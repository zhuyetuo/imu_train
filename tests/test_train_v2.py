"""新版训练流程（src/ml/train_v2.py）：平台参数怎么传过去、细类怎么并回、整条流程能跑通且产物推理 / 端侧认得。"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

jobs = pytest.importorskip("label_service.jobs")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(ROOT, "src", "ml"), os.path.join(ROOT, "src", "data")]


def _spec_file(job_id):
    return os.path.join(ROOT, "tmp", f"train_v2_job{job_id}.json")


def test_平台勾新版流程_命令和输入都对():
    from label_service.app import DatasetSpec
    from label_service import config as cfg

    d = DatasetSpec(date="pool_x", export_json="data_train/pool_x/merged_tmp.json", source_hz=50, hz=15.625,
                    extra_datasets=[{"date": "old16", "export_json": "data_train/old16/merged_tmp.json", "source_hz": 16}],
                    label_remap={"进食": "活动", "抓挠-躯干": "抓挠", "活动": "活动", "抓挠": "抓挠"},
                    pipeline="v2", fine_labels=["进食"], class_scale={"抓挠": 1.5}, axes=3,
                    edge_size=True, edge_trees=20, edge_depth=8).model_dump()
    try:
        cmd = jobs.build_command(d, "rf", None, 99911)
        assert cmd[1] == "src/ml/train_v2.py"
        spec = json.load(open(_spec_file(99911), encoding="utf-8"))
        assert spec["hz"] == 15.625 and spec["window"] == 62          # 默认 4 秒 → 62 点
        assert spec["classes"] == ["活动", "抓挠"] and spec["fine_labels"] == ["进食"]
        assert spec["class_scale"] == {"抓挠": 1.5}
        assert spec["model_params"] == {"n_estimators": 20, "max_depth": 8}
        assert [x["source_hz"] for x in spec["datasets"]] == [50, 16]
        assert spec["axes"] == 3
        jobs.build_command({**d, "axes": 6}, "rf", None, 99911)
        assert json.load(open(_spec_file(99911), encoding="utf-8"))["axes"] == 6
        assert spec["datasets"][1]["export_json"] == os.path.join(cfg.NAS_ROOT, "data_train/old16/merged_tmp.json")
        # 窗口跟着采样率走：25Hz × 4s = 100 点（上限 128）；不填采样率默认 20Hz → 80 点
        jobs.build_command({**d, "hz": 25.0}, "rf", None, 99911)
        assert json.load(open(_spec_file(99911), encoding="utf-8"))["window"] == 100
        jobs.build_command({**d, "hz": None}, "rf", None, 99911)
        sp_ = json.load(open(_spec_file(99911), encoding="utf-8"))
        assert sp_["hz"] == 20.0 and sp_["window"] == 80
        with pytest.raises(RuntimeError):
            jobs.build_command(d, "cnn", None, 99911)
    finally:
        if os.path.exists(_spec_file(99911)):
            os.remove(_spec_file(99911))


def test_细类并回_概率等于手算():
    from sklearn.ensemble import RandomForestClassifier
    from train_v2 import merge_forest

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    y = rng.integers(0, 4, 600)                  # 0,1,2 是正式类别，3 是细类（并回 0）
    X[y == 1, 0] += 2
    m = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=0).fit(X, y)
    per_tree = []
    for e in m.estimators_:
        p = e.predict_proba(X[:50])
        q = np.stack([p[:, 0] + p[:, 3], p[:, 1], p[:, 2] * 2.0], 1)   # 并回 + 第 2 类 ×2
        per_tree.append(q / q.sum(1, keepdims=True))
    want = np.mean(per_tree, 0)
    merged = merge_forest(m, ["a", "b", "c", "a_fine"], ["a", "b", "c"], [0, 1, 2, 0], np.array([1.0, 1.0, 2.0]))
    got = merged.predict_proba(X[:50])
    assert merged.n_classes_ == 3 and got.shape == (50, 3)
    np.testing.assert_allclose(got, want, atol=1e-9)


def _fake_csv(path, seconds, rng, pattern):
    """50Hz、蓝牙包时间戳（5 个点一个时间），pattern(t) 给每个时刻的加速度"""
    import pandas as pd

    n = int(seconds * 50)
    t = np.arange(n) / 50
    pkt = (np.arange(n) // 5) * 5 / 50
    ts = pd.Timestamp("2026-10-30 10:00:00") + pd.to_timedelta(pkt, unit="s")
    acc = np.stack([pattern(t, k) for k in range(3)], 1) + rng.normal(0, 0.02, (n, 3))
    df = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
                       "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
                       "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0})
    df.to_csv(path, index=False)


def test_整条流程跑通_产物推理和端侧都认得(tmp_path):
    rng = np.random.default_rng(1)
    nas = tmp_path / "nas"
    (nas / "raw").mkdir(parents=True)
    still = lambda t, k: np.full_like(t, [0.0, 0.0, -1.0][k])
    move = lambda t, k: np.sin(2 * np.pi * (1 + k) * 0.7 * t) * 0.4 + [0, 0, -1.0][k]
    scratch = lambda t, k: np.sin(2 * np.pi * 6 * t + k) * 1.2 + [0, 0, -1.0][k]
    tasks = []
    for i, imu in enumerate(["imu1", "imu2", "imu3", "imu4"]):
        rel = f"raw/multicam_2026103{i}_100000000_cam1_{imu}_raw.csv"
        # 前 60 秒静止、60~120 秒活动、120~180 秒抓挠（中间夹 3 秒短抓挠）
        f = lambda t, k: np.where(t < 60, still(t, k), np.where(t < 120, move(t, k), scratch(t, k)))
        _fake_csv(nas / rel, 180, rng, f)

        def seg(a, b, labs):
            return {"value": {"start_ms": int(a * 1000), "end_ms": int(b * 1000), "timeserieslabels": labs}}
        res = [seg(1, 59, ["睡觉"]), seg(61, 100, ["活动"]), seg(100, 119, ["进食"]),
               seg(121, 150, ["抓挠", "抓挠-躯干"]), seg(160, 163, ["抓挠"])]
        tasks.append({"id": i, "data": {"csv": rel, "sample_code": f"multicam_2026103{i}_100000000_{imu}"},
                      "annotations": [{"id": i, "result": res}]})
    exp = tmp_path / "merged_tmp.json"
    exp.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    spec = {"hz": 15.625, "window": 62, "stride_s": 1.0, "classes": ["静止/休息", "活动", "抓挠"],
            "label_remap": {"睡觉": "静止/休息", "活动": "活动", "进食": "活动", "抓挠": "抓挠", "抓挠-躯干": "抓挠"},
            "fine_labels": ["进食"], "class_scale": {"抓挠": 1.5}, "model_params": {"n_estimators": 5, "max_depth": 6},
            "run_name": "v2pytest", "nas_root": str(nas),
            "datasets": [{"name": "fake", "export_json": str(exp), "source_hz": 50}]}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "ml", "train_v2.py"), "--spec", str(sp),
                        "--repo", str(tmp_path), "--workers", "1", "--n_jobs", "1"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("纯标注: ")]
    assert line == ["纯标注: results/processed_v2pytest/15.625hz_v2/rf/ml_rf.pkl"]
    assert jobs.parse_model_path(r.stdout) == "results/processed_v2pytest/15.625hz_v2/rf/ml_rf.pkl"
    model_path = tmp_path / "results/processed_v2pytest/15.625hz_v2/rf/ml_rf.pkl"
    meta = json.load(open(str(model_path).replace(".pkl", ".json"), encoding="utf-8"))
    assert meta["classes"] == spec["classes"] and meta["window_size"] == 62 and meta["n_channels"] == 5
    assert meta["train_classes"] == ["静止/休息", "活动", "抓挠", "进食"]
    assert meta["macro_f1"] > 0.8                      # 三种信号差别很大，分不开说明流程有问题
    assert meta["main_dataset"] == "fake"
    pdm = meta["per_dataset"]["fake"]
    assert pdm["confusion"]["labels"] == spec["classes"] and len(pdm["confusion"]["matrix"]) == 3
    assert set(pdm["per_class"]) <= set(spec["classes"])
    # 6 轴：同一份数据加陀螺仪，输入 8 个通道
    spec6 = {**spec, "axes": 6, "run_name": "v2pytest6"}
    sp.write_text(json.dumps(spec6, ensure_ascii=False), encoding="utf-8")
    r6 = subprocess.run([sys.executable, os.path.join(ROOT, "src", "ml", "train_v2.py"), "--spec", str(sp),
                         "--repo", str(tmp_path), "--workers", "1", "--n_jobs", "1"], capture_output=True, text=True, timeout=600)
    assert r6.returncode == 0, r6.stdout[-2000:] + r6.stderr[-2000:]
    m6 = json.load(open(tmp_path / "results/processed_v2pytest6/15.625hz_v2/rf/ml_rf.json", encoding="utf-8"))
    assert m6["n_channels"] == 8
    import joblib
    m = joblib.load(model_path)
    assert list(m.classes_) == [0, 1, 2]
    # 推理侧读得懂：窗口 / 步长 / 通道
    from label_service.model_loader import load_model_bundle
    b = load_model_bundle(str(model_path))
    assert b["hz"] == 15.625 and round(b["window_s"] * b["hz"]) == 62 and b["channel_mode"] == "inv"
    # 端侧导出要的留出集
    d = np.load(tmp_path / "data/processed_v2pytest/15.625hz/val.npz")
    assert d["X"].shape[1:] == (62, 5) and set(np.unique(d["y"])) <= {0, 1, 2}
    assert meta["drop_gravity_level"] is False and meta["masked_features"] == []
    # 不用重力绝对大小：树里一次都不用那几列，特征维度不变（推理照常喂完整的向量）
    from train_v2 import gravity_level_mask
    specg = {**spec, "drop_gravity_level": True, "run_name": "v2pytestg"}
    sp.write_text(json.dumps(specg, ensure_ascii=False), encoding="utf-8")
    rg = subprocess.run([sys.executable, os.path.join(ROOT, "src", "ml", "train_v2.py"), "--spec", str(sp),
                         "--repo", str(tmp_path), "--workers", "1", "--n_jobs", "1"], capture_output=True, text=True, timeout=600)
    assert rg.returncode == 0, rg.stdout[-2000:] + rg.stderr[-2000:]
    mg_path = tmp_path / "results/processed_v2pytestg/15.625hz_v2/rf/ml_rf.pkl"
    mg = joblib.load(mg_path)
    metag = json.load(open(str(mg_path).replace(".pkl", ".json"), encoding="utf-8"))
    mask, names = gravity_level_mask(5)
    assert metag["drop_gravity_level"] is True and metag["masked_features"] == names and "acc_z_mean" in names
    assert mg.n_features_in_ == m.n_features_in_ == mask.size
    for est in mg.estimators_:
        f = est.tree_.feature
        assert not mask[f[f >= 0]].any()
    assert metag["macro_f1"] > 0.8
    assert load_model_bundle(str(mg_path))["hz"] == 15.625
    # XGBoost：直接训归并后的类别，产物是 ml_xgb.pkl + ml_xgb.json，推理侧照样认得
    specx = {**spec, "model": "xgb", "drop_gravity_level": True, "run_name": "v2pytestx",
             "model_params": {"n_estimators": 10, "max_depth": 4}}
    sp.write_text(json.dumps(specx, ensure_ascii=False), encoding="utf-8")
    rx = subprocess.run([sys.executable, os.path.join(ROOT, "src", "ml", "train_v2.py"), "--spec", str(sp),
                         "--repo", str(tmp_path), "--workers", "1", "--n_jobs", "1"], capture_output=True, text=True, timeout=600)
    assert rx.returncode == 0, rx.stdout[-2000:] + rx.stderr[-2000:]
    assert jobs.parse_model_path(rx.stdout) == "results/processed_v2pytestx/15.625hz_v2/xgb/ml_xgb.pkl"
    mx_path = tmp_path / "results/processed_v2pytestx/15.625hz_v2/xgb/ml_xgb.pkl"
    metax = json.load(open(str(mx_path).replace(".pkl", ".json"), encoding="utf-8"))
    assert metax["model"] == "xgb" and metax["train_classes"] == spec["classes"] and metax["fine_labels"] == []
    assert metax["macro_f1"] > 0.8
    bx = load_model_bundle(str(mx_path))
    assert bx["classes"] == spec["classes"] and hasattr(bx["model"], "get_booster")
    assert len(bx["model"].get_booster().get_dump()) == 10 * 3          # 10 轮 × 3 类


def test_xgb_输出校准做进叶子_等于乘系数再归一化():
    import xgboost  # noqa: F401
    from train_v2 import bake_class_scale, fit_xgb, xgb_params

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6)).astype(np.float32)
    y = rng.integers(0, 3, 600)
    X[y == 1, 0] += 1.5
    m, _ = fit_xgb(X, y, 3, xgb_params({"n_estimators": 8, "max_depth": 3}), 1)
    before = m.predict_proba(X[:50])
    s = np.array([1.0, 2.5, 0.5])
    bake_class_scale(m, s, X[:200])
    want = before * s
    want /= want.sum(1, keepdims=True)
    np.testing.assert_allclose(m.predict_proba(X[:50]), want, rtol=1e-4, atol=1e-6)
    with pytest.raises(SystemExit):
        xgb_params({"max_depth": 7})                                   # 端侧紧凑编码最多深 6


def test_平台提交_xgb_走新版流程_端侧默认50轮深6():
    from label_service.app import DatasetSpec

    d = DatasetSpec(date="pool", export_json="a.json", pipeline="v2", label_remap={"活动": "活动"},
                    edge_size=True).model_dump()
    try:
        jobs.build_command(d, "xgb", None, 99914)
        spec = json.load(open(_spec_file(99914), encoding="utf-8"))
        assert spec["model"] == "xgb" and spec["model_params"] == {"n_estimators": 50, "max_depth": 6}
        jobs.build_command(d, "rf", None, 99914)
        spec = json.load(open(_spec_file(99914), encoding="utf-8"))
        assert spec["model"] == "rf" and spec["model_params"] == {"n_estimators": 20, "max_depth": 8}
    finally:
        if os.path.exists(_spec_file(99914)):
            os.remove(_spec_file(99914))


def test_静止活动按动作强度分_只改这两类本身():
    from train_v2 import motion_of, still_by_motion

    rng = np.random.default_rng(0)
    X = np.zeros((6, 80, 8), np.float32)
    X[:, :, 2] = 1.0
    amp = [0.0005, 0.02, 0.0005, 0.02, 0.0005, 0.02]
    for i, a in enumerate(amp):
        X[i, :, 2] += rng.normal(0, a, 80)
    mot = motion_of(X)
    assert (mot[::2] < 0.003).all() and (mot[1::2] > 0.01).all()
    # 0 静止 1 活动 2 进食（细类，不动）
    y = np.array([1, 1, 0, 0, 2, 2])
    plain = np.array([True, True, True, True, False, False])
    y2, info = still_by_motion(y, mot, plain, 0, 1, 0.005)
    assert y2.tolist() == [0, 1, 0, 1, 2, 2]
    assert info["活动→静止"] == 1 and info["静止→活动"] == 1


def test_平台把不用重力绝对大小传到训练():
    from label_service.app import DatasetSpec

    d = DatasetSpec(date="pool", export_json="a.json", pipeline="v2", label_remap={"活动": "活动"}).model_dump()
    try:
        jobs.build_command(d, "rf", None, 99913)
        assert json.load(open(_spec_file(99913), encoding="utf-8"))["drop_gravity_level"] is False
        jobs.build_command({**d, "drop_gravity_level": True}, "rf", None, 99913)
        assert json.load(open(_spec_file(99913), encoding="utf-8"))["drop_gravity_level"] is True
        assert json.load(open(_spec_file(99913), encoding="utf-8"))["still_motion_g"] is None
        jobs.build_command({**d, "still_motion_g": 0.004}, "rf", None, 99913)
        assert json.load(open(_spec_file(99913), encoding="utf-8"))["still_motion_g"] == 0.004
    finally:
        if os.path.exists(_spec_file(99913)):
            os.remove(_spec_file(99913))


def test_一起训练的数据集可以不用某几类(tmp_path):
    from train_v2 import collect_segments

    tasks = [{"id": 1, "data": {"csv": "a.csv", "sample_code": "x_imu1"},
              "annotations": [{"result": [{"value": {"start_ms": 0, "end_ms": 5000, "timeserieslabels": ["活动"]}},
                                          {"value": {"start_ms": 5000, "end_ms": 9000, "timeserieslabels": ["抓挠"]}}]}]}]
    p = tmp_path / "e.json"
    p.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    remap = {"活动": "活动", "抓挠": "抓挠"}
    cls = ["活动", "抓挠"]
    all_ = collect_segments([{"name": "old", "export_json": str(p)}], str(tmp_path), remap, cls, [])
    only = collect_segments([{"name": "old", "export_json": str(p), "exclude_classes": ["活动"]}], str(tmp_path), remap, cls, [])
    assert [s["coarse"] for s in all_] == ["活动", "抓挠"] and [s["coarse"] for s in only] == ["抓挠"]


def test_平台把不用的类别传到训练():
    from label_service.app import DatasetSpec

    d = DatasetSpec(date="pool", export_json="a.json", pipeline="v2", label_remap={"活动": "活动", "抓挠": "抓挠"},
                    extra_datasets=[{"date": "old", "export_json": "b.json", "source_hz": 50, "exclude_classes": ["活动"]}]).model_dump()
    try:
        jobs.build_command(d, "rf", None, 99912)
        spec = json.load(open(_spec_file(99912), encoding="utf-8"))
        assert spec["datasets"][0].get("exclude_classes") is None and spec["datasets"][1]["exclude_classes"] == ["活动"]
    finally:
        if os.path.exists(_spec_file(99912)):
            os.remove(_spec_file(99912))


def test_新版流程的训练集统计_按它自己的窗口_短片段不算进不了(tmp_path, monkeypatch):
    from label_service import config as cfg, dataset_stats

    monkeypatch.setattr(cfg, "REPO_ROOT", str(tmp_path))
    d = tmp_path / "data" / "raw_custom" / "old__job7"
    d.mkdir(parents=True)

    def seg(a, b, lab):
        return {"value": {"start": f"2026-10-30 10:00:{a:02d}.000", "end": f"2026-10-30 10:00:{b:02d}.000", "timeserieslabels": [lab]}}
    tasks = [{"id": 1, "data": {"csv": "x_20261030.csv"},
              "annotations": [{"result": [seg(0, 1, "甩头/抖身"), seg(2, 3, "甩头/抖身"), seg(10, 40, "活动")]}]}]
    (d / "merged_tmp.json").write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    spec = {"date": "pool", "pipeline": "v2", "hz": 15.625, "label_remap": {"甩头/抖身": "甩头/抖身", "活动": "活动"},
            "extra_datasets": [{"date": "old", "exclude_classes": ["活动"]}]}
    st = dataset_stats.compute(spec, ["old__job7"], *jobs.v2_stats_geom(spec))
    assert abs(st["window_s"] - 62 / 15.625) < 1e-9
    rows = {r["label"]: r for r in st["rows"]}
    assert rows["甩头/抖身"]["windows"] == 2 and "活动" not in rows
    assert not any("进不了训练" in h or "切不出来" in h for h in st["hints"])
    assert not any("归并表里没有" in h for h in st["hints"])
    # 老流程还是按 2 秒窗口说「切不出来」
    old = dataset_stats.compute({**spec, "pipeline": None}, ["old__job7"])
    assert any("切不出来" in h or "进不了训练" in h for h in old["hints"])
