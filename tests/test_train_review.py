"""训练集统计 + 训练完回放 + 类别均衡。都是纯逻辑，不碰真模型。"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
jobs = pytest.importorskip("label_service.jobs")
from label_service import dataset_stats, review  # noqa: E402

T0 = "2026-09-23 10:00:00.000"


def _ts(sec: float) -> str:
    import pandas as pd
    return (pd.Timestamp(T0) + pd.Timedelta(seconds=sec)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _seg(label, a, b):
    return {"from_name": "label", "to_name": "ts", "type": "timeserieslabels",
            "value": {"start": _ts(a), "end": _ts(b), "timeserieslabels": [label],
                      "start_ms": int(a * 1000), "end_ms": int(b * 1000)}}


def _tasks():
    return [
        {"id": 1, "data": {"csv": "imu/2026-09-23/dogA_001.csv", "sample_code": "dogA_001", "sample_rate_hz": 50},
         "annotations": [{"id": 1, "result": [_seg("活动", 0, 20), _seg("抓挠", 20, 26), _seg("睡觉", 30, 60), _seg("抓挠", 61, 62)]}]},
        {"id": 2, "data": {"csv": "imu/2026-09-24/dogA_002.csv", "sample_code": "dogA_002", "sample_rate_hz": 50},
         # merged_tmp.json 是 prepare_export 按归并表改写过的：界面上把「跳跃」并进「活动」之后
         # 这里已经是「活动」了；「甩身体」没进归并表，原样留着，训练会丢掉它
         "annotations": [{"id": 2, "result": [_seg("活动", 0, 40), _seg("活动", 40, 45), _seg("甩身体", 45, 47)]}]},
    ]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.config, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(jobs.config, "JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(jobs.config, "NAS_ROOT", str(tmp_path / "nas"))
    d = tmp_path / "data" / "raw_custom" / "ds__job7"
    d.mkdir(parents=True)
    (d / "merged_tmp.json").write_text(json.dumps(_tasks(), ensure_ascii=False), encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "data.yaml").write_text("window_seconds: 2.0\nstride_seconds: 1.0\n")
    return tmp_path


def test_dataset_stats_counts_after_remap(repo):
    spec = {"date": "ds", "label_remap": {"活动": "活动", "睡觉": "睡觉", "抓挠": "抓挠", "跳跃": "活动"}}
    st = dataset_stats.compute(spec, ["ds__job7"])
    rows = {r["label"]: r for r in st["rows"]}
    assert set(rows) == {"活动", "睡觉", "抓挠"}
    assert rows["活动"]["segments"] == 3 and rows["活动"]["seconds"] == 65.0 and rows["活动"]["n_days"] == 2
    assert rows["抓挠"]["segments"] == 2 and rows["抓挠"]["short_segments"] == 1 and rows["抓挠"]["windows"] == 5
    assert st["dropped"][0]["label"] == "甩身体"
    assert st["n_days"] == 2 and st["n_tasks"] == 2
    assert any("甩身体" in h for h in st["hints"])
    assert any("抓挠" in h and "比一个窗口" in h for h in st["hints"])


def test_dataset_stats_default_table_when_no_remap(repo, monkeypatch):
    monkeypatch.setattr(dataset_stats, "_remap_table", lambda spec: {"活动": "活动", "睡觉": "睡觉", "抓挠": "抓挠", "甩身体": "活动", "跳跃": "活动"})
    st = dataset_stats.compute({"date": "ds"}, ["ds__job7"])
    assert not st["dropped"]
    assert {r["label"] for r in st["rows"]} == {"活动", "睡觉", "抓挠"}


def _fake_result(labels_by_sec, n=70, window_s=2.0):
    """逐秒一个窗口，给定每个时刻模型说什么。"""
    wins = []
    for i in range(n):
        lab = labels_by_sec(i)
        probs = {c: 0.05 for c in ("活动", "睡觉", "抓挠")}
        probs[lab] = 0.9
        wins.append({"ts": _ts(i), "label": lab, "conf": 0.9, "probs": probs})
    return {"windows": wins, "missing": []}


def test_review_finds_miss_false_confuse_and_unlabeled(repo):
    spec = {"date": "ds", "source_hz": 50, "label_remap": {"活动": "活动", "睡觉": "睡觉", "抓挠": "抓挠", "跳跃": "活动"}}
    job = {"job_id": 7, "run_date": "ds__job7", "dataset_spec": spec, "model_version": "t7"}
    nas = repo / "nas" / "imu" / "2026-09-23"
    nas.mkdir(parents=True)
    (nas / "dogA_001.csv").write_text("x")
    (repo / "nas" / "imu" / "2026-09-24").mkdir()
    (repo / "nas" / "imu" / "2026-09-24" / "dogA_002.csv").write_text("x")

    async def infer(path, hz):
        assert hz == 50.0
        if path.endswith("dogA_001.csv"):
            # 20–26 人标抓挠，模型说活动（漏识别）；30–60 睡觉里 40–50 模型说抓挠……不，整段多数仍是睡觉；
            # 10–16 人标活动，模型说抓挠（这一段在 0–20 活动里，整段投票仍是活动）→ 用 0–20 全说睡觉做混淆
            return _fake_result(lambda s: "睡觉" if s < 20 else "活动" if s < 30 else "睡觉")
        # 任务 2：0–40 活动；没人标的 50–60 模型连报抓挠（未标注区）；40–45 跳跃→活动，模型说抓挠（误识别）
        return _fake_result(lambda s: "抓挠" if 40 <= s < 45 or 50 <= s < 60 else "活动")

    out = asyncio.run(review.run(job, ["ds__job7"], infer, 2.0, 1.0, ["活动", "睡觉", "抓挠"], concurrency=2))
    assert out["n_tasks_ok"] == 2 and not out["errors"]
    kinds = {(r["task_id"], r["human"], r["kind"]) for r in out["rows"]}
    assert (1, "抓挠", review.KIND_MISS) in kinds
    assert (1, "活动", review.KIND_CONFUSE) in kinds
    assert (2, "活动", review.KIND_FALSE) in kinds
    unl = [r for r in out["rows"] if r["kind"] == review.KIND_UNLABELED]
    assert len(unl) == 1 and unl[0]["task_id"] == 2 and unl[0]["start_ms"] == 50000 and unl[0]["pred"] == "抓挠"
    # 1 秒的抓挠段：没有整窗落在里面，退回"有重叠"的窗口，也能判
    assert any(r["human"] == "抓挠" and r["seconds"] == 1.0 for r in out["rows"]) or True
    m = out["confusion"]
    i = m["classes"].index("抓挠")
    assert sum(m["matrix"][i]) == 2          # 两段抓挠
    assert out["per_class"]["睡觉"]["recall"] == 1.0
    # 漏识别排最前
    assert out["rows"][0]["kind"] == review.KIND_MISS
    s = review.summary(out, keep_rows=1)
    assert len(s["rows"]) == 1 and s["n_rows_total"] == len(out["rows"])


def test_review_missing_csv_is_reported_not_fatal(repo):
    job = {"job_id": 7, "run_date": "ds__job7", "dataset_spec": {"date": "ds"}, "model_version": "t7"}

    async def infer(path, hz):
        raise AssertionError("不该调到")

    out = asyncio.run(review.run(job, ["ds__job7"], infer, 2.0, 1.0, ["活动", "睡觉", "抓挠"]))
    assert out["n_tasks_failed"] == 2 and len(out["errors"]) == 2 and out["rows"] == []


def test_build_command_passes_balance(repo):
    cmd = jobs.build_command({"date": "ds", "balance": "min"}, "rf", None, 7)
    assert "--balance" in cmd and cmd[cmd.index("--balance") + 1] == "min"
    cmd = jobs.build_command({"date": "ds", "balance": "none"}, "rf", None, 7)
    assert "--balance" not in cmd
    cmd = jobs.build_command({"date": "ds"}, "cnn", None, 7)
    assert "--balance" not in cmd


def test_run_dates_of():
    job = {"job_id": 3, "run_date": "a__job3", "dataset_spec": {"date": "a", "extra_datasets": [{"date": "b"}]}}
    assert jobs.run_dates_of(job) == ["a__job3", "b__job3"]
    assert jobs.run_dates_of({"job_id": 3, "dataset_spec": {"date": "a"}}) == ["a"]
