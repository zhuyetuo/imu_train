"""短促动作（甩动，一两秒）补成训练窗口：以前比 2 秒窗口短的片段要么被丢、要么被多数投票吞掉。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
import labelstudio_to_custom as L  # noqa: E402

HZ = 16


def _csv(tmp_path):
    n = 20 * HZ
    t0 = pd.Timestamp("2026-11-01 10:00:00")
    ts = [t0 + pd.Timedelta(seconds=i / HZ) for i in range(n)]
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"time": [t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] for t in ts],
                       **{c: rng.normal(size=n) for c in ("AX", "AY", "AZ", "GX", "GY", "GZ")}})
    p = tmp_path / "a.csv"
    df.to_csv(p, index=False)
    return p


def _seg(label, a, b):
    t = lambda s: (pd.Timestamp("2026-11-01 10:00:00") + pd.Timedelta(seconds=s)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return {"value": {"timeserieslabels": [label], "start": t(a), "end": t(b)}}


def _tasks(p):
    return [{"id": 1, "data": {"csv": p.name}, "annotations": [{"result": [
        _seg("抓挠", 0, 5), _seg("甩动", 5.5, 6.5), _seg("抓挠", 7, 12), _seg("甩动", 15, 15.8), _seg("活动", 16, 20),
    ]}]}]


def test_without_option_short_segments_stay_as_before(tmp_path):
    p = _csv(tmp_path)
    out = L.convert(_tasks(p), str(tmp_path), "ms2")
    assert not out["record_id"].str.contains("_short").any()
    assert (out["label"] == "甩动").sum() <= 2 * HZ + 2       # 原样：只有片段自己那一两秒


def test_short_events_become_context_records_every_window_contains_event(tmp_path, capsys):
    p = _csv(tmp_path)
    out = L.convert(_tasks(p), str(tmp_path), "ms2", short_labels=["甩动"], window_s=2.0)
    short = out[out["record_id"].str.contains("_short")]
    assert set(short["label"]) == {"甩动"} and short["record_id"].nunique() == 2
    for rid, g in short.groupby("record_id"):
        span = (pd.to_datetime(g["timestamp"]).max() - pd.to_datetime(g["timestamp"]).min()).total_seconds()
        assert span >= 2.0                                   # 至少能切出一个完整窗口
    # 第一次甩动夹在两段抓挠中间（5~5.5、6.5~7 才是空的）：上下文截在抓挠边上，不够长用镜像补齐，
    # 不能把抓挠的数据标成甩动
    first = short[short["record_id"] == short["record_id"].iloc[0]]
    raw = pd.read_csv(p)
    scratch_vals = set(np.round(raw["AX"].iloc[0:5 * HZ], 6)) | set(np.round(raw["AX"].iloc[7 * HZ + 1:12 * HZ], 6))
    assert not (set(np.round(first["acc_x"], 6)) & scratch_vals)
    # 原来那几行甩动不再单独留在主记录里（不然又被吞）
    main = out[~out["record_id"].str.contains("_short")]
    assert "甩动" not in set(main["label"])
    assert "短事件补齐" in capsys.readouterr().out


def test_long_enough_segments_untouched(tmp_path):
    p = _csv(tmp_path)
    out = L.convert(_tasks(p), str(tmp_path), "ms2", short_labels=["抓挠"], window_s=2.0)
    assert not out["record_id"].str.contains("_short").any()  # 抓挠段都 5 秒，比窗口长，照旧


def test_platform_passes_short_labels_to_script():
    import pytest
    jobs = pytest.importorskip("label_service.jobs")
    cmd = jobs.build_command({"date": "d", "short_labels": ["甩动", " ", "甩头/抖身"]}, "rf", None)
    assert cmd[cmd.index("--short_labels") + 1] == "甩动,甩头/抖身"
    assert "--short_labels" not in jobs.build_command({"date": "d"}, "rf", None)


def test_short_event_recognized_by_original_name_after_remap(tmp_path):
    """甩动并进「活动」当负样本：改名之后也要按原名认出来补齐（标签是新名）"""
    import pytest
    jobs = pytest.importorskip("label_service.jobs")
    p = _csv(tmp_path)
    tasks = _tasks(p)
    jobs.apply_label_remap(tasks, {"甩动": "活动"})
    out = L.convert(tasks, str(tmp_path), "ms2", short_labels=["甩动"], window_s=2.0)
    short = out[out["record_id"].str.contains("_short")]
    assert short["record_id"].nunique() == 2 and set(short["label"]) == {"活动"}
