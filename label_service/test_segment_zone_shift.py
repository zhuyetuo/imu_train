"""label_mode=segment（train_v2 的模型）：窗口负责的时间段整体往后挪 (窗口 - 步长)/2；
别的 label_mode 一点不动。挪的是整条时间轴，所以片段只是平移、长度和切法都不变。"""
from datetime import datetime, timedelta

from label_service import postprocess as pp

CLASSES = ["活动", "抓挠"]
T0 = datetime(2026, 11, 15, 10, 0, 0)


def _windows(seq):
    return [{"ts": pp._fmt_ts(T0 + timedelta(seconds=i)), "label": lab, "conf": 0.9,
             "probs": {c: (0.9 if c == lab else 0.1) for c in CLASSES}} for i, lab in enumerate(seq)]


def _seg(label_mode):
    seq = ["活动"] * 20 + ["抓挠"] * 6 + ["活动"] * 20
    out = pp.stabilize(_windows(seq), CLASSES, ["抓挠"], 4.0, 1.0, label_mode, pp.StableParams(), algo="viterbi")
    s = out["抓挠"]
    assert len(s) == 1
    return pp._parse_ts(s[0]["start_ts"]), pp._parse_ts(s[0]["end_ts"])


def test_segment_整体后移半个窗口减半个步长():
    a0, a1 = _seg("majority")
    b0, b1 = _seg("segment")
    assert a0 == T0 + timedelta(seconds=20)                       # 老规矩：从窗口起点算，一点不动
    assert (b0 - a0).total_seconds() == 1.5 and (b1 - a1).total_seconds() == 1.5
