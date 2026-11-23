"""严格版（viterbi_strict）：解码跟 v2 一样，只把抓挠的门槛收紧；别的类别、疑似抓挠候选不变。"""
from datetime import datetime, timedelta

from label_service import postprocess as pp

CLASSES = ["活动", "抓挠", "甩身体"]
T0 = datetime(2026, 11, 24, 10, 0, 0)


def _windows(spans, total=60, gstd=60.0, arhythm=0.6):
    """spans: [(起, 止, 类别, 概率)]，其余时间是活动"""
    out = []
    for i in range(total):
        lab, p = "活动", 0.9
        for a, b, c, q in spans:
            if a <= i < b:
                lab, p = c, q
        probs = {c: (p if c == lab else (1 - p) / (len(CLASSES) - 1)) for c in CLASSES}
        out.append({"ts": pp._fmt_ts(T0 + timedelta(seconds=i)), "label": lab, "conf": p, "probs": probs,
                    "gstd": gstd, "arhythm": arhythm, "amag": 1.2, "missing": 0.0})
    return out


def _n(segs, lab="抓挠"):
    return len(segs.get(lab) or [])


def _run(w, algo, p=None):
    return pp.stabilize(w, CLASSES, CLASSES, 4.0, 1.0, "segment", p or pp.StableParams(), algo=algo)


def test_有把握的抓挠两版都留():
    w = _windows([(20, 30, "抓挠", 0.8)])
    assert _n(_run(w, "viterbi")) == 1 and _n(_run(w, "viterbi_strict")) == 1


def test_平均概率不到0点7的抓挠严格版去掉():
    w = _windows([(20, 30, "抓挠", 0.6)])
    assert _n(_run(w, "viterbi")) == 1
    assert _n(_run(w, "viterbi_strict")) == 0


def test_物理门槛默认沿用线上值():
    """评估和线上的陀螺仪 / 节律口径不同，严格版默认不另收这两道"""
    w = _windows([(20, 30, "抓挠", 0.8)], gstd=10.0, arhythm=0.35)
    on = pp.StableParams(gyro_std_min=5.0, acc_rhythm_min=0.25)
    assert _n(_run(w, "viterbi", on)) == 1 and _n(_run(w, "viterbi_strict", on)) == 1
    assert _n(_run(w, "viterbi_strict", pp.StableParams(gyro_std_min=20.0))) == 0


def test_配了严格物理门槛才收():
    w = _windows([(20, 30, "抓挠", 0.8)], arhythm=0.35)
    p = pp.StableParams(acc_rhythm_min=0.25, strict_acc_rhythm_min=0.45)
    assert _n(_run(w, "viterbi", p)) == 1 and _n(_run(w, "viterbi_strict", p)) == 0


def test_甩身体不跟着收紧():
    w = _windows([(20, 30, "甩身体", 0.6)])
    assert _run(w, "viterbi")["甩身体"] == _run(w, "viterbi_strict")["甩身体"]


def test_线上门槛比严格版还严时不放松():
    p = pp.strict_params(pp.StableParams(gyro_std_min=30.0, acc_rhythm_min=0.5))
    assert p.gyro_std_min == 30.0 and p.acc_rhythm_min == 0.5 and p.scratch_min_mean == 0.7


def test_不是严格版就跟以前一模一样():
    w = _windows([(20, 30, "抓挠", 0.6), (40, 45, "甩身体", 0.7)])
    assert _run(w, "viterbi") == _run(w, "viterbi", pp.StableParams(scratch_min_mean=0.0))
    assert not pp.is_strict("viterbi") and not pp.is_strict("viterbi_precise") and pp.is_strict("viterbi_strict")
