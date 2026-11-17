"""精准版（viterbi_precise）起止：平移校正 + 3–9 Hz 能量对齐。片段个数、类别跟 v2 一模一样，只改起止。"""
from datetime import datetime, timedelta

from label_service import postprocess as pp

CLASSES = ["活动", "抓挠"]
T0 = datetime(2026, 11, 18, 10, 0, 0)


def _windows(seq):
    return [{"ts": pp._fmt_ts(T0 + timedelta(seconds=i)), "label": lab, "conf": 0.9,
             "probs": {c: (0.9 if c == lab else 0.1) for c in CLASSES}} for i, lab in enumerate(seq)]


def _span(seg):
    return ((pp._parse_ts(seg["start_ts"]) - T0).total_seconds(), (pp._parse_ts(seg["end_ts"]) - T0).total_seconds())


def _env(on_from_s, on_to_s, total_s=80, hz=10.0):
    """3–9 Hz 能量：[on_from, on_to) 秒里在抓（1.0），别处 0.01"""
    band = [1.0 if on_from_s * hz <= i < on_to_s * hz else 0.01 for i in range(int(total_s * hz))]
    return {"t0": pp._fmt_ts(T0), "hz": hz, "energy": band, "band": band}


def test_解码跟v2完全一样():
    seq = ["活动"] * 20 + ["抓挠"] * 10 + ["活动"] * 20
    a = pp.stabilize(_windows(seq), CLASSES, ["抓挠"], 8.0, 1.0, "segment", pp.StableParams(), algo="viterbi")
    b = pp.stabilize(_windows(seq), CLASSES, ["抓挠"], 8.0, 1.0, "segment", pp.StableParams(), algo="viterbi_precise")
    assert a == b


def test_没有包络时只平移():
    seg = [{"start_ts": pp._fmt_ts(T0 + timedelta(seconds=20)), "end_ts": pp._fmt_ts(T0 + timedelta(seconds=30))}]
    pp.precise_boundaries(seg, None, pp.precise_bias("抓挠", 8.0, 1.0, "segment"))
    s, e = _span(seg[0])
    assert abs(s - 20.5) < 1e-6 and abs(e - 28.9) < 1e-6        # 起点往后 0.5，终点往前 1.1


def test_按能量对齐到真正开始和停下():
    seg = [{"start_ts": pp._fmt_ts(T0 + timedelta(seconds=20)), "end_ts": pp._fmt_ts(T0 + timedelta(seconds=31))}]
    # 真正在抓的是 21.0 ~ 29.0 秒
    pp.precise_boundaries(seg, _env(21.0, 29.0), pp.precise_bias("抓挠", 8.0, 1.0, "segment"))
    s, e = _span(seg[0])
    assert abs(s - 21.0) <= 0.11 and abs(e - 29.0) <= 0.11


def test_抓之前的低频大动作不影响起点():
    """平台原来的微调按总能量找：抓之前在转身，起点就被带早了。band 里没有转身，起点不动"""
    env = _env(21.0, 29.0)
    env["energy"] = [5.0 if 180 <= i < 210 else v for i, v in enumerate(env["energy"])]   # 18~21 s 转身：总能量大
    seg = [{"start_ts": pp._fmt_ts(T0 + timedelta(seconds=20)), "end_ts": pp._fmt_ts(T0 + timedelta(seconds=31))}]
    pp.precise_boundaries(seg, env, pp.precise_bias("抓挠", 8.0, 1.0, "segment"))
    assert abs(_span(seg[0])[0] - 21.0) <= 0.11


def test_别的窗口长度不平移():
    assert pp.precise_bias("抓挠", 4.0, 1.0, "segment") == (0.0, 0.0)
    assert pp.precise_bias("抓挠", 8.0, 1.0, "majority") == (0.0, 0.0)
    assert pp.precise_bias("甩头/抖身", 8.0, 1.0, "segment") == (-0.5, 2.0)


def test_短片段不会被收没():
    seg = [{"start_ts": pp._fmt_ts(T0 + timedelta(seconds=20)), "end_ts": pp._fmt_ts(T0 + timedelta(seconds=21))}]
    pp.precise_boundaries(seg, None, pp.precise_bias("甩头/抖身", 8.0, 1.0, "segment"))
    s, e = _span(seg[0])
    assert e - s >= 0.5


def test_只管事件类():
    assert pp.is_event_for_precise("抓挠") and pp.is_event_for_precise("抓挠-头颈耳") and pp.is_event_for_precise("甩头/抖身")
    assert not pp.is_event_for_precise("静止/休息") and not pp.is_event_for_precise("活动")
