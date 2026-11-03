"""短的「未佩戴」碎片并回挨着的静止；长的、两边不是静止的不动。

    python -m pytest label_service/test_notworn_absorb.py -q
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postprocess as pp  # noqa: E402

CLASSES = ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]


def _windows(seq):
    """seq：[(类别, 窗口数)]，stride 1s、window 1s，每个窗口那一类 0.9"""
    t0 = datetime(2026, 11, 1, 0, 0, 0)
    out, k = [], 0
    for lab, n in seq:
        for _ in range(n):
            probs = {c: 0.025 for c in CLASSES}
            probs[lab] = 0.9
            out.append({"ts": (t0 + timedelta(seconds=k)).strftime(pp._TS_FMT)[:-3], "probs": probs, "label": lab})
            k += 1
    return out


def _labels_of(segs):
    return {lab: [(s["start_ts"][11:19], s["end_ts"][11:19]) for s in v] for lab, v in segs.items() if v}


def _run(seq, notworn_min_s, algo="viterbi"):
    p = pp.StableParams(notworn_min_s=notworn_min_s, min_state_s=10.0)
    return pp.stabilize(_windows(seq), CLASSES, CLASSES, 1.0, 1.0, "majority", p, algo=algo)


def test_short_notworn_inside_sleep_becomes_sleep():
    seq = [("睡觉", 120), ("未佩戴", 60), ("睡觉", 120)]
    for algo in ("viterbi", "stable"):
        on = _run(seq, 300, algo)
        assert not on.get("未佩戴"), algo
        assert len(on["睡觉"]) == 1, algo
        off = _run(seq, 0, algo)
        assert len(off["未佩戴"]) == 1, algo           # 关掉时照旧


def test_long_notworn_kept():
    seq = [("睡觉", 120), ("未佩戴", 600), ("睡觉", 120)]
    assert len(_run(seq, 300)["未佩戴"]) == 1          # 10 分钟：真摘下来了


def test_short_notworn_between_activity_kept():
    seq = [("活动", 120), ("未佩戴", 60), ("活动", 120)]
    assert len(_run(seq, 300)["未佩戴"]) == 1          # 两边不是静止：交给人判断


def test_one_side_rest_is_enough():
    seq = [("活动", 120), ("未佩戴", 60), ("睡觉", 200)]
    r = _run(seq, 300)
    assert not r.get("未佩戴") and len(r["睡觉"]) == 1
