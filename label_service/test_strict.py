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


def test_三轴节律占比不因快抓挠模长二倍频误杀(tmp_path):
    """6.5 Hz 快抓挠对模长做 FFT 会二倍频到 13 Hz 掉出 2-10 Hz；三轴功率谱之和能正确识别高节律，同时仍拦住 0.8 Hz 低频漂移。"""
    import os
    import sys
    import numpy as np
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from label_service import pool

    hz = 50.0
    n = int(hz * 20)
    t = np.arange(n) / hz
    base = np.datetime64("2026-08-21T19:59:00.000")
    ts_str = [str(base + np.timedelta64(int(round(x * 1000)), "ms")).replace("T", " ") for x in t]

    # 前 10 秒：6.5 Hz 快抓挠（正负对称振荡，模长会整流到 13 Hz）；后 10 秒：0.8 Hz 低频慢晃
    ax = np.where(t < 10.0, 1.2 * np.sin(2 * np.pi * 6.5 * t), 0.5 * np.sin(2 * np.pi * 0.8 * t))
    ay = np.zeros(n)
    az = np.ones(n) * 0.05
    gx = np.where(t < 10.0, 60.0 * np.sin(2 * np.pi * 6.5 * t), 2.0 * np.sin(2 * np.pi * 0.8 * t))
    gy = np.zeros(n)
    gz = np.zeros(n)

    csv_p = tmp_path / "fast_scratch.csv"
    with open(csv_p, "w", encoding="utf-8") as f:
        f.write("timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
        for i in range(n):
            f.write(f"{ts_str[i]},{ax[i]:.4f},{ay[i]:.4f},{az[i]:.4f},{gx[i]:.2f},{gy[i]:.2f},{gz[i]:.2f}\n")

    wins = [
        {"ts": "2026-08-21 19:59:02.000", "label": "抓挠", "conf": 0.9, "probs": {"抓挠": 0.9, "活动": 0.1}},
        {"ts": "2026-08-21 19:59:12.000", "label": "抓挠", "conf": 0.9, "probs": {"抓挠": 0.9, "活动": 0.1}},
    ]
    pool._spectral_ratio_per_window(str(csv_p), wins, window_s=4.0, device_hz=hz)
    assert wins[0]["arhythm"] is not None and wins[0]["arhythm"] > 0.7
    assert wins[1]["arhythm"] is not None and wins[1]["arhythm"] < 0.15


def test_高置信度且陀螺仪在动时豁免节律门限_但静止仍拦截():
    p = pp.StableParams(gyro_std_min=5.0, acc_rhythm_min=0.25)
    # 1. 高置信(0.92) + 陀螺仪正常在动(60) + 节律偏低(0.15) -> 豁免 acc_rhythm_min，不误杀
    w_hi_move = _windows([(20, 30, "抓挠", 0.92)], gstd=60.0, arhythm=0.15)
    assert _n(_run(w_hi_move, "viterbi", p)) == 1
    assert _n(_run(w_hi_move, "viterbi_strict", p)) == 1
    # 2. 高置信(0.92) + 陀螺仪静止(1.0) -> 仍被 gyro_std_min 拦住（设备没戴/放桌上）
    w_hi_still = _windows([(20, 30, "抓挠", 0.92)], gstd=1.0, arhythm=0.15)
    assert _n(_run(w_hi_still, "viterbi", p)) == 0
    assert _n(_run(w_hi_still, "viterbi_strict", p)) == 0
    # 3. 中等置信(0.72) + 节律偏低(0.15) -> 不触发豁免，照常由 acc_rhythm_min 拦住
    w_mid_low_r = _windows([(20, 30, "抓挠", 0.72)], gstd=60.0, arhythm=0.15)
    assert _n(_run(w_mid_low_r, "viterbi", p)) == 0
    assert _n(_run(w_mid_low_r, "viterbi_strict", p)) == 0
