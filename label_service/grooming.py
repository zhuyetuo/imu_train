"""疑似舔/啃候选：只靠三轴加速度、不靠模型，从整天的 IMU 里把"像在理毛"的时段挑出来。

为什么要它：想训「舔 / 啃 / 抓哪个部位」的模型，缺的是标注。标注最贵的一步是
**从 24 小时视频里找到那几分钟**。这里用 IMU 先挑候选，人只看候选那几段并
标上部位，走的还是「疑似抓挠」那条现成的确认/排除通道。

判据（全部只用加速度，陀螺仪砍掉了也一样）：

  1. **姿态偏离基准**。舔/啃时舌头和下颌的力传不到脖子，真正到项圈的是
     **头必须伸到那个部位并保持住**——项圈上的重力方向明显偏离这只狗
     平时的姿态。基准 = 前后五分钟里没在走/跑的窗口的重力方向中位数，
     所以是**相对本次佩戴、相对当时状态**的：项圈转了角度、睡着和醒着
     姿态不同，都跟得上。
  2. **动作量适中**。太小是趴着不动（睡觉/发呆，姿态偏了也不算），
     太大是走/跑（走路头也低，但整个身体在颠）。
  3. 节律只用来打分不用来卡门槛：舔有 2–4 Hz 的轻微节律，但慢舔没有。

这是**候选**，不是判断。目标是"一小时的清单人能看完"，误报靠人排除——
排除记录本身也是训练时的负样本。站着嗅地面、低头喝水会被挑进来，正常。

跟 postprocess.scratch_candidates 一样返回片段字典，多两个字段：
label（写进候选表的 label_name，默认「舔身体」）和 tilt_deg。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from label_service.postprocess import _fmt_ts, _parse_ts


@dataclass(frozen=True)
class GroomParams:
    window_s: float = 1.0
    stride_s: float = 0.5
    # 基准：前后这么多秒内、没在走/跑的窗口，重力方向的中位数（见 baseline_directions）
    baseline_span_s: float = 300.0
    # 头偏离基准姿态多少度以上才算"伸到某个部位去了"
    tilt_min_deg: float = 25.0
    # 动作量（|acc| 标准差 / 重力）的合理区间：低于 = 趴着不动，高于 = 走/跑
    motion_min: float = 0.03
    motion_max: float = 0.35
    # 一个窗口里有效样本占比下限（掉数据的窗口不看）
    valid_min: float = 0.8
    # 节律频段（舔/啃的下颌与头部小幅往复）
    rhythm_lo_hz: float = 1.5
    rhythm_hi_hz: float = 4.5
    # 成段：中间断开不超过 gap_s 就并成一段；短于 min_s 的丢掉
    gap_s: float = 2.0
    min_s: float = 3.0
    label: str = "舔身体"


def _window_stats(acc: np.ndarray, valid: np.ndarray, hz: float, p: GroomParams):
    """逐窗口：重力方向、动作量、有效占比、节律占比。返回 (starts, means, motion, vfrac, rhythm)。"""
    n_samp = max(2, int(round(p.window_s * hz)))
    step = max(1, int(round(p.stride_s * hz)))
    n = len(acc)
    if n < n_samp:
        return np.zeros(0, int), np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0)
    starts = np.arange(0, n - n_samp + 1, step)
    idx = starts[:, None] + np.arange(n_samp)[None, :]
    w = acc[idx].astype(np.float32)                       # (W, n_samp, 3)
    means = w.mean(axis=1)                                 # (W, 3)
    mag = np.linalg.norm(w, axis=2)                        # (W, n_samp)
    g = float(np.median(np.linalg.norm(means, axis=1))) or 1.0
    motion = mag.std(axis=1) / g
    vfrac = valid[idx].mean(axis=1)
    # 节律：|acc| 去均值后的功率里，1.5–4.5 Hz 占 0.5–12 Hz 的比例
    freqs = np.fft.rfftfreq(n_samp, d=1.0 / hz)
    spec = np.abs(np.fft.rfft((mag - mag.mean(axis=1, keepdims=True)) * np.hanning(n_samp), axis=1)) ** 2
    band = (freqs >= p.rhythm_lo_hz) & (freqs <= p.rhythm_hi_hz)
    wide = (freqs >= 0.5) & (freqs <= min(12.0, hz / 2))
    denom = spec[:, wide].sum(axis=1)
    rhythm = np.where(denom > 0, spec[:, band].sum(axis=1) / np.maximum(denom, 1e-12), 0.0)
    return starts, means, motion, vfrac, rhythm


def baseline_directions(means: np.ndarray, motion: np.ndarray, vfrac: np.ndarray,
                        starts: np.ndarray, hz: float, p: GroomParams) -> np.ndarray | None:
    """每个窗口的"平时姿态"（单位向量，(W,3)）：**前后 baseline_span_s 秒内**、有数据、
    又不在走/跑的窗口，重力方向的中位数。

    为什么是局部的而不是整份文件一个：一份文件里狗会睡也会醒，睡着的姿态和
    醒着的姿态差得远。整份文件取一个基准，晚上那份就成了"睡姿"——白天站着
    看看四周都算偏了 40°，一整天全是候选。取前后五分钟的中位数，基准跟着当时
    的状态走；舔一次几十秒，占不到中位数。项圈白天滑转一下也跟得上。

    走/跑的窗口不参与：走路时头是低的，狗场白天大半时间在走，让它进中位数的话
    "平时姿态"就成了走路姿态，舔的偏转量被吃掉一半。
    """
    ok = (vfrac >= p.valid_min) & (motion <= p.motion_max)
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    span = p.baseline_span_s * hz
    lo = np.searchsorted(starts[idx], starts - span, side="left")
    hi = np.searchsorted(starts[idx], starts + span, side="right")
    global_med = np.median(means[idx], axis=0)
    out = np.empty_like(means, dtype=np.float32)
    for i in range(len(means)):
        a, b = lo[i], hi[i]
        v = np.median(means[idx[a:b]], axis=0) if b > a else global_med
        n = float(np.linalg.norm(v))
        out[i] = v / n if n > 1e-6 else global_med / max(float(np.linalg.norm(global_med)), 1e-6)
    return out


def tilt_degrees(means: np.ndarray, base: np.ndarray) -> np.ndarray:
    """每个窗口的重力方向跟它自己的基准夹多少度。base 是 (W,3)，逐窗口。"""
    norms = np.linalg.norm(means, axis=1)
    cos = (means * base).sum(axis=1) / np.maximum(norms, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _bouts(flags: np.ndarray, starts: np.ndarray, hz: float, p: GroomParams) -> list[tuple[int, int]]:
    """连续为真的窗口成段，中间空隙 ≤ gap_s 的并起来，短于 min_s 的丢掉。返回 (i0, i1) 含两端。"""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    gap_samples = p.gap_s * hz
    out: list[list[int]] = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if starts[i] - starts[out[-1][1]] <= gap_samples + p.stride_s * hz:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    n_samp = p.window_s * hz
    keep = []
    for i0, i1 in out:
        dur = (starts[i1] + n_samp - starts[i0]) / hz
        if dur >= p.min_s:
            keep.append((i0, i1))
    return keep


def mine(acc: np.ndarray, ts, valid_mask, hz: float, p: GroomParams | None = None) -> list[dict]:
    """→ 候选列表（结构同 scratch_candidates 的元素，多 label / tilt_deg）。

    acc (N,3)；ts 是逐行时间（pandas Series / datetime 数组），None 就给不出时间戳、返回 []；
    valid_mask (N,) True = 这行有数据。
    """
    p = p or GroomParams()
    if ts is None or acc is None or len(acc) == 0:
        return []
    acc = np.asarray(acc, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else np.ones(len(acc), bool)
    # 六轴全 0 的占位行也当无效（跟 pool._spectral_ratio_per_window 同一套判断）
    valid = valid & ~((np.abs(acc) < 1e-9).all(axis=1))
    starts, means, motion, vfrac, rhythm = _window_stats(acc, valid, hz, p)
    if len(starts) == 0:
        return []
    base = baseline_directions(means, motion, vfrac, starts, hz, p)
    if base is None:
        return []
    tilt = tilt_degrees(means, base)
    flags = (vfrac >= p.valid_min) & (tilt >= p.tilt_min_deg) & (motion >= p.motion_min) & (motion <= p.motion_max)

    ts_vals = np.asarray(getattr(ts, "values", ts))
    n_samp = int(round(p.window_s * hz))
    out = []
    for i0, i1 in _bouts(flags, starts, hz, p):
        s, e = int(starts[i0]), min(int(starts[i1]) + n_samp, len(acc)) - 1
        t0, t1 = _to_datetime(ts_vals[s]), _to_datetime(ts_vals[e])
        if t0 is None or t1 is None or t1 <= t0:
            continue
        sel = slice(i0, i1 + 1)
        # 打分：偏得越多、节律越明显越像。只用来排序，不是概率
        score = np.clip((tilt[sel] - p.tilt_min_deg) / 45.0, 0, 1) * 0.6 + np.clip(rhythm[sel], 0, 1) * 0.4
        out.append({
            "start_ts": _fmt_ts(t0),
            "end_ts": _fmt_ts(t1),
            "conf_max": round(float(score.max()), 3),
            "conf_mean": round(float(score.mean()), 3),
            "n_windows": int(i1 - i0 + 1),
            "spec": None,
            "reason": "grooming",
            "label": p.label,
            "tilt_deg": round(float(tilt[sel].mean()), 1),
        })
    out.sort(key=lambda c: (c["conf_mean"], c["n_windows"]), reverse=True)
    return out


def _to_datetime(v):
    """逐行时间戳 → datetime。pandas 读出来的是 datetime64，NaT 给 None。
    不走 postprocess._parse_ts：那个只认 "%Y-%m-%d %H:%M:%S.%f" 这一种字符串。"""
    import pandas as pd
    try:
        t = pd.Timestamp(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    return t.to_pydatetime().replace(tzinfo=None)


def drop_overlapping(cands: list[dict], spans: list[dict]) -> list[dict]:
    """去掉跟已有片段（抓挠正式片段 / 疑似抓挠）时间上重叠的候选。

    抓挠时头也会歪过去，姿态判据会把它当成理毛；那段已经有人在看了，别重复。
    """
    taken = []
    for s in spans or []:
        a, b = _parse_ts(s.get("start_ts")), _parse_ts(s.get("end_ts"))
        if a and b:
            taken.append((a, b))
    keep = []
    for c in cands:
        a, b = _parse_ts(c["start_ts"]), _parse_ts(c["end_ts"])
        if a is None or b is None:
            continue
        if any(a < tb and b > ta for ta, tb in taken):
            continue
        keep.append(c)
    return keep


def mine_file(full_path: str, device_hz: float, p: GroomParams | None = None) -> list[dict]:
    """从 CSV 直接挖。load_csv 在 worker 里是 memoize 过的（见 pool._memoize_load_csv），
    跟频谱那一步共用一次读盘。"""
    from infer_csv_scratch import load_csv
    acc, _gyro, ts, valid_mask, _null = load_csv(full_path)
    return mine(acc, ts, valid_mask, float(device_hz), p)
