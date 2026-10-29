"""疑似进食 / 饮水 / 吠叫候选：只靠三轴加速度的规则，给标注员一个起点，不是判断。

跟 grooming.py（疑似舔/啃）是一套思路、一条通道：规则挑段 → 平台「疑似片段」里
人工确认 / 改类别 / 排除。目标是"不用从零翻 24 小时视频"，误报靠人排，排除记录
本身就是以后训真模型的负样本。**这些规则没有用标注数据调过**（还没有这几类的标注），
阈值都是按动作的物理特点给的初值，攒够确认 / 排除之后要回来重调。

## 进食 / 饮水：从「头伸出去并保持住」那批段里分出来

grooming.mine 挑的是"头明显偏离平时姿态、动作量适中"的段——舔身体、低头吃饭、
低头喝水在这一步上长得一样（grooming.py 顶上就写了"低头喝水会被挑进来"）。
所以不另挑一遍，而是给同一批段**换一个更像的类别**：

  饮水  舔水是很规整的 3–5.5 Hz 往复（狗一秒舔三四下），一段里节律集中在一个频率上，
        动作量小，一般不超过两三分钟
  进食  咀嚼 + 低头叼食是 0.7–2 Hz 的慢往复（舔身体一般 2–4 Hz，所以上限压在 2），
        一顿饭一般十几秒以上
  其余  还是「舔」（原来的类别）

段一条没多也一条没少——只是标签猜得更像，猜错了人点「改成别的」就行。
GROOM_SPLIT_ENABLED=0 就退回全叫「舔」。

## 吠叫：一串孤立的冲击

每叫一声，胸腔收缩 + 头往前一顶，项圈上是一个几十毫秒宽的尖脉冲；连叫是
0.2–2 秒一声。跟别的动作的区别是**脉冲之间是安静的**：

  走 / 跑    也有周期性的峰（落脚），但峰之间一直在动——底噪高
  甩身体     4–6 Hz 连续大幅振荡，没有安静的间隔
  抓挠       已经有抓挠片段 / 疑似抓挠盖着，重叠的去掉

所以判据是：高通后的 |acc| 上找比周围底噪高很多倍的尖峰，连着 ≥3 个、间隔合理、
峰之间底噪低。只用加速度（50 Hz 足够看到几十毫秒的峰）。

## 甩头 / 抖身：短促、大幅、规整的 3–9 Hz 振荡

狗甩水是 4–6 Hz（小狗更快）的左右扭转，甩头更短更快。项圈上是**每个轴**都在
大幅往复（不看 |acc|：左右扭转时 |acc| 的频率会翻倍），而且很短——一两秒就停。
跟别的动作的区别：

  走 / 跑    也有周期，但步频 1.5–3.5 Hz，能量不集中在 3–9 Hz；而且一走就是好久
  抓挠       已经有抓挠片段 / 疑似抓挠盖着，重叠的去掉
  吠叫       是孤立的尖峰，不是连续振荡

判据：逐 1 秒窗口，高通后三轴合起来的振幅够大、3–9 Hz 能量占大头；连续成段，
0.75–5 秒（再长多半是在跑）。类别给「甩头/抖身」，甩头还是抖身由人选。

## 如厕（排尿 / 排便）：走着走着停下来、换个姿势、不动一会儿、又走了

**这个判据很弱，误报会多**：项圈在脖子上，蹲下 / 抬腿时脖子的姿态变化不大，
跟坐下、站着发呆、低头闻长得很像。能用的只有「上下文」：

  - 停下来之前和之后都在走（如厕前常转圈闻地，完事就走开）
  - 停下来那段几乎不动，姿态偏离平时（蹲 / 抬腿 / 弓背）
  - 4–90 秒：排尿一般十几秒以内，排便更久

按时长猜排尿 / 排便（< 20 秒猜排尿），猜错了人改。坐下歇一会儿再走、趴下又
起来会被挑进来——误报靠人排，排除记录攒够了再回来调。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from label_service.postprocess import _fmt_ts


@dataclass(frozen=True)
class BehaviorParams:
    # ── 姿态段分类（进食 / 饮水）──
    drink_f_lo: float = 3.0          # 舔水节律（舔身体常在 2–3 Hz，下限压在 3）
    drink_f_hi: float = 5.5
    drink_periodicity_min: float = 0.30   # 主频 ±0.3 Hz 内的能量占 0.5–8 Hz 的比例
    drink_motion_max: float = 0.20
    drink_max_s: float = 180.0
    eat_f_lo: float = 0.7            # 咀嚼 / 叼食节律
    eat_f_hi: float = 2.0
    eat_min_s: float = 15.0
    eat_motion_min: float = 0.05
    label_drink: str = "饮水"
    label_eat: str = "进食"
    # ── 吠叫 ──
    bark_hp_s: float = 0.2           # 减掉这么长的滑动均值 = 高通，留下尖峰
    bark_peak_min_g: float = 0.12    # 峰至少这么高（g）
    bark_peak_k: float = 6.0         # 且至少是周围底噪的这么多倍
    bark_noise_s: float = 2.0        # 底噪 = 前后这么长里 |高通| 的中位数
    bark_width_max_s: float = 0.12   # 半高宽上限：叫一声是尖脉冲，不是一个鼓包
    bark_ioi_min_s: float = 0.2      # 两声之间至少隔这么久
    bark_ioi_max_s: float = 2.0      # 超过这么久算另一串
    bark_min_n: int = 3              # 一串至少几声
    bark_quiet_max_g: float = 0.05   # 一串里峰以外的 |高通| 中位数上限（走 / 跑会超）
    bark_pad_s: float = 0.5          # 段前后各留一点
    label_bark: str = "吠叫"
    # ── 甩头 / 抖身 ──
    shake_window_s: float = 1.0
    shake_stride_s: float = 0.25
    shake_hp_s: float = 0.4          # 减掉这么长的滑动均值（逐轴）= 去掉重力和姿态
    shake_f_lo: float = 3.0
    shake_f_hi: float = 9.0
    shake_rms_min_g: float = 0.35    # 三轴高通后的 RMS 至少这么大（甩是大动作）
    shake_band_min: float = 0.6      # 3–9 Hz 能量占 0.5–12 Hz 的比例下限（走 / 跑过不了）
    shake_gap_s: float = 0.5
    shake_min_s: float = 0.75
    shake_max_s: float = 5.0         # 再长多半是在跑
    label_shake: str = "甩头/抖身"
    # ── 如厕 ──
    toilet_still_max: float = 0.05   # 那段的动作量上限（|acc| 标准差 / 重力）
    toilet_tilt_min_deg: float = 15.0
    toilet_walk_min: float = 0.08    # 前后"在走"的动作量下限（低头闻地慢慢走也要算上）
    toilet_ctx_s: float = 20.0       # 前后各看这么长
    toilet_ctx_walk_s: float = 3.0   # 前后各至少走了这么久
    toilet_min_s: float = 4.0
    toilet_max_s: float = 90.0
    toilet_feces_min_s: float = 20.0  # 不短于这个猜排便，短的猜排尿
    label_urinate: str = "疑似如厕-排尿"
    label_defecate: str = "疑似如厕-排便"


def _moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    if n == 1 or len(x) < n:
        return np.full_like(x, x.mean() if len(x) else 0.0)
    c = np.cumsum(np.concatenate([[0.0], x.astype(np.float64)]))
    out = (c[n:] - c[:-n]) / n
    pad_l = (n - 1) // 2
    return np.concatenate([np.full(pad_l, out[0]), out, np.full(len(x) - len(out) - pad_l, out[-1])]).astype(np.float32)


def rhythm_of(mag: np.ndarray, hz: float) -> tuple[float, float]:
    """一段 |acc| 的主频和"节律集中度"（主频 ±0.3 Hz 的能量 / 0.5–8 Hz 的能量）。
    Welch：2 秒一段、半重叠，段内去均值加 Hann——一顿饭几十秒，整段一次 FFT 的话
    慢漂移会把低频糊满。"""
    seg = int(round(2.0 * hz))
    x = np.asarray(mag, dtype=np.float64)
    if len(x) < seg:
        seg = len(x)
    if seg < 8:
        return 0.0, 0.0
    step = max(1, seg // 2)
    win = np.hanning(seg)
    acc = None
    for s in range(0, len(x) - seg + 1, step):
        w = x[s:s + seg]
        p = np.abs(np.fft.rfft((w - w.mean()) * win)) ** 2
        acc = p if acc is None else acc + p
    freqs = np.fft.rfftfreq(seg, d=1.0 / hz)
    band = (freqs >= 0.5) & (freqs <= min(8.0, hz / 2))
    if acc is None or not band.any() or acc[band].sum() <= 0:
        return 0.0, 0.0
    i = int(np.flatnonzero(band)[np.argmax(acc[band])])
    f0 = float(freqs[i])
    near = band & (np.abs(freqs - f0) <= 0.3)
    return f0, float(acc[near].sum() / acc[band].sum())


def classify_posture_bout(acc: np.ndarray, hz: float, motion_mean: float, dur_s: float,
                          p: BehaviorParams | None = None) -> tuple[str, str, dict] | None:
    """grooming 挑出来的一段（acc 是这段的原始加速度）→ (类别, reason, 线索)；
    返回 None = 不像进食也不像饮水，保持原来的「舔」。"""
    p = p or BehaviorParams()
    mag = np.linalg.norm(np.asarray(acc, dtype=np.float32), axis=1)
    f0, conc = rhythm_of(mag, hz)
    info = {"f0_hz": round(f0, 2), "rhythm": round(conc, 3)}
    if (p.drink_f_lo <= f0 <= p.drink_f_hi and conc >= p.drink_periodicity_min
            and motion_mean <= p.drink_motion_max and dur_s <= p.drink_max_s):
        return p.label_drink, "drinking", info
    if p.eat_f_lo <= f0 <= p.eat_f_hi and dur_s >= p.eat_min_s and motion_mean >= p.eat_motion_min:
        return p.label_eat, "eating", info
    return None


def bark_peaks(acc: np.ndarray, valid: np.ndarray, hz: float, p: BehaviorParams) -> tuple[np.ndarray, np.ndarray]:
    """→ (像「叫了一声」的尖峰的样本下标（升序）, 高通后的 |acc|)。"""
    mag = np.linalg.norm(np.asarray(acc, dtype=np.float32), axis=1)
    mag = np.where(valid, mag, 0.0).astype(np.float32)
    hp = np.abs(mag - _moving_mean(mag, round(p.bark_hp_s * hz)))
    hp[~valid] = 0.0
    # 底噪：分块中位数再插回去（逐点滚动中位数在一整天的数据上太慢）
    blk = max(1, int(round(p.bark_noise_s * hz)))
    n_blk = int(np.ceil(len(hp) / blk))
    meds = np.array([np.median(hp[i * blk:(i + 1) * blk]) for i in range(n_blk)], dtype=np.float32)
    noise = np.repeat(meds, blk)[: len(hp)]
    thr = np.maximum(p.bark_peak_min_g, p.bark_peak_k * noise)
    # 局部极大且过门槛
    cand = np.flatnonzero((hp[1:-1] >= thr[1:-1]) & (hp[1:-1] >= hp[:-2]) & (hp[1:-1] > hp[2:])) + 1
    half_w = int(round(p.bark_width_max_s * hz / 2)) + 1
    min_gap = max(1, int(round(p.bark_ioi_min_s * hz)))
    out: list[int] = []
    for i in cand:
        # 半高宽：峰两边各 half_w 个点之外要掉到峰高一半以下，不然是个鼓包（走路落脚、翻身）
        lo, hi = max(0, i - 3 * half_w), min(len(hp), i + 3 * half_w + 1)
        above = np.flatnonzero(hp[lo:hi] >= hp[i] / 2) + lo
        if len(above) and (above[-1] - above[0] + 1) > 2 * half_w:
            continue
        if out and i - out[-1] < min_gap:
            if hp[i] > hp[out[-1]]:
                out[-1] = int(i)
            continue
        out.append(int(i))
    return np.asarray(out, dtype=int), hp


def mine_barks(acc: np.ndarray, ts, valid_mask, hz: float, p: BehaviorParams | None = None) -> list[dict]:
    """→ 疑似吠叫候选（结构同 grooming.mine 的元素，reason="barking"）。"""
    from label_service.grooming import _to_datetime

    p = p or BehaviorParams()
    if ts is None or acc is None or len(acc) < int(hz * 2):
        return []
    acc = np.asarray(acc, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else np.ones(len(acc), bool)
    valid = valid & ~((np.abs(acc) < 1e-9).all(axis=1))
    peaks, hp = bark_peaks(acc, valid, hz, p)
    if len(peaks) < p.bark_min_n:
        return []
    max_gap = p.bark_ioi_max_s * hz
    runs: list[list[int]] = [[int(peaks[0])]]
    for i in peaks[1:]:
        if i - runs[-1][-1] <= max_gap:
            runs[-1].append(int(i))
        else:
            runs.append([int(i)])
    ts_vals = np.asarray(getattr(ts, "values", ts))
    pad = int(round(p.bark_pad_s * hz))
    excl = int(round(p.bark_width_max_s * hz))
    out = []
    for r in runs:
        if len(r) < p.bark_min_n:
            continue
        a, b = r[0], r[-1]
        # 峰之间要安静：挖掉每个峰附近一小截，剩下的 |高通| 中位数要低（走 / 跑 / 甩身体过不了）
        mask = np.ones(b - a + 1, bool)
        for k in r:
            mask[max(0, k - excl - a): k + excl + 1 - a] = False
        between = hp[a:b + 1][mask]
        quiet = float(np.median(between)) if len(between) else 0.0
        if quiet > p.bark_quiet_max_g:
            continue
        s, e = max(0, a - pad), min(len(acc) - 1, b + pad)
        t0, t1 = _to_datetime(ts_vals[s]), _to_datetime(ts_vals[e])
        if t0 is None or t1 is None or t1 <= t0:
            continue
        heights = hp[r]
        # 打分：声数越多、峰比底噪越突出越像。只用来排序
        score = min(1.0, len(r) / 8.0) * 0.5 + float(np.clip(np.median(heights) / max(quiet, 1e-3) / 40.0, 0, 1)) * 0.5
        out.append({
            "start_ts": _fmt_ts(t0),
            "end_ts": _fmt_ts(t1),
            "conf_max": round(score, 3),
            "conf_mean": round(score, 3),
            "n_windows": len(r),
            "spec": None,
            "reason": "barking",
            "label": p.label_bark,
            "n_barks": len(r),
        })
    out.sort(key=lambda c: (c["conf_mean"], c["n_barks"]), reverse=True)
    return out


def _valid_rows(acc: np.ndarray, valid_mask) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool) if valid_mask is not None else np.ones(len(acc), bool)
    return valid & ~((np.abs(acc) < 1e-9).all(axis=1))


def _runs(flags: np.ndarray, starts: np.ndarray, hz: float, gap_s: float, win_s: float,
          min_s: float, max_s: float) -> list[tuple[int, int]]:
    """连续为真的窗口成段（空隙 ≤ gap_s 并起来），只留 min_s–max_s 的。(i0, i1) 含两端。"""
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    out: list[list[int]] = [[int(idx[0]), int(idx[0])]]
    for i in idx[1:]:
        if (starts[i] - starts[out[-1][1]]) / hz <= gap_s + win_s:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    keep = []
    for i0, i1 in out:
        dur = (starts[i1] - starts[i0]) / hz + win_s
        if min_s <= dur <= max_s:
            keep.append((i0, i1))
    return keep


def _cand(ts_vals, s: int, e: int, score: float, n: int, reason: str, label: str, **info) -> dict | None:
    from label_service.grooming import _to_datetime

    t0, t1 = _to_datetime(ts_vals[s]), _to_datetime(ts_vals[e])
    if t0 is None or t1 is None or t1 <= t0:
        return None
    return {"start_ts": _fmt_ts(t0), "end_ts": _fmt_ts(t1), "conf_max": round(score, 3),
            "conf_mean": round(score, 3), "n_windows": n, "spec": None, "reason": reason,
            "label": label, **info}


def mine_shakes(acc: np.ndarray, ts, valid_mask, hz: float, p: BehaviorParams | None = None) -> list[dict]:
    """→ 疑似甩头/抖身候选（reason="shaking"）。"""
    p = p or BehaviorParams()
    n_samp = max(4, int(round(p.shake_window_s * hz)))
    if ts is None or acc is None or len(acc) < n_samp:
        return []
    acc = np.asarray(acc, dtype=np.float32)
    valid = _valid_rows(acc, valid_mask)
    k = round(p.shake_hp_s * hz)
    hp = np.stack([acc[:, j] - _moving_mean(acc[:, j], k) for j in range(3)], axis=1)
    hp[~valid] = 0.0
    step = max(1, int(round(p.shake_stride_s * hz)))
    starts = np.arange(0, len(acc) - n_samp + 1, step)
    idx = starts[:, None] + np.arange(n_samp)[None, :]
    w = hp[idx]                                                  # (W, n, 3)
    rms = np.sqrt((w ** 2).sum(axis=2).mean(axis=1))
    vfrac = valid[idx].mean(axis=1)
    freqs = np.fft.rfftfreq(n_samp, d=1.0 / hz)
    spec = (np.abs(np.fft.rfft((w - w.mean(axis=1, keepdims=True)) * np.hanning(n_samp)[None, :, None],
                               axis=1)) ** 2).sum(axis=2)       # 三轴功率加起来
    band = (freqs >= p.shake_f_lo) & (freqs <= min(p.shake_f_hi, hz / 2))
    wide = (freqs >= 0.5) & (freqs <= min(12.0, hz / 2))
    denom = spec[:, wide].sum(axis=1)
    frac = np.where(denom > 0, spec[:, band].sum(axis=1) / np.maximum(denom, 1e-12), 0.0)
    flags = (vfrac >= 0.8) & (rms >= p.shake_rms_min_g) & (frac >= p.shake_band_min)
    ts_vals = np.asarray(getattr(ts, "values", ts))
    out = []
    for i0, i1 in _runs(flags, starts, hz, p.shake_gap_s, p.shake_window_s, p.shake_min_s, p.shake_max_s):
        sel = slice(i0, i1 + 1)
        f0 = float(freqs[band][np.argmax(spec[sel][:, band].sum(axis=0))]) if band.any() else 0.0
        # 打分：越猛、能量越集中在甩的频段越像。只用来排序
        score = float(np.clip(rms[sel].max() / 1.5, 0, 1)) * 0.5 + float(frac[sel].mean()) * 0.5
        c = _cand(ts_vals, int(starts[i0]), min(int(starts[i1]) + n_samp, len(acc)) - 1, score,
                  int(i1 - i0 + 1), "shaking", p.label_shake, f0_hz=round(f0, 1),
                  rms_g=round(float(rms[sel].max()), 2))
        if c:
            out.append(c)
    out.sort(key=lambda c: c["conf_mean"], reverse=True)
    return out


def mine_toilet(acc: np.ndarray, ts, valid_mask, hz: float, gp=None,
                p: BehaviorParams | None = None) -> list[dict]:
    """→ 疑似如厕候选（reason="toileting"），类别按时长猜排尿 / 排便。"""
    from label_service import grooming as _g

    p = p or BehaviorParams()
    gp = gp or _g.GroomParams()
    if ts is None or acc is None or len(acc) == 0:
        return []
    acc = np.asarray(acc, dtype=np.float32)
    valid = _valid_rows(acc, valid_mask)
    starts, means, motion, vfrac, _rhythm = _g._window_stats(acc, valid, hz, gp)
    if len(starts) == 0:
        return []
    base = _g.baseline_directions(means, motion, vfrac, starts, hz, gp)
    if base is None:
        return []
    tilt = _g.tilt_degrees(means, base)
    ok = vfrac >= gp.valid_min
    still = ok & (motion <= p.toilet_still_max) & (tilt >= p.toilet_tilt_min_deg)
    walking = ok & (motion >= p.toilet_walk_min)
    ctx = int(round(p.toilet_ctx_s / gp.stride_s))
    need = int(np.ceil(p.toilet_ctx_walk_s / gp.stride_s))
    ts_vals = np.asarray(getattr(ts, "values", ts))
    n_samp = int(round(gp.window_s * hz))
    out = []
    for i0, i1 in _runs(still, starts, hz, gp.stride_s * 2, gp.window_s, p.toilet_min_s, p.toilet_max_s):
        before = int(walking[max(0, i0 - ctx):i0].sum())
        after = int(walking[i1 + 1:i1 + 1 + ctx].sum())
        if before < need or after < need:
            continue
        s, e = int(starts[i0]), min(int(starts[i1]) + n_samp, len(acc)) - 1
        dur = (e - s) / hz
        sel = slice(i0, i1 + 1)
        label = p.label_defecate if dur >= p.toilet_feces_min_s else p.label_urinate
        # 打分：姿态偏得越多、前后走得越明显越像。只用来排序
        score = (float(np.clip((tilt[sel].mean() - p.toilet_tilt_min_deg) / 45.0, 0, 1)) * 0.5
                 + min(1.0, (before + after) / (4.0 * need)) * 0.5)
        c = _cand(ts_vals, s, e, score, int(i1 - i0 + 1), "toileting", label,
                  tilt_deg=round(float(tilt[sel].mean()), 1))
        if c:
            out.append(c)
    out.sort(key=lambda c: c["conf_mean"], reverse=True)
    return out


def mine_file(full_path: str, device_hz: float, gp=None, bp: BehaviorParams | None = None,
              split: bool = True, barks: bool = True, shakes: bool = True,
              toilet: bool = True) -> list[dict]:
    """读一次 CSV，出姿态类候选（舔 / 进食 / 饮水）+ 吠叫 / 甩头抖身 / 如厕候选。
    load_csv 在 worker 里 memoize 过，跟频谱那一步共用一次读盘。"""
    from infer_csv_scratch import load_csv
    from label_service import grooming as _grooming

    bp = bp or BehaviorParams()
    acc, _gyro, ts, valid_mask, _null = load_csv(full_path)
    hz = float(device_hz)
    classify = (lambda a, h, m, d: classify_posture_bout(a, h, m, d, bp)) if split else None
    out = _grooming.mine(acc, ts, valid_mask, hz, gp, classify=classify)
    if barks:
        out = out + mine_barks(acc, ts, valid_mask, hz, bp)
    if shakes:
        out = out + mine_shakes(acc, ts, valid_mask, hz, bp)
    if toilet:
        out = out + mine_toilet(acc, ts, valid_mask, hz, gp, bp)
    return out
