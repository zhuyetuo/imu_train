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


def mine_file(full_path: str, device_hz: float, gp=None, bp: BehaviorParams | None = None,
              split: bool = True, barks: bool = True) -> list[dict]:
    """读一次 CSV，出姿态类候选（舔 / 进食 / 饮水）+ 吠叫候选。load_csv 在 worker 里
    memoize 过，跟频谱那一步共用一次读盘。"""
    from infer_csv_scratch import load_csv
    from label_service import grooming as _grooming

    bp = bp or BehaviorParams()
    acc, _gyro, ts, valid_mask, _null = load_csv(full_path)
    hz = float(device_hz)
    classify = (lambda a, h, m, d: classify_posture_bout(a, h, m, d, bp)) if split else None
    out = _grooming.mine(acc, ts, valid_mask, hz, gp, classify=classify)
    if barks:
        out = out + mine_barks(acc, ts, valid_mask, hz, bp)
    return out
