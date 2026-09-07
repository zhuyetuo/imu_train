"""
"稳定版"后处理：把逐窗口的原始预测（调试版，模型逐窗口 argmax 的真实输出）
整理成人看着可信的片段。原始输出里活动/睡觉这类"状态"经常一秒一换来回闪，
抓挠/甩身体这类"事件"又常常是孤立的单窗口噪声——这里分两套规则处理：

状态类（活动/睡觉/未佩戴 …，除 EVENT_LABELS 之外的全部）
  1. 对状态类的概率向量做滑动平均（SMOOTH_WINDOWS 个窗口），再取 argmax；
  2. 连续同状态的一段短于 MIN_STATE_S 的，并入相邻更长的那段，反复直到没有碎片。

事件类（抓挠/甩身体，EVENT_LABELS）
  1. 用原始 argmax 找出事件窗口，同类事件之间间隔 ≤ EVENT_GAP_S 的合成一个 bout；
  2. 一个 bout 要么窗口数 ≥ EVENT_MIN_WINDOWS 且平均概率 ≥ EVENT_MIN_MEAN，
     要么最高概率 ≥ EVENT_SINGLE_CONF（单窗口但特别确定），否则丢掉；
  3. 留下的事件覆盖在状态时间轴上（事件期间不算活动/睡觉）。

输入就是 /infer 返回的 windows（ts/label/conf/probs），不碰模型、不碰特征，
调试版原样保留，两个版本用同一次推理的结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass
class StableParams:
    event_labels: tuple[str, ...] = ("抓挠", "甩身体")
    smooth_windows: int = 7        # 状态概率滑动平均的窗口数（stride 1s 时 ≈ 7 秒）
    min_state_s: float = 10.0      # 状态片段最短时长，短于这个并入邻居
    event_gap_s: float = 2.0       # 同类事件之间隔多久以内合成一个 bout
    event_min_windows: int = 2     # bout 至少几个窗口
    event_min_mean: float = 0.45   # bout 内该事件的平均概率下限
    event_single_conf: float = 0.85  # 单窗口也保留的最高概率下限


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, _TS_FMT)
    except ValueError:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None


def _fmt_ts(t: datetime) -> str:
    return t.strftime(_TS_FMT)[:-3]


def _zones(ts: list[datetime], window_s: float, stride_s: float, label_mode: str) -> list[tuple[datetime, datetime]]:
    """每个窗口在时间轴上负责的区间，跟 infer_csv_scratch.window_zone 一致：
    majority 模式 = 自己起点到下一个窗口起点；center 模式 = 中心点前后各 stride/2。"""
    out = []
    n = len(ts)
    for k in range(n):
        if label_mode == "center":
            c = ts[k] + timedelta(seconds=window_s / 2)
            out.append((c - timedelta(seconds=stride_s / 2), c + timedelta(seconds=stride_s / 2)))
        else:
            end = ts[k + 1] if k + 1 < n else ts[k] + timedelta(seconds=window_s)
            out.append((ts[k], end))
    return out


def _runs(labels: list[str]) -> list[list]:
    """[[label, i0, i1], ...] 连续同标签的区间（闭区间，窗口下标）"""
    runs: list[list] = []
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])
    return runs


def _absorb_short_runs(labels: list[str], zones, min_s: float) -> list[str]:
    """把短于 min_s 的状态片段并进相邻更长的那段，直到没有碎片。"""
    labels = list(labels)
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        dur = [(zones[i1][1] - zones[i0][0]).total_seconds() for _, i0, i1 in runs]
        short = [k for k, d in enumerate(dur) if d < min_s]
        if not short:
            return labels
        k = min(short, key=lambda kk: dur[kk])   # 先处理最短的那段
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        if left is None:
            target = right
        elif right is None:
            target = left
        else:
            target = left if dur[k - 1] >= dur[k + 1] else right
        for i in range(runs[k][1], runs[k][2] + 1):
            labels[i] = target[0]


def stabilize(windows: list[dict], classes: list[str], target_labels: list[str],
              window_s: float, stride_s: float, label_mode: str,
              params: StableParams | None = None) -> dict[str, list[dict]]:
    """返回 {label: [{start_ts, end_ts, conf_max, conf_mean, n_windows}]}，跟调试版 segments 同结构。"""
    p = params or StableParams()
    n = len(windows)
    if n == 0:
        return {lab: [] for lab in target_labels}
    ts = [_parse_ts(w.get("ts")) for w in windows]
    if any(t is None for t in ts):
        # 没有时间戳没法算时长/间隔，稳定版做不了，返回空让调用方退回调试版
        return {lab: [] for lab in target_labels}
    zones = _zones(ts, window_s, stride_s, label_mode)  # type: ignore[arg-type]

    events = [lab for lab in p.event_labels if lab in classes]
    states = [lab for lab in classes if lab not in events]
    probs = [w.get("probs") or {} for w in windows]

    # ── 状态：概率滑动平均 → argmax → 吸收碎片 ──
    half = max(0, p.smooth_windows // 2)
    state_label: list[str] = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        best, best_v = None, -1.0
        for lab in states:
            v = sum(probs[j].get(lab, 0.0) for j in range(lo, hi)) / (hi - lo)
            if v > best_v:
                best, best_v = lab, v
        state_label.append(best or (states[0] if states else ""))
    state_label = _absorb_short_runs(state_label, zones, p.min_state_s)

    # ── 事件：原始 argmax 找事件窗口 → 间隙合并 → 门槛过滤 ──
    final = list(state_label)
    raw_label = [w.get("label") for w in windows]
    for ev in events:
        idx = [i for i in range(n) if raw_label[i] == ev]
        bouts: list[list[int]] = []
        for i in idx:
            if bouts and (zones[i][0] - zones[bouts[-1][-1]][1]).total_seconds() <= p.event_gap_s:
                bouts[-1].append(i)
            else:
                bouts.append([i])
        for b in bouts:
            pv = [probs[i].get(ev, 0.0) for i in b]
            mean_c, max_c = sum(pv) / len(pv), max(pv)
            keep = (len(b) >= p.event_min_windows and mean_c >= p.event_min_mean) or max_c >= p.event_single_conf
            if not keep:
                continue
            for i in range(b[0], b[-1] + 1):   # 间隙里的窗口也归进这个 bout
                final[i] = ev

    # ── 按最终标签切片段，置信度用该标签在片段内窗口的原始概率 ──
    out: dict[str, list[dict]] = {lab: [] for lab in target_labels}
    for lab, i0, i1 in _runs(final):
        if lab not in out:
            continue
        pv = [probs[i].get(lab, 0.0) for i in range(i0, i1 + 1)]
        out[lab].append({
            "start_ts": _fmt_ts(zones[i0][0]),
            "end_ts": _fmt_ts(zones[i1][1]),
            "conf_max": float(max(pv)) if pv else 0.0,
            "conf_mean": float(sum(pv) / len(pv)) if pv else 0.0,
            "n_windows": i1 - i0 + 1,
        })
    return out
