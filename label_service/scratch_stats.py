"""
抓挠日统计的纯计算部分，照抄 src/imu_scratch_daily_stats.py 里的常量和函数
（_day_features / _cluster_count / _sleep_disruption_count / _delta_score / _union_seconds）。

为什么不直接 import 那个模块：它顶部 `from extract_clips import ...`，而 extract_clips
在 import 时就 subprocess 跑 ffmpeg 探测 CUDA，服务进程里没有 ffmpeg 就直接炸。
这里只要几个纯函数，复制一份最稳；改口径时两边一起改。
"""

from datetime import timedelta

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 6
CLUSTER_EVENTS_PER_HOUR = 5
INTERRUPT_MERGE_GAP_MINUTES = 5
LONG_SCRATCH_RED_FLAG_SEC = 60
MIN_GOOD_WEAR_HOURS = 12

_DELTA_TIERS = [
    (30, 20, 15 * 60, 3.0),
    (20, 10, 10 * 60, 2.0),
    (10, 5, 5 * 60, 1.5),
    (5, 3, 3 * 60, 1.3),
]
BASELINE_DENOM_FLOOR = 3


def union_seconds(spans) -> float:
    """[(start, end), ...] → 区间并集的总秒数，重叠的录制时段只算一次。"""
    if not spans:
        return 0.0
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return sum((end - start).total_seconds() for start, end in merged)


def cluster_count(starts) -> int:
    """1小时滑窗内事件数 >= CLUSTER_EVENTS_PER_HOUR 记一个聚集时段。"""
    starts = sorted(starts)
    n = len(starts)
    clusters = 0
    i = 0
    while i < n:
        window_end = starts[i] + timedelta(hours=1)
        j = i
        while j < n and starts[j] < window_end:
            j += 1
        if j - i >= CLUSTER_EVENTS_PER_HOUR:
            clusters += 1
            i = j
        else:
            i += 1
    return clusters


def sleep_disruption_count(night_events) -> int:
    """相邻夜间事件间隔 < INTERRUPT_MERGE_GAP_MINUTES 视为同一次中断，合并计数。"""
    if not night_events:
        return 0
    events = sorted(night_events)
    count = 1
    prev_end = events[0][1]
    for start, end in events[1:]:
        gap = (start - prev_end).total_seconds() / 60
        if gap >= INTERRUPT_MERGE_GAP_MINUTES:
            count += 1
        prev_end = end
    return count


def day_features(events) -> dict:
    """events: [(start_dt, end_dt), ...]，一个 (imu, date) 当天的全部抓挠事件。"""
    starts = [s for s, _ in events]
    durations = [(e - s).total_seconds() for s, e in events]
    is_night = lambda dt: dt.hour >= NIGHT_START_HOUR or dt.hour < NIGHT_END_HOUR  # noqa: E731
    night_events = [(s, e) for s, e in events if is_night(s)]

    event_count = len(events)
    total_duration_sec = sum(durations)
    max_event_duration_sec = max(durations) if durations else 0.0
    zd = sleep_disruption_count(night_events)
    return {
        "event_count": event_count,
        "total_duration_sec": total_duration_sec,
        "total_duration_min": round(total_duration_sec / 60, 2),
        "max_event_duration_sec": round(max_event_duration_sec, 1),
        "cluster_count": cluster_count(starts) if starts else 0,
        "night_event_count": len(night_events),
        "zn": max(0, event_count - zd),
        "zd": zd,
        "long_scratch": max_event_duration_sec >= LONG_SCRATCH_RED_FLAG_SEC,
    }


def delta_score(current_count, baseline_count, current_dur_min, baseline_dur_min) -> int:
    def one(current, baseline, is_duration):
        denom = max(baseline, BASELINE_DENOM_FLOOR)
        ratio = current / denom
        abs_increase = current - baseline
        for score, abs_min_count, abs_min_dur_sec, ratio_min in _DELTA_TIERS:
            abs_min = (abs_min_dur_sec / 60) if is_duration else abs_min_count
            if abs_increase >= abs_min and ratio >= ratio_min:
                return score
        return 0
    return max(one(current_count, baseline_count, False), one(current_dur_min, baseline_dur_min, True))
