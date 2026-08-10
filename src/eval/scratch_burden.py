"""
Scratch Burden Score 计算引擎，按 docs/skin_health.md / Wardyn V0.4 算法定义文档实现。

输入是抓挠事件列表（起止时间），不关心事件是怎么识别出来的——可以是合成测试数据，
也可以是 infer_csv_scratch.py 产出的 *_infer.json（用 load_events_from_infer_json 加载）。

核心流程：事件 → 按天聚合特征 → 个人基线（跳过最近7天）→ 四个子项打分 + 红旗信号 → 总分 → C0/C1/C2。

个人基线建立前（设备刚绑定，还不满21天历史）不强行用不完整数据凑基线，"变化幅度"分项
改用 Whistle FIT 已用 Pruritus VAS 验证过的绝对分级兜底（见 BOOTSTRAP_* 常量），
"聚集程度"和"长时间抓挠红旗"本身不依赖基线，从第一天就正常生效；只有"持续程度"
（概念上依赖基线偏离的连续性）在引导期内不计分。基线一旦建立就切回正常的相对基线评分。
"""
import json
from datetime import timedelta
from statistics import median

import pandas as pd

CLUSTER_EVENTS_PER_HOUR = 5      # 1小时内≥N次抓挠事件记为1个聚集时段
BASELINE_LOOKBACK_START = 8      # 基线窗口起点：d-8
BASELINE_LOOKBACK_END = 21       # 基线窗口终点：d-21（含），跳过最近7天
MIN_BASELINE_DAYS = 7            # 基线窗口内至少要有几天有效数据才建立基线
PERSISTENCE_ENTRY_SCORE = 10     # 触发"当日算作持续"的变化幅度门槛
INTERRUPT_SLEEP_MIN_MINUTES = 5
INTERRUPT_EVENT_MIN_SEC = 3
INTERRUPT_MERGE_GAP_MINUTES = 5  # 相邻中断事件合并间隔
LONG_SCRATCH_RED_FLAG_SEC = 60   # 连续/高聚集抓挠达到1分钟 -> 红旗
BASELINE_DENOM_FLOOR = 1         # 变化幅度比值分母下限，只为避免除以0，不是用来压低低基线狗信号的

# 基线未建立期间的绝对阈值兜底（来自 Whistle FIT，已用 Pruritus VAS 验证，见 docs/skin_health.md §6）
BOOTSTRAP_ELEVATED_SEC_PER_DAY = 120   # 对应 Whistle "Elevated" 档下限
BOOTSTRAP_SEVERE_SEC_PER_DAY = 300     # 对应 Whistle "Severe" 档下限
BOOTSTRAP_ELEVATED_SCORE = 10
BOOTSTRAP_SEVERE_SCORE = 30


# ── 事件加载 ──────────────────────────────────────────────────────────

def load_events_from_infer_json(json_paths, pet_id):
    """从 infer_csv_scratch.py 产出的 *_infer.json 里读 scratch_segments，转成统一的事件表。"""
    rows = []
    for p in json_paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for seg in data.get("scratch_segments", []):
            start = pd.to_datetime(seg["start_ts"])
            end = pd.to_datetime(seg["end_ts"])
            rows.append({
                "pet_id": pet_id,
                "start": start,
                "end": end,
                "duration_sec": (end - start).total_seconds(),
            })
    return pd.DataFrame(rows, columns=["pet_id", "start", "end", "duration_sec"])


# ── 按天聚合 ──────────────────────────────────────────────────────────

def _cluster_segments(day_events):
    """1小时滑窗内事件数>=CLUSTER_EVENTS_PER_HOUR，记一个聚集时段；聚集窗口内的事件消耗掉，避免重复计数。"""
    starts = sorted(day_events["start"].tolist())
    clusters = 0
    i = 0
    n = len(starts)
    while i < n:
        window_end = starts[i] + timedelta(hours=1)
        j = i
        while j < n and starts[j] < window_end:
            j += 1
        count = j - i
        if count >= CLUSTER_EVENTS_PER_HOUR:
            clusters += 1
            i = j  # 跳过已经计入这个聚集时段的事件
        else:
            i += 1
    return clusters


def _sleep_interruptions(night_events):
    """相邻夜间抓挠事件，间隔<INTERRUPT_MERGE_GAP_MINUTES 视为未重新稳定入睡，合并为同一次中断。

    简化实现：只用事件时间序列的间隔做判断，不依赖独立的睡眠状态识别流；
    生产环境接入真实睡眠分类结果后，需要额外校验"抓挠前稳定睡眠>=5分钟"这个前置条件。
    """
    starts = sorted(night_events["start"].tolist())
    ends = sorted(night_events["end"].tolist())
    if not starts:
        return 0
    events = sorted(zip(night_events["start"], night_events["end"]))
    count = 1
    prev_end = events[0][1]
    for start, end in events[1:]:
        gap = (start - prev_end).total_seconds() / 60
        if gap >= INTERRUPT_MERGE_GAP_MINUTES:
            count += 1
        prev_end = end
    return count


def daily_features(events, wear_hours, night_start="22:00", night_end="06:00"):
    """
    events: DataFrame[pet_id, start, end, duration_sec]
    wear_hours: DataFrame[pet_id, date, valid_wear_hours]，date 为 python date
    返回按 (pet_id, date) 聚合的日特征表。
    """
    events = events.copy()
    events["date"] = events["start"].dt.date
    events["hour"] = events["start"].dt.hour
    is_night = events["hour"].apply(lambda h: h >= 22 or h < 6)
    events["is_night"] = is_night

    rows = []
    for (pet_id, date), grp in events.groupby(["pet_id", "date"]):
        wear_row = wear_hours[(wear_hours["pet_id"] == pet_id) & (wear_hours["date"] == date)]
        wear = float(wear_row["valid_wear_hours"].iloc[0]) if len(wear_row) else 0.0
        night_grp = grp[grp["is_night"]]
        rows.append({
            "pet_id": pet_id,
            "date": date,
            "event_count": len(grp),
            "total_duration_sec": grp["duration_sec"].sum(),
            "max_event_duration_sec": grp["duration_sec"].max() if len(grp) else 0.0,
            "valid_wear_hours": wear,
            "cluster_count": _cluster_segments(grp) if len(grp) else 0,
            "sleep_disruption_count": _sleep_interruptions(night_grp) if len(night_grp) else 0,
            "night_ratio": len(night_grp) / len(grp) if len(grp) else 0.0,
        })

    # 把有佩戴但没有抓挠事件的日子也补上（event_count=0），基线计算需要完整日历
    all_days = wear_hours[["pet_id", "date", "valid_wear_hours"]].drop_duplicates()
    df = all_days.merge(
        pd.DataFrame(rows).drop(columns=["valid_wear_hours"], errors="ignore"),
        on=["pet_id", "date"], how="left",
    )
    for col in ["event_count", "total_duration_sec", "max_event_duration_sec",
                "cluster_count", "sleep_disruption_count", "night_ratio"]:
        df[col] = df[col].fillna(0)
    df["data_quality_flag"] = df["valid_wear_hours"].apply(
        lambda h: "good" if h >= 12 else ("partial" if h > 0 else "insufficient")
    )
    return df.sort_values(["pet_id", "date"]).reset_index(drop=True)


# ── 个人基线（§2.3：跳过最近7天，取此前14天中位数） ──────────────────────

def compute_baseline(daily_df, pet_id, target_date):
    dates = sorted(daily_df.loc[daily_df.pet_id == pet_id, "date"].unique())
    idx = {d: i for i, d in enumerate(dates)}
    if target_date not in idx:
        return None
    t = idx[target_date]
    lo, hi = t - BASELINE_LOOKBACK_END, t - BASELINE_LOOKBACK_START
    if lo < 0:
        return None  # 历史不足21天

    sub = daily_df[(daily_df.pet_id == pet_id)
                    & (daily_df.date >= dates[lo]) & (daily_df.date <= dates[hi])]
    sub = sub[sub.data_quality_flag == "good"]
    if len(sub) < MIN_BASELINE_DAYS:
        return None  # 有效基线天数不足

    return {
        "baseline_event_count": median(sub.event_count),
        "baseline_duration_sec": median(sub.total_duration_sec),
        "n_baseline_days": len(sub),
    }


# ── §2.1/2.3 变化幅度打分（双门槛，AND 语义，取满足条件的最高档；count/duration 取高分） ──

_DELTA_TIERS = [
    # (score, abs_min_count, abs_min_duration_sec, ratio_min)
    (30, 20, 15 * 60, 3.0),
    (20, 10, 10 * 60, 2.0),
    (10, 5, 5 * 60, 1.5),
    (5, 3, 3 * 60, 1.3),
]


def _delta_score_one(current, baseline, abs_min_idx, ratio_min_idx):
    denom = max(baseline, BASELINE_DENOM_FLOOR)
    ratio = current / denom
    abs_increase = current - baseline
    for score, abs_min_count, abs_min_dur, ratio_min in _DELTA_TIERS:
        abs_min = abs_min_dur if abs_min_idx == "duration" else abs_min_count
        if abs_increase >= abs_min and ratio >= ratio_min:
            return score, ratio
    return 0, ratio


def score_delta(day_row, baseline):
    """§2.3：事件数、总时长分别算，取分高的一个（"双值取大"）。返回 (score, detail dict)。"""
    if baseline is None:
        return None, {"reason": "基线正在建立中"}
    count_score, count_ratio = _delta_score_one(
        day_row.event_count, baseline["baseline_event_count"], "count", "count")
    dur_score, dur_ratio = _delta_score_one(
        day_row.total_duration_sec, baseline["baseline_duration_sec"], "duration", "duration")
    if count_score >= dur_score:
        return count_score, {"by": "count", "ratio": count_ratio}
    return dur_score, {"by": "duration", "ratio": dur_ratio}


def score_delta_bootstrap(day_row):
    """基线未建立期间的兜底：不看相对变化，直接用 Whistle 验证过的绝对分级。"""
    dur = day_row.total_duration_sec
    if dur >= BOOTSTRAP_SEVERE_SEC_PER_DAY:
        return BOOTSTRAP_SEVERE_SCORE, {"by": "bootstrap_absolute", "level": "severe"}
    if dur >= BOOTSTRAP_ELEVATED_SEC_PER_DAY:
        return BOOTSTRAP_ELEVATED_SCORE, {"by": "bootstrap_absolute", "level": "elevated"}
    return 0, {"by": "bootstrap_absolute", "level": "infrequent_or_occasional"}


# ── §2.1 聚集程度 ──────────────────────────────────────────────────────

def score_cluster(day_row):
    c = day_row.cluster_count
    if c >= 3:
        return 20
    if c >= 1:
        return 10
    return 0


# ── §2.1 持续程度：需要外部传入"过去7天每天的变化幅度分"序列 ──────────────

def score_persistence(delta_scores_last_7_days):
    """delta_scores_last_7_days：从今天往前，最多7天的变化幅度分（None=数据不足日，不重置不增加）。

    简化实现：数据不足日跳过（既不重置也不计入连续），一旦出现某天分数<PERSISTENCE_ENTRY_SCORE
    （回到基线范围）就重置连续计数为0；连续两个数据不足日则提前终止本次统计窗口。
    """
    consecutive = 0
    insufficient_streak = 0
    for score in delta_scores_last_7_days:  # 已按时间从远到近排好序
        if score is None:
            insufficient_streak += 1
            if insufficient_streak >= 2:
                consecutive = 0
            continue
        insufficient_streak = 0
        if score >= PERSISTENCE_ENTRY_SCORE:
            consecutive += 1
        else:
            consecutive = 0
    if consecutive >= 3:
        return 20
    if consecutive == 2:
        return 10
    if consecutive == 1:
        return 5
    return 0


# ── §2.4 正常行为影响（中断）──────────────────────────────────────────

def score_interruption(day_row):
    zn = day_row.event_count - day_row.sleep_disruption_count  # 粗略近似：非中断相关的日间抓挠次数
    zd = day_row.sleep_disruption_count
    long_scratch = day_row.max_event_duration_sec >= LONG_SCRATCH_RED_FLAG_SEC
    if zd >= 2 or long_scratch:
        return 30, True  # 红旗
    if zd == 1:
        return 20, False
    if zn > 5:
        return 10, False
    return 0, False


# ── 汇总：单日 SBS ─────────────────────────────────────────────────────

def score_day(day_row, baseline, delta_scores_last_7_days):
    bootstrap_mode = baseline is None
    if bootstrap_mode:
        # 基线还没建立（<21天历史）：变化幅度换成绝对阈值兜底，持续程度在这期间不计分
        # （"连续偏离基线"这个概念本身依赖基线，引导期内没有意义）。
        delta_score, delta_detail = score_delta_bootstrap(day_row)
        persistence_score = 0
    else:
        delta_score, delta_detail = score_delta(day_row, baseline)
        persistence_score = score_persistence(delta_scores_last_7_days)

    cluster_score = score_cluster(day_row)
    interrupt_score, interrupt_red_flag = score_interruption(day_row)

    red_flags = []
    if not bootstrap_mode and delta_detail.get("ratio", 0) > 3:
        red_flags.append("baseline>3x")
        delta_score = 30
    if bootstrap_mode and delta_detail.get("level") == "severe":
        red_flags.append("bootstrap_absolute_severe")
    if day_row.cluster_count >= 3:
        red_flags.append("cluster>=3")
        cluster_score = 20
    if persistence_score >= 20:
        red_flags.append("persistence>=3days")
    if interrupt_red_flag:
        red_flags.append("interrupt_or_long_scratch")

    total = delta_score + cluster_score + persistence_score + interrupt_score
    total = max(0, min(100, total))
    if total >= 50:
        tier = "C2"
    elif total >= 30:
        tier = "C1"
    else:
        tier = "C0"

    return {
        "date": day_row.date, "pet_id": day_row.pet_id,
        "delta_score": delta_score, "cluster_score": cluster_score,
        "persistence_score": persistence_score, "interrupt_score": interrupt_score,
        "total": total, "tier": tier, "red_flags": red_flags,
        "delta_detail": delta_detail, "insufficient_baseline": False,
        "bootstrap_mode": bootstrap_mode,
    }


def run_pipeline(events, wear_hours):
    """输入原始事件表+佩戴时长表，输出每只狗每天完整的 SBS 明细表（DataFrame）。"""
    daily = daily_features(events, wear_hours)
    results = []
    for pet_id in daily.pet_id.unique():
        pet_daily = daily[daily.pet_id == pet_id].sort_values("date").reset_index(drop=True)
        delta_history = []  # 按日期顺序保存每天的 delta_score（None=基线不足/数据不足）
        for _, row in pet_daily.iterrows():
            if row.data_quality_flag != "good":
                # 当天佩戴不足，不是"基线未建立"，两种"没有分数"的原因要分开标记，
                # 不能用0代替无数据（PM文档 §15）。
                results.append({
                    "date": row.date, "pet_id": row.pet_id, "total": None,
                    "tier": "insufficient_data", "insufficient_baseline": False,
                    "bootstrap_mode": False,
                })
                delta_history.append(None)
                continue
            baseline = compute_baseline(daily, pet_id, row.date)
            delta_score, _ = score_delta(row, baseline) if baseline else (None, {})
            last_7 = delta_history[-7:]
            res = score_day(row, baseline, last_7)
            results.append(res)
            delta_history.append(delta_score)
    return pd.DataFrame(results)
