"""
按 rf_feature_spec.md 定义的特征集，从事件明细+佩戴时长表算出RF训练用的
每日特征宽表。跟 scratch_burden.py 的 daily_features() 是互补关系：那边
是SBS引擎用的聚合（次数/时长/聚集/夜间占比这几个足够SBS用的），这里额外
补上RF需要但SBS不需要的部分（时长统计量、间隔统计量、比率特征、多窗口
滚动特征、history_days_available、品种类别特征）。

复用 daily_features() 算基础聚合，不重复造轮子。
"""
import numpy as np
import pandas as pd

from scratch_burden import daily_features

ROLLING_WINDOWS = (3, 7, 14, 30)

# 抓挠基线候选窗口——不直接照抄scratch_burden.py里SBS用的那一组具体参数
# (21天窗口/跳过最近7天/分母保护=1)，那是PM当初拍脑袋定的一组特定超参数，
# 不是业务上唯一正确的定义。这里给出好几组不同"跳过最近几天/看多长历史"
# 的候选基线定义，都当独立特征喂给模型，让训练自己挑出哪个时间尺度真的
# 有区分力——skip=0(不跳过，包含最近几天)代表"响应快的短期基线"，
# skip=7(参考SBS的跳过思路，但不是唯一选择)代表"排除掉可能已经开始异常
# 的最近一段时间，看更早、更可能代表真实正常水平的历史"这个业务直觉，
# 两种直觉都值得让模型自己验证谁更有用，不是predetermine哪个对。
# 2026-08更新：产品希望基线能"多快建立多快用"，同时长期看又要够稳，不是
# 只顾一头——之前只有7/14/30三档"短期"，现在把"recent"(不跳过，响应快)这组
# 扩到短/中/长完整覆盖3/7/14/21/30/60/180天，同时保留"excl_recent"(跳过
# 最近7天，排除近期可能已异常的时段，看更早、更可能代表真实正常水平的
# 历史)这个独立业务直觉在14/30两档上——两种直觉(快vs稳、含近期vs排除近期)
# 都留着让训练自己验证谁在什么阶段更有用，不是predetermine哪个对，也不是
# 要拿新的窗口去替换旧的。
BASELINE_CONFIGS = [
    {"name": "recent3", "skip": 0, "window": 3, "min_valid": 2},
    {"name": "recent7", "skip": 0, "window": 7, "min_valid": 4},
    {"name": "recent14", "skip": 0, "window": 14, "min_valid": 5},
    {"name": "recent21", "skip": 0, "window": 21, "min_valid": 7},
    {"name": "recent30", "skip": 0, "window": 30, "min_valid": 10},
    {"name": "recent60", "skip": 0, "window": 60, "min_valid": 15},
    {"name": "recent180", "skip": 0, "window": 180, "min_valid": 30},
    {"name": "excl_recent14", "skip": 7, "window": 14, "min_valid": 5},
    {"name": "excl_recent30", "skip": 7, "window": 30, "min_valid": 10},
]
# min_valid按窗口长度递增(不是统一硬阈值)：窗口越长，要求窗口内"good"天数
# 占比越高才建立基线，避免长窗口只有寥寥几天有效数据就被短窗口那套宽松
# 标准放行——比如recent180如果只要5天good数据就成立，180天里绝大部分是
# 缺失/插值的情况下算出的中位数没有代表性。recent3用min_valid=2，参考
# ROLLING_WINDOWS滚动特征min_periods=max(2, w//2)的同一套"至少2天才算数"
# 的宽松下限逻辑——这一档本来就是牺牲稳定性换响应速度的，不强求高占比。
BASELINE_MIN_VALID_DAYS = 5  # 兜底默认值，配置里没显式给min_valid的窗口用这个
# ——比SBS的MIN_BASELINE_DAYS=7独立设定，不是照抄，数值上凑巧接近是因为
# 业务直觉相通(至少要有接近一周的数据才谈得上"典型水平")，不是刻意对齐
# SBS的选择。
BASELINE_DENOM_MIN = 1.0  # 分母下限，纯粹是防止除0/被极小基线值放大，不是业务参数


def _duration_stats(events, pet_id, date):
    day_events = events[(events["pet_id"] == pet_id) & (events["date"] == date)]
    if len(day_events) == 0:
        return {"duration_mean": 0.0, "duration_median": 0.0, "duration_std": 0.0}
    d = day_events["duration_sec"]
    return {
        "duration_mean": float(d.mean()),
        "duration_median": float(d.median()),
        "duration_std": float(d.std()) if len(d) > 1 else 0.0,
    }


def _interval_stats(events, pet_id, date):
    day_events = events[(events["pet_id"] == pet_id) & (events["date"] == date)].sort_values("start")
    if len(day_events) < 2:
        return {"interval_mean": np.nan, "interval_std": np.nan}
    starts = day_events["start"].tolist()
    gaps = [(b - a).total_seconds() for a, b in zip(starts[:-1], starts[1:])]
    return {
        "interval_mean": float(np.mean(gaps)),
        "interval_std": float(np.std(gaps)) if len(gaps) > 1 else 0.0,
    }


def compute_rf_features(events, wear_hours, breed_map, target_wear_hours=24.0):
    """
    events: DataFrame[pet_id, start, end, duration_sec]（跟 scratch_burden 的事件表一致）
    wear_hours: DataFrame[pet_id, date, valid_wear_hours]
    breed_map: dict pet_id -> breed_or_size_class 字符串
    返回按 (pet_id, date) 一行的特征宽表，跟 scratch_burden.daily_features() 的
    行范围完全一致（同样的日历补全逻辑），只是列更多。
    """
    events = events.copy()
    events["date"] = events["start"].dt.date

    base = daily_features(events, wear_hours)  # pet_id/date/event_count/total_duration_sec/
    # max_event_duration_sec/valid_wear_hours/cluster_count/sleep_disruption_count/
    # night_ratio/data_quality_flag

    extra_rows = []
    for _, row in base.iterrows():
        extra_rows.append({
            "pet_id": row["pet_id"], "date": row["date"],
            **_duration_stats(events, row["pet_id"], row["date"]),
            **_interval_stats(events, row["pet_id"], row["date"]),
        })
    extra = pd.DataFrame(extra_rows)
    df = base.merge(extra, on=["pet_id", "date"], how="left")

    df["total_duration_min"] = df["total_duration_sec"] / 60.0
    df["wear_completeness_ratio"] = (df["valid_wear_hours"] / target_wear_hours).clip(upper=1.0)
    # 分母用有效佩戴小时数，佩戴时长为0的天（off_body/insufficient）比率特征没有意义，置0
    # 而不是除0报错——这些天data_quality_flag已经是insufficient，训练时会被过滤掉，
    # 这里置0只是避免NaN/inf污染下面的滚动窗口计算
    safe_wear = df["valid_wear_hours"].replace(0, np.nan)
    df["event_rate_per_wear_hour"] = (df["event_count"] / safe_wear).fillna(0.0)
    df["duration_rate_per_wear_hour"] = (df["total_duration_min"] / safe_wear).fillna(0.0)

    df["breed_or_size_class"] = df["pet_id"].map(breed_map)

    df = df.sort_values(["pet_id", "date"]).reset_index(drop=True)

    # 多窗口滚动均值/标准差（在event_rate_per_wear_hour这个比率量上算，不在原始次数上算，
    # 避免佩戴时长波动混进"抓挠是不是变多了"这个判断）+ history_days_available，
    # 按狗分组各自独立计算，min_periods=1保证有多少天算多少天，不足窗口长度时不强制NaN
    # （但下面单独存一份"数据够不够"的天数信息，模型自己学会怎么权衡）
    out_frames = []
    for pet_id, grp in df.groupby("pet_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["history_days_available"] = np.arange(1, len(grp) + 1)
        for w in ROLLING_WINDOWS:
            grp[f"rolling_mean_{w}d"] = grp["event_rate_per_wear_hour"].rolling(
                window=w, min_periods=max(2, w // 2)).mean()
            grp[f"rolling_std_{w}d"] = grp["event_rate_per_wear_hour"].rolling(
                window=w, min_periods=max(2, w // 2)).std()

        # 抓挠基线特征——多组候选窗口(见BASELINE_CONFIGS)分别算一遍，不是
        # 只信SBS那一组参数。每组都用"用当天事件数/时长 对比 窗口内(排除
        # 当天)历史中位数"这个共同逻辑，窗口定义(跳过几天/看多长)不同。
        n = len(grp)
        event_arr = grp["event_count"].values
        dur_arr = grp["total_duration_sec"].values
        good_mask = (grp["data_quality_flag"] == "good").values
        any_baseline_valid = np.zeros(n, dtype=bool)
        for cfg in BASELINE_CONFIGS:
            name, skip, window = cfg["name"], cfg["skip"], cfg["window"]
            min_valid = cfg.get("min_valid", BASELINE_MIN_VALID_DAYS)
            base_event = np.full(n, np.nan)
            base_dur = np.full(n, np.nan)
            for i in range(n):
                hi = i - skip  # 不含当天(i)本身，也不含skip天之内的最近历史
                lo = hi - window
                if lo < 0:
                    continue
                idx = np.arange(max(0, lo), hi)
                idx = idx[good_mask[idx]]
                if len(idx) < min_valid:
                    continue
                base_event[i] = np.median(event_arr[idx])
                base_dur[i] = np.median(dur_arr[idx])
            denom_event = np.maximum(base_event, BASELINE_DENOM_MIN)
            denom_dur = np.maximum(base_dur, BASELINE_DENOM_MIN)
            grp[f"baseline_ratio_count_{name}"] = event_arr / denom_event
            grp[f"baseline_ratio_duration_{name}"] = dur_arr / denom_dur
            any_baseline_valid |= ~np.isnan(base_event)
        grp["has_any_baseline"] = any_baseline_valid.astype(int)

        # consecutive_days_above_baseline：对齐SBS的persistence_score维度——
        # 那边显式track"最近连续几天变化幅度分超过阈值"，是一个时序状态量，
        # 不是rolling_mean/std这类窗口统计量能精确重现的信息（详见
        # rf_synthetic_validation_findings.md的覆盖度分析：4个SBS维度里，
        # 变化幅度/聚集度/睡眠中断都有更细粒度的连续特征对应，只有持续性
        # 这个"streak"信息缺失）。用excl_recent14基线（实测最有区分力的一组，
        # 见rf_synthetic_validation_findings.md）的count/duration比率中较高
        # 者判断"当天是否偏高"，阈值2.0对应SBS _DELTA_TIERS里score=20那档的
        # ratio_min（score=10更宽松，这里选稍严格的2.0避免把太多天数计入
        # 连续，具体阈值不是照抄SBS的PERSISTENCE_ENTRY_SCORE=10那一档，只是
        # 业务直觉上的"明显偏高"参考点）。基线缺失(NaN)的天不重置也不计入
        # ——跟SBS score_persistence()里"数据不足日跳过，既不重置也不计入
        # 连续"的处理方式一致。
        ratio_c = grp["baseline_ratio_count_excl_recent14"].values
        ratio_d = grp["baseline_ratio_duration_excl_recent14"].values
        elevated = np.maximum(ratio_c, ratio_d) >= 2.0
        has_baseline = ~np.isnan(ratio_c)
        streak = np.zeros(n, dtype=int)
        cur = 0
        for i in range(n):
            if not has_baseline[i]:
                streak[i] = cur  # 数据不足：不重置也不累加，保持上一次的streak值
                continue
            if elevated[i]:
                cur += 1
            else:
                cur = 0
            streak[i] = cur
        grp["consecutive_days_above_baseline"] = streak

        # z_score_vs_self：用30天滚动均值/标准差近似这只狗自己的历史抓挠分布
        # （标准差用来标准化偏离程度，不依赖上面任何一组固定窗口的基线定义）
        eps = 1e-6
        grp["z_score_vs_self"] = (
            (grp["event_rate_per_wear_hour"] - grp["rolling_mean_30d"])
            / (grp["rolling_std_30d"] + eps)
        )
        out_frames.append(grp)
    df = pd.concat(out_frames, ignore_index=True)

    # 基线/z-score类特征在历史不足时天然是NaN，不插补——本项目选用
    # HistGradientBoostingClassifier，原生支持NaN，让模型自己学习"缺失"这个
    # 状态该怎么处理，比人工插补(均值/哨兵值)更准，见rf_feature_engineering_
    # plan.md第4节的方案选型。has_any_baseline这个0/1伴随特征显式告诉模型
    # "有没有任何一组候选基线窗口建立起来了"，跟NaN共同存在不冲突。

    return df


BASELINE_FEATURE_COLUMNS = [
    f"baseline_ratio_count_{cfg['name']}" for cfg in BASELINE_CONFIGS
] + [
    f"baseline_ratio_duration_{cfg['name']}" for cfg in BASELINE_CONFIGS
]

FEATURE_COLUMNS = [
    "event_count", "event_rate_per_wear_hour",
    "total_duration_min", "duration_rate_per_wear_hour",
    "duration_mean", "duration_median", "max_event_duration_sec", "duration_std",
    "night_ratio",
    "interval_mean", "interval_std",
    "sleep_disruption_count",
    "wear_completeness_ratio",
    "history_days_available",
    "breed_or_size_class",
    "has_any_baseline", "z_score_vs_self", "consecutive_days_above_baseline",
] + BASELINE_FEATURE_COLUMNS \
  + [f"rolling_mean_{w}d" for w in ROLLING_WINDOWS] + [f"rolling_std_{w}d" for w in ROLLING_WINDOWS]
