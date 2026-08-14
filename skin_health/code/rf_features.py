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

from scratch_burden import (BASELINE_DENOM_FLOOR, BASELINE_LOOKBACK_END,
                            BASELINE_LOOKBACK_START, MIN_BASELINE_DAYS, daily_features)

ROLLING_WINDOWS = (3, 7, 14, 30)


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

        # 个人基线特征——跟scratch_burden.compute_baseline()用完全一样的窗口
        # (d-21~d-8，跳过最近7天)和分母保护(BASELINE_DENOM_FLOOR)，因为SBS的
        # 标签本身就是从这个基线偏离算出来的，这几个特征此前在rf_feature_spec.md
        # 里标了"待实现"但实际代码一直没加，是模型A/B宏F1上不去的一个主要原因：
        # 之前只给了多窗口"滚动均值"这种平滑趋势特征，没给"跟SBS同一套定义的
        # 基线偏离量"，模型只能自己近似重建SBS的判断依据，现在直接把这个信息
        # 显式喂进去。
        n = len(grp)
        baseline_event = np.full(n, np.nan)
        baseline_dur = np.full(n, np.nan)
        for i in range(n):
            lo, hi = i - BASELINE_LOOKBACK_END, i - BASELINE_LOOKBACK_START
            if lo < 0:
                continue
            window = grp.iloc[lo:hi + 1]
            window = window[window["data_quality_flag"] == "good"]
            if len(window) < MIN_BASELINE_DAYS:
                continue
            baseline_event[i] = window["event_count"].median()
            baseline_dur[i] = window["total_duration_sec"].median()
        grp["baseline_event_count"] = baseline_event
        grp["baseline_duration_sec"] = baseline_dur
        denom_event = np.maximum(baseline_event, BASELINE_DENOM_FLOOR)
        denom_dur = np.maximum(baseline_dur, BASELINE_DENOM_FLOOR)
        grp["baseline_ratio_count"] = grp["event_count"] / denom_event
        grp["baseline_ratio_duration"] = grp["total_duration_sec"] / denom_dur
        grp["baseline_delta_count"] = grp["event_count"] - baseline_event
        grp["baseline_delta_duration"] = grp["total_duration_sec"] - baseline_dur
        grp["has_baseline"] = (~np.isnan(baseline_event)).astype(int)

        # z_score_vs_self：用30天滚动均值/标准差近似个人历史分布（比14天基线窗口
        # 覆盖面更广，标准差用来标准化偏离程度）
        eps = 1e-6
        grp["z_score_vs_self"] = (
            (grp["event_rate_per_wear_hour"] - grp["rolling_mean_30d"])
            / (grp["rolling_std_30d"] + eps)
        )
        out_frames.append(grp)
    df = pd.concat(out_frames, ignore_index=True)

    # 基线/z-score类特征在历史不足(<21天)时天然是NaN，不插补——本项目选用
    # HistGradientBoostingClassifier，原生支持NaN，让模型自己学习"缺失"这个
    # 状态该怎么处理，比人工插补(均值/哨兵值)更准，见rf_feature_engineering_
    # plan.md第4节的方案选型。has_baseline这个0/1伴随特征则是显式告诉模型
    # "基线是不是真的建立了"，跟NaN共同存在不冲突。

    return df


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
    "baseline_ratio_count", "baseline_ratio_duration",
    "baseline_delta_count", "baseline_delta_duration",
    "has_baseline", "z_score_vs_self",
] + [f"rolling_mean_{w}d" for w in ROLLING_WINDOWS] + [f"rolling_std_{w}d" for w in ROLLING_WINDOWS]
