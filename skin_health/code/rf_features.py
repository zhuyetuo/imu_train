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
        out_frames.append(grp)
    df = pd.concat(out_frames, ignore_index=True)

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
] + [f"rolling_mean_{w}d" for w in ROLLING_WINDOWS] + [f"rolling_std_{w}d" for w in ROLLING_WINDOWS]
