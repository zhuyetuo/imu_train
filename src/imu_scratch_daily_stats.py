"""
统计每天每个机位(IMU)的抓挠情况，产出符合pm_skin_scoring「C值计算」标签
需要的统计量(次数/时长/聚集时段数/持续天数/中断次数等)，供那边的Web页面
直接读取、按日期+IMU选择自动填充，不用每次手动一个个数字去数。

跟review_to_labelstudio.py用的是同一套"camN=IMU{N}"命名约定(--ml_full_video
模式那边，见其build_tasks_from_infer_ml)——不是extract_clips.py那套"cam是
固定机位、imu是狗自己的编号、两者不对应"的旧约定，这里直接复用
review_to_labelstudio.py里已经验证过的parse_cam()/session_key()，保证
"IMU几"这个标签在两份脚本之间的含义完全一致。

输入: infer_result_xxx/{day}/_infer/*_infer.json（infer_csv_scratch.py的产出）
输出: 一份CSV，每行是(日期, IMU)的一天统计量。

用法:
  python src/imu_scratch_daily_stats.py \
    --infer_root infer_result_majority_syn \
    --output infer_result_majority_syn/imu_daily_scratch_stats.csv

  # 只统计某几天（逗号分隔），默认扫描infer_root下所有天的目录
  python src/imu_scratch_daily_stats.py \
    --infer_root infer_result_majority_syn --days 2026_8_18,2026_8_19 \
    --output infer_result_majority_syn/imu_daily_scratch_stats.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime, timedelta
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_to_labelstudio import parse_cam, session_key  # noqa: E402

import json

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 6
CLUSTER_EVENTS_PER_HOUR = 5        # 跟scratch_burden.py的CLUSTER_EVENTS_PER_HOUR一致
INTERRUPT_MERGE_GAP_MINUTES = 5    # 跟scratch_burden.py的INTERRUPT_MERGE_GAP_MINUTES一致
LONG_SCRATCH_RED_FLAG_SEC = 60     # 跟scratch_burden.py/pm_skin_scoring的常量一致

# 基线基本比对用的简化档位——只用来估"这天算不算持续变化幅度大"（delta_score>=10），
# 不是最终C值计算本身，最终C值由pm_skin_scoring网页那边用PM的原始规则重新算，
# 这里只是给"持续天数"这个字段一个自动估算的参考值，用户在网页上仍然可以手动改
_DELTA_TIERS = [
    (30, 20, 15 * 60, 3.0),
    (20, 10, 10 * 60, 2.0),
    (10, 5, 5 * 60, 1.5),
    (5, 3, 3 * 60, 1.3),
]
BASELINE_DENOM_FLOOR = 3


def _parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")


def _load_day_events(infer_dir, min_conf=0.8, conf_field="conf_mean"):
    """返回 {imu_label: [(start_dt, end_dt), ...]}，一个session下每个机位
    自己的抓挠片段（跟review_to_labelstudio.py的build_tasks_from_infer_ml
    分组方式完全一致：按csv_basename里的camN分组）。

    只保留conf_field(默认conf_mean，跟--ml_conf_field/ML_CONF_FIELD一个
    语义)>=min_conf的片段才算"抓挠"计入统计——不是模型报出来的每一段
    都算数，跟ML_PRELABEL那边"只标注高置信度片段"是同一个筛选标准，两边
    置信度阈值不一致的话，网页上看到的C值统计跟Label Studio里实际标注出
    来的片段对不上，容易confusing。"""
    infer_jsons = glob.glob(os.path.join(infer_dir, "**", "*_infer.json"), recursive=True)
    infer_jsons += glob.glob(os.path.join(infer_dir, "*_infer.json"))
    infer_jsons = sorted(set(infer_jsons))

    events_by_imu = {}
    for path in infer_jsons:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        csv_basename = data.get("csv_basename", os.path.basename(path))
        cam = parse_cam(csv_basename) or "cam1"
        m = re.search(r"\d+", cam)
        cam_num = int(m.group()) if m else 1
        imu_label = f"IMU{cam_num}"

        for seg in data.get("scratch_segments", []):
            if not seg.get("start_ts") or not seg.get("end_ts"):
                continue
            if seg.get(conf_field, 0.0) < min_conf:
                continue
            try:
                start = _parse_ts(seg["start_ts"])
                end = _parse_ts(seg["end_ts"])
            except ValueError:
                continue
            events_by_imu.setdefault(imu_label, []).append((start, end))

    return events_by_imu


def _cluster_count(starts):
    """1小时滑窗内事件数>=CLUSTER_EVENTS_PER_HOUR记一个聚集时段，跟
    scratch_burden.py的_cluster_segments算法完全一致。"""
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


def _sleep_disruption_count(night_events):
    """相邻夜间抓挠事件间隔<INTERRUPT_MERGE_GAP_MINUTES视为同一次中断没重新
    入睡，合并计数——跟scratch_burden.py的_sleep_interruptions算法一致。"""
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


def _day_features(events):
    """events: [(start_dt, end_dt), ...]，一个(imu, date)当天的全部抓挠事件。"""
    starts = [s for s, _ in events]
    durations = [(e - s).total_seconds() for s, e in events]
    is_night = lambda dt: dt.hour >= NIGHT_START_HOUR or dt.hour < NIGHT_END_HOUR
    night_events = [(s, e) for s, e in events if is_night(s)]

    event_count = len(events)
    total_duration_sec = sum(durations)
    max_event_duration_sec = max(durations) if durations else 0.0
    cluster_count = _cluster_count(starts) if starts else 0
    night_event_count = len(night_events)
    zd = _sleep_disruption_count(night_events)
    zn = max(0, event_count - zd)

    return {
        "event_count": event_count,
        "total_duration_sec": total_duration_sec,
        "total_duration_min": round(total_duration_sec / 60, 2),
        "max_event_duration_sec": round(max_event_duration_sec, 1),
        "cluster_count": cluster_count,
        "night_event_count": night_event_count,
        "zn": zn,
        "zd": zd,
        "long_scratch": max_event_duration_sec >= LONG_SCRATCH_RED_FLAG_SEC,
    }


def _delta_score(current_count, baseline_count, current_dur_min, baseline_dur_min):
    """跟pm_skin_scoring的_c_score_delta同一套档位，粗略估一个每日"变化幅度"分，
    只用来推算"持续天数"这个辅助字段，不是正式C值计算（正式计算在网页上用
    用户在网页上核对/调整过的基线值重新算一遍，这里只是给个自动参考起点）。"""
    def one(current, baseline, is_duration):
        denom = max(baseline, BASELINE_DENOM_FLOOR if not is_duration else BASELINE_DENOM_FLOOR)
        ratio = current / denom
        abs_increase = current - baseline
        for score, abs_min_count, abs_min_dur_sec, ratio_min in _DELTA_TIERS:
            abs_min = (abs_min_dur_sec / 60) if is_duration else abs_min_count
            if abs_increase >= abs_min and ratio >= ratio_min:
                return score
        return 0
    return max(one(current_count, baseline_count, False), one(current_dur_min, baseline_dur_min, True))


def compute_stats(infer_root, days=None, min_conf=0.8, conf_field="conf_mean"):
    """返回按(date, imu_label)排序的统计行列表。min_conf/conf_field跟
    ML_PRELABEL那边筛"算不算抓挠"用的是同一个标准，默认conf_mean>=0.8
    （跟ML_CONF_FIELD/ML_MIN_CONF保持一致，不是conf_max）。"""
    if days:
        day_dirs = [(d, os.path.join(infer_root, d, "_infer")) for d in days]
    else:
        day_dirs = []
        for d in sorted(os.listdir(infer_root)):
            infer_dir = os.path.join(infer_root, d, "_infer")
            if os.path.isdir(infer_dir):
                day_dirs.append((d, infer_dir))

    # 先把每个(imu, date)的原始特征都算出来，再统一算基线/持续天数——
    # 基线需要看同一个IMU在别的日子的数据，必须等所有天都读完才能算
    raw = {}  # imu -> {date_str: features}
    for day, infer_dir in day_dirs:
        if not os.path.isdir(infer_dir):
            print(f"[跳过] {infer_dir} 不存在")
            continue
        events_by_imu = _load_day_events(infer_dir, min_conf=min_conf, conf_field=conf_field)
        for imu_label, events in events_by_imu.items():
            feat = _day_features(events)
            raw.setdefault(imu_label, {})[day] = feat

    rows = []
    for imu_label in sorted(raw):
        by_date = raw[imu_label]
        dates_sorted = sorted(by_date)  # 字符串"2026_8_19"这种格式天然可排序

        # 基线：用该IMU除当天以外的其它天的中位数（不是PM文档要求的严格
        # "21天历史、跳过最近7天"，那需要长期积累的数据，数据不够21天时
        # 这里退化成"有多少天算多少天"的简化基线，仅供网页上参考/核对，
        # 用户在C值计算标签里仍然可以手动覆盖这两个数字
        delta_history = []  # 按日期顺序，跟这个IMU的dates_sorted对齐
        for date in dates_sorted:
            other_dates = [d for d in dates_sorted if d != date]
            if other_dates:
                baseline_count = median(by_date[d]["event_count"] for d in other_dates)
                baseline_duration_min = median(by_date[d]["total_duration_min"] for d in other_dates)
                n_baseline_days = len(other_dates)
            else:
                baseline_count = 0
                baseline_duration_min = 0
                n_baseline_days = 0

            feat = by_date[date]
            delta = (_delta_score(feat["event_count"], baseline_count,
                                  feat["total_duration_min"], baseline_duration_min)
                    if n_baseline_days else None)
            delta_history.append(delta)

            # 持续天数：往前数连续几天(含今天)delta_score>=10，None(基线不足)
            # 不重置也不计入，跟scratch_burden.py的score_persistence同一个思路
            consecutive = 0
            insufficient_streak = 0
            for d in reversed(delta_history):
                if d is None:
                    insufficient_streak += 1
                    if insufficient_streak >= 2:
                        break
                    continue
                insufficient_streak = 0
                if d >= 10:
                    consecutive += 1
                else:
                    break
            persistence_days = min(consecutive, 7)

            rows.append({
                "date": date, "imu": imu_label,
                "event_count": feat["event_count"],
                "total_duration_min": feat["total_duration_min"],
                "max_event_duration_sec": feat["max_event_duration_sec"],
                "cluster_count": feat["cluster_count"],
                "night_event_count": feat["night_event_count"],
                "zn": feat["zn"], "zd": feat["zd"],
                "long_scratch": feat["long_scratch"],
                "baseline_count": round(baseline_count, 1),
                "baseline_duration_min": round(baseline_duration_min, 2),
                "n_baseline_days": n_baseline_days,
                "persistence_days": persistence_days,
            })

    rows.sort(key=lambda r: (r["date"], r["imu"]))
    return rows


COLUMNS = [
    "date", "imu", "event_count", "total_duration_min", "max_event_duration_sec",
    "cluster_count", "night_event_count", "zn", "zd", "long_scratch",
    "baseline_count", "baseline_duration_min", "n_baseline_days", "persistence_days",
]


def main():
    parser = argparse.ArgumentParser(description="统计每天每个IMU的抓挠情况，产出C值计算需要的统计量")
    parser.add_argument("--infer_root", required=True,
                        help="推理结果根目录，比如infer_result_majority_syn（下面是{day}/_infer/*_infer.json）")
    parser.add_argument("--days", default="",
                        help="只统计这几天，逗号分隔，比如2026_8_18,2026_8_19；默认扫描infer_root下所有天")
    parser.add_argument("--output", required=True, help="输出CSV路径")
    parser.add_argument("--min_conf", type=float, default=0.8,
                        help="只有conf_field>=这个值的抓挠片段才算数（默认0.8，"
                             "跟ML_PRELABEL那边的ML_MIN_CONF保持一致）")
    parser.add_argument("--conf_field", default="conf_mean", choices=["conf_max", "conf_mean"],
                        help="用哪个置信度字段判断（默认conf_mean，不是conf_max——"
                             "跟ML_CONF_FIELD保持一致）")
    args = parser.parse_args()

    days = [d.strip() for d in args.days.split(",") if d.strip()] or None
    rows = compute_stats(args.infer_root, days, min_conf=args.min_conf, conf_field=args.conf_field)

    if not rows:
        print(f"[警告] 没有统计出任何数据，检查--infer_root/--days是否正确")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    days_found = sorted({r["date"] for r in rows})
    imus_found = sorted({r["imu"] for r in rows})
    print(f"→ {args.output}  共{len(rows)}行，覆盖{len(days_found)}天({', '.join(days_found)})，"
          f"{len(imus_found)}个机位({', '.join(imus_found)})")


if __name__ == "__main__":
    main()
