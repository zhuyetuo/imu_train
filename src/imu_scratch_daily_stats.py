"""
统计每天每个机位(IMU)的抓挠情况，产出符合pm_skin_scoring「C值计算」标签
需要的统计量(次数/时长/聚集时段数/持续天数/中断次数等)，供那边的Web页面
直接读取、按日期+IMU选择自动填充，不用每次手动一个个数字去数。

"IMU几"按文件名里的 _imu{N} 取（复用extract_clips.py的extract_imu_label），
不是按 cam{N} —— 机位号跟狗身上的IMU编号没有对应关系：房间固定就2个机位，
但可以同时挂4条狗，同一个cam1下面会有 cam1_imu1 / cam1_imu3 / cam1_imu4
三条不同狗的CSV。按cam分组会把好几条狗的数据合并成一条，佩戴时长和抓挠
次数都会翻好几倍（实测8-14那天IMU1的佩戴时长算出40.7小时，一天才24小时）。

输入: infer_result_xxx/{day}/_infer/*_infer.json（infer_csv_scratch.py的产出）
输出: 每天一份CSV，写在 infer_result_xxx/{day}/imu_daily_scratch_stats.csv，
      每行是这天某个IMU的统计量。按天分开存是为了不互相覆盖——所有天共用
      root下一个文件的话，跑完8-14再跑8-15就把8-14那份冲掉了。

基线跟输出范围是两回事：--days 只限制"输出哪几天"，基线永远拿root下全部
已跑过推理的天来算。只读请求那一天的话永远算不出基线（第一次跑8-19就只有
8-19一天数据）。反过来说，以后补跑了更早的天，把之前的天重跑一次就能拿到
更新后的、更准的基线。

佩戴时长(valid_wear_hours)按每个session推理窗口的时间跨度近似，同一天同
一个IMU的多个session取区间并集（不是累加——录制时段会重叠，累加会算出
超过24小时）；不足MIN_GOOD_WEAR_HOURS小时的天标成partial/
insufficient(data_quality_flag列)——这种天的抓挠次数天然偏低，不是狗不痒
了，是没多少数据可看，所以这些天不参与基线中位数、当天的"持续天数"也不
计入(按数据不足处理，既不重置也不增加连续计数)。

用法:
  # 输出root下所有天，每天各自一份
  python src/imu_scratch_daily_stats.py --infer_root infer_result_majority_syn

  # 只输出某几天（基线仍然用全部天算）
  python src/imu_scratch_daily_stats.py \
    --infer_root infer_result_majority_syn --days 2026_8_18,2026_8_19
"""
import argparse
import csv
import glob
import os
import sys
from datetime import datetime, timedelta
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_clips import extract_imu_label  # noqa: E402

import json

NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 6
CLUSTER_EVENTS_PER_HOUR = 5        # 跟scratch_burden.py的CLUSTER_EVENTS_PER_HOUR一致
INTERRUPT_MERGE_GAP_MINUTES = 5    # 跟scratch_burden.py的INTERRUPT_MERGE_GAP_MINUTES一致
LONG_SCRATCH_RED_FLAG_SEC = 60     # 跟scratch_burden.py/pm_skin_scoring的常量一致
MIN_GOOD_WEAR_HOURS = 12           # 跟scratch_burden.py的daily_features()同一个阈值：
                                    # 有效佩戴<12小时的天标成"partial"/"insufficient"，
                                    # 不是"抓挠次数少"，是"根本没多少数据可看"，两者不能
                                    # 混为一谈——佩戴不到的天次数天然就少，不代表狗真的
                                    # 不痒了

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
    """返回 ({imu_label: [(start_dt, end_dt), ...]}, {imu_label: 佩戴秒数})，
    按csv文件名里的 _imu{N} 分组（不是camN，见模块开头的说明）。

    只保留conf_field(默认conf_mean，跟--ml_conf_field/ML_CONF_FIELD一个
    语义)>=min_conf的片段才算"抓挠"计入统计——不是模型报出来的每一段
    都算数，跟ML_PRELABEL那边"只标注高置信度片段"是同一个筛选标准，两边
    置信度阈值不一致的话，网页上看到的C值统计跟Label Studio里实际标注出
    来的片段对不上，容易confusing。"""
    infer_jsons = glob.glob(os.path.join(infer_dir, "**", "*_infer.json"), recursive=True)
    infer_jsons += glob.glob(os.path.join(infer_dir, "*_infer.json"))
    infer_jsons = sorted(set(infer_jsons))

    events_by_imu = {}
    wear_spans_by_imu = {}
    for path in infer_jsons:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        csv_basename = data.get("csv_basename", os.path.basename(path))
        # 按 _imu{N} 取，不是按 cam{N}——同一个机位下会挂多条狗的IMU
        # (cam1_imu1 / cam1_imu3 / cam1_imu4)，按cam分组会把它们合并成一条
        imu_label = extract_imu_label(os.path.splitext(csv_basename)[0])

        # 佩戴时长：这个session(csv文件)有推理覆盖的时间跨度=第一个窗口到
        # 最后一个窗口的时间差，近似当作这段时间设备实际戴着(每个窗口本身
        # 只有1-2秒，相对小时级的佩戴时长可以忽略不计)。这不是精确的"真的
        # 贴身佩戴"判断(比如摘下来放在旁边但设备还在采集也会被算进去)，
        # 只是"这段时间有数据"的近似，跟scratch_burden.py要求外部单独传入
        # 真实佩戴时长表是同一个概念，这里没有更精确的信号源，用推理窗口
        # 覆盖范围顶上。
        #
        # 各session的跨度先收集起来，最后取区间并集再算总时长，不是直接
        # 累加——同一天同一个IMU的录制时段是会重叠的(实际数据里有
        # ..._100000446_... 和 ..._100117864_... 这种只差1分钟就又起一段
        # 录制的情况)，直接累加会把重叠部分重复计算，算出超过24小时的
        # 佩戴时长
        windows = data.get("windows", [])
        window_ts = []
        for w in windows:
            if not w.get("ts"):
                continue
            try:
                window_ts.append(_parse_ts(w["ts"]))
            except ValueError:
                continue
        if window_ts:
            wear_spans_by_imu.setdefault(imu_label, []).append((min(window_ts), max(window_ts)))

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

    wear_seconds_by_imu = {imu: _union_seconds(spans)
                           for imu, spans in wear_spans_by_imu.items()}
    return events_by_imu, wear_seconds_by_imu


def _union_seconds(spans):
    """[(start, end), ...] → 区间并集的总秒数。重叠的录制时段只算一次，
    不重复计入——同一天同一个IMU经常有重叠的录制片段，直接把各段时长
    加起来会算出超过24小时的"佩戴时长"。"""
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
    （跟ML_CONF_FIELD/ML_MIN_CONF保持一致，不是conf_max）。

    days 只筛"输出哪几天"，不限制"读哪几天"——基线是拿这只狗别的日子
    的数据算中位数，只读请求的那一天的话永远算不出基线（第一次跑8-19
    就只有8-19一天数据，n_baseline_days恒等于0）。所以不管days传什么，
    infer_root下所有已经跑过推理的天都要读进来参与基线计算，最后只把
    请求的那几天的行返回出去。这也意味着：以后补跑了更早的天，之前那
    几天的基线会自动变准，重跑一次就能拿到更新后的数字。"""
    day_dirs = []
    for d in sorted(os.listdir(infer_root)):
        infer_dir = os.path.join(infer_root, d, "_infer")
        if os.path.isdir(infer_dir):
            day_dirs.append((d, infer_dir))

    if days:
        missing = [d for d in days if d not in {day for day, _ in day_dirs}]
        for d in missing:
            print(f"[跳过] {os.path.join(infer_root, d, '_infer')} 不存在")

    # 先把每个(imu, date)的原始特征都算出来，再统一算基线/持续天数——
    # 基线需要看同一个IMU在别的日子的数据，必须等所有天都读完才能算
    raw = {}  # imu -> {date_str: features}
    for day, infer_dir in day_dirs:
        if not os.path.isdir(infer_dir):
            print(f"[跳过] {infer_dir} 不存在")
            continue
        events_by_imu, wear_seconds_by_imu = _load_day_events(
            infer_dir, min_conf=min_conf, conf_field=conf_field)
        # 佩戴时长按当天出现过的全部IMU算(不只是有抓挠事件的那些)——一个
        # IMU这天佩戴时长有数据、但一次抓挠都没有检测到，也得出现在统计里，
        # 不然"这天佩戴够但没抓挠"跟"这天数据不足看不出来"这两种情况又会
        # 混在一起分不清
        for imu_label in set(events_by_imu) | set(wear_seconds_by_imu):
            events = events_by_imu.get(imu_label, [])
            feat = _day_features(events)
            wear_hours = round(wear_seconds_by_imu.get(imu_label, 0.0) / 3600, 2)
            feat["valid_wear_hours"] = wear_hours
            feat["data_quality_flag"] = (
                "good" if wear_hours >= MIN_GOOD_WEAR_HOURS
                else ("partial" if wear_hours > 0 else "insufficient")
            )
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
            # 基线只用佩戴时长达标(good)的天——佩戴不足的天次数天然偏低，
            # 拿进来算中位数会把基线压低，反过来让别的天看着"涨得特别多"，
            # 跟scratch_burden.py的compute_baseline只取data_quality_flag
            # =="good"的天是同一个道理
            other_dates = [d for d in dates_sorted
                          if d != date and by_date[d]["data_quality_flag"] == "good"]
            if other_dates:
                baseline_count = median(by_date[d]["event_count"] for d in other_dates)
                baseline_duration_min = median(by_date[d]["total_duration_min"] for d in other_dates)
                n_baseline_days = len(other_dates)
            else:
                baseline_count = 0
                baseline_duration_min = 0
                n_baseline_days = 0

            feat = by_date[date]
            # 当天佩戴不足时delta给None(数据不足)，不是给0分——0分的语义是
            # "看过了，确实没怎么涨"，None是"这天压根没足够数据下结论"，
            # 两者在下面算持续天数时的处理方式完全不同(None不重置也不计入)
            delta = (_delta_score(feat["event_count"], baseline_count,
                                  feat["total_duration_min"], baseline_duration_min)
                    if n_baseline_days and feat["data_quality_flag"] == "good" else None)
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
                "valid_wear_hours": feat["valid_wear_hours"],
                "data_quality_flag": feat["data_quality_flag"],
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

    # 到这里为止全部天都参与了基线计算，现在才按days筛输出
    if days:
        wanted = set(days)
        rows = [r for r in rows if r["date"] in wanted]

    rows.sort(key=lambda r: (r["date"], r["imu"]))
    return rows


COLUMNS = [
    "date", "imu", "valid_wear_hours", "data_quality_flag",
    "event_count", "total_duration_min", "max_event_duration_sec",
    "cluster_count", "night_event_count", "zn", "zd", "long_scratch",
    "baseline_count", "baseline_duration_min", "n_baseline_days", "persistence_days",
]

# 每天一份，落在 {infer_root}/{day}/ 下——不是所有天共用root下的一个文件，
# 那样跑完一天再跑另一天会直接覆盖掉前一天的结果
OUTPUT_BASENAME = "imu_daily_scratch_stats.csv"


def main():
    parser = argparse.ArgumentParser(description="统计每天每个IMU的抓挠情况，产出C值计算需要的统计量")
    parser.add_argument("--infer_root", required=True,
                        help="推理结果根目录，比如infer_result_majority_syn（下面是{day}/_infer/*_infer.json）")
    parser.add_argument("--days", default="",
                        help="只输出这几天，逗号分隔，比如2026_8_18,2026_8_19；默认输出infer_root下"
                             "所有天。注意：不管这里传什么，基线都是拿root下全部已跑过推理的天算的，"
                             "只是输出被限制在这几天")
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
        return

    # 每天一个文件、落在各自的日期目录下——之前所有天共用root下的一个
    # 文件，跑完8-14再跑8-15就会把8-14那份直接覆盖掉，只剩最后一次运行
    # 的结果。按天分开存就不会互相覆盖，各天的统计可以累积下来
    by_day = {}
    for row in rows:
        by_day.setdefault(row["date"], []).append(row)

    for day in sorted(by_day):
        day_rows = by_day[day]
        out_path = os.path.join(args.infer_root, day, OUTPUT_BASENAME)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for row in day_rows:
                writer.writerow(row)
        imus = sorted({r["imu"] for r in day_rows})
        n_base = {r["imu"]: r["n_baseline_days"] for r in day_rows}
        base_note = ("无基线" if all(v == 0 for v in n_base.values())
                     else f"基线取自{max(n_base.values())}天历史")
        print(f"→ {out_path}  {len(day_rows)}个机位({', '.join(imus)})，{base_note}")


if __name__ == "__main__":
    main()
