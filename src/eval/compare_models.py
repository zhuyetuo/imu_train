"""
对比多个模型（比如 majority/center/带合成/不带合成）在同一批数据上的推理结果。

两类对比：
  1. 跨模型：给定几对模型，同一条record上，两边都检测到的事件配对比较
     （起止时间戳差多少、时长差多少、conf_max/conf_mean差多少），以及
     只有一边检测到的事件（单边漏检/多检）。
  2. 单模型内部：同一批事件按conf_max分桶 vs 按conf_mean分桶，找出两种
     分桶方式结果不一样的事件（比如conf_max分到0.9-1.0，conf_mean却
     分到0.7-0.8）。

数据来源：每个模型 run_review_bins_all_days.sh 产出的
  <RESULT_ROOT>/<day>/_infer/*.json 里的 scratch_segments 字段——
  这是模型实际检测到的事件（未经clips目录的conf_max/conf_mean分桶影响），
  最准确的比较基准。

用法:
  python src/eval/compare_models.py \\
    --day 2026_7_17 \\
    --model majority=infer_result_majority \\
    --model majority_syn=infer_result_majority_syn \\
    --model center=infer_result_center \\
    --model center_syn=infer_result_center_syn \\
    --pairs majority:center majority:majority_syn center:center_syn majority_syn:center_syn \\
    --log_file logs/compare_2026_7_17.log
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def get_bin(conf: float) -> str:
    for i in range(len(BINS) - 1):
        if BINS[i] <= conf < BINS[i + 1]:
            lo = BINS[i]
            hi = BINS[i + 1] if BINS[i + 1] <= 1.0 else 1.0
            return f"{lo:.1f}-{hi:.1f}"
    return "0.9-1.0"


def parse_ts(s):
    if not s:
        return None
    return datetime.strptime(s[:23], "%Y-%m-%d %H:%M:%S.%f")


def load_model_events(result_root, day):
    """扫描 <result_root>/<day>/_infer/*.json，返回
    (events_by_record, total_records)：
    events_by_record = {record(csv_basename): [事件dict, ...]}（只包含至少有
    一个检测事件的record），事件dict含start/end(datetime)、start_ts/end_ts
    (原始字符串)、conf_max、conf_mean、duration(秒)。
    total_records = 这一天这个模型总共跑了推理的record数（不管有没有检测到
    事件），用来算"总共多少条record、其中多少条有抓挠"这个比例，而不是
    只看"有抓挠的record数"（那个数字容易让人误以为是总量）。"""
    infer_dir = os.path.join(result_root, day, "_infer")
    events_by_record = {}
    infer_jsons = sorted(glob.glob(os.path.join(infer_dir, "*_infer.json")))
    for path in infer_jsons:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        record = data.get("csv_basename", os.path.basename(path))
        segs = data.get("scratch_segments", [])
        evs = []
        for seg in segs:
            t0 = parse_ts(seg.get("start_ts"))
            t1 = parse_ts(seg.get("end_ts"))
            if t0 is None or t1 is None:
                continue
            evs.append({
                "start": t0, "end": t1,
                "start_ts": seg.get("start_ts"), "end_ts": seg.get("end_ts"),
                "conf_max": seg.get("conf_max", 0.0),
                "conf_mean": seg.get("conf_mean", 0.0),
                "duration": (t1 - t0).total_seconds(),
            })
        if evs:
            events_by_record[record] = evs
    return events_by_record, len(infer_jsons)


def overlaps(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def match_events(events_a, events_b):
    """两条record内部事件按时间重叠配对（贪心：重叠了就算配对，一个事件
    只配一次）。返回 (matched_pairs, only_a, only_b)。"""
    matched = []
    used_b = set()
    only_a = []
    for ea in events_a:
        found = None
        for j, eb in enumerate(events_b):
            if j in used_b:
                continue
            if overlaps(ea, eb):
                found = j
                break
        if found is not None:
            used_b.add(found)
            matched.append((ea, events_b[found]))
        else:
            only_a.append(ea)
    only_b = [eb for j, eb in enumerate(events_b) if j not in used_b]
    return matched, only_a, only_b


def fmt_ts(t):
    # 精确到百分之一秒——之前只显示到整秒，会出现"两边时间戳看起来只差1秒，
    # 但时长差却是2.25秒"这种视觉上对不上的情况（时长差是按完整精度算的，
    # 显示被截断了），加上毫秒位方便直接核对
    return t.strftime("%H:%M:%S.%f")[:-4] if t else "?"


def compare_pair(label_a, events_a_by_record, label_b, events_b_by_record):
    print(f"\n{'='*90}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'='*90}")

    all_records = sorted(set(events_a_by_record) | set(events_b_by_record))
    total_matched = 0
    total_only_a = 0
    total_only_b = 0
    start_diffs, end_diffs, dur_diffs, max_diffs, mean_diffs = [], [], [], [], []

    for record in all_records:
        evs_a = events_a_by_record.get(record, [])
        evs_b = events_b_by_record.get(record, [])
        if not evs_a and not evs_b:
            continue
        matched, only_a, only_b = match_events(evs_a, evs_b)
        total_matched += len(matched)
        total_only_a += len(only_a)
        total_only_b += len(only_b)

        if not matched and not only_a and not only_b:
            continue
        print(f"\n  record: {record}")
        for ea, eb in matched:
            start_diff = abs((ea["start"] - eb["start"]).total_seconds())
            end_diff = abs((ea["end"] - eb["end"]).total_seconds())
            dur_diff = abs(ea["duration"] - eb["duration"])
            max_diff = abs(ea["conf_max"] - eb["conf_max"])
            mean_diff = abs(ea["conf_mean"] - eb["conf_mean"])
            start_diffs.append(start_diff)
            end_diffs.append(end_diff)
            dur_diffs.append(dur_diff)
            max_diffs.append(max_diff)
            mean_diffs.append(mean_diff)
            print(f"    [两边都测到] {label_a}:{fmt_ts(ea['start'])}-{fmt_ts(ea['end'])}"
                  f"(max={ea['conf_max']:.2f},mean={ea['conf_mean']:.2f})  "
                  f"{label_b}:{fmt_ts(eb['start'])}-{fmt_ts(eb['end'])}"
                  f"(max={eb['conf_max']:.2f},mean={eb['conf_mean']:.2f})  "
                  f"起点差{start_diff:.2f}s 终点差{end_diff:.2f}s 时长差{dur_diff:.2f}s "
                  f"max差{max_diff:.2f} mean差{mean_diff:.2f}")
        for ea in only_a:
            print(f"    [仅{label_a}测到] {fmt_ts(ea['start'])}-{fmt_ts(ea['end'])}"
                  f"  max={ea['conf_max']:.2f} mean={ea['conf_mean']:.2f}"
                  f"  ← {label_b} 没检测到")
        for eb in only_b:
            print(f"    [仅{label_b}测到] {fmt_ts(eb['start'])}-{fmt_ts(eb['end'])}"
                  f"  max={eb['conf_max']:.2f} mean={eb['conf_mean']:.2f}"
                  f"  ← {label_a} 没检测到")

    print(f"\n  ── 汇总 ──")
    print(f"  两边都测到: {total_matched}")
    print(f"  仅{label_a}测到: {total_only_a}")
    print(f"  仅{label_b}测到: {total_only_b}")
    if start_diffs:
        print(f"  配对事件的起点时间差: 平均{sum(start_diffs)/len(start_diffs):.2f}s"
              f"  最大{max(start_diffs):.2f}s")
        print(f"  配对事件的终点时间差: 平均{sum(end_diffs)/len(end_diffs):.2f}s"
              f"  最大{max(end_diffs):.2f}s")
        print(f"  配对事件的时长差:     平均{sum(dur_diffs)/len(dur_diffs):.2f}s"
              f"  最大{max(dur_diffs):.2f}s")
        print(f"  配对事件的conf_max差: 平均{sum(max_diffs)/len(max_diffs):.3f}"
              f"  最大{max(max_diffs):.3f}")
        print(f"  配对事件的conf_mean差:平均{sum(mean_diffs)/len(mean_diffs):.3f}"
              f"  最大{max(mean_diffs):.3f}")


def report_bin_divergence(label, events_by_record):
    """单模型内部：同一事件按conf_max/conf_mean分桶，找出两种分桶不一样的事件。"""
    print(f"\n{'='*90}")
    print(f"  {label}：conf_max分桶 vs conf_mean分桶 不一致的事件")
    print(f"{'='*90}")
    n_total = 0
    n_diff = 0
    rows = []
    for record, evs in sorted(events_by_record.items()):
        for e in evs:
            n_total += 1
            bin_max = get_bin(e["conf_max"])
            bin_mean = get_bin(e["conf_mean"])
            if bin_max != bin_mean:
                n_diff += 1
                rows.append((record, e, bin_max, bin_mean))

    print(f"  共 {n_total} 个事件，其中 {n_diff} 个（{n_diff/n_total*100:.1f}%）分桶不一致"
          if n_total else "  没有事件")
    for record, e, bin_max, bin_mean in rows:
        print(f"    {record}  {fmt_ts(e['start'])}-{fmt_ts(e['end'])}  "
              f"conf_max={e['conf_max']:.2f}→桶[{bin_max}]  "
              f"conf_mean={e['conf_mean']:.2f}→桶[{bin_mean}]")


def main():
    ap = argparse.ArgumentParser(description="对比多个模型的推理结果")
    ap.add_argument("--day", required=True, help="日期子目录名，比如 2026_7_17")
    ap.add_argument("--model", action="append", required=True,
                     help="格式 label=result_root，可传多次，比如 "
                          "--model majority=infer_result_majority")
    ap.add_argument("--pairs", nargs="+", default=[],
                     help="要对比的模型对，格式 labelA:labelB，可传多个，"
                          "留空则自动对比所有相邻传入的--model两两组合")
    ap.add_argument("--log_file", default="")
    args = ap.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_f)
        print(f"[日志] 输出同时保存到: {args.log_file}\n")

    models = {}
    total_records_seen = None
    for item in args.model:
        if "=" not in item:
            print(f"[错误] --model 格式应为 label=result_root，收到: {item}")
            return
        label, result_root = item.split("=", 1)
        events_by_record, n_total = load_model_events(result_root, args.day)
        models[label] = events_by_record
        n_events = sum(len(v) for v in events_by_record.values())
        n_with_events = len(events_by_record)
        pct = n_with_events / n_total * 100 if n_total else 0.0
        print(f"[加载] {label} ({result_root}/{args.day})：{args.day} 当天共 {n_total} 条record，"
              f"其中 {n_with_events} 条（{pct:.1f}%）检测到抓挠，共 {n_events} 个抓挠事件")
        # 各模型理论上应该扫的是同一批原始record（同一天的同一批文件），
        # 数量对不上说明有的模型推理没跑全，或者用的数据目录不一致，提醒一下
        if total_records_seen is not None and n_total != total_records_seen:
            print(f"  [提示] record总数({n_total})跟之前加载的模型不一致"
                  f"(之前是{total_records_seen})，可能是这个模型的推理没跑全"
                  f"或者用的数据目录不一样")
        total_records_seen = n_total

    # ── 跨模型交集：哪些record是4个模型都判定有抓挠的，哪些是某个模型独有的 ──
    if len(models) >= 2:
        print(f"\n{'='*90}")
        print(f"  跨模型 record 交集分析（判定标准：这条record里模型检测到>=1个抓挠事件）")
        print(f"{'='*90}")
        record_sets = {label: set(events.keys()) for label, events in models.items()}
        common = set.intersection(*record_sets.values())
        print(f"  {len(models)}个模型共同判定有抓挠的record: {len(common)} 条")
        for r in sorted(common):
            print(f"    {r}")
        for label, rset in record_sets.items():
            others = set.union(*(s for l, s in record_sets.items() if l != label))
            only_this = rset - others
            print(f"\n  仅 {label} 判定有抓挠、其余{len(models)-1}个模型都没有的record: {len(only_this)} 条")
            for r in sorted(only_this):
                print(f"    {r}")

    pairs = []
    for p in args.pairs:
        if ":" not in p:
            print(f"[错误] --pairs 格式应为 labelA:labelB，收到: {p}")
            return
        a, b = p.split(":", 1)
        if a not in models or b not in models:
            print(f"[错误] {p} 里的模型没有通过 --model 提供")
            return
        pairs.append((a, b))

    for label_a, label_b in pairs:
        compare_pair(label_a, models[label_a], label_b, models[label_b])

    for label, events in models.items():
        report_bin_divergence(label, events)


if __name__ == "__main__":
    main()
