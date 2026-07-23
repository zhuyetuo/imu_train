"""
分析 Label Studio 导出的标注 JSON，展示类别分布和数据量。

用法:
  python src/data/analyze_annotations.py --json data/raw_custom/labelstudio_export.json
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime

KEEP_LABELS = {"活动", "睡觉", "抓挠"}


def parse_dt(s):
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Label Studio 导出 JSON")
    parser.add_argument("--labels", nargs="*", default=list(KEEP_LABELS),
                        help="只统计这些标签（默认: 活动 睡觉 抓挠）")
    args = parser.parse_args()

    keep = set(args.labels)

    with open(args.json, encoding="utf-8") as f:
        tasks = json.load(f)

    # label → total_seconds, segment_count
    label_sec   = defaultdict(float)
    label_count = defaultdict(int)
    # sensor → label → seconds
    sensor_sec  = defaultdict(lambda: defaultdict(float))

    skipped_labels = defaultdict(float)
    total_tasks = len(tasks)
    annotated   = 0

    for task in tasks:
        anns = task.get("annotations", [])
        if not anns:
            continue
        annotated += 1
        for ann in anns:
            for seg in ann.get("result", []):
                val   = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                if not labels:
                    continue
                label = labels[0]
                t0 = parse_dt(val.get("start", ""))
                t1 = parse_dt(val.get("end",   ""))
                if t0 is None or t1 is None:
                    continue
                dur = (t1 - t0).total_seconds()
                sensor = seg.get("from_name", "unknown")  # label1 or label2

                if label in keep:
                    label_sec[label]   += dur
                    label_count[label] += 1
                    sensor_sec[sensor][label] += dur
                else:
                    skipped_labels[label] += dur

    print(f"\n{'='*52}")
    print(f"  标注分析  （共 {total_tasks} 个 task，{annotated} 个已标注）")
    print(f"{'='*52}")

    # ── 总览 ──────────────────────────────────────────────
    total_sec = sum(label_sec.values())
    print(f"\n【类别总览】（只统计: {', '.join(sorted(keep))}）")
    print(f"  {'类别':<8} {'片段数':>6} {'总时长':>10} {'占比':>7}")
    print(f"  {'-'*35}")
    for label in sorted(label_sec, key=lambda x: -label_sec[x]):
        sec  = label_sec[label]
        cnt  = label_count[label]
        pct  = sec / total_sec * 100 if total_sec > 0 else 0
        mins = int(sec // 60)
        secs = sec % 60
        print(f"  {label:<8} {cnt:>6}   {mins:>3}m{secs:>5.1f}s  {pct:>6.1f}%")
    print(f"  {'合计':<8} {sum(label_count.values()):>6}   "
          f"{int(total_sec//60):>3}m{total_sec%60:>5.1f}s  100.0%")

    # ── 按传感器 ──────────────────────────────────────────
    if len(sensor_sec) > 1:
        print(f"\n【按传感器分布】")
        for sensor in sorted(sensor_sec):
            s_total = sum(sensor_sec[sensor].values())
            parts = [f"{label}={sensor_sec[sensor][label]:.0f}s"
                     for label in sorted(sensor_sec[sensor])]
            print(f"  {sensor}: 总 {s_total:.0f}s  |  {',  '.join(parts)}")

    # ── 类别平衡建议 ─────────────────────────────────────
    if label_sec:
        max_sec = max(label_sec.values())
        min_sec = min(label_sec.values())
        ratio   = max_sec / min_sec if min_sec > 0 else float("inf")
        print(f"\n【平衡性】最多/最少 = {ratio:.1f}x", end="")
        if ratio > 5:
            min_label = min(label_sec, key=lambda x: label_sec[x])
            print(f"  ⚠️  '{min_label}' 数据偏少，建议补充标注")
        elif ratio > 2:
            print(f"  ⚠️  轻微不均衡，训练时已自动加权")
        else:
            print(f"  ✅  较均衡")

    # ── 跳过的其他类别 ────────────────────────────────────
    if skipped_labels:
        print(f"\n【其他类别（已跳过）】")
        for label, sec in sorted(skipped_labels.items(), key=lambda x: -x[1]):
            print(f"  {label}: {sec:.0f}s")

    # ── 估算窗口数 ────────────────────────────────────────
    WINDOW_S, STRIDE_S = 2, 1
    print(f"\n【估算可用窗口数】(窗口={WINDOW_S}s, 步长={STRIDE_S}s, 采样率=16Hz)")
    win_counts = {}
    for label in sorted(label_sec):
        sec = label_sec[label]
        est = max(0, int(sec - WINDOW_S) + 1)
        win_counts[label] = est
        print(f"  {label}: ~{est} 个窗口")

    # ── 合成数据建议 ──────────────────────────────────────
    if win_counts:
        max_wins  = max(win_counts.values())
        # 目标：最少类别窗口数 >= max 的 50%（3倍以内差距）
        TARGET_RATIO = 0.5
        # 单类别绝对下限
        ABS_MIN = 100
        need_syn = []
        for label, wins in sorted(win_counts.items()):
            target = max(int(max_wins * TARGET_RATIO), ABS_MIN)
            if wins < target:
                need_syn.append((label, wins, target))

        print(f"\n【合成数据建议】")
        if not need_syn:
            print(f"  ✅  各类别窗口数均衡，无需合成数据")
        else:
            for label, wins, target in need_syn:
                shortage = target - wins
                print(f"  ⚠️  {label}: 当前 ~{wins} 窗口，目标 ~{target}（还差 ~{shortage}），推荐命令：")
                print(f"      python src/data/synthesize_scratch.py \\")
                print(f"        --json <merged_tmp.json> --csv_dir data/raw_wit/ \\")
                print(f"        --output data/synthetic/{label}_<DATE>.npz \\")
                print(f"        --label {label} --hz 16 --n_aug 50 --target_windows {target}")

    print()


if __name__ == "__main__":
    main()
