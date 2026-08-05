"""
统计某个标签（默认"抓挠"）在 Label Studio 标注JSON里，每一段原始标注片段的
时长分布 + 详细定位信息（project文件、task_id、csv/video链接），方便：
  1. 看清楚"189个原始片段"里，长短分布到底是什么样的（比如一大堆<1s的碎片会
     严重拉低实际可用的完整事件数——event_eval.py 里"识别出多少个事件"看的是
     合并后的连续事件，不是这里统计的原始标注片段数，两者概念不同）
  2. 挨个去 Label Studio 网页版核实/调整某一条标注（比如时长异常短或异常长的）

用法:
  python src/data/analyze_label_segments.py \\
    --json data/raw_custom/2026_7_30/merged_tmp.json \\
    --json_dir data/raw_custom/2026_7_30 \\
    --label 抓挠 \\
    --log_file logs/scratch_segments_2026_7_30.log
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_dt(s):
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def extract_task_id(s):
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


def extract_project_no(filename):
    m = re.search(r"project-(\d+)-at-", os.path.basename(filename))
    return m.group(1) if m else None


def build_project_lookup(json_dir, pattern="project-*.json"):
    """扫描 project-*.json，建立 task_id -> (文件名, project编号, task字典) 的查找表"""
    lookup = {}
    files = sorted(glob.glob(os.path.join(json_dir, pattern)))
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                tasks = json.load(f)
        except Exception:
            continue
        project_no = extract_project_no(fpath)
        for task in tasks:
            tid = task.get("id")
            if tid is not None and tid not in lookup:
                lookup[tid] = (os.path.basename(fpath), project_no, task)
    return lookup


DUR_BUCKETS = [
    ("<1s",    0,    1),
    ("1-3s",   1,    3),
    ("3-9s",   3,    9),
    ("9s以上",  9, float("inf")),
]


def bucket_of(dur):
    for name, lo, hi in DUR_BUCKETS:
        if lo <= dur < hi:
            return name
    return DUR_BUCKETS[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Label Studio 导出/合并后的 JSON")
    ap.add_argument("--label", default="抓挠", help="要统计的标签（默认: 抓挠）")
    ap.add_argument("--json_dir", default="", help="存放 project-*.json 的目录，"
                     "用于反查每条片段属于哪个 project（不传则跳过定位信息）")
    ap.add_argument("--sort_by", choices=["time", "duration"], default="duration",
                     help="逐条列表排序方式：duration=按时长从短到长（默认，方便优先核实异常短的）"
                          "，time=按标注开始时间")
    ap.add_argument("--log_file", default="", help="同时把输出写到这个文件")
    args = ap.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = Tee(sys.__stdout__, log_f)
        print(f"[日志] 输出同时保存到: {args.log_file}\n")

    with open(args.json, encoding="utf-8") as f:
        tasks = json.load(f)

    project_lookup = build_project_lookup(args.json_dir) if args.json_dir else None
    if args.json_dir and not project_lookup:
        print(f"[警告] {args.json_dir} 下没找到 project-*.json，跳过 project 反查")

    segments = []  # (dur, t0, t1, task_id, sensor_key)
    for task in tasks:
        task_id = task["id"]
        for ann in task.get("annotations", []):
            for seg in ann.get("result", []):
                val = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                if not labels or labels[0] != args.label:
                    continue
                t0_str, t1_str = val.get("start", ""), val.get("end", "")
                t0, t1 = parse_dt(t0_str), parse_dt(t1_str)
                if t0 is None or t1 is None:
                    continue
                dur = (t1 - t0).total_seconds()
                sensor = seg.get("from_name", "")  # label1/label2/单sensor为空
                segments.append((dur, t0, t0_str, t1_str, task_id, sensor))

    print(f"\n{'='*78}")
    print(f"  标签 '{args.label}'  共 {len(segments)} 个原始标注片段")
    print(f"{'='*78}")

    if not segments:
        print("  没有找到任何片段")
        return

    durs = [s[0] for s in segments]
    total_sec = sum(durs)
    print(f"  总时长: {total_sec:.1f}s ({total_sec/60:.1f}min)   "
          f"最短: {min(durs):.2f}s   最长: {max(durs):.2f}s   "
          f"平均: {total_sec/len(durs):.2f}s")

    # ── 时长分布直方图 ──────────────────────────────────────
    print(f"\n【时长分布】")
    print(f"  {'区间':<10}{'片段数':>8}{'占比':>8}{'总时长':>12}")
    print(f"  {'-'*40}")
    for name, lo, hi in DUR_BUCKETS:
        in_bucket = [d for d in durs if lo <= d < hi]
        pct = len(in_bucket) / len(durs) * 100
        print(f"  {name:<10}{len(in_bucket):>8}{pct:>7.1f}%{sum(in_bucket):>10.1f}s")

    # ── 逐条列表：project/task_id/csv链接，方便去 Label Studio 核实调整 ──
    print(f"\n{'='*78}")
    print(f"  逐条片段明细（按 {'时长从短到长' if args.sort_by == 'duration' else '时间先后'} 排序）")
    print(f"{'='*78}")

    if args.sort_by == "duration":
        segments.sort(key=lambda s: s[0])
    else:
        segments.sort(key=lambda s: s[1])

    for i, (dur, t0, t0_str, t1_str, task_id, sensor) in enumerate(segments, 1):
        print(f"\n  [{i}/{len(segments)}]  时长={dur:.2f}s  {t0_str} → {t1_str}"
              f"{'  sensor=' + sensor if sensor else ''}")
        if project_lookup is not None:
            if task_id not in project_lookup:
                print(f"      [未找到 task_id={task_id} 对应的 project 信息]")
            else:
                fname, project_no, task = project_lookup[task_id]
                inner_id = task.get("inner_id")
                inner_str = f"  inner_id(项目内序号)={inner_id}" if inner_id is not None else ""
                print(f"      project文件: {fname}  project编号: {project_no or '未知'}  "
                      f"task_id={task_id}{inner_str}")
                data = task.get("data", {})
                for k in ("csv", "csv1", "csv2", "video1", "video2", "cam1", "cam2"):
                    if k in data:
                        print(f"      data.{k}: {data[k]}")
        else:
            print(f"      task_id={task_id}（未传 --json_dir，跳过 project 反查）")

    print()


if __name__ == "__main__":
    main()
