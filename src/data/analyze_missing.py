"""
统计每个类别的标注数据里，acc/gyro缺失值(NaN，蓝牙断联导致)有多少——
配合--missing_strategy用，跑训练前先看一眼缺失情况严不严重、集中在
哪个类别/哪个task，比蒙着头对比drop/ffill/none三个版本的最终指标更
直接：如果某个类别缺失率特别高，说明那批数据本身录制质量有问题，
不管选哪种missing_strategy，那个类别的效果都可能受影响。

跟labelstudio_to_custom.py读同样的Label Studio导出JSON+原始传感器CSV，
复用同一套_load_sensor_df()/_extract_rows()逻辑（--missing_strategy
用"none"强制不处理，这样才能看到真实的缺失情况，不然drop/ffill会
把NaN提前处理掉，统计出来的就是0）。

用法:
  python src/data/analyze_missing.py \\
    --json data/raw_custom/2026_8_11-2026_8_27_raw/merged_tmp.json \\
    --csv_dir data/raw_wit/
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labelstudio_to_custom import _load_sensor_df, _extract_rows  # noqa: E402

SENSOR_VALUE_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def _iter_task_segments(task, csv_dir):
    """跟labelstudio_to_custom.py的convert()同一套task解析逻辑，
    yield (task_id, subject_id, label, t0, t1, rows)。missing_strategy
    强制"none"——统计缺失情况必须看到真实NaN，不能被提前处理掉。"""
    task_id = task["id"]
    data = task.get("data", {})
    annotations = task.get("annotations", [])
    if not annotations:
        return

    is_multi = "csv1" in data or "csv2" in data
    if is_multi:
        sensor_map = {}
        for idx in ("1", "2"):
            url = data.get(f"csv{idx}", "")
            if url:
                res = _load_sensor_df(url, csv_dir, f"imu{idx}", missing_strategy="none")
                if res:
                    sensor_map[f"label{idx}"] = (res[0], res[1], res[2], f"task{task_id}_imu{idx}")
        if not sensor_map:
            return
        if len(sensor_map) == 1:
            sensor_map["label"] = next(iter(sensor_map.values()))

        for ann in annotations:
            for seg in ann.get("result", []):
                val = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                t0, t1 = val.get("start", ""), val.get("end", "")
                fn = seg.get("from_name", "")
                if not labels or not t0 or not t1 or fn not in sensor_map:
                    continue
                df, acc_cols, gyro_cols, subject_id = sensor_map[fn]
                rows = _extract_rows(df, acc_cols, gyro_cols, labels[0], t0, t1,
                                      subject_id, "ms2", None)
                if rows:
                    yield task_id, subject_id, labels[0], t0, t1, rows
    else:
        csv_url = data.get("csv", "")
        if not csv_url:
            return
        res = _load_sensor_df(csv_url, csv_dir, "imu", missing_strategy="none")
        if not res:
            return
        df, acc_cols, gyro_cols = res
        subject_id = f"task{task_id}"
        for ann in annotations:
            for seg in ann.get("result", []):
                val = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                t0, t1 = val.get("start", ""), val.get("end", "")
                if not labels or not t0 or not t1:
                    continue
                rows = _extract_rows(df, acc_cols, gyro_cols, labels[0], t0, t1,
                                      subject_id, "ms2", None)
                if rows:
                    yield task_id, subject_id, labels[0], t0, t1, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="merged_tmp.json（多个project-*.json合并后的）")
    ap.add_argument("--csv_dir", default="data/raw_wit/")
    ap.add_argument("--top_n", type=int, default=15,
                     help="缺失率最高的task明细，打印前N个（默认15）")
    args = ap.parse_args()

    tasks = json.load(open(args.json, encoding="utf-8"))
    print(f"[analyze_missing] 共 {len(tasks)} 个task")

    # 按类别统计
    label_total = defaultdict(int)
    label_nan = defaultdict(int)
    # 按task统计（找出缺失最集中的那几个task，方便去核实是不是那次录制
    # 蓝牙确实不稳）
    task_stats = []  # (task_id, subject_id, label, n_rows, n_nan_rows, nan_rate)

    for task in tasks:
        for task_id, subject_id, label, t0, t1, rows in _iter_task_segments(task, args.csv_dir):
            n_rows = len(rows)
            n_nan = sum(1 for r in rows if any(pd.isna(r[c]) for c in SENSOR_VALUE_COLS))
            label_total[label] += n_rows
            label_nan[label] += n_nan
            if n_nan:
                task_stats.append((task_id, subject_id, label, n_rows, n_nan, n_nan / n_rows))

    print(f"\n{'类别':<10}{'总行数':>10}{'缺失行数':>10}{'缺失率':>10}")
    print("-" * 42)
    for label in sorted(label_total, key=lambda k: -label_total[k]):
        total, nan = label_total[label], label_nan[label]
        rate = nan / total * 100 if total else 0.0
        flag = "  ⚠️ 缺失率偏高" if rate > 5 else ""
        print(f"{label:<10}{total:>10}{nan:>10}{rate:>9.2f}%{flag}")

    grand_total = sum(label_total.values())
    grand_nan = sum(label_nan.values())
    print("-" * 42)
    print(f"{'合计':<10}{grand_total:>10}{grand_nan:>10}"
          f"{(grand_nan/grand_total*100 if grand_total else 0):>9.2f}%")

    if task_stats:
        task_stats.sort(key=lambda t: -t[5])
        print(f"\n[analyze_missing] 缺失率最高的{min(args.top_n, len(task_stats))}个标注段"
              f"（可能是这几次录制蓝牙不稳）:")
        print(f"  {'task':<8}{'record_id':<20}{'label':<8}{'总行数':>8}{'缺失行数':>8}{'缺失率':>8}")
        for task_id, subject_id, label, n_rows, n_nan, rate in task_stats[:args.top_n]:
            print(f"  {task_id:<8}{subject_id:<20}{label:<8}{n_rows:>8}{n_nan:>8}{rate*100:>7.1f}%")
    else:
        print("\n[analyze_missing] 没有发现任何缺失值")


if __name__ == "__main__":
    main()
