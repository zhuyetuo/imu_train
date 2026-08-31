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

用法（行级统计，默认）:
  python src/data/analyze_missing.py \\
    --json data/raw_custom/2026_8_11-2026_8_27_raw/merged_tmp.json \\
    --csv_dir data/raw_wit/

用法（额外加窗口级统计，跟训练时的切窗方式对齐才有意义，
--source_hz/--hz/--window_s/--stride_s要和train_custom.sh里用的一致）:
  python src/data/analyze_missing.py \\
    --json data/raw_custom/2026_8_11-2026_8_27_raw/merged_tmp.json \\
    --csv_dir data/raw_wit/ \\
    --windows --source_hz 16 --hz 16 --window_s 1 --stride_s 0.5
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labelstudio_to_custom import _load_sensor_df, _extract_rows  # noqa: E402
from preprocess import downsample  # noqa: E402

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


def _analyze_windows(tasks, csv_dir, source_hz, target_hz, window_s, stride_s):
    """按训练管道同样的方式切窗口（downsample()+滑窗），统计每个类别里
    完整窗口(不含NaN) vs 含NaN窗口的数量/占比。跟preprocess.py的窗口
    定义对齐——window_size/stride都是在target_hz下按秒数换算的采样点数，
    这样这里数出来的"能不能用"才是训练时实际会遇到的窗口，不是随便
    切一刀的数字。

    每个标注段（segment）单独切窗口，不跨段拼接——preprocess.py里
    process_split/process_label_concat也是按segment切，段与段之间
    时间不连续，拼起来窗口没有物理意义。

    source_hz==target_hz时跳过downsample()（常见情况，比如本来就是
    16Hz采集16Hz训练）；不等时才真正重采样，重采样本身可能因为
    resample_poly的滤波把个别行的NaN扩散到邻近输出点，这也是真实
    训练时会发生的情况，所以这里不特意规避。"""
    window_size = int(round(window_s * target_hz))
    stride = int(round(stride_s * target_hz))
    if window_size <= 0 or stride <= 0:
        raise ValueError(f"window_s/stride_s换算到target_hz={target_hz}后得到"
                          f"window_size={window_size}, stride={stride}，必须>0")

    label_clean = defaultdict(int)
    label_nan = defaultdict(int)

    for task in tasks:
        for task_id, subject_id, label, t0, t1, rows in _iter_task_segments(task, csv_dir):
            arr = np.array([[r[c] for c in SENSOR_VALUE_COLS] for r in rows], dtype=np.float64)
            nan_mask = np.isnan(arr).any(axis=1)
            if source_hz != target_hz:
                # downsample()第二个参数是标签数组，随便传个占位的，
                # 这里只要重采样后的数据arr_ds，标签不用
                dummy_labels = np.zeros(len(arr), dtype=np.int64)
                arr, _ = downsample(arr, dummy_labels, source_hz, target_hz)
                nan_mask = np.isnan(arr).any(axis=1)

            n = len(arr)
            if n < window_size:
                continue
            for start in range(0, n - window_size + 1, stride):
                end = start + window_size
                if nan_mask[start:end].any():
                    label_nan[label] += 1
                else:
                    label_clean[label] += 1

    return label_clean, label_nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="merged_tmp.json（多个project-*.json合并后的）")
    ap.add_argument("--csv_dir", default="data/raw_wit/")
    ap.add_argument("--top_n", type=int, default=15,
                     help="缺失率最高的task明细，打印前N个（默认15）")
    ap.add_argument("--windows", action="store_true",
                     help="额外统计窗口级别的缺失情况（每个类别切分成多少个训练"
                          "窗口，其中多少个完整/多少个含NaN），跟train_custom.sh"
                          "用同样的--hz/--window_s/--stride_s才有意义")
    ap.add_argument("--source_hz", type=int, default=16, help="原始采集频率（--windows用）")
    ap.add_argument("--hz", type=int, default=16, help="训练目标频率（--windows用）")
    ap.add_argument("--window_s", type=float, default=2.0, help="窗口长度秒数（--windows用）")
    ap.add_argument("--stride_s", type=float, default=1.0, help="窗口步长秒数（--windows用）")
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

    if args.windows:
        label_clean, label_nan = _analyze_windows(
            tasks, args.csv_dir, args.source_hz, args.hz, args.window_s, args.stride_s)
        print(f"\n[analyze_missing] 窗口级统计（source_hz={args.source_hz} -> hz={args.hz}, "
              f"window_s={args.window_s}, stride_s={args.stride_s}）:")
        print(f"\n{'类别':<10}{'总窗口数':>10}{'完整窗口':>10}{'含NaN窗口':>10}{'NaN占比':>10}")
        print("-" * 52)
        all_labels = sorted(set(label_clean) | set(label_nan),
                             key=lambda k: -(label_clean[k] + label_nan[k]))
        for label in all_labels:
            clean, nan = label_clean[label], label_nan[label]
            total = clean + nan
            rate = nan / total * 100 if total else 0.0
            flag = "  ⚠️ 建议丢弃部分窗口" if rate > 5 else ""
            print(f"{label:<10}{total:>10}{clean:>10}{nan:>10}{rate:>9.2f}%{flag}")


if __name__ == "__main__":
    main()
