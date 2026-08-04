"""
事件级评估（借助 ward-metrics 库）：对比模型输出的目标行为片段（如"抓挠"）
与真实标注在事件层面的对应关系——不只是逐帧准确率，而是区分
漏检(D)、碎片化(F)、合并(M)、纯误报(I') 这几类错误模式。

依赖: pip install ward-metrics   （import 名是 wardmetrics，注意不一致）

用法:
  # 对比不同 merge_gap 取值对碎片化/合并的影响
  python src/eval/event_eval.py \\
    --labeled_csv data/raw_custom/2026_7_30/merged_all_labels_2026_7_30.csv \\
    --model results/processed_2026_7_30/16hz_remap_custom_3class_syn/ml_rf.pkl \\
    --hz 16 --target_label 抓挠 \\
    --merge_gap 0 3 10

原理:
  1. 从已标注CSV按 dog_id 提取真实的目标行为连续片段（事件），单位转成秒。
  2. 用给定模型对同一份数据做滑窗推理，得到逐窗口预测，合并成预测片段
     （按不同 merge_gap 值分别合并，复现 infer_csv_scratch.py 的合并逻辑）。
  3. 把真实事件和预测事件喂给 wardmetrics.eval_events，得到事件级
     precision/recall，以及漏检/碎片化/合并/误报的具体数量。
  4. 跨多个 merge_gap 值对比，直接回答"调大/调小 merge_gap 对碎片化和
     合并谁更有利"这个问题，而不是凭感觉猜。
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))

from gravity_align import gravity_align_batch, append_raw_tilt_batch
from features import extract_features

try:
    from wardmetrics.core_methods import eval_events
    from wardmetrics.utils import print_standard_event_metrics, print_detailed_event_metrics
except ImportError:
    print("[错误] 未安装 ward-metrics，请先: pip install ward-metrics")
    sys.exit(1)

ACC_CANDIDATES  = [["acc_x", "acc_y", "acc_z"], ["AccX", "AccY", "AccZ"], ["AX", "AY", "AZ"], ["ax", "ay", "az"]]
GYRO_CANDIDATES = [["gyro_x", "gyro_y", "gyro_z"], ["gyr_x", "gyr_y", "gyr_z"], ["GyroX", "GyroY", "GyroZ"], ["GX", "GY", "GZ"]]

DOG_TIME_OFFSET = 1e7  # 每条狗的时间轴偏移量，避免跨狗事件在拼接后重叠


def find_cols(cols, candidates):
    for g in candidates:
        if all(c in cols for c in g):
            return g
    return None


def find_contiguous_segments(labels, min_len=1):
    segs = []
    n = len(labels)
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        if j - i >= min_len:
            segs.append((i, j, labels[i]))
        i = j
    return segs


def sliding_windows(data, window_size, stride):
    starts = list(range(0, len(data) - window_size + 1, stride))
    if not starts:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32), []
    return np.stack([data[s:s + window_size] for s in starts]), starts


def predict_dog(data6, model, classes, window_size, stride, hz):
    """对单条狗的完整原始信号做滑窗推理，返回 (pred_labels, start_indices)"""
    X, starts = sliding_windows(data6, window_size, stride)
    if len(X) == 0:
        return [], []
    tilt = append_raw_tilt_batch(X)[:, :, 6:8]
    X_aligned = gravity_align_batch(X)
    X_full = np.concatenate([X_aligned, tilt], axis=2)
    feats = extract_features(X_full, hz, show_progress=False)
    preds = model.predict(feats)
    pred_labels = [classes[int(p)] for p in preds]
    return pred_labels, starts


def merge_segments(intervals, merge_gap):
    """intervals: [(start_sec, end_sec), ...] 已排序，合并间隔<=merge_gap的相邻段"""
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def pred_windows_to_segments(pred_labels, starts, window_size, hz, target_label):
    """把逐窗口预测里等于 target_label 的窗口，合并成连续的原始片段（未做 merge_gap 合并）"""
    raw_segs = []
    in_seg = False
    seg_start = None
    prev_end = None
    for lbl, s in zip(pred_labels, starts):
        start_sec = s / hz
        end_sec = (s + window_size) / hz
        if lbl == target_label:
            if not in_seg:
                in_seg = True
                seg_start = start_sec
            prev_end = end_sec
        else:
            if in_seg:
                raw_segs.append((seg_start, prev_end))
                in_seg = False
    if in_seg:
        raw_segs.append((seg_start, prev_end))
    return raw_segs


def main():
    ap = argparse.ArgumentParser(description="事件级评估：真实标注 vs 模型预测的目标行为片段对应关系")
    ap.add_argument("--labeled_csv", required=True)
    ap.add_argument("--model", required=True, help="模型 .pkl 路径")
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--target_label", default="抓挠")
    ap.add_argument("--dog_id_col", default="dog_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--merge_gap", type=float, nargs="+", default=[0, 3, 10],
                     help="要对比的 merge_gap 取值列表（秒），默认对比 0/3/10")
    args = ap.parse_args()

    model = joblib.load(args.model)
    meta_path = args.model.replace(".pkl", ".json")
    classes, window_s, stride_s = [], 2.0, 1.0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        classes    = meta.get("classes", [])
        window_s   = float(meta.get("window_s", 2.0))
        stride_s   = float(meta.get("stride_s", 1.0))
    else:
        classes = list(model.classes_) if hasattr(model, "classes_") else []
    if args.target_label not in classes:
        print(f"[错误] 模型类别中没有 '{args.target_label}': {classes}")
        return
    window_size = int(window_s * args.hz)
    stride = max(1, int(stride_s * args.hz))
    print(f"[模型] 类别={classes}  窗口={window_s}s  步长={stride_s}s")

    df = pd.read_csv(args.labeled_csv)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    acc_cols = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    if acc_cols is None or gyro_cols is None:
        print(f"[错误] 找不到 acc/gyro 列: {list(df.columns)}")
        return

    gt_events_all = []       # 拼接所有狗的真实事件（带偏移）
    raw_pred_segs_all = []   # 拼接所有狗的"原始"预测片段（未合并，带偏移）

    dog_ids = sorted(df[args.dog_id_col].unique())
    print(f"共 {len(dog_ids)} 条狗，逐条推理中...")
    for i, dog_id in enumerate(dog_ids):
        sub = df[df[args.dog_id_col] == dog_id].reset_index(drop=True)
        labels = sub[args.label_col].values
        offset = i * DOG_TIME_OFFSET

        gt_segs = find_contiguous_segments(labels)
        for start, end, lbl in gt_segs:
            if lbl == args.target_label:
                gt_events_all.append((start / args.hz + offset, end / args.hz + offset))

        data6 = np.concatenate(
            [sub[acc_cols].values, sub[gyro_cols].values], axis=1
        ).astype(np.float32)
        pred_labels, starts = predict_dog(data6, model, classes, window_size, stride, args.hz)
        raw_segs = pred_windows_to_segments(pred_labels, starts, window_size, args.hz, args.target_label)
        raw_pred_segs_all.extend((s + offset, e + offset) for s, e in raw_segs)

    gt_events_all = sorted(gt_events_all)
    raw_pred_segs_all = sorted(raw_pred_segs_all)
    print(f"\n真实 '{args.target_label}' 事件数: {len(gt_events_all)}")
    print(f"模型原始预测片段数（合并前）: {len(raw_pred_segs_all)}")

    if not gt_events_all:
        print("[警告] 没有真实事件，无法评估")
        return

    print(f"\n{'='*78}")
    print(f"  不同 merge_gap 取值下的事件级评估对比")
    print(f"{'='*78}")
    header = f"  {'merge_gap':>10}{'预测事件数':>10}{'precision':>12}{'recall':>10}{'F1':>8}" \
             f"{'漏检D':>8}{'碎片F':>8}{'合并M':>8}{'误报I‘':>8}"
    print(header)

    for mg in args.merge_gap:
        det_events = merge_segments(raw_pred_segs_all, mg)
        if not det_events:
            # wardmetrics 对空预测列表直接报错，手动兜底：全部真实事件都算漏检
            p, r, f1 = 0.0, 0.0, 0.0
            n_d, n_f, n_m, n_insert = len(gt_events_all), 0, 0, 0
        else:
            gt_scores, det_scores, detailed, standard = eval_events(gt_events_all, det_events)
            p, r = standard["precision"], standard["recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            n_d, n_f, n_m, n_insert = detailed["D"], detailed["F"], detailed["M"], detailed["I'"]
        row = (f"  {mg:>10.1f}{len(det_events):>10}{p:>12.3f}{r:>10.3f}{f1:>8.3f}"
               f"{n_d:>8}{n_f:>8}{n_m:>8}{n_insert:>8}")
        print(row)

    print(f"""
{'='*78}
  列说明：
    precision/recall/F1  ward-metrics库内置的事件级指标——⚠️ 注意：这个
                          precision/recall 把"合并"(M)当作命中处理，不算
                          错误，所以 merge_gap 很大、把好几次抓挠合并成一次
                          时，precision/recall 依然会显示很高（如上面例子），
                          不能只看这两个数字判断好坏。
    漏检D   真实事件完全没被检测到
    碎片F   一个真实事件被切成多个预测片段（merge_gap太小容易出现）
    合并M   多个真实事件被错误合并成一个预测片段（merge_gap太大容易出现，
            但不计入库自带的precision/recall，需要单独盯着这一列看）
    误报I'  预测出来但根本不存在对应真实事件的纯虚警

  实际判断 merge_gap 好不好，主要看 D/F/M/I' 这四列的绝对数量，而不是
  precision/recall——如果你关心"统计每天抓挠次数"，合并(M)对你来说是
  实实在在的错误（次数被低估了），但库自带指标不会体现这一点。
  典型规律：merge_gap 增大 → 碎片F减少，但合并M可能增加，需要根据下游
  用途（比如次数统计 vs 只关心有没有发生）权衡选择。
{'='*78}
""")


if __name__ == "__main__":
    main()
