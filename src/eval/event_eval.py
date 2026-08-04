"""
事件级评估（借助 ward-metrics 库）：对比模型输出的目标行为片段（如"抓挠"）
与真实标注在事件层面的对应关系——不只是逐帧准确率，而是区分
漏检(D)、碎片化(F)、合并(M)、纯误报(I') 这几类错误模式。

依赖: pip install ward-metrics   （import 名是 wardmetrics，注意不一致）

用法:
  # 扫描不同置信度阈值，看预测事件的可信度分布 + 各阈值下的事件级表现
  python src/eval/event_eval.py \\
    --labeled_csv data/raw_custom/2026_7_30/merged_all_labels_2026_7_30.csv \\
    --model results/processed_2026_7_30/16hz_remap_custom_3class_syn/ml_rf.pkl \\
    --hz 16 --target_label 抓挠 \\
    --confidence_threshold 0.0 0.4 0.5 0.6 0.7 0.8 0.9

  # 对比不同 merge_gap 取值（固定置信度阈值为0）
  python src/eval/event_eval.py \\
    --labeled_csv ... --model ... --confidence_threshold 0.0 \\
    --merge_gap 0 1 3 10   # 传给 --merge_gap 的仍是单值时忽略，见下方参数说明

原理:
  1. 从已标注CSV按 dog_id 提取真实的目标行为连续片段（事件），单位转成秒。
  2. 用给定模型对同一份数据做滑窗推理，取每个窗口 argmax 标签 + 该标签的
     置信度（predict_proba 最大值），逻辑与 infer_csv_scratch.py 一致。
  3. 先打印"窗口级置信度分布"：预测为 target_label 的窗口里，有多少落在
     各置信度阈值以上——直接回答"预测抓挠的窗口是不是大多集中在高置信度
     区间"这个问题，不需要先转成事件。
  4. 再对每个置信度阈值，只保留 conf>=阈值 的目标窗口合并成预测事件
     （merge_gap 固定，默认1s），喂给 wardmetrics.eval_events，看事件级的
     漏检/碎片化/合并/误报数量随阈值如何变化。
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
    """对单条狗的完整原始信号做滑窗推理，返回 (pred_labels, pred_confs, start_indices)。
    pred_confs 是每个窗口 argmax 类别自身的概率（与 infer_csv_scratch.py 语义一致）。"""
    X, starts = sliding_windows(data6, window_size, stride)
    if len(X) == 0:
        return [], [], []
    tilt = append_raw_tilt_batch(X)[:, :, 6:8]
    X_aligned = gravity_align_batch(X)
    X_full = np.concatenate([X_aligned, tilt], axis=2)
    feats = extract_features(X_full, hz, show_progress=False)
    probs = model.predict_proba(feats)
    pred_ids = np.argmax(probs, axis=1)
    pred_confs = probs[np.arange(len(probs)), pred_ids]
    pred_labels = [classes[int(p)] for p in pred_ids]
    return pred_labels, pred_confs, starts


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


def pred_windows_to_segments(pred_labels, pred_confs, starts, window_size, hz, target_label, conf_threshold):
    """把逐窗口预测里 label==target_label 且 conf>=conf_threshold 的窗口，
    合并成连续的原始片段（未做 merge_gap 合并）"""
    raw_segs = []
    in_seg = False
    seg_start = None
    prev_end = None
    for lbl, conf, s in zip(pred_labels, pred_confs, starts):
        hit = (lbl == target_label) and (conf >= conf_threshold)
        start_sec = s / hz
        end_sec = (s + window_size) / hz
        if hit:
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


def eval_at_merge_gap(gt_events, raw_segs, merge_gap):
    det_events = merge_segments(sorted(raw_segs), merge_gap)
    if not det_events:
        return len(det_events), 0.0, 0.0, 0.0, len(gt_events), 0, 0, 0
    gt_scores, det_scores, detailed, standard = eval_events(gt_events, det_events)
    p, r = standard["precision"], standard["recall"]
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return len(det_events), p, r, f1, detailed["D"], detailed["F"], detailed["M"], detailed["I'"]


def main():
    ap = argparse.ArgumentParser(description="事件级评估：真实标注 vs 模型预测的目标行为片段对应关系")
    ap.add_argument("--labeled_csv", required=True)
    ap.add_argument("--model", required=True, help="模型 .pkl 路径")
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--target_label", default="抓挠")
    ap.add_argument("--dog_id_col", default="dog_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--merge_gap", type=float, default=1.0,
                     help="事件级评估用的合并间隔秒数（默认1s，单值，见 README 说明）")
    ap.add_argument("--confidence_threshold", type=float, nargs="+",
                     default=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                     help="要扫描的置信度阈值列表（默认 0.0 0.4 0.5 0.6 0.7 0.8 0.9）")
    args = ap.parse_args()

    model = joblib.load(args.model)
    meta_path = args.model.replace(".pkl", ".json")
    classes, window_s, stride_s = [], 2.0, 1.0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        classes  = meta.get("classes", [])
        window_s = float(meta.get("window_s", 2.0))
        stride_s = float(meta.get("stride_s", 1.0))
    else:
        classes = list(model.classes_) if hasattr(model, "classes_") else []
    if args.target_label not in classes:
        print(f"[错误] 模型类别中没有 '{args.target_label}': {classes}")
        return
    window_size = int(window_s * args.hz)
    stride = max(1, int(stride_s * args.hz))
    print(f"[模型] 类别={classes}  窗口={window_s}s  步长={stride_s}s  merge_gap={args.merge_gap}s")

    df = pd.read_csv(args.labeled_csv)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    acc_cols = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    if acc_cols is None or gyro_cols is None:
        print(f"[错误] 找不到 acc/gyro 列: {list(df.columns)}")
        return

    gt_events_all = []             # 拼接所有狗的真实事件（带偏移）
    target_window_confs_all = []   # 所有 argmax==target_label 窗口的置信度（不带偏移，纯统计用）
    dog_window_records = []        # [(pred_labels, pred_confs, starts, offset), ...] 每条狗，供后续按阈值重建片段

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
        pred_labels, pred_confs, starts = predict_dog(data6, model, classes, window_size, stride, args.hz)
        dog_window_records.append((pred_labels, pred_confs, starts, offset))
        target_window_confs_all.extend(
            c for lbl, c in zip(pred_labels, pred_confs) if lbl == args.target_label
        )

    gt_events_all = sorted(gt_events_all)
    target_window_confs_all = np.array(target_window_confs_all)
    print(f"\n真实 '{args.target_label}' 事件数: {len(gt_events_all)}")
    print(f"模型预测为 '{args.target_label}' 的窗口总数（未按置信度过滤）: {len(target_window_confs_all)}")

    if not gt_events_all:
        print("[警告] 没有真实事件，无法评估")
        return

    # ── 窗口级置信度分布 ──────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  窗口级置信度分布（预测为 '{args.target_label}' 的窗口，按阈值累计保留比例）")
    print(f"{'='*78}")
    print(f"  {'阈值>=':>8}{'保留窗口数':>12}{'占比':>10}")
    n_total_windows = len(target_window_confs_all)
    for thr in args.confidence_threshold:
        n_keep = int((target_window_confs_all >= thr).sum()) if n_total_windows else 0
        pct = n_keep / n_total_windows * 100 if n_total_windows else 0.0
        print(f"  {thr:>8.2f}{n_keep:>12}{pct:>9.1f}%")

    # ── 事件级评估，按置信度阈值扫描 ──────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  不同置信度阈值下的事件级评估（merge_gap={args.merge_gap}s 固定）")
    print(f"{'='*78}")
    header = f"  {'阈值>=':>8}{'预测事件数':>10}{'precision':>12}{'recall':>10}{'F1':>8}" \
             f"{'漏检D':>8}{'碎片F':>8}{'合并M':>8}{'误报I‘':>8}"
    print(header)

    for thr in args.confidence_threshold:
        raw_segs = []
        for pred_labels, pred_confs, starts, offset in dog_window_records:
            segs = pred_windows_to_segments(pred_labels, pred_confs, starts, window_size,
                                            args.hz, args.target_label, thr)
            raw_segs.extend((s + offset, e + offset) for s, e in segs)
        n_det, p, r, f1, n_d, n_f, n_m, n_insert = eval_at_merge_gap(gt_events_all, raw_segs, args.merge_gap)
        row = (f"  {thr:>8.2f}{n_det:>10}{p:>12.3f}{r:>10.3f}{f1:>8.3f}"
               f"{n_d:>8}{n_f:>8}{n_m:>8}{n_insert:>8}")
        print(row)

    print(f"""
{'='*78}
  怎么解读：
  - 窗口级分布表：如果预测为目标行为的窗口大量集中在低置信度区间（比如
    阈值0.5时保留比例已经腰斩），说明模型对这个类别整体不够自信，提高
    阈值虽然能过滤掉一部分误报，但可能连带砍掉不少真实检测。
  - 事件级表：⚠️ precision/recall 由 ward-metrics 库计算，把"合并"(M)
    当命中处理，不算错误，不能只看这两个数字。重点看 D/F/M/I' 的绝对
    数量随阈值提高怎么变化——理想情况下，阈值提高时 I'（纯误报）应该
    明显下降，而 D（漏检）不应该涨得太快；如果阈值刚提到0.5、0.6，D就
    涨得很猛，说明模型对真实抓挠的置信度普遍也不高，问题不是"阈值没调
    好"，而是训练数据/特征本身对这个类别的区分度还不够，需要回到标注
    数据质量和数量上想办法，而不是靠调阈值解决。
{'='*78}
""")


if __name__ == "__main__":
    main()
