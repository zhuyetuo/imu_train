"""
事件级评估（借助 ward-metrics 库）：对比模型输出的目标行为片段（如"抓挠"）
与真实标注在事件层面的对应关系——不只是逐帧准确率，而是区分
漏检(D)、碎片化(F)、合并(M)、纯误报(I') 这几类错误模式。

依赖: pip install ward-metrics   （import 名是 wardmetrics，注意不一致）

用法:
  # 网格搜索：置信度阈值 × merge_gap，自动找出综合最优组合
  # （每次换新模型/新数据重新跑一遍即可，不用每次手动试）
  python src/eval/event_eval.py \\
    --labeled_csv data/raw_custom/2026_7_30/merged_all_labels_2026_7_30.csv \\
    --model results/processed_2026_7_30/16hz_remap_custom_3class_syn/ml_rf.pkl \\
    --hz 16 --target_label 抓挠 \\
    --confidence_threshold 0.0 0.4 0.5 0.6 0.7 0.8 0.9 \\
    --merge_gap 1 1.5 2 2.5 3

原理:
  1. 从已标注CSV按 dog_id 提取真实的目标行为连续片段（事件），单位转成秒。
  2. 用给定模型对同一份数据做滑窗推理一次，取每个窗口 argmax 标签 + 该标签
     的置信度（predict_proba 最大值），逻辑与 infer_csv_scratch.py 一致。
  3. 先打印"窗口级置信度分布"：预测为 target_label 的窗口里，有多少落在
     各置信度阈值以上——不需要先转成事件就能看出模型整体自不自信。
  4. 对 (置信度阈值, merge_gap) 的每一种组合都跑一遍事件级评估，打印完整
     网格，并按论文里定义的 F1e 指标（把碎片化F、合并M都算作错误，不像
     ward-metrics库自带的precision/recall那样把M当命中）自动挑出最优组合，
     直接给出下次推理该用的参数，而不是人工盯着表格猜。
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))

from gravity_align import gravity_align_batch, append_raw_tilt_batch
from features import extract_features
from find_task_project import extract_task_id, extract_project_no

try:
    from wardmetrics.core_methods import eval_events
except ImportError:
    print("[错误] 未安装 ward-metrics，请先: pip install ward-metrics")
    sys.exit(1)

ACC_CANDIDATES  = [["acc_x", "acc_y", "acc_z"], ["AccX", "AccY", "AccZ"], ["AX", "AY", "AZ"], ["ax", "ay", "az"]]
GYRO_CANDIDATES = [["gyro_x", "gyro_y", "gyro_z"], ["gyr_x", "gyr_y", "gyr_z"], ["GyroX", "GyroY", "GyroZ"], ["GX", "GY", "GZ"]]

DOG_TIME_OFFSET = 1e7  # 每条狗的时间轴偏移量，避免跨狗事件在拼接后重叠


class Tee:
    """把 print 输出同时写到终端和日志文件"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


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


def window_majority_labels(labels, starts, window_size):
    """每个窗口内真实标签的多数投票，与 preprocess.py 的 sliding_window 逻辑一致，
    用于跟模型的逐窗口预测对齐，算窗口级（逐帧）分类报告"""
    from collections import Counter
    return [Counter(labels[s:s + window_size]).most_common(1)[0][0] for s in starts]


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


def print_project_info(dog_id, project_lookup):
    """按 dog_id（如 task496_imu1）反查并打印所属 project 文件 + video/csv 链接"""
    tid = extract_task_id(dog_id)
    if tid is None or tid not in project_lookup:
        print(f"      [未找到 {dog_id} 对应的 project 信息]")
        return
    fname, project_no, task = project_lookup[tid]
    print(f"      project文件: {fname}  project编号: {project_no or '未知'}")
    data = task.get("data", {})
    for k in ("csv", "video1", "video2", "cam1", "cam2"):
        if k in data:
            print(f"      data.{k}: {data[k]}")


def find_merge_groups(gt_events_meta, det_events):
    """找出被同一个预测事件"吃进去"的多个真实事件（>=2个才算合并组）。
    gt_events_meta: [(dog_id, local_start, local_end, global_start, global_end), ...]
    det_events: [(global_start, global_end), ...] 已按 merge_gap 合并后的预测事件
    返回: [[(dog_id, local_start, local_end), ...], ...] 每个子列表是一组被合并的真实事件，按时间排序"""
    groups = []
    for det_start, det_end in det_events:
        hits = [
            (dog_id, ls, le) for dog_id, ls, le, gs, ge in gt_events_meta
            if gs < det_end and ge > det_start  # 与预测事件有重叠
        ]
        if len(hits) >= 2:
            hits.sort(key=lambda h: h[1])
            groups.append(hits)
    return groups


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


def compute_f1e(detailed):
    """按论文定义的事件级 F1e：TP=C，FN=D+F+FM+M（碎片化/合并都算错误），
    FP=M'+FM'+F'+I'。与 ward-metrics 库自带、把"合并"当命中的 precision/
    recall 不同，这个才是自动选参数时该用的指标。"""
    tp = detailed["C"]
    fn = detailed["D"] + detailed["F"] + detailed["FM"] + detailed["M"]
    fp = detailed["M'"] + detailed["FM'"] + detailed["F'"] + detailed["I'"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1e = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1e


def run_grid(thr_values, mg_values, dog_window_records, gt_events_all,
             target_label, window_size, hz, desc="网格搜索", show_progress=True):
    """跑一遍 (置信度阈值 x merge_gap) 网格，返回结果列表
    [(thr, mg, n_det, n_d, n_f, n_m, n_insert, f1e_p, f1e_r, f1e, lib_p, lib_r), ...]
    耗时主要在按阈值重建预测片段这一层（外层循环），gap 合并很快，
    所以进度条按阈值粒度显示，能大致反映整体进度。"""
    results = []
    best_f1e_so_far = 0.0
    it = thr_values
    if show_progress:
        from tqdm import tqdm
        it = tqdm(thr_values, desc=desc, unit="thr")
    for thr in it:
        raw_segs = []
        for pred_labels, pred_confs, starts, offset in dog_window_records:
            segs = pred_windows_to_segments(pred_labels, pred_confs, starts, window_size,
                                            hz, target_label, thr)
            raw_segs.extend((s + offset, e + offset) for s, e in segs)
        raw_segs = sorted(raw_segs)
        for mg in mg_values:
            (n_det, lib_p, lib_r, lib_f1, n_d, n_f, n_m, n_insert,
             f1e_p, f1e_r, f1e) = eval_at_merge_gap(gt_events_all, raw_segs, mg)
            results.append((thr, mg, n_det, n_d, n_f, n_m, n_insert, f1e_p, f1e_r, f1e, lib_p, lib_r))
            best_f1e_so_far = max(best_f1e_so_far, f1e)
        if show_progress:
            it.set_postfix(best_f1e=f"{best_f1e_so_far:.3f}")
    return results


def print_grid_table(results, title, top_n=None):
    insert_label = "I'"
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    rows = results if top_n is None else results[:top_n]
    if top_n is not None:
        header = f"  {'rank':<4}{'thr>=':>7}{'gap':>6}{'n_pred':>7}{'D':>7}{'F':>7}{'M':>7}{insert_label:>7}{'F1e':>8}"
        print(header)
        for rank, r in enumerate(rows, 1):
            thr, mg, n_det, n_d, n_f, n_m, n_insert = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
            f1e = r[9]
            print(f"  {rank:<4}{thr:>7.3f}{mg:>6.1f}{n_det:>7}{n_d:>7}{n_f:>7}{n_m:>7}{n_insert:>7}{f1e:>8.3f}")
    else:
        header = (f"  {'thr>=':>7}{'gap':>6}{'n_pred':>7}{'D':>7}{'F':>7}{'M':>7}{insert_label:>7}"
                  f"{'F1e':>8}{'lib_P':>8}{'lib_R':>8}")
        print(header)
        for r in rows:
            thr, mg, n_det, n_d, n_f, n_m, n_insert, f1e_p, f1e_r, f1e, lib_p, lib_r = r
            print(f"  {thr:>7.3f}{mg:>6.1f}{n_det:>7}{n_d:>7}{n_f:>7}{n_m:>7}{n_insert:>7}"
                  f"{f1e:>8.3f}{lib_p:>8.3f}{lib_r:>8.3f}")


def eval_at_merge_gap(gt_events, raw_segs, merge_gap):
    """返回 (n_det, lib_p, lib_r, lib_f1, D, F, M, I', f1e_p, f1e_r, f1e)"""
    det_events = merge_segments(sorted(raw_segs), merge_gap)
    n_gt = len(gt_events)
    if not det_events:
        return 0, 0.0, 0.0, 0.0, n_gt, 0, 0, 0, 0.0, 0.0, 0.0
    gt_scores, det_scores, detailed, standard = eval_events(gt_events, det_events)
    lib_p, lib_r = standard["precision"], standard["recall"]
    lib_f1 = 2 * lib_p * lib_r / (lib_p + lib_r) if (lib_p + lib_r) > 0 else 0.0
    f1e_p, f1e_r, f1e = compute_f1e(detailed)
    return (len(det_events), lib_p, lib_r, lib_f1,
            detailed["D"], detailed["F"], detailed["M"], detailed["I'"],
            f1e_p, f1e_r, f1e)


def main():
    ap = argparse.ArgumentParser(description="事件级评估：真实标注 vs 模型预测的目标行为片段对应关系")
    ap.add_argument("--labeled_csv", required=True)
    ap.add_argument("--model", required=True, help="模型 .pkl 路径")
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--target_label", default="抓挠")
    ap.add_argument("--dog_id_col", default="dog_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--merge_gap", type=float, nargs="+",
                     default=[1.0, 1.5, 2.0, 2.5, 3.0],
                     help="要扫描的 merge_gap 取值列表（秒，默认 1 1.5 2 2.5 3）")
    ap.add_argument("--confidence_threshold", type=float, nargs="+",
                     default=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                     help="要扫描的置信度阈值列表（默认 0.0 0.4 0.5 0.6 0.7 0.8 0.9），"
                          "这是第一阶段粗网格，用来定位大致范围")
    ap.add_argument("--scan_mode", choices=["auto", "full"], default="auto",
                     help="auto=先粗网格再在Top5附近精细化（默认，更快但可能错过全局最优）；"
                          "full=直接在完整范围内做一次精细网格（更慢但保证覆盖到全局最优，"
                          "范围/步长由 --full_thr_* / --full_gap_* 指定）")
    ap.add_argument("--refine", dest="refine", action="store_true", default=True,
                     help="[auto模式] 是否在粗网格Top5范围附近自动做精细网格搜索（默认开启）")
    ap.add_argument("--no_refine", dest="refine", action="store_false",
                     help="[auto模式] 关闭精细网格搜索，只用粗网格结果")
    ap.add_argument("--refine_thr_step", type=float, default=0.01,
                     help="[auto模式] 精细网格的置信度阈值步长（默认0.01）")
    ap.add_argument("--refine_gap_step", type=float, default=0.1,
                     help="[auto模式] 精细网格的merge_gap步长（默认0.1s）")
    ap.add_argument("--refine_thr_pad", type=float, default=0.05,
                     help="[auto模式] 精细网格阈值范围在粗网格Top5边界基础上向外扩展的幅度（默认0.05）")
    ap.add_argument("--refine_gap_pad", type=float, default=0.5,
                     help="[auto模式] 精细网格gap范围在粗网格Top5边界基础上向外扩展的幅度（默认0.5s）")
    ap.add_argument("--full_thr_start", type=float, default=0.0)
    ap.add_argument("--full_thr_stop", type=float, default=1.0)
    ap.add_argument("--full_thr_step", type=float, default=0.01)
    ap.add_argument("--full_gap_start", type=float, default=0.5)
    ap.add_argument("--full_gap_stop", type=float, default=5.0)
    ap.add_argument("--full_gap_step", type=float, default=0.1)
    ap.add_argument("--json_dir", default="",
                     help="Label Studio project-*.json 所在目录（可选）。传了的话，"
                          "被合并事件组/逐条对应表会顺带打印每个 dog_id 对应的 project "
                          "文件和 video/csv 链接，不用再单独跑 find_task_project.py")
    ap.add_argument("--log_file", default="",
                     help="可选：把完整输出（含逐条事件对应表）同时保存到这个文件，"
                          "方便去 Label Studio 复查时对照")
    args = ap.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_f)
        print(f"[日志] 输出同时保存到: {args.log_file}\n")

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
    print(f"[模型] 类别={classes}  窗口={window_s}s  步长={stride_s}s")

    df = pd.read_csv(args.labeled_csv)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    acc_cols = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    if acc_cols is None or gyro_cols is None:
        print(f"[错误] 找不到 acc/gyro 列: {list(df.columns)}")
        return

    gt_events_all = []             # 拼接所有狗的真实事件（带偏移，供 eval_events 用）
    gt_events_meta = []            # [(dog_id, local_start_s, local_end_s, global_start, global_end), ...]
    target_window_confs_all = []   # 所有 argmax==target_label 窗口的置信度（不带偏移，纯统计用）
    dog_window_records = []        # [(pred_labels, pred_confs, starts, offset), ...] 每条狗，供后续按阈值重建片段
    y_true_windows = []            # 全部窗口的真实标签（多数投票），跟 y_pred_windows 一一对应
    y_pred_windows = []            # 全部窗口的模型预测标签（argmax，不做置信度过滤）

    dog_ids = sorted(df[args.dog_id_col].unique())
    print(f"共 {len(dog_ids)} 条狗，逐条推理中...")
    for i, dog_id in enumerate(dog_ids):
        sub = df[df[args.dog_id_col] == dog_id].reset_index(drop=True)
        labels = sub[args.label_col].values
        offset = i * DOG_TIME_OFFSET

        gt_segs = find_contiguous_segments(labels)
        for start, end, lbl in gt_segs:
            if lbl == args.target_label:
                local_start, local_end = start / args.hz, end / args.hz
                gt_events_all.append((local_start + offset, local_end + offset))
                gt_events_meta.append((dog_id, local_start, local_end,
                                       local_start + offset, local_end + offset))

        data6 = np.concatenate(
            [sub[acc_cols].values, sub[gyro_cols].values], axis=1
        ).astype(np.float32)
        pred_labels, pred_confs, starts = predict_dog(data6, model, classes, window_size, stride, args.hz)
        dog_window_records.append((pred_labels, pred_confs, starts, offset))
        target_window_confs_all.extend(
            c for lbl, c in zip(pred_labels, pred_confs) if lbl == args.target_label
        )
        y_true_windows.extend(window_majority_labels(labels, starts, window_size))
        y_pred_windows.extend(pred_labels)

    # gt_events_all 和 gt_events_meta 必须按同一顺序排序（wardmetrics 要求输入按时间排好序，
    # 且返回的 gt_scores 顺序与输入顺序一一对应，两个列表如果各自排序会错位）
    order = sorted(range(len(gt_events_all)), key=lambda i: gt_events_all[i][0])
    gt_events_all = [gt_events_all[i] for i in order]
    gt_events_meta = [gt_events_meta[i] for i in order]
    target_window_confs_all = np.array(target_window_confs_all)
    print(f"\n真实 '{args.target_label}' 事件数: {len(gt_events_all)}")
    print(f"模型预测为 '{args.target_label}' 的窗口总数（未按置信度过滤）: {len(target_window_confs_all)}")

    if not gt_events_all:
        print("[警告] 没有真实事件，无法评估")
        return

    # ── 窗口级（逐帧）分类报告：不合并成事件，就是标准的多分类precision/recall/f1 ──
    print(f"\n{'='*78}")
    print("  窗口级分类报告（逐帧/逐窗口，未合并成事件，argmax预测，不做置信度过滤）")
    print(f"{'='*78}")
    labels_present = sorted(set(y_true_windows) | set(y_pred_windows))
    print(classification_report(y_true_windows, y_pred_windows, labels=labels_present,
                                zero_division=0))

    # ── 窗口级置信度分布 ──────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print(f"  窗口级置信度分布（预测为 '{args.target_label}' 的窗口，按阈值累计保留比例）")
    print(f"{'='*78}")
    print(f"  {'thr>=':>8}{'n_keep':>12}{'pct':>10}")
    n_total_windows = len(target_window_confs_all)
    for thr in args.confidence_threshold:
        n_keep = int((target_window_confs_all >= thr).sum()) if n_total_windows else 0
        pct = n_keep / n_total_windows * 100 if n_total_windows else 0.0
        print(f"  {thr:>8.2f}{n_keep:>12}{pct:>9.1f}%")

    if args.scan_mode == "full":
        # ── 全局精细网格搜索：一次性覆盖完整范围，不依赖粗网格定位，保证不错过全局最优 ──
        n_thr = int(round((args.full_thr_stop - args.full_thr_start) / args.full_thr_step)) + 1
        n_mg = int(round((args.full_gap_stop - args.full_gap_start) / args.full_gap_step)) + 1
        full_thr = [round(args.full_thr_start + i * args.full_thr_step, 4) for i in range(n_thr)]
        full_mg = [round(args.full_gap_start + i * args.full_gap_step, 4) for i in range(n_mg)]
        print(f"\n[全局网格] thr∈[{args.full_thr_start},{args.full_thr_stop}]步长{args.full_thr_step} "
              f"× gap∈[{args.full_gap_start},{args.full_gap_stop}]步长{args.full_gap_step}"
              f"，共 {len(full_thr)}×{len(full_mg)}={len(full_thr)*len(full_mg)} 组合")
        full_results = run_grid(full_thr, full_mg, dog_window_records, gt_events_all,
                                args.target_label, window_size, args.hz, desc="全局网格搜索")
        results_sorted = sorted(full_results, key=lambda x: -x[9])
        print_grid_table(results_sorted,
                         f"全局网格搜索 Top 20（{len(full_results)} 组合中选出）", top_n=20)
    else:
        # ── 第一阶段：粗网格搜索 ────────────────────────────────────────────
        coarse_results = run_grid(args.confidence_threshold, args.merge_gap, dog_window_records,
                                  gt_events_all, args.target_label, window_size, args.hz,
                                  desc="粗网格搜索")
        print_grid_table(coarse_results,
                         f"粗网格搜索: 置信度阈值 × merge_gap（{len(args.confidence_threshold)}×"
                         f"{len(args.merge_gap)}={len(coarse_results)} 组合）")

        coarse_sorted = sorted(coarse_results, key=lambda x: -x[9])
        print_grid_table(coarse_sorted, "粗网格 Top 5（F1e 把碎片化F、合并M都算作错误，不像库自带precision/recall那样纵容合并）",
                         top_n=5)

        # ── 第二阶段：在粗网格 Top5 的范围附近做精细网格搜索 ───────────────
        results_sorted = coarse_sorted
        if args.refine:
            top5 = coarse_sorted[:5]
            thr_lo = max(0.0, min(r[0] for r in top5) - args.refine_thr_pad)
            thr_hi = min(1.0, max(r[0] for r in top5) + args.refine_thr_pad)
            mg_lo = max(0.0, min(r[1] for r in top5) - args.refine_gap_pad)
            mg_hi = max(r[1] for r in top5) + args.refine_gap_pad

            n_thr = int(round((thr_hi - thr_lo) / args.refine_thr_step)) + 1
            n_mg = int(round((mg_hi - mg_lo) / args.refine_gap_step)) + 1
            fine_thr = [round(thr_lo + i * args.refine_thr_step, 4) for i in range(n_thr)]
            fine_mg = [round(mg_lo + i * args.refine_gap_step, 4) for i in range(n_mg)]

            fine_results = run_grid(fine_thr, fine_mg, dog_window_records,
                                    gt_events_all, args.target_label, window_size, args.hz,
                                    desc="精细网格搜索")
            fine_sorted = sorted(fine_results, key=lambda x: -x[9])
            print_grid_table(
                fine_sorted,
                f"精细网格搜索（基于粗网格Top5范围扩展）: thr∈[{thr_lo:.2f},{thr_hi:.2f}]"
                f"步长{args.refine_thr_step} × gap∈[{mg_lo:.1f},{mg_hi:.1f}]步长{args.refine_gap_step}"
                f"（{len(fine_thr)}×{len(fine_mg)}={len(fine_results)} 组合），仅显示 Top 10",
                top_n=10)
            results_sorted = fine_sorted

    best = results_sorted[0]
    print(f"""
{'='*90}
  推荐配置: confidence_threshold={best[0]:.2f}  merge_gap={best[1]:.1f}s  (F1e={best[9]:.3f})
  对应到实际推理命令：
    python src/infer_csv_scratch.py ... --confidence_threshold {best[0]:.2f} --merge_gap {best[1]:.1f}
""")

    # ── 推荐配置下，重新算一遍拿到完整明细（C/D/F/FM/M/I' 等），供下面拆解和逐条表复用 ──
    best_thr, best_mg = best[0], best[1]
    raw_segs = []
    for pred_labels, pred_confs, starts, offset in dog_window_records:
        segs = pred_windows_to_segments(pred_labels, pred_confs, starts, window_size,
                                        args.hz, args.target_label, best_thr)
        raw_segs.extend((s + offset, e + offset) for s, e in segs)
    raw_segs = sorted(raw_segs)
    det_events_best = merge_segments(raw_segs, best_mg)
    gt_scores_best, det_scores_best, detailed_best, standard_best = eval_events(gt_events_all, det_events_best)

    project_lookup = build_project_lookup(args.json_dir) if args.json_dir else None
    if args.json_dir and not project_lookup:
        print(f"[警告] {args.json_dir} 下没找到 project-*.json，跳过 project 反查")

    # ── 详细拆解：真实事件总数 = C(精确匹配) + D(漏检) + F(碎片) + FM(碎片且合并) + M(合并) ──
    n_gt = len(gt_events_all)
    c_count = detailed_best["C"]
    d_count = detailed_best["D"]
    f_count = detailed_best["F"]
    fm_count = detailed_best["FM"]
    m_count = detailed_best["M"]
    insert_count = detailed_best["I'"]
    print(f"{'='*90}")
    print(f"  推荐配置详细拆解（真实 '{args.target_label}' 事件共 {n_gt} 个，预测事件 {len(det_events_best)} 个）")
    print(f"{'='*90}")
    print(f"  ✅ 精确匹配 C  = {c_count:>3}  个真实事件被干净利落地一对一识别对了")
    print(f"  ❌ 漏检     D  = {d_count:>3}  个真实事件完全没被检测到")
    print(f"  🔀 碎片     F  = {f_count:>3}  个真实事件被切成了多段预测")
    if fm_count:
        print(f"  🔀🔗 碎片且合并 FM = {fm_count:>3}  个真实事件既被切碎又被合并（复合错误）")
    print(f"  🔗 合并     M  = {m_count:>3}  个真实事件跟旁边事件被粘到了同一个预测片段里")
    print(f"  👻 纯误报   I' = {insert_count:>3}  个预测事件压根没有对应的真实事件")
    print(f"  验算: C+D+F+FM+M = {c_count + d_count + f_count + fm_count + m_count}"
          f"  （应等于真实事件总数 {n_gt}）")

    # ── 全部真实事件逐条对应表：每一条的类别 + 匹配到的预测区间 + project信息 ──
    print(f"\n{'='*90}")
    print(f"  全部 {n_gt} 个真实事件逐条对应表（供逐条去 Label Studio 复查）")
    print(f"{'='*90}")
    print("  图例: C=精确匹配 D=漏检 F=碎片化 M=合并 FM=碎片且合并")
    # gt_events_meta 按全局时间排序，同一条狗的事件天然连续排在一起，
    # 所以只需检测 dog_id 变化就能正确分块（不需要重新分组）
    last_dog_id = None
    for (dog_id, ls, le, gs, ge), score in zip(gt_events_meta, gt_scores_best):
        if dog_id != last_dog_id:
            print(f"\n  {'-'*86}")
            print(f"  dog_id={dog_id}")
            if project_lookup is not None:
                print_project_info(dog_id, project_lookup)
            last_dog_id = dog_id
        overlaps = [(ds - (gs - ls), de - (ge - le)) for ds, de in det_events_best if ds < ge and de > gs]
        overlap_str = ", ".join(f"{ds:.2f}s-{de:.2f}s" for ds, de in overlaps) if overlaps else "(none)"
        print(f"    [{score:>3}]  real:{ls:>8.2f}s-{le:>8.2f}s   pred:{overlap_str}")

    # ── 被合并的真实事件组（旧有小结，方便快速定位问题最集中的几组）──
    merge_groups = find_merge_groups(gt_events_meta, det_events_best)

    print(f"\n{'='*90}")
    print(f"  推荐配置下被合并的真实事件组（共 {len(merge_groups)} 组，供人工核实）")
    print(f"{'='*90}")
    if not merge_groups:
        print("  没有发现被合并的事件组")
    else:
        for gi, group in enumerate(merge_groups, 1):
            dog_id = group[0][0]
            print(f"  组{gi}  dog_id={dog_id}  共{len(group)}个真实事件被合并:")
            if project_lookup is not None:
                print_project_info(dog_id, project_lookup)
            for j, (dog_id, ls, le) in enumerate(group):
                gap_str = ""
                if j > 0:
                    gap = ls - group[j - 1][2]
                    gap_str = f"    (距上一事件间隔 {gap:.2f}s)"
                print(f"      {ls:>8.2f}s → {le:>8.2f}s{gap_str}")
    print(f"""
  上面的秒数是该条狗在合并CSV里的行序号/hz（从这条狗的第一行开始算），
  可以据此去对应的原始录制/Label Studio标注里定位具体时间点核实：
  - 如果间隔很短（<1秒）且动作听起来像是连续的一次抓挠中间偶尔停顿，
    大概率是标注时被切成了多段，属于标注粒度问题，不是模型的错。
  - 如果间隔有一两秒甚至更长，更像是两次独立的抓挠，说明合并确实是
    模型/参数层面的问题，可以考虑要不要接受这种程度的合并（次数统计
    会因此偏少），或者未来标注时尽量避免把间隔很短的重复动作拆开标。
""")

    print(f"""
{'='*90}
  怎么解读：
  - 窗口级分布表：如果预测为目标行为的窗口大量集中在低置信度区间（比如
    阈值0.5时保留比例已经腰斩），说明模型对这个类别整体不够自信，提高
    阈值虽然能过滤掉一部分误报，但可能连带砍掉不少真实检测。
  - 网格表里的 lib_P/lib_R 是 ward-metrics 库自带的事件级 precision/
    recall，⚠️ 它把"合并"(M)当命中处理，不算错误，数值会偏乐观，仅供
    参考；真正用来挑参数的是 F1e（论文定义，把D/F/M都算错误）。
  - "推荐配置"是网格里 F1e 最高的一组，不代表绝对最优（毕竟只在你当前
    这批标注数据上评估），换新数据/重训模型后建议重新跑一遍这个脚本，
    不要一直沿用旧的推荐值。
  - 如果 Top5 里 F1e 普遍不高（比如都低于0.5），说明问题的瓶颈不在
    confidence_threshold/merge_gap 这两个后处理参数上，而是模型本身对
    这个类别的区分度不够，需要回到标注数据质量和数量上想办法。
{'='*90}
""")


if __name__ == "__main__":
    main()
