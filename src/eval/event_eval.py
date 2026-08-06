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


def find_contiguous_segments(labels, min_len=1, break_before=None):
    """把标签数组切成连续同标签的片段。

    break_before: 可选的布尔数组（跟 labels 等长），True 的位置强制从这里切断，
    即使前后标签相同也不合并成一个片段。

    为什么需要这个：labelstudio_to_custom.py 生成CSV时只保留标注覆盖到的行，
    两次标注之间没有被标注的时间段（哪怕是几分钟）在CSV里根本不存在任何行——
    不是"被打上了别的标签"，是"直接消失"。这样一来，如果两次独立发生的抓挠
    之间没有被标注别的行为，它们在CSV数组里会紧挨在一起，单看标签数组分不出
    这是"一次连续事件"还是"两次相隔很远、只是中间没被标注任何东西的独立事件"。
    调用方可以传入基于真实时间戳算出的 break_before（时间跳变超过阈值的位置），
    强制在这些位置切断，避免虚假合并成一个事件。"""
    segs = []
    n = len(labels)
    i = 0
    while i < n:
        j = i + 1
        while (j < n and labels[j] == labels[i]
               and not (break_before is not None and break_before[j])):
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


def _process_one_dog(dog_id, i, df, acc_cols, gyro_cols, dog_id_col, label_col,
                     model, classes, window_size, stride, hz, target_label,
                     has_timestamp=False, gap_tolerance=1.5):
    """单条狗的完整处理（真实事件提取 + 滑窗推理），供并行调用。

    has_timestamp: CSV是否带有真实时间戳列。有的话用真实时间跳变来判断两段
    同标签的行是不是真的时间连续，而不是只看数组下标是否相邻——因为CSV只保留
    了标注覆盖到的行，未标注的时间段直接消失不留痕迹，两次相隔很远的独立事件
    如果中间没有别的标注，会在数组里紧挨着，单靠标签数组分不出来。
    gap_tolerance: 相邻两行时间差超过 gap_tolerance/hz 秒就认为不是真连续
    （正常单帧间隔是1/hz，给1.5倍冗余容忍轻微的采样抖动）。"""
    sub = df[df[dog_id_col] == dog_id].reset_index(drop=True)
    labels = sub[label_col].values
    offset = i * DOG_TIME_OFFSET

    break_before = None
    if has_timestamp:
        ts = pd.to_datetime(sub["timestamp"], errors="coerce").values.astype("datetime64[ns]")
        deltas = np.empty(len(ts))
        deltas[0] = 0.0
        deltas[1:] = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
        break_before = deltas > (gap_tolerance / hz)

    gt_events, gt_meta = [], []
    gt_segs = find_contiguous_segments(labels, break_before=break_before)
    for start, end, lbl in gt_segs:
        if lbl == target_label:
            local_start, local_end = start / hz, end / hz
            gt_events.append((local_start + offset, local_end + offset))
            gt_meta.append((dog_id, local_start, local_end,
                            local_start + offset, local_end + offset))

    # break_before 强制切开的两个事件，边界值本来就是"行下标/hz"算出来的，
    # 如果两段在数组里正好是相邻行（只是因为真实时间跳变才被拆开），算出来的
    # 前一段end跟后一段start会是完全相等的浮点数——ward-metrics库自己内部有个
    # "首尾正好相接就合并"的逻辑（eval_events()里的merge_events_if_necessary），
    # 会把我们刚拆开的两个事件在它自己的统计口径里悄悄合并回去，等于白拆。
    # 加一个微小的epsilon错开边界，两个事件保留可忽略不计的确定性小间隔，
    # 不会影响任何显示精度（都是2位小数展示），但能防止被库重新合并。
    EPS = 1e-6
    for i in range(1, len(gt_events)):
        prev_end = gt_events[i - 1][1]
        cur_start, cur_end = gt_events[i]
        if cur_start <= prev_end:
            shift = (prev_end - cur_start) + EPS
            gt_events[i] = (cur_start + shift, cur_end + shift)
            m_dog, m_ls, m_le, m_gs, m_ge = gt_meta[i]
            gt_meta[i] = (m_dog, m_ls + shift, m_le + shift, m_gs + shift, m_ge + shift)

    data6 = np.concatenate(
        [sub[acc_cols].values, sub[gyro_cols].values], axis=1
    ).astype(np.float32)
    pred_labels, pred_confs, starts = predict_dog(data6, model, classes, window_size, stride, hz)
    target_confs = [c for lbl, c in zip(pred_labels, pred_confs) if lbl == target_label]
    y_true = window_majority_labels(labels, starts, window_size)

    return {
        "gt_events": gt_events, "gt_meta": gt_meta,
        "window_record": (pred_labels, pred_confs, starts, offset),
        "target_confs": target_confs,
        "y_true": y_true, "y_pred": list(pred_labels),
    }


def build_dataset(df, acc_cols, gyro_cols, dog_ids, dog_id_col, label_col,
                  model, classes, window_size, stride, hz, target_label,
                  show_progress=True, workers=-1, has_timestamp=False):
    """对全部狗跑一遍推理 + 提取真实事件，返回本次评估需要的全部中间数据。
    每次调用都会重新做完整的特征提取+模型推理，换 stride 时必须重新调用
    （跟 confidence_threshold/merge_gap 不同，那两个可以复用同一份推理结果）。

    每条狗的处理彼此独立，用 joblib 多进程并行（特征提取是纯CPU计算，
    多进程能真正利用多核，多线程会被GIL卡住基本没用）。"""
    gt_events_all = []
    gt_events_meta = []
    target_window_confs_all = []
    dog_window_records = []
    y_true_windows = []
    y_pred_windows = []

    from joblib import Parallel, delayed
    from tqdm import tqdm
    n_jobs = workers if workers > 0 else -1
    dog_iter = enumerate(dog_ids)
    if show_progress:
        dog_iter = enumerate(tqdm(dog_ids, desc="逐狗推理", unit="狗"))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_one_dog)(dog_id, i, df, acc_cols, gyro_cols, dog_id_col, label_col,
                                  model, classes, window_size, stride, hz, target_label,
                                  has_timestamp=has_timestamp)
        for i, dog_id in dog_iter
    )

    for r in results:
        gt_events_all.extend(r["gt_events"])
        gt_events_meta.extend(r["gt_meta"])
        dog_window_records.append(r["window_record"])
        target_window_confs_all.extend(r["target_confs"])
        y_true_windows.extend(r["y_true"])
        y_pred_windows.extend(r["y_pred"])

    order = sorted(range(len(gt_events_all)), key=lambda i: gt_events_all[i][0])
    gt_events_all = [gt_events_all[i] for i in order]
    gt_events_meta = [gt_events_meta[i] for i in order]
    target_window_confs_all = np.array(target_window_confs_all)

    return {
        "gt_events_all": gt_events_all,
        "gt_events_meta": gt_events_meta,
        "target_window_confs_all": target_window_confs_all,
        "dog_window_records": dog_window_records,
        "y_true_windows": y_true_windows,
        "y_pred_windows": y_pred_windows,
    }


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
    """按 dog_id（如 task496_imu1）反查并打印所属 project 文件 + task_id + video/csv 链接。
    project 下通常有很多个待标注文件，光有 project 编号定位不到具体是哪一条，
    必须带上 task_id（Label Studio Data Manager 里按这个搜索/筛选）。"""
    tid = extract_task_id(dog_id)
    if tid is None or tid not in project_lookup:
        print(f"      [未找到 {dog_id} 对应的 project 信息]")
        return
    fname, project_no, task = project_lookup[tid]
    inner_id = task.get("inner_id")
    inner_str = f"  inner_id(项目内序号)={inner_id}" if inner_id is not None else ""
    print(f"      project文件: {fname}  project编号: {project_no or '未知'}  task_id={tid}{inner_str}")
    print(f"      → Label Studio 网页版: 进入 project {project_no}，Data Manager 里按 task_id={tid} 搜索/筛选定位")
    data = task.get("data", {})
    for k in ("csv", "csv1", "csv2", "video1", "video2", "cam1", "cam2"):
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


def local_sec_to_ts_str(dog_id, local_sec, hz, dog_ts_map):
    """把某条狗内部的相对秒数换算成 CSV 里的绝对时间戳字符串，供人工去原始CSV/视频里核对。
    没有 timestamp 列（旧数据）或越界时返回 None。"""
    ts_series = dog_ts_map.get(dog_id)
    if ts_series is None or len(ts_series) == 0:
        return None
    idx = int(round(local_sec * hz))
    idx = max(0, min(idx, len(ts_series) - 1))
    ts = ts_series.iloc[idx]
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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


def pred_windows_to_segments(pred_labels, pred_confs, starts, window_size, hz, target_label,
                             conf_threshold, stride=None, label_mode="majority"):
    """把逐窗口预测里 label==target_label 且 conf>=conf_threshold 的窗口，
    合并成连续的原始片段（未做 merge_gap 合并）。

    label_mode 必须跟训练模型时用的一致，否则会错误地重建事件边界：
      "majority"（默认）：一个正例窗口代表"这2秒(window_size)都是目标行为"，
        跟窗口起点到窗口终点整段对应，这是训练时多数投票的语义。
      "center"：一个正例窗口只代表"窗口正中心这一瞬间是目标行为"，只用窗口
        中心点前后半个步长(stride/2)去覆盖时间轴，不铺满整个窗口——这样才能
        真正发挥中心点标注法带来的边界精度收益，否则等于白训练。
    """
    raw_segs = []
    in_seg = False
    seg_start = None
    prev_end = None
    half_pad = (stride / 2 / hz) if (label_mode == "center" and stride) else 0.0
    for lbl, conf, s in zip(pred_labels, pred_confs, starts):
        hit = (lbl == target_label) and (conf >= conf_threshold)
        if label_mode == "center":
            center_sec = (s + window_size / 2) / hz
            start_sec = center_sec - half_pad
            end_sec   = center_sec + half_pad
        else:
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


def _flatten_records(dog_window_records, target_label):
    """把 dog_window_records（687条狗，每条是 python list-of-str 的标签+置信度+起点）
    转成几个大的 numpy 数组：starts/confs/hit 拼接成三个扁平数组，bounds 记录
    每条狗在扁平数组里的 [start_idx, end_idx) 区间，offsets 是每条狗的时间偏移。

    这是为了给 run_grid 的多进程并行用。之前直接把 dog_window_records（687个
    python list，每个list里是几千个字符串）传给 joblib 的多进程worker，每次
    派发任务都要重新pickle这一大坨python对象，101个阈值重复101次，开销比省下
    的计算时间还大。改成先转成大的numpy数组，joblib/loky 对超过一定大小的
    numpy数组会自动用内存映射共享（不用每次都完整拷贝+序列化），才能真正
    发挥多进程的加速——用多线程规避这个问题是错的，纯python循环受GIL限制，
    线程之间根本不会并行，等于白跑。"""
    starts_parts, confs_parts, hit_parts = [], [], []
    bounds = [0]
    offsets = np.empty(len(dog_window_records), dtype=np.float64)
    for i, (pred_labels, pred_confs, starts, offset) in enumerate(dog_window_records):
        starts_parts.append(np.asarray(starts, dtype=np.int64))
        confs_parts.append(np.asarray(pred_confs, dtype=np.float32))
        hit_parts.append(np.asarray([lbl == target_label for lbl in pred_labels], dtype=bool))
        offsets[i] = offset
        bounds.append(bounds[-1] + len(starts))
    starts_flat = np.concatenate(starts_parts) if starts_parts else np.empty(0, dtype=np.int64)
    confs_flat = np.concatenate(confs_parts) if confs_parts else np.empty(0, dtype=np.float32)
    hit_flat = np.concatenate(hit_parts) if hit_parts else np.empty(0, dtype=bool)
    bounds = np.array(bounds, dtype=np.int64)
    return starts_flat, confs_flat, hit_flat, bounds, offsets


def _segments_from_hits(hits, starts, window_size, hz, stride, label_mode):
    """跟 pred_windows_to_segments 逻辑一样，只是输入是预先算好的布尔命中数组
    （hit标签 & conf>=阈值 已经在外面向量化算完），避免每个窗口都重复做
    字符串比较+置信度比较。"""
    raw_segs = []
    in_seg = False
    seg_start = None
    prev_end = None
    half_pad = (stride / 2 / hz) if (label_mode == "center" and stride) else 0.0
    for idx in range(len(hits)):
        s = starts[idx]
        if label_mode == "center":
            center_sec = (s + window_size / 2) / hz
            start_sec = center_sec - half_pad
            end_sec = center_sec + half_pad
        else:
            start_sec = s / hz
            end_sec = (s + window_size) / hz
        if hits[idx]:
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


def _eval_one_threshold_flat(thr, mg_values, starts_flat, confs_flat, hit_flat, bounds, offsets,
                             gt_events_all, window_size, hz, stride, label_mode):
    """run_grid 并行版用：接收扁平numpy数组而不是 dog_window_records，
    每条狗切片按阈值算命中区间、重建片段，再扫一遍 merge_gap。"""
    hits_over_thr = hit_flat & (confs_flat >= thr)
    raw_segs = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if s == e:
            continue
        segs = _segments_from_hits(hits_over_thr[s:e], starts_flat[s:e], window_size, hz, stride, label_mode)
        raw_segs.extend((ss + offsets[i], ee + offsets[i]) for ss, ee in segs)
    raw_segs.sort()
    rows = []
    for mg in mg_values:
        (n_det, lib_p, lib_r, lib_f1, n_d, n_f, n_m, n_insert,
         f1e_p, f1e_r, f1e) = eval_at_merge_gap(gt_events_all, raw_segs, mg)
        rows.append((thr, mg, n_det, n_d, n_f, n_m, n_insert, f1e_p, f1e_r, f1e, lib_p, lib_r))
    return rows


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


def _eval_one_threshold(thr, mg_values, dog_window_records, gt_events_all,
                        target_label, window_size, hz, stride, label_mode):
    """单个置信度阈值下，扫完所有 merge_gap 取值，返回这一批结果行。
    按阈值拆成独立函数是为了给 run_grid 的多进程并行用（每个阈值互不依赖，
    重建预测片段+跑 wardmetrics 都是纯CPU计算，核多、狗多、事件多时并行收益明显）。"""
    raw_segs = []
    for pred_labels, pred_confs, starts, offset in dog_window_records:
        segs = pred_windows_to_segments(pred_labels, pred_confs, starts, window_size,
                                        hz, target_label, thr, stride, label_mode)
        raw_segs.extend((s + offset, e + offset) for s, e in segs)
    raw_segs = sorted(raw_segs)
    rows = []
    for mg in mg_values:
        (n_det, lib_p, lib_r, lib_f1, n_d, n_f, n_m, n_insert,
         f1e_p, f1e_r, f1e) = eval_at_merge_gap(gt_events_all, raw_segs, mg)
        rows.append((thr, mg, n_det, n_d, n_f, n_m, n_insert, f1e_p, f1e_r, f1e, lib_p, lib_r))
    return rows


def run_grid(thr_values, mg_values, dog_window_records, gt_events_all,
             target_label, window_size, hz, stride=None, label_mode="majority",
             desc="网格搜索", show_progress=True, workers=1):
    """跑一遍 (置信度阈值 x merge_gap) 网格，返回结果列表
    [(thr, mg, n_det, n_d, n_f, n_m, n_insert, f1e_p, f1e_r, f1e, lib_p, lib_r), ...]
    耗时主要在按阈值重建预测片段这一层（外层循环），gap 合并很快。

    workers != 1 时用 joblib 的 loky（多进程）按阈值并行——纯Python循环受GIL
    限制，多线程根本无法并行、白搭；真正要加速必须用多进程绕开GIL。多进程的
    代价是每次派发任务都要序列化数据传给子进程：如果直接传 dog_window_records
    （687个python list，每个list几千个字符串），101个阈值重复序列化101次，
    这个开销比省下的计算时间还大。所以先用 _flatten_records() 把数据转成几个
    大的numpy数组——joblib/loky 对超过阈值大小的numpy数组会自动用内存映射
    共享（写一次磁盘文件，各子进程直接映射读取，不需要每次完整拷贝+反序列化），
    这样才能真正拿到多进程的加速。"""
    if workers and workers != 1:
        from joblib import Parallel, delayed
        from tqdm import tqdm
        starts_flat, confs_flat, hit_flat, bounds, offsets = _flatten_records(
            dog_window_records, target_label)
        n_jobs = workers if workers > 0 else -1
        it = thr_values
        if show_progress:
            it = tqdm(thr_values, desc=desc, unit="thr")
        nested = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_eval_one_threshold_flat)(thr, mg_values, starts_flat, confs_flat, hit_flat,
                                              bounds, offsets, gt_events_all,
                                              window_size, hz, stride, label_mode)
            for thr in it
        )
        return [row for rows in nested for row in rows]

    results = []
    best_f1e_so_far = 0.0
    it = thr_values
    if show_progress:
        from tqdm import tqdm
        it = tqdm(thr_values, desc=desc, unit="thr")
    for thr in it:
        rows = _eval_one_threshold(thr, mg_values, dog_window_records, gt_events_all,
                                   target_label, window_size, hz, stride, label_mode)
        results.extend(rows)
        if rows:
            best_f1e_so_far = max(best_f1e_so_far, max(r[9] for r in rows))
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
    # 传副本进去：ward-metrics 的 eval_events() 内部会用 del 原地修改传入的列表
    # （合并首尾正好相接的事件），如果直接传引用，我们自己这份 gt_events_all
    # 会被library悄悄改短，后面再用 len(gt_events_all) 就会跟之前打印的对不上
    gt_scores, det_scores, detailed, standard = eval_events(list(gt_events), list(det_events))
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
    ap.add_argument("--infer_stride_s", type=float, default=None,
                     help="推理步长（秒），跟训练步长解耦，不传则沿用模型meta里的训练步长。"
                          "步长越小，事件边界定位越精细，但要重新做一遍完整推理，成本按比例上升")
    ap.add_argument("--stride_compare", type=float, nargs="+", default=None,
                     help="步长对比模式：给出多个候选步长（秒），每个都重新推理一遍并各自做"
                          "网格搜索，最后对比哪个步长的最优F1e最高。例: --stride_compare 1.0 0.5 0.25 0.0625"
                          "（0.0625s=16Hz下1个采样点，是能做到的最细步长）。传了这个参数会跳过"
                          "正常的单步长完整分析，只输出步长对比汇总")
    ap.add_argument("--workers", type=int, default=-1,
                     help="逐狗推理的并行进程数（默认-1=用全部CPU核）。特征提取是纯CPU计算，"
                          "多进程能真正利用多核；步长越密、狗越多，并行收益越明显")
    args = ap.parse_args()

    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_f)
        print(f"[日志] 输出同时保存到: {args.log_file}\n")

    model = joblib.load(args.model)
    meta_path = args.model.replace(".pkl", ".json")
    classes, window_s, stride_s, label_mode = [], 2.0, 1.0, "majority"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        classes  = meta.get("classes", [])
        window_s = float(meta.get("window_s", 2.0))
        stride_s = float(meta.get("stride_s", 1.0))
        label_mode = meta.get("label_mode", "majority")  # 旧模型没这个字段，退化为majority（兼容原有行为）
    else:
        classes = list(model.classes_) if hasattr(model, "classes_") else []
    if args.target_label not in classes:
        print(f"[错误] 模型类别中没有 '{args.target_label}': {classes}")
        return
    window_size = int(window_s * args.hz)
    # 推理步长可以跟训练步长解耦：--infer_stride_s 显式覆盖，不传则沿用训练meta里的stride_s
    infer_stride_s = args.infer_stride_s if args.infer_stride_s is not None else stride_s
    stride = max(1, round(infer_stride_s * args.hz))
    print(f"[模型] label_mode={label_mode}  类别={classes}  窗口={window_s}s  训练步长={stride_s}s  "
          f"推理步长={infer_stride_s}s（{stride}个采样点）")

    df = pd.read_csv(args.labeled_csv)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    acc_cols = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    if acc_cols is None or gyro_cols is None:
        print(f"[错误] 找不到 acc/gyro 列: {list(df.columns)}")
        return
    dog_ids = sorted(df[args.dog_id_col].unique())

    # ── 可选的绝对时间戳列（labelstudio_to_custom.py 新版才有，旧数据没有则优雅降级）──
    has_ts = "timestamp" in df.columns
    dog_ts_map = {}
    if has_ts:
        ts_all = pd.to_datetime(df["timestamp"], errors="coerce")
        for dog_id in dog_ids:
            dog_ts_map[dog_id] = ts_all[df[args.dog_id_col] == dog_id].reset_index(drop=True)
    else:
        print("[提示] CSV 里没有 timestamp 列（用旧版 labelstudio_to_custom.py 生成的），"
              "逐条对应表将只显示相对秒数，不显示绝对时间戳；"
              "同时无法判断两段同标签标注是否真的时间连续，可能把中间隔了很久的独立事件误合并成一个")

    # ── stride 对比模式：每个候选步长都要重新做一遍完整推理（不能复用），
    # 所以只跑网格搜索拿到每个 stride 的最优 F1e 做对比，不打印逐条明细 ──
    if args.stride_compare:
        print(f"\n[stride对比] 候选步长: {args.stride_compare}（每个都要重新推理，比较耗时）")
        stride_results = []
        for si, cand_stride_s in enumerate(args.stride_compare, 1):
            cand_stride = max(1, round(cand_stride_s * args.hz))
            print(f"\n[{si}/{len(args.stride_compare)}] 步长={cand_stride_s}s（{cand_stride}个采样点）推理中...")
            ds = build_dataset(df, acc_cols, gyro_cols, dog_ids, args.dog_id_col, args.label_col,
                               model, classes, window_size, cand_stride, args.hz, args.target_label,
                               workers=args.workers, has_timestamp=has_ts)
            if not ds["gt_events_all"]:
                print("  [警告] 没有真实事件，跳过")
                continue
            n_windows = sum(len(r[0]) for r in ds["dog_window_records"])
            grid_results = run_grid(args.confidence_threshold, args.merge_gap, ds["dog_window_records"],
                                    ds["gt_events_all"], args.target_label, window_size, args.hz,
                                    stride=cand_stride, label_mode=label_mode,
                                    desc=f"  阈值×gap网格(stride={cand_stride_s}s)",
                                    workers=args.workers)
            best = sorted(grid_results, key=lambda x: -x[9])[0]
            thr, mg, n_det, n_d, n_f, n_m, n_insert = best[0], best[1], best[2], best[3], best[4], best[5], best[6]
            f1e = best[9]
            stride_results.append((cand_stride_s, n_windows, thr, mg, n_det, n_d, n_f, n_m, n_insert, f1e))
            print(f"  该步长下最优: thr={thr:.2f} gap={mg:.1f}s  F1e={f1e:.3f}  "
                  f"(D={n_d} F={n_f} M={n_m} I'={n_insert})")

        print(f"\n{'='*90}")
        print(f"  步长对比汇总（每个步长各自网格搜索后的最优表现）")
        print(f"{'='*90}")
        insert_label = "I'"
        print(f"  {'stride_s':>10}{'n_windows':>12}{'best_thr':>10}{'best_gap':>10}{'n_pred':>8}"
              f"{'D':>5}{'F':>5}{'M':>5}{insert_label:>5}{'F1e':>8}")
        for r in stride_results:
            cand_stride_s, n_windows, thr, mg, n_det, n_d, n_f, n_m, n_insert, f1e = r
            print(f"  {cand_stride_s:>10.4f}{n_windows:>12}{thr:>10.2f}{mg:>10.1f}{n_det:>8}"
                  f"{n_d:>5}{n_f:>5}{n_m:>5}{n_insert:>5}{f1e:>8.3f}")

        if stride_results:
            best_stride = max(stride_results, key=lambda r: r[-1])
            print(f"\n  推荐步长: {best_stride[0]}s（F1e={best_stride[-1]:.3f}）")
            print(f"  对应命令: --infer_stride_s {best_stride[0]} --confidence_threshold {best_stride[2]:.2f}"
                  f" --merge_gap {best_stride[3]:.1f}")
        print(f"""
{'='*90}
  怎么解读：
  - 这里每个步长都只跑了一次网格搜索找最优 (thr, gap)，没有像单步长模式
    那样再做二次精细网格，是为了控制总耗时；如果某个步长看起来明显更好，
    建议单独用 --infer_stride_s 那个值重跑一次完整流程（不加 --stride_compare），
    拿到更精细的推荐参数和逐条对应表。
  - n_windows 是这个步长下总共产生的窗口数，直接反映计算量差异，步长越小
    这个数字涨得越快，可以对照实际跑的时间感受一下代价。
  - 如果 F1e 随步长变小并没有实质提升，说明当前瓶颈不在步长，不需要为了
    这点收益长期承受更高的计算成本。
{'='*90}
""")
        return

    # ── 单一步长模式（默认，跟原来一样）──────────────────────────────────
    ds = build_dataset(df, acc_cols, gyro_cols, dog_ids, args.dog_id_col, args.label_col,
                       model, classes, window_size, stride, args.hz, args.target_label,
                       workers=args.workers, has_timestamp=has_ts)
    gt_events_all = ds["gt_events_all"]
    gt_events_meta = ds["gt_events_meta"]
    target_window_confs_all = ds["target_window_confs_all"]
    dog_window_records = ds["dog_window_records"]
    y_true_windows = ds["y_true_windows"]
    y_pred_windows = ds["y_pred_windows"]

    print(f"\n真实 '{args.target_label}' 事件数: {len(gt_events_all)}")
    print(f"模型预测为 '{args.target_label}' 的窗口总数（未按置信度过滤）: {len(target_window_confs_all)}")

    if not gt_events_all:
        print("[警告] 没有真实事件，无法评估")
        return

    # ── 窗口级（逐帧）分类报告：不合并成事件，就是标准的多分类precision/recall/f1 ──
    # 只保留真实标签属于模型已知类别的窗口：标注CSV可能包含模型没训练过的类别（比如
    # 啃身体/奔跑/舔身体等），拿这些去跟模型预测比较没有意义（模型压根不认识这个类，
    # 结果必然是0，只会拉低总体指标、把报告搅乱），所以先过滤掉。
    n_before = len(y_true_windows)
    filtered = [(t, p) for t, p in zip(y_true_windows, y_pred_windows) if t in classes]
    n_skipped = n_before - len(filtered)
    y_true_eval = [t for t, p in filtered]
    y_pred_eval = [p for t, p in filtered]

    print(f"\n{'='*78}")
    print("  窗口级分类报告（逐帧/逐窗口，未合并成事件，argmax预测，不做置信度过滤）")
    print(f"{'='*78}")
    if n_skipped:
        print(f"  [说明] 跳过 {n_skipped} 个真实标签不在模型类别 {classes} 内的窗口"
              f"（比如啃身体/奔跑等模型没训练过的类别，比较没有意义）")
    print(classification_report(y_true_eval, y_pred_eval, labels=classes, zero_division=0))
    print("  指标含义：")
    print("    precision  预测为该类的窗口里，真的是该类的比例（预测准不准）")
    print("    recall     真实为该类的窗口里，被正确预测出来的比例（有没有漏）")
    print("    f1-score   precision 和 recall 的调和平均，兼顾两者")
    print("    support    该类别在真实标签里的窗口数量")
    print("    accuracy   全部窗口里预测完全正确的比例")
    print("    macro avg    各类别指标的简单平均，不考虑样本量多少（小类别权重被拉高）")
    print("    weighted avg 按各类别样本量加权平均，更能反映整体实际表现")

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
                                args.target_label, window_size, args.hz,
                                stride=stride, label_mode=label_mode, desc="全局网格搜索",
                                workers=args.workers)
        results_sorted = sorted(full_results, key=lambda x: -x[9])
        print_grid_table(results_sorted,
                         f"全局网格搜索 Top 20（{len(full_results)} 组合中选出）", top_n=20)
    else:
        # ── 第一阶段：粗网格搜索 ────────────────────────────────────────────
        coarse_results = run_grid(args.confidence_threshold, args.merge_gap, dog_window_records,
                                  gt_events_all, args.target_label, window_size, args.hz,
                                  stride=stride, label_mode=label_mode, desc="粗网格搜索",
                                  workers=args.workers)
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
                                    stride=stride, label_mode=label_mode, desc="精细网格搜索",
                                    workers=args.workers)
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
                                        args.hz, args.target_label, best_thr, stride, label_mode)
        raw_segs.extend((s + offset, e + offset) for s, e in segs)
    raw_segs = sorted(raw_segs)
    det_events_best = merge_segments(raw_segs, best_mg)
    gt_scores_best, det_scores_best, detailed_best, standard_best = eval_events(list(gt_events_all), list(det_events_best))

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

    # ── F1e 计算过程明细：TP=C, FN=D+F+FM+M, FP=M'+FM'+F'+I' ──────────────
    mp_count = detailed_best["M'"]
    fmp_count = detailed_best["FM'"]
    fp_count_label = detailed_best["F'"]
    tp = c_count
    fn = d_count + f_count + fm_count + m_count
    fp = mp_count + fmp_count + fp_count_label + insert_count
    f1e_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1e_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1e_val = 2 * f1e_p * f1e_r / (f1e_p + f1e_r) if (f1e_p + f1e_r) > 0 else 0.0
    print(f"\n  F1e 计算过程：")
    print(f"    TP = C = {tp}")
    print(f"    FN = D+F+FM+M = {d_count}+{f_count}+{fm_count}+{m_count} = {fn}")
    print(f"    FP = M'+FM'+F'+I' = {mp_count}+{fmp_count}+{fp_count_label}+{insert_count} = {fp}"
          f"  （M'/FM'/F' 是造成合并/碎片化的那些预测片段本身，I'是纯误报）")
    print(f"    precision_e = TP/(TP+FP) = {tp}/{tp+fp} = {f1e_p:.3f}")
    print(f"    recall_e    = TP/(TP+FN) = {tp}/{tp+fn} = {f1e_r:.3f}")
    print(f"    F1e = 2×P×R/(P+R) = {f1e_val:.3f}")

    # ── 全部真实事件 + 纯误报逐条对应表：每一条的类别 + 匹配区间 + project信息 ──
    SCORE_INFO = {
        "C":  ("✅", "精确匹配"),
        "D":  ("❌", "漏检"),
        "F":  ("🔀", "碎片化"),
        "M":  ("🔗", "合并"),
        "FM": ("🔀🔗", "碎片且合并"),
        "I'": ("👻", "纯误报"),
    }

    def score_tag(score):
        emoji, name = SCORE_INFO.get(score, ("", score))
        return f"{emoji} {score}={name}" if emoji else f"{score}"

    # 把纯误报（I'）也定位到所属 dog_id + 局部时间，混入同一份逐条表里一起按时间/dog展示
    insertion_rows = []  # (dog_id, local_start, local_end, global_start)
    for (ds, de), dscore in zip(det_events_best, det_scores_best):
        if dscore == "I'":
            dog_idx = int(ds // DOG_TIME_OFFSET)
            d_id = dog_ids[dog_idx] if 0 <= dog_idx < len(dog_ids) else f"(未知,offset={ds:.0f})"
            offset = dog_idx * DOG_TIME_OFFSET
            insertion_rows.append((d_id, ds - offset, de - offset, ds))

    all_rows = [(dog_id, ls, gs, "event", score, le, ge)
                for (dog_id, ls, le, gs, ge), score in zip(gt_events_meta, gt_scores_best)] + \
               [(dog_id, ls, gs, "insertion", "I'", le, None)
                for dog_id, ls, le, gs in insertion_rows]
    all_rows.sort(key=lambda r: r[2])  # 按全局时间排（天然按 dog 分块）

    print(f"\n{'='*90}")
    print(f"  全部 {n_gt} 个真实事件 + {len(insertion_rows)} 个纯误报逐条对应表（供逐条去 Label Studio 复查）")
    print(f"{'='*90}")
    print("  图例: " + "  ".join(f"{v[0]} {k}={v[1]}" for k, v in SCORE_INFO.items()))
    last_dog_id = None
    for dog_id, ls, gs, kind, score, le, ge in all_rows:
        if dog_id != last_dog_id:
            print(f"\n  {'-'*86}")
            print(f"  dog_id={dog_id}")
            if project_lookup is not None:
                print_project_info(dog_id, project_lookup)
            last_dog_id = dog_id
        if kind == "event":
            overlaps = [(ds2 - (gs - ls), de2 - (ge - le)) for ds2, de2 in det_events_best if ds2 < ge and de2 > gs]
            overlap_str = ", ".join(f"{ds2:.2f}s-{de2:.2f}s" for ds2, de2 in overlaps) if overlaps else "(none)"
            print(f"    [{score_tag(score)}]  real:{ls:>8.2f}s-{le:>8.2f}s   pred:{overlap_str}")
            real_ts_start = local_sec_to_ts_str(dog_id, ls, args.hz, dog_ts_map)
            real_ts_end = local_sec_to_ts_str(dog_id, le, args.hz, dog_ts_map)
            if real_ts_start is not None:
                print(f"        真实标注时间戳: {real_ts_start} → {real_ts_end}")
            for ds2, de2 in overlaps:
                pred_ts_start = local_sec_to_ts_str(dog_id, ds2, args.hz, dog_ts_map)
                pred_ts_end = local_sec_to_ts_str(dog_id, de2, args.hz, dog_ts_map)
                if pred_ts_start is not None:
                    print(f"        预测事件时间戳: {pred_ts_start} → {pred_ts_end}")
        else:
            print(f"    [{score_tag(score)}]  real:(none)              pred:{ls:.2f}s-{le:.2f}s"
                  f"  ← 模型预测但没有对应真实事件，建议核实是不是漏标")
            pred_ts_start = local_sec_to_ts_str(dog_id, ls, args.hz, dog_ts_map)
            pred_ts_end = local_sec_to_ts_str(dog_id, le, args.hz, dog_ts_map)
            if pred_ts_start is not None:
                print(f"        预测事件时间戳: {pred_ts_start} → {pred_ts_end}")

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
                ts_start = local_sec_to_ts_str(dog_id, ls, args.hz, dog_ts_map)
                ts_end = local_sec_to_ts_str(dog_id, le, args.hz, dog_ts_map)
                if ts_start is not None:
                    print(f"        绝对时间戳: {ts_start} → {ts_end}")
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
