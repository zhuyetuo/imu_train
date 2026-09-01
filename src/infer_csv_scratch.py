"""
对自采 CSV 文件做离线推理，输出抓挠识别结果。

支持单文件或整个目录批量处理。
输出每个时间窗口的预测标签，并汇总抓挠片段的起止时间和置信度。

用法:
  # 单个 CSV
  python src/infer_csv_scratch.py \
    --csv data/raw_wit/multicam_20260715_084939_cam1_imu1_resampled16hz.csv \
    --model results/processed_merged_all/16hz/ml_rf.pkl

  # 目录批量（处理所有 *imu1*.csv）
  python src/infer_csv_scratch.py \
    --csv_dir data/raw_wit/ \
    --pattern "*imu1*.csv" \
    --model results/processed_merged_all/16hz/ml_rf.pkl

  # 设备 100Hz CSV，模型 16Hz（自动降采样）
  python src/infer_csv_scratch.py \
    --csv data/raw_wit/rec_wit_20260629.csv \
    --model results/processed_merged_all/16hz/ml_rf.pkl \
    --device_hz 100 --model_hz 16
"""

import argparse
import os
import sys
import glob
import io
import urllib.request
from math import gcd

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))

from gravity_align import gravity_align, append_raw_tilt_batch
from features import extract_features

import joblib
import json


def _load_dl_model(model_path):
    """加载DL模型(.pt，src/dl/train.py训出来的state_dict)+同名.json推理
    元数据，包装成一个有predict_proba(X)接口的对象——这样main()/infer_file()
    里"加载模型→predict_proba"这条主流程不用为了兼容DL模型整个重写，
    ML(.pkl，joblib+sklearn predict_proba)和DL(.pt，torch state_dict)
    两条分支在“怎么产出probs”这一步就统一了。

    DL模型和ML模型吃的输入完全不是一回事：ML走src/ml/features.py手工
    提取统计特征，DL直接吃重力对齐+tilt拼接后的原始窗口(N,window_size,
    n_channels)——所以调用方(infer_file)要知道is_dl，对DL跳过
    extract_features()那一步，直接把窗口数组传给这里的predict_proba。

    还原模型结构需要model_name+超参cfg（跟src/dl/train.py用的是同一份
    load_model()，不重复写一份模型工厂函数）；标准化必须用训练时保存的
    ch_mean/ch_std重新做，不能用这批推理数据自己的统计量——否则输入分布
    跟训练时对不上，这是src/dl/train.py训练那次修好的NaN/loss坍缩问题
    的同一类坑（分布不匹配不会报错，只会让效果悄悄变差）。见
    src/dl/train.py里infer_meta的注释。
    """
    import torch
    import torch.nn.functional as F

    dl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl")
    if dl_dir not in sys.path:
        sys.path.insert(0, dl_dir)
    from train import load_model as _build_model  # noqa: E402

    meta_path = model_path.replace(".pt", ".json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"找不到DL模型的推理元数据: {meta_path}（需要跟{model_path}同目录同名，"
            f"src/dl/train.py训练完会自动生成，旧版训出来的.pt没有这个文件，得重新训一次）")
    with open(meta_path, encoding="utf-8") as f:
        dl_meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = dl_meta["model"]
    cfg_wrapped = {model_name: dl_meta["model_cfg"]}
    net = _build_model(model_name, dl_meta["n_channels"], dl_meta["window_size"],
                       len(dl_meta["classes"]), cfg_wrapped)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    ch_mean = np.array(dl_meta["ch_mean"], dtype=np.float32)
    ch_std  = np.array(dl_meta["ch_std"],  dtype=np.float32)
    m2m     = bool(dl_meta["m2m"])

    class _DLModelWrapper:
        classes_ = dl_meta["classes"]

        def predict_proba(self, X):
            # X: (N, window_size, n_channels) 原始窗口，重力对齐+tilt拼接后、
            # 还没标准化——用训练时的ch_mean/ch_std做同样的z-score
            Xs = (X.astype(np.float32) - ch_mean) / ch_std
            xb = torch.from_numpy(Xs).permute(0, 2, 1).to(device)  # (N, C, T)
            with torch.no_grad():
                logits = net(xb)
                if m2m:
                    # (N, n_classes, T) 逐帧softmax后按时间维取平均得到窗口级
                    # 概率——训练时评估用逐帧多数投票选类别，是离散的、没有
                    # 自然的置信度；这里下游要用置信度做阈值过滤/片段合并，
                    # 换成连续的时间维平均概率，跟多数投票在大多数情况下
                    # 结果一致，但能给出一个有意义的置信度数值
                    probs = F.softmax(logits, dim=1).mean(dim=2)
                else:
                    probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

    return _DLModelWrapper(), dl_meta

# 默认目标类别只有"抓挠"——保持旧调用方式（不传 --target_labels / 不设
# TARGET_LABELS 环境变量）时行为跟改造前完全一致，不会因为这次多类别
# 改造而意外多出别的类别的输出目录，也不会改变旧版脚本/流水线的产出结构。
DEFAULT_TARGET_LABELS = ["抓挠"]

ACC_CANDIDATES  = [["acc_x","acc_y","acc_z"],["AccX","AccY","AccZ"],["AX","AY","AZ"],["ax","ay","az"]]
GYRO_CANDIDATES = [["gyro_x","gyro_y","gyro_z"],["gyr_x","gyr_y","gyr_z"],["GyroX","GyroY","GyroZ"],["GX","GY","GZ"]]
TS_KEYWORDS     = ["time", "timestamp", "datetime", "chip_time", "pc_ms"]
# pc_ms 是 witmotion_imu 采集端存原始未降采样数据（"_raw.csv"）时用的时间戳列名，
# 存的是毫秒级epoch时间戳（数字），跟 timestamp 列（字符串日期）的解析方式不一样，
# 用 pd.to_datetime 时必须显式指定 unit="ms"，否则数值会被当成纳秒解析，全部错乱。
EPOCH_MS_TS_COLS = ["pc_ms"]


def find_cols(cols, candidates):
    for g in candidates:
        if all(c in cols for c in g):
            return g
    return None


def find_ts_col(cols):
    low = [c.lower() for c in cols]
    for kw in TS_KEYWORDS:
        for i, cl in enumerate(low):
            if kw in cl:
                return cols[i]
    return None


def load_csv(path):
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()))
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    acc_cols  = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    ts_col    = find_ts_col(df.columns.tolist())
    if acc_cols is None:
        raise ValueError(f"找不到加速度列: {list(df.columns)}")
    valid_mask = df[acc_cols].notnull().all(axis=1).values  # True = 有效行
    # 空文件（0行）时 valid_mask.mean() 对空数组求均值会触发 RuntimeWarning 且结果是nan，
    # nan跟>0.1比较恒为False，会把"这个文件根本没有数据"的情况悄悄当成缺失率正常放过去；
    # 显式判断成缺失率100%，既消掉警告又能让下游"数据缺失率过高"的告警正确触发
    null_ratio = (1 - valid_mask.mean()) if len(valid_mask) > 0 else 1.0
    acc  = df[acc_cols].ffill().bfill().values.astype(np.float32)
    gyro = df[gyro_cols].ffill().bfill().values.astype(np.float32) if gyro_cols \
           else np.zeros((len(df), 3), dtype=np.float32)
    if ts_col and ts_col.strip().lower() in EPOCH_MS_TS_COLS:
        from timestamp_utils import pc_ms_to_local_datetime
        ts = pc_ms_to_local_datetime(df[ts_col])
    elif ts_col:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        ts = None
    return acc, gyro, ts, valid_mask, null_ratio


def downsample(data, device_hz, model_hz, method="poly"):
    """method="poly"（默认，向后兼容）：scipy resample_poly，多相FIR滤波。
    method="training_match"：复刻 witmotion_imu 生成训练数据时用的算法（滑动平均低通+
    线性插值），device_hz != model_hz 且要求跟训练数据用同一套重采样算法时用这个——
    见 src/data/resample_training_match.py 顶部注释，两种方法实测有约6~8%的输出差异。
    """
    if device_hz == model_hz:
        return data
    if method == "training_match":
        from resample_training_match import resample_training_match
        return resample_training_match(data, device_hz, model_hz)
    g = gcd(device_hz, model_hz)
    up, down = model_hz // g, device_hz // g
    if up == 1:
        return data[::down]
    from scipy.signal import resample_poly
    return resample_poly(data, up, down, axis=0).astype(np.float32)


def sliding_windows(data, window_size, stride):
    windows, indices = [], []
    for start in range(0, len(data) - window_size + 1, stride):
        windows.append(data[start:start + window_size])
        indices.append(start)
    return np.stack(windows) if windows else np.empty((0, window_size, data.shape[1])), indices


def _extract_label_segments(preds, confs, start_indices, classes, window_bounds, idx_to_ts,
                            label, confidence_threshold):
    """从逐窗口预测里抽出某一个目标类别（比如"抓挠"或"甩身体"）的连续片段。
    以前这段逻辑写死判断 label=="抓挠"，现在改成单类别提取的独立函数，每个
    目标类别各自调用一次——一个窗口序列里完全可能同时含多个类别的片段
    （一段抓挠、另一段甩身体），必须分别扫描、分别产出，不能只扫一遍、
    混在一个列表里（下游按类别分文件输出时会分不清）。"""
    segs = []
    in_run = False
    run_first_i = None
    run_last_i = None
    for pred_id, conf, start_i in zip(preds, confs, start_indices):
        is_hit = (classes[pred_id] == label) and (conf >= confidence_threshold)
        if is_hit:
            if not in_run:
                in_run = True
                run_first_i = start_i
            run_last_i = start_i
        elif in_run:
            in_run = False
            s0, _ = window_bounds(run_first_i)
            _, e1 = window_bounds(run_last_i)
            segs.append((idx_to_ts(s0), idx_to_ts(e1), run_first_i, run_last_i))
    if in_run:
        s0, _ = window_bounds(run_first_i)
        _, e1 = window_bounds(run_last_i)
        segs.append((idx_to_ts(s0), idx_to_ts(e1), run_first_i, run_last_i))
    return segs


def infer_file(path, model, classes, window_size, stride, device_hz, model_hz, gravity_aligned,
               confidence_threshold=0.0, quiet=False, scratch_only=False, merge_gap_s=10,
               output_dir=None, min_windows=1, keep_isolated=True, label_mode="majority",
               resample_method="poly", target_labels=None, is_dl=False, **kwargs):
    # target_labels：要独立统计/输出的目标类别列表，默认只有"抓挠"，跟改造前
    # 行为完全一致。多个类别时，output_dir 下会按类别各建一个子目录，互不
    # 混淆（见函数末尾"保存 JSON 结果"部分）——CSV读取/降采样/重力对齐/特征
    # 提取/模型预测这些步骤全部只做一遍，只有"从逐窗口预测里抽片段"这一步
    # 按类别各跑一次，避免多类别时重复读CSV、重复跑模型（那部分才是真正
    # 耗时的部分，跟类别数无关的这些步骤没有理由跑N遍）。
    target_labels = target_labels or DEFAULT_TARGET_LABELS
    display_name = path.split("/")[-1].split("?")[0]  # works for both file paths and URLs
    if not scratch_only:
        print(f"\n── {display_name} ──")
    acc, gyro, ts, valid_mask, null_ratio = load_csv(path)
    if not scratch_only:
        print(f"  行数={len(acc)}  device_hz={device_hz}  model_hz={model_hz}")
        if null_ratio > 0.1:
            print(f"  [警告] 数据缺失率={null_ratio*100:.1f}%（蓝牙断联？），将跳过含缺失的窗口")

    # 降采样
    acc_ds       = downsample(acc,        device_hz, model_hz, method=resample_method)
    gyro_ds      = downsample(gyro,       device_hz, model_hz, method=resample_method)
    valid_mask_ds = downsample(valid_mask.astype(np.float32).reshape(-1, 1),
                               device_hz, model_hz, method=resample_method).reshape(-1) > 0.5

    # 时间戳对应（降采样后的索引 → 原始时间戳）
    ratio = device_hz / model_hz
    def idx_to_ts(i):
        orig = min(int(i * ratio), len(ts) - 1) if ts is not None else -1
        return ts.iloc[orig] if ts is not None and orig >= 0 else None

    # 重力对齐 + 滑窗（跳过缺失率>30%的窗口）
    data6 = np.concatenate([acc_ds, gyro_ds], axis=1)
    X, start_indices = sliding_windows(data6, window_size, stride)
    if len(X) > 0:
        valid_windows = [
            valid_mask_ds[s:s + window_size].mean() >= 0.7
            for s in start_indices
        ]
        X            = X[valid_windows]
        start_indices = [s for s, v in zip(start_indices, valid_windows) if v]
        n_skipped = sum(not v for v in valid_windows)
        if n_skipped and not scratch_only:
            print(f"  [过滤] 跳过 {n_skipped} 个缺失率>30% 的窗口")
    if len(X) == 0:
        return

    tilt = append_raw_tilt_batch(X)[:, :, 6:8]  # 原始（未对齐）姿态角，须在重力对齐前算
    if gravity_aligned:
        X_aligned = np.stack([gravity_align(X[i]) for i in range(len(X))])
    else:
        X_aligned = X
    X_aligned = np.concatenate([X_aligned, tilt], axis=2)

    # 提取特征 + 预测——DL模型直接吃X_aligned这个原始窗口数组（模型内部
    # 自己学卷积/时序特征，不需要ml/features.py那套手工统计特征），
    # 标准化(ch_mean/ch_std)在_DLModelWrapper.predict_proba()内部做
    if is_dl:
        feats = X_aligned
    else:
        feats = extract_features(X_aligned, model_hz, show_progress=not quiet and not scratch_only)
    probs = model.predict_proba(feats)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    # 单个窗口在原始时间轴上对应的 (起点采样点, 终点采样点)，取决于训练时的 label_mode：
    #   "majority"（默认）：一个正例窗口代表"这一整个window_size都是目标行为"，
    #     窗口起点到窗口终点整段对应，这是训练时多数投票的语义。
    #   "center"：一个正例窗口只代表"窗口正中心这一瞬间是目标行为"，只用窗口
    #     中心点前后半个步长(stride/2)去覆盖时间轴，不铺满整个窗口——否则中心点
    #     标注法带来的边界精度收益在推理这一步就白费了（跟 event_eval.py 里
    #     pred_windows_to_segments() 的逻辑保持一致）。
    def window_bounds(s):
        if label_mode == "center":
            center = s + window_size / 2
            half_pad = stride / 2
            return center - half_pad, center + half_pad
        return s, s + window_size

    # 打印逐窗口结果（多类别时，一个窗口只可能是argmax出来的那一个类别，
    # marker标出它是不是命中了target_labels里的某一个，不再写死"抓挠"）
    if not quiet:
        print(f"  {'时间':<22} {'预测':<6} {'置信度':>6}")
        print(f"  {'-'*38}")
        for pred_id, conf, start_i in zip(preds, confs, start_indices):
            label = classes[pred_id]
            t = idx_to_ts(start_i)
            t_str = t.strftime("%Y-%m-%d %H:%M:%S") if t is not None else f"帧{start_i}"
            is_target = label in target_labels and conf >= confidence_threshold
            marker = f" ⬅ {label}" if is_target else ""
            print(f"  {t_str:<22} {label:<6} {conf:>6.2f}{marker}")

    def fmt(t, i, suffix=""):
        return t.strftime(f"%H:%M:%S{suffix}") if t else f"帧{i}"

    def count_windows(s):
        return sum(1 for k in range(len(preds))
                   if start_indices[k] >= s[2] and start_indices[k] <= s[3])

    ts_fmt = lambda t: t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if t is not None else None

    # 每个目标类别独立扫描/合并/过滤/输出——一个窗口序列里完全可能同时含
    # 多个类别的片段（一段抓挠、另一段甩身体），必须分开处理，不能只跑
    # 一遍、混在同一份结果里（下游按类别拆目录就无从下手了）
    merged_by_label = {}
    for target_label in target_labels:
        raw_segs = _extract_label_segments(preds, confs, start_indices, classes,
                                           window_bounds, idx_to_ts,
                                           target_label, confidence_threshold)

        # 之前这里 scratch_only 且没检测到目标类别时会直接return，连下面
        # "保存JSON结果"那块都被跳过了——批量推理时某个文件如果真的一段
        # 目标行为都没有（比如模型误报率低了之后完全正常出现的情况），会
        # 导致它压根没有对应的 {stem}_infer.json，后续 extract_clips.py
        # 找不到文件、甚至在"这一批全部文件都零检出"时报错说 _infer 目录下
        # 没有任何 *_infer.json。改成不提前return，只在没检出时跳过下面的
        # 汇总打印（scratch_only本来的目的就是减少输出噪音），JSON该保存
        # 还是要保存。
        skip_print = scratch_only and not raw_segs

        # 合并相邻片段（间隔 <= merge_gap_s 秒视为同一段）——但不跨越"这段
        # 间隔里其实是另一个类别置信预测"的情况去合并：ML_PRELABEL_MULTI把
        # 多个类别的片段画在同一条时间轴上之后才发现，比如"抓挠"只持续了
        # 不到1s(1~2个窗口)，前后都是"活动"，之前这里只看时间间隔够不够
        # 短(<=merge_gap_s)就无条件桥接，桥接后的"活动"片段会整段盖住中间
        # 那段真实是"抓挠"的时间区间——单独看"抓挠"这一个类别的文件时这个
        # 问题完全看不出来(只有它自己的片段，不会跟别的类别比对)，多类别
        # 合并到同一条时间轴才会暴露出"同一段时间被两个类别同时标注"这个
        # 问题。改成合并前检查间隔内每个窗口的argmax预测——只要有任何一个
        # 窗口被置信预测(conf>=confidence_threshold)成了不是target_label
        # 的类别，就不桥接，保留成两段独立片段，避免吞掉中间那段真实发生
        # 的别的行为。
        def _gap_has_confident_other_label(prev_last_i, next_first_i):
            for pred_id, conf, start_i in zip(preds, confs, start_indices):
                if prev_last_i < start_i < next_first_i:
                    if classes[pred_id] != target_label and conf >= confidence_threshold:
                        return True
            return False

        merged = []
        for t0, t1, i0, i1 in raw_segs:
            if merged and t0 is not None and merged[-1][1] is not None:
                gap = (t0 - merged[-1][1]).total_seconds()
                if gap <= merge_gap_s and not _gap_has_confident_other_label(merged[-1][3], i0):
                    merged[-1] = (merged[-1][0], t1, merged[-1][2], i1)
                    continue
            merged.append([t0, t1, i0, i1])

        # 过滤孤立单窗口片段（前后均不是目标类别）
        if not keep_isolated:
            before = len(merged)
            merged = [s for s in merged if count_windows(s) > 1]
            if before != len(merged):
                dropped = before - len(merged)
                msg = f"  [过滤][{target_label}] 丢弃 {dropped} 段孤立单窗口（keep_isolated=False）"
                if scratch_only:
                    print(msg)
                elif not quiet:
                    print(msg)

        # 过滤窗口数不足的短片段
        if min_windows > 1:
            before = len(merged)
            merged = [s for s in merged if count_windows(s) >= min_windows]
            if not scratch_only and before != len(merged):
                print(f"  [过滤][{target_label}] 丢弃 {before - len(merged)} 段（窗口数 < {min_windows}）")

        merged_by_label[target_label] = merged

        n_hit = int(sum(1 for pid, c in zip(preds, confs)
                        if classes[pid] == target_label and c >= confidence_threshold))
        if not skip_print:
            if scratch_only:
                print(f"\n── {display_name} [{target_label}] ──")
            seg_str = "  ".join(fmt(t0, i0) + "→" + fmt(t1, i1) for t0, t1, i0, i1 in raw_segs) \
                      if raw_segs else f"未检测到{target_label}"
            merged_str = "  ".join(fmt(t0, i0) + "→" + fmt(t1, i1) for t0, t1, i0, i1 in merged) \
                         if merged else f"未检测到{target_label}"
            print(f"  【汇总:{target_label}】总窗口={len(preds)}  {target_label}窗口={n_hit}  "
                  f"({n_hit/len(preds)*100:.1f}%)")
            print(f"  【片段】{seg_str}")
            print(f"  【合并】{merged_str}")

    # ── 保存 JSON 结果（供后续复查和 Label Studio 上传）────────────────
    # 每个目标类别各写一份 JSON，落在 output_dir/{label}/ 下——不同类别的
    # 片段绝不能混进同一个 *_infer.json，否则下游 extract_clips.py/
    # review_to_labelstudio.py 没法区分"这段是抓挠还是甩身体"。逐窗口全
    # 概率(windows)是模型一次预测算出来的、跟具体目标类别无关，每个类别
    # 的文件里都完整存一份——好处是单看某个类别的_infer.json时仍能查到
    # 当时模型对其它类别的判断，代价是同一次推理的windows在多个类别目录
    # 下重复存了一份，用磁盘换清晰度（数据量本身不大，可接受）。
    if output_dir:
        import json as _json
        windows_out = []
        for i, (pred_id, start_i) in enumerate(zip(preds, start_indices)):
            t = idx_to_ts(start_i)
            prob_vec = {classes[j]: float(probs[i, j]) for j in range(len(classes))}
            windows_out.append({
                "ts":    ts_fmt(t),
                "label": classes[pred_id],
                "conf":  float(probs[i, pred_id]),
                "probs": prob_vec,
            })

        stem = os.path.splitext(os.path.basename(path))[0]
        for target_label in target_labels:
            merged = merged_by_label[target_label]
            segs_out = []
            for t0, t1, i0, i1 in merged:
                seg_probs = [probs[k, classes.index(target_label)]
                             for k in range(len(preds))
                             if start_indices[k] >= i0 and start_indices[k] <= i1
                             and target_label in classes]
                segs_out.append({
                    "start_ts":  ts_fmt(t0),
                    "end_ts":    ts_fmt(t1),
                    "conf_max":  float(max(seg_probs)) if seg_probs else 0.0,
                    "conf_mean": float(sum(seg_probs) / len(seg_probs)) if seg_probs else 0.0,
                    "n_windows": len(seg_probs),
                })

            n_hit = int(sum(1 for pid, c in zip(preds, confs)
                            if classes[pid] == target_label and c >= confidence_threshold))
            out = {
                "csv_file":        os.path.abspath(path),
                "csv_basename":    os.path.basename(path),
                "n_windows":       len(preds),
                "n_scratch":       n_hit,  # 字段名沿用旧版（曾经只有"抓挠"一个类别），
                                            # 语义变成"本文件所在label子目录对应类别的命中窗口数"，
                                            # 不改字段名是为了不用同步改review_to_labelstudio.py/
                                            # imu_scratch_daily_stats.py/extract_clips.py里所有
                                            # 读这个字段的地方——它们本来读的就是"目标类别的片段"，
                                            # 现在目标类别通过目录结构区分，字段语义不需要变
                "windows":         windows_out,
                "scratch_segments": segs_out,
            }
            # output_dir 是"这一天"这一级的目录，每个类别的_infer.json落在
            # output_dir/{label}/_infer/ 下——跟以前"--output_dir直接就是
            # 落盘目录"不同（以前只有一个类别，调用方/bash脚本自己拼好
            # .../_infer再传进来）；现在一次调用要覆盖多个类别，"往哪个
            # label子目录、要不要_infer这一层"必须由这里统一决定，不能再
            # 依赖调用方每个类别各传一次不同路径
            label_out_dir = os.path.join(output_dir, target_label, "_infer")
            os.makedirs(label_out_dir, exist_ok=True)
            out_path = os.path.join(label_out_dir, f"{stem}_infer.json")
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump(out, f, ensure_ascii=False, indent=2)

    return preds, classes, merged_by_label


def main():
    parser = argparse.ArgumentParser(description="离线 CSV 抓挠识别")
    parser.add_argument("--csv",       default="", help="单个 CSV 文件路径")
    parser.add_argument("--csv_dir",   default="", help="CSV 目录（批量处理）")
    parser.add_argument("--pattern",   default="*.csv", help="文件名通配符（默认 *.csv）")
    parser.add_argument("--model",     required=True, help="ML 模型路径（.pkl）")
    parser.add_argument("--device_hz", type=int, default=0,
                        help="CSV 采样率（0=自动从模型元数据读取）")
    parser.add_argument("--model_hz",  type=int, default=0,
                        help="模型训练采样率（0=与device_hz相同）")
    parser.add_argument("--window_s",  type=float, default=0,
                        help="窗口秒数（0=从模型元数据读取，默认2.0）")
    parser.add_argument("--stride_s",  type=float, default=0,
                        help="步长秒数（0=从模型元数据读取，默认1.0）")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="置信度阈值，低于此值的预测忽略（默认0=不过滤，建议0.65-0.75）")
    parser.add_argument("--quiet", action="store_true",
                        help="只输出每个文件的汇总行，不打印逐窗口详情")
    parser.add_argument("--scratch_only", action="store_true",
                        help="只输出检测到抓挠的文件，忽略无抓挠的文件")
    parser.add_argument("--merge_gap", type=float, default=1.0,
                        help="合并相邻抓挠片段的最大间隔秒数（默认1s，"
                             "event_eval.py 验证过3s会导致约一半真实事件被错误合并）")
    parser.add_argument("--min_windows", type=int, default=1,
                        help="片段最少窗口数，不足则丢弃（默认1=不过滤）")
    parser.add_argument("--no_keep_isolated", action="store_true",
                        help="丢弃孤立单窗口片段（前后均不是抓挠的单帧抓挠，默认保留）")
    parser.add_argument("--workers", type=int, default=-1,
                        help="并行进程数（默认-1=用全部CPU核，1=单进程）")
    parser.add_argument("--output_dir", default="",
                        help="保存每个文件推理结果 JSON 的目录（留空不保存），通常传当天的"
                             "结果根目录（比如 RESULT_ROOT/{day}）。每个目标类别会各自落在"
                             "output_dir/{label}/_infer/{stem}_infer.json——注意跟旧版不同，"
                             "旧版output_dir就是json直接落盘的目录（调用方自己拼好.../_infer"
                             "再传进来）；现在一次调用要覆盖多个类别，'{label}/_infer'这一层"
                             "统一由本脚本自己拼，不用调用方每个类别各传一次不同路径")
    parser.add_argument("--no_gravity_align", action="store_true")
    parser.add_argument("--target_labels", default="抓挠",
                        help="要独立检测/统计/输出的目标类别，逗号分隔，比如'抓挠,甩身体'"
                             "（默认只有'抓挠'，保持跟改造前完全一致的行为——不传这个参数的"
                             "旧调用方式不受影响，也不会多出别的类别子目录）。CSV只读一次、"
                             "特征只提一次、模型只预测一次，只有片段提取/合并/JSON输出这几步"
                             "按类别各跑一次，不是每个类别重新推理一遍")
    parser.add_argument("--resample_method", default="poly", choices=["poly", "training_match"],
                        help="device_hz != model_hz 时用哪种降采样算法（默认poly=scipy "
                             "resample_poly）。training_match 复刻 witmotion_imu 生成训练数据"
                             "时用的算法（滑动平均低通+线性插值），两种方法实测有约6~8%的输出"
                             "差异（src/eval/compare_resample_methods.py），源数据不是已经"
                             "预先降采样好的16Hz witmotion文件时（比如TF设备）值得两种都试试")
    args = parser.parse_args()

    target_labels = [t.strip() for t in args.target_labels.split(",") if t.strip()]
    if not target_labels:
        target_labels = list(DEFAULT_TARGET_LABELS)

    # 加载模型 + 元数据——.pt是DL模型(src/dl/train.py训出来的state_dict)，
    # .pkl是ML模型(src/ml/train.py, joblib)，两者元数据格式和推理输入
    # 都不一样，靠后缀名分流
    is_dl = args.model.endswith(".pt")
    classes, gravity_aligned, t_hz, t_window_s, t_stride_s = [], True, 16, 2.0, 1.0
    label_mode = "majority"
    if is_dl:
        model, dl_meta = _load_dl_model(args.model)
        classes         = dl_meta["classes"]
        gravity_aligned = dl_meta["gravity_aligned"]
        t_hz            = int(dl_meta["hz"])
        t_window_s      = dl_meta["window_size"] / t_hz
        t_stride_s      = dl_meta["stride"] / t_hz
        label_mode      = dl_meta.get("label_mode", "majority")
        print(f"[模型] DL模型: {dl_meta['model']}  训练参数: 采样率={t_hz}Hz  "
              f"窗口={t_window_s}s  步长={t_stride_s}s  重力对齐={gravity_aligned}  "
              f"label_mode={label_mode}  many-to-many={dl_meta['m2m']}")
        print(f"[模型] 类别: {classes}")
    else:
        model = joblib.load(args.model)
        meta_path = args.model.replace(".pkl", ".json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            classes        = meta.get("classes", [])
            gravity_aligned= meta.get("gravity_aligned", True)
            t_hz           = int(meta.get("hz", 16))
            t_window_s     = float(meta.get("window_s", 2.0))
            t_stride_s     = float(meta.get("stride_s", 1.0))
            label_mode     = meta.get("label_mode", "majority")  # 旧模型没这个字段，退化为majority（兼容原有行为）
            print(f"[模型] 训练参数: 采样率={t_hz}Hz  窗口={t_window_s}s  步长={t_stride_s}s  "
                  f"重力对齐={gravity_aligned}  label_mode={label_mode}")
            print(f"[模型] 类别: {classes}")
        else:
            classes = list(model.classes_) if hasattr(model, "classes_") else []
            print(f"[模型] 未找到元数据 JSON，类别: {classes}")

    if args.no_gravity_align:
        gravity_aligned = False

    model_hz  = args.model_hz  or t_hz
    device_hz = args.device_hz or model_hz
    window_s  = args.window_s  or t_window_s
    stride_s  = args.stride_s  or t_stride_s
    window_size = int(window_s * model_hz)
    stride      = int(stride_s * model_hz)

    print(f"[推理] 设备Hz={device_hz}  模型Hz={model_hz}  窗口={window_s}s  步长={stride_s}s")

    print(f"[推理] 目标类别: {target_labels}")
    for tl in target_labels:
        if tl not in classes:
            print(f"[警告] 模型类别中没有'{tl}': {classes}")

    # 收集文件列表
    files = []
    if args.csv:
        files = [args.csv]
    elif args.csv_dir:
        files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    if not files:
        print("[错误] 请指定 --csv 或 --csv_dir")
        return

    print(f"\n共 {len(files)} 个文件")

    from tqdm import tqdm
    from joblib import Parallel, delayed

    def _run_one(path):
        try:
            out_dir = args.output_dir or None
            return infer_file(path, model, classes, window_size, stride,
                              device_hz, model_hz, gravity_aligned,
                              confidence_threshold=args.confidence_threshold,
                              quiet=args.quiet,
                              scratch_only=args.scratch_only,
                              merge_gap_s=args.merge_gap,
                              min_windows=args.min_windows,
                              keep_isolated=not args.no_keep_isolated,
                              label_mode=label_mode,
                              output_dir=out_dir,
                              resample_method=args.resample_method,
                              target_labels=target_labels,
                              is_dl=is_dl)
        except Exception as e:
            tqdm.write(f"  [错误] {os.path.basename(path)}: {e}")
            return None

    # DL模型(torch.nn.Module)跨进程传给loky worker既慢（每个任务都要
    # 重新pickle整个模型+权重）又容易因为动态定义的_DLModelWrapper类
    # 序列化不稳定而出问题，ML路径没有这个包袱（sklearn模型走joblib
    # 本来就是为了多进程设计的）。DL强制单进程——神经网络单窗口推理本身
    # 很快，真正的批量瓶颈在ML的手工特征提取那步，DL跳过了那步，
    # 单进程也不会明显慢
    n_jobs = 1 if is_dl else (args.workers if args.workers > 0 else -1)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one)(p) for p in tqdm(files, desc="推理进度", unit="文件")
    )

    # 每个目标类别独立累计窗口命中数——以前只认"抓挠"一个类别，现在每个
    # target_label各自一份计数器，不能混在一起
    all_hits  = {tl: 0 for tl in target_labels}
    all_total = 0
    for result in results:
        if result:
            preds, _, _merged_by_label = result
            for tl in target_labels:
                if tl in classes:
                    all_hits[tl] += int((np.array(preds) == classes.index(tl)).sum())
            all_total += len(preds)

    if len(files) > 1:
        print(f"\n{'='*50}")
        if all_total:
            for tl in target_labels:
                print(f"批量汇总[{tl}]: 总窗口={all_total}  {tl}窗口={all_hits[tl]}  "
                      f"({all_hits[tl]/all_total*100:.1f}%)")
        else:
            print("无有效数据")


if __name__ == "__main__":
    main()
