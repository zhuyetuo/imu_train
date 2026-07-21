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

from gravity_align import gravity_align
from features import extract_features

import joblib
import json

ACC_CANDIDATES  = [["acc_x","acc_y","acc_z"],["AccX","AccY","AccZ"],["AX","AY","AZ"],["ax","ay","az"]]
GYRO_CANDIDATES = [["gyro_x","gyro_y","gyro_z"],["gyr_x","gyr_y","gyr_z"],["GyroX","GyroY","GyroZ"],["GX","GY","GZ"]]
TS_KEYWORDS     = ["time", "timestamp", "datetime", "chip_time"]


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
    null_ratio = 1 - valid_mask.mean()
    acc  = df[acc_cols].ffill().bfill().values.astype(np.float32)
    gyro = df[gyro_cols].ffill().bfill().values.astype(np.float32) if gyro_cols \
           else np.zeros((len(df), 3), dtype=np.float32)
    ts   = pd.to_datetime(df[ts_col], errors="coerce") if ts_col else None
    return acc, gyro, ts, valid_mask, null_ratio


def downsample(data, device_hz, model_hz):
    if device_hz == model_hz:
        return data
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


def infer_file(path, model, classes, window_size, stride, device_hz, model_hz, gravity_aligned,
               confidence_threshold=0.0, quiet=False, scratch_only=False):
    display_name = path.split("/")[-1].split("?")[0]  # works for both file paths and URLs
    if not scratch_only:
        print(f"\n── {display_name} ──")
    acc, gyro, ts, valid_mask, null_ratio = load_csv(path)
    if not scratch_only:
        print(f"  行数={len(acc)}  device_hz={device_hz}  model_hz={model_hz}")
        if null_ratio > 0.1:
            print(f"  [警告] 数据缺失率={null_ratio*100:.1f}%（蓝牙断联？），将跳过含缺失的窗口")

    # 降采样
    acc_ds       = downsample(acc,        device_hz, model_hz)
    gyro_ds      = downsample(gyro,       device_hz, model_hz)
    valid_mask_ds = downsample(valid_mask.astype(np.float32).reshape(-1, 1),
                               device_hz, model_hz).reshape(-1) > 0.5

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

    if gravity_aligned:
        X_aligned = np.stack([gravity_align(X[i]) for i in range(len(X))])
    else:
        X_aligned = X

    # 提取特征 + 预测
    feats = extract_features(X_aligned, model_hz, show_progress=not quiet and not scratch_only)
    probs = model.predict_proba(feats)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    # 打印逐窗口结果
    scratch_segs = []
    in_scratch   = False
    seg_start_ts = None
    seg_start_i  = None

    if not quiet:
        print(f"  {'时间':<22} {'预测':<6} {'置信度':>6}")
        print(f"  {'-'*38}")

    for i, (pred_id, conf, start_i) in enumerate(zip(preds, confs, start_indices)):
        label = classes[pred_id]
        # 置信度低于阈值时，将抓挠预测视为非抓挠
        if label == "抓挠" and conf < confidence_threshold:
            label = f"({classes[pred_id]}?)"
        t = idx_to_ts(start_i)
        if not quiet:
            t_str = t.strftime("%Y-%m-%d %H:%M:%S") if t is not None else f"帧{start_i}"
            marker = " ⬅ 抓挠" if label == "抓挠" else ""
            print(f"  {t_str:<22} {label:<6} {conf:>6.2f}{marker}")

        # 合并连续抓挠片段
        if label == "抓挠" and not in_scratch:
            in_scratch   = True
            seg_start_ts = t
            seg_start_i  = start_i
        elif label != "抓挠" and in_scratch:
            in_scratch = False
            seg_end_ts = idx_to_ts(start_i)
            scratch_segs.append((seg_start_ts, seg_end_ts, seg_start_i, start_i))

    if in_scratch:
        seg_end_ts = idx_to_ts(start_indices[-1] + window_size)
        scratch_segs.append((seg_start_ts, seg_end_ts, seg_start_i, start_indices[-1]))

    # 汇总
    n_scratch = int((preds == classes.index("抓挠")).sum()) if "抓挠" in classes else 0
    if scratch_only and not scratch_segs:
        return preds, classes, scratch_segs
    seg_str = "  ".join(
        f"{t0.strftime('%H:%M:%S') if t0 else f'帧{i0}'}→{t1.strftime('%H:%M:%S') if t1 else f'帧{i1}'}"
        for t0, t1, i0, i1 in scratch_segs
    ) if scratch_segs else "未检测到抓挠"
    if scratch_only:
        print(f"\n── {display_name} ──")
    print(f"  【汇总】总窗口={len(preds)}  抓挠窗口={n_scratch}  ({n_scratch/len(preds)*100:.1f}%)  {seg_str}")

    return preds, classes, scratch_segs


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
    parser.add_argument("--no_gravity_align", action="store_true")
    args = parser.parse_args()

    # 加载模型 + 元数据
    model = joblib.load(args.model)
    meta_path = args.model.replace(".pkl", ".json")
    classes, gravity_aligned, t_hz, t_window_s, t_stride_s = [], True, 16, 2.0, 1.0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        classes        = meta.get("classes", [])
        gravity_aligned= meta.get("gravity_aligned", True)
        t_hz           = int(meta.get("hz", 16))
        t_window_s     = float(meta.get("window_s", 2.0))
        t_stride_s     = float(meta.get("stride_s", 1.0))
        print(f"[模型] 训练参数: 采样率={t_hz}Hz  窗口={t_window_s}s  步长={t_stride_s}s  重力对齐={gravity_aligned}")
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

    if "抓挠" not in classes:
        print(f"[警告] 模型类别中没有'抓挠': {classes}")

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
    all_scratch = 0
    all_total   = 0
    for path in tqdm(files, desc="推理进度", unit="文件"):
        try:
            result = infer_file(path, model, classes, window_size, stride,
                                device_hz, model_hz, gravity_aligned,
                                confidence_threshold=args.confidence_threshold,
                                quiet=args.quiet,
                                scratch_only=args.scratch_only)
            if result:
                preds, _, _ = result
                all_scratch += int((np.array(preds) == classes.index("抓挠")).sum()) if "抓挠" in classes else 0
                all_total   += len(preds)
        except Exception as e:
            print(f"  [错误] {e}")

    if len(files) > 1:
        print(f"\n{'='*50}")
        print(f"批量汇总: 总窗口={all_total}  抓挠窗口={all_scratch}  ({all_scratch/all_total*100:.1f}%)" if all_total else "无有效数据")


if __name__ == "__main__":
    main()
