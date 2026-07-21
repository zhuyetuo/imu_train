"""
根据真实抓挠 IMU 片段合成更多抓挠数据。

原理:
  1. 从带标注的真实 CSV 中切出所有抓挠片段
  2. 对每个片段做多种数据增强：时间拉伸、幅值缩放、轴向翻转、加噪声、时间偏移
  3. 将增强后的片段滑窗切成 (window_size, 6) 格式
  4. 保存为 .npz（与 --synthetic 参数兼容）

用法:
  python src/data/synthesize_scratch.py \
    --csv1 data/raw_wit/multicam_20260715_084939_cam1_imu1_resampled16hz.csv \
    --csv2 data/raw_wit/multicam_20260715_084939_cam2_imu2_resampled16hz.csv \
    --output data/synthetic/scratch_synthetic.npz \
    --hz 16 \
    --n_aug 30

  # 查看生成的窗口数
  python -c "import numpy as np; d=np.load('data/synthetic/scratch_synthetic.npz'); print(d['X'].shape)"
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime

# ── task 472 的抓挠标注时间段 ──────────────────────────────────────────────────
# 来自 project-24 JSON，label1 对应 csv1，label2 对应 csv2
SCRATCH_SEGS = {
    "csv1": [
        ("2026-07-15 08:52:37.577", "2026-07-15 08:52:44.265"),
        ("2026-07-15 08:52:50.202", "2026-07-15 08:52:54.702"),
        ("2026-07-15 08:54:23.327", "2026-07-15 08:54:28.015"),
        ("2026-07-15 08:55:13.390", "2026-07-15 08:55:17.140"),
        ("2026-07-15 08:55:34.390", "2026-07-15 08:55:35.827"),
        ("2026-07-15 08:55:39.202", "2026-07-15 08:55:43.702"),
        ("2026-07-15 08:55:59.577", "2026-07-15 08:56:12.952"),
        ("2026-07-15 08:56:16.515", "2026-07-15 08:56:19.952"),
        ("2026-07-15 08:56:39.452", "2026-07-15 08:56:43.952"),
        ("2026-07-15 08:56:52.265", "2026-07-15 08:56:54.452"),
    ],
    "csv2": [
        ("2026-07-15 08:51:54.884", "2026-07-15 08:52:06.259"),
        ("2026-07-15 08:52:19.072", "2026-07-15 08:52:27.009"),
        ("2026-07-15 08:52:43.384", "2026-07-15 08:52:48.072"),
    ],
}

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    # 自动找时间戳列
    ts_col = next((c for c in df.columns if "time" in c.lower()), None)
    if ts_col is None:
        raise ValueError(f"找不到时间戳列: {list(df.columns)}")
    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    # 自动找传感器列
    acc_cols  = next((g for g in [["acc_x","acc_y","acc_z"],["AccX","AccY","AccZ"]] if all(c in df.columns for c in g)), None)
    gyro_cols = next((g for g in [["gyro_x","gyro_y","gyro_z"],["gyr_x","gyr_y","gyr_z"],["GyroX","GyroY","GyroZ"]] if all(c in df.columns for c in g)), None)
    if acc_cols is None:
        raise ValueError(f"找不到加速度列: {list(df.columns)}")
    if gyro_cols is None:
        gyro_cols = None
    return df, acc_cols, gyro_cols


def extract_segments(df, acc_cols, gyro_cols, segs, min_rows=16):
    """切出标注时间段，返回 list of (N,6) ndarray。"""
    results = []
    for t0_str, t1_str in segs:
        t0 = pd.to_datetime(t0_str)
        t1 = pd.to_datetime(t1_str)
        mask = (df["_ts"] >= t0) & (df["_ts"] <= t1)
        seg = df[mask]
        if len(seg) < min_rows:
            print(f"  [跳过] {t0_str} 只有 {len(seg)} 行，小于最小 {min_rows} 行")
            continue
        acc  = seg[acc_cols].values.astype(np.float32)
        gyro = seg[gyro_cols].values.astype(np.float32) if gyro_cols else np.zeros((len(seg), 3), dtype=np.float32)
        data = np.concatenate([acc, gyro], axis=1)  # (N, 6)
        results.append(data)
        print(f"  提取 {t0_str} → {t1_str}: {len(seg)} 行")
    return results


# ── 增强函数 ──────────────────────────────────────────────────────────────────

def aug_noise(seg, scale=0.02):
    """加高斯噪声。"""
    noise = np.random.randn(*seg.shape).astype(np.float32) * scale
    return seg + noise


def aug_scale(seg, low=0.85, high=1.15):
    """幅值缩放（acc 和 gyro 分别随机）。"""
    s = np.random.uniform(low, high, size=(1, seg.shape[1])).astype(np.float32)
    return seg * s


def aug_flip_axis(seg):
    """随机翻转某一轴（模拟佩戴方向差异）。"""
    axis = np.random.randint(0, 3)
    out = seg.copy()
    out[:, axis] *= -1
    out[:, axis + 3] *= -1
    return out


def aug_time_stretch(seg, low=0.8, high=1.2, target_len=None):
    """时间拉伸/压缩后插值回原长度。"""
    from scipy.signal import resample
    factor = np.random.uniform(low, high)
    new_len = max(4, int(len(seg) * factor))
    stretched = resample(seg, new_len, axis=0).astype(np.float32)
    if target_len is not None:
        stretched = resample(stretched, target_len, axis=0).astype(np.float32)
    return stretched


def aug_time_shift(seg, max_shift_frac=0.1):
    """循环时移（让波形相位不同）。"""
    shift = np.random.randint(1, max(2, int(len(seg) * max_shift_frac)))
    return np.roll(seg, shift, axis=0)


def augment_segment(seg, n_aug, rng):
    """对单个原始片段生成 n_aug 个增强版本（每次随机组合多种增强）。"""
    variants = []
    aug_fns = [aug_noise, aug_scale, aug_flip_axis, aug_time_shift]
    for _ in range(n_aug):
        out = seg.copy()
        # 每次随机选 1-3 种增强叠加
        chosen = rng.choice(len(aug_fns), size=rng.integers(1, 4), replace=False)
        for i in chosen:
            try:
                out = aug_fns[i](out)
            except Exception:
                pass
        # 20% 概率加时间拉伸
        if rng.random() < 0.2:
            out = aug_time_stretch(out, target_len=len(out))
        variants.append(out)
    return variants


def sliding_windows(data, window_size, stride):
    """从 (N,6) 数据切滑窗，返回 list of (window_size, 6)。"""
    wins = []
    for start in range(0, len(data) - window_size + 1, stride):
        wins.append(data[start:start + window_size])
    return wins


def main():
    parser = argparse.ArgumentParser(description="抓挠合成数据生成器")
    parser.add_argument("--csv1",   required=True, help="imu1 CSV 路径")
    parser.add_argument("--csv2",   default="",    help="imu2 CSV 路径（可选）")
    parser.add_argument("--output", required=True, help="输出 .npz 路径")
    parser.add_argument("--hz",     type=int, default=16, help="采样率")
    parser.add_argument("--window_s", type=float, default=2.0, help="窗口秒数")
    parser.add_argument("--stride_s", type=float, default=1.0, help="步长秒数")
    parser.add_argument("--n_aug",  type=int, default=30, help="每个原始片段生成的增强数量")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    window_size = int(args.window_s * args.hz)
    stride      = int(args.stride_s * args.hz)
    print(f"窗口={window_size}点  步长={stride}点  采样率={args.hz}Hz")

    all_segs = []

    # ── 加载 csv1 ─────────────────────────────────────────────────────────────
    print(f"\n[csv1] {args.csv1}")
    df1, acc1, gyro1 = load_csv(args.csv1)
    segs1 = extract_segments(df1, acc1, gyro1, SCRATCH_SEGS["csv1"])
    all_segs.extend(segs1)

    # ── 加载 csv2 ─────────────────────────────────────────────────────────────
    if args.csv2:
        print(f"\n[csv2] {args.csv2}")
        df2, acc2, gyro2 = load_csv(args.csv2)
        segs2 = extract_segments(df2, acc2, gyro2, SCRATCH_SEGS["csv2"])
        all_segs.extend(segs2)

    print(f"\n共提取 {len(all_segs)} 个原始片段")

    # ── 从原始片段切窗 ────────────────────────────────────────────────────────
    raw_windows = []
    for seg in all_segs:
        wins = sliding_windows(seg, window_size, stride)
        raw_windows.extend(wins)
    print(f"原始片段滑窗: {len(raw_windows)} 个窗口")

    # ── 增强 ──────────────────────────────────────────────────────────────────
    aug_windows = []
    for seg in all_segs:
        variants = augment_segment(seg, args.n_aug, rng)
        for v in variants:
            wins = sliding_windows(v, window_size, stride)
            aug_windows.extend(wins)

    all_windows = raw_windows + aug_windows
    print(f"增强后总窗口: {len(all_windows)} 个")

    if not all_windows:
        print("[错误] 没有生成任何窗口，请检查 CSV 路径和标注时间段")
        return

    X = np.stack(all_windows, axis=0)  # (N, window_size, 6)
    print(f"\n输出形状: {X.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, X=X)
    print(f"已保存: {args.output}")

    print(f"\n下一步（注入合成数据训练）:")
    print(f"  python src/ml/train.py --hz {args.hz} --model rf \\")
    print(f"    --processed_dir data/processed_merged_all \\")
    print(f"    --synthetic {args.output} --synthetic_label 抓挠")


if __name__ == "__main__":
    main()
