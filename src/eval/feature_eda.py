"""
特征 EDA：对已标注数据的每个行为类别，走完整的特征提取流程
（与训练时一致：重力对齐 + 姿态角 + 全部182维特征），
按类别统计并计算每个特征的判别力（ANOVA F-score），
用于回答"该重点关注哪些特征——尤其是非常规特征（模长/Jerk/轴间相关/
分频段能量/姿态角）——对区分当前类别有没有用"。

用法:
  python src/eval/feature_eda.py \\
    --labeled_csv data/raw_custom/2026_7_30/merged_all_labels_2026_7_30.csv --hz 16

样本量太少的类别（默认<10个窗口）只展示窗口数，不参与判别力排名，
避免小样本噪声误导结论。
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))

from gravity_align import gravity_align_batch, append_raw_tilt_batch
from features import extract_features, feature_names

ACC_CANDIDATES  = [["acc_x", "acc_y", "acc_z"], ["AccX", "AccY", "AccZ"], ["AX", "AY", "AZ"], ["ax", "ay", "az"]]
GYRO_CANDIDATES = [["gyro_x", "gyro_y", "gyro_z"], ["gyr_x", "gyr_y", "gyr_z"], ["GyroX", "GyroY", "GyroZ"], ["GX", "GY", "GZ"]]

UNCONVENTIONAL_MARKERS = ("mag", "jerk", "corr", "sma", "band_energy", "pitch", "roll")


def find_cols(cols, candidates):
    for g in candidates:
        if all(c in cols for c in g):
            return g
    return None


def find_contiguous_segments(labels, min_len):
    """返回 [(start, end, label), ...]，end 不含"""
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


def build_windows(seg_df, acc_cols, gyro_cols, window_size, stride):
    data = np.concatenate(
        [seg_df[acc_cols].values, seg_df[gyro_cols].values], axis=1
    ).astype(np.float32)
    n = len(data)
    wins = [data[s:s + window_size] for s in range(0, n - window_size + 1, stride)]
    if not wins:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32)
    return np.stack(wins)


def is_unconventional(name: str) -> bool:
    return any(k in name for k in UNCONVENTIONAL_MARKERS)


def main():
    ap = argparse.ArgumentParser(description="特征判别力 EDA：哪些特征（尤其非常规特征）对区分当前行为类别有用")
    ap.add_argument("--labeled_csv", required=True, help="合并CSV路径（dog_id/label/acc_x.../gyr_z 逐行标签）")
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--window_s", type=float, default=2.0)
    ap.add_argument("--stride_s", type=float, default=1.0)
    ap.add_argument("--min_seg_s", type=float, default=2.0, help="连续同标签片段最短秒数")
    ap.add_argument("--dog_id_col", default="dog_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--min_windows_per_class", type=int, default=10,
                     help="窗口数低于此值的类别只展示、不参与判别力排名（默认10）")
    ap.add_argument("--top_n", type=int, default=30)
    ap.add_argument("--csv_out", default="", help="可选：把完整排名保存为 CSV")
    args = ap.parse_args()

    print(f"读取 {args.labeled_csv} ...")
    df = pd.read_csv(args.labeled_csv)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    acc_cols = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    if acc_cols is None or gyro_cols is None:
        print(f"[错误] 找不到 acc/gyro 列，现有列: {list(df.columns)}")
        return
    missing = [c for c in [args.dog_id_col, args.label_col] if c not in df.columns]
    if missing:
        print(f"[错误] 缺少列: {missing}")
        return

    window_size = int(args.window_s * args.hz)
    stride = max(1, int(args.stride_s * args.hz))
    min_len = max(window_size, int(args.hz * args.min_seg_s))

    label_wins = {}
    for dog_id, sub in df.groupby(args.dog_id_col):
        labels = sub[args.label_col].values
        for start, end, label in find_contiguous_segments(labels, min_len):
            wins = build_windows(sub.iloc[start:end], acc_cols, gyro_cols, window_size, stride)
            if len(wins):
                label_wins.setdefault(label, []).append(wins)

    label_X = {lbl: np.concatenate(arrs, axis=0) for lbl, arrs in label_wins.items()}
    if not label_X:
        print("[警告] 没有提取到任何窗口，检查 --min_seg_s 是否太大")
        return

    print(f"\n窗口={window_size}点({args.window_s}s)  步长={stride}点({args.stride_s}s)")
    print("各类别窗口数:")
    for lbl in sorted(label_X, key=lambda l: -len(label_X[l])):
        n = len(label_X[lbl])
        flag = "" if n >= args.min_windows_per_class else "  ⚠️ 样本太少，仅展示不参与排名"
        print(f"  {lbl}: {n} 窗口{flag}")

    print("\n提取特征（重力对齐 + 姿态角 + 182维完整特征）...")
    label_feats = {}
    from tqdm import tqdm
    for lbl, X in tqdm(label_X.items(), desc="按类别提取特征", unit="类别"):
        tilt = append_raw_tilt_batch(X)[:, :, 6:8]
        X_aligned = gravity_align_batch(X)
        X_full = np.concatenate([X_aligned, tilt], axis=2)
        label_feats[lbl] = extract_features(X_full, args.hz, show_progress=True)

    names = feature_names(8)
    n_feat = len(names)

    valid_labels = [l for l in label_feats if len(label_feats[l]) >= args.min_windows_per_class]
    if len(valid_labels) < 2:
        print(f"\n[警告] 窗口数 >= {args.min_windows_per_class} 的类别不足2个，无法计算判别力排名")
        return

    print(f"\n参与判别力排名的类别: {valid_labels}")

    f_scores = np.zeros(n_feat)
    p_values = np.ones(n_feat)
    for i in range(n_feat):
        groups = [label_feats[l][:, i] for l in valid_labels]
        try:
            f, p = scipy_stats.f_oneway(*groups)
        except Exception:
            f, p = 0.0, 1.0
        f_scores[i] = 0.0 if np.isnan(f) else f
        p_values[i] = 1.0 if np.isnan(p) else p

    order = np.argsort(-f_scores)

    print(f"\n{'='*78}")
    print(f"  判别力 Top {args.top_n} 特征（ANOVA F-score，越高说明该特征在类别间差异越显著）")
    print(f"{'='*78}")
    print(f"  {'排名':<4}{'类型':<8}{'特征名':<32}{'F值':>10}{'p值':>12}")
    for rank, idx in enumerate(order[:args.top_n], 1):
        tag = "🔶非常规" if is_unconventional(names[idx]) else "  常规"
        sig = "***" if p_values[idx] < 0.001 else ("**" if p_values[idx] < 0.01 else ("*" if p_values[idx] < 0.05 else ""))
        print(f"  {rank:<4}{tag:<8}{names[idx]:<32}{f_scores[idx]:>10.1f}{p_values[idx]:>12.2e}{sig}")

    n_unconv = sum(1 for idx in order[:args.top_n] if is_unconventional(names[idx]))
    print(f"\nTop{args.top_n} 中有 {n_unconv} 个是非常规特征"
          f"（模长/Jerk/轴间相关/SMA/分频段能量/姿态角，均为最近新增）")

    if args.csv_out:
        out_df = pd.DataFrame({
            "feature": names,
            "f_score": f_scores,
            "p_value": p_values,
            "unconventional": [is_unconventional(n) for n in names],
        }).sort_values("f_score", ascending=False)
        out_df.to_csv(args.csv_out, index=False)
        print(f"\n完整排名已保存: {args.csv_out}")

    print(f"""
{'='*78}
  怎么解读：
  - F值越大，说明这个特征在不同类别间的均值差异相对组内波动越显著，
    对分类越有潜在贡献。p值是显著性检验的参考，样本量小时不必太较真。
  - 只有窗口数>={args.min_windows_per_class}的类别参与了排名，样本太少的类别
    （比如你标注比较粗糙、段数个位数的那几类）不参与计算，避免小样本
    噪声混进结论。
  - 如果 Top{args.top_n} 里"非常规特征"占比高，说明最近新加的模长/Jerk/
    轴间相关/分频段能量/姿态角这些确实在起作用，不是白加；如果占比低、
    仍然是均值/标准差/RMS这些老特征霸榜，说明现有类别本身用常规特征
    就能分开，非常规特征的价值会在未来更细分的类别上才体现出来。
{'='*78}
""")


if __name__ == "__main__":
    main()
