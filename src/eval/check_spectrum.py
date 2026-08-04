"""
诊断脚本：检查真实IMU数据的频谱分布，评估16Hz采样率下是否存在混叠(aliasing)风险。

两种模式：

1. 已标注数据模式 --labeled_csv（推荐先跑这个，标签是真值，不受模型误判干扰）
   读取 loader_custom.py 同格式的合并CSV（dog_id/label/acc_x.../gyr_z 逐行标签），
   按 dog_id 内连续同标签片段切分，按标签分组统计频谱。

     python src/eval/check_spectrum.py \\
       --labeled_csv data/raw_custom/2026_7_30/merged_2026_7_30.csv --hz 16

2. 推理预测数据模式 --csv（看模型实际部署时输入信号的样子，按置信度桶自动分组）
   读取 extract_clips.py 产出的 clip CSV，按父目录名（clips_0.8-1.0 等）自动分组。

     python src/eval/check_spectrum.py \\
       --csv "infer_result/*/clips_*/*.csv" --hz 16

原理：
  奈奎斯特频率 = hz/2 = 8Hz（16Hz采样时）。真实生物运动信号如果本身
  能量集中在8Hz以内，频谱在接近8Hz处应该自然衰减。如果频谱在接近8Hz
  边缘处反而"抬升"，通常说明存在混叠——更高频率的真实运动成分被
  折叠回了这个频段，特征会失真。

  这个脚本没法给出100%确定的诊断（需要更高采样率的参考数据才能确证），
  但可以给出初步信号，尤其是"已标注数据"模式下按真值标签分组对比，
  如果某个类别（比如抓挠）的高频边缘能量占比系统性地高于其他类别，
  说明这不是纯噪声——要么是真实的高频动作特征（好事），要么需要找
  硬件同事确认抗混叠滤波（如果怀疑是伪影）。
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

ACC_CANDIDATES  = [["acc_x", "acc_y", "acc_z"], ["AccX", "AccY", "AccZ"], ["AX", "AY", "AZ"], ["ax", "ay", "az"]]
GYRO_CANDIDATES = [["gyro_x", "gyro_y", "gyro_z"], ["gyr_x", "gyr_y", "gyr_z"], ["GyroX", "GyroY", "GyroZ"], ["GX", "GY", "GZ"]]


def find_cols(cols, candidates):
    for g in candidates:
        if all(c in cols for c in g):
            return g
    return None


def load_signal_from_file(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    acc_cols = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    if acc_cols is None:
        return None
    return arrays_from_df(df, acc_cols, gyro_cols)


def arrays_from_df(df, acc_cols, gyro_cols):
    data = {}
    for name, col in zip(("acc_x", "acc_y", "acc_z"), acc_cols):
        data[name] = df[col].ffill().bfill().values.astype(np.float64)
    if gyro_cols:
        for name, col in zip(("gyr_x", "gyr_y", "gyr_z"), gyro_cols):
            data[name] = df[col].ffill().bfill().values.astype(np.float64)
    return data


def band_energy_report(x, hz, edge_frac=0.85):
    """返回 (freqs, psd_norm, edge_ratio)：edge_ratio 是高于 edge_frac*Nyquist 频段的能量占比"""
    nperseg = min(len(x), 256)
    if nperseg < 8:
        return None
    freqs, psd = signal.welch(x - x.mean(), fs=hz, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + 1e-12)
    nyq = hz / 2.0
    edge_mask = freqs >= edge_frac * nyq
    edge_ratio = float(psd_norm[edge_mask].sum())
    return freqs, psd_norm, edge_ratio


def summarize_signal(data, hz, edge_frac):
    """data: {channel_name: 1D array} -> {channel_name: {peak_freq, edge_ratio}}"""
    results = {}
    for ch_name, x in data.items():
        rep = band_energy_report(x, hz, edge_frac)
        if rep is None:
            continue
        freqs, psd_norm, edge_ratio = rep
        results[ch_name] = {"peak_freq": float(freqs[np.argmax(psd_norm)]), "edge_ratio": edge_ratio}
    return results


def print_group_summary(title, group_edge_ratios):
    """group_edge_ratios: {group_name: {ch_name: [edge_ratio,...]}}"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = "  分组".ljust(24) + "".join(ch.ljust(12) for ch in sorted({ch for g in group_edge_ratios.values() for ch in g}))
    print(header)
    for group_name in sorted(group_edge_ratios):
        chans = group_edge_ratios[group_name]
        row = f"  {group_name}".ljust(24)
        for ch in sorted({ch for g in group_edge_ratios.values() for ch in g}):
            if ch in chans and chans[ch]:
                mean_r = np.mean(chans[ch])
                row += f"{mean_r*100:5.1f}%      ".ljust(12)
            else:
                row += "-".ljust(12)
        print(row)
    print("  （数值 = 高频边缘能量占比，均值越高越值得关注混叠风险，也可能是真实高频动作）")


# ── 模式1：已标注数据 ──────────────────────────────────────────────────────

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


def run_labeled_mode(args):
    print(f"[标注数据模式] 读取 {args.labeled_csv} ...")
    df = pd.read_csv(args.labeled_csv)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    missing = [c for c in [args.dog_id_col, args.label_col] if c not in df.columns]
    if missing:
        print(f"[错误] CSV 缺少列: {missing}，现有列: {list(df.columns)}")
        return

    acc_cols = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    if acc_cols is None:
        print(f"[错误] 找不到加速度列，现有列: {list(df.columns)}")
        return

    min_len = max(8, int(args.hz * args.min_seg_s))
    group_edge_ratios = {}  # label -> {ch_name: [ratios]}
    n_segs_total = 0

    for dog_id, sub in df.groupby(args.dog_id_col):
        labels = sub[args.label_col].values
        segs = find_contiguous_segments(labels, min_len)
        for start, end, label in segs:
            seg_df = sub.iloc[start:end]
            data = arrays_from_df(seg_df, acc_cols, gyro_cols)
            results = summarize_signal(data, args.hz, args.edge_frac)
            for ch_name, r in results.items():
                group_edge_ratios.setdefault(label, {}).setdefault(ch_name, []).append(r["edge_ratio"])
            n_segs_total += 1

    if n_segs_total == 0:
        print(f"[警告] 没有找到长度 >= {min_len} 帧（{args.min_seg_s}秒）的连续同标签片段，"
              f"尝试调小 --min_seg_s")
        return

    print(f"共提取 {n_segs_total} 个连续同标签片段（每段 >= {args.min_seg_s}秒）")
    for label in sorted(group_edge_ratios):
        n = len(next(iter(group_edge_ratios[label].values())))
        print(f"  {label}: {n} 段")

    print_group_summary(
        f"按真值标签分组的频谱统计（采样率={args.hz}Hz, Nyquist={args.hz/2}Hz, "
        f"高频边缘阈值>{args.edge_frac*args.hz/2:.1f}Hz）",
        group_edge_ratios)


# ── 模式2：推理预测数据（clip CSV，按父目录名分组）──────────────────────────

def run_predicted_mode(args):
    files = []
    for pattern in args.csv:
        matched = glob.glob(pattern)
        files.extend(matched if matched else ([pattern] if os.path.exists(pattern) else []))
    files = sorted(set(files))

    if not files:
        print(f"[错误] 没有匹配到任何文件: {args.csv}")
        return

    print(f"{'='*70}")
    print(f"  推理预测数据频谱检查{'  [' + args.label + ']' if args.label else ''}")
    print(f"  采样率={args.hz}Hz  Nyquist={args.hz/2}Hz  高频边缘阈值={args.edge_frac*args.hz/2:.1f}Hz以上")
    print(f"  共 {len(files)} 个文件")
    print(f"{'='*70}")

    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)

    group_edge_ratios = {}  # 父目录名（置信度桶）-> {ch_name: [ratios]}

    for fpath in files:
        data = load_signal_from_file(fpath)
        if data is None:
            print(f"  [跳过] {os.path.basename(fpath)}: 无法解析")
            continue
        results = summarize_signal(data, args.hz, args.edge_frac)
        group_name = os.path.basename(os.path.dirname(fpath))  # clips_0.8-1.0 等
        fname = os.path.basename(fpath)
        print(f"\n  [{group_name}] {fname}")
        for ch_name, r in results.items():
            flag = " ⚠️" if r["edge_ratio"] > 0.15 else ""
            print(f"    {ch_name:8s}  主频={r['peak_freq']:.2f}Hz  "
                  f"高频边缘能量占比={r['edge_ratio']*100:.1f}%{flag}")
            group_edge_ratios.setdefault(group_name, {}).setdefault(ch_name, []).append(r["edge_ratio"])

        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(len(data), 1, figsize=(8, 2 * len(data)), sharex=True)
                if len(data) == 1:
                    axes = [axes]
                for ax, (ch_name, x) in zip(axes, data.items()):
                    freqs, psd = signal.welch(x - x.mean(), fs=args.hz, nperseg=min(len(x), 256))
                    ax.plot(freqs, psd)
                    ax.set_ylabel(ch_name)
                    ax.axvline(args.edge_frac * args.hz / 2, color="r", linestyle="--", linewidth=0.8)
                axes[-1].set_xlabel("Hz")
                fig.suptitle(f"{group_name}/{fname}")
                fig.tight_layout()
                out_png = os.path.join(args.plot_dir, f"{group_name}_{os.path.splitext(fname)[0]}.png")
                fig.savefig(out_png, dpi=100)
                plt.close(fig)
            except ImportError:
                print("    [警告] matplotlib 未安装，跳过画图（pip install matplotlib）")

    print_group_summary("按置信度桶（父目录名）分组的频谱统计", group_edge_ratios)


# ── 入口 ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="检查真实数据频谱，评估混叠风险")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--labeled_csv", default=None,
                       help="已标注数据模式：合并CSV路径（含 dog_id/label/acc_x.../gyr_z 逐行标签）")
    mode.add_argument("--csv", nargs="+", default=None,
                       help="推理预测数据模式：clip CSV 路径或 glob 模式（可多个）")
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--label", default="", help="推理模式下仅用于打印标题")
    ap.add_argument("--dog_id_col", default="dog_id", help="标注模式：狗ID列名")
    ap.add_argument("--label_col", default="label", help="标注模式：标签列名")
    ap.add_argument("--min_seg_s", type=float, default=2.0,
                     help="标注模式：连续同标签片段最短秒数，短于此长度的片段丢弃（默认2秒）")
    ap.add_argument("--edge_frac", type=float, default=0.85,
                     help="判定为'高频边缘'的阈值，占 Nyquist 的比例（默认0.85，即16Hz下 >6.8Hz）")
    ap.add_argument("--plot", action="store_true", help="推理模式：为每个文件画频谱图（需要 matplotlib）")
    ap.add_argument("--plot_dir", default="/tmp/spectrum_plots")
    args = ap.parse_args()

    if args.labeled_csv:
        run_labeled_mode(args)
    else:
        run_predicted_mode(args)

    print(f"""
{'='*70}
  怎么解读：
  - 高频边缘能量占比低（<15%左右）且组间稳定：混叠风险较小，现有
    16Hz+FFT特征可以放心使用。
  - 已标注数据模式下，如果"抓挠"组的占比系统性高于"活动"/"睡觉"组：
    说明这不是纯噪声，是有判别力的真实信号——除非怀疑是混叠伪影，
    否则这本身是好消息（分频段能量特征对分类有用）。
  - 推理预测数据模式下，如果低置信度桶（如 clips_0.3-0.4）的占比
    明显不同于高置信度桶：可能提示模型在这类边界样本上依赖了不该
    依赖的高频噪声特征，值得进一步排查误报样本。
{'='*70}
""")


if __name__ == "__main__":
    main()
