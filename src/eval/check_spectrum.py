"""
诊断脚本：检查真实IMU数据的频谱分布，评估16Hz采样率下是否存在混叠(aliasing)风险。

用法:
  # 检查已裁剪的高置信度抓挠 clip（推荐先看这个）
  python src/eval/check_spectrum.py --csv infer_result/*/clips_0.8-1.0/*cam1_imu1*.csv --hz 16

  # 检查单个文件，并可选画图（需要 matplotlib）
  python src/eval/check_spectrum.py --csv path/to/file.csv --hz 16 --plot --plot_dir /tmp/spectrum_plots

  # 对比多个行为类型的 clip（放到不同目录，分别跑）
  python src/eval/check_spectrum.py --csv "infer_result/*/clips_0.8-1.0/*.csv" --hz 16 --label 高置信度抓挠

原理：
  奈奎斯特频率 = hz/2 = 8Hz（16Hz采样时）。真实生物运动信号如果本身
  能量集中在8Hz以内，频谱在接近8Hz处应该自然衰减（能量逐渐变小）。
  如果频谱在接近8Hz边缘处反而"抬升"或出现异常尖峰，通常说明存在
  混叠——更高频率的真实运动成分被折叠回了这个频段，特征会失真。

  这个脚本没法给出100%确定的诊断（需要更高采样率的参考数据才能确证），
  但可以给出一个初步信号：如果多数窗口在接近8Hz处能量占比很低、且
  平稳衰减，说明混叠风险较小，现有特征工程可以放心用；如果高频段
  能量占比异常高，值得进一步找硬件同事确认抗混叠滤波情况。
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../infer"))

ACC_CANDIDATES  = [["acc_x", "acc_y", "acc_z"], ["AccX", "AccY", "AccZ"], ["AX", "AY", "AZ"], ["ax", "ay", "az"]]
GYRO_CANDIDATES = [["gyro_x", "gyro_y", "gyro_z"], ["gyr_x", "gyr_y", "gyr_z"], ["GyroX", "GyroY", "GyroZ"], ["GX", "GY", "GZ"]]


def find_cols(cols, candidates):
    for g in candidates:
        if all(c in cols for c in g):
            return g
    return None


def load_signal(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    acc_cols = find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    if acc_cols is None:
        return None
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


def summarize_file(csv_path, hz, edge_frac):
    data = load_signal(csv_path)
    if data is None:
        return None
    results = {}
    for ch_name, x in data.items():
        rep = band_energy_report(x, hz, edge_frac)
        if rep is None:
            continue
        freqs, psd_norm, edge_ratio = rep
        peak_freq = freqs[np.argmax(psd_norm)]
        results[ch_name] = {"peak_freq": peak_freq, "edge_ratio": edge_ratio}
    return results


def main():
    ap = argparse.ArgumentParser(description="检查真实数据频谱，评估混叠风险")
    ap.add_argument("--csv", nargs="+", required=True, help="CSV 文件路径或 glob 模式（可多个）")
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--label", default="", help="仅用于打印标题，不影响计算")
    ap.add_argument("--edge_frac", type=float, default=0.85,
                     help="判定为'高频边缘'的阈值，占 Nyquist 的比例（默认0.85，即16Hz下 >6.8Hz）")
    ap.add_argument("--plot", action="store_true", help="为每个文件画频谱图（需要 matplotlib）")
    ap.add_argument("--plot_dir", default="/tmp/spectrum_plots")
    args = ap.parse_args()

    files = []
    for pattern in args.csv:
        matched = glob.glob(pattern)
        files.extend(matched if matched else ([pattern] if os.path.exists(pattern) else []))
    files = sorted(set(files))

    if not files:
        print(f"[错误] 没有匹配到任何文件: {args.csv}")
        return

    print(f"{'='*70}")
    print(f"  频谱混叠风险检查{'  [' + args.label + ']' if args.label else ''}")
    print(f"  采样率={args.hz}Hz  Nyquist={args.hz/2}Hz  高频边缘阈值={args.edge_frac*args.hz/2:.1f}Hz以上")
    print(f"  共 {len(files)} 个文件")
    print(f"{'='*70}")

    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)

    all_edge_ratios = {}  # ch_name -> list of edge_ratio across files

    for fpath in files:
        results = summarize_file(fpath, args.hz, args.edge_frac)
        if not results:
            print(f"  [跳过] {os.path.basename(fpath)}: 无法解析")
            continue
        fname = os.path.basename(fpath)
        print(f"\n  {fname}")
        for ch_name, r in results.items():
            flag = " ⚠️ 高频边缘能量偏高" if r["edge_ratio"] > 0.15 else ""
            print(f"    {ch_name:8s}  主频={r['peak_freq']:.2f}Hz  "
                  f"高频边缘能量占比={r['edge_ratio']*100:.1f}%{flag}")
            all_edge_ratios.setdefault(ch_name, []).append(r["edge_ratio"])

        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                data = load_signal(fpath)
                fig, axes = plt.subplots(len(data), 1, figsize=(8, 2 * len(data)), sharex=True)
                if len(data) == 1:
                    axes = [axes]
                for ax, (ch_name, x) in zip(axes, data.items()):
                    freqs, psd = signal.welch(x - x.mean(), fs=args.hz, nperseg=min(len(x), 256))
                    ax.plot(freqs, psd)
                    ax.set_ylabel(ch_name)
                    ax.axvline(args.edge_frac * args.hz / 2, color="r", linestyle="--", linewidth=0.8)
                axes[-1].set_xlabel("Hz")
                fig.suptitle(fname)
                fig.tight_layout()
                out_png = os.path.join(args.plot_dir, os.path.splitext(fname)[0] + ".png")
                fig.savefig(out_png, dpi=100)
                plt.close(fig)
            except ImportError:
                print("    [警告] matplotlib 未安装，跳过画图（pip install matplotlib）")

    print(f"\n{'='*70}")
    print("  汇总（跨所有文件取平均高频边缘能量占比）")
    print(f"{'='*70}")
    for ch_name, ratios in all_edge_ratios.items():
        mean_r = np.mean(ratios)
        flag = " ⚠️ 建议进一步检查抗混叠滤波" if mean_r > 0.15 else " ✅ 看起来正常"
        print(f"  {ch_name:8s}  平均高频边缘能量占比={mean_r*100:.1f}%{flag}")

    print(f"""
{'='*70}
  怎么解读：
  - 高频边缘能量占比低（<15%左右）且各文件间稳定：说明该轴的能量
    主要集中在中低频，混叠风险较小，现有16Hz+FFT特征可以放心使用。
  - 占比偏高、或个别文件突然飙升：可能是真实的高频动作（比如某些
    抓挠/甩头确实很快），也可能是混叠伪影，仅凭这个脚本无法100%
    区分两者——但如果这类文件里"抓挠"标注很多，且占比持续偏高，
    值得找硬件同事确认IMU芯片是否有抗混叠低通滤波。
  - 建议同时对比"抓挠" vs "活动" vs "睡觉" 三类clip各跑一遍，看看
    高频边缘能量占比在类别之间是否有系统性差异（如果抓挠明显更高，
    说明这个信息本身对分类是有用的，不一定是坏事，除非怀疑是纯噪声）。
{'='*70}
""")


if __name__ == "__main__":
    main()
