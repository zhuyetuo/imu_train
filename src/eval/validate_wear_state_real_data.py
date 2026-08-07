"""
用真实录制的 CSV 验证 wear_state.py 的两个检测器在真机上到底有没有信号可用。

合成数据（gen_synthetic_wear_state_scenarios.py）只证明了"检测逻辑本身没写错"，
回答不了"WitMotion/HICC 真机上心跳/呼吸微振动能不能被测到""厂商自带的降噪滤波
会不会已经把信号磨掉了"这些问题——这些必须拿真实录制的数据跑一遍才知道，
见 docs/wear_state_detection.md §4/§5.3。

用法（每个 --group 是一批同类场景的录制文件，label 自己起名）：

    python src/eval/validate_wear_state_real_data.py \
      --group worn_sleep:"data/real_wear_test/sleep/*.csv" \
      --group off_body_static:"data/real_wear_test/static_table/*.csv" \
      --group off_body_charging:"data/real_wear_test/charging/*.csv" \
      --group baseline_normal:"data/raw_wit/*.csv" \
      --mode bio --hz 50

    python src/eval/validate_wear_state_real_data.py \
      --group loose_various:"data/real_wear_test/loose_walk/*.csv" \
      --group baseline_normal:"data/raw_wit/*imu1*.csv" \
      --mode loose --hz 50

--hz 是设备采样率（witmotion常见25/50/100Hz，HICC自研设备看实际情况），必须传对，
不传的话会尝试从时间戳列估算，估算不出来就报错，不会瞎猜。

--downsample_to 16 可以同时看一遍降到16Hz（最终成品采样率）之后信号还在不在，
直接回答"未佩戴检测到底需不需要原始高采样率数据"这个开放问题。
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # for infer_csv_scratch (src/)

from wear_state import (  # noqa: E402
    bio_signal_energy, is_macro_still, tilt_drift,
    MACRO_STILL_WINDOW_SEC, GAIT_SUBSEG_SEC,
    BIO_SIGNAL_ENERGY_THRESHOLD, LOOSE_WOBBLE_STD_THRESHOLD_RAD,
)
from infer_csv_scratch import load_csv, downsample  # noqa: E402


def estimate_hz(ts):
    if ts is None or ts.isna().all():
        return None
    dt = ts.diff().dropna().dt.total_seconds()
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    return round(1.0 / dt.median())


def iter_windows(acc, hz, window_sec):
    n = int(window_sec * hz)
    if n <= 0:
        return
    for start in range(0, len(acc) - n + 1, n):
        yield acc[start:start + n]


def run_bio_mode(files, hz, downsample_to=None):
    rows = []
    for path in files:
        acc, _, ts, valid_mask, null_ratio = load_csv(path)
        file_hz = hz or estimate_hz(ts)
        if file_hz is None:
            print(f"  [跳过] {path}: 传不了 --hz 也估不出采样率")
            continue
        if downsample_to and downsample_to != file_hz:
            acc = downsample(acc, file_hz, downsample_to)
            use_hz = downsample_to
        else:
            use_hz = file_hz
        n_still = n_total = 0
        for w in iter_windows(acc, use_hz, MACRO_STILL_WINDOW_SEC):
            n_total += 1
            if is_macro_still(w, use_hz):
                n_still += 1
                rows.append({"file": os.path.basename(path), "hz": use_hz,
                             "bio_energy": bio_signal_energy(w, use_hz)})
        if n_total:
            print(f"  {os.path.basename(path)}: {n_total}个{MACRO_STILL_WINDOW_SEC:.0f}s窗口，"
                  f"{n_still}个宏观静止（{n_still/n_total*100:.0f}%），hz={use_hz}"
                  + (f"，缺失率={null_ratio*100:.1f}%" if null_ratio > 0.05 else ""))
    return pd.DataFrame(rows)


def run_loose_mode(files, hz, downsample_to=None):
    rows = []
    for path in files:
        acc, _, ts, valid_mask, null_ratio = load_csv(path)
        file_hz = hz or estimate_hz(ts)
        if file_hz is None:
            print(f"  [跳过] {path}: 传不了 --hz 也估不出采样率")
            continue
        if downsample_to and downsample_to != file_hz:
            acc = downsample(acc, file_hz, downsample_to)
            use_hz = downsample_to
        else:
            use_hz = file_hz
        for w in iter_windows(acc, use_hz, GAIT_SUBSEG_SEC * 3):  # 每3个子段一个窗口
            drift = tilt_drift(w, use_hz)
            rows.append({"file": os.path.basename(path), "hz": use_hz, "tilt_drift": drift})
        if null_ratio > 0.05:
            print(f"  [警告] {os.path.basename(path)} 缺失率={null_ratio*100:.1f}%")
    return pd.DataFrame(rows)


def summarize(df, col, threshold, threshold_name):
    if df.empty:
        print("  (无有效窗口)")
        return
    q = df[col].quantile([0.1, 0.5, 0.9]).to_dict()
    above = (df[col] > threshold).mean()
    print(f"  n={len(df)}  p10={q[0.1]:.6g}  p50={q[0.5]:.6g}  p90={q[0.9]:.6g}  "
          f"max={df[col].max():.6g}  超过当前阈值({threshold_name}={threshold:g})的比例={above*100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True,
                    help='label:glob模式，可重复传多个，如 worn_sleep:"data/x/*.csv"')
    ap.add_argument("--mode", choices=["bio", "loose"], required=True,
                    help="bio=验证未佩戴/静置检测；loose=验证松动检测")
    ap.add_argument("--hz", type=int, default=None, help="设备采样率，不传则尝试从时间戳估算")
    ap.add_argument("--downsample_to", type=int, default=None,
                    help="额外跑一遍降到这个采样率后的结果（比如16，对应最终成品采样率）")
    ap.add_argument("--out_csv", default=None, help="把逐窗口结果存成CSV，方便自己画图看分布")
    args = ap.parse_args()

    groups = {}
    for g in args.group:
        label, pattern = g.split(":", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[警告] group '{label}' 的模式 '{pattern}' 没匹配到任何文件")
        groups[label] = files

    run_fn = run_bio_mode if args.mode == "bio" else run_loose_mode
    col = "bio_energy" if args.mode == "bio" else "tilt_drift"
    threshold = BIO_SIGNAL_ENERGY_THRESHOLD if args.mode == "bio" else LOOSE_WOBBLE_STD_THRESHOLD_RAD
    threshold_name = "BIO_SIGNAL_ENERGY_THRESHOLD" if args.mode == "bio" else "LOOSE_WOBBLE_STD_THRESHOLD_RAD"

    all_dfs = []
    for label, files in groups.items():
        print(f"\n=== {label}（{len(files)}个文件） ===")
        df = run_fn(files, args.hz, args.downsample_to)
        df["group"] = label
        all_dfs.append(df)
        print(f"[{label}] {col} 分布：")
        summarize(df, col, threshold, threshold_name)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    print("\n" + "=" * 70)
    print("组间对比（能不能看出清晰分界，是这次验证最重要的结论）：")
    if not combined.empty:
        print(combined.groupby("group")[col].describe(percentiles=[0.1, 0.5, 0.9]).to_string())

    if args.out_csv and not combined.empty:
        combined.to_csv(args.out_csv, index=False)
        print(f"\n逐窗口结果已保存到 {args.out_csv}，可以自己拉出来画直方图看分布是否重叠")


if __name__ == "__main__":
    main()
