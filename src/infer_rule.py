"""
基于规则的 IMU 行为判断（活动 / 睡觉 / 抓挠）

不需要训练模型，直接用信号特征阈值判断。

用法:
  python src/infer_rule.py --input_url "http://192.168.2.140:8182/rec_wit.csv"
  python src/infer_rule.py --input_file data/infer/rec.csv
  python src/infer_rule.py --input_file data/infer/rec.csv --show_features  # 打印特征值帮助调阈值
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


# ── 默认阈值（可通过命令行覆盖）────────────────────────────────────────────
DEFAULTS = {
    # 睡觉判断：加速度合力标准差极低（几乎静止）
    "sleep_std_thresh":    0.08,   # m/s² 或 g，低于此 → 睡觉

    # 抓挠判断：在主频 3-6Hz 且标准差中等
    "scratch_freq_low":    2.5,    # Hz，抓挠主频下限
    "scratch_freq_high":   7.0,    # Hz，抓挠主频上限
    "scratch_freq_power":  0.25,   # 该频段能量占比下限（0-1）
    "scratch_std_low":     0.08,   # 标准差下限（排除睡觉）
    "scratch_std_high":    3.0,    # 标准差上限（排除剧烈运动）

    # 窗口参数
    "window_s":  2.0,    # 秒，窗口长度
    "stride_s":  1.0,    # 秒，步长
}


# ── 数据加载（复用 infer.py 的逻辑）─────────────────────────────────────────
ACC_COLS  = ["AX", "AY", "AZ"]
GYRO_COLS = ["GX", "GY", "GZ"]
ALT_ACC   = ["acc_x", "acc_y", "acc_z"]
ALT_GYRO  = ["gyro_x", "gyro_y", "gyro_z"]

BEHAVIOR_ZH = {"活动": "活动", "睡觉": "睡觉", "抓挠": "抓挠"}


def load_csv(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if all(c in df.columns for c in ACC_COLS):
        acc  = df[ACC_COLS].values.astype(np.float32)
        gyro = df[GYRO_COLS].values.astype(np.float32) if all(c in df.columns for c in GYRO_COLS) else None
    elif all(c in df.columns for c in ALT_ACC):
        acc  = df[ALT_ACC].values.astype(np.float32)
        gyro = df[ALT_GYRO].values.astype(np.float32) if all(c in df.columns for c in ALT_GYRO) else None
    else:
        raise ValueError(f"找不到加速度列，现有列: {list(df.columns)}")

    # 起始时间戳 + 采样率检测
    start_ts = None
    detected_hz = 50
    ts_col = next((c for c in df.columns if "time" in c.lower()), None)
    if ts_col:
        try:
            ts = pd.to_datetime(df[ts_col])
            start_ts = ts.iloc[0]
            unique_ts = ts.drop_duplicates().sort_values()
            interval = unique_ts.diff().dropna().dt.total_seconds().median()
            if interval > 0:
                detected_hz = max(1, round(1.0 / interval))
        except Exception:
            pass

    return acc, gyro, detected_hz, start_ts


def download_url(url: str) -> str:
    import tempfile, urllib.request
    from urllib.parse import urlparse
    fname = os.path.basename(urlparse(url).path) or "download.csv"
    suffix = ".csv" if url.lower().endswith(".csv") else ".txt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    print(f"[rule] 下载: {url}")
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name, fname


# ── 特征提取 ─────────────────────────────────────────────────────────────────
def extract_features(acc_win: np.ndarray, hz: int) -> dict:
    """
    acc_win: (N, 3)
    返回单窗口特征字典
    """
    mag = np.linalg.norm(acc_win, axis=1)       # 合力幅值

    acc_std  = float(mag.std())                  # 运动强度
    acc_mean = float(mag.mean())                 # 平均合力

    # FFT 找主频
    n = len(mag)
    fft_vals = np.abs(np.fft.rfft(mag - mag.mean()))
    freqs    = np.fft.rfftfreq(n, d=1.0 / hz)
    # 去掉 0Hz 直流分量
    fft_vals[0] = 0
    total_power = fft_vals.sum() + 1e-9

    # 抓挠频段能量占比
    scratch_mask  = (freqs >= DEFAULTS["scratch_freq_low"]) & (freqs <= DEFAULTS["scratch_freq_high"])
    scratch_power = float(fft_vals[scratch_mask].sum() / total_power)

    # 主频
    dominant_freq = float(freqs[fft_vals.argmax()]) if len(freqs) > 1 else 0.0

    return {
        "acc_std":        acc_std,
        "acc_mean":       acc_mean,
        "dominant_freq":  dominant_freq,
        "scratch_power":  scratch_power,
    }


# ── 规则判断 ─────────────────────────────────────────────────────────────────
def classify(feat: dict, cfg: dict) -> str:
    std   = feat["acc_std"]
    sp    = feat["scratch_power"]

    # 规则1：睡觉 — 几乎静止
    if std < cfg["sleep_std_thresh"]:
        return "睡觉"

    # 规则2：抓挠 — 3-6Hz 频段能量占主导，运动强度中等
    if (sp >= cfg["scratch_freq_power"]
            and cfg["scratch_std_low"] <= std <= cfg["scratch_std_high"]):
        return "抓挠"

    # 规则3：默认活动
    return "活动"


# ── 主流程 ───────────────────────────────────────────────────────────────────
def run(fpath: str, fname: str, cfg: dict, show_features: bool, output_dir: str):
    acc, gyro, hz, start_ts = load_csv(fpath)
    print(f"[rule] {fname}: {len(acc)} 行  采样率={hz}Hz  起始={start_ts}")

    window_n = max(4, int(cfg["window_s"] * hz))
    stride_n = max(1, int(cfg["stride_s"] * hz))

    results  = []
    feat_log = []

    i = 0
    while i + window_n <= len(acc):
        win   = acc[i:i + window_n]
        feat  = extract_features(win, hz)
        label = classify(feat, cfg)
        t_start = i / hz
        t_end   = t_start + cfg["stride_s"]

        row = {
            "file":         fname,
            "time_start_s": round(t_start, 3),
            "time_end_s":   round(t_end,   3),
            "prediction":   label,
            "prediction_zh": label,
        }
        if start_ts is not None:
            row["abs_start"] = (start_ts + pd.Timedelta(seconds=t_start)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            row["abs_end"]   = (start_ts + pd.Timedelta(seconds=t_end  )).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        results.append(row)
        if show_features:
            feat_log.append({**{"time_s": round(t_start, 2), "label": label}, **feat})

        i += stride_n

    if show_features:
        print("\n── 特征值明细 ──")
        feat_df = pd.DataFrame(feat_log)
        print(feat_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    # 合并连续相同行为
    merged = []
    for row in results:
        if not merged or row["prediction"] != merged[-1]["prediction"]:
            merged.append(dict(row))
        else:
            merged[-1]["time_end_s"] = row["time_end_s"]
            if "abs_end" in row:
                merged[-1]["abs_end"] = row["abs_end"]

    for seg in merged:
        seg["duration_s"] = round(seg["time_end_s"] - seg["time_start_s"], 1)

    # 统计
    from collections import Counter
    counts = Counter(r["prediction"] for r in results)
    total  = len(results)
    print(f"  {fname}: {total} 窗口  " +
          "  ".join(f"{k}:{v/total:.0%}" for k, v in counts.most_common()))

    print()
    for seg in merged:
        t = f"{seg['abs_start']} → {seg['abs_end']}" if "abs_start" in seg \
            else f"{seg['time_start_s']}s → {seg['time_end_s']}s"
        print(f"  {t}  {seg['prediction']}  {seg['duration_s']}秒")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(fname)[0]
    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, f"{stem}_rule.csv"), index=False)
    pd.DataFrame(merged).to_csv(
        os.path.join(output_dir, f"{stem}_rule_segments.csv"), index=False)
    print(f"\n[rule] 已保存至 {output_dir}/{stem}_rule*.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="")
    parser.add_argument("--input_url",  default="")
    parser.add_argument("--output_dir", default="results/infer")
    parser.add_argument("--show_features", action="store_true",
                        help="打印每窗口特征值，用于调整阈值")

    # 阈值覆盖
    parser.add_argument("--sleep_std",        type=float, default=DEFAULTS["sleep_std_thresh"],
                        help=f"睡觉判断：acc合力标准差阈值（默认{DEFAULTS['sleep_std_thresh']}）")
    parser.add_argument("--scratch_power",    type=float, default=DEFAULTS["scratch_freq_power"],
                        help=f"抓挠判断：3-6Hz频段能量占比阈值（默认{DEFAULTS['scratch_freq_power']}）")
    parser.add_argument("--scratch_std_high", type=float, default=DEFAULTS["scratch_std_high"],
                        help=f"抓挠判断：标准差上限（默认{DEFAULTS['scratch_std_high']}）")
    parser.add_argument("--window_s", type=float, default=DEFAULTS["window_s"],
                        help=f"窗口长度秒（默认{DEFAULTS['window_s']}）")
    parser.add_argument("--stride_s", type=float, default=DEFAULTS["stride_s"],
                        help=f"步长秒（默认{DEFAULTS['stride_s']}）")
    args = parser.parse_args()

    cfg = {**DEFAULTS,
           "sleep_std_thresh":  args.sleep_std,
           "scratch_freq_power": args.scratch_power,
           "scratch_std_high":  args.scratch_std_high,
           "window_s":          args.window_s,
           "stride_s":          args.stride_s}

    if args.input_url:
        fpath, fname = download_url(args.input_url)
    elif args.input_file:
        fpath, fname = args.input_file, os.path.basename(args.input_file)
    else:
        print("请指定 --input_file 或 --input_url")
        sys.exit(1)

    run(fpath, fname, cfg, args.show_features, args.output_dir)


if __name__ == "__main__":
    main()
