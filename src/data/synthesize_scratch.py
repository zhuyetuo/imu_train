"""
从 Label Studio JSON 标注中提取指定类别的真实片段，数据增强后生成合成数据。

不再硬编码时间段，直接读取当天导出的 JSON，自动提取任意类别的所有标注片段。

用法:
  # 合成抓挠数据（默认）
  python src/data/synthesize_scratch.py \
    --json data/raw_custom/2026_7_23/merged_tmp.json \
    --csv_dir data/raw_wit/ \
    --output data/synthetic/scratch_2026_7_23.npz \
    --label 抓挠 --hz 16 --n_aug 30

  # 合成睡觉数据（类别数据少时同理）
  python src/data/synthesize_scratch.py \
    --json data/raw_custom/2026_7_23/merged_tmp.json \
    --csv_dir data/raw_wit/ \
    --output data/synthetic/sleep_2026_7_23.npz \
    --label 睡觉 --hz 16 --n_aug 10

  # 验证生成数量
  python -c "import numpy as np; d=np.load('data/synthetic/scratch_2026_7_23.npz'); print(d['X'].shape)"
"""

import argparse
import json
import os
import urllib.request
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ACC_CANDIDATES  = [
    ["acc_x", "acc_y", "acc_z"],
    ["AccX",  "AccY",  "AccZ"],
    ["AX",    "AY",    "AZ"],
    ["ax",    "ay",    "az"],
    ["Ax",    "Ay",    "Az"],
]
GYRO_CANDIDATES = [
    ["gyro_x", "gyro_y", "gyro_z"],
    ["GyroX",  "GyroY",  "GyroZ"],
    ["GX",     "GY",     "GZ"],
    ["gx",     "gy",     "gz"],
    ["Gx",     "Gy",     "Gz"],
    ["wx",     "wy",     "wz"],
]
TS_KEYWORDS = ["time", "timestamp", "datetime", "date", "chip_time"]


def _find_cols(cols, candidates):
    for grp in candidates:
        if all(c in cols for c in grp):
            return grp
    return None


def _find_ts_col(cols):
    low = [c.lower() for c in cols]
    for kw in TS_KEYWORDS:
        for i, cl in enumerate(low):
            if kw in cl:
                return cols[i]
    return None


def _load_sensor(url, csv_dir, name):
    if not url:
        return None
    try:
        fname = os.path.basename(urlparse(url).path)
        if csv_dir:
            local = os.path.join(csv_dir, fname)
            if os.path.exists(local):
                df = pd.read_csv(local)
            else:
                tmp = f"/tmp/_syn_{fname}"
                urllib.request.urlretrieve(url, tmp)
                df = pd.read_csv(tmp)
        else:
            tmp = f"/tmp/_syn_{fname}"
            urllib.request.urlretrieve(url, tmp)
            df = pd.read_csv(tmp)
    except Exception as e:
        print(f"  [错误] {name} 加载失败: {e}")
        return None
    df.columns = [c.strip() for c in df.columns]
    acc_cols  = _find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = _find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    ts_col    = _find_ts_col(df.columns.tolist())
    if acc_cols is None or ts_col is None:
        print(f"  [错误] {name}: 找不到加速度列或时间戳列")
        return None
    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    return df, acc_cols, gyro_cols


def extract_segments_from_json(tasks, csv_dir, target_label, min_rows=16):
    """从 Label Studio JSON 中提取所有 target_label 的原始片段，返回 list of (N,6) ndarray。"""
    segments = []
    # 缓存已加载的 CSV，避免同一文件重复读取
    _csv_cache = {}

    for task in tasks:
        task_id = task["id"]
        data    = task.get("data", {})
        anns    = task.get("annotations", [])
        if not anns:
            continue

        is_multi = "csv1" in data or "csv2" in data

        if is_multi:
            sensor_map = {}
            for idx in ("1", "2"):
                url = data.get(f"csv{idx}", "")
                if not url:
                    continue
                if url not in _csv_cache:
                    _csv_cache[url] = _load_sensor(url, csv_dir, f"task{task_id}_imu{idx}")
                if _csv_cache[url]:
                    sensor_map[f"label{idx}"] = _csv_cache[url]
        else:
            url = data.get("csv", "")
            if not url:
                continue
            if url not in _csv_cache:
                _csv_cache[url] = _load_sensor(url, csv_dir, f"task{task_id}_imu")
            sensor_map = {"label": _csv_cache[url]} if _csv_cache[url] else {}

        for ann in anns:
            for seg in ann.get("result", []):
                val    = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                t0_str = val.get("start", "")
                t1_str = val.get("end",   "")
                fn     = seg.get("from_name", "")
                if not labels or labels[0] != target_label or not t0_str or not t1_str:
                    continue
                if fn not in sensor_map:
                    continue
                df, acc_cols, gyro_cols = sensor_map[fn]
                t0   = pd.to_datetime(t0_str)
                t1   = pd.to_datetime(t1_str)
                mask = (df["_ts"] >= t0) & (df["_ts"] <= t1)
                sub  = df[mask]
                if len(sub) < min_rows:
                    print(f"  [跳过] task{task_id} {t0_str} 只有 {len(sub)} 行")
                    continue
                acc  = sub[acc_cols].values.astype(np.float32)
                gyro = sub[gyro_cols].values.astype(np.float32) if gyro_cols \
                       else np.zeros((len(sub), 3), dtype=np.float32)
                segments.append(np.concatenate([acc, gyro], axis=1))
                print(f"  task{task_id} [{fn}] {target_label}: {t0_str} → {t1_str}  ({len(sub)} 行)")

    return segments


# ── 增强函数 ──────────────────────────────────────────────────────────────────

def aug_noise(seg, scale=0.02):
    return seg + np.random.randn(*seg.shape).astype(np.float32) * scale


def aug_scale(seg, low=0.85, high=1.15):
    s = np.random.uniform(low, high, size=(1, seg.shape[1])).astype(np.float32)
    return seg * s


def aug_flip_axis(seg):
    out  = seg.copy()
    axis = np.random.randint(0, 3)
    out[:, axis]     *= -1
    out[:, axis + 3] *= -1
    return out


def aug_time_shift(seg, max_frac=0.1):
    shift = np.random.randint(1, max(2, int(len(seg) * max_frac)))
    return np.roll(seg, shift, axis=0)


def aug_time_stretch(seg, low=0.8, high=1.2):
    from scipy.signal import resample
    factor    = np.random.uniform(low, high)
    new_len   = max(4, int(len(seg) * factor))
    stretched = resample(seg, new_len, axis=0).astype(np.float32)
    return resample(stretched, len(seg), axis=0).astype(np.float32)


def augment_segment(seg, n_aug, rng):
    aug_fns  = [aug_noise, aug_scale, aug_flip_axis, aug_time_shift]
    variants = []
    for _ in range(n_aug):
        out    = seg.copy()
        chosen = rng.choice(len(aug_fns), size=rng.integers(1, 4), replace=False)
        for i in chosen:
            try:
                out = aug_fns[i](out)
            except Exception:
                pass
        if rng.random() < 0.2:
            out = aug_time_stretch(out)
        variants.append(out)
    return variants


def sliding_windows(data, window_size, stride):
    wins = []
    for start in range(0, len(data) - window_size + 1, stride):
        wins.append(data[start:start + window_size])
    return wins


def main():
    parser = argparse.ArgumentParser(description="从 Label Studio JSON 生成指定类别的合成数据")
    parser.add_argument("--json",     required=True, help="Label Studio 导出的 JSON（支持 merged_tmp.json）")
    parser.add_argument("--csv_dir",  default="",    help="本地 CSV 目录（不填则从 URL 下载）")
    parser.add_argument("--output",   required=True, help="输出 .npz 路径")
    parser.add_argument("--label",    default="抓挠", help="要合成的类别名称（默认：抓挠）")
    parser.add_argument("--hz",       type=int,   default=16,  help="采样率（默认 16）")
    parser.add_argument("--window_s", type=float, default=2.0, help="窗口秒数（默认 2.0）")
    parser.add_argument("--stride_s", type=float, default=1.0, help="步长秒数（默认 1.0）")
    parser.add_argument("--n_aug",    type=int,   default=30,  help="每个原始片段生成的增强数量（默认 30）")
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    rng         = np.random.default_rng(args.seed)
    window_size = int(args.window_s * args.hz)
    stride      = int(args.stride_s * args.hz)
    print(f"目标类别='{args.label}'  窗口={window_size}点  步长={stride}点  采样率={args.hz}Hz")

    with open(args.json, encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"\n加载 JSON: {len(tasks)} 个 task")

    print(f"\n── 提取 '{args.label}' 片段 ──")
    segments = extract_segments_from_json(tasks, args.csv_dir, args.label)
    print(f"\n共提取 {len(segments)} 个原始片段")
    if not segments:
        print("[错误] 未找到任何片段，请检查 --label 名称和 JSON 内容")
        return

    raw_windows = []
    for seg in segments:
        raw_windows.extend(sliding_windows(seg, window_size, stride))
    print(f"原始片段滑窗: {len(raw_windows)} 个窗口")

    aug_windows = []
    for seg in segments:
        for v in augment_segment(seg, args.n_aug, rng):
            aug_windows.extend(sliding_windows(v, window_size, stride))

    all_windows = raw_windows + aug_windows
    print(f"增强后总窗口: {len(all_windows)} 个")

    if not all_windows:
        print("[错误] 没有生成任何窗口")
        return

    X = np.stack(all_windows, axis=0)
    print(f"\n输出形状: {X.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, X=X)
    print(f"已保存: {args.output}")

    print(f"\n下一步（注入合成数据训练）:")
    print(f"  python src/ml/train.py --hz {args.hz} --model rf \\")
    print(f"    --processed_dir <your_processed_dir> \\")
    print(f"    --remap configs/remap_custom_3class.yaml \\")
    print(f"    --synthetic {args.output} --synthetic_label '{args.label}'")


if __name__ == "__main__":
    main()
