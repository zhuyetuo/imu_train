"""
将 Label Studio 导出的 IMU 标注 JSON 转换为训练用 CSV。

Label Studio 导出格式（时间序列标注）:
  每个 task 对应一个 CSV 文件（data.csv 字段），
  annotations 里包含多个时间段 + 标签（timeserieslabels）。

输出格式（与 loader_custom.py 兼容）:
  subject_id, label, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

用法:
  # CSV 从 URL 下载（需要设备在线）
  python src/data/labelstudio_to_custom.py \\
    --json data/raw_custom/labelstudio_export.json \\
    --output data/raw_custom/data.csv

  # CSV 在本地目录（文件名与 URL 里的文件名一致）
  python src/data/labelstudio_to_custom.py \\
    --json data/raw_custom/labelstudio_export.json \\
    --csv_dir data/raw_wit/ \\
    --output data/raw_custom/data.csv

  # WitMotion acc 单位是 g，转换成 m/s²
  python src/data/labelstudio_to_custom.py \\
    --json data/raw_custom/labelstudio_export.json \\
    --output data/raw_custom/data.csv \\
    --acc_unit g
"""

import argparse
import json
import os
import sys
import urllib.request
from urllib.parse import urlparse

import numpy as np
import pandas as pd

G = 9.81

# ── 自动检测列名 ──────────────────────────────────────────────────────────────
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


def _find_cols(df_cols, candidates):
    for grp in candidates:
        if all(c in df_cols for c in grp):
            return grp
    return None


def _find_ts_col(df_cols):
    low = [c.lower() for c in df_cols]
    for kw in TS_KEYWORDS:
        for i, cl in enumerate(low):
            if kw in cl:
                return df_cols[i]
    return None


# ── CSV 加载 ──────────────────────────────────────────────────────────────────
def load_csv(path_or_url: str, csv_dir: str) -> pd.DataFrame:
    """从本地目录或 URL 加载 CSV。"""
    fname = os.path.basename(urlparse(path_or_url).path)

    if csv_dir:
        local = os.path.join(csv_dir, fname)
        if os.path.exists(local):
            print(f"  [local] {local}")
            return pd.read_csv(local)

    print(f"  [download] {path_or_url}")
    tmp = f"/tmp/_ls_imu_{fname}"
    urllib.request.urlretrieve(path_or_url, tmp)
    return pd.read_csv(tmp)


# ── 时间戳解析 ────────────────────────────────────────────────────────────────
def parse_ts(df: pd.DataFrame, ts_col: str) -> pd.Series:
    """把时间戳列解析为 datetime，容忍多种格式。"""
    return pd.to_datetime(df[ts_col], errors="coerce")


# ── 主转换逻辑 ────────────────────────────────────────────────────────────────
def convert(tasks: list, csv_dir: str, acc_unit: str) -> pd.DataFrame:
    rows = []

    for task in tasks:
        task_id   = task["id"]
        csv_url   = task["data"].get("csv", "")
        if not csv_url:
            print(f"[跳过] task {task_id}：无 csv 字段")
            continue

        annotations = task.get("annotations", [])
        if not annotations:
            print(f"[跳过] task {task_id}：无标注")
            continue

        print(f"\n[task {task_id}] {os.path.basename(csv_url)}")

        try:
            df = load_csv(csv_url, csv_dir)
        except Exception as e:
            print(f"  [错误] 无法加载 CSV: {e}")
            continue

        df.columns = [c.strip() for c in df.columns]
        acc_cols  = _find_cols(df.columns.tolist(), ACC_CANDIDATES)
        gyro_cols = _find_cols(df.columns.tolist(), GYRO_CANDIDATES)
        ts_col    = _find_ts_col(df.columns.tolist())

        if acc_cols is None:
            print(f"  [错误] 找不到加速度列，现有: {list(df.columns)}")
            continue
        if ts_col is None:
            print(f"  [错误] 找不到时间戳列，现有: {list(df.columns)}")
            continue

        df["_ts"] = parse_ts(df, ts_col)
        if df["_ts"].isna().all():
            print(f"  [错误] 时间戳解析失败，列: {ts_col}，样本值: {df[ts_col].iloc[0]}")
            continue

        print(f"  acc={acc_cols}  gyro={gyro_cols}  ts={ts_col}")
        print(f"  行数={len(df)}  时间范围: {df['_ts'].min()} → {df['_ts'].max()}")

        subject_id = f"task{task_id}"

        for ann in annotations:
            for seg in ann.get("result", []):
                val    = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                t_start_str = val.get("start", "")
                t_end_str   = val.get("end",   "")
                if not labels or not t_start_str or not t_end_str:
                    continue

                label   = labels[0]
                t_start = pd.to_datetime(t_start_str)
                t_end   = pd.to_datetime(t_end_str)

                mask = (df["_ts"] >= t_start) & (df["_ts"] <= t_end)
                seg_df = df[mask]

                if len(seg_df) == 0:
                    print(f"  [警告] 标注段 {label} {t_start_str}→{t_end_str} 没有匹配到任何行")
                    continue

                print(f"  {label}: {t_start_str} → {t_end_str}  ({len(seg_df)} 行)")

                acc = seg_df[acc_cols].values.astype(np.float64)
                if acc_unit == "g":
                    acc = acc * G

                if gyro_cols:
                    gyro = seg_df[gyro_cols].values.astype(np.float64)
                else:
                    gyro = np.zeros((len(seg_df), 3))

                for i in range(len(seg_df)):
                    rows.append({
                        "dog_id": subject_id,
                        "label":  label,
                        "acc_x":  acc[i, 0],
                        "acc_y":  acc[i, 1],
                        "acc_z":  acc[i, 2],
                        "gyr_x":  gyro[i, 0],
                        "gyr_y":  gyro[i, 1],
                        "gyr_z":  gyro[i, 2],
                    })

    if not rows:
        raise RuntimeError("没有任何有效数据，请检查输入 JSON 和 CSV 路径")

    out = pd.DataFrame(rows)
    return out


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Label Studio IMU 标注 → 训练 CSV")
    parser.add_argument("--json",    required=True, help="Label Studio 导出的 JSON 文件")
    parser.add_argument("--output",  required=True, help="输出 CSV 路径（如 data/raw_custom/data.csv）")
    parser.add_argument("--csv_dir", default="",   help="本地 CSV 目录（不填则从 URL 下载）")
    parser.add_argument("--acc_unit", default="ms2", choices=["ms2", "g"],
                        help="加速度单位：ms2=m/s²（默认），g=重力单位（自动×9.81）")
    args = parser.parse_args()

    with open(args.json, encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"共 {len(tasks)} 个 task")

    out_df = convert(tasks, args.csv_dir, args.acc_unit)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"\n── 汇总 ──")
    print(f"总行数: {len(out_df)}")
    print(f"subject 数: {out_df['dog_id'].nunique()}")
    counts = out_df["label"].value_counts()
    for label, cnt in counts.items():
        print(f"  {label}: {cnt} 行")
    print(f"\n已保存: {args.output}")
    print(f"\n下一步（预处理）:")
    print(f"  bash setup.sh --dataset custom")
    print(f"下一步（训练）:")
    print(f"  python src/ml/train.py --hz 50 --model rf --processed_dir data/processed_custom")


if __name__ == "__main__":
    main()
