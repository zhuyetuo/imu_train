"""
读取原始 CSV 文件，只提取项圈传感器列。
CSV 列名格式待首次运行 setup.sh 后根据实际输出确认。
"""

import os
import glob
import pandas as pd
import numpy as np


# 项圈传感器列名（运行 setup.sh 后根据实际列名更新）
# 格式参考: Acc_collar_X / Gyro_collar_X 或类似命名
COLLAR_COLS_PLACEHOLDER = None  # 由 detect_collar_cols() 自动探测

LABEL_COL = None  # 由 detect_label_col() 自动探测


def detect_collar_cols(df: pd.DataFrame) -> list[str]:
    """自动探测项圈传感器列（含 'collar' 关键字）。"""
    cols = [c for c in df.columns if 'collar' in c.lower()]
    if not cols:
        # fallback: 尝试常见命名
        cols = [c for c in df.columns if any(k in c.lower() for k in ['acc', 'gyro', 'accel'])]
    return cols


def detect_label_col(df: pd.DataFrame) -> str:
    """自动探测标签列。"""
    candidates = [c for c in df.columns if any(k in c.lower() for k in ['label', 'behavior', 'activity', 'class'])]
    if candidates:
        return candidates[0]
    raise ValueError(f"未找到标签列，现有列: {list(df.columns)}")


def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """读取单个 CSV 文件。"""
    df = pd.read_csv(csv_path)
    return df


def load_dataset_files(csv_dir: str, dog_info_path: str = None) -> list[dict]:
    """
    遍历 csv_dir 下所有 CSV，返回每条记录：
    {dog_id, df_collar, labels}
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"在 {csv_dir} 下未找到 CSV 文件")

    records = []
    collar_cols = None
    label_col = None

    for path in csv_files:
        df = load_raw_csv(path)

        # 首次运行时自动探测列名
        if collar_cols is None:
            collar_cols = detect_collar_cols(df)
            label_col = detect_label_col(df)
            print(f"[loader] 项圈传感器列: {collar_cols}")
            print(f"[loader] 标签列: {label_col}")

        dog_id = os.path.splitext(os.path.basename(path))[0]
        df_collar = df[collar_cols].copy()
        labels = df[label_col].values

        records.append({
            "dog_id": dog_id,
            "data": df_collar.values.astype(np.float32),
            "labels": labels,
            "collar_cols": collar_cols,
            "n_channels": len(collar_cols),
        })

    print(f"[loader] 加载完成: {len(records)} 个文件，{len(collar_cols)} 个传感器通道")
    return records, collar_cols, label_col
