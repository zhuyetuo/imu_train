"""
数据集B加载器: Mendeley mpph6bmn7g
  42只辅助犬，ActiGraph GT9X，颈/背/胸，100Hz
  5类姿势: Standing / Sitting / Lying down / Walking / Body shake

列名在首次运行时自动探查并打印，如与实际不符请更新 NECK_COLS / LABEL_COL。
"""

import os
import glob
import pandas as pd
import numpy as np

# 项圈（Neck）传感器列 —— 待首次运行后确认
# 数据集B与A同用 ActiGraph GT9X，预期格式相同
NECK_COLS_CANDIDATES = [
    # 数据集A格式
    ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"],
    # 备选格式1
    ["Acc_Neck_X", "Acc_Neck_Y", "Acc_Neck_Z", "Gyr_Neck_X", "Gyr_Neck_Y", "Gyr_Neck_Z"],
    # 备选格式2
    ["neck_ax", "neck_ay", "neck_az", "neck_gx", "neck_gy", "neck_gz"],
]

LABEL_COL_CANDIDATES = ["posture", "Posture", "label", "Label", "behavior",
                         "Behavior", "Behavior_1", "activity", "Activity"]

DOG_ID_COL_CANDIDATES = ["DogID", "dog_id", "Dog", "dog", "ID", "id", "subject"]


def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_cols(df: pd.DataFrame, candidates_list: list) -> list:
    for candidates in candidates_list:
        if all(c in df.columns for c in candidates):
            return candidates
    return []


def load_dataset_b(csv_dir: str) -> tuple:
    """
    读取数据集B的 CSV 文件，按 DogID 列（或按文件）拆分。
    返回 (records, collar_cols, label_col)，格式与 loader_a 相同。
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"[loader_b] 在 {csv_dir} 下未找到 CSV 文件")

    print(f"[loader_b] 找到 {len(csv_files)} 个 CSV 文件")

    # 读第一个文件探查结构
    sample = pd.read_csv(csv_files[0], nrows=5)
    print(f"[loader_b] 列名: {list(sample.columns)}")

    neck_cols = _find_cols(sample, NECK_COLS_CANDIDATES)
    if not neck_cols:
        # 自动探测含 neck 的列
        neck_cols = [c for c in sample.columns if "neck" in c.lower()]
        if not neck_cols:
            # fallback: 所有数值列（去掉ID/时间/标签列）
            neck_cols = [c for c in sample.select_dtypes(include="number").columns
                         if not any(k in c.lower() for k in ["id", "time", "sec", "num"])]
        print(f"[loader_b] ⚠️ 未匹配预设列名，自动选择: {neck_cols}")
    else:
        print(f"[loader_b] 项圈传感器列: {neck_cols}")

    label_col = _find_col(sample, LABEL_COL_CANDIDATES)
    if label_col is None:
        raise ValueError(f"[loader_b] 未找到标签列，现有列: {list(sample.columns)}")
    print(f"[loader_b] 标签列: {label_col}")

    dog_id_col = _find_col(sample, DOG_ID_COL_CANDIDATES)

    records = []

    if dog_id_col and len(csv_files) == 1:
        # 单大文件，按 DogID 列拆分（同数据集A）
        df = pd.read_csv(csv_files[0])
        dog_ids = sorted(df[dog_id_col].unique())
        print(f"[loader_b] 按 {dog_id_col} 列拆分，共 {len(dog_ids)} 只狗")
        for dog_id in dog_ids:
            sub = df[df[dog_id_col] == dog_id]
            records.append({
                "dog_id": f"B_{dog_id}",   # 加前缀避免与数据集A的ID冲突
                "data": sub[neck_cols].values.astype(np.float32),
                "labels": sub[label_col].values,
            })
    else:
        # 多文件，每文件一只狗
        print(f"[loader_b] 每文件一只狗，共 {len(csv_files)} 只")
        for path in csv_files:
            df = pd.read_csv(path)
            dog_id = os.path.splitext(os.path.basename(path))[0]
            if len(neck_cols) > len(df.columns):
                continue
            records.append({
                "dog_id": f"B_{dog_id}",
                "data": df[neck_cols].values.astype(np.float32),
                "labels": df[label_col].values,
            })

    print(f"[loader_b] 加载完成: {len(records)} 只狗，{len(neck_cols)} 个传感器通道")
    return records, neck_cols, label_col
