"""
数据集B加载器: Mendeley mpph6bmn7g
  42只辅助犬，ActiGraph GT9X，颈/背/胸，100Hz
  标签列: Position (standing/sitting/lying down/walking/body shake)
  狗ID列: Subject
"""

import os
import pandas as pd
import numpy as np

# 项圈传感器列（只取 Acc + Gyr，不含 Mag，与数据集A保持6通道一致）
COLLAR_COLS = [
    "Neck.Acc.X", "Neck.Acc.Y", "Neck.Acc.Z",
    "Neck.Gyr.X", "Neck.Gyr.Y", "Neck.Gyr.Z",
]

LABEL_COL = "Position"
DOG_ID_COL = "Subject"


def load_dataset_b(csv_path: str) -> tuple:
    """
    读取 df_raw.csv，按 Subject 列拆分为每只狗的记录。
    返回 (records, collar_cols, label_col)，格式与 loader_a 相同。
    """
    print(f"[loader_b] 读取 {csv_path} ...")
    df = pd.read_csv(csv_path)

    missing = [c for c in COLLAR_COLS + [LABEL_COL, DOG_ID_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"[loader_b] CSV 缺少列: {missing}，现有列: {list(df.columns)}")

    print(f"[loader_b] 项圈传感器列: {COLLAR_COLS}")
    print(f"[loader_b] 标签列: {LABEL_COL}")

    dog_ids = sorted(df[DOG_ID_COL].unique())
    print(f"[loader_b] 共 {len(dog_ids)} 只狗")

    records = []
    for dog_id in dog_ids:
        sub = df[df[DOG_ID_COL] == dog_id]
        records.append({
            "dog_id": f"B_{dog_id}",   # 加前缀避免与数据集A的ID冲突
            "data": sub[COLLAR_COLS].values.astype(np.float32),
            "labels": sub[LABEL_COL].values,
        })

    print(f"[loader_b] 加载完成: {len(records)} 只狗，{len(COLLAR_COLS)} 个传感器通道")
    return records, COLLAR_COLS, LABEL_COL
