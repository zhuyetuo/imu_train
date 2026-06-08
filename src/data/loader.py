"""
读取原始 CSV，按 DogID 列拆分为每条狗的记录，只保留项圈（Neck）传感器列。
实际列名: ANeck_x/y/z (加速度计), GNeck_x/y/z (陀螺仪)
"""

import os
import glob
import pandas as pd
import numpy as np

# 项圈传感器列（Neck = 项圈位置）
COLLAR_COLS = ["ANeck_x", "ANeck_y", "ANeck_z", "GNeck_x", "GNeck_y", "GNeck_z"]

# 标签列
LABEL_COL = "Behavior_1"

# 狗 ID 列
DOG_ID_COL = "DogID"


def load_dataset_files(csv_dir: str, dog_info_path: str = None) -> tuple:
    """
    读取大 CSV，按 DogID 拆分，返回每条狗的记录列表。
    每条记录: {dog_id, data (np.float32), labels (np.ndarray)}
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True))
    if not csv_files:
        csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"在 {csv_dir} 下未找到 CSV 文件")

    print(f"[loader] 读取 {csv_files[0]} ...")
    df = pd.read_csv(csv_files[0])

    # 验证必要列存在
    missing = [c for c in COLLAR_COLS + [LABEL_COL, DOG_ID_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}，现有列: {list(df.columns)}")

    print(f"[loader] 项圈传感器列: {COLLAR_COLS}")
    print(f"[loader] 标签列: {LABEL_COL}")

    # 按 DogID 拆分
    dog_ids = sorted(df[DOG_ID_COL].unique())
    print(f"[loader] 共 {len(dog_ids)} 条狗: {dog_ids[:5]}{'...' if len(dog_ids) > 5 else ''}")

    records = []
    for dog_id in dog_ids:
        sub = df[df[DOG_ID_COL] == dog_id]
        records.append({
            "dog_id": str(dog_id),
            "data": sub[COLLAR_COLS].values.astype(np.float32),
            "labels": sub[LABEL_COL].values,
        })

    print(f"[loader] 加载完成: {len(records)} 条狗，{len(COLLAR_COLS)} 个传感器通道")
    return records, COLLAR_COLS, LABEL_COL
