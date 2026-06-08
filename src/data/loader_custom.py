"""
自采数据集加载器。

CSV 格式要求（列名在 configs/data.yaml custom 节配置）:
  dog_id  label  acc_x  acc_y  acc_z  gyr_x  gyr_y  gyr_z
  dog1    Walk   0.12   -0.03  9.81   0.01   0.02  -0.01
  ...

- dog_id : 每条狗的唯一标识，用于 leave-some-dogs-out 划分
- label  : 行为标签字符串
- acc/gyr: 项圈加速度计 + 陀螺仪，6通道，单位不限（训练时不做量纲统一）

多条狗的数据可以放在同一个 CSV 里，按 dog_id 列自动拆分。
"""

import pandas as pd
import numpy as np


def load_dataset_custom(csv_path: str, cfg: dict) -> tuple:
    """
    读取自采 CSV，按 dog_id 列拆分为每条狗的记录。

    cfg: configs/data.yaml 中 custom 节的内容
    返回 (records, sensor_cols, label_col)，格式与其他 loader 相同。
    """
    dog_id_col = cfg["dog_id_col"]
    label_col = cfg["label_col"]
    sensor_cols = cfg["sensor_cols"]

    print(f"[loader_custom] 读取 {csv_path} ...")
    df = pd.read_csv(csv_path)

    missing = [c for c in sensor_cols + [label_col, dog_id_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"[loader_custom] CSV 缺少列: {missing}\n"
            f"  现有列: {list(df.columns)}\n"
            f"  请检查 configs/data.yaml 的 custom.sensor_cols / dog_id_col / label_col 配置"
        )

    print(f"[loader_custom] 传感器列: {sensor_cols}")
    print(f"[loader_custom] 标签列: {label_col}  狗ID列: {dog_id_col}")

    dog_ids = sorted(df[dog_id_col].unique())
    print(f"[loader_custom] 共 {len(dog_ids)} 条狗")

    records = []
    for dog_id in dog_ids:
        sub = df[df[dog_id_col] == dog_id]
        records.append({
            "dog_id": str(dog_id),
            "data": sub[sensor_cols].values.astype(np.float32),
            "labels": sub[label_col].values,
        })

    print(f"[loader_custom] 加载完成: {len(records)} 条狗，{len(sensor_cols)} 个传感器通道")
    return records, sensor_cols, label_col
