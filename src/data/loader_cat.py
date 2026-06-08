"""
猫咪数据集加载器。

支持以下公开数据集（列名通过 configs/data.yaml cat 节配置）：

  cat_smit2023   — Smit et al. 2023, Sensors
    原始格式: R dataframe (.RDATA)，需先用 src/data/convert_cat_rdata.py 转为 CSV
    FigShare: https://figshare.com/articles/dataset/23605842
    12只猫，项圈+胸背带，三轴加速度计，24类行为，1秒 epoch

  cat_dunford2024 — Dunford et al. 2024, Ecology and Evolution
    原始格式: CSV（Dryad）
    Dryad DOI: 10.5061/dryad.q2bvq83sx
    9只猫，项圈，三轴加速度计，多类行为

  cat_smit2024   — Smit et al. 2024, PMC (家庭环境)
    原始格式: R dataframe (.RDATA)，需先转为 CSV
    FigShare: https://figshare.com/articles/dataset/24848292
    28只猫，项圈，三轴加速度计，8类行为

注意: 猫咪数据集通常只有加速度计（3通道），无陀螺仪。
本 loader 支持 3 或 6 通道，实际通道数由 sensor_cols 决定。
"""

import pandas as pd
import numpy as np


def load_dataset_cat(csv_path: str, cfg: dict) -> tuple:
    """
    读取猫咪 CSV，按 cat_id 列拆分为每只猫的记录。

    cfg: configs/data.yaml 中对应 cat 数据集节的内容
    返回 (records, sensor_cols, label_col)，格式与其他 loader 相同。
    """
    cat_id_col  = cfg.get("cat_id_col", "cat_id")
    label_col   = cfg.get("label_col", "behavior")
    sensor_cols = cfg.get("sensor_cols", ["acc_x", "acc_y", "acc_z"])

    print(f"[loader_cat] 读取 {csv_path} ...")
    df = pd.read_csv(csv_path)

    missing = [c for c in sensor_cols + [label_col, cat_id_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"[loader_cat] CSV 缺少列: {missing}\n"
            f"  现有列: {list(df.columns)}\n"
            f"  请检查 configs/data.yaml 对应 cat 节的 sensor_cols / cat_id_col / label_col"
        )

    print(f"[loader_cat] 传感器列 ({len(sensor_cols)}ch): {sensor_cols}")
    print(f"[loader_cat] 标签列: {label_col}  猫ID列: {cat_id_col}")

    # 去掉传感器列有缺失值的行
    before = len(df)
    df = df.dropna(subset=sensor_cols)
    if len(df) < before:
        print(f"[loader_cat] 丢弃 {before - len(df)} 行缺失值")

    cat_ids = sorted(df[cat_id_col].unique())
    print(f"[loader_cat] 共 {len(cat_ids)} 只猫")

    records = []
    for cat_id in cat_ids:
        sub = df[df[cat_id_col] == cat_id]
        records.append({
            "dog_id": f"cat_{cat_id}",      # 统一用 dog_id 字段，前缀 cat_ 避免与狗 ID 冲突
            "data":   sub[sensor_cols].values.astype(np.float32),
            "labels": sub[label_col].values,
        })

    print(f"[loader_cat] 加载完成: {len(records)} 只猫，{len(sensor_cols)} 个传感器通道")
    return records, sensor_cols, label_col
