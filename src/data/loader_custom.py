"""
自采数据集加载器。

CSV 格式要求（列名在 configs/data.yaml custom 节配置）:
  record_id      label   acc_x  acc_y  acc_z  gyr_x  gyr_y  gyr_z
  task496_imu1   Walk    0.12   -0.03  9.81   0.01   0.02  -0.01
  ...

- record_id: 每次录制+每路传感器的唯一标识，用于 leave-some-out 划分。
  注意：这不是真实的狗ID，是"task编号+传感器路数"拼出来的分组键
  （比如 task496_imu1），一次录制/一路传感器算一个record_id，跟实际
  有多少条不同的狗没有直接对应关系。
- label  : 行为标签字符串
- acc/gyr: 项圈加速度计 + 陀螺仪，6通道，单位不限（训练时不做量纲统一）

多条记录的数据可以放在同一个 CSV 里，按 record_id 列自动拆分。
"""

import pandas as pd
import numpy as np


def load_dataset_custom(csv_path: str, cfg: dict) -> tuple:
    """
    读取自采 CSV，按 record_id 列拆分为每条记录。

    cfg: configs/data.yaml 中 custom 节的内容
    返回 (records, sensor_cols, label_col)，格式与其他 loader 相同。
    """
    record_id_col = cfg["record_id_col"]
    label_col = cfg["label_col"]
    sensor_cols = cfg["sensor_cols"]

    print(f"[loader_custom] 读取 {csv_path} ...")
    # 省内存地读：只读要用的列，传感器 float32，标签 / 记录 ID 用 category。
    # 默认读法下几百小时的数据（上亿行）光字符串列就要几十 GB——每行一个
    # Python 字符串对象（2026-10-29 训练 #15 在上一步就被 OOM 杀了，这一步同样会）
    head = pd.read_csv(csv_path, nrows=0).columns
    use = [c for c in dict.fromkeys(sensor_cols + [label_col, record_id_col]) if c in head]
    dtype = {c: "float32" for c in sensor_cols if c in head}
    dtype.update({label_col: "category", record_id_col: "category"})
    df = pd.read_csv(csv_path, usecols=use, dtype=dtype)

    missing = [c for c in sensor_cols + [label_col, record_id_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"[loader_custom] CSV 缺少列: {missing}\n"
            f"  现有列: {list(df.columns)}\n"
            f"  请检查 configs/data.yaml 的 custom.sensor_cols / record_id_col / label_col 配置"
        )

    print(f"[loader_custom] 传感器列: {sensor_cols}")
    print(f"[loader_custom] 标签列: {label_col}  记录ID列: {record_id_col}")

    # groupby 一遍拆完，不再逐个记录在整张表上比一次（上千条记录 × 上亿行）。
    # sort=False + 下面按 str 排：顺序跟以前 sorted(unique()) 一致
    groups = {str(k): g for k, g in df.groupby(record_id_col, sort=False, observed=True)}
    record_ids = sorted(groups)
    print(f"[loader_custom] 共 {len(record_ids)} 条记录（每条记录=一次录制的一路传感器，"
          f"不等于实际狗的数量）")

    records = []
    for record_id in record_ids:
        sub = groups[record_id]
        records.append({
            "record_id": record_id,
            "data": sub[sensor_cols].values.astype(np.float32),
            # 还原成普通字符串数组：下游按字符串比较 / 编码
            "labels": np.asarray(sub[label_col].astype(str), dtype=object),
        })
    del groups, df

    print(f"[loader_custom] 加载完成: {len(records)} 条记录，{len(sensor_cols)} 个传感器通道")
    return records, sensor_cols, label_col
