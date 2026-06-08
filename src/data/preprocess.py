"""
预处理流程：
  单数据集: 原始 100Hz CSV → 降采样 → 窗口切分 → 保存 .npz
  合并模式: 数据集A + 数据集B → 标签统一 → 合并 → 保存 .npz
"""

import os
import argparse
import numpy as np
import yaml
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from loader import load_dataset_files
from label_map import LABEL_MAP_A, LABEL_MAP_B, UNIFIED_LABELS, apply_map


def downsample(data, labels, source_hz, target_hz):
    step = source_hz // target_hz
    return data[::step], labels[::step]


def sliding_window(data, labels, window_size, stride, keep_label_set=None):
    X, y = [], []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        majority = Counter(labels[start:end]).most_common(1)[0][0]
        if keep_label_set is not None and majority not in keep_label_set:
            continue
        X.append(data[start:end])
        y.append(majority)
    if not X:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32), np.empty((0,))
    return np.array(X, dtype=np.float32), np.array(y)


def split_by_dog(records, train_r, val_r, seed):
    rng = np.random.default_rng(seed)
    dog_ids = [r["dog_id"] for r in records]
    rng.shuffle(dog_ids)
    n = len(dog_ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return (set(dog_ids[:n_train]),
            set(dog_ids[n_train:n_train + n_val]),
            set(dog_ids[n_train + n_val:]))


def process_split(records, dog_ids_set, window_size, stride, le, keep_label_set=None):
    X_all, y_all = [], []
    valid_encoded = set(le.transform(list(keep_label_set))) if keep_label_set else None
    for r in records:
        if r["dog_id"] not in dog_ids_set:
            continue
        data, labels = r["data"], r["labels"]
        mask = np.isin(labels, list(keep_label_set)) if keep_label_set else np.ones(len(labels), bool)
        labels_enc = np.full(len(labels), -1, dtype=np.int64)
        labels_enc[mask] = le.transform(labels[mask])
        X, y = sliding_window(data, labels_enc, window_size, stride, valid_encoded)
        if len(X) == 0:
            continue
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(X_all), np.concatenate(y_all)


def load_merged_records(cfg, args):
    """加载并合并数据集A和B，统一标签到 UNIFIED_LABELS。"""
    from loader_b import load_dataset_b

    print(f"[preprocess] 模式: 合并数据集A + B")

    # 数据集A
    records_a, _, _ = load_dataset_files(args.raw_csv_dir, args.dog_info)
    for r in records_a:
        r["labels"] = apply_map(r["labels"], LABEL_MAP_A)

    # 数据集B
    records_b, _, _ = load_dataset_b(args.raw_csv_b)
    for r in records_b:
        r["labels"] = apply_map(r["labels"], LABEL_MAP_B)

    records = records_a + records_b
    print(f"[preprocess] 合并后: {len(records_a)} + {len(records_b)} = {len(records)} 只狗")
    return records, UNIFIED_LABELS


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    source_hz = cfg["source_hz"]
    target_hz_list = cfg["target_hz_list"]
    window_sec = cfg["window_seconds"]
    stride_sec = cfg["stride_seconds"]
    seed = cfg["seed"]
    train_r = cfg["train_ratio"]
    val_r = cfg["val_ratio"]

    # 决定加载模式
    merge_mode = args.raw_csv_b and os.path.exists(args.raw_csv_b)

    if merge_mode:
        records, keep_labels_list = load_merged_records(cfg, args)
        keep_label_set = set(keep_labels_list)
    else:
        print(f"[preprocess] 模式: 单数据集A")
        records, collar_cols, _ = load_dataset_files(args.raw_csv_dir, args.dog_info)
        keep_labels_list = cfg.get("keep_labels", None)
        keep_label_set = set(keep_labels_list) if keep_labels_list else None
        if keep_label_set:
            print(f"[preprocess] 过滤标签，只保留: {keep_labels_list}")

    # 拟合标签编码器
    if keep_label_set:
        all_labels = np.concatenate([
            r["labels"][np.isin(r["labels"], list(keep_label_set))] for r in records
        ])
    else:
        all_labels = np.concatenate([r["labels"] for r in records])

    # 去掉 None（合并模式下未映射的标签）
    all_labels = all_labels[all_labels != np.array(None)]

    le = LabelEncoder()
    le.fit(all_labels)
    classes = list(le.classes_)
    print(f"[preprocess] 行为类别 ({len(classes)}): {classes}")

    train_ids, val_ids, test_ids = split_by_dog(records, train_r, val_r, seed)
    print(f"[preprocess] 狗 ID 划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    suffix = "_merged" if merge_mode else ""

    for target_hz in target_hz_list:
        step = source_hz // target_hz
        window_size = int(window_sec * target_hz)
        stride = int(stride_sec * target_hz)

        print(f"\n[preprocess] 处理 {target_hz}Hz ...")

        ds_records = []
        for r in records:
            data_ds, labels_ds = downsample(r["data"], r["labels"], source_hz, target_hz)
            ds_records.append({**r, "data": data_ds, "labels": labels_ds})

        X_train, y_train = process_split(ds_records, train_ids, window_size, stride, le, keep_label_set)
        X_val, y_val = process_split(ds_records, val_ids, window_size, stride, le, keep_label_set)
        X_test, y_test = process_split(ds_records, test_ids, window_size, stride, le, keep_label_set)

        print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

        out_dir = os.path.join(args.output_dir + suffix, f"{target_hz}hz")
        os.makedirs(out_dir, exist_ok=True)

        meta = {
            "hz": target_hz, "window_size": window_size, "stride": stride,
            "n_channels": str(X_train.shape[2] if X_train.ndim == 3 else 6),
            "classes": str(classes),
            "train_dog_ids": str(list(train_ids)),
            "val_dog_ids": str(list(val_ids)),
            "test_dog_ids": str(list(test_ids)),
            "mode": "merged" if merge_mode else "single",
        }

        np.savez_compressed(os.path.join(out_dir, "train.npz"), X=X_train, y=y_train, **meta)
        np.savez_compressed(os.path.join(out_dir, "val.npz"), X=X_val, y=y_val, **meta)
        np.savez_compressed(os.path.join(out_dir, "test.npz"), X=X_test, y=y_test, **meta)
        print(f"  ✅ 保存至 {out_dir}/")

    print("\n[preprocess] 全部完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv_dir", default="data/raw/csv")
    parser.add_argument("--dog_info", default="data/raw/DogInfo.csv")
    parser.add_argument("--raw_csv_b", default="data/raw_b/df_raw.csv",
                        help="数据集B的 df_raw.csv 路径，不传则只用数据集A")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--config", default="configs/data.yaml")
    main(parser.parse_args())
