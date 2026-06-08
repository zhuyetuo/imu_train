"""
预处理流程：
  原始 100Hz CSV → 降采样 → 窗口切分 → 按狗划分 train/val/test → 保存 .npz
"""

import os
import argparse
import numpy as np
import yaml
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from loader import load_dataset_files


def downsample(data: np.ndarray, labels: np.ndarray, source_hz: int, target_hz: int):
    """按固定步长降采样（整数比）。"""
    step = source_hz // target_hz
    return data[::step], labels[::step]


def sliding_window(data: np.ndarray, labels: np.ndarray, window_size: int, stride: int):
    """
    滑动窗口切分。
    返回 X: (N, window_size, n_channels)，y: (N,) 取窗口内众数标签
    """
    X, y = [], []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]
        window_labels = labels[start:end]
        majority = Counter(window_labels).most_common(1)[0][0]
        X.append(window)
        y.append(majority)
    return np.array(X, dtype=np.float32), np.array(y)


def split_by_dog(records: list, train_r: float, val_r: float, seed: int):
    """按狗 ID 划分，避免同一条狗同时出现在训练和测试集。"""
    rng = np.random.default_rng(seed)
    dog_ids = [r["dog_id"] for r in records]
    rng.shuffle(dog_ids)
    n = len(dog_ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    train_ids = set(dog_ids[:n_train])
    val_ids = set(dog_ids[n_train:n_train + n_val])
    test_ids = set(dog_ids[n_train + n_val:])
    return train_ids, val_ids, test_ids


def process_split(records, dog_ids_set, window_size, stride, le):
    X_all, y_all = [], []
    for r in records:
        if r["dog_id"] not in dog_ids_set:
            continue
        data, labels = r["data"], r["labels"]
        # 编码标签
        labels_enc = le.transform(labels)
        X, y = sliding_window(data, labels_enc, window_size, stride)
        X_all.append(X)
        y_all.append(y)
    if not X_all:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(X_all), np.concatenate(y_all)


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

    print(f"[preprocess] 加载原始数据: {args.raw_csv_dir}")
    records, collar_cols, label_col = load_dataset_files(args.raw_csv_dir, args.dog_info)

    # 拟合标签编码器
    all_labels = np.concatenate([r["labels"] for r in records])
    le = LabelEncoder()
    le.fit(all_labels)
    classes = list(le.classes_)
    print(f"[preprocess] 行为类别 ({len(classes)}): {classes}")

    # 按狗 ID 划分
    train_ids, val_ids, test_ids = split_by_dog(records, train_r, val_r, seed)
    print(f"[preprocess] 狗 ID 划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    for target_hz in target_hz_list:
        step = source_hz // target_hz
        window_size = int(window_sec * target_hz)
        stride = int(stride_sec * target_hz)

        print(f"\n[preprocess] 处理 {target_hz}Hz (step={step}, window={window_size}pts, stride={stride}pts)")

        # 降采样所有记录
        ds_records = []
        for r in records:
            data_ds, labels_ds = downsample(r["data"], r["labels"], source_hz, target_hz)
            ds_records.append({**r, "data": data_ds, "labels": labels_ds})

        # 切窗口
        X_train, y_train = process_split(ds_records, train_ids, window_size, stride, le)
        X_val, y_val = process_split(ds_records, val_ids, window_size, stride, le)
        X_test, y_test = process_split(ds_records, test_ids, window_size, stride, le)

        print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

        # 保存
        out_dir = os.path.join(args.output_dir, f"{target_hz}hz")
        os.makedirs(out_dir, exist_ok=True)

        meta = {
            "hz": target_hz,
            "window_size": window_size,
            "stride": stride,
            "n_channels": len(collar_cols),
            "collar_cols": collar_cols,
            "classes": classes,
            "train_dog_ids": list(train_ids),
            "val_dog_ids": list(val_ids),
            "test_dog_ids": list(test_ids),
        }

        np.savez_compressed(os.path.join(out_dir, "train.npz"), X=X_train, y=y_train, **{k: str(v) for k, v in meta.items()})
        np.savez_compressed(os.path.join(out_dir, "val.npz"), X=X_val, y=y_val, **{k: str(v) for k, v in meta.items()})
        np.savez_compressed(os.path.join(out_dir, "test.npz"), X=X_test, y=y_test, **{k: str(v) for k, v in meta.items()})

        print(f"  ✅ 保存至 {out_dir}/")

    print("\n[preprocess] 全部完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv_dir", required=True)
    parser.add_argument("--dog_info", default=None)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--config", default="configs/data.yaml")
    main(parser.parse_args())
