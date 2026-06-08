"""
统一数据加载接口，ML 和 DL 都从这里取数据。
"""

import os
import numpy as np


def load_dataset(hz: int, split: str, processed_dir: str = "data/processed"):
    """
    加载指定采样率和分割的数据。

    返回:
        X: (N, window_size, n_channels) float32
        y: (N,) int
        meta: dict，含 classes / n_channels / hz 等
    """
    path = os.path.join(processed_dir, f"{hz}hz", f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，请先运行 bash setup.sh")

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(np.int64)
    y_seq = data["y_seq"].astype(np.int64) if "y_seq" in data.files else None
    meta = {k: data[k].item() if data[k].ndim == 0 else data[k].tolist()
            for k in data.files if k not in ("X", "y", "y_seq")}
    return X, y, y_seq, meta


def load_all_splits(hz: int, processed_dir: str = "data/processed"):
    """一次性返回 train/val/test 三个分割。每个分割返回 (X, y, y_seq)。"""
    X_train, y_train, y_seq_train, meta = load_dataset(hz, "train", processed_dir)
    X_val,   y_val,   y_seq_val,   _    = load_dataset(hz, "val",   processed_dir)
    X_test,  y_test,  y_seq_test,  _    = load_dataset(hz, "test",  processed_dir)
    return (X_train, y_train, y_seq_train), (X_val, y_val, y_seq_val), (X_test, y_test, y_seq_test), meta
