"""
从姿态 JSON + 标注 CSV 构建行为分类数据集。

标注 CSV 格式:
  start_s,end_s,label
  0.0,5.2,Lying chest
  5.2,8.7,Standing

支持两种模式:
  - single_frame: 每帧独立预测（MLP）
  - sequence:     滑窗序列预测（LSTM / Transformer）
"""

import json
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def load_pose_json(path: str) -> tuple[float, list[dict]]:
    """返回 (fps, frames)。每帧结构: {frame, time_s, dogs: [{keypoints, ...}]}"""
    with open(path) as f:
        d = json.load(f)
    return d["fps"], d["frames"]


def kp_to_vector(keypoints: list, n_kp: int = 17) -> np.ndarray:
    """
    将关键点列表 (N, 3) → 归一化特征向量 (n_kp * 2,)。
    只取 x, y，归一化到 bbox 范围，忽略置信度。
    """
    kp = np.array(keypoints, dtype=np.float32)   # (N, 3)
    if len(kp) == 0:
        return np.zeros(n_kp * 2, dtype=np.float32)
    kp = kp[:n_kp]
    if len(kp) < n_kp:
        pad = np.zeros((n_kp - len(kp), 3), dtype=np.float32)
        kp = np.vstack([kp, pad])
    xy = kp[:, :2]
    # 中心化 + 尺度归一化（用最大距离）
    center = xy.mean(axis=0)
    xy = xy - center
    scale = np.abs(xy).max() + 1e-6
    xy = xy / scale
    return xy.flatten()


def build_windows(frames: list[dict], labels_df: pd.DataFrame,
                  window_size: int, stride: int, fps: float,
                  n_kp: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """
    对一条视频的帧序列做滑窗，返回 (X, y)。
    X shape: (N, window_size, n_kp*2)  或  (N, n_kp*2) if window_size==1
    """
    # 时间戳 → 标签映射
    def get_label(t: float) -> str | None:
        for _, row in labels_df.iterrows():
            if row["start_s"] <= t < row["end_s"]:
                return row["label"]
        return None

    # 每帧提取特征向量
    vecs = []
    times = []
    for frm in frames:
        t = frm["time_s"]
        dogs = frm.get("dogs", [])
        if dogs:
            kp = max(dogs, key=lambda d: d.get("confidence", 0))["keypoints"]
        else:
            kp = []
        vecs.append(kp_to_vector(kp, n_kp))
        times.append(t)

    vecs = np.array(vecs, dtype=np.float32)
    times = np.array(times, dtype=np.float32)

    X, y = [], []
    for start in range(0, len(vecs) - window_size + 1, stride):
        end = start + window_size
        window_vecs = vecs[start:end]
        # 用窗口中点时间查标签
        mid_t = float(times[start + window_size // 2])
        label = get_label(mid_t)
        if label is None:
            continue
        X.append(window_vecs if window_size > 1 else window_vecs[0])
        y.append(label)

    if not X:
        feat_dim = n_kp * 2
        shape = (0, window_size, feat_dim) if window_size > 1 else (0, feat_dim)
        return np.empty(shape, dtype=np.float32), np.array([])
    return np.array(X, dtype=np.float32), np.array(y)


class DogBehaviorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, classes: list[str]):
        self.X = X
        # y 可能已经是整数索引（来自 npz），也可能是字符串标签
        if y.dtype.kind in ("i", "u"):
            self.y = y.astype(np.int64)
        else:
            self.y = np.array([classes.index(c) for c in y], dtype=np.int64)
        self.classes = classes

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


def load_all_sessions(pose_dir: str, annot_dir: str, classes: list[str],
                      window_size: int, stride: int,
                      split: str = "train", train_ratio: float = 0.7,
                      val_ratio: float = 0.15, seed: int = 42,
                      n_kp: int = 17) -> tuple[np.ndarray, np.ndarray]:
    """
    扫描 pose_dir 下所有 *_pose.json，匹配 annot_dir 下同名 *_labels.csv，
    按 session 划分 train/val/test，返回 (X, y)。
    """
    pose_files = sorted([f for f in os.listdir(pose_dir) if f.endswith("_pose.json")])
    rng = np.random.default_rng(seed)
    rng.shuffle(pose_files)
    n = len(pose_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": pose_files[:n_train],
        "val":   pose_files[n_train:n_train + n_val],
        "test":  pose_files[n_train + n_val:],
    }
    selected = splits.get(split, [])

    all_X, all_y = [], []
    for pf in selected:
        stem = pf.replace("_pose.json", "")
        annot_path = os.path.join(annot_dir, f"{stem}_labels.csv")
        if not os.path.exists(annot_path):
            print(f"  [跳过] 找不到标注: {annot_path}")
            continue
        fps, frames = load_pose_json(os.path.join(pose_dir, pf))
        labels_df = pd.read_csv(annot_path)
        X, y = build_windows(frames, labels_df, window_size, stride, fps, n_kp)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)

    if not all_X:
        return np.empty((0,), dtype=np.float32), np.array([])
    return np.concatenate(all_X), np.concatenate(all_y)
