"""
合并 YOLO 关键点 + LLM 行为标签，生成训练用 npz。

前置步骤:
  1. 训练 YOLO pose 模型（runs/pose/train/weights/best.pt）
  2. 用 LLM 标注行为标签（src/vision/label_with_llm.py）

用法:
  python src/vision/build_behavior_dataset.py
  python src/vision/build_behavior_dataset.py \
      --dataset_dir datasets/dog-pose \
      --labels_dir  data/processed/llm_labels \
      --output_dir  data/processed/behavior

输出:
  data/processed/behavior/train.npz
  data/processed/behavior/val.npz
  格式: X=(N, n_kp*2), y=(N,), classes, n_kp
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


BEHAVIORS = ["Lying", "Sitting", "Standing", "Walking", "Trotting", "Sniffing"]


def normalize_keypoints(kps: np.ndarray, n_kp: int) -> np.ndarray:
    """(n_kp, 3) → (n_kp*2,) 中心化+尺度归一化"""
    kps = kps[:n_kp]
    if len(kps) < n_kp:
        kps = np.vstack([kps, np.zeros((n_kp - len(kps), 3))])
    xy     = kps[:, :2].copy()
    center = xy.mean(axis=0)
    xy    -= center
    scale  = np.abs(xy).max() + 1e-6
    xy    /= scale
    return xy.flatten().astype(np.float32)


def load_labels_csv(csv_path: str) -> dict:
    """stem → behavior"""
    mapping = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            b = row["behavior"].strip()
            if b in BEHAVIORS:
                mapping[row["stem"]] = b
    return mapping


def load_split(dataset_dir: Path, split: str, labels_map: dict, n_kp: int):
    labels_dir = dataset_dir / split / "labels"
    if not labels_dir.exists():
        print(f"  [跳过] {labels_dir} 不存在")
        return np.empty((0, n_kp * 2), dtype=np.float32), np.empty((0,), dtype=np.int64)

    label_files = sorted(labels_dir.glob("*.txt"))
    X, y = [], []
    skipped = 0

    for lf in label_files:
        stem = lf.stem
        behavior = labels_map.get(stem)
        if behavior is None:
            skipped += 1
            continue

        class_idx = BEHAVIORS.index(behavior)

        with open(lf) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            skipped += 1
            continue

        # 取第一行（通常只有一只狗）
        parts = lines[0].split()
        kp_start = 5
        kps = []
        for i in range(n_kp):
            idx = kp_start + i * 3
            if idx + 2 < len(parts):
                kps.append([float(parts[idx]), float(parts[idx+1]), float(parts[idx+2])])
            else:
                kps.append([0.0, 0.0, 0.0])

        vec = normalize_keypoints(np.array(kps, dtype=np.float32), n_kp)
        X.append(vec)
        y.append(class_idx)

    print(f"  {split}: {len(X)} 条  跳过 {skipped} 条（无标签或空文件）")
    if not X:
        return np.empty((0, n_kp * 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="datasets/dog-pose")
    parser.add_argument("--labels_dir",  default="data/processed/llm_labels")
    parser.add_argument("--output_dir",  default="data/processed/behavior")
    parser.add_argument("--n_kp",        type=int, default=24)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    labels_dir  = Path(args.labels_dir)

    print(f"[build] 数据集: {dataset_dir}")
    print(f"[build] 标签目录: {labels_dir}")
    print(f"[build] 关键点数: {args.n_kp}")
    print(f"[build] 行为类别: {BEHAVIORS}")

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ("train", "val"):
        csv_path = labels_dir / f"{split}_labels.csv"
        if not csv_path.exists():
            print(f"  [跳过] {csv_path} 不存在，先运行 label_with_llm.py")
            continue

        labels_map = load_labels_csv(str(csv_path))
        print(f"\n[build] {split}: 加载 {len(labels_map)} 条 LLM 标签")

        X, y = load_split(dataset_dir, split, labels_map, args.n_kp)

        out_path = os.path.join(args.output_dir, f"{split}.npz")
        np.savez_compressed(out_path, X=X, y=y,
                            classes=np.array(BEHAVIORS),
                            n_kp=np.array(args.n_kp))
        print(f"  ✅ 保存 → {out_path}  shape={X.shape}")

        # 标签分布
        from collections import Counter
        counts = Counter(BEHAVIORS[i] for i in y)
        for b, c in counts.most_common():
            print(f"    {b:<12} {c:4d}  ({c/len(y):.1%})")

    print(f"\n[build] 完成！训练命令:")
    print(f"  python src/vision/train.py --mode mlp --npz_dir {args.output_dir}")


if __name__ == "__main__":
    main()
