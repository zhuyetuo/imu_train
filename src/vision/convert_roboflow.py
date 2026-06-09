"""
将 Roboflow 下载的 YOLOv8-Pose 格式数据集转换为训练用 npz。

YOLOv8-Pose 标签格式（每行）:
  class_id  cx cy w h  x1 y1 v1  x2 y2 v2  ...  xN yN vN
  坐标均归一化到 [0,1]，v 是可见度/置信度。

用法:
  python src/data/convert_roboflow.py \
    --dataset_dir animal-pose-xhq7l-1 \
    --output_dir data/processed/roboflow

转换后生成:
  data/processed/roboflow/train.npz   (X, y, classes)
  data/processed/roboflow/val.npz
  data/processed/roboflow/test.npz
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.dataset import kp_to_vector


def parse_label_file(label_path: str, n_kp: int) -> list[tuple[int, np.ndarray]]:
    """
    解析单个 YOLO Pose 标签文件，返回 [(class_id, kp_vector), ...]。
    一个文件可能有多只狗，取置信度最高的那只。
    """
    results = []
    if not os.path.exists(label_path):
        return results
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            # parts[1:5] = cx cy w h，之后每3个是 x y v
            kp_start = 5
            kps = []
            for i in range(n_kp):
                idx = kp_start + i * 3
                if idx + 2 < len(parts):
                    x, y, v = float(parts[idx]), float(parts[idx+1]), float(parts[idx+2])
                    kps.append([x, y, v])
                else:
                    kps.append([0.0, 0.0, 0.0])
            results.append((class_id, np.array(kps, dtype=np.float32)))
    return results


def kp_normalized_to_vector(kps: np.ndarray, n_kp: int) -> np.ndarray:
    """
    输入: (N, 3) 归一化关键点 [x, y, v]（坐标已在0-1范围）
    输出: (n_kp * 2,) 中心化+尺度归一化特征向量
    """
    kps = kps[:n_kp]
    if len(kps) < n_kp:
        pad = np.zeros((n_kp - len(kps), 3), dtype=np.float32)
        kps = np.vstack([kps, pad])
    xy = kps[:, :2]
    center = xy.mean(axis=0)
    xy = xy - center
    scale = np.abs(xy).max() + 1e-6
    xy = xy / scale
    return xy.flatten()


def load_split(dataset_dir: str, split: str, classes: list[str],
               n_kp: int) -> tuple[np.ndarray, np.ndarray]:
    split_dir = os.path.join(dataset_dir, split)
    if not os.path.isdir(split_dir):
        print(f"  [跳过] {split} 目录不存在: {split_dir}")
        return np.empty((0, n_kp * 2), dtype=np.float32), np.empty((0,), dtype=np.int64)

    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    if not os.path.isdir(images_dir):
        print(f"  [跳过] images 目录不存在: {images_dir}")
        return np.empty((0, n_kp * 2), dtype=np.float32), np.empty((0,), dtype=np.int64)

    img_files = [f for f in os.listdir(images_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    X, y = [], []
    skipped = 0
    for img_file in img_files:
        stem = os.path.splitext(img_file)[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        records = parse_label_file(label_path, n_kp)
        if not records:
            skipped += 1
            continue
        # 每张图取第一个（通常只有一只狗）
        class_id, kps = records[0]
        if class_id >= len(classes):
            skipped += 1
            continue
        vec = kp_normalized_to_vector(kps, n_kp)
        X.append(vec)
        y.append(class_id)

    print(f"  {split}: {len(X)} 条  (跳过 {skipped} 条无标注)")
    if not X:
        return np.empty((0, n_kp * 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main(args):
    # 读取 data.yaml 获取类别名和关键点数
    yaml_path = os.path.join(args.dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[convert] 找不到 {yaml_path}")
        sys.exit(1)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    classes = cfg.get("names", [])
    n_kp = cfg.get("kpt_shape", [17, 3])[0]

    print(f"[convert] 数据集: {args.dataset_dir}")
    print(f"[convert] 类别 ({len(classes)}): {classes}")
    print(f"[convert] 关键点数: {n_kp}")

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ("train", "valid", "test"):
        X, y = load_split(args.dataset_dir, split, classes, n_kp)
        out_split = "val" if split == "valid" else split
        out_path = os.path.join(args.output_dir, f"{out_split}.npz")
        np.savez_compressed(out_path, X=X, y=y,
                            classes=np.array(classes),
                            n_kp=np.array(n_kp))
        print(f"  ✅ 保存 → {out_path}  shape={X.shape}")

    print(f"\n[convert] 完成！输出目录: {args.output_dir}")
    print(f"  训练命令:")
    print(f"  python src/train.py --mode mlp --npz_dir {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="Roboflow 下载目录，如 animal-pose-xhq7l-1")
    parser.add_argument("--output_dir", default="data/processed/roboflow",
                        help="输出 npz 目录（默认 data/processed/roboflow）")
    main(parser.parse_args())
