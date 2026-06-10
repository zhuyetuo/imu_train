"""
将 dog-pose 数据集的 YOLO label 文件中的 class_id=0
替换为 LLM 标注的行为 class_id（0-5），
并生成新的 data.yaml，用于重新训练 YOLO26 行为识别模型。

用法:
  python src/vision/relabel_for_yolo.py
"""

import csv
import os
import shutil
from pathlib import Path

BEHAVIORS = ["Lying", "Sitting", "Standing", "Walking", "Trotting", "Sniffing"]
BEHAVIOR2ID = {b: i for i, b in enumerate(BEHAVIORS)}


def load_llm_labels(csv_path: str) -> dict:
    mapping = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            b = row["behavior"].strip()
            if b in BEHAVIOR2ID:
                mapping[row["stem"]] = BEHAVIOR2ID[b]
    return mapping


def relabel_split(src_labels_dir: Path, dst_labels_dir: Path, stem2id: dict):
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = 0
    for lf in sorted(src_labels_dir.glob("*.txt")):
        class_id = stem2id.get(lf.stem)
        if class_id is None:
            skip += 1
            continue
        lines = lf.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            parts[0] = str(class_id)   # 替换 class_id
            new_lines.append(" ".join(parts))
        (dst_labels_dir / lf.name).write_text("\n".join(new_lines) + "\n")
        ok += 1
    print(f"  {src_labels_dir.name}: {ok} 条已更新  跳过 {skip} 条（无LLM标签）")
    return ok


def main():
    src_dataset = Path("datasets/dog-pose")
    dst_dataset = Path("datasets/dog-pose-behavior")
    llm_dir     = Path("data/processed/llm_labels")

    print(f"[relabel] 源数据集: {src_dataset}")
    print(f"[relabel] 目标数据集: {dst_dataset}")
    print(f"[relabel] 行为类别: {BEHAVIORS}")

    # 复制图片（软链接，省磁盘）
    for split in ("train", "val"):
        src_img = src_dataset / "images" / split
        dst_img = dst_dataset / "images" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        for img in src_img.glob("*"):
            dst = dst_img / img.name
            if not dst.exists():
                dst.symlink_to(img.resolve())
    print("[relabel] 图片软链接已创建")

    # 重写 label 文件
    for split in ("train", "val"):
        csv_path = llm_dir / f"{split}_labels.csv"
        if not csv_path.exists():
            print(f"  [跳过] {csv_path} 不存在")
            continue
        stem2id = load_llm_labels(str(csv_path))
        relabel_split(
            src_dataset / "labels" / split,
            dst_dataset / "labels" / split,
            stem2id,
        )

    # 生成新的 data.yaml
    yaml_content = f"""path: {dst_dataset.resolve()}
train: images/train
val: images/val

nc: {len(BEHAVIORS)}
names: {BEHAVIORS}

kpt_shape: [24, 3]
"""
    yaml_path = dst_dataset / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\n[relabel] 完成！新数据集: {dst_dataset}")
    print(f"[relabel] data.yaml: {yaml_path}")
    print(f"\n[relabel] 训练命令:")
    print(f"  python train_dog_pose.py --model yolo26n-pose.pt --data {yaml_path} --batch 128")


if __name__ == "__main__":
    main()
