"""
把多个Label Studio "YOLO with Images"导出目录合并成一份统一的YOLO检测
训练集（ultralytics格式：images/train,val + labels/train,val + data.yaml）。

为什么需要这一步，不能直接把两个project的导出目录简单拼一起：
  1. 两个project是分开建的，各自的classes.txt顺序不一定一样（哪怕两边
     标注时用的都是同一套"正常/异常"选项，Label Studio是按"这个project
     里第一次出现的标签顺序"生成classes.txt的，不保证两边一致）——直接
     合并会导致同一个class_id在两份数据里代表不同类别，模型会学到错误
     的东西且没有任何报错提示，是最容易被忽略的一个坑
  2. 两个project导出的文件名可能重复（都是"1.jpg"这种默认命名），直接
     合并到一个目录会互相覆盖，静默丢数据
  3. 需要切train/val，且要保证同一张图不会被简单地按文件名排序切分导致
     两个project的样本在train/val里分布不均衡（比如异常样本全被切进
     train，val里一个异常样本都没有，这样看着效果好实际没验证到位）

用法：
    python3 tooth_health/code/merge_labelstudio_yolo_exports.py \
        --exports data/raw_exports/normal_project data/raw_exports/abnormal_project \
        --out_dir tooth_health/data/yolo_dataset \
        --val_ratio 0.2 --seed 42
"""
import argparse
import random
import shutil
from pathlib import Path


def load_classes(export_dir: Path):
    classes_path = export_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(
            f"{export_dir} 下没有classes.txt——确认导出格式选的是"
            "'YOLO with Images'，且zip解压后保留了Label Studio给的原始目录结构")
    with open(classes_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_canonical_classes(per_export_classes: dict) -> list:
    """按"第一次出现"的顺序合并各project的类别列表，得到全局统一的class_id映射。"""
    canonical = []
    for classes in per_export_classes.values():
        for c in classes:
            if c not in canonical:
                canonical.append(c)
    return canonical


def remap_label_file(src_label: Path, dst_label: Path, old_classes: list, canonical: list):
    """把YOLO标注txt里的class_id从这个project自己的classes.txt顺序，
    重新映射成全局canonical classes列表里的顺序。"""
    remap = {old_idx: canonical.index(name) for old_idx, name in enumerate(old_classes)}
    lines_out = []
    with open(src_label, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            new_id = remap[old_id]
            lines_out.append(" ".join([str(new_id)] + parts[1:]))
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", nargs="+", required=True,
                    help="多个Label Studio 'YOLO with Images'导出目录（各自含images/labels/classes.txt）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    export_dirs = [Path(e) for e in args.exports]
    per_export_classes = {}
    for d in export_dirs:
        per_export_classes[d] = load_classes(d)
        print(f"{d.name}: classes.txt = {per_export_classes[d]}")

    canonical = build_canonical_classes(per_export_classes)
    print(f"\n合并后的全局类别顺序（class_id按这个顺序编号）: {canonical}")
    for d, classes in per_export_classes.items():
        if classes != canonical:
            print(f"  [提示] {d.name}的原始classes.txt顺序是{classes}，"
                  f"跟全局顺序不同，已自动重新映射class_id，不用手动处理")

    out_dir = Path(args.out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 收集所有(project, 图片文件)配对，按project加前缀避免文件名冲突，
    # 然后统一打乱切分——保证train/val里两个project的样本比例大致一致
    all_items = []  # (project_prefix, img_path, label_path)
    for d in export_dirs:
        img_dir, label_dir = d / "images", d / "labels"
        images = sorted(img_dir.iterdir())
        for img_path in images:
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"  [跳过] {img_path.name} 没有对应的标注文件 {label_path.name}")
                continue
            all_items.append((d.name, img_path, label_path))

    print(f"\n共找到 {len(all_items)} 张带标注的图片（{len(export_dirs)}个project合计）")

    rng = random.Random(args.seed)
    # 按project分层切分，保证每个project(正常为主/异常为主)在train/val里都有覆盖，
    # 不是整体打乱后随机切（那样小概率会把某个project的样本全切进一边）
    by_project = {}
    for item in all_items:
        by_project.setdefault(item[0], []).append(item)

    train_items, val_items = [], []
    for project, items in by_project.items():
        rng.shuffle(items)
        n_val = max(1, round(len(items) * args.val_ratio)) if len(items) > 1 else 0
        val_items.extend(items[:n_val])
        train_items.extend(items[n_val:])
        print(f"  {project}: {len(items)}张 -> train {len(items) - n_val} / val {n_val}")

    for split, items in (("train", train_items), ("val", val_items)):
        for project, img_path, label_path in items:
            new_stem = f"{project}__{img_path.stem}"
            dst_img = out_dir / "images" / split / (new_stem + img_path.suffix)
            dst_label = out_dir / "labels" / split / (new_stem + ".txt")
            shutil.copy2(img_path, dst_img)
            old_classes = per_export_classes[Path(
                [d for d in export_dirs if d.name == project][0]
            )]
            remap_label_file(label_path, dst_label, old_classes, canonical)

    data_yaml = out_dir / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"names:\n")
        for i, name in enumerate(canonical):
            f.write(f"  {i}: {name}\n")

    print(f"\n合并完成: train {len(train_items)}张, val {len(val_items)}张")
    print(f"数据集配置: {data_yaml}")


if __name__ == "__main__":
    main()
