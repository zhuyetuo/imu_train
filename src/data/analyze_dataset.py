"""
分析预处理后的数据集（.npz），展示 train/val/test 类别分布。

用法:
  python src/data/analyze_dataset.py --processed_dir data/processed_custom --hz 16
"""

import argparse
import os
import numpy as np
from collections import Counter


def load_split(path):
    if not os.path.exists(path):
        return None, None
    d = np.load(path, allow_pickle=True)
    y = d["y"]
    classes = eval(str(d["classes"])) if "classes" in d else []
    return y, classes


def print_split(name, y, classes):
    if y is None:
        print(f"  {name}: 文件不存在")
        return
    total = len(y)
    cnt = Counter(y.tolist())
    print(f"  {name} ({total} 窗口):")
    for cls_id in sorted(cnt):
        label = classes[int(cls_id)] if classes and int(cls_id) < len(classes) else str(cls_id)
        n   = cnt[cls_id]
        pct = n / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<8} {n:>5} ({pct:>5.1f}%)  {bar}")
    if total > 0:
        counts = list(cnt.values())
        ratio  = max(counts) / min(counts) if min(counts) > 0 else float("inf")
        if ratio > 5:
            min_id    = min(cnt, key=cnt.get)
            min_label = classes[int(min_id)] if classes else str(min_id)
            print(f"    ⚠️  最多/最少={ratio:.1f}x，'{min_label}' 样本偏少")
        elif ratio > 2:
            print(f"    ⚠️  轻微不均衡（最多/最少={ratio:.1f}x），训练时已自动加权")
        else:
            print(f"    ✅  较均衡（最多/最少={ratio:.1f}x）")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", required=True,
                        help="预处理输出目录，如 data/processed_custom")
    parser.add_argument("--hz", type=int, default=0,
                        help="指定采样率，0=自动扫描所有可用 Hz")
    args = parser.parse_args()

    base = args.processed_dir
    if not os.path.isdir(base):
        print(f"[错误] 目录不存在: {base}")
        return

    # 找所有 hz 子目录
    if args.hz:
        hz_dirs = [f"{args.hz}hz"]
    else:
        hz_dirs = sorted(
            d for d in os.listdir(base)
            if d.endswith("hz") and os.path.isdir(os.path.join(base, d))
        )

    if not hz_dirs:
        print(f"[错误] 找不到任何 *hz/ 子目录: {base}")
        return

    for hz_dir in hz_dirs:
        dir_path = os.path.join(base, hz_dir)
        print(f"\n{'='*52}")
        print(f"  {hz_dir}  —  {dir_path}")
        print(f"{'='*52}")

        y_tr, classes = load_split(os.path.join(dir_path, "train.npz"))
        y_va, _       = load_split(os.path.join(dir_path, "val.npz"))
        y_te, _       = load_split(os.path.join(dir_path, "test.npz"))

        if classes:
            print(f"  类别: {classes}")

        print_split("train", y_tr, classes)
        print_split("val",   y_va, classes)
        print_split("test",  y_te, classes)

        # 总量
        totals = [y for y in [y_tr, y_va, y_te] if y is not None]
        if totals:
            all_y = np.concatenate(totals)
            total = len(all_y)
            cnt   = Counter(all_y.tolist())
            print(f"  合计 {total} 窗口: " +
                  "  ".join(f"{classes[int(k)] if classes else k}={v}" for k, v in sorted(cnt.items())))
    print()


if __name__ == "__main__":
    main()
