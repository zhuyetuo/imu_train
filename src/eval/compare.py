"""
汇总所有 hz × 模型的结果，输出对比表格。
用法:
  python src/eval/compare.py                          # 单数据集
  python src/eval/compare.py --results_dir results/processed_merged  # 合并数据集
  python src/eval/compare.py --all                    # 同时显示所有数据集
"""

import os
import json
import argparse


HZ_LIST = [5, 10, 25, 50]
ML_MODELS = ["rf", "xgb"]
DL_MODELS = ["cnn", "cnn_lstm", "transformer"]


def load_result(results_dir, hz, prefix, model_name):
    path = os.path.join(results_dir, f"{hz}hz", f"{prefix}_{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_table(results_dir, title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    all_models = [("ml", m) for m in ML_MODELS] + [("dl", m) for m in DL_MODELS]

    for metric, key in [("Test Accuracy", "accuracy"), ("Macro F1", "macro_f1")]:
        print(f"\n  ({metric})")
        header = f"{'模型':<18}" + "".join(f"{'%dHz'%hz:>10}" for hz in HZ_LIST)
        print(header)
        print("-" * 58)
        for prefix, model in all_models:
            row = f"{prefix.upper()+'/'+model:<18}"
            for hz in HZ_LIST:
                r = load_result(results_dir, hz, prefix, model)
                row += f"{'—':>10}" if r is None else f"{r[key]:.4f}".rjust(10)
            print(row)

    print("=" * 70)


def main(args):
    if args.all:
        base = args.results_dir
        candidates = []
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                d = os.path.join(base, name)
                if os.path.isdir(d):
                    candidates.append((name, d))
        if not candidates:
            candidates = [("results", base)]
        for name, d in candidates:
            print_table(d, f"IMU 犬只行为识别 — {name}")
    else:
        print_table(args.results_dir, "IMU 犬只行为识别 — ML vs DL × 采样率对比")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/processed",
                        help="结果目录（含各 {hz}hz/ 子目录）")
    parser.add_argument("--all", action="store_true",
                        help="遍历 results_dir 下所有子目录并分别展示")
    main(parser.parse_args())
