"""
汇总各数据集、各模型的测试结果，输出对比表格。
用法:
  python src/eval/compare.py                      # 列出 results/ 下所有数据集
  python src/eval/compare.py --dataset processed_a  # 只看数据集A的结果
"""

import os
import json
import argparse


HZ_LIST = [5, 10, 25, 50]
ML_MODELS = ["rf", "xgb", "lgbm", "catboost"]
DL_MODELS = ["cnn", "cnn_lstm", "transformer"]
RESULTS_ROOT = "results"


def load_result(results_dir, hz, prefix, model_name):
    path = os.path.join(results_dir, f"{hz}hz", f"{prefix}_{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_table(results_dir, dataset_name):
    all_models = [("ml", m) for m in ML_MODELS] + [("dl", m) for m in DL_MODELS]

    # 检查该目录下是否有任何结果
    has_any = any(
        load_result(results_dir, hz, p, m) is not None
        for hz in HZ_LIST for p, m in all_models
    )
    if not has_any:
        return

    print(f"\n{'=' * 70}")
    print(f"  数据集: {dataset_name}")
    print(f"{'=' * 70}")

    for metric, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1")]:
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


def main(args):
    if args.dataset:
        d = os.path.join(RESULTS_ROOT, args.dataset)
        print_table(d, args.dataset)
    else:
        # 遍历 results/ 下所有子目录
        if not os.path.isdir(RESULTS_ROOT):
            print(f"[compare] 还没有结果，请先训练模型。")
            return
        found = False
        for name in sorted(os.listdir(RESULTS_ROOT)):
            d = os.path.join(RESULTS_ROOT, name)
            if os.path.isdir(d):
                print_table(d, name)
                found = True
        if not found:
            print(f"[compare] results/ 下没有找到任何训练结果。")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="",
                        help="只显示指定数据集结果，如 processed_a。留空则显示所有。")
    main(parser.parse_args())
