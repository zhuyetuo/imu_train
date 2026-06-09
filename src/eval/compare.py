"""
汇总各数据集、各模型的测试结果，输出对比表格。
用法:
  python src/eval/compare.py                        # 列出 results/ 下所有数据集
  python src/eval/compare.py --dataset processed_a  # 只看数据集A的结果
  python src/eval/compare.py --per_class            # 同时展示每类 P/R/F1
  python src/eval/compare.py --per_class --hz 50 --model xgb  # 指定模型和采样率
"""

import os
import json
import argparse


HZ_LIST = [5, 10, 25, 50]
ML_MODELS = ["rf", "xgb", "lgbm", "catboost"]
DL_MODELS = ["cnn", "collar_cnn", "cnn_lstm", "transformer", "filternet", "filternet_m2m"]
RESULTS_ROOT = "results"


def load_result(results_dir, hz, prefix, model_name):
    path = os.path.join(results_dir, f"{hz}hz", f"{prefix}_{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_overview(results_dir, dataset_name):
    all_models = [("ml", m) for m in ML_MODELS] + [("dl", m) for m in DL_MODELS]

    has_any = any(
        load_result(results_dir, hz, p, m) is not None
        for hz in HZ_LIST for p, m in all_models
    )
    if not has_any:
        return False

    print(f"\n{'=' * 70}")
    print(f"  数据集: {dataset_name}")
    print(f"{'=' * 70}")

    for metric, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1")]:
        print(f"\n  ({metric})")
        header = f"{'模型':<22}" + "".join(f"{'%dHz'%hz:>10}" for hz in HZ_LIST)
        print(header)
        print("-" * 62)
        for prefix, model in all_models:
            row = f"{prefix.upper()+'/'+model:<22}"
            for hz in HZ_LIST:
                r = load_result(results_dir, hz, prefix, model)
                row += f"{'—':>10}" if r is None else f"{r[key]:.4f}".rjust(10)
            print(row)
    return True


def print_per_class(results_dir, dataset_name, hz_filter, model_filter):
    """展示每个类别的 Precision / Recall / F1，按指定 hz 和模型过滤。"""
    all_models = [("ml", m) for m in ML_MODELS] + [("dl", m) for m in DL_MODELS]
    hz_list = [hz_filter] if hz_filter else HZ_LIST

    printed_header = False
    for hz in hz_list:
        for prefix, model in all_models:
            if model_filter and model != model_filter:
                continue
            r = load_result(results_dir, hz, prefix, model)
            if r is None or "per_class" not in r:
                continue

            if not printed_header:
                print(f"\n{'─' * 70}")
                print(f"  每类指标 — 数据集: {dataset_name}")
                print(f"{'─' * 70}")
                printed_header = True

            label = f"{prefix.upper()}/{model}  {hz}Hz"
            print(f"\n  [{label}]")
            classes = r.get("classes", list(r["per_class"].keys()))
            header = f"  {'类别':<18}{'Precision':>12}{'Recall':>10}{'F1':>10}"
            print(header)
            print(f"  {'-'*48}")
            for cls in classes:
                if cls not in r["per_class"]:
                    continue
                m = r["per_class"][cls]
                print(f"  {cls:<18}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1-score']:>10.4f}")


def main(args):
    datasets = []
    if args.dataset:
        datasets = [(args.dataset, os.path.join(RESULTS_ROOT, args.dataset))]
    else:
        if not os.path.isdir(RESULTS_ROOT):
            print(f"[compare] 还没有结果，请先训练模型。")
            return
        for name in sorted(os.listdir(RESULTS_ROOT)):
            d = os.path.join(RESULTS_ROOT, name)
            if os.path.isdir(d) and name != "infer":
                datasets.append((name, d))

    if not datasets:
        print(f"[compare] results/ 下没有找到任何训练结果。")
        return

    for name, d in datasets:
        found = print_overview(d, name)
        if found and args.per_class:
            print_per_class(d, name, args.hz, args.model)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="",
                        help="只显示指定数据集结果，如 processed_a。留空则显示所有。")
    parser.add_argument("--per_class", action="store_true",
                        help="同时展示每类 Precision / Recall / F1")
    parser.add_argument("--hz", type=int, default=0,
                        help="只展示指定采样率的 per-class 结果（配合 --per_class）")
    parser.add_argument("--model", default="",
                        help="只展示指定模型的 per-class 结果，如 xgb（配合 --per_class）")
    main(parser.parse_args())
