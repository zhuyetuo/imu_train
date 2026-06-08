"""
汇总所有 hz × 模型的结果，输出对比表格。
用法: python src/eval/compare.py
"""

import os
import json
import glob


HZ_LIST = [5, 10, 25, 50]
ML_MODELS = ["rf", "svm", "xgb"]
DL_MODELS = ["cnn", "cnn_lstm", "transformer"]
RESULTS_DIR = "results"


def load_result(hz, prefix, model_name):
    path = os.path.join(RESULTS_DIR, f"{hz}hz", f"{prefix}_{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    print("\n" + "=" * 70)
    print("  IMU 犬只行为识别 — ML vs DL × 采样率 对比结果 (Test Accuracy)")
    print("=" * 70)

    all_models = [("ml", m) for m in ML_MODELS] + [("dl", m) for m in DL_MODELS]

    # 表头
    header = f"{'模型':<18}" + "".join(f"{'%dHz'%hz:>10}" for hz in HZ_LIST)
    print(header)
    print("-" * 58)

    for prefix, model in all_models:
        row = f"{prefix.upper()+'/'+model:<18}"
        for hz in HZ_LIST:
            r = load_result(hz, prefix, model)
            if r is None:
                row += f"{'—':>10}"
            else:
                row += f"{r['accuracy']:.4f}".rjust(10)
        print(row)

    print("\n  (Macro F1)")
    print("-" * 58)
    for prefix, model in all_models:
        row = f"{prefix.upper()+'/'+model:<18}"
        for hz in HZ_LIST:
            r = load_result(hz, prefix, model)
            if r is None:
                row += f"{'—':>10}"
            else:
                row += f"{r['macro_f1']:.4f}".rjust(10)
        print(row)

    print("=" * 70)


if __name__ == "__main__":
    main()
