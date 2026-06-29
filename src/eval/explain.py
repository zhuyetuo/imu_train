"""
SHAP 特征重要性分析（仅适用于 ML 模型）。

用法:
  python src/eval/explain.py --hz 50 --model xgb --processed_dir data/processed_a
  python src/eval/explain.py --hz 50 --model rf  --processed_dir data/processed_a --max_samples 500

输出（保存到 results/<dataset>/<hz>hz/shap_<model>/）:
  summary_bar.png   — 全局特征重要性（条形图，Top-N 特征）
  summary_dot.png   — beeswarm 图（每个样本的 SHAP 值分布）
  class_<name>.png  — 每个行为类别的 Top 特征（多分类时）
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))

import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from dataset import load_all_splits
from features import extract_features


# ── 特征名生成 ─────────────────────────────────────────────────────────────────

CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]

TIME_FEAT_NAMES = ["mean", "std", "min", "max", "range", "rms", "skew", "kurtosis", "zcr"]
FREQ_FEAT_NAMES = ["spec_mean", "spec_std", "peak_freq", "spec_entropy"]


def build_feature_names(n_channels: int) -> list[str]:
    names = []
    for ch in range(n_channels):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
        for feat in TIME_FEAT_NAMES:
            names.append(f"{ch_name}_{feat}")
    for ch in range(n_channels):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
        for feat in FREQ_FEAT_NAMES:
            names.append(f"{ch_name}_{feat}")
    return names


# ── 绘图工具 ───────────────────────────────────────────────────────────────────

def save_summary_bar(shap_values, feature_names, out_path, title, max_display=20):
    plt.figure(figsize=(8, max_display * 0.35 + 1))
    shap.summary_plot(
        shap_values, feature_names=feature_names,
        plot_type="bar", max_display=max_display, show=False,
    )
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out_path}")


def save_summary_dot(shap_values, X_feat, feature_names, out_path, title, max_display=20):
    plt.figure(figsize=(9, max_display * 0.35 + 1))
    shap.summary_plot(
        shap_values, X_feat, feature_names=feature_names,
        max_display=max_display, show=False,
    )
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {out_path}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main(args):
    print(f"\n[explain] hz={args.hz}, model={args.model}, processed_dir={args.processed_dir}")

    # 加载数据
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), meta = load_all_splits(args.hz, args.processed_dir)
    classes = eval(meta["classes"]) if isinstance(meta["classes"], str) else meta["classes"]
    n_channels = X_tr.shape[2]
    feature_names = build_feature_names(n_channels)

    # 加载特征（优先读缓存）
    feat_cache = os.path.join(args.processed_dir, f"{args.hz}hz", "ml_features.npz")
    if os.path.exists(feat_cache):
        print(f"[explain] 加载缓存特征: {feat_cache}")
        cache = np.load(feat_cache)
        X_tr_f, X_te_f = cache["X_tr"], cache["X_te"]
    else:
        print(f"[explain] 提取特征...")
        X_tr_f = extract_features(X_tr, args.hz)
        X_te_f = extract_features(X_te, args.hz)

    # 加载模型
    dataset_tag = os.path.basename(args.processed_dir.rstrip("/"))
    model_path = os.path.join(args.results_dir, dataset_tag, f"{args.hz}hz", f"ml_{args.model}.pkl")
    if not os.path.exists(model_path):
        print(f"[explain] ❌ 找不到模型: {model_path}")
        print(f"  请先运行: python src/ml/train.py --hz {args.hz} --model {args.model} --processed_dir {args.processed_dir}")
        return
    model = joblib.load(model_path)
    print(f"[explain] 加载模型: {model_path}")

    # 采样（SHAP 计算较慢，默认最多 1000 个测试样本）
    max_s = min(args.max_samples, len(X_te_f))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_te_f), max_s, replace=False)
    X_explain = X_te_f[idx]
    print(f"[explain] SHAP 分析样本数: {max_s}")

    # 背景数据（用训练集子集）
    bg_size = min(200, len(X_tr_f))
    bg_idx = rng.choice(len(X_tr_f), bg_size, replace=False)
    X_bg = X_tr_f[bg_idx]

    # 构建 Explainer（rf 需要背景数据，tree boosters 不需要）
    if args.model in ("xgb", "lgbm", "catboost"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model, data=X_bg)

    print(f"[explain] 计算 SHAP 值（可能需要几分钟）...")
    shap_values = explainer(X_explain)   # shap.Explanation 对象

    # 输出目录
    out_dir = os.path.join(args.results_dir, dataset_tag, f"{args.hz}hz", f"shap_{args.model}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[explain] 保存图表至 {out_dir}/")

    n_classes = len(classes)
    dataset_label = f"{dataset_tag} | {args.hz}Hz | {args.model.upper()}"

    if n_classes == 2 or shap_values.values.ndim == 2:
        sv = shap_values.values
        save_summary_bar(sv, feature_names, os.path.join(out_dir, "summary_bar.png"),
                         f"Feature Importance — {dataset_label}")
        save_summary_dot(sv, X_explain, feature_names, os.path.join(out_dir, "summary_dot.png"),
                         f"SHAP Values — {dataset_label}")
    else:
        # 多分类：shap_values.values shape = (N, n_features, n_classes)
        sv_all = shap_values.values  # (N, F, C)

        # 全局重要性：各类别平均绝对值求和
        global_importance = np.abs(sv_all).mean(axis=(0, 2))  # (F,)
        top_idx = np.argsort(global_importance)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh([feature_names[i] for i in top_idx[::-1]], global_importance[top_idx[::-1]])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Global Feature Importance (Top 20) — {dataset_label}")
        plt.tight_layout()
        bar_path = os.path.join(out_dir, "summary_bar.png")
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ {bar_path}")

        # 每个类别的 beeswarm
        for c_idx, c_name in enumerate(classes):
            sv_c = sv_all[:, :, c_idx]  # (N, F)
            dot_path = os.path.join(out_dir, f"class_{c_name}.png")
            save_summary_dot(sv_c, X_explain, feature_names, dot_path,
                             f"SHAP — {dataset_label} — Class: {c_name}")

    print(f"\n[explain] 完成！图表保存至 {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=int, required=True, choices=[5, 10, 15, 16, 20, 25, 50])
    parser.add_argument("--model", default="xgb", choices=["rf", "xgb", "lgbm", "catboost"])
    parser.add_argument("--processed_dir", default="data/processed_a")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="用于 SHAP 分析的测试集样本上限（越多越慢）")
    main(parser.parse_args())
