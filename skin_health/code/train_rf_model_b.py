"""
训练"模型B"（综合严重度分类器，预测S0/S1/S2，见two_stage_rf_architecture.md）。

特征 = 模型A的全部输入特征 + 模型A的输出（用GroupKFold的out-of-fold预测
概率，不是同一份数据train完直接predict——那样会泄漏，模型B会学到"抄
模型A在训练集上背下来的答案"而不是学到问答信息真正带来的增量价值）+
问答特征（questionnaire_features.py）。

用法：
    python skin_health/code/gen_model_b_training_data.py --data_dir skin_health/data/rf_synthetic
    python skin_health/code/train_rf_model_b.py --data_dir skin_health/data/rf_synthetic
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS
from rf_features import FEATURE_COLUMNS

MODEL_FILENAME = "model_b.joblib"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    table = pd.read_csv(os.path.join(args.data_dir, "model_b_training_table.csv"))
    print(f"模型B训练表: {len(table)}行, {table.pet_id.nunique()}个场景狗")

    # 长窗口基线(比如recent180)在合成数据里可能整列全NaN(场景库最长历史不够
    # 覆盖这个窗口)，HistGradientBoostingClassifier分箱阶段会因此报错，训练时
    # 先排除——不是硬编码列名，判断条件是"全NaN"，真实数据积累够了会自动纳入。
    feature_cols_a = [c for c in FEATURE_COLUMNS if not table[c].isna().all()]
    dropped_a = [c for c in FEATURE_COLUMNS if c not in feature_cols_a]
    if dropped_a:
        print(f"以下模型A特征在当前数据里全为NaN，stacking训练时跳过: {dropped_a}")

    X_a = table[feature_cols_a].copy()
    X_a["breed_or_size_class"] = X_a["breed_or_size_class"].astype("category")
    groups = table["pet_id"].values
    n_folds_a = min(5, table.pet_id.nunique())

    # ── 模型A的out-of-fold预测概率，当stacking特征喂给模型B ──
    cat_idx_a = [X_a.columns.get_loc("breed_or_size_class")]
    model_a_for_stack = HistGradientBoostingClassifier(
        categorical_features=cat_idx_a, max_depth=5, min_samples_leaf=8,
        class_weight="balanced", random_state=42,
    )
    gkf_a = GroupKFold(n_splits=n_folds_a)
    oof_proba = cross_val_predict(model_a_for_stack, X_a, table["c_tier"].values,
                                   groups=groups, cv=gkf_a, method="predict_proba")
    classes_a = np.unique(table["c_tier"].values)
    for i, cls in enumerate(classes_a):
        table[f"model_a_proba_{cls}"] = oof_proba[:, i]
    stack_cols = [f"model_a_proba_{cls}" for cls in classes_a]

    feature_cols_b_all = FEATURE_COLUMNS + stack_cols + QUESTIONNAIRE_FEATURE_COLUMNS
    feature_cols_b = [c for c in feature_cols_b_all if not table[c].isna().all()]
    dropped_b = [c for c in feature_cols_b_all if c not in feature_cols_b]
    if dropped_b:
        print(f"以下模型B特征在当前数据里全为NaN，训练时跳过: {dropped_b}")
    X = table[feature_cols_b].copy()
    X["breed_or_size_class"] = X["breed_or_size_class"].astype("category")
    y = table["s_tier"].values

    cat_idx = [X.columns.get_loc("breed_or_size_class")]
    model_b = HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_depth=5, min_samples_leaf=8,
        class_weight="balanced", random_state=42,
    )

    n_folds = min(args.n_folds, table.pet_id.nunique())
    gkf = GroupKFold(n_splits=n_folds)
    fold_f1s, all_true, all_pred = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        model_b.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model_b.predict(X.iloc[te_idx])
        f1 = f1_score(y[te_idx], pred, average="macro")
        fold_f1s.append(f1)
        all_true.extend(y[te_idx])
        all_pred.extend(pred)
        print(f"  fold {fold+1}/{n_folds}: 宏F1={f1:.3f} (测试集{len(te_idx)}行, "
              f"{len(set(groups[te_idx]))}只场景狗)")

    print(f"\n交叉验证宏F1: 均值={np.mean(fold_f1s):.3f} 标准差={np.std(fold_f1s):.3f}")
    print("\n汇总分类报告(跨全部fold的测试集预测):")
    print(classification_report(all_true, all_pred, digits=3))
    labels_order = sorted(set(all_true) | set(all_pred))
    print("混淆矩阵 (行=真值, 列=预测):", labels_order)
    print(confusion_matrix(all_true, all_pred, labels=labels_order))

    print("\n用全量数据重新训练，计算特征重要性(permutation importance)...")
    model_b.fit(X, y)

    # 持久化，供rf_infer.py加载做真实推理用。stack_cols用来在推理时对齐——
    # 推理时不会重新训一个model_a_for_stack，而是直接用已经持久化的
    # model_a.joblib对新数据算predict_proba，按这里存的stack_cols名字
    # (对应模型A的每个类别C0/C1/C2)拼进模型B的输入特征，不依赖训练时这里
    # 用cross_val_predict算出来的具体概率值(那是训练/评估阶段防泄漏用的，
    # 不是推理该用的模型A)
    model_path = os.path.join(args.data_dir, MODEL_FILENAME)
    joblib.dump({
        "model": model_b,
        "feature_cols": feature_cols_b,
        "stack_cols": stack_cols,
        "breed_categories": list(X["breed_or_size_class"].cat.categories),
        "classes": list(model_b.classes_),
    }, model_path)
    print(f"模型B已持久化: {model_path}")

    result = permutation_importance(model_b, X, y, n_repeats=15, random_state=42,
                                     scoring="f1_macro", n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": feature_cols_b,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print(importance_df.to_string(index=False))

    out_path = os.path.join(args.data_dir, "model_b_feature_importance.csv")
    importance_df.to_csv(out_path, index=False)

    report_path = os.path.join(args.data_dir, "model_b_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 模型B（综合严重度分类器）训练报告——合成数据\n\n")
        f.write(f"训练表：{len(table)}行，{table.pet_id.nunique()}个场景狗（仅C1/C2触发问答"
                f"且用户回答的天）\n\n")
        f.write(f"交叉验证({n_folds}折，按场景狗分组)宏F1：均值={np.mean(fold_f1s):.3f}"
                f" 标准差={np.std(fold_f1s):.3f}\n\n")
        f.write("## 特征重要性(permutation importance，全量数据)\n\n")
        f.write("| 特征 | 重要性均值 | 标准差 |\n|---|---|---|\n")
        for _, r in importance_df.iterrows():
            f.write(f"| {r['feature']} | {r['importance_mean']:.4f} | {r['importance_std']:.4f} |\n")
    print(f"\n报告已保存: {report_path}")
    print(f"特征重要性CSV已保存: {out_path}")


if __name__ == "__main__":
    main()
