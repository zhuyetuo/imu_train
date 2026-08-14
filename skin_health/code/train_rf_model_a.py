"""
用 gen_rf_synthetic_scenarios.py 生成的合成数据，训练"模型A"（行为严重度
分类器，见 two_stage_rf_architecture.md）——HistGradientBoostingClassifier，
特征集见 rf_feature_spec.md（IMU/行为特征部分）。

标签来源：复用 scratch_burden.py 的SBS引擎（run_pipeline）算出的C0/C1/C2
真值——这么做是有意的，不是偷懒：现在没有真实兽医标签，用当前"官方"的SBS
规则给合成数据打真值标签，训练出来的模型本质上是"SBS规则的一个可学习近似"，
用来验证两件事：
  1. 特征集本身够不够——如果RF在这批合成数据上都学不好（比如某个特征
     在真值标签里明显该有区分力，但RF没学到），说明特征工程有问题，
     不是数据不够的问题（合成数据可以要多少有多少）
  2. 特征重要性排序——哪些特征RF觉得没用，是"这份合成数据没设计到能
     体现这个特征价值的场景"（需要回去补场景），还是"这个特征本身在
     rf_feature_spec.md里就是冗余的"（那份文档已经标注过几个候选冗余项，
     这里可以用实际训练结果验证那些猜测对不对）

注意：这个结果不能代表真实准确率，等真实兽医标签攒够后要重新训练评估，
这里只验证代码/特征管道本身没问题、看合成数据设计得够不够全面。

用法：
    python skin_health/code/train_rf_model_a.py --data_dir skin_health/data/rf_synthetic
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from rf_features import FEATURE_COLUMNS, compute_rf_features
from scratch_burden import run_pipeline


def build_training_table(data_dir):
    events = pd.read_csv(os.path.join(data_dir, "all_events.csv"), parse_dates=["start", "end"])
    wear = pd.read_csv(os.path.join(data_dir, "all_wear.csv"), parse_dates=["date"])
    wear["date"] = wear["date"].dt.date
    breed_map = json.load(open(os.path.join(data_dir, "breed_map.json"), encoding="utf-8"))

    feats = compute_rf_features(events, wear, breed_map)
    labels = run_pipeline(events, wear)[["pet_id", "date", "tier"]]

    table = feats.merge(labels, on=["pet_id", "date"], how="inner")
    table = table[table["tier"] != "insufficient_data"].reset_index(drop=True)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    table = build_training_table(args.data_dir)
    print(f"训练表: {len(table)}行, {table.pet_id.nunique()}只(场景)狗, "
          f"标签分布:\n{table.tier.value_counts()}")

    X = table[FEATURE_COLUMNS].copy()
    X["breed_or_size_class"] = X["breed_or_size_class"].astype("category")
    y = table["tier"].values
    groups = table["pet_id"].values

    cat_idx = [X.columns.get_loc("breed_or_size_class")]
    model = HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_depth=5, min_samples_leaf=8,
        class_weight="balanced", random_state=42,
    )

    n_folds = min(args.n_folds, table.pet_id.nunique())
    gkf = GroupKFold(n_splits=n_folds)
    fold_f1s, all_true, all_pred = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        model.fit(X.iloc[tr_idx], y[tr_idx])
        pred = model.predict(X.iloc[te_idx])
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

    # 全量数据上再训一版，专门用来看特征重要性（permutation importance，
    # 比HistGradientBoosting自带的重要性更可靠，且能处理类别特征）
    print("\n用全量数据重新训练，计算特征重要性(permutation importance)...")
    model.fit(X, y)
    # 直接用X（保留breed_or_size_class的category dtype），不要转成整数编码——
    # HistGradientBoostingClassifier内部按fit时看到的category dtype做预处理，
    # permutation_importance重新预测时如果喂整数编码会跟内部的类别映射对不上
    result = permutation_importance(model, X, y, n_repeats=15, random_state=42,
                                     scoring="f1_macro", n_jobs=-1)
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    print(importance_df.to_string(index=False))

    out_path = os.path.join(args.data_dir, "model_a_feature_importance.csv")
    importance_df.to_csv(out_path, index=False)

    report_path = os.path.join(args.data_dir, "model_a_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 模型A（行为严重度分类器）训练报告——合成数据\n\n")
        f.write(f"训练表：{len(table)}行，{table.pet_id.nunique()}个场景狗\n\n")
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
