"""
模型B效果展示：对比"只用C兜底(用户不填问答时的规则)" vs "模型B综合判断
(用户填了问答)" vs "真值S标签"，按questionnaire_behavior三类场景各挑
例子。

跟模型A的showcase不同，这里是站得住脚的"效果对比"，不是单纯流程验证：
真值s_tier由questionnaire_features.py的true_s_tier()生成，这个函数
不是PM的固定加权公式，也不是模型B自己的输出，是我们独立设计的合成规则
(0.35*C_ordinal + 0.65*皮肤严重度)，模型B是在学这个独立规则，不是在
模仿"只用C兜底"这个naive方法，两者不循环，可以正当比较。

三类questionnaire_behavior：
  answered_consistent          问答内容跟C同步，naive兜底大概率也对
  answered_conflicting_worse   问答显示皮肤比C推断的更严重，naive兜底会漏判
  answered_conflicting_better  问答显示皮肤其实比C推断的更轻，naive兜底会过度紧张

用法：
    python skin_health/code/gen_model_b_pattern_showcase.py --data_dir skin_health/data/rf_synthetic
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS
from rf_features import FEATURE_COLUMNS

NAIVE_FALLBACK = {"C1": "S1", "C2": "S2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.data_dir, "model_b_pattern_showcase.json")

    table = pd.read_csv(os.path.join(args.data_dir, "model_b_training_table.csv"))

    feature_cols_a = [c for c in FEATURE_COLUMNS if not table[c].isna().all()]
    X_a = table[feature_cols_a].copy()
    X_a["breed_or_size_class"] = X_a["breed_or_size_class"].astype("category")
    groups = table["pet_id"].values
    n_folds = min(5, table.pet_id.nunique())

    cat_idx_a = [X_a.columns.get_loc("breed_or_size_class")]
    model_a_for_stack = HistGradientBoostingClassifier(
        categorical_features=cat_idx_a, max_depth=5, min_samples_leaf=8,
        class_weight="balanced", random_state=42,
    )
    gkf = GroupKFold(n_splits=n_folds)
    oof_proba_a = cross_val_predict(model_a_for_stack, X_a, table["c_tier"].values,
                                     groups=groups, cv=gkf, method="predict_proba")
    classes_a = list(np.unique(table["c_tier"].values))
    for i, cls in enumerate(classes_a):
        table[f"model_a_proba_{cls}"] = oof_proba_a[:, i]
    stack_cols = [f"model_a_proba_{cls}" for cls in classes_a]

    feature_cols_b_all = FEATURE_COLUMNS + stack_cols + QUESTIONNAIRE_FEATURE_COLUMNS
    feature_cols_b = [c for c in feature_cols_b_all if not table[c].isna().all()]
    X = table[feature_cols_b].copy()
    X["breed_or_size_class"] = X["breed_or_size_class"].astype("category")
    y = table["s_tier"].values

    cat_idx = [X.columns.get_loc("breed_or_size_class")]
    model_b = HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_depth=5, min_samples_leaf=8,
        class_weight="balanced", random_state=42,
    )
    oof_proba_b = cross_val_predict(model_b, X, y, groups=groups, cv=gkf, method="predict_proba")
    classes_b = list(np.unique(y))
    for i, cls in enumerate(classes_b):
        table[f"model_b_proba_{cls}"] = oof_proba_b[:, i]
    table["model_b_pred"] = [classes_b[i] for i in oof_proba_b.argmax(axis=1)]
    table["naive_fallback_pred"] = table["c_tier"].map(NAIVE_FALLBACK)

    table["naive_correct"] = table["naive_fallback_pred"] == table["s_tier"]
    table["model_b_correct"] = table["model_b_pred"] == table["s_tier"]

    summary = {}
    showcase = []
    for behavior, grp in table.groupby("questionnaire_behavior"):
        naive_acc = grp["naive_correct"].mean()
        model_b_acc = grp["model_b_correct"].mean()
        summary[behavior] = {
            "n": len(grp), "naive_fallback_accuracy": round(float(naive_acc), 3),
            "model_b_accuracy": round(float(model_b_acc), 3),
        }
        # 挑几个例子：优先挑"naive错、模型B对"的（最能体现问答价值），
        # 每类最多挑3个，不够就补充其他例子
        rows = []
        corrected = grp[(~grp["naive_correct"]) & (grp["model_b_correct"])]
        others = grp[~((~grp["naive_correct"]) & (grp["model_b_correct"]))]
        picked = pd.concat([corrected.head(3), others.head(max(0, 3 - len(corrected)))])
        for _, row in picked.iterrows():
            rows.append({
                "pet_id": row["pet_id"], "date": str(row["date"]),
                "c_tier": row["c_tier"], "s_tier_true": row["s_tier"],
                "naive_fallback_pred": row["naive_fallback_pred"],
                "naive_correct": bool(row["naive_correct"]),
                "model_b_pred": row["model_b_pred"],
                "model_b_proba": {cls: round(float(row[f"model_b_proba_{cls}"]), 3) for cls in classes_b},
                "model_b_correct": bool(row["model_b_correct"]),
                "questionnaire": {c: (None if pd.isna(row[c]) else row[c]) for c in QUESTIONNAIRE_FEATURE_COLUMNS},
            })
        showcase.append({"questionnaire_behavior": behavior, "examples": rows})

    out = {"summary": summary, "showcase": showcase}
    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"已生成: {out_path}")
    for behavior, s in summary.items():
        print(f"  {behavior:30s} n={s['n']:4d} naive兜底准确率={s['naive_fallback_accuracy']} "
              f"模型B准确率={s['model_b_accuracy']}")


if __name__ == "__main__":
    main()
