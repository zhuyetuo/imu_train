"""
按抓挠模式(pattern)逐类抽样例子，对比SBS(scratch_burden.py)和模型A(RF)的
判断，生成流程验证展示文档的原始数据(JSON)，供人工写成md。

重要：这不是"哪个更准"的对比——模型A的训练标签就是SBS自己的run_pipeline()
输出(见train_rf_model_a.py开头的说明)，模型A本质是在学"用RF去逼近SBS"，
两者不一致的地方只能说明RF没学准，不能说明SBS判断错误。这份脚本只做两件事：
  1. 验证RF在各种抓挠模式下都能学会跟SBS方向一致的判断(流程/特征工程没问题)
  2. 展示RF的概率输出比SBS的离散档位更细粒度，並用具体特征值做可解释性说明

用法：
    python skin_health/code/gen_model_a_pattern_showcase.py --data_dir skin_health/data/rf_synthetic
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from rf_features import FEATURE_COLUMNS, compute_rf_features
from scratch_burden import run_pipeline

EXPLAIN_FEATURES = [
    "event_count", "total_duration_min", "z_score_vs_self",
    "baseline_ratio_count_excl_recent14", "baseline_ratio_duration_excl_recent14",
    "consecutive_days_above_baseline", "interval_std", "night_ratio",
    "flare_episode_count_90d", "sleep_disruption_count",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.data_dir, "model_a_pattern_showcase.json")

    events = pd.read_csv(os.path.join(args.data_dir, "all_events.csv"), parse_dates=["start", "end"])
    wear = pd.read_csv(os.path.join(args.data_dir, "all_wear.csv"), parse_dates=["date"])
    wear["date"] = wear["date"].dt.date
    breed_map = json.load(open(os.path.join(args.data_dir, "breed_map.json"), encoding="utf-8"))
    scenarios = {s["scenario_id"]: s for s in
                 json.load(open(os.path.join(args.data_dir, "scenarios.json"), encoding="utf-8"))["scenarios"]}

    feats = compute_rf_features(events, wear, breed_map)
    sbs = run_pipeline(events, wear)
    sbs_full = sbs.rename(columns={"tier": "sbs_tier", "total": "sbs_total"})
    table = feats.merge(
        sbs_full[["pet_id", "date", "sbs_tier", "sbs_total", "delta_score", "cluster_score",
                  "persistence_score", "interrupt_score", "bootstrap_mode"]],
        on=["pet_id", "date"], how="inner")
    table = table[table["sbs_tier"] != "insufficient_data"].reset_index(drop=True)

    feature_cols = [c for c in FEATURE_COLUMNS if not table[c].isna().all()]
    dropped = [c for c in FEATURE_COLUMNS if c not in feature_cols]
    if dropped:
        print(f"以下特征在当前数据里全为NaN，跳过: {dropped}")
    X = table[feature_cols].copy()
    X["breed_or_size_class"] = X["breed_or_size_class"].astype("category")
    y = table["sbs_tier"].values
    groups = table["pet_id"].values
    n_folds = min(5, table.pet_id.nunique())
    cat_idx = [X.columns.get_loc("breed_or_size_class")]
    model = HistGradientBoostingClassifier(
        categorical_features=cat_idx, max_depth=5, min_samples_leaf=8,
        class_weight="balanced", random_state=42,
    )
    gkf = GroupKFold(n_splits=n_folds)
    oof_proba = cross_val_predict(model, X, y, groups=groups, cv=gkf, method="predict_proba")
    classes = list(model.fit(X, y).classes_)  # fit once for class order (labels only, not used for oof)
    for i, cls in enumerate(classes):
        table[f"rf_proba_{cls}"] = oof_proba[:, i]
    table["rf_pred_tier"] = [classes[i] for i in oof_proba.argmax(axis=1)]

    # 按pattern分组，每类选一个场景(取该pattern里天数最多的那个，故事最完整)
    pattern_to_scenarios = {}
    for sid, s in scenarios.items():
        pattern_to_scenarios.setdefault(s["pattern"], []).append(sid)

    showcase = []
    for pattern, sids in pattern_to_scenarios.items():
        sub_all = table[table["pet_id"].isin(sids)]
        best_sid = sub_all.groupby("pet_id").size().idxmax()
        s_meta = scenarios[best_sid]
        sub = table[table["pet_id"] == best_sid].sort_values("date").reset_index(drop=True)

        days = []
        for _, row in sub.iterrows():
            days.append({
                "date": str(row["date"]),
                "event_count": int(row["event_count"]),
                "total_duration_min": round(float(row["total_duration_min"]), 1),
                "sbs_tier": row["sbs_tier"],
                "sbs_total": None if pd.isna(row["sbs_total"]) else round(float(row["sbs_total"]), 1),
                "sbs_components": {
                    "delta": row["delta_score"], "cluster": row["cluster_score"],
                    "persistence": row["persistence_score"], "interrupt": row["interrupt_score"],
                    "bootstrap_mode": bool(row["bootstrap_mode"]),
                },
                "rf_pred_tier": row["rf_pred_tier"],
                "rf_proba": {cls: round(float(row[f"rf_proba_{cls}"]), 3) for cls in classes},
                "agree": row["sbs_tier"] == row["rf_pred_tier"],
                "explain_features": {
                    f: (None if pd.isna(row[f]) else round(float(row[f]), 3))
                    for f in EXPLAIN_FEATURES if f in row
                },
            })

        n_days = len(days)
        n_agree = sum(d["agree"] for d in days)
        showcase.append({
            "pattern": pattern,
            "scenario_id": best_sid,
            "scenario_meta": s_meta,
            "n_days": n_days,
            "agree_rate": round(n_agree / n_days, 3) if n_days else None,
            "days": days,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(showcase, f, ensure_ascii=False, indent=2)
    print(f"已生成: {out_path}")
    for item in showcase:
        print(f"  {item['pattern']:28s} scenario={item['scenario_id']:45s} "
              f"天数={item['n_days']:3d} 一致率={item['agree_rate']}")


if __name__ == "__main__":
    main()
