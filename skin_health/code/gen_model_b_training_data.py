"""
用已有的86个模型A合成场景（skin_health/data/rf_synthetic/scenarios.json，
里面本来就带questionnaire_behavior标注），生成模型B（综合严重度分类器，见
two_stage_rf_architecture.md）的训练数据——不需要重新设计新场景，直接复用。

只有两类天会进模型B训练集：
  1. 当天SBS真值tier是C1或C2（C0不触发问答，不产出模型B样本）
  2. 场景的questionnaire_behavior是"answered_*"三种之一（"no_answer"和
     "not_triggered"不产出问答，S直接按two_stage_rf_architecture.md的
     兜底规则出，不是模型B学出来的，不算模型B的训练样本）

用法：
    python skin_health/code/gen_model_b_training_data.py \
        --data_dir skin_health/data/rf_synthetic
（复用同目录下已经生成好的all_events.csv/all_wear.csv/scenarios.json，
输出model_b_training_table.csv到同目录）
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from questionnaire_features import (QUESTIONNAIRE_FEATURE_COLUMNS,
                                     simulate_questionnaire_row, true_s_tier)
from rf_features import FEATURE_COLUMNS, compute_rf_features
from scratch_burden import run_pipeline

TIER_ORDINAL = {"C0": 0, "C1": 1, "C2": 2}
ANSWERED_BEHAVIORS = {"answered_consistent", "answered_conflicting_worse", "answered_conflicting_better"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    events = pd.read_csv(os.path.join(args.data_dir, "all_events.csv"), parse_dates=["start", "end"])
    wear = pd.read_csv(os.path.join(args.data_dir, "all_wear.csv"), parse_dates=["date"])
    wear["date"] = wear["date"].dt.date
    breed_map = json.load(open(os.path.join(args.data_dir, "breed_map.json"), encoding="utf-8"))
    with open(os.path.join(args.data_dir, "scenarios.json"), encoding="utf-8") as f:
        scenario_meta = {s["scenario_id"]: s for s in json.load(f)["scenarios"]}

    feats = compute_rf_features(events, wear, breed_map)
    labels = run_pipeline(events, wear)[["pet_id", "date", "tier", "red_flags"]]
    table = feats.merge(labels, on=["pet_id", "date"], how="inner")

    rng = np.random.default_rng(args.seed)
    rows = []
    for _, r in table.iterrows():
        if r["tier"] not in ("C1", "C2"):
            continue
        s = scenario_meta.get(r["pet_id"])
        if s is None:
            continue
        qb = s.get("questionnaire_behavior", "not_triggered")
        if qb not in ANSWERED_BEHAVIORS:
            continue
        c_ordinal = TIER_ORDINAL[r["tier"]]
        has_red_flag = isinstance(r["red_flags"], list) and len(r["red_flags"]) > 0
        q = simulate_questionnaire_row(rng, c_ordinal, qb, has_red_flag)
        s_label = true_s_tier(c_ordinal, q["_true_skin_latent"])
        row = {c: r[c] for c in FEATURE_COLUMNS}
        row["pet_id"] = r["pet_id"]
        row["date"] = r["date"]
        row["c_tier"] = r["tier"]
        row["questionnaire_behavior"] = qb
        for qc in QUESTIONNAIRE_FEATURE_COLUMNS:
            row[qc] = q[qc]
        row["s_tier"] = s_label
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = os.path.join(args.data_dir, "model_b_training_table.csv")
    out.to_csv(out_path, index=False)
    print(f"模型B训练表: {len(out)}行, {out.pet_id.nunique()}个场景狗")
    print(f"c_tier分布:\n{out.c_tier.value_counts()}")
    print(f"s_tier分布:\n{out.s_tier.value_counts()}")
    print(f"questionnaire_behavior分布:\n{out.questionnaire_behavior.value_counts()}")
    print(f"保存到 {out_path}")


if __name__ == "__main__":
    main()
