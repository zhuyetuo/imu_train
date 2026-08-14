"""
把 scenario spec（见 rf_synthetic_scenarios.json，由多个agent设计后汇总）转成
逐事件的合成抓挠数据 + 佩戴时长表，喂给 rf_features.py 算特征、scratch_burden.py
算SBS真值标签，最终训练 HistGradientBoostingClassifier（train_rf_model_a.py）。

设计原则（跟用户要求对应）：
  1. 合成数据可复现——所有随机性都用每个场景自己的固定seed（从scenario_id哈希出来），
     同一份scenario json重跑输出完全一致
  2. 覆盖多种场景（不同pattern/品种/基线/噪声/佩戴质量/问答行为组合）
  3. 生成的CSV全部保存到 skin_health/data/rf_synthetic/，随代码一起提交

用法：
    python skin_health/code/gen_rf_synthetic_scenarios.py \
        --scenarios skin_health/data/rf_synthetic/scenarios.json \
        --out_dir skin_health/data/rf_synthetic
"""
import argparse
import hashlib
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

DAY0 = date(2026, 1, 1)  # 合成数据统一起点，跟真实日期无关，只是给时间戳一个锚点
NIGHT_HOURS = set(list(range(22, 24)) + list(range(0, 6)))
DAY_HOURS = [h for h in range(24) if h not in NIGHT_HOURS]


def _seed_for(scenario_id):
    return int(hashlib.sha256(scenario_id.encode()).hexdigest(), 16) % (2**32)


def _multiplier(day, s):
    """给定第几天(0-based)，返回这天相对基线的严重度倍数。"""
    pattern = s["pattern"]
    sev = s["severity_multiplier"]
    if pattern == "stable":
        return 1.0
    if pattern == "chronic_high_from_start":
        return sev
    if pattern == "step_onset":
        return sev if day >= s["onset_day"] else 1.0
    if pattern == "ramp_onset":
        onset, peak = s["onset_day"], s["peak_day"]
        if day < onset:
            return 1.0
        if day >= peak:
            return sev
        return 1.0 + (sev - 1.0) * (day - onset) / max(1, peak - onset)
    if pattern == "spike_recover":
        onset, recover = s["onset_day"], s["recover_day"]
        return sev if onset <= day < recover else 1.0
    if pattern == "gradual_decline_recovery":
        onset, recover = s["onset_day"], s["recover_day"]
        if day < onset:
            return sev
        if day >= recover:
            return 1.0
        return sev - (sev - 1.0) * (day - onset) / max(1, recover - onset)
    if pattern == "single_day_spike":
        return sev if day == s["spike_day"] else 1.0
    if pattern == "cyclical":
        period, n_cycles = s["cycle_period_days"], s["n_cycles"]
        onset = s.get("onset_day") or 0
        if day < onset:
            return 1.0
        cycle_pos = (day - onset) % period
        in_high_phase = cycle_pos < period // 2
        cycles_elapsed = (day - onset) // period
        return sev if (in_high_phase and cycles_elapsed < n_cycles) else 1.0
    raise ValueError(f"未知pattern: {pattern}")


def _sample_event_count(rng, base_rate, mult, noise_dispersion):
    """noise_dispersion控制天然过度离散程度：low接近Poisson，high明显过度离散
    （用负二项，alpha小=离散度大）。"""
    lam = max(0.05, base_rate * mult)
    if noise_dispersion == "low":
        return int(rng.poisson(lam))
    alpha = {"medium": 4.0, "high": 1.2}[noise_dispersion]
    p = alpha / (alpha + lam)
    return int(rng.negative_binomial(alpha, p))


def _sample_event_hour(rng, night_bias, is_anomaly_extra):
    """is_anomaly_extra=True表示这是"异常期新增"的事件，按night_bias决定夜间/白天；
    基线事件固定用较低的天然夜间占比(0.15)，模拟正常昼夜作息。"""
    p_night = night_bias if is_anomaly_extra else 0.15
    if rng.random() < p_night:
        return int(rng.choice(sorted(NIGHT_HOURS)))
    return int(rng.choice(DAY_HOURS))


def simulate_scenario(s):
    rng = np.random.default_rng(_seed_for(s["scenario_id"]))
    pet_id = s["scenario_id"]
    n_days = s["n_days"]
    base_rate = s.get("baseline_events_per_day", 5.0)
    dur_min = s.get("baseline_duration_sec_min", 1.5)
    dur_max = s.get("baseline_duration_sec_max", 6.0)
    noise = s.get("noise_dispersion", "medium")
    night_bias = s.get("night_bias", 0.2)
    clustering = s.get("clustering_bias", "diffuse")
    wear_quality = s.get("wear_quality", "good")
    sparse_history = wear_quality == "sparse_history"

    events, wear_rows = [], []
    sleep_interrupt_done = False
    long_scratch_done = False

    # sparse_history: 整条时间线本身就很短（截断到21天以内），模拟"刚绑设备没多久"
    effective_days = min(n_days, 9) if sparse_history else n_days

    for day in range(effective_days):
        d = DAY0 + timedelta(days=day)
        mult = _multiplier(day, s)
        is_anomaly_day = mult > 1.0001 or (s["pattern"] == "gradual_decline_recovery" and mult > 1.0001)

        # ── 佩戴时长 ──
        if wear_quality == "off_body_days" and day % 9 in (3, 4):
            wear_h = 0.0
        elif wear_quality == "partial" and day % 5 == 0:
            wear_h = float(rng.uniform(2, 9))
        elif wear_quality == "loose_days" and day % 7 in (2,):
            wear_h = float(rng.uniform(8, 14))  # 松动期间有效佩戴打折但没归零
        else:
            wear_h = float(rng.uniform(16, 23))
        wear_rows.append({"pet_id": pet_id, "date": d, "valid_wear_hours": round(wear_h, 2)})

        if wear_h <= 0:
            continue  # 完全没数据的天不生成事件

        n_events = _sample_event_count(rng, base_rate, mult, noise)
        n_extra = max(0, n_events - int(round(base_rate))) if is_anomaly_day else 0

        # 需要强制注入的红旗信号，优先在这天安排（如果这天恰好是异常期）
        force_sleep_interrupt = (
            s.get("sleep_interrupt_injection") and is_anomaly_day and not sleep_interrupt_done
            and day == (s.get("onset_day") or day)
        )
        force_long_scratch = (
            s.get("long_scratch_injection") and is_anomaly_day and not long_scratch_done
            and day == (s.get("onset_day") or day) + 1
        )

        if clustering == "clustered" and n_extra >= 3:
            cluster_start_hour = int(rng.choice(DAY_HOURS if rng.random() > night_bias else sorted(NIGHT_HOURS)))
            base_minute = 0
            for i in range(n_events):
                if i < n_extra:
                    minute = min(59, base_minute + i * 6)
                    hour = cluster_start_hour
                else:
                    hour = int(rng.choice(DAY_HOURS))
                    minute = int(rng.integers(0, 60))
                start = datetime(d.year, d.month, d.day, hour % 24, minute, int(rng.integers(0, 60)))
                dur = float(rng.uniform(dur_min, dur_max)) * (1.0 + 0.3 * (mult - 1.0) if is_anomaly_day else 1.0)
                events.append({"pet_id": pet_id, "start": start,
                              "end": start + timedelta(seconds=dur), "duration_sec": dur})
        else:
            for i in range(n_events):
                is_extra = i < n_extra
                hour = _sample_event_hour(rng, night_bias, is_extra)
                minute = int(rng.integers(0, 60))
                start = datetime(d.year, d.month, d.day, hour, minute, int(rng.integers(0, 60)))
                dur = float(rng.uniform(dur_min, dur_max)) * (1.0 + 0.3 * (mult - 1.0) if is_anomaly_day else 1.0)
                events.append({"pet_id": pet_id, "start": start,
                              "end": start + timedelta(seconds=dur), "duration_sec": dur})

        if force_sleep_interrupt:
            start = datetime(d.year, d.month, d.day, 2, 0, 0)
            events.append({"pet_id": pet_id, "start": start,
                          "end": start + timedelta(seconds=5), "duration_sec": 5.0})
            sleep_interrupt_done = True

        if force_long_scratch:
            start = datetime(d.year, d.month, d.day, 15, 0, 0)
            events.append({"pet_id": pet_id, "start": start,
                          "end": start + timedelta(seconds=75), "duration_sec": 75.0})
            long_scratch_done = True

    events_df = pd.DataFrame(events, columns=["pet_id", "start", "end", "duration_sec"])
    wear_df = pd.DataFrame(wear_rows, columns=["pet_id", "date", "valid_wear_hours"])
    return events_df, wear_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True, help="scenario spec JSON路径")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    with open(args.scenarios, encoding="utf-8") as f:
        scenarios = json.load(f)["scenarios"]

    os.makedirs(args.out_dir, exist_ok=True)
    events_dir = os.path.join(args.out_dir, "events")
    wear_dir = os.path.join(args.out_dir, "wear")
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(wear_dir, exist_ok=True)

    all_events, all_wear, meta_rows = [], [], []
    for s in scenarios:
        events_df, wear_df = simulate_scenario(s)
        events_df.to_csv(os.path.join(events_dir, f"{s['scenario_id']}_events.csv"), index=False)
        wear_df.to_csv(os.path.join(wear_dir, f"{s['scenario_id']}_wear.csv"), index=False)
        all_events.append(events_df)
        all_wear.append(wear_df)
        meta_rows.append({
            "scenario_id": s["scenario_id"], "theme": s.get("theme", ""), "breed": s.get("breed", ""),
            "pattern": s["pattern"], "severity_multiplier": s["severity_multiplier"],
            "wear_quality": s.get("wear_quality", "good"), "noise_dispersion": s.get("noise_dispersion", "medium"),
            "questionnaire_behavior": s.get("questionnaire_behavior", "not_triggered"),
            "n_events_generated": len(events_df), "n_days_generated": len(wear_df),
            "description": s.get("description", ""),
        })

    combined_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    combined_wear = pd.concat(all_wear, ignore_index=True) if all_wear else pd.DataFrame()
    combined_events.to_csv(os.path.join(args.out_dir, "all_events.csv"), index=False)
    combined_wear.to_csv(os.path.join(args.out_dir, "all_wear.csv"), index=False)
    pd.DataFrame(meta_rows).to_csv(os.path.join(args.out_dir, "scenario_meta.csv"), index=False)

    breed_map = {s["scenario_id"]: s.get("breed", "未知") for s in scenarios}
    with open(os.path.join(args.out_dir, "breed_map.json"), "w", encoding="utf-8") as f:
        json.dump(breed_map, f, ensure_ascii=False, indent=2)

    print(f"生成 {len(scenarios)} 个场景，共 {len(combined_events)} 个事件，"
          f"{len(combined_wear)} 天佩戴记录，保存到 {args.out_dir}")


if __name__ == "__main__":
    main()
