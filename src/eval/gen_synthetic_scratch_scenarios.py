"""
生成覆盖多种场景的合成抓挠事件数据，跑一遍 scratch_burden.py 的 SBS 引擎，
验证机制本身（基线/四个子项/红旗/分档）在各类场景下的行为是否符合预期。

注意：这里验证的是"SBS 打分逻辑"这一层，不是"抓挠事件识别准确率"——
后者依赖真实标注数据，见 docs/skin_health.md §4 待验证参数清单。

用法：
    python src/eval/gen_synthetic_scratch_scenarios.py
"""
import random
from datetime import datetime, timedelta

import pandas as pd

from scratch_burden import run_pipeline

random.seed(42)
N_DAYS = 35          # 21天基线窗口(d-8~d-21) + 14天观察期
DAY0 = datetime(2026, 6, 1)


def _mk_event(day, hour, minute, duration_sec):
    start = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return {"start": start, "end": start + timedelta(seconds=duration_sec), "duration_sec": duration_sec}


def gen_baseline_events(pet_id, n_days, events_per_day, dur_range, hours_range=(8, 20)):
    """生成"日常正常"抓挠：次数、每次时长在给定范围内随机波动。"""
    rows = []
    for d in range(n_days):
        day = DAY0 + timedelta(days=d)
        n = max(0, round(random.gauss(events_per_day, max(0.5, events_per_day * 0.2))))
        for _ in range(n):
            hour = random.randint(*hours_range)
            minute = random.randint(0, 59)
            dur = random.uniform(*dur_range)
            rows.append({"pet_id": pet_id, **_mk_event(day, hour, minute, dur)})
    return rows


def gen_wear_hours(pet_id, n_days, wear_hours=18.0, missing_days=()):
    rows = []
    for d in range(n_days):
        day = (DAY0 + timedelta(days=d)).date()
        h = 0.0 if d in missing_days else wear_hours
        rows.append({"pet_id": pet_id, "date": day, "valid_wear_hours": h})
    return rows


def scenario_stable_low_baseline():
    """场景1：低基线狗，日常小幅波动，全程应保持 C0。"""
    pet = "dog_stable_low"
    events = gen_baseline_events(pet, N_DAYS, events_per_day=2, dur_range=(1.5, 4))
    wear = gen_wear_hours(pet, N_DAYS)
    return pet, events, wear, "全程应保持 C0（低基线、无明显偏离）"


def scenario_sudden_spike_from_zero():
    """场景2：前21天几乎不抓，后14天陡增（对应此前讨论的 0→3/天 场景），测试红旗/基线灵敏度。"""
    pet = "dog_sudden_spike"
    events = gen_baseline_events(pet, BASELINE_DAYS := 21, events_per_day=0.1, dur_range=(1, 2))
    for d in range(BASELINE_DAYS, N_DAYS):
        day = DAY0 + timedelta(days=d)
        for i in range(3):
            events.append({"pet_id": pet, **_mk_event(day, 10 + i * 3, 0, random.uniform(3, 6))})
    wear = gen_wear_hours(pet, N_DAYS)
    return pet, events, wear, "后14天从每天3次开始，应在基线建立后逐步触发 C1/C2（验证低基线狗的相对变化是否能被捕捉到）"


def scenario_gradual_worsening():
    """场景3：从第21天起逐日递增，测试"持续程度"计分。"""
    pet = "dog_gradual_worsening"
    events = gen_baseline_events(pet, 21, events_per_day=2, dur_range=(1.5, 4))
    for d in range(21, N_DAYS):
        day = DAY0 + timedelta(days=d)
        n = 2 + (d - 21)  # 每天多一次
        for i in range(n):
            events.append({"pet_id": pet, **_mk_event(day, 8 + (i % 10), i * 5 % 60, random.uniform(2, 6))})
    wear = gen_wear_hours(pet, N_DAYS)
    return pet, events, wear, "应观察到变化幅度分逐日上升、持续程度分在偏离持续>=3天后达到20"


def scenario_high_baseline_modest_change():
    """场景4：高基线狗，小幅波动（对应 PM 文档示例五：30→35），双门槛应判定为不触发。"""
    pet = "dog_high_baseline_modest"
    events = gen_baseline_events(pet, 21, events_per_day=30 / 12, dur_range=(1, 3), hours_range=(6, 22))
    for d in range(21, N_DAYS):
        day = DAY0 + timedelta(days=d)
        for i in range(3):  # 约35/12次日抓挠水平，小幅高于基线
            events.append({"pet_id": pet, **_mk_event(day, 6 + i, 0, random.uniform(1, 3))})
    # 强制补齐到接近 30/35 次量级（用固定次数场景更好验证，覆盖随机基线生成的波动）
    wear = gen_wear_hours(pet, N_DAYS)
    return pet, events, wear, "PM文档示例五（30→35，倍数1.17）：应判定为无明显变化，变化幅度分=0"


def scenario_clustered_single_day():
    """场景5：总量不多但某天高度聚集在1小时内，测试聚集程度分项。"""
    pet = "dog_clustered"
    events = gen_baseline_events(pet, 21, events_per_day=2, dur_range=(1.5, 4))
    spike_day = DAY0 + timedelta(days=25)
    for i in range(6):  # 1小时内6次，触发聚集
        events.append({"pet_id": pet, **_mk_event(spike_day, 9, i * 8, random.uniform(2, 4))})
    wear = gen_wear_hours(pet, N_DAYS)
    return pet, events, wear, "第25天(day25)应触发聚集程度=10分（1个聚集时段），其他天不触发"


def scenario_night_sleep_disruption():
    """场景6：夜间反复抓挠导致睡眠中断，测试正常行为影响分项。"""
    pet = "dog_night_disruption"
    events = gen_baseline_events(pet, 21, events_per_day=2, dur_range=(1.5, 4))
    bad_night = DAY0 + timedelta(days=28)
    # 23:00、23:20、02:00 三次夜间抓挠：前两次间隔20分钟(<5分钟合并阈值? 20min>5min -> 应算2次独立中断)
    events.append({"pet_id": pet, **_mk_event(bad_night, 23, 0, 5)})
    events.append({"pet_id": pet, **_mk_event(bad_night, 23, 20, 5)})
    events.append({"pet_id": pet, **_mk_event((bad_night + timedelta(days=1)), 2, 0, 5)})
    wear = gen_wear_hours(pet, N_DAYS)
    return pet, events, wear, "第28天应识别出2次睡眠中断（23:00/23:20间隔20min不合并，02:00算第2或第3次），触发中断分>=20"


def scenario_long_continuous_scratch():
    """场景7：单次连续抓挠超过60秒，测试"长时间抓挠"红旗。"""
    pet = "dog_long_scratch"
    events = gen_baseline_events(pet, 21, events_per_day=2, dur_range=(1.5, 4))
    day = DAY0 + timedelta(days=30)
    events.append({"pet_id": pet, **_mk_event(day, 15, 0, 75)})  # 75秒连续抓挠
    wear = gen_wear_hours(pet, N_DAYS)
    return pet, events, wear, "第30天应触发红旗 interrupt_or_long_scratch，正常行为影响分=30"


def scenario_missing_wear_days():
    """场景8：中间若干天设备脱落/未佩戴，测试基线计算和 data_quality_flag 是否正确跳过。"""
    pet = "dog_missing_wear"
    events = gen_baseline_events(pet, N_DAYS, events_per_day=2, dur_range=(1.5, 4))
    missing = {10, 11, 12, 26}
    events = [e for e in events if (e["start"] - DAY0).days not in missing]
    wear = gen_wear_hours(pet, N_DAYS, missing_days=missing)
    return pet, events, wear, "缺失日应标记 insufficient，不应计入基线，也不应被当成'0次抓挠=正常'"


def scenario_cold_start():
    """场景9：设备刚绑定，历史不足21天，测试冷启动时不打分。"""
    pet = "dog_cold_start"
    events = gen_baseline_events(pet, 10, events_per_day=2, dur_range=(1.5, 4))
    wear = gen_wear_hours(pet, 10)
    return pet, events, wear, "前21天内任何一天都应 insufficient_baseline=True，不产出分数"


SCENARIOS = [
    scenario_stable_low_baseline,
    scenario_sudden_spike_from_zero,
    scenario_gradual_worsening,
    scenario_high_baseline_modest_change,
    scenario_clustered_single_day,
    scenario_night_sleep_disruption,
    scenario_long_continuous_scratch,
    scenario_missing_wear_days,
    scenario_cold_start,
]


def main():
    all_events, all_wear, expectations = [], [], {}
    for fn in SCENARIOS:
        pet, events, wear, expect = fn()
        all_events += events
        all_wear += wear
        expectations[pet] = expect

    events_df = pd.DataFrame(all_events)
    wear_df = pd.DataFrame(all_wear)
    result = run_pipeline(events_df, wear_df)

    print("=" * 100)
    for pet, expect in expectations.items():
        print(f"\n### {pet}")
        print(f"预期: {expect}")
        sub = result[result.pet_id == pet].reset_index(drop=True)
        cols = ["date", "total", "tier", "delta_score", "cluster_score",
                "persistence_score", "interrupt_score", "red_flags", "insufficient_baseline"]
        print(sub[cols].to_string(index=False))

    out_path = "data/synthetic_skin_health_sbs_report.csv"
    result.to_csv(out_path, index=False)
    print(f"\n完整结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
