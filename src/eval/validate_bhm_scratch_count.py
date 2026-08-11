"""
用合成数据验证 bhm_scratch_count.py 这套"品种→狗→每日"三层贝叶斯计数模型，
用的是我们影棚实际的4个品种（金毛/中华田园犬/比熊/马尔济斯，大中小体型都有）。

现实里每个品种现在只有1只真实狗，组内方差完全没法估——这里给每个品种额外
"陪跑"几只合成狗，纯粹是为了在合理的样本规模下验证"部分池化"这个机制本身
работает（狗与狗之间借力估计），不代表我们真的有这么多狗，每个品种里名字
带"real_"前缀的才是对应我们影棚实际那4只狗的占位符。

验证三件事（对应 bhm_scratch_count.py 的三个函数）：
  1. 部分池化：数据少的狗，估计值会更靠近品种基线；数据多的狗更靠近自己的真实值
  2. 异常检测：一只狗从某天起患病（抓挠率显著上升），模型能不能用后验预测检验及时发现
  3. 冷启动：一只全新狗，不确定性怎么随佩戴天数从"品种先验"收窄

用法：
    python src/eval/validate_bhm_scratch_count.py
"""
import numpy as np
import pandas as pd

from bhm_scratch_count import fit_model, check_convergence, posterior_predictive_check, sequential_cold_start

rng = np.random.default_rng(42)

# 品种 = 我们影棚实际的4个，大中小体型都覆盖到
BREEDS = ["金毛", "中华田园犬", "比熊", "马尔济斯"]
SYNTHETIC_DOGS_PER_BREED = 9  # 加上1只"real_"占位狗，每个品种凑够10只用于验证
N_DAYS = 35
SICK_BREED = "中华田园犬"
SICK_DOG_LOCAL_IDX = 3
SICK_ONSET_DAY = 20
SICK_EFFECT = 1.1  # log-rate增量，约3倍抓挠频率

mu_pop_true = 1.6       # log(基线抓挠次数/天)，约等于 exp(1.6)≈5次/天
sigma_breed_true = 0.35  # 品种间差异
sigma_dog_true = 0.30    # 品种内个体差异
nb_alpha_true = 6.0      # 越大越接近Poisson，越小越过度离散


def simulate_data():
    breed_effects = {b: rng.normal(0, sigma_breed_true) for b in BREEDS}
    rows, dog_meta = [], []
    for breed in BREEDS:
        for local_idx in range(SYNTHETIC_DOGS_PER_BREED + 1):
            dog_id = f"{'real' if local_idx == 0 else 'syn'}_{breed}_{local_idx}"
            dog_effect = rng.normal(0, sigma_dog_true)
            is_sick = (breed == SICK_BREED and local_idx == SICK_DOG_LOCAL_IDX)
            dog_meta.append({"dog_id": dog_id, "breed": breed, "is_sick_dog": is_sick})
            for day in range(N_DAYS):
                extra = SICK_EFFECT if (is_sick and day >= SICK_ONSET_DAY) else 0.0
                log_rate = mu_pop_true + breed_effects[breed] + dog_effect + extra
                mu = np.exp(log_rate)
                p = nb_alpha_true / (nb_alpha_true + mu)
                count = rng.negative_binomial(nb_alpha_true, p)
                rows.append({"dog_id": dog_id, "breed": breed, "day": day, "scratch_count": count})
    return pd.DataFrame(rows), pd.DataFrame(dog_meta)


def main():
    df, dog_meta = simulate_data()
    print(f"合成数据: {len(df)} 行, {df.dog_id.nunique()} 只狗（每个品种1只real_占位 + "
          f"{SYNTHETIC_DOGS_PER_BREED}只合成陪跑）, {df.breed.nunique()} 个品种")
    print(df.groupby("breed")["scratch_count"].agg(["mean", "std", "count"]))

    sick_dog_id = dog_meta[dog_meta.is_sick_dog].dog_id.values[0]
    print(f"\n合成的'患病狗': {sick_dog_id}（第{SICK_ONSET_DAY}天起抓挠率显著上升）")

    BASELINE_CUTOFF = 19
    train_df = df[df.day < BASELINE_CUTOFF].reset_index(drop=True)

    print("\n拟合三层贝叶斯计数模型（品种→狗→每日）...")
    idata, breed_ids, dog_ids, dog_to_breed = fit_model(train_df)

    max_rhat, n_div = check_convergence(idata)
    print(f"收敛诊断: 最大r_hat={max_rhat:.3f}  发散样本数={n_div}")

    # ── 1. 部分池化：挑几只不同数据量的狗对比 ──
    print("\n" + "=" * 90)
    print("部分池化效果抽样对比（原始均值 vs 模型后验估计）")
    print("=" * 90)
    post = idata.posterior
    for dog_id in dog_meta.dog_id.values[:4]:
        if dog_id not in dog_ids:
            continue
        d_idx = dog_ids.index(dog_id)
        raw_mean = train_df[train_df.dog_id == dog_id].scratch_count.mean()
        n_obs = (train_df.dog_id == dog_id).sum()
        bhm_log_rate = (post["mu_pop"] + post["dog_effect"].sel(dog=d_idx)).mean().values
        bhm_rate = float(np.exp(bhm_log_rate))
        print(f"  {dog_id:20s} (基线期{n_obs}天观测): 原始均值={raw_mean:.2f}  "
              f"BHM后验估计={bhm_rate:.2f}")

    # ── 2. 异常检测：患病狗的完整35天数据做后验预测检验 ──
    print("\n" + "=" * 90)
    print(f"异常检测: {sick_dog_id}（后验预测检验，基于前{BASELINE_CUTOFF}天估计的个体基线）")
    print("=" * 90)
    sick_df = df[df.dog_id == sick_dog_id].sort_values("day")
    sick_d_idx = dog_ids.index(sick_dog_id)
    percentiles, is_anomaly = posterior_predictive_check(
        idata, sick_d_idx, sick_df.scratch_count.values)

    n_false_before = is_anomaly[sick_df.day.values < BASELINE_CUTOFF].sum()
    n_baseline_days = (sick_df.day.values < BASELINE_CUTOFF).sum()
    onset_mask = sick_df.day.values >= SICK_ONSET_DAY
    n_alerts_after = is_anomaly[onset_mask].sum()
    n_days_after = onset_mask.sum()
    first_alert_days = sick_df.day.values[onset_mask][is_anomaly[onset_mask]]
    detect_delay = (first_alert_days.min() - SICK_ONSET_DAY) if len(first_alert_days) else None

    print(f"  基线期({n_baseline_days}天)误报: {n_false_before}天 "
          f"误报率{n_false_before/n_baseline_days*100:.0f}%（理论期望约5%，双边2.5%+2.5%）")
    print(f"  发病期({n_days_after}天)命中: {n_alerts_after}天 "
          f"命中率{n_alerts_after/n_days_after*100:.0f}%")
    print(f"  发病后首次成功报警延迟: "
          f"{f'{detect_delay}天' if detect_delay is not None else '未检出'}")

    # ── 3. 冷启动：模拟一只全新狗，看不确定性怎么收窄 ──
    print("\n" + "=" * 90)
    print("冷启动模拟：一只全新的中华田园犬，从0天到30天")
    print("=" * 90)
    target_breed_idx = breed_ids.index("中华田园犬")
    true_new_dog_rate = np.exp(mu_pop_true + rng.normal(0, sigma_breed_true) - 0.15)
    new_dog_daily_counts = rng.poisson(true_new_dog_rate, size=30)
    checkpoints = [0, 1, 3, 7, 14, 21, 30]
    means, lo, hi = sequential_cold_start(idata, target_breed_idx, new_dog_daily_counts, checkpoints)
    for day, m, l, h in zip(checkpoints, means, lo, hi):
        print(f"  第{day:2d}天: 估计日均抓挠={m:.2f}  95%区间=[{l:.2f}, {h:.2f}]  "
              f"区间宽度={h-l:.2f}")
    print(f"\n  该狗真实基线(仅供验证，模型未知): {true_new_dog_rate:.2f}")
    print("  预期看到区间宽度随天数增加逐渐收窄，第0天的区间应该最宽（完全靠品种先验）")


if __name__ == "__main__":
    main()
