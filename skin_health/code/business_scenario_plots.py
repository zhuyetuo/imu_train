"""
用于回答几个具体业务问题的实验+出图，配合 docs/bayesian_model_business_guide.md 使用：
  1. 群体阈值要多少只狗才可信？（不同规模的品种群体分别拟合，看群体层参数的
     后验不确定性怎么随狗的数量收窄）
  2. 个体判断要多少天才可信？（单只狗的冷启动曲线，扩展到3/7/14/30/60/90/180天）
  3. 贝叶斯先验的"有效样本量"随实际数据积累怎么被稀释（回答"多少数据后
     该更快/更慢调整阈值"这个问题）

都是在合成数据上跑的实验，用来说明"机制的形状"，不是真实的产品参数。

用法：
    python skin_health/code/business_scenario_plots.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

for cand in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

from bhm_scratch_count import fit_model, sequential_cold_start

rng = np.random.default_rng(42)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(OUT_DIR, exist_ok=True)

BREEDS = ["金毛", "中华田园犬", "比熊", "马尔济斯"]
mu_pop_true = 1.6
sigma_breed_true = 0.35
sigma_dog_true = 0.30
nb_alpha_true = 6.0
N_DAYS = 25


def simulate_fleet(dogs_per_breed):
    breed_effects = {b: rng.normal(0, sigma_breed_true) for b in BREEDS}
    rows = []
    for breed in BREEDS:
        for local_idx in range(dogs_per_breed):
            dog_id = f"{breed}_{local_idx}"
            dog_effect = rng.normal(0, sigma_dog_true)
            for day in range(N_DAYS):
                log_rate = mu_pop_true + breed_effects[breed] + dog_effect
                mu = np.exp(log_rate)
                p = nb_alpha_true / (nb_alpha_true + mu)
                count = rng.negative_binomial(nb_alpha_true, p)
                rows.append({"dog_id": dog_id, "breed": breed, "day": day, "scratch_count": count})
    return pd.DataFrame(rows)


# ── 图A：群体规模 vs 群体层参数的可信度 ──────────────────────────────────
def chart_a_fleet_size_vs_confidence():
    print("图A：拟合不同规模的狗群体，看群体层参数的后验不确定性...")
    fleet_sizes_per_breed = [1, 3, 6, 12, 25]  # 总狗数 = 这个数 x 4个品种
    sigma_breed_sds, mu_pop_sds, total_dogs_list = [], [], []

    for n_per_breed in fleet_sizes_per_breed:
        df = simulate_fleet(n_per_breed)
        total_dogs = df.dog_id.nunique()
        idata, breed_ids, dog_ids, dog_to_breed = fit_model(
            df, draws=500, tune=500, chains=2, seed=42)
        post = idata.posterior
        sigma_breed_sds.append(float(post["sigma_breed"].std().values))
        mu_pop_sds.append(float(post["mu_pop"].std().values))
        total_dogs_list.append(total_dogs)
        print(f"  {total_dogs}只狗: sigma_breed后验标准差={sigma_breed_sds[-1]:.3f}  "
              f"mu_pop后验标准差={mu_pop_sds[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(total_dogs_list, sigma_breed_sds, "-o", color="#d62728",
            label="品种间差异(sigma_breed)的后验不确定性")
    ax.plot(total_dogs_list, mu_pop_sds, "-s", color="#1f77b4",
            label="群体均值(mu_pop)的后验不确定性")
    ax.set_xlabel("狗的总数量（4个品种加起来）")
    ax.set_ylabel("参数后验标准差（越低=估计越可信）")
    ax.set_title("群体层面参数的可信度，随狗的数量增加而提高\n"
                  "（用于回答：多少只狗之后，'自己的群体阈值'才能替代借来的外部阈值）")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "chart_a_fleet_size_vs_confidence.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  已保存: {out}")
    return total_dogs_list, sigma_breed_sds


# ── 图B：单只狗的冷启动曲线，扩展到180天 ────────────────────────────────
def chart_b_individual_cold_start():
    print("\n图B：单只新狗的个体判断可信度，随佩戴天数变化...")
    # 用一个中等规模的群体（每品种12只）当"已经比较成熟"的群体先验来源
    df = simulate_fleet(12)
    idata, breed_ids, dog_ids, dog_to_breed = fit_model(df, draws=800, tune=800, chains=2, seed=1)

    target_breed_idx = breed_ids.index("中华田园犬")
    true_new_dog_rate = np.exp(mu_pop_true + rng.normal(0, sigma_breed_true) - 0.1)
    new_dog_daily_counts = rng.poisson(true_new_dog_rate, size=180)

    checkpoints = [0, 1, 3, 7, 14, 21, 30, 60, 90, 180]
    means, lo, hi = sequential_cold_start(idata, target_breed_idx, new_dog_daily_counts, checkpoints)
    width = [h - l for h, l in zip(hi, lo)]
    # "个体判断可信度"用区间宽度相对最终(180天)宽度的比例来定义，方便讲清楚
    # "已经收敛到接近最终精度的百分之多少"这个业务问题
    final_width = width[-1]
    confidence_pct = [max(0, 100 * (1 - (w - final_width) / (width[0] - final_width))) for w in width]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    ax.plot(checkpoints, means, "-o", color="#1f77b4", label="个体基线后验均值")
    ax.fill_between(checkpoints, lo, hi, alpha=0.25, color="#1f77b4", label="95%置信区间")
    ax.axhline(true_new_dog_rate, color="#d62728", linestyle=":", label="真实基线(仅供验证)")
    ax.set_xlabel("已佩戴天数")
    ax.set_ylabel("估计的日均抓挠次数")
    ax.set_title("个体基线估计值 + 不确定性区间")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(checkpoints, confidence_pct, "-o", color="#2ca02c")
    for day, pct in zip(checkpoints, confidence_pct):
        ax2.annotate(f"{pct:.0f}%", (day, pct), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax2.axhline(90, color="gray", linestyle="--", alpha=0.6, label="90%参考线")
    ax2.set_xlabel("已佩戴天数")
    ax2.set_ylabel("相对最终精度的收敛百分比")
    ax2.set_title("个体判断可信度收敛曲线\n(回答：多少天后个体判断已经'足够可信')")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "chart_b_individual_cold_start.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  已保存: {out}")
    for day, pct, w in zip(checkpoints, confidence_pct, width):
        print(f"  第{day:3d}天: 收敛度={pct:5.1f}%  区间宽度={w:.2f}")
    return checkpoints, confidence_pct


# ── 图C：先验的"有效样本量"随实际数据积累被稀释 ──────────────────────────
def chart_c_prior_dilution():
    print("\n图C：先验权重 vs 实际数据权重，随数据量变化...")
    # Gamma-Poisson共轭：先验相当于k0个"虚拟观测"，实际数据每多1天就多1个真实观测，
    # "先验还占多少权重"直接等于 k0 / (k0 + n)，用来回答"数据量多大之后就该
    # 主要相信数据、少相信先验（也就是阈值该变得更快）"这个问题。
    df = simulate_fleet(12)
    idata, breed_ids, dog_ids, dog_to_breed = fit_model(df, draws=800, tune=800, chains=2, seed=1)
    target_breed_idx = breed_ids.index("中华田园犬")
    post = idata.posterior
    breed_log_rate_post = (post["mu_pop"] + post["breed_effect"].sel(breed=target_breed_idx)).stack(
        sample=("chain", "draw")).values
    sigma_dog_post = post["sigma_dog"].stack(sample=("chain", "draw")).values
    individual_draw = rng.normal(0, sigma_dog_post)
    prior_samples = breed_log_rate_post + individual_draw
    prior_mean = np.exp(prior_samples).mean()
    prior_var = np.exp(prior_samples).var()
    theta0 = prior_var / prior_mean
    k0 = prior_mean / theta0  # 先验的"等效观测天数"

    n_days = np.arange(0, 61)
    prior_weight_pct = k0 / (k0 + n_days) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(n_days, prior_weight_pct, color="#9467bd", linewidth=2, label="先验(群体经验)权重")
    ax.plot(n_days, 100 - prior_weight_pct, color="#ff7f0e", linewidth=2, label="这只狗自己数据的权重")
    ax.axvline(k0, color="gray", linestyle="--", alpha=0.6,
               label=f"先验的'等效观测天数' ≈ {k0:.1f}天（此处两条线各占50%）")
    ax.set_xlabel("这只狗已经佩戴的天数")
    ax.set_ylabel("权重占比 (%)")
    ax.set_title("个体判断里，'群体先验'和'自己数据'各占多少权重\n"
                  "(回答：为什么早期该让阈值变化更快，后期该让阈值变化更慢)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "chart_c_prior_dilution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  已保存: {out}  (先验等效观测天数 k0={k0:.2f})")
    return k0


if __name__ == "__main__":
    chart_a_fleet_size_vs_confidence()
    chart_b_individual_cold_start()
    chart_c_prior_dilution()
    print("\n全部完成，图表在", OUT_DIR)
