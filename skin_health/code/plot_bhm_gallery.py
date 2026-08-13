"""
给贝叶斯分层模型（bhm_scratch_count.py）出图+保留数据，配合 docs/bhm_gallery.md
使用，跟 plot_scenario_gallery.py（SBS引擎那份画廊）是同一个思路。

场景定义在 gen_bhm_scenarios.py（6个场景狗：急性发作/渐进恶化/发作后恢复/
轻微异常/全程正常真阴性/天然高噪声），单次联合模型拟合，拟合完针对每只场景狗
分别跑后验预测检验出图，另外单独出部分池化（含一只稀疏数据狗）+两个品种的
冷启动对比图。

用法（跑一次MCMC大概40-60秒，场景狗比原来validate脚本多，略慢一点）：
    python skin_health/code/plot_bhm_gallery.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
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

from bhm_scratch_count import fit_model, check_convergence, posterior_predictive_check, sequential_cold_start
from gen_bhm_scenarios import simulate_data, SCENARIOS, BASELINE_CUTOFF, mu_pop_true, sigma_breed_true

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "skin_health", "data", "bhm_validation")
ASSET_DIR = os.path.join(_REPO_ROOT, "skin_health", "docs", "assets")
DOC_PATH = os.path.join(_REPO_ROOT, "skin_health", "docs", "bhm_gallery.md")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

rng = np.random.default_rng(7)

# 稀疏基线狗：从背景种群里挑一只，训练时只给模型看基线期最后5天数据（而不是完整
# 19天），跟同品种、完整19天数据的狗放一起对比，直观看"数据量少不少"这个维度
# 单独对收缩幅度的影响（之前只对比了"原始值离群程度"这一个维度）
SPARSE_DOG_ID = "syn_金毛_5"
SPARSE_DOG_VISIBLE_DAYS = 5


def plot_scenario_timeline(scenario, dog_df, percentiles, is_anomaly, idx, total):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [1, 1]})

    days = dog_df["day"].values
    counts = dog_df["scratch_count"].values
    ax1.bar(days, counts, color="#4a7fb5", width=0.8)
    ax1.axvline(BASELINE_CUTOFF - 0.5, color="#555", linestyle="--", linewidth=1,
               label=f"基线截止（第{BASELINE_CUTOFF}天，之前的数据用来估计个体基线）")
    onset = scenario.get("onset_day")
    if onset is not None:
        ax1.axvline(onset - 0.5, color="#c0392b", linestyle="--", linewidth=1,
                   label=f"异常注入起点（第{onset}天，模型不知道这个信息）")
    if scenario.get("recover_day") is not None:
        ax1.axvline(scenario["recover_day"] - 0.5, color="#4c9a5c", linestyle="--", linewidth=1,
                   label=f"恢复正常起点（第{scenario['recover_day']}天）")
    ax1.set_ylabel("当天抓挠次数（原始观测）")
    ax1.set_title(f"场景{idx}/{total}：{scenario['key']}（{scenario['breed']}）— 每日抓挠次数原始观测",
                  fontsize=10, loc="left")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlabel("day（横轴每格=1天，共35天）")

    colors = ["#c0392b" if a else "#4c9a5c" for a in is_anomaly]
    ax2.scatter(days, percentiles, c=colors, s=30, zorder=3)
    ax2.plot(days, percentiles, color="#999", linewidth=0.8, zorder=2)
    ax2.axhspan(2.5, 97.5, color="#4c9a5c", alpha=0.06)
    ax2.axhline(97.5, color="#c0392b", linewidth=0.8, linestyle="--")
    ax2.axhline(2.5, color="#c0392b", linewidth=0.8, linestyle="--")
    ax2.axvline(BASELINE_CUTOFF - 0.5, color="#555", linestyle="--", linewidth=1)
    if onset is not None:
        ax2.axvline(onset - 0.5, color="#c0392b", linestyle="--", linewidth=1)
    if scenario.get("recover_day") is not None:
        ax2.axvline(scenario["recover_day"] - 0.5, color="#4c9a5c", linestyle="--", linewidth=1)
    ax2.set_ylim(-3, 103)
    ax2.set_ylabel("后验预测百分位")
    ax2.set_xlabel("day（横轴每格=1天，共35天）")
    ax2.set_title("红点=判定异常（百分位>97.5或<2.5，双边检验），绿点=正常范围内",
                  fontsize=10, loc="left")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c9a5c", markersize=8, label="正常"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b", markersize=8, label="判定异常"),
    ]
    ax2.legend(handles=legend_handles, loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(ASSET_DIR, f"bhm_scenario_{scenario['key']}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return os.path.relpath(out_path, os.path.dirname(DOC_PATH))


def plot_partial_pooling(pooling_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pooling_df))
    w = 0.35
    ax.bar(x - w / 2, pooling_df["raw_mean"], width=w, color="#9aa0a6",
          label="原始均值（只看这只狗自己能看到的数据）")
    ax.bar(x + w / 2, pooling_df["bhm_estimate"], width=w, color="#4a7fb5",
          label="BHM后验估计（借了品种层的力）")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.dog_id}\n({r.n_obs}天数据)" for r in pooling_df.itertuples()],
                       fontsize=8, rotation=0)
    ax.set_ylabel("日均抓挠次数")
    ax.set_title("部分池化效果：前4只是同品种(金毛)内数据量一致(19天)但原始均值不同的狗，\n"
                 "最后1只是数据量少很多(仅5天)的狗——离群程度和数据量两个维度分别看收缩幅度",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(ASSET_DIR, "bhm_partial_pooling.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return os.path.relpath(out_path, os.path.dirname(DOC_PATH))


def plot_cold_start_compare(results):
    """results: list of (breed, checkpoints, means, lo, hi, true_rate)"""
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 5), sharey=False)
    if len(results) == 1:
        axes = [axes]
    for ax, (breed, checkpoints, means, lo, hi, true_rate) in zip(axes, results):
        means, lo, hi = np.array(means), np.array(lo), np.array(hi)
        ax.fill_between(checkpoints, lo, hi, color="#4a7fb5", alpha=0.2, label="95%可信区间")
        ax.plot(checkpoints, means, color="#4a7fb5", marker="o", label="估计日均抓挠次数")
        ax.axhline(true_rate, color="#c0392b", linestyle="--", linewidth=1,
                  label=f"真实基线={true_rate:.2f}（仅供验证）")
        ax.set_xlabel("佩戴天数")
        ax.set_ylabel("日均抓挠次数估计")
        ax.set_title(f"全新{breed}的冷启动收窄过程", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(ASSET_DIR, "bhm_cold_start.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return os.path.relpath(out_path, os.path.dirname(DOC_PATH))


def main():
    df, dog_meta = simulate_data()
    print(f"合成数据: {len(df)} 行, {df.dog_id.nunique()} 只狗"
          f"（背景种群 + {len(SCENARIOS)} 只场景狗）, {df.breed.nunique()} 个品种")

    train_df = df[df.day < BASELINE_CUTOFF].reset_index(drop=True)
    # 稀疏基线狗：只保留基线期最后5天，模拟"这只狗刚绑设备没多久、数据量本来就少"
    sparse_cutoff_day = BASELINE_CUTOFF - SPARSE_DOG_VISIBLE_DAYS
    train_df_for_fit = train_df[
        ~((train_df.dog_id == SPARSE_DOG_ID) & (train_df.day < sparse_cutoff_day))
    ].reset_index(drop=True)

    print(f"拟合三层贝叶斯计数模型（品种→狗→每日），{train_df_for_fit.dog_id.nunique()}只狗训练数据，大概1分钟...")
    idata, breed_ids, dog_ids, dog_to_breed = fit_model(train_df_for_fit)
    max_rhat, n_div = check_convergence(idata)
    print(f"收敛诊断: 最大r_hat={max_rhat:.3f}  发散样本数={n_div}")

    df.to_csv(os.path.join(DATA_DIR, "simulated_full_data.csv"), index=False)
    dog_meta.to_csv(os.path.join(DATA_DIR, "dog_meta.csv"), index=False)
    train_df_for_fit.to_csv(os.path.join(DATA_DIR, "train_data_baseline_period.csv"), index=False)

    post = idata.posterior

    # ── 部分池化：同品种4只满数据狗 + 1只稀疏数据狗 ──
    sample_dogs = dog_meta[(dog_meta.breed == "金毛") & (dog_meta.scenario_key.isna())].dog_id.values[:4]
    pooling_rows = []
    for dog_id in list(sample_dogs) + [SPARSE_DOG_ID]:
        d_idx = dog_ids.index(dog_id)
        visible = train_df_for_fit[train_df_for_fit.dog_id == dog_id]
        raw_mean = visible.scratch_count.mean()
        n_obs = len(visible)
        bhm_log_rate = (post["mu_pop"] + post["dog_effect"].sel(dog=d_idx)).mean().values
        bhm_rate = float(np.exp(bhm_log_rate))
        pooling_rows.append({"dog_id": dog_id, "n_obs": int(n_obs), "raw_mean": raw_mean, "bhm_estimate": bhm_rate})
    pooling_df = pd.DataFrame(pooling_rows)
    pooling_df.to_csv(os.path.join(DATA_DIR, "partial_pooling_comparison.csv"), index=False)
    pooling_img = plot_partial_pooling(pooling_df)

    # ── 逐场景异常检测 ──
    scenario_summaries = []
    scenario_imgs = {}
    for i, s in enumerate(SCENARIOS, start=1):
        dog_id = f"scenario_{s['key']}"
        dog_df = df[df.dog_id == dog_id].sort_values("day")
        d_idx = dog_ids.index(dog_id)
        percentiles, is_anomaly = posterior_predictive_check(idata, d_idx, dog_df.scratch_count.values)
        out = dog_df.copy()
        out["ppc_percentile"] = percentiles
        out["is_anomaly"] = is_anomaly
        out.to_csv(os.path.join(DATA_DIR, f"scenario_{s['key']}.csv"), index=False)
        img = plot_scenario_timeline(s, dog_df, percentiles, is_anomaly, i, len(SCENARIOS))
        scenario_imgs[s["key"]] = img

        days = dog_df.day.values
        n_baseline = (days < BASELINE_CUTOFF).sum()
        n_false_before = is_anomaly[days < BASELINE_CUTOFF].sum()
        onset = s.get("onset_day")
        if onset is not None:
            after_mask = days >= onset
            n_after = after_mask.sum()
            n_alerts_after = is_anomaly[after_mask].sum()
            first_alert = days[after_mask][is_anomaly[after_mask]]
            delay = int(first_alert.min() - onset) if len(first_alert) else None
        else:
            n_after = n_alerts_after = None
            delay = None
        scenario_summaries.append({
            "key": s["key"], "breed": s["breed"], "desc": s["desc"],
            "n_baseline_days": int(n_baseline), "n_false_before": int(n_false_before),
            "n_after_days": n_after, "n_alerts_after": n_alerts_after, "detect_delay": delay,
        })

    # ── 冷启动：两个品种对比 ──
    cold_start_results = []
    cold_start_rows = []
    for breed in ["中华田园犬", "比熊"]:
        target_breed_idx = breed_ids.index(breed)
        true_rate = np.exp(mu_pop_true + rng.normal(0, sigma_breed_true) - 0.15)
        daily_counts = rng.poisson(true_rate, size=30)
        checkpoints = [0, 1, 3, 7, 14, 21, 30]
        means, lo, hi = sequential_cold_start(idata, target_breed_idx, daily_counts, checkpoints)
        cold_start_results.append((breed, checkpoints, means, lo, hi, true_rate))
        for day, m, l, h in zip(checkpoints, means, lo, hi):
            cold_start_rows.append({"breed": breed, "day": day, "estimate": m, "lo95": l, "hi95": h,
                                    "true_rate": true_rate})
    cold_start_df = pd.DataFrame(cold_start_rows)
    cold_start_df.to_csv(os.path.join(DATA_DIR, "cold_start_new_dog.csv"), index=False)
    cold_start_img = plot_cold_start_compare(cold_start_results)

    # ── 汇总md ──
    doc_lines = [
        "# 贝叶斯分层模型（BHM）验证画廊",
        "",
        "> 对应 `bhm_scratch_count.py`（品种→狗→每日三层贝叶斯计数模型）+ "
        "`gen_bhm_scenarios.py`（场景定义）。跟`scenario_gallery.md`（SBS规则引擎"
        "那份画廊）是同一个思路，验证的是贝叶斯分层模型这条路，两者互补不是替代"
        "（什么时候该上哪个见`model_roadmap.md`）。",
        "",
        f"背景种群：4个品种（金毛/中华田园犬/比熊/马尔济斯），每个品种1只real_占位"
        "（对应影棚实际那4只狗） + 9只syn_陪跑（现实里每品种只有1只真狗，组内方差"
        f"没法估，陪跑狗纯粹用于验证机制），再额外加{len(SCENARIOS)}只场景狗，"
        "单次联合拟合一个模型，拟合完分别检验每只场景狗在自己的场景下表现如何。",
        "",
        f"本次拟合收敛诊断：最大r_hat={max_rhat:.3f}（越接近1.0越好，一般<1.01算收敛），"
        f"发散样本数={n_div}（越接近0越好）。",
        "",
        "## 场景总览",
        "",
        "| # | 场景 | 品种 | 说明 |",
        "|---|---|---|---|",
        *[f"| {i} | {s['key']} | {s['breed']} | {s['desc']} |" for i, s in enumerate(SCENARIOS, start=1)],
        "",
        "---",
        "",
    ]

    for i, summary in enumerate(scenario_summaries, start=1):
        doc_lines += [f"## 场景{i}/{len(SCENARIOS)}：{summary['key']}（{summary['breed']}）", ""]
        doc_lines += [f"**说明**：{summary['desc']}", ""]
        doc_lines += [f"![{summary['key']}]({scenario_imgs[summary['key']]})", ""]
        doc_lines += [
            f"- 基线期({summary['n_baseline_days']}天)误报: {summary['n_false_before']}天 "
            f"误报率{summary['n_false_before']/summary['n_baseline_days']*100:.0f}%"
            "（理论期望约5%，双边2.5%+2.5%）",
        ]
        if summary["n_after_days"] is not None:
            delay_text = f"{summary['detect_delay']}天" if summary['detect_delay'] is not None else "未检出"
            doc_lines += [
                f"- 异常期({summary['n_after_days']}天)命中: {summary['n_alerts_after']}天 "
                f"命中率{summary['n_alerts_after']/summary['n_after_days']*100:.0f}%",
                f"- 首次成功报警延迟: {delay_text}",
            ]
        else:
            doc_lines += ["- 全程未注入异常，此项为真阴性对照，只看基线期误报率是否符合理论期望"]
        doc_lines += [
            "",
            f"数据：[逐日百分位+异常判定](../data/bhm_validation/scenario_{summary['key']}.csv)",
            "",
            "---",
            "",
        ]

    doc_lines += [
        "## 部分池化（partial pooling）——数据量少/原始值离群的狗，估计值怎么被拉回品种基线",
        "",
        "前4只是同品种（金毛）内基线期数据量一致（19天）但原始均值不同的狗，"
        "最后1只（" + SPARSE_DOG_ID + "）是数据量少很多（只给模型看5天数据，"
        "模拟\"刚绑设备没多久\"）的狗——原始值离群程度和数据量两个维度分别看"
        "收缩幅度：离群越远/数据越少，蓝柱（模型估计）偏离灰柱（原始均值）越明显。",
        "",
        f"![部分池化]({pooling_img})",
        "",
        "| 狗 | 训练时可见天数 | 原始均值 | BHM后验估计 |",
        "|---|---|---|---|",
        *[f"| {r.dog_id} | {r.n_obs} | {r.raw_mean:.2f} | {r.bhm_estimate:.2f} |"
          for r in pooling_df.itertuples()],
        "",
        "数据：[部分池化对比表](../data/bhm_validation/partial_pooling_comparison.csv) | "
        "[基线期训练数据](../data/bhm_validation/train_data_baseline_period.csv)",
        "",
        "---",
        "",
        "## 冷启动——两个品种对比，全新狗的不确定性怎么随佩戴天数收窄",
        "",
        "同样用Gamma-Poisson共轭近似（不用每天重跑完整MCMC），对比两个方差/均值"
        "水平不同的品种（中华田园犬 vs 比熊），全新狗的95%可信区间从第0天（纯品种"
        "先验，最宽）到第30天怎么收窄——品种间基线/方差本身不同，收窄的绝对宽度"
        "和起点也会不一样，这是预期内的正常现象。",
        "",
        f"![冷启动对比]({cold_start_img})",
        "",
        "数据：[两品种逐checkpoint明细](../data/bhm_validation/cold_start_new_dog.csv)",
        "",
        "---",
        "",
        "完整合成数据：[全部品种全部狗35天数据](../data/bhm_validation/simulated_full_data.csv) | "
        "[狗-品种-场景对照表](../data/bhm_validation/dog_meta.csv)",
        "",
    ]

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines))
    print(f"已生成 {DOC_PATH}，图片在 {ASSET_DIR}，CSV在 {DATA_DIR}")


if __name__ == "__main__":
    main()
