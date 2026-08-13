"""
给 validate_bhm_scratch_count.py 验证的"品种→狗→每日"三层贝叶斯计数模型
（bhm_scratch_count.py）出图+保留数据，配合 docs/bhm_gallery.md 使用，跟
plot_scenario_gallery.py（SBS引擎那份画廊）是同一个思路，但这边是给BHM的：
  1. 部分池化：数据量不同的狗，模型估计值怎么在"自己的原始均值"和"品种基线"
     之间取舍
  2. 异常检测：一只"患病"狗从某天起抓挠率上升，后验预测检验能不能及时发现，
     基线期误报率是不是符合双边检验的理论值
  3. 冷启动：一只全新狗，不确定性区间怎么随佩戴天数从"只有品种先验"收窄到
     "有个体数据支撑"

场景定义、合成数据生成、模型拟合全部复用 validate_bhm_scratch_count.py 里
已经写好的逻辑，这里只多做出图+存数据+汇总md三件事。

用法（跑一次MCMC大概40秒左右）：
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
from validate_bhm_scratch_count import (
    simulate_data, BREEDS, SICK_BREED, SICK_ONSET_DAY, mu_pop_true, sigma_breed_true,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "skin_health", "data", "bhm_validation")
ASSET_DIR = os.path.join(_REPO_ROOT, "skin_health", "docs", "assets")
DOC_PATH = os.path.join(_REPO_ROOT, "skin_health", "docs", "bhm_gallery.md")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

BASELINE_CUTOFF = 19
rng = np.random.default_rng(42)


def plot_partial_pooling(train_df, post, dog_ids, sample_dogs):
    """每只样本狗：原始均值 vs BHM后验估计，配对柱状图，直观看'借力'效果——
    数据量固定看不同狗的收缩幅度（这批样本狗基线期观测天数一致，都是同一批
    金毛，方便同条件对比原始值离群程度对收缩幅度的影响）。"""
    rows = []
    for dog_id in sample_dogs:
        d_idx = dog_ids.index(dog_id)
        raw_mean = train_df[train_df.dog_id == dog_id].scratch_count.mean()
        n_obs = (train_df.dog_id == dog_id).sum()
        bhm_log_rate = (post["mu_pop"] + post["dog_effect"].sel(dog=d_idx)).mean().values
        bhm_rate = float(np.exp(bhm_log_rate))
        rows.append({"dog_id": dog_id, "n_obs": int(n_obs), "raw_mean": raw_mean, "bhm_estimate": bhm_rate})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["raw_mean"], width=w, color="#9aa0a6", label="原始均值（只看这只狗自己的数据）")
    ax.bar(x + w / 2, df["bhm_estimate"], width=w, color="#4a7fb5", label="BHM后验估计（借了品种层的力）")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.dog_id}\n(基线期{r.n_obs}天)" for r in df.itertuples()], fontsize=8)
    ax.set_ylabel("日均抓挠次数")
    ax.set_title(f"部分池化效果：{df.iloc[0]['dog_id'].split('_')[1] if '_' in df.iloc[0]['dog_id'] else ''}"
                 f"同品种内几只狗的原始均值 vs 模型估计\n"
                 "（原始值离群越远，BHM估计被往品种基线拉回得越明显——这就是'借力'）",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(ASSET_DIR, "bhm_partial_pooling.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return df, os.path.relpath(out_path, os.path.dirname(DOC_PATH))


def plot_anomaly_detection(sick_df, percentiles, is_anomaly, sick_dog_id):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), gridspec_kw={"height_ratios": [1, 1]})

    days = sick_df["day"].values
    counts = sick_df["scratch_count"].values
    ax1.bar(days, counts, color="#4a7fb5", width=0.8)
    ax1.axvline(BASELINE_CUTOFF - 0.5, color="#555", linestyle="--", linewidth=1,
               label=f"基线截止（第{BASELINE_CUTOFF}天，之前的数据用来估计个体基线）")
    ax1.axvline(SICK_ONSET_DAY - 0.5, color="#c0392b", linestyle="--", linewidth=1,
               label=f"真实发病起点（第{SICK_ONSET_DAY}天起注入抓挠率升高，模型不知道这个信息）")
    ax1.set_ylabel("当天抓挠次数（原始观测）")
    ax1.set_title(f"{sick_dog_id}：每日抓挠次数原始观测", fontsize=10, loc="left")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlabel("day（横轴每格=1天，共35天）")

    colors = ["#c0392b" if a else "#4c9a5c" for a in is_anomaly]
    ax2.scatter(days, percentiles, c=colors, s=30, zorder=3)
    ax2.plot(days, percentiles, color="#999", linewidth=0.8, zorder=2)
    ax2.axhspan(2.5, 97.5, color="#4c9a5c", alpha=0.06)
    ax2.axhline(97.5, color="#c0392b", linewidth=0.8, linestyle="--")
    ax2.axhline(2.5, color="#c0392b", linewidth=0.8, linestyle="--")
    ax2.axvline(BASELINE_CUTOFF - 0.5, color="#555", linestyle="--", linewidth=1)
    ax2.axvline(SICK_ONSET_DAY - 0.5, color="#c0392b", linestyle="--", linewidth=1)
    ax2.set_ylim(-3, 103)
    ax2.set_ylabel("后验预测百分位（这个观测值在'这只狗正常应该是多少'的\n预测分布里排第几百分位）")
    ax2.set_xlabel("day（横轴每格=1天，共35天）")
    ax2.set_title("后验预测检验：红点=判定异常（百分位>97.5或<2.5，双边检验），绿点=正常范围内",
                  fontsize=10, loc="left")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c9a5c", markersize=8, label="正常（2.5~97.5百分位内）"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b", markersize=8, label="判定异常（>97.5或<2.5百分位）"),
    ]
    ax2.legend(handles=legend_handles, loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(ASSET_DIR, "bhm_anomaly_detection.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return os.path.relpath(out_path, os.path.dirname(DOC_PATH))


def plot_cold_start(checkpoints, means, lo, hi, true_rate):
    fig, ax = plt.subplots(figsize=(8, 5))
    means, lo, hi = np.array(means), np.array(lo), np.array(hi)
    ax.fill_between(checkpoints, lo, hi, color="#4a7fb5", alpha=0.2, label="95%可信区间")
    ax.plot(checkpoints, means, color="#4a7fb5", marker="o", label="估计日均抓挠次数")
    ax.axhline(true_rate, color="#c0392b", linestyle="--", linewidth=1,
              label=f"这只狗的真实基线={true_rate:.2f}（仅供验证用，模型看不到这个数）")
    ax.set_xlabel("佩戴天数")
    ax.set_ylabel("日均抓挠次数估计")
    ax.set_title("冷启动：一只全新狗的不确定性区间，随佩戴天数从'只有品种先验'\n收窄到'有个体数据支撑'",
                fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(ASSET_DIR, "bhm_cold_start.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return os.path.relpath(out_path, os.path.dirname(DOC_PATH))


def main():
    df, dog_meta = simulate_data()
    sick_dog_id = dog_meta[dog_meta.is_sick_dog].dog_id.values[0]
    train_df = df[df.day < BASELINE_CUTOFF].reset_index(drop=True)

    print("拟合三层贝叶斯计数模型（品种→狗→每日），大概40秒...")
    idata, breed_ids, dog_ids, dog_to_breed = fit_model(train_df)
    max_rhat, n_div = check_convergence(idata)
    print(f"收敛诊断: 最大r_hat={max_rhat:.3f}  发散样本数={n_div}")

    df.to_csv(os.path.join(DATA_DIR, "simulated_full_data.csv"), index=False)
    dog_meta.to_csv(os.path.join(DATA_DIR, "dog_meta.csv"), index=False)
    train_df.to_csv(os.path.join(DATA_DIR, "train_data_baseline_period.csv"), index=False)

    # ── 1. 部分池化：挑同一个品种里几只数据量一致但原始均值离群程度不同的狗 ──
    post = idata.posterior
    sample_breed = "金毛"
    sample_dogs = dog_meta[dog_meta.breed == sample_breed].dog_id.values[:4]
    pooling_df, pooling_img = plot_partial_pooling(train_df, post, dog_ids, sample_dogs)
    pooling_df.to_csv(os.path.join(DATA_DIR, "partial_pooling_comparison.csv"), index=False)

    # ── 2. 异常检测 ──
    sick_df = df[df.dog_id == sick_dog_id].sort_values("day")
    sick_d_idx = dog_ids.index(sick_dog_id)
    percentiles, is_anomaly = posterior_predictive_check(idata, sick_d_idx, sick_df.scratch_count.values)
    anomaly_out = sick_df.copy()
    anomaly_out["ppc_percentile"] = percentiles
    anomaly_out["is_anomaly"] = is_anomaly
    anomaly_out.to_csv(os.path.join(DATA_DIR, f"anomaly_detection_{sick_dog_id}.csv"), index=False)
    anomaly_img = plot_anomaly_detection(sick_df, percentiles, is_anomaly, sick_dog_id)

    n_baseline_days = (sick_df.day.values < BASELINE_CUTOFF).sum()
    n_false_before = is_anomaly[sick_df.day.values < BASELINE_CUTOFF].sum()
    onset_mask = sick_df.day.values >= SICK_ONSET_DAY
    n_alerts_after = is_anomaly[onset_mask].sum()
    n_days_after = onset_mask.sum()
    first_alert_days = sick_df.day.values[onset_mask][is_anomaly[onset_mask]]
    detect_delay = int(first_alert_days.min() - SICK_ONSET_DAY) if len(first_alert_days) else None

    # ── 3. 冷启动 ──
    target_breed_idx = breed_ids.index("中华田园犬")
    true_new_dog_rate = np.exp(mu_pop_true + rng.normal(0, sigma_breed_true) - 0.15)
    new_dog_daily_counts = rng.poisson(true_new_dog_rate, size=30)
    checkpoints = [0, 1, 3, 7, 14, 21, 30]
    means, lo, hi = sequential_cold_start(idata, target_breed_idx, new_dog_daily_counts, checkpoints)
    cold_start_df = pd.DataFrame({"day": checkpoints, "estimate": means, "lo95": lo, "hi95": hi})
    cold_start_df.to_csv(os.path.join(DATA_DIR, "cold_start_new_dog.csv"), index=False)
    cold_start_img = plot_cold_start(checkpoints, means, lo, hi, true_new_dog_rate)

    doc_lines = [
        "# 贝叶斯分层模型（BHM）验证画廊",
        "",
        "> 对应 `bhm_scratch_count.py`（品种→狗→每日三层贝叶斯计数模型）和 "
        "`validate_bhm_scratch_count.py`（合成数据验证脚本），用的是影棚实际4个品种"
        "（金毛/中华田园犬/比熊/马尔济斯），每个品种额外配9只合成'陪跑'狗凑够能验证"
        "'借力'机制的样本规模（现实里每个品种只有1只真狗，组内方差没法估）。"
        "这份文档跟`scenario_gallery.md`（SBS规则引擎那份画廊）是同一个思路，但验证的"
        "是贝叶斯分层模型这条路，不是SBS打分公式那条路，两个是互补关系，不是替代关系"
        "（具体什么时候该上哪个，见`model_roadmap.md`）。",
        "",
        f"本次拟合收敛诊断：最大r_hat={max_rhat:.3f}（越接近1.0越好，一般<1.01算收敛），"
        f"发散样本数={n_div}（越接近0越好）。",
        "",
        "## 1. 部分池化（partial pooling）——数据量少的狗，估计值会更靠近品种基线",
        "",
        "同一个品种（金毛）里挑4只基线期观测天数一样但原始均值不同的狗，对比"
        "\"只看这只狗自己数据算出来的均值\"（灰柱）和\"模型综合了同品种其他狗信息后的"
        "估计值\"（蓝柱）。原始均值离品种平均水平越远的狗，蓝柱被往品种基线拉回得"
        "越明显——这就是分层模型的核心机制：既不完全相信单只狗的少量数据（容易被"
        "噪声带偏），也不完全无视个体差异（不会把所有狗都拉成同一个数）。",
        "",
        f"![部分池化]({pooling_img})",
        "",
        "| 狗 | 基线期观测天数 | 原始均值 | BHM后验估计 |",
        "|---|---|---|---|",
        *[f"| {r.dog_id} | {r.n_obs} | {r.raw_mean:.2f} | {r.bhm_estimate:.2f} |"
          for r in pooling_df.itertuples()],
        "",
        "数据：[部分池化对比表](../data/bhm_validation/partial_pooling_comparison.csv) | "
        "[基线期训练数据](../data/bhm_validation/train_data_baseline_period.csv)",
        "",
        "---",
        "",
        "## 2. 异常检测——一只狗从某天起患病，后验预测检验能不能及时发现",
        "",
        f"合成的\"患病狗\"：**{sick_dog_id}**，第{SICK_ONSET_DAY}天起注入抓挠率显著上升"
        "（约3倍），模型拟合时完全不知道这个信息，只用前19天数据估计这只狗的个体基线，"
        "之后每天拿实际观测值去跟\"这只狗正常情况下应该是多少\"的后验预测分布比对，"
        "算出百分位，超出[2.5, 97.5]区间就判定异常（双边检验）。",
        "",
        f"![异常检测]({anomaly_img})",
        "",
        f"- 基线期({n_baseline_days}天)误报: {n_false_before}天，"
        f"误报率{n_false_before/n_baseline_days*100:.0f}%（理论期望约5%，双边2.5%+2.5%）",
        f"- 发病期({n_days_after}天)命中: {n_alerts_after}天，"
        f"命中率{n_alerts_after/n_days_after*100:.0f}%",
        f"- 发病后首次成功报警延迟: "
        f"{f'{detect_delay}天' if detect_delay is not None else '未检出'}",
        "",
        f"数据：[逐日百分位+异常判定](../data/bhm_validation/anomaly_detection_{sick_dog_id}.csv)",
        "",
        "---",
        "",
        "## 3. 冷启动——一只全新狗，不确定性怎么随佩戴天数收窄",
        "",
        "模拟一只全新绑定设备的中华田园犬，用Gamma-Poisson共轭近似（不用每天重跑"
        "完整MCMC）看不确定性区间怎么从\"只有品种先验、完全没见过这只狗的数据\"逐步"
        "收窄到\"有一定个体数据支撑\"。第0天区间最宽（纯品种先验），随着天数增加"
        "区间逐渐收窄并向这只狗的真实基线靠拢。",
        "",
        f"![冷启动]({cold_start_img})",
        "",
        "| 佩戴天数 | 估计日均抓挠 | 95%区间下限 | 95%区间上限 | 区间宽度 |",
        "|---|---|---|---|---|",
        *[f"| {r.day} | {r.estimate:.2f} | {r.lo95:.2f} | {r.hi95:.2f} | {r.hi95-r.lo95:.2f} |"
          for r in cold_start_df.itertuples()],
        "",
        f"该狗真实基线（仅供验证，模型未知）：{true_new_dog_rate:.2f}",
        "",
        "数据：[冷启动逐checkpoint明细](../data/bhm_validation/cold_start_new_dog.csv)",
        "",
        "---",
        "",
        "完整合成数据：[全部品种全部狗35天数据](../data/bhm_validation/simulated_full_data.csv) | "
        "[狗-品种-是否患病对照表](../data/bhm_validation/dog_meta.csv)",
        "",
    ]

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines))
    print(f"已生成 {DOC_PATH}，图片在 {ASSET_DIR}，CSV在 {DATA_DIR}")


if __name__ == "__main__":
    main()
