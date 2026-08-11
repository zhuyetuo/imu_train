"""
用合成数据验证分层贝叶斯模型（bayesian_skin_model.py）的机制本身对不对：
  1. 数据多的狗，个体基线(alpha)的后验应该更接近它自己的真实水平、不确定性更小
  2. 数据少的狗，alpha 应该被往群体均值方向拉（收缩），不确定性更大
  3. 全新狗（零数据）的预测应该完全依赖群体先验

不是验证"真实狗的皮肤状况预测准不准"（那需要真实兽医标签，见 docs/skin_health.md），
是验证"这套贝叶斯机制本身有没有表现出该有的统计行为"。

用法：
    python src/eval/validate_bayesian_skin_model.py
"""
import numpy as np
import pandas as pd

from bayesian_skin_model import fit_hierarchical_model, predict_new_dog, summarize

rng = np.random.default_rng(42)

TRUE_POP_MEAN = 0.0
TRUE_POP_SD = 1.2
TRUE_BETA = 0.8  # 特征（抓挠强度）对严重程度的真实影响力


def simulate_dog(true_alpha, n_days, label_noise=0.1):
    """给一只狗模拟 n_days 天的数据：每天一个"抓挠强度"特征 + 一个兽医打的
    有序标签（可能有噪声，label_noise概率随机打错一档，模拟兽医主观误差）。"""
    x = rng.normal(0, 1, n_days)  # 标准化后的当天抓挠强度特征
    eta = true_alpha + TRUE_BETA * x
    # 用固定切点把连续的eta离散成3类，模拟"真实"标签生成过程
    cutpoints = [-0.5, 0.5]
    true_label = np.digitize(eta + rng.normal(0, 0.3, n_days), cutpoints)
    label = true_label.copy()
    flip_mask = rng.random(n_days) < label_noise
    label[flip_mask] = rng.integers(0, 3, flip_mask.sum())
    return x, label


def main():
    # 4只"数据深度"不同的合成狗，模拟真实场景里陆续接入、数据量参差不齐
    dogs = {
        "dog_rich_history": {"true_alpha": 1.0, "n_days": 60},   # 数据多，真实水平偏高
        "dog_medium_history": {"true_alpha": -0.5, "n_days": 20},  # 数据中等，真实水平偏低
        "dog_sparse_history": {"true_alpha": 0.3, "n_days": 5},   # 数据很少，真实水平中等偏高
        "dog_baseline_normal": {"true_alpha": 0.0, "n_days": 30},  # 数据中等，真实水平正好在群体均值
    }

    rows = []
    for dog_id, cfg in dogs.items():
        x, label = simulate_dog(cfg["true_alpha"], cfg["n_days"])
        for xi, li in zip(x, label):
            rows.append({"pet_id": dog_id, "scratch_intensity": xi, "vet_label": li})
    df = pd.DataFrame(rows)

    print(f"合成数据: {len(df)} 行, {df.pet_id.nunique()} 只狗")
    print(df.groupby("pet_id").size().rename("天数"))

    print("\n拟合分层贝叶斯模型...")
    model, trace, dog_ids = fit_hierarchical_model(
        df, feature_cols=["scratch_intensity"], label_col="vet_label", dog_col="pet_id")

    summary, alpha_rows = summarize(trace, dog_ids)

    max_rhat = summary["r_hat"].max()
    n_divergences = int(trace.sample_stats["diverging"].sum())
    print(f"\n收敛诊断: 最大r_hat={max_rhat:.3f}（应接近1.0，>1.01说明没收敛好），"
          f"发散样本数={n_divergences}（应该是0或很少）")

    print("\n" + "=" * 90)
    print("各狗的个体基线(alpha)后验估计 vs 真实值")
    print("=" * 90)
    # 模型里 alpha 的群体均值被固定为0（"整体位置"由cutpoints承担，见
    # bayesian_skin_model.py 里的说明，避免mu_alpha和cutpoints互相抵消导致不可识别），
    # 所以 alpha 代表的是"相对这4只合成狗真实均值的偏移"，不是绝对值，
    # 拿真实alpha减去这4只狗的真实均值才是苹果对苹果的比较对象。
    true_alphas = {d: cfg["true_alpha"] for d, cfg in dogs.items()}
    true_pop_mean = np.mean(list(true_alphas.values()))
    for _, row in alpha_rows.iterrows():
        true_alpha = true_alphas[row.dog_id]
        true_relative = true_alpha - true_pop_mean
        n_days = dogs[row.dog_id]["n_days"]
        print(f"  {row.dog_id:22s} (真实历史{n_days:3d}天): "
              f"后验均值={row['mean']:+.2f}  后验标准差(不确定性)={row['sd']:.2f}  "
              f"真实相对值={true_relative:+.2f}（真实alpha={true_alpha:+.2f} - 4只狗真实均值{true_pop_mean:+.2f}）")

    print("\n预期看到的现象:")
    print("  1. dog_rich_history (60天) 的后验标准差应该明显小于 dog_sparse_history (5天)")
    print("  2. dog_sparse_history 的后验均值应该比它的真实相对值更靠近0"
          "（被群体先验往0拉了一把，这就是收缩/shrinkage）")

    sigma_alpha_mean = summary.loc[summary.param == "sigma_alpha", "mean"].values[0]
    print(f"\n群体层面估计: sigma_alpha={sigma_alpha_mean:.2f} "
          f"(真实4只狗alpha的标准差={np.std(list(true_alphas.values())):.2f}，"
          "样本只有4只狗，这个真实标准差本身也不准，仅供参考)")

    print("\n" + "=" * 90)
    print("全新狗（零数据）预测——完全依赖群体先验")
    print("=" * 90)
    new_dog_samples = predict_new_dog(trace)
    print(f"  全新狗的alpha后验预测: 均值={new_dog_samples.mean():.2f}  "
          f"标准差={new_dog_samples.std():.2f}")
    print(f"  均值应该接近0（群体平均水平），标准差应该明显大于任何一只有数据的狗"
          f"（因为完全没有个体信息，全靠群体先验猜）")

    beta_mean = summary.loc[summary.param == "beta[0]", "mean"].values[0]
    print(f"\n特征影响力(beta)估计: {beta_mean:.2f} (真实值={TRUE_BETA}，"
          "cutpoints跟beta也有一定程度的尺度关联，不要求完全精确匹配，看数量级和方向对不对)")


if __name__ == "__main__":
    main()
