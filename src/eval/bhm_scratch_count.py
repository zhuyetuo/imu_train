"""
分层贝叶斯模型（品种→狗→每日观测三层结构），直接对"每日抓挠次数"这个原始
计数建模，不需要兽医标签。跟 bayesian_skin_model.py（对兽医打的有序标签建模）
是两条互补的路：

- 这个模型（NB计数模型）：只要有IMU算出来的抓挠次数，不需要任何兽医参与，
  就能算出"个体基线"+做异常检测（今天是不是明显偏离这只狗的正常水平）。
  兽医标签留到后面，只用来验证"统计异常"是不是"真的有皮肤问题"，两件事解耦。
- bayesian_skin_model.py（有序标签模型）：需要兽医打分才能训练，产出"正常/
  关注/异常"这种分类结果，直接对齐产品要展示的等级。

两个模型的定位不同：这个模型适合先跑起来做异常检测的地基（数据要求低），
后者适合有了标签之后去校准/验证"异常"这个统计概念在多大程度上等同于真实
皮肤问题的临床意义。

模型结构：
    scratch_count[i] ~ NegativeBinomial(mu[i], alpha)   # 处理"有的狗某几天特别多"的过度离散
    log(mu[i]) = mu_pop + breed_effect[breed[i]] + dog_effect[dog[i]]
    breed_effect ~ Normal(0, sigma_breed)                # 非中心化参数化，见 fit_model()
    dog_effect   ~ Normal(breed_effect[该狗品种], sigma_dog)

用非中心化参数化（noncentered parameterization）：不直接从 Normal(mu, sigma)
里采样效应值，而是先采样一个标准正态"原始值"，再乘以sigma——这是PyMC分层模型
的标准做法，避免sigma和效应值在后验几何上高度相关导致的"漏斗"病态、采样效率差、
容易发散的问题。
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def fit_model(df, breed_col="breed", dog_col="dog_id", count_col="scratch_count",
              draws=1000, tune=1000, chains=4, seed=42):
    """
    df: 每行一个 (狗, 天)，包含品种、狗ID、当天抓挠次数。
    返回 (idata, breed_ids, dog_ids, dog_to_breed_idx)。
    """
    breed_ids = sorted(df[breed_col].unique())
    breed_idx_map = {b: i for i, b in enumerate(breed_ids)}
    dog_ids = sorted(df[dog_col].unique())
    dog_idx_map = {d: i for i, d in enumerate(dog_ids)}

    dog_to_breed = df.drop_duplicates(dog_col).set_index(dog_col)[breed_col].map(breed_idx_map)
    dog_to_breed = dog_to_breed.reindex(dog_ids).values

    breed_idx = df[breed_col].map(breed_idx_map).values
    dog_idx = df[dog_col].map(dog_idx_map).values
    y = df[count_col].values

    n_breeds, n_dogs = len(breed_ids), len(dog_ids)
    coords = {"breed": range(n_breeds), "dog": range(n_dogs), "obs": range(len(df))}

    with pm.Model(coords=coords) as model:
        mu_pop = pm.Normal("mu_pop", mu=1.5, sigma=1.0)

        sigma_breed = pm.HalfNormal("sigma_breed", sigma=0.5)
        breed_effect_raw = pm.Normal("breed_effect_raw", mu=0, sigma=1, dims="breed")
        breed_effect = pm.Deterministic("breed_effect", breed_effect_raw * sigma_breed, dims="breed")

        sigma_dog = pm.HalfNormal("sigma_dog", sigma=0.5)
        dog_effect_raw = pm.Normal("dog_effect_raw", mu=0, sigma=1, dims="dog")
        dog_effect = pm.Deterministic(
            "dog_effect", dog_effect_raw * sigma_dog + breed_effect[dog_to_breed], dims="dog")

        alpha = pm.HalfNormal("alpha", sigma=10)  # NB离散度参数，越小越过度离散
        log_mu = mu_pop + dog_effect[dog_idx]
        mu = pm.Deterministic("mu_obs", pm.math.exp(log_mu), dims="obs")

        pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y, dims="obs")

        # 注意：这里不调用 pm.sample_posterior_predictive 存整份后验预测分布——
        # posterior_predictive_check() 需要的是"给定某只狗的个体后验，模拟它自己的
        # 预测分布"，用 mu_pop/dog_effect/alpha 的后验样本手动模拟即可，不需要
        # PyMC自带的posterior_predictive组（存进idata还会因为arviz/xarray版本
        # 兼容性问题报错，没必要冒这个险）。
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                           progressbar=False, target_accept=0.95)

    return idata, breed_ids, dog_ids, dog_to_breed


def check_convergence(idata):
    summary = az.summary(idata, var_names=["mu_pop", "sigma_breed", "sigma_dog", "alpha"])
    max_rhat = summary["r_hat"].max()
    n_div = int(idata.sample_stats["diverging"].sum())
    return max_rhat, n_div


def posterior_predictive_check(idata, dog_id, observed_counts, n_sim=4000, seed=0):
    """
    异常检测的核心：用这只狗【基线期】数据拟合出的个体后验，模拟"正常情况下今天
    应该是多少次"的预测分布，看新的实际观测值落在这个分布的第几百分位。

    百分位越极端（比如>97.5%），说明今天的抓挠次数越不像是这只狗正常水平能
    产生的结果——这是有统计意义的检验，不是"倍数超过3就报警"这种手调阈值。

    返回每个观测值对应的百分位（0-100）和是否判定异常（>97.5或<2.5，双边检验）。
    """
    post = idata.posterior
    dog_log_rate_samples = (post["mu_pop"] + post["dog_effect"].sel(dog=dog_id)).stack(
        sample=("chain", "draw")).values
    alpha_samples = post["alpha"].stack(sample=("chain", "draw")).values

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dog_log_rate_samples), n_sim, replace=True)
    mu_sim = np.exp(dog_log_rate_samples[idx])
    alpha_sim = alpha_samples[idx]
    p_sim = alpha_sim / (alpha_sim + mu_sim)
    ppc_counts = rng.negative_binomial(alpha_sim, p_sim)

    percentiles, is_anomaly = [], []
    for cnt in observed_counts:
        pct = (ppc_counts <= cnt).mean() * 100
        percentiles.append(pct)
        is_anomaly.append(pct > 97.5 or pct < 2.5)
    return np.array(percentiles), np.array(is_anomaly)


def sequential_cold_start(idata, breed_id, daily_counts, days_checkpoints, seed=123):
    """
    新狗从0天到N天，个体基线的不确定性怎么从"品种先验"收窄到"个体后验"——
    用 Gamma-Poisson 共轭近似做在线更新（不需要每天重跑完整MCMC，闭式解，计算量很小），
    适合回答"产品的标定期该设几天"这类问题，也是"每天要不要全量重新拟合"这个
    工程问题的一个轻量解法：只有品种/群体层参数需要定期完整重新拟合，单只狗的
    日常更新可以用这种共轭近似代替。

    daily_counts: 这只狗按天顺序排列的观测计数（比如 [3,5,4,...]）。
    days_checkpoints: 想看哪几个时间点的后验，比如 [0,1,3,7,14,21,30]。
    返回 (means, lo95, hi95) 三个跟 days_checkpoints 等长的列表（次/天尺度）。
    """
    post = idata.posterior
    breed_log_rate_post = (post["mu_pop"] + post["breed_effect"].sel(breed=breed_id)).stack(
        sample=("chain", "draw")).values
    sigma_dog_post = post["sigma_dog"].stack(sample=("chain", "draw")).values

    rng = np.random.default_rng(seed)
    individual_dog_draw = rng.normal(0, sigma_dog_post)
    new_dog_prior_log_rate_samples = breed_log_rate_post + individual_dog_draw
    prior_rate_mean = np.exp(new_dog_prior_log_rate_samples).mean()
    prior_rate_var = np.exp(new_dog_prior_log_rate_samples).var()

    # Gamma先验矩匹配（Poisson似然的共轭先验是Gamma，用矩匹配把"品种+个体差异"的
    # 不确定性近似成一个Gamma分布，作为新狗的先验起点）
    theta0 = prior_rate_var / prior_rate_mean
    k0 = prior_rate_mean / theta0

    means, lo, hi = [], [], []
    for n in days_checkpoints:
        data_sum = sum(daily_counts[:n])
        k_post = k0 + data_sum
        theta_post = theta0 / (1 + n * theta0)
        post_mean = k_post * theta_post
        post_sd = np.sqrt(k_post) * theta_post
        means.append(post_mean)
        lo.append(max(0, post_mean - 1.96 * post_sd))
        hi.append(post_mean + 1.96 * post_sd)
    return means, lo, hi
