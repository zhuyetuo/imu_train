"""
分层贝叶斯模型：狗的皮肤/抓挠严重程度评估，theory见对话记录/docs/skin_health.md。

两层结构：
  第1层（每天，第i只狗）：
    当天严重程度（有序类别：0=正常/1=关注/2=异常）~ OrderedLogistic(alpha_i + beta·特征, cutpoints)
  第2层（狗与狗之间）：
    alpha_i ~ Normal(mu_alpha, sigma_alpha)   ← 个体基线跟群体之间的桥梁

alpha_i 就是"个体基线"，但不是简单点估计——数据少的狗，它的 alpha_i 后验会被
群体先验(mu_alpha, sigma_alpha)往群体均值方向拉（收缩），数据多的狗则主要由
它自己的数据决定，这是自动发生的，不需要写"21天"这种硬边界。

beta 是特征对严重程度的影响力，对应PM文档手写的那些权重——这里是从数据里
估计出来的（哪怕数据少时基本等于先验，不是凭空编的）。

用法（先跑合成数据验证机制本身）：
    python skin_health/code/bayesian_skin_model.py
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def fit_hierarchical_model(df, feature_cols, label_col="vet_label", dog_col="pet_id",
                            n_classes=3, draws=1000, tune=1000, chains=4, seed=42):
    """
    df: 每行一个 (狗, 天)，包含特征列 + 兽医打的有序标签（0/1/2）。
    返回 (model, trace, dog_ids)。
    """
    dog_ids = sorted(df[dog_col].unique())
    dog_idx_map = {d: i for i, d in enumerate(dog_ids)}
    dog_idx = df[dog_col].map(dog_idx_map).values
    n_dogs = len(dog_ids)

    X = df[feature_cols].values.astype(float)
    # 标准化特征，方便先验用统一尺度、beta 系数之间可比
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    y = df[label_col].values.astype(int)

    with pm.Model() as model:
        # 群体的"整体位置"完全由 cutpoints 承担，alpha 的群体均值固定为0，只表示
        # "每只狗相对群体典型水平的偏移"——否则 mu_alpha 和 cutpoints 可以互相
        # 抵消（同时平移、似然不变），模型参数不可识别，估出来的数字会systematically偏。
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.5)
        alpha = pm.Normal("alpha", mu=0.0, sigma=sigma_alpha, shape=n_dogs)

        beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=len(feature_cols))

        eta = alpha[dog_idx] + pm.math.dot(X, beta)

        # n_classes 个类别需要 n_classes-1 个切点，pm.OrderedLogistic 要求严格递增
        cutpoints = pm.Normal(
            "cutpoints", mu=np.linspace(-1, 1, n_classes - 1), sigma=2.0,
            shape=n_classes - 1,
            transform=pm.distributions.transforms.ordered,
            initval=np.linspace(-1, 1, n_classes - 1),
        )

        pm.OrderedLogistic("y_obs", eta=eta, cutpoints=cutpoints, observed=y)

        trace = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed,
                           progressbar=False, target_accept=0.9)

    return model, trace, dog_ids


def predict_new_dog(trace):
    """全新狗（没有任何自己的数据）的基线该怎么估：直接用群体先验的后验分布，
    不掺入任何特定狗的信息——对应现在系统里"引导期用Whistle兜底"这一步，
    只是这里的"群体先验"是从我们自己的狗群体数据学出来的，不是借用外部文献数字。
    群体均值固定为0（见模型定义里的说明），所以新狗的alpha就是 Normal(0, sigma_alpha)。
    """
    sigma_alpha_samples = trace.posterior["sigma_alpha"].values.flatten()
    rng = np.random.default_rng(0)
    new_dog_alpha_samples = rng.normal(0.0, sigma_alpha_samples)
    return new_dog_alpha_samples


def summarize(trace, dog_ids):
    summary = az.summary(trace, var_names=["sigma_alpha", "alpha", "beta", "cutpoints"])
    summary = summary.reset_index().rename(columns={"index": "param"})
    alpha_rows = summary[summary.param.str.startswith("alpha[")].copy()
    alpha_rows["dog_id"] = [dog_ids[int(p.split("[")[1][:-1])] for p in alpha_rows.param]
    return summary, alpha_rows
