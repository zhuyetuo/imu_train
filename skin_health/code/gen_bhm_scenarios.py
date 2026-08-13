"""
给 plot_bhm_gallery.py 用的合成场景定义——在 validate_bhm_scratch_count.py
那个"背景种群"（每个品种1只real_占位 + 9只syn_陪跑）基础上，额外加几只打了
不同异常模式标签的"场景狗"，验证同一个联合拟合出来的模型在不同情况下分别
表现如何：不是每个场景单独拟合一次模型（成本高很多——一次MCMC大概40秒，
6个场景分别拟合要4分钟+），而是把所有场景狗放进同一份训练数据联合拟合，
拟合完之后针对每只场景狗单独跑 posterior_predictive_check() 看结果，这样
更贴近真实业务场景本身就是"一个模型同时服务所有狗"。

场景狗的onset_day统一设在BASELINE_CUTOFF(=19)之后，保证每只场景狗自己的
基线期(day<19)数据都是干净的、没被异常污染——这样"用基线期数据估计个体
基线、用之后的数据做异常检测"这个前提对所有场景狗都成立，能公平对比。
"""
import numpy as np
import pandas as pd

BREEDS = ["金毛", "中华田园犬", "比熊", "马尔济斯"]
SYNTHETIC_DOGS_PER_BREED = 9  # 每个品种背景种群陪跑狗数量，同 validate_bhm_scratch_count.py
N_DAYS = 35
BASELINE_CUTOFF = 19

mu_pop_true = 1.6
sigma_breed_true = 0.35
sigma_dog_true = 0.30
nb_alpha_true = 6.0  # 默认离散度，越大越接近Poisson

# 场景狗定义：kind决定异常怎么随天数变化。
#   step:  onset_day起阶跃式恒定升高，升高量=effect（log-rate增量）
#   ramp:  onset_day起线性爬升到peak_day，之后维持在effect水平
#   spike_recover: onset_day到recover_day之间维持effect水平，之后恢复正常
#   none:  全程不注入异常（stable_normal / high_variance_noisy用这个）
SCENARIOS = [
    {"key": "sudden_onset", "breed": "中华田园犬", "kind": "step",
     "onset_day": 20, "effect": 1.1,
     "desc": "第20天起阶跃式上升（约3倍），急性发作模式，验证检测延迟"},
    {"key": "gradual_onset", "breed": "金毛", "kind": "ramp",
     "onset_day": 20, "peak_day": 32, "effect": 1.1,
     "desc": "第20天起线性爬升到第32天到达约3倍水平，渐进恶化模式，验证模型"
             "会不会等爬到很高才反应过来，还是能在爬升途中就报警"},
    {"key": "recovering", "breed": "比熊", "kind": "spike_recover",
     "onset_day": 20, "recover_day": 28, "effect": 1.1,
     "desc": "第20-27天升高（约3倍）后第28天起恢复正常，验证异常信号会不会"
             "跟着恢复消退，而不是一直挂着报警"},
    {"key": "mild_onset", "breed": "马尔济斯", "kind": "step",
     "onset_day": 20, "effect": 0.4,
     "desc": "第20天起小幅上升（约1.5倍，远轻于sudden_onset的3倍），对比"
             "检测延迟/命中率随异常幅度变化——异常越轻微，本来就应该越难/越慢发现"},
    {"key": "stable_normal", "breed": "金毛", "kind": "none",
     "desc": "全程正常，不注入任何异常，验证35天里误报率是否符合双边检验"
             "约5%的理论期望（真阴性对照）"},
    {"key": "high_variance_noisy", "breed": "中华田园犬", "kind": "none",
     "alpha_override": 1.2,
     "desc": "均值水平跟同品种正常狗一样，但这只狗自己数据的离散度大得多"
             "（模型的alpha是全局共享参数，没法单独识别这只狗天生噪声更大），"
             "验证天然噪声大的狗会不会被误判成异常——这是在暴露模型的一个"
             "已知局限，不是要证明模型没问题"},
]


def _daily_counts(rng, base_log_rate, day, alpha, scenario=None):
    extra = 0.0
    if scenario is not None:
        kind = scenario["kind"]
        onset = scenario.get("onset_day")
        if kind == "step" and day >= onset:
            extra = scenario["effect"]
        elif kind == "ramp":
            peak = scenario["peak_day"]
            if day >= onset:
                extra = scenario["effect"] * min(1.0, (day - onset) / (peak - onset))
        elif kind == "spike_recover":
            recover = scenario["recover_day"]
            if onset <= day < recover:
                extra = scenario["effect"]
    mu = np.exp(base_log_rate + extra)
    p = alpha / (alpha + mu)
    return int(rng.negative_binomial(alpha, p))


def simulate_data(seed=42):
    """返回 (df, dog_meta)。df: [dog_id, breed, day, scratch_count]。
    dog_meta: [dog_id, breed, scenario_key(None=背景种群普通狗), scenario_desc]。"""
    rng = np.random.default_rng(seed)
    breed_effects = {b: rng.normal(0, sigma_breed_true) for b in BREEDS}
    rows, dog_meta = [], []

    # 背景种群：每个品种1只real_占位 + 9只syn_陪跑，跟validate_bhm_scratch_count.py
    # 里的标准种群一致（不含任何注入异常），用来撑住品种层/群体层的估计
    for breed in BREEDS:
        for local_idx in range(SYNTHETIC_DOGS_PER_BREED + 1):
            dog_id = f"{'real' if local_idx == 0 else 'syn'}_{breed}_{local_idx}"
            dog_effect = rng.normal(0, sigma_dog_true)
            dog_meta.append({"dog_id": dog_id, "breed": breed,
                             "scenario_key": None, "scenario_desc": ""})
            for day in range(N_DAYS):
                log_rate = mu_pop_true + breed_effects[breed] + dog_effect
                rows.append({"dog_id": dog_id, "breed": breed, "day": day,
                            "scratch_count": _daily_counts(rng, log_rate, day, nb_alpha_true)})

    # 场景狗：每个额外加一只，各自品种复用上面抽好的breed_effect保持一致
    for s in SCENARIOS:
        dog_id = f"scenario_{s['key']}"
        breed = s["breed"]
        dog_effect = rng.normal(0, sigma_dog_true)
        alpha = s.get("alpha_override", nb_alpha_true)
        dog_meta.append({"dog_id": dog_id, "breed": breed,
                         "scenario_key": s["key"], "scenario_desc": s["desc"]})
        for day in range(N_DAYS):
            log_rate = mu_pop_true + breed_effects[breed] + dog_effect
            rows.append({"dog_id": dog_id, "breed": breed, "day": day,
                        "scratch_count": _daily_counts(rng, log_rate, day, alpha, scenario=s)})

    return pd.DataFrame(rows), pd.DataFrame(dog_meta)
