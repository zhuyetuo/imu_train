"""
问答特征的合成生成 + 特征列定义，对应 questionnaire_feature_spec.md。

跟IMU特征不同，问答特征没有"事件"可以模拟，是直接在"这一天的真实皮肤状态"
这个潜变量基础上合成的类别/有序取值。潜变量设计：

  true_skin_latent（0~3，不给模型看，只用来生成问答取值和S真值标签）：
    0=完全正常 1=轻微 2=中等 3=严重
  跟当天的C（IMU行为严重度）挂钩的方式取决于scenario的questionnaire_behavior：
    answered_consistent          latent 跟 C_ordinal 同步（C0→0，C1→1，C2→2/3看红旗）
    answered_conflicting_worse   latent 比 C_ordinal 推断的更严重（皮肤已经有问题但
                                  抓挠次数还不算特别多，比如舔舐主导而非抓挠）
    answered_conflicting_better  latent 比 C_ordinal 推断的更轻（次数多但皮肤基本正常，
                                  可能是行为习惯而非病理性）

主人自报的抓挠次数/时长桶（owner_reported_*）反映的是"抓挠行为本身"，跟当天
真实event_count/duration相关（加一点主观感知噪声），不受true_skin_latent影响——
这是跟IMU测量同一件事的两个信息源，不是跟皮肤状态相关的问题。
"""
import numpy as np
import pandas as pd

# 每个特征的有序取值数（0=最轻/正常，数字越大越严重），对应questionnaire_feature_spec.md
SKIN_REDNESS_LEVELS = 3          # 正常/轻微泛红/鲜红无破损
SKIN_LESION_LEVELS = 4           # 完整/干燥少量皮屑/斑块成片/糜烂渗液
ODOR_LEVELS = 4                  # 无/凑近/30-50cm/进屋即闻
HAIR_LOSS_SPOT_LEVELS = 4        # 无/1-2处/>=3处/连续成片
HAIR_LOSS_DIAMETER_LEVELS = 4    # 无/<1-2cm/2-3cm/>3cm
COAT_QUALITY_LEVELS = 4          # 光亮/油腻打结/小范围断裂/大部分干枯
SCRATCH_COUNT_BUCKETS = 5        # <5/5-15/15-30/30-50/>50，owner_reported
SCRATCH_DURATION_BUCKETS = 5     # <0/0-1h/1-3h/3-6h/>6h，owner_reported
PIGMENT_ABNORMAL_LEVELS = 2      # 二值：有没有黑色油脂/色素沉着（不是有序严重度）

QUESTIONNAIRE_FEATURE_COLUMNS = [
    "owner_reported_scratch_count_bucket",
    "owner_reported_scratch_duration_bucket",
    "skin_redness_level",
    "skin_pigment_abnormal",
    "skin_lesion_severity",
    "odor_level",
    "hair_loss_spot_count_level",
    "hair_loss_max_diameter_level",
    "coat_quality_level",
]


def _latent_for(c_ordinal, questionnaire_behavior, has_red_flag, rng):
    """c_ordinal: 0/1/2 对应C0/C1/C2。返回0~3的true_skin_latent。

    注意：不再叠加has_red_flag的额外加成——红旗信号已经影响了SBS算出来的
    c_ordinal本身（红旗会把total_score直接推高甚至强制变成C2），在这里
    再加一次会重复计入同一个信号，之前就是这个重复计入 + Python round()
    的banker's rounding（0.5→0但1.5→2，不对称）两个问题叠加，导致S标签
    几乎全部倒向S2，已经在86场景实测里发现并修正。has_red_flag参数保留
    在函数签名里但不再使用，调用方不用改。
    """
    base = c_ordinal
    if questionnaire_behavior == "answered_conflicting_worse":
        return min(3, base + 1 + int(rng.integers(0, 2)))
    if questionnaire_behavior == "answered_conflicting_better":
        return max(0, base - 1 - int(rng.integers(0, 2)))
    # answered_consistent：跟c_ordinal同步，加±1的小噪声制造合理变化，不是100%确定性映射
    return int(np.clip(base + rng.integers(-1, 2), 0, 3))


def _bucket_from_count(rng, event_count):
    """主人自报抓挠次数桶，围绕真实event_count加感知噪声后落桶。"""
    perceived = max(0, event_count + rng.normal(0, max(1.0, event_count * 0.25)))
    edges = [5, 15, 30, 50]
    for i, e in enumerate(edges):
        if perceived < e:
            return i
    return 4


def _bucket_from_duration_min(rng, duration_min):
    perceived_hours = max(0.0, duration_min / 60.0 + rng.normal(0, 0.3))
    edges = [0.0001, 1, 3, 6]
    for i, e in enumerate(edges):
        if perceived_hours < e:
            return i
    return 4


def _jitter(rng, value, max_level, p_jitter=0.3):
    """每个问答特征独立在latent基础上抖动±1（30%概率），不让任何一个观测特征
    精确等于用来算真值标签的latent——之前skin_redness_level=min(2,latent)是
    latent的精确拷贝，跟true_s_tier()里的skin_capped是同一个数，导致permutation
    importance测出这个特征"重要性"高达0.63、宏F1到0.957，这不是真实的特征
    价值，是标签生成过程本身的循环论证（这个特征本质上就是标签的另一种写法）。
    加独立抖动后，每个特征都是latent的一个带噪声的观测，不再有任何单一特征
    能精确还原标签，重要性排序才有参考价值。"""
    if rng.random() < p_jitter:
        value += int(rng.choice([-1, 1]))
    return int(np.clip(value, 0, max_level))


def simulate_questionnaire_row(rng, c_ordinal, questionnaire_behavior, has_red_flag,
                               event_count, duration_min):
    latent = _latent_for(c_ordinal, questionnaire_behavior, has_red_flag, rng)
    lvl3 = min(2, latent)  # 3档取值特征封顶到2
    return {
        "owner_reported_scratch_count_bucket": _bucket_from_count(rng, event_count),
        "owner_reported_scratch_duration_bucket": _bucket_from_duration_min(rng, duration_min),
        "skin_redness_level": _jitter(rng, lvl3, 2),
        "skin_pigment_abnormal": int(rng.random() < (0.15 + 0.15 * latent)),
        "skin_lesion_severity": _jitter(rng, min(HAIR_LOSS_SPOT_LEVELS - 1, latent), HAIR_LOSS_SPOT_LEVELS - 1),
        "odor_level": _jitter(rng, min(ODOR_LEVELS - 1, latent), ODOR_LEVELS - 1),
        "hair_loss_spot_count_level": _jitter(
            rng, min(HAIR_LOSS_SPOT_LEVELS - 1, max(0, latent - 1)), HAIR_LOSS_SPOT_LEVELS - 1),
        "hair_loss_max_diameter_level": _jitter(
            rng, min(HAIR_LOSS_DIAMETER_LEVELS - 1, max(0, latent - 1)), HAIR_LOSS_DIAMETER_LEVELS - 1),
        "coat_quality_level": _jitter(rng, min(COAT_QUALITY_LEVELS - 1, latent), COAT_QUALITY_LEVELS - 1),
        "_true_skin_latent": latent,  # 仅用于算S真值标签，不进模型
    }


def true_s_tier(c_ordinal, true_skin_latent):
    """合成S真值标签的组合规则（仅供合成数据训练用，不是PM公式的复刻）：
    皮肤外观权重0.65 > 行为权重0.35（问答本来就是为了在行为信号之外补充
    "皮肤到底长什么样"这个更直接的信息，所以让它在合成真值里占大头），
    用向上取整式四舍五入（不用Python内置round()——它是banker's rounding，
    0.5会舍去、1.5会进位，不对称，之前就是这个不对称导致S标签几乎全部
    倒向S2，已实测发现并改成这里的标准四舍五入）。

    c_ordinal只会是1或2（模型B只在C1/C2触发问答后才有样本，C0不触发）。
    这个权重下，C1+皮肤完全正常（skin=0）能压到S0，C2+皮肤明显异常
    （skin=2）稳定给S2，中间地带按皮肤严重程度自然分布到三档，不会像
    之前那样结构性地基本只剩S2一个结果。"""
    skin_capped = min(2, true_skin_latent)
    raw = c_ordinal * 0.35 + skin_capped * 0.65
    s_ordinal = int(np.floor(raw + 0.5))  # 标准四舍五入，不用banker's rounding
    s_ordinal = max(0, min(2, s_ordinal))
    return ["S0", "S1", "S2"][s_ordinal]
