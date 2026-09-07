"""
PM 皮肤评估规则——从 pm_skin_scoring/code/questionnaire_app.py 逐字抽出来的纯函数和常量
（用 ast/inspect 自动生成，不是手抄），给 label_service 的 /api/v1/skin/* 接口用；Gradio
那边照旧用它自己的。规则有改动时重新生成：python label_service/tools/sync_skin_rules.py
（生成后会自动跑一遍两边输出一致性校验）。
"""
import math
import re

DOG_NAME_OPTIONS = ["比熊-BB", "金毛-巴利", "中华田园犬-露露", "马尔济斯-小满"]

IMU_DOG_DEFAULT_MAP = {
    "IMU1": "比熊-BB",
    "IMU2": "金毛-巴利",
    "IMU3": "中华田园犬-露露",
    "IMU4": "马尔济斯-小满",
}

SKIN_GROUP_WEIGHT = 0.35

HAIR_GROUP_WEIGHT = 0.25

C_WEIGHT = 0.40

SKIN_COLOR_OPTIONS = [
    ("A. 粉粉的、肉色的，看起来比较正常。", 0),
    ("B. 有一点点发红。", 5),
    ("C. 明显鲜红，但还没有破皮。", 15),
    ("D. 皮肤表面有黑色油油的东西，可以擦下来。", 10),
    ("E. 皮肤变黑、变褐色或变成淡褐色，像是颜色沉下去了，无法擦下来。", 5),
]

ODOR_OPTIONS = [
    ("A. 没有什么异味。", 0),
    ("B. 只有凑近闻，才能闻到一点油脂味、潮湿味。", 5),
    ("C. 离它大概30-50cm，就能闻到比较明显的臭味。", 10),
    ("D. 一进屋或者离得很远，就能闻到恶臭。", 20),
]

LESION_OPTIONS = [
    ("A. 皮肤完整，看不出异常。", 0),
    ("B. 只是有点干，有少量细小白色皮屑。", 5),
    ("C. 皮肤上有一块块大于1厘米异常区域，或者有成片、成块的皮屑/表皮脱落。", 15),
    ("D. 有糜烂、液体、结痂、脓包、红疙瘩，或者皮肤裂开。", 20),
]

HAIR_SPOT_OPTIONS = [
    ("A. 没有明显的无毛或毛发稀疏。", 0),
    ("B. 有1-2个小地方没有毛或毛发稀疏，比如爪子、耳朵边、肚子局部。", 10),
    ("C. 有3处或更多明显没有毛或毛发稀疏的地方。", 15),
    ("D. 脱毛连成一大片，不是零星小块。", 20),
]

HAIR_DIAMETER_OPTIONS = [
    ("A. 没有脱毛区域。", 0),
    ("B. 最大的一块很小，直径不到1-2cm。", 10),
    ("C. 最大的一块大约超过2-3cm。", 15),
    ("D. 最大的一块超过3cm，面积比较明显。", 20),
]

COAT_QUALITY_OPTIONS = [
    ("A. 毛发光亮、顺滑、浓密，看起来比较健康。", 0),
    ("B. 毛发有点油、容易打结，摸起来不太清爽。", 5),
    ("C. 有小范围毛发断裂、变稀、变少。", 10),
    ("D. 大部分毛发都明显干枯、易断、毛质很差。", 20),
]

C_BASELINE_DENOM_FLOOR = 3

C_DELTA_TIERS = [
    (30, 20, 15, 3.0),
    (20, 10, 10, 2.0),
    (10, 5, 5, 1.5),
    (5, 3, 3, 1.3),
]

STATS_CSV_NAME = "imu_daily_scratch_stats.csv"

DEFAULT_IMU_STATS_ROOTS = "infer_result_majority_syn, infer_result_majority"

RECORD_COLUMNS = [
    "狗狗名字", "填表日期", "填写人",
    "有无毛发稀疏", "皮肤颜色", "体味", "皮损", "秃毛分布", "秃毛面积", "整体毛质",
    "问答分数", "保存时间",
]

WEEKLY_REPORT_COLUMNS = [
    "日期", "今日佩戴时长", "今日佩戴时间段",
    "模型-抓挠次数", "模型-抓挠总时长(分钟)", "模型-C级评级", "模型-S级评级", "模型-导出事件记录",
    "人工-抓挠次数", "人工-抓挠总时长(分钟)", "人工-睡眠打断次数", "人工-导出事件记录",
    "对比-错误集(模型vs人工)", "对比-误差(模型vs人工)",
    "兽医1-姓名", "兽医1-抓挠次数", "兽医1-抓挠总时长(分钟)", "兽医1-导出事件记录", "兽医1-睡眠打断次数",
    "兽医1-C级评级", "兽医1-S评分", "兽医1-状态评估描述", "兽医1-问答分数",
    "兽医2-姓名", "兽医2-睡眠打断次数", "兽医2-C级评级", "兽医2-S评分", "兽医2-状态评估描述", "兽医2-问答分数",
    "对比-错误集(人工vs兽医)", "对比-误差(人工vs兽医)",
    "误差分析-抓挠次数误差(人工vs兽医1)", "误差分析-抓挠时长误差(人工vs兽医1)",
    "误差分析-C评级误差(兽医1vs兽医2)", "误差分析-S评分误差(兽医1vs兽医2)",
    "素材库地址",
]

(W_DATE, W_WEAR_HOURS, W_WEAR_RANGE,
 W_M_COUNT, W_M_DUR, W_M_C, W_M_S, W_M_EVENTS,
 W_H_COUNT, W_H_DUR, W_H_INTERRUPT, W_H_EVENTS,
 W_CMP_MH_ERR_SET, W_CMP_MH_ERR,
 W_V1_NAME, W_V1_COUNT, W_V1_DUR, W_V1_EVENTS, W_V1_INTERRUPT,
 W_V1_C, W_V1_S, W_V1_DESC, W_V1_QSCORE,
 W_V2_NAME, W_V2_INTERRUPT, W_V2_C, W_V2_S, W_V2_DESC, W_V2_QSCORE,
 W_CMP_HV_ERR_SET, W_CMP_HV_ERR,
 W_ERR_COUNT_HV1, W_ERR_DUR_HV1, W_ERR_C_V1V2, W_ERR_S_V1V2,
 W_MATERIAL_URL) = range(len(WEEKLY_REPORT_COLUMNS))

_WEEKLY_DEFAULT_CHAIN = [
    (W_H_COUNT, W_M_COUNT),
    (W_H_DUR, W_M_DUR),
    (W_V1_COUNT, W_H_COUNT),
    (W_V1_DUR, W_H_DUR),
    (W_V1_C, W_M_C),
    (W_V2_C, W_V1_C),
    (W_V2_S, W_V1_S),
]

WEEKLY_FORM_INDICES = [i for i in range(len(WEEKLY_REPORT_COLUMNS)) if i != W_DATE]

WEEKLY_AUTOFILL_INDICES = {W_WEAR_HOURS, W_M_COUNT, W_M_DUR, W_M_C}


def _round_half_up(value: float, ndigits: int) -> float:
    """标准四舍五入，不用Python内置round()——它是banker's rounding
    (round-half-to-even)，round(74.25, 1)算出来是74.2不是74.25，之前
    在skin_health/合成数据标签生成那边就踩过同一个坑(见
    rf_synthetic_validation_findings.md)，这里用floor(x+0.5)方式避开。"""
    factor = 10 ** ndigits
    return math.floor(value * factor + 0.5) / factor


def _score_of(choice: str, options: list) -> int:
    if not choice:
        return 0
    for text, score in options:
        if text == choice:
            return score
    return 0


def _question_group_scores(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat):
    """问答部分的共用计算——"填写问答"标签和"S总分"标签都要用到同一套
    算法，抽成一个函数避免两处各写一遍、以后改分值表容易漏改一处。
    前置问题选"否"时，秃毛分布/秃毛面积这两题不需要回答，按0分计（不是
    "扣分"，是这两题在这只狗身上根本不适用，PM原表里这种情况也是记0分，
    不是留空导致的缺失）。整体毛质题不受前置问题限制，始终参与计分。
    返回每题原始分 + 两组小计，供调用方自己决定要不要乘权重/怎么展示。"""
    s_color = _score_of(color, SKIN_COLOR_OPTIONS)
    s_odor = _score_of(odor, ODOR_OPTIONS)
    s_lesion = _score_of(lesion, LESION_OPTIONS)
    s_coat = _score_of(coat, COAT_QUALITY_OPTIONS)

    if has_hair_loss == "是":
        s_spot = _score_of(hair_spot, HAIR_SPOT_OPTIONS)
        s_diameter = _score_of(hair_diameter, HAIR_DIAMETER_OPTIONS)
    else:
        s_spot = 0
        s_diameter = 0

    skin_group_raw = s_color + s_odor + s_lesion
    hair_group_raw = s_spot + s_diameter + s_coat
    return {
        "color": s_color, "odor": s_odor, "lesion": s_lesion,
        "spot": s_spot, "diameter": s_diameter, "coat": s_coat,
        "skin_group_raw": skin_group_raw, "hair_group_raw": hair_group_raw,
    }


def compute_score(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat):
    """"填写问答"标签用——只算问答部分，不含C值，这个函数的输入/输出
    签名不再变动。"""
    g = _question_group_scores(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat)
    skin_group_score = g["skin_group_raw"] * SKIN_GROUP_WEIGHT
    hair_group_score = g["hair_group_raw"] * HAIR_GROUP_WEIGHT
    total = _round_half_up(skin_group_score + hair_group_score, 2)

    # 红旗提示——纯信息展示，不影响total这个数字本身，"填写问答"标签的
    # 计算逻辑/输出签名不变。体味/皮损/秃毛分布/秃毛面积/整体毛质这5题
    # 满分都是20(皮肤颜色满分只有15，够不到)，选中满分选项时提前告诉
    # 填表人"这题去S总分标签会直接判S2"，不用等填完切到S总分标签才发现
    red_flag_titles = {"odor": "体味", "lesion": "皮损", "spot": "秃毛分布",
                       "diameter": "秃毛面积", "coat": "整体毛质"}
    red_flag_hit = [title for key, title in red_flag_titles.items() if g[key] == 20]
    red_flag_line = (f"\n> 🚩 「{'、'.join(red_flag_hit)}」选中了满分选项——这几题只要"
                     f"有一题是满分，去「S总分」标签算的时候会不看加权总分、直接判S2"
                     if red_flag_hit else "")

    breakdown = (
        f"| 题目 | 原始分 |\n|---|---|\n"
        f"| 皮肤颜色 | {g['color']} |\n"
        f"| 体味 | {g['odor']} |\n"
        f"| 皮损 | {g['lesion']} |\n"
        f"| **皮肤状态组小计 × {SKIN_GROUP_WEIGHT:.0%}** | **{g['skin_group_raw']} × {SKIN_GROUP_WEIGHT:.0%} = {skin_group_score:.2f}** |\n"
        f"| 秃毛分布 | {g['spot']}{'（未评估，计0分）' if has_hair_loss != '是' else ''} |\n"
        f"| 秃毛面积 | {g['diameter']}{'（未评估，计0分）' if has_hair_loss != '是' else ''} |\n"
        f"| 整体毛质 | {g['coat']} |\n"
        f"| **毛发状态组小计 × {HAIR_GROUP_WEIGHT:.0%}** | **{g['hair_group_raw']} × {HAIR_GROUP_WEIGHT:.0%} = {hair_group_score:.2f}** |\n"
        f"| **问答分数合计** | **{total}** |\n"
        f"{red_flag_line}"
    )
    return total, breakdown


def c_tier_of(c_value):
    """C0：0≤C<30，C1：30≤C<50，C2：50≤C≤100，跟scratch_burden.py的
    score_day()判定完全一致。"""
    if c_value is None:
        return ""
    if c_value >= 50:
        return "C2"
    if c_value >= 30:
        return "C1"
    return "C0"


def s_tier_of(s_value, red_flag=False):
    """PM确认后的精确边界：S0：0≤S<12；S1：12≤S<20；S2：20≤S≤74.25。
    74.25正好是公式本身能算出的理论最大值(C=100×40%+皮肤组满分55×35%+
    毛发组满分60×25%)，S2不用额外设上限，公式自身就封顶在这。

    另外有一条红旗规则：任意单项（体味/皮损/秃毛分布/秃毛面积/整体
    毛质，这5题满分都是20，皮肤颜色满分只有15、够不到这条规则）打了
    20分满分，不管总分算出来多少，直接判S2——跟C值那边"红旗信号触发
    直接给C2"是同一类"某个维度已经严重到可以跳过加权总分直接定档"的
    设计。"""
    if s_value is None:
        return ""
    if red_flag:
        return "S2"
    if s_value >= 20:
        return "S2"
    if s_value >= 12:
        return "S1"
    return "S0"


def _c_delta_score_one(current, baseline, use_duration: bool):
    """PM表格里"次数/时长最小绝对增加"这一栏写的是"≥"(下限含)，但"相对倍数"
    这一栏写的是"1.3倍 < 相对倍数 ≤ 1.5倍"这种——下限用`<`(不含)、上限用`≤`
    (含)，只有最高档"相对倍数 > 3倍"没有上限、纯大于。两栏的边界含义不一样，
    不能统一用>=判断：之前两个条件都用>=，导致倍数刚好等于某一档下限时
    (比如正好3.00倍、正好1.5倍)会被错误地算进上一档——比如倍数=3.00时，
    PM表格里"2倍<相对倍数≤3倍"这档(20分)才是倍数=3.00该落的档，因为上限
    3倍是"含"的，"相对倍数>3倍"那档(30分/红旗)要求严格大于3，3.00不满足；
    但之前用>=3.0判断的话3.00会被误判成命中最高档，多算10分还多触发一个
    不该有的红旗。"""
    denom = max(baseline, C_BASELINE_DENOM_FLOOR)
    ratio = current / denom
    abs_increase = current - baseline
    for score, abs_min_count, abs_min_dur_min, ratio_min in C_DELTA_TIERS:
        abs_min = abs_min_dur_min if use_duration else abs_min_count
        if abs_increase >= abs_min and ratio > ratio_min:
            return score, ratio
    return 0, ratio


def _c_score_delta(baseline_count, baseline_duration_min, today_count, today_duration_min):
    """双门槛+双值取大：次数、时长分别判定，取分高的一个当"变化幅度"分。"""
    count_score, count_ratio = _c_delta_score_one(today_count, baseline_count, use_duration=False)
    dur_score, dur_ratio = _c_delta_score_one(today_duration_min, baseline_duration_min, use_duration=True)
    if count_score >= dur_score:
        return count_score, "次数", count_ratio
    return dur_score, "时长", dur_ratio


def _c_score_cluster(cluster_count):
    if cluster_count >= 3:
        return 20, True  # 红旗
    if cluster_count >= 1:
        return 10, False
    return 0, False


def _c_score_persistence(consecutive_days):
    if consecutive_days >= 3:
        return 20, True  # 红旗
    if consecutive_days == 2:
        return 10, False
    if consecutive_days == 1:
        return 5, False
    return 0, False


def _c_score_interruption(zn, zd, long_scratch):
    """2026-08更新：跟兽医同事重新讨论后的版本——ZD<3(即1或2次)算20分，
    ZD>=3或触发长时间抓挠(满足一项即可，不用同时满足)才是30分红旗，之前
    版本的门槛是ZD==1给20分、ZD>=2就跳到30分红旗，已经改成PM文档最新的
    这版数值。"""
    if zd >= 3 or long_scratch:
        return 30, True  # 红旗
    if zd >= 1:
        return 20, False
    if zn > 5:
        return 10, False
    return 0, False


def compute_c_score(baseline_count, baseline_duration_min, today_count, today_duration_min,
                    cluster_count, persistence_days, zn, zd, long_scratch,
                    has_baseline=True):
    """has_baseline=False表示这只狗还没有建立个人基线（比如设备刚戴上的
    第一天、或者之前的天佩戴时长都不达标）。这种情况下"变化幅度"这一项
    不计分、也不触发红旗——不能把"没有基线"当成"基线是0次"来算：那样
    今天抓了26次会被算成"相对基线涨了8.67倍"，凭空造出一个>3倍的红旗，
    直接把这天判成C2，其实根本没有可比的历史数据。其余三项(聚集/持续/
    中断)本身不依赖基线，照常计分。"""
    # 数字输入框在没填的时候Gradio给的是None，统一按0处理，不报错
    baseline_count = baseline_count or 0
    baseline_duration_min = baseline_duration_min or 0
    today_count = today_count or 0
    today_duration_min = today_duration_min or 0
    cluster_count = cluster_count or 0
    persistence_days = persistence_days or 0
    zn = zn or 0
    zd = zd or 0

    if has_baseline:
        delta_score, delta_by, delta_ratio = _c_score_delta(
            baseline_count, baseline_duration_min, today_count, today_duration_min)
        # 红旗＝真的落进了30分那一档(_c_delta_score_one按"绝对增加和相对
        # 倍数同时满足"判定出来的)，不能只看"相对倍数>3"这一个条件就直接
        # 判红旗——这是之前一个真实bug：≥20次/≥15分钟这个绝对门槛没达到，
        # 但相对倍数因为基线值很小意外超过3倍时(比如基线1次涨到10次，
        # 倍数3.33但只增加了9次，够不到20次这个绝对门槛)，之前的代码会
        # 不看绝对门槛直接把delta_score强改成30分红旗，跟_c_delta_score_one
        # 自己按"双门槛"算出来的分数(这个例子应该是20分，时长那边10分钟
        # 时长增量够不到15分钟的绝对门槛)不一致，页面上会多算出10分、还会
        # 凭空触发一个不该有的红旗
        delta_red_flag = delta_score >= 30
        delta_note = f"按{delta_by}判定，相对基线比值={delta_ratio:.2f}"
    else:
        # 没有基线：这一项不计分也不触发红旗。注意这样算出来的C值会偏低
        # （最多只剩70分），是"信息不足所以先不下这个结论"，不是"确实没
        # 涨"——页面上会单独提示，不要拿这天的C值直接跟有基线的天比
        delta_score, delta_red_flag = 0, False
        delta_note = "⚠️ 还没有个人基线，这一项暂不计分（不是0分，是没法判定）"

    cluster_score, cluster_red_flag = _c_score_cluster(cluster_count)
    persistence_score, persistence_red_flag = _c_score_persistence(persistence_days)
    interrupt_score, interrupt_red_flag = _c_score_interruption(zn, zd, bool(long_scratch))

    total = max(0, min(100, delta_score + cluster_score + persistence_score + interrupt_score))

    # 红旗直接定档C2，不是"红旗那一项封顶分数、再按总分套阈值"——PM原文的
    # 举例七写得很明确："满足显著变化条件5，得分30分。触发红旗信号，直接
    # 进入C2"：这天的变化幅度红旗单独触发时，delta_score=30、其余三项都是
    # 0，加总只有30分，按30/50阈值本该只是C1，但她的例子明确说"直接进入
    # C2"，说明红旗是跳过总分阈值判断、直接定档的，不是只把红旗那一项的
    # 分数封顶。四个维度任意一个触发红旗都适用这条规则，不只是变化幅度。
    # （这一点skin_health/code/scratch_burden.py目前的实现里没有做——那边
    # 只把红旗项的分数封顶、还是按总分走30/50阈值，等于是漏了这条规则，
    # 已经记录在pm_rules_verification.md，需要跟算法组同步一下）
    any_red_flag = delta_red_flag or cluster_red_flag or persistence_red_flag or interrupt_red_flag
    if any_red_flag:
        tier = "C2"
    elif total >= 50:
        tier = "C2"
    elif total >= 30:
        tier = "C1"
    else:
        tier = "C0"

    red_flags = []
    if delta_red_flag:
        red_flags.append(f"变化幅度({delta_by})相对基线>3倍")
    if cluster_red_flag:
        red_flags.append("聚集时段≥3个")
    if persistence_red_flag:
        red_flags.append("持续≥3天")
    if interrupt_red_flag:
        red_flags.append("睡眠中断≥3次或触发长时间抓挠")
    red_flag_line = (f"\n> 🚩 红旗信号：{'、'.join(red_flags)}，直接判定C2，不看加权总分是否达到50分"
                     if red_flags else "")

    no_baseline_line = ("\n> ⚠️ 这只狗还没有个人基线，「变化幅度」这一项暂不计分，"
                        "C值上限只有70分，会偏低——这是「信息不足、先不下结论」，"
                        "不是「确实没变化」，不要直接拿这天的C值跟有基线的天比"
                        if not has_baseline else "")

    breakdown = (
        f"| 打分项 | 得分 | 说明 |\n|---|---|---|\n"
        f"| 变化幅度(0-30) | {delta_score} | {delta_note} |\n"
        f"| 聚集程度(0-20) | {cluster_score} | 今日聚集时段数={cluster_count} |\n"
        f"| 持续程度(0-20) | {persistence_score} | 连续天数={persistence_days} |\n"
        f"| 中断影响(0-30) | {interrupt_score} | 普通抓挠ZN={zn}次，打断睡眠ZD={zd}次，"
        f"长时间抓挠={'是' if long_scratch else '否'} |\n"
        f"| **C值合计** | **{total}** | → **{tier}** |\n"
        f"{red_flag_line}"
        f"{no_baseline_line}"
    )
    return total, tier, breakdown


def compute_s_total(c_value, c_tier_hint, has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat):
    """"S总分"标签用——读"填写问答"标签里已经选好的答案（复用
    _question_group_scores，跟"填写问答"标签算问答分数是同一套逻辑，
    不会算出两个不一致的数字）+ 这个标签自己填的C值，组合成最终S总分。

    c_tier_hint是从"C值计算"标签同步过来的真实档位(那边已经按红旗规则
    算好了，比如"聚集时段≥3个"这种红旗会直接判C2，不是单纯看C值总分够
    不够50)。这里不能拿到c_value之后自己用c_tier_of()按30/50阈值重新
    粗算一遍——那样会漏掉红旗，把该判C2的算成C1。c_tier_hint留空(比如
    用户没去"C值计算"标签算过、是直接手动填的C值)时才退回c_tier_of()的
    简单阈值判断，这是唯一能做到的兜底，因为手动填的C值本身就不带红旗
    信息。

    C是C2(不管是总分够50、还是红旗触发)时，S也直接判S2，不看加权总分——
    跟问答单项满分20分触发红旗是同一个"某个维度已经严重到能跳过总分
    直接定档"的设计，PM确认过这条规则。"""
    g = _question_group_scores(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat)
    skin_group_score = g["skin_group_raw"] * SKIN_GROUP_WEIGHT
    hair_group_score = g["hair_group_raw"] * HAIR_GROUP_WEIGHT

    c = c_value if c_value is not None else 0
    c_score = c * C_WEIGHT
    total = _round_half_up(c_score + skin_group_score + hair_group_score, 2)

    c_tier = c_tier_hint if c_tier_hint else c_tier_of(c_value)
    question_red_flag = 20 in (g["odor"], g["lesion"], g["spot"], g["diameter"], g["coat"])
    c2_red_flag = c_tier == "C2"
    red_flag = question_red_flag or c2_red_flag
    s_tier = s_tier_of(total, red_flag)

    c_line = (f"| C值 | {c} × {C_WEIGHT:.0%} = {c_score:.2f} | [{c_tier}] |\n"
             if c_value is not None else
             "| ⚠️ 还没填C值 | 按C=0算 | — |\n")
    red_flag_reasons = []
    if question_red_flag:
        red_flag_reasons.append("问答部分有单项打了20分满分")
    if c2_red_flag:
        red_flag_reasons.append("C值判定为C2")
    red_flag_line = (f"\n> 🚩 触发红旗信号：{'、'.join(red_flag_reasons)}，S档位直接判S2，不看加权总分"
                     if red_flag_reasons else "")

    breakdown = (
        f"| 组成部分 | 加权分 | 档位 |\n|---|---|---|\n"
        f"{c_line}"
        f"| 皮肤状态组 | {g['skin_group_raw']} × {SKIN_GROUP_WEIGHT:.0%} = {skin_group_score:.2f} | — |\n"
        f"| 毛发状态组 | {g['hair_group_raw']} × {HAIR_GROUP_WEIGHT:.0%} = {hair_group_score:.2f} | — |\n"
        f"| **S总分 = C×40%+皮肤组×35%+毛发组×25%** | **{total}** | **{s_tier}** |\n"
        f"{red_flag_line}"
    )
    return total, c_tier, s_tier, breakdown


def _letter_of(choice: str) -> str:
    """把"D. 皮肤表面有黑色油油的东西，可以擦下来。"这种完整选项文案
    压缩成只留最前面的选项字母"D"——历史记录表格/导出CSV只需要知道
    选了哪个选项，不需要每次都把一整句选项原文堆在格子里。has_hair_loss
    这题本身答案就是"是"/"否"两个字，不是"A./B."这种格式，原样返回。"""
    if not choice:
        return ""
    if len(choice) >= 2 and choice[0] in "ABCDE" and choice[1] == ".":
        return choice[0]
    return choice


def _missing_questions(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat):
    """检查必填题有没有漏填。皮肤颜色/体味/皮损/整体毛质这4题始终必填；
    秃毛分布/秃毛面积只有前置问题选"是"时才必填——选"否"时这两题界面上
    本来就是隐藏的，不该也不会要求填，留空(NaN)是它们在"否"这个分支下
    唯一合法的状态，不是"漏填"。返回缺失的题目名称列表，空列表=都填好了。"""
    missing = []
    if not has_hair_loss:
        missing.append("前置问题（有无毛发稀疏）")
    if not color:
        missing.append("皮肤颜色")
    if not odor:
        missing.append("体味")
    if not lesion:
        missing.append("皮损")
    if not coat:
        missing.append("整体毛质")
    if has_hair_loss == "是":
        if not hair_spot:
            missing.append("秃毛分布")
        if not hair_diameter:
            missing.append("秃毛面积")
    return missing


def _parse_bool(value) -> bool:
    """CSV里读出来的都是字符串，"False"这种非空字符串直接bool()会误判成
    True，要按内容判断。"""
    return str(value).strip().lower() in ("true", "1", "是")


def _to_float_or_none(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_iso_date(day_str: str) -> str:
    """"2026_8_19" → "2026-08-19"，用来填进gr.DateTime(type="string")的
    填表日期组件——那边存的是标准"YYYY-MM-DD"格式，目录名格式不一样，
    这里转一下。转不了就原样返回，不报错，让用户自己在「填写问答」标签
    核对/改一下日期就好。"""
    m = re.match(r"^(\d{4})_(\d{1,2})_(\d{1,2})$", day_str or "")
    if not m:
        return day_str or ""
    y, mo, d = m.groups()
    return f"{y}-{int(mo):02d}-{int(d):02d}"


def _dog_breed(dog_name_val):
    """"比熊-BB" → "比熊"——DOG_NAME_OPTIONS里的命名约定本来就是"品种-
    名字"，取"-"前面那段就是品种，刚好也是rf_infer模型训练时breed_map
    用的同一套品种字符串(比熊/金毛/中华田园犬/马尔济斯)，不用额外维护
    一份映射表。"""
    if not dog_name_val or "-" not in dog_name_val:
        return None
    return dog_name_val.split("-", 1)[0]


def _pm_answers_to_rf_ordinals(color, odor, lesion, hair_spot, hair_diameter, coat):
    """任意一题没选(None/空字符串)时那一项直接不放进返回的dict——调用方
    (rf_infer.predict_s)会把没给的问答特征当NaN处理，不是当成"选了最轻
    档"，缺答案跟"确认是最轻档"是两回事，不能用0悄悄顶替。"""

    def _letter(choice):
        if not choice or len(choice) < 2 or choice[1] != ".":
            return None
        return choice[0]

    ordinals = {}

    color_letter = _letter(color)
    if color_letter in ("A", "B", "C"):
        ordinals["skin_redness_level"] = {"A": 0, "B": 1, "C": 2}[color_letter]
        ordinals["skin_pigment_abnormal"] = 0
    elif color_letter in ("D", "E"):
        ordinals["skin_redness_level"] = 0
        ordinals["skin_pigment_abnormal"] = 1

    letter_to_ordinal = {"A": 0, "B": 1, "C": 2, "D": 3}
    odor_letter = _letter(odor)
    if odor_letter in letter_to_ordinal:
        ordinals["odor_level"] = letter_to_ordinal[odor_letter]
    lesion_letter = _letter(lesion)
    if lesion_letter in letter_to_ordinal:
        ordinals["skin_lesion_severity"] = letter_to_ordinal[lesion_letter]
    spot_letter = _letter(hair_spot)
    if spot_letter in letter_to_ordinal:
        ordinals["hair_loss_spot_count_level"] = letter_to_ordinal[spot_letter]
    diameter_letter = _letter(hair_diameter)
    if diameter_letter in letter_to_ordinal:
        ordinals["hair_loss_max_diameter_level"] = letter_to_ordinal[diameter_letter]
    coat_letter = _letter(coat)
    if coat_letter in letter_to_ordinal:
        ordinals["coat_quality_level"] = letter_to_ordinal[coat_letter]

    return ordinals


def _tier_match_note(a, b) -> str:
    """两个档位字符串（C0/C1/C2或S0/S1/S2这类）对比，都非空时给"一致"/
    "不一致"，任意一边空着就不下结论（不能把"还没填"当成"不一致"）。"""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return ""
    return "一致" if a == b else "不一致"


def recompute_weekly_errors(rows: list) -> list:
    """基于表格里当前已经填的"模型-"/"人工-"/"兽医1-"/"兽医2-"这几列，
    重新算一遍"对比-"/"误差分析-"这几列：
      - 对比-误差(模型vs人工)：次数+时长的绝对误差，合并成一句话
        （原模板这里只有一个"误差"格，不像"误差分析"那组拆成4个独立列）
      - 误差分析-抓挠次数/时长误差(人工vs兽医1)：原模板里只有人工和兽医1
        两边都填了具体的次数/时长数字，兽医2这个模板本身没有次数/时长列，
        没法比
      - 误差分析-C评级/S评分误差(兽医1vs兽医2)：C级/S评分是档位
        （C0/C1/C2、S0/S1/S2），不是连续数值，算不出"误差"这种数值，
        只能标两位兽医的判断是不是一致；人工标注这个模板本身没有单独的
        C级/S评分列，没法拿人工的档位去跟兽医比，所以这两列比的是
        兽医1vs兽医2这两位审核人互相之间是否一致，不是跟人工比
      - 对比-错误集/误差(模型vs人工)、(人工vs兽医)这两组文字性的"错误集"
        列不做任何自动填写，具体差在哪需要人工去看数据后自己写
    每次点按钮都是基于当前表格内容重新算一遍、覆盖旧的对比结果，不是
    累加。"""
    out = []
    for row in rows:
        row = list(row) + [""] * (len(WEEKLY_REPORT_COLUMNS) - len(row))  # 兼容旧的、列数还没补齐的行

        model_count, human_count = _to_float_or_none(row[W_M_COUNT]), _to_float_or_none(row[W_H_COUNT])
        model_dur,   human_dur   = _to_float_or_none(row[W_M_DUR]),   _to_float_or_none(row[W_H_DUR])
        if model_count is not None and human_count is not None and model_dur is not None and human_dur is not None:
            row[W_CMP_MH_ERR] = f"次数差{abs(model_count - human_count):g}次，时长差{abs(model_dur - human_dur):.1f}分钟"
        else:
            row[W_CMP_MH_ERR] = ""

        vet1_count, vet1_dur = _to_float_or_none(row[W_V1_COUNT]), _to_float_or_none(row[W_V1_DUR])
        row[W_ERR_COUNT_HV1] = abs(human_count - vet1_count) if human_count is not None and vet1_count is not None else ""
        row[W_ERR_DUR_HV1] = round(abs(human_dur - vet1_dur), 1) if human_dur is not None and vet1_dur is not None else ""

        row[W_ERR_C_V1V2] = _tier_match_note(row[W_V1_C], row[W_V2_C])
        row[W_ERR_S_V1V2] = _tier_match_note(row[W_V1_S], row[W_V2_S])

        out.append(row)
    return out


def _apply_weekly_defaults(row: list) -> list:
    row = list(row)
    for target, source in _WEEKLY_DEFAULT_CHAIN:
        if not (row[target] or "").strip() and (row[source] or "").strip():
            row[target] = row[source]
    return row

