"""
PM版皮肤问答打分页面——一比一复刻PM原始Wardyn题库的固定加权公式，不用
机器学习，每个选项对应她CSV里写死的分值，选完直接加总出"问答分数"。

跟skin_health/那套RF方案是两个完全独立的东西，不要混——那边是算法组
自己设计的、权重靠训练学出来的方案；这个目录专门给PM的原始方案一个
独立的、可以直接用起来的页面，方便对照/过渡期两边同时用。

分值表来源：PM提供的Wardyn题库CSV（PFSY01-06），字段名/选项文案跟
questionnaire_paper_form.md保持一致（那份是纯离线纸质表，这个是同一套
题目的在线版，方便自动算分）。

页面分三个独立标签：
  1. 填写问答——纯问答部分，算出"问答分数"（皮肤组×35%+毛发组×25%），
     不含C值，这个标签的行为/输出格式不再改动
  2. C值计算——IMU行为严重度打分器，按PM文档的变化幅度/聚集程度/
     持续程度/中断影响四项规则(跟skin_health/code/scratch_burden.py
     的SBS引擎已核对一致，见pm_rules_verification.md)，手动填结构化的
     统计量(次数/时长/聚集时段数等)算出C值，不需要接原始事件时间戳
  3. S总分——读取"填写问答"标签里已经选好的答案 + "C值计算"标签算出
     的C值，实时组合成最终S总分/S档位，三个标签各自确认好就能看到
     最终结果，不用来回抄数字

用法：
    python pm_skin_scoring/code/questionnaire_app.py
    浏览器打开 http://localhost:7860
"""
import csv
import glob
import json
import math
import os
import re
import socket
import sys
from datetime import datetime

import gradio as gr


def _round_half_up(value: float, ndigits: int) -> float:
    """标准四舍五入，不用Python内置round()——它是banker's rounding
    (round-half-to-even)，round(74.25, 1)算出来是74.2不是74.25，之前
    在skin_health/合成数据标签生成那边就踩过同一个坑(见
    rf_synthetic_validation_findings.md)，这里用floor(x+0.5)方式避开。"""
    factor = 10 ** ndigits
    return math.floor(value * factor + 0.5) / factor


DOG_NAME_OPTIONS = ["比熊-BB", "金毛-巴利", "中华田园犬-露露", "马尔济斯-小满"]

# IMU编号→狗狗名字的默认对应关系，给「导入IMU统计数据」标签用——选好机位
# 后自动带出这个默认值，省得每次手动选。但这只是默认值，不是写死的规则，
# 下拉框始终保持可编辑，不能锁死——尤其IMU1，历史数据里对应过的狗不一定
# 一直是同一只，选完务必核对一下再点应用。
IMU_DOG_DEFAULT_MAP = {
    "IMU1": "比熊-BB",
    "IMU2": "金毛-巴利",
    "IMU3": "中华田园犬-露露",
    "IMU4": "马尔济斯-小满",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RECORDS_CSV = os.path.join(DATA_DIR, "records.csv")
RECORD_COLUMNS = [
    "狗狗名字", "填表日期", "填写人",
    "有无毛发稀疏", "皮肤颜色", "体味", "皮损", "秃毛分布", "秃毛面积", "整体毛质",
    "问答分数", "保存时间",
]


def get_lan_ip() -> str:
    """探测这台机器对外的局域网IP（不实际发送数据，只是借这个UDP连接动作
    让操作系统选一个出口网卡地址）。用来在启动日志里打印局域网内其他设备
    能访问到的地址，不然server_name="0.0.0.0"打印出来的0.0.0.0本身不是
    一个能在浏览器里直接打开的地址，容易让人误以为要打开"0.0.0.0"。
    探测不到就退回127.0.0.1，只有本机能用，不影响本机自己访问。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


# ── PM原始分值表（PFSY01-06，来自Wardyn题库CSV）──────────────────────────
# 每题: (题干, [(选项文案, 分值), ...])
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


def _score_of(choice: str, options: list) -> int:
    if not choice:
        return 0
    for text, score in options:
        if text == choice:
            return score
    return 0


# PM原表按大类分组加权：皮肤状态(皮肤颜色+体味+皮损)组内原始分相加后
# 乘以35%权重；毛发状态(秃毛部位+秃毛面积+整体毛质)组内原始分相加后
# 乘以25%权重，两组加权分相加得到最终问答分数——不是6道题原始分直接
# 相加，PM原表里这两组权重加起来是60%，还有别的类别(抓挠行为，对应
# C值那部分)占剩下的40%。
SKIN_GROUP_WEIGHT = 0.35
HAIR_GROUP_WEIGHT = 0.25
C_WEIGHT = 0.40


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


# ── C值计算：IMU行为严重度打分器（变化幅度/聚集程度/持续程度/中断影响）──
#
# 规则跟skin_health/code/scratch_burden.py的SBS引擎逐条核对过、完全一致
# （见pm_rules_verification.md第一节），这里是同一套规则的独立实现，用
# 结构化统计量当输入（次数/时长/聚集时段数……），不需要接原始逐次抓挠
# 事件时间戳——那个属于要连上真实IMU数据管线才能做的事，这个页面只是
# 手动核对/演示用的计算器。
#
# 分母保护值：PM文档写"分母最低按3计算"，这里按字面数字用3，故意跟
# scratch_burden.py里的BASELINE_DENOM_FLOOR=1不一致——那边是那个模块
# 自己的设计选择(注释写"只为避免除以0，不是用来压低低基线狗信号的")，
# 这里既然是"一比一复刻"就按PM原文字面数字来，两处不同已经记录在
# pm_rules_verification.md备注1，不是疏漏。
C_BASELINE_DENOM_FLOOR = 3

# (score, 次数最小绝对增加, 时长最小绝对增加(分钟), 相对倍数下限)
C_DELTA_TIERS = [
    (30, 20, 15, 3.0),
    (20, 10, 10, 2.0),
    (10, 5, 5, 1.5),
    (5, 3, 3, 1.3),
]


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


# ── 历史记录：本地CSV持久化（只属于"填写问答"标签，行为不变）──────────────

def load_records() -> list:
    """返回records.csv里的全部记录（不含表头），每条是一个list，按
    RECORD_COLUMNS的顺序。文件不存在时返回空列表，不报错——第一次用
    这个页面、还没保存过任何记录是正常状态。"""
    if not os.path.exists(RECORDS_CSV):
        return []
    with open(RECORDS_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return rows[1:] if rows else []  # 去掉表头行


def _write_all(rows: list) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RECORDS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(RECORD_COLUMNS)
        writer.writerows(rows)


def find_duplicate_index(rows: list, dog_name: str, fill_date: str, filler: str):
    """狗狗名字+填表日期+填写人三者都相同才算重复——这三个凑一起唯一
    标识"这次填表"，避免同一天同一个人给同一只狗重复交了两份不同答案的
    表，后一份把前一份静默覆盖掉。返回命中的行下标，没有命中返回None。"""
    for i, row in enumerate(rows):
        if len(row) >= 3 and row[0] == dog_name and row[1] == fill_date and row[2] == filler:
            return i
    return None


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


def save_record(dog_name, fill_date, filler, has_hair_loss, color, odor, lesion,
                hair_spot, hair_diameter, coat, total_score, confirm_overwrite):
    """保存一条问答记录。狗狗名字/填表日期/填写人三者组合已存在时，默认
    拒绝保存并提示——避免手滑/误操作把之前填好的记录覆盖掉；用户确认
    要覆盖后勾选"确认覆盖"复选框再保存一次，才会真的替换旧记录。"""
    if not dog_name or not fill_date or not filler:
        return "❌ 请先选择狗狗名字、填表日期、填写人再保存", load_records(), gr.update(value=False)

    missing = _missing_questions(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat)
    if missing:
        msg = "❌ 还有必填题没填：" + "、".join(missing)
        return msg, load_records(), gr.update(value=False)

    fill_date_str = fill_date.split(" ")[0] if fill_date else fill_date  # 只保留日期部分
    rows = load_records()
    dup_idx = find_duplicate_index(rows, dog_name, fill_date_str, filler)

    new_row = [dog_name, fill_date_str, filler, has_hair_loss or "",
               _letter_of(color), _letter_of(odor), _letter_of(lesion),
               _letter_of(hair_spot), _letter_of(hair_diameter), _letter_of(coat),
               total_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

    if dup_idx is not None and not confirm_overwrite:
        msg = (f"⚠️ 已存在完全相同的记录（{dog_name} / {fill_date_str} / {filler}），"
               "为避免误操作覆盖旧数据，没有保存。如果确实要用这次的答案覆盖旧记录，"
               "勾选下面「确认覆盖」后再点一次保存")
        return msg, rows, gr.update(value=False)

    if dup_idx is not None:
        rows[dup_idx] = new_row
        msg = f"✅ 已覆盖旧记录：{dog_name} / {fill_date_str} / {filler}"
    else:
        rows.append(new_row)
        msg = f"✅ 已保存新记录：{dog_name} / {fill_date_str} / {filler}"

    _write_all(rows)
    return msg, rows, gr.update(value=False)  # 保存完把"确认覆盖"复选框重置回未勾选


def export_records_csv():
    """导出当前全部历史记录为一个独立CSV文件供下载——直接把records.csv
    本身返回给gr.File就行，不需要另外拷贝一份，本来就是CSV格式。文件
    不存在（还没保存过任何记录）时先创建一个只有表头的空文件，避免
    下载按钮点了报错。"""
    if not os.path.exists(RECORDS_CSV):
        _write_all([])
    return RECORDS_CSV


# ── IMU统计数据导入：读 {root}/{day}/imu_daily_scratch_stats.csv（
# src/imu_scratch_daily_stats.py / run_review_bins_all_days.sh的IMU_STATS=1
# 产出的那份，每天一份、落在各自的日期目录下），供「C值计算」标签按
# (天,机位)选择自动填充。
#
# 不是现场扫描/现算全部_infer.json——之前那样做会把root下所有已经跑过
# ML_PRELABEL的天都翻出来，哪怕IMU_STATS=1只对某一天(比如8-19)跑过，
# 别的天(比如8-18之前那些)照样会被扫出来显示成"有数据"，其实那些天根本
# 没跑过统计，容易误导。改成只认imu_daily_scratch_stats.csv这份文件——
# 这份文件本身就是"用户明确跑过IMU_STATS=1的天才会出现在里面"，没跑过
# 统计的天不会出现在下拉框里，不会看着像有数据。
STATS_CSV_NAME = "imu_daily_scratch_stats.csv"

# 常见的两个推理结果根目录（不同模型跑出来的），预填在输入框里，用户可以
# 改成自己实际用的路径——不写死绝对路径，因为不同机器上的repo根目录
# 不一样，这里给的是相对当前工作目录的常见默认值
DEFAULT_IMU_STATS_ROOTS = "infer_result_majority_syn, infer_result_majority"


# ── ML版对比：加载skin_health/的RF模型，跟PM固定权重公式跑同一份真实数据
# 做对比。skin_health/是算法组自己的独立目录(rf_infer.py/rf_features.py/
# train_rf_model_a.py/train_rf_model_b.py)，不属于pm_skin_scoring自己的
# 代码，import前要把它的code目录加进sys.path。
SKIN_HEALTH_CODE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "skin_health", "code"))
if SKIN_HEALTH_CODE_DIR not in sys.path:
    sys.path.insert(0, SKIN_HEALTH_CODE_DIR)

try:
    import rf_infer  # noqa: E402
    _RF_INFER_IMPORT_ERROR = ""
except Exception as e:  # noqa: BLE001 —— 只是想在页面上给出清楚的提示，
    # 不是要吞掉所有异常类型；导入失败最常见的原因是skin_health/code/
    # rf_features.py依赖的numpy/pandas/scikit-learn版本环境问题，不应该
    # 让整个questionnaire_app.py直接起不来，只影响"ML版对比"这一个标签
    rf_infer = None
    _RF_INFER_IMPORT_ERROR = f"{type(e).__name__}: {e}"

# PM问答选项 → questionnaire_features.py的有序整数特征——PM原题库跟RF
# 问答特征不是一一对应的：皮肤颜色这一题PM原表里同时混了"泛红程度"和
# "色素异常"两件事(A正常/B发红/C鲜红/D黑色油油/E变黑变褐)，RF那边是拆成
# 两个独立特征(skin_redness_level有序0-2 + skin_pigment_abnormal二值)，
# D/E选项映射到色素异常=1、泛红程度给0(这两个选项本身不是在描述泛红，
# 给0不是"没问题"的意思，只是泛红程度这个维度在这两个选项下没有信息)。
# 其余5题(体味/皮损/秃毛分布/秃毛面积/整体毛质)PM的4个选项本来就是从轻到
# 重的有序梯度，直接按选项顺序映射成0/1/2/3，跟questionnaire_features.py
# 里各特征的LEVELS定义(都是4档，0~3)一一对应，不用额外设计映射规则。
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


def _dog_breed(dog_name_val):
    """"比熊-BB" → "比熊"——DOG_NAME_OPTIONS里的命名约定本来就是"品种-
    名字"，取"-"前面那段就是品种，刚好也是rf_infer模型训练时breed_map
    用的同一套品种字符串(比熊/金毛/中华田园犬/马尔济斯)，不用额外维护
    一份映射表。"""
    if not dog_name_val or "-" not in dog_name_val:
        return None
    return dog_name_val.split("-", 1)[0]


def ml_scan_roots(roots_str: str):
    """扫描逗号分隔的推理结果根目录，找每个根目录下每个IMU、每一天的真实
    佩戴数据——直接扫{root}/*/_infer/*_infer.json文件名(不需要
    imu_daily_scratch_stats.csv这份预生成的统计文件，RF这边要的是多天
    历史事件明细，不是聚合过的单日统计量)，用extract_imu_label()取法
    保证跟别处"IMU几"的含义一致。

    返回的rows结构故意跟scan_imu_roots()（"导入IMU统计数据"标签用的那个）
    保持一样的字段名(date/imu/root/root_label)，这样_date_label()/
    update_imu_choices()/default_dog_for_imu()这几个函数可以直接复用，
    两个标签的"先选日期、再选机位、机位带出默认狗狗"这套交互是同一套
    代码，不是照着外观抄一遍再重新实现一遍逻辑。"""
    if rf_infer is None:
        return [], gr.update(choices=[], value=None), f"❌ rf_infer模块导入失败：{_RF_INFER_IMPORT_ERROR}"

    roots = [r.strip() for r in (roots_str or "").split(",") if r.strip()]
    if not roots:
        return [], gr.update(choices=[], value=None), "❌ 请先填至少一个根目录"

    from extract_clips import extract_imu_label  # noqa: E402（skin_health/code已在sys.path里）

    all_rows = []
    skipped = []
    for root in roots:
        if not os.path.isdir(root):
            skipped.append(f"{root}（目录不存在）")
            continue
        infer_jsons = glob.glob(os.path.join(root, "*", "_infer", "*_infer.json"))
        imus_seen = set()
        for path in infer_jsons:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            stem = os.path.splitext(data.get("csv_basename", os.path.basename(path)))[0]
            imus_seen.add(extract_imu_label(stem))
        if not imus_seen:
            skipped.append(f"{root}（没有找到任何_infer.json）")
            continue

        root_label = os.path.basename(root.rstrip("/\\")) or root
        for imu in sorted(imus_seen):
            _, wear = rf_infer.load_events_and_wear(root, imu)
            for d in sorted(wear["date"].astype(str).tolist()):
                all_rows.append({"date": d, "imu": imu, "root": root, "root_label": root_label})

    if not all_rows:
        msg = "❌ 没有扫描到任何天/机位的佩戴数据"
        if skipped:
            msg += "；" + "、".join(skipped)
        return [], gr.update(choices=[], value=None), msg

    date_labels = sorted({_date_label(r) for r in all_rows})
    status = f"✅ 扫描到{len(all_rows)}条(天,机位)真实数据，共{len(date_labels)}个日期"
    if skipped:
        status += "；跳过：" + "、".join(skipped)
    return all_rows, gr.update(choices=date_labels, value=date_labels[-1]), status


def ml_preview(rows, date_label, imu, dog_name_val):
    """选好日期+机位+狗狗后，先把这一天算出来的原始统计量列出来——跟
    「导入IMU统计数据」标签的预览表格是同一个用途：让人在看模型预测结果
    之前，先确认喂给模型的这份数据本身对不对（次数/时长这些看着眼熟，
    才有必要往下看模型给的档位；这天数据本身就不对，模型给什么档位都
    没意义）。"""
    if rf_infer is None or not rows or not date_label or not imu:
        return ""
    match = next((r for r in rows if _date_label(r) == date_label and r["imu"] == imu), None)
    if not match:
        return ""

    import datetime as _dt
    import pandas as pd  # noqa: E402（只在这个函数里用一次，就近导入）
    from rf_features import compute_rf_features  # noqa: E402（skin_health/code已在sys.path里）

    breed = _dog_breed(dog_name_val)
    events, wear = rf_infer.load_events_and_wear(match["root"], imu)
    breed_map = {imu: breed or "未知"}
    target_date = _dt.date.fromisoformat(match["date"])
    feats = compute_rf_features(events, wear, breed_map)
    row = feats[(feats["pet_id"] == imu) & (feats["date"] == target_date)]
    if row.empty:
        return "（这天没有可用的佩戴数据）"
    r = row.iloc[0]

    def _fmt(v, suffix="", nd=1):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v:.{nd}f}{suffix}" if isinstance(v, float) else f"{v}{suffix}"

    return (
        f"| 字段 | 值 |\n|---|---|\n"
        f"| 佩戴时长 | {_fmt(r['valid_wear_hours'], '小时')}（{r['data_quality_flag']}） |\n"
        f"| 今日次数/时长 | {r['event_count']}次 / {_fmt(r['total_duration_min'], '分钟', 2)} |\n"
        f"| 最长单次抓挠 | {_fmt(r['max_event_duration_sec'], '秒')} |\n"
        f"| 聚集时段数 | {r['cluster_count']} |\n"
        f"| 夜间占比 | {_fmt(r['night_ratio']*100 if pd.notna(r['night_ratio']) else None, '%', 0)} |\n"
        f"| 睡眠中断次数 | {r['sleep_disruption_count']} |\n"
        f"| 历史天数（这个IMU目前累计多少天数据） | {r['history_days_available']} |\n"
        f"| 是否已建立基线（8组候选窗口任意一组） | {'是' if r['has_any_baseline'] else '否'} |\n"
        f"| 相对自身历史的z分数 | {_fmt(r['z_score_vs_self'], nd=2)} |\n"
        f"| 连续偏高天数 | {r['consecutive_days_above_baseline']} |\n"
    )


def _proba_table(proba: dict, predicted_tier: str) -> str:
    """概率分布渲染成表格，预测出来的那一档加粗标出来，不是塞进一行内联
    文字——用户反馈"最好给出结果说明置信度"，表格比一行"C0=0%、C1=74%、
    C2=26%"这种更容易一眼看出模型到底有多确定。"""
    lines = ["| 档位 | 模型给出的概率 |", "|---|---|"]
    for k, v in sorted(proba.items()):
        mark = "**" if k == predicted_tier else ""
        lines.append(f"| {mark}{k}{mark} | {mark}{v:.1%}{mark} |")
    return "\n".join(lines)


_ML_CAVEAT = (
    "\n---\n⚠️ 这两个模型目前只在合成数据上训练过，没有真实兽医标签校准过，"
    "预测结果不代表真实准确率，只能看个大概方向、跟PM版的C值计算/S总分"
    "对照看两套方案在同一天差多少，不能当成真实诊断依据。"
)


def _ml_resolve(rows, date_label, imu, dog_name_val):
    """三个ML函数(预测C/预测S/去填问答)共用的"选好了没有"校验+定位当前
    (root,imu,date,breed)——抽成一个函数，三处保持完全一致的校验逻辑和
    报错文案，不是各自重复写一遍容易改漏。返回(match, breed, error_msg)，
    error_msg非空时前两个是None，调用方直接返回error_msg。"""
    if rf_infer is None:
        return None, None, f"❌ rf_infer模块导入失败：{_RF_INFER_IMPORT_ERROR}"
    if not rows or not date_label or not imu:
        return None, None, "❌ 请先扫描根目录，并选好日期和机位"
    match = next((r for r in rows if _date_label(r) == date_label and r["imu"] == imu), None)
    if not match:
        return None, None, f"❌ 没找到 {date_label}-{imu} 对应的数据"
    breed = _dog_breed(dog_name_val)
    if not breed:
        return None, None, "❌ 请先选好「对应狗狗」（用来对应品种，机位选好后会自动带出默认值）"
    return match, breed, ""


def ml_predict_c(rows, date_label, imu, dog_name_val):
    """第①步——只跑模型A，出C档位。跟模型B分开成两个按钮/两次调用，
    不是一次性把C/S都算完：用户可能只想看C档位就够了，不一定每次都要
    去填问答、等模型B；分开之后也能让"要不要填问答"这个决定发生在看到
    C档位结果之后，符合用户描述的"先模型A看结果，再决定填不填问答"这个
    使用顺序。"""
    match, breed, err = _ml_resolve(rows, date_label, imu, dog_name_val)
    if err:
        return err
    a_avail, _ = rf_infer.models_available()
    if not a_avail:
        return ("❌ 模型A文件不存在，先在skin_health/code/下跑：\n\n"
               "```\npython train_rf_model_a.py --data_dir ../data/rf_synthetic\n```")

    import datetime as _dt
    target_date = _dt.date.fromisoformat(match["date"])
    events, wear = rf_infer.load_events_and_wear(match["root"], imu)
    model_a = rf_infer.load_model_a()

    c_res = rf_infer.predict_c(model_a, events, wear, imu, breed, target_date)
    if not c_res["available"]:
        return f"❌ 模型A无法预测：{c_res['reason']}"

    lines = [
        "## 模型A —— IMU行为严重度（C档位）",
        "*权重是从合成数据训练出来的，不是PM那套固定加权公式*\n",
        f"### 预测：**{c_res['tier']}**\n",
        _proba_table(c_res["proba"], c_res["tier"]),
        _ML_CAVEAT,
    ]
    return "\n".join(lines)


def ml_goto_questionnaire(rows, date_label, imu, dog_name_val):
    """"去「填写问答」标签"按钮——把ML标签当前选好的日期/狗狗同步进
    「填写问答」标签的填表日期/狗狗名字(不用两个标签分别选一遍)，然后
    切换到那个标签。跳转回来不需要额外处理：Gradio的Tab只是同一个页面
    里的显示/隐藏切换，不是分开的页面，「填写问答」标签里选的问答答案
    本来就是实时共享状态，切回"ML版对比"标签点"预测S档位"按钮时能直接
    读到，不用专门做"返回时自动带回结果"这一步。"""
    match = next((r for r in rows if _date_label(r) == date_label and r["imu"] == imu), None) \
        if rows and date_label and imu else None
    date_update = gr.update(value=match["date"]) if match else gr.update()
    dog_update = gr.update(value=dog_name_val) if dog_name_val else gr.update()
    # 「填写问答」标签是Tabs里定义的第一个Tab，Gradio没显式给id时按定义
    # 顺序从0开始编号，selected=0就是切过去那个标签，不用额外给每个Tab
    # 手动设id
    return date_update, dog_update, gr.Tabs(selected=0)


def ml_predict_s(rows, date_label, imu, dog_name_val,
                 has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat):
    """第③步——跑模型B，出S档位。问答是可选的：「填写问答」标签没填就把
    问答那7个特征当缺失值喂给模型，HistGradientBoostingClassifier原生
    支持缺失值，照样能出一个预测（置信度可能会因为少了问答这部分信号
    更不确定，具体信不信看下面显示的概率分布自己判断）。这跟PM版"S总分
    必须先有C值+问答"的强制要求不一样：PM那套是固定公式，公式里没数字
    就没法算；这边是训练出来的模型，缺问答不等于没法预测。

    重新跑一次模型A(不复用ml_predict_c的结果)——模型B需要模型A的预测
    概率当stacking特征，两次调用之间没有跨请求缓存，重新算一遍开销很小
    (就是读文件+算特征+两次predict_proba)，比维护一份"上次C预测结果"的
    跨按钮状态简单可靠，不会出现"C按钮跟S按钮点的不是同一份数据"这种
    因为状态没同步好导致的不一致。"""
    match, breed, err = _ml_resolve(rows, date_label, imu, dog_name_val)
    if err:
        return err
    a_avail, b_avail = rf_infer.models_available()
    if not (a_avail and b_avail):
        return ("❌ 模型文件不存在，先在skin_health/code/下跑：\n\n"
               "```\npython train_rf_model_a.py --data_dir ../data/rf_synthetic\n"
               "python gen_model_b_training_data.py --data_dir ../data/rf_synthetic\n"
               "python train_rf_model_b.py --data_dir ../data/rf_synthetic\n```")

    import datetime as _dt
    target_date = _dt.date.fromisoformat(match["date"])
    events, wear = rf_infer.load_events_and_wear(match["root"], imu)
    model_a = rf_infer.load_model_a()
    model_b = rf_infer.load_model_b()

    ordinals = _pm_answers_to_rf_ordinals(color, odor, lesion, hair_spot, hair_diameter, coat)
    s_res = rf_infer.predict_s(model_a, model_b, events, wear, imu, breed, target_date, ordinals)
    if not s_res["available"]:
        return f"## 模型B —— 综合严重度（S档位）\n⚠️ {s_res['reason']}{_ML_CAVEAT}"

    lines = [
        "## 模型B —— 综合严重度（S档位）",
        "*模型A的特征+输出概率 + 问答特征（如果有）→ S0/S1/S2*\n",
    ]
    q_note = ("（用了「填写问答」标签的答案）" if s_res["used_questionnaire"]
             else "（**没有问答数据**，只用IMU特征预测，置信度可能因此更不确定，"
                  "想更准就去「填写问答」标签填一下再回来点这个按钮）")
    lines.append(f"### 预测：**{s_res['tier']}** {q_note}\n")
    lines.append(_proba_table(s_res["proba"], s_res["tier"]))
    if s_res.get("missing_features"):
        lines.append(f"\n（另有{len(s_res['missing_features'])}个特征因历史数据不够/没填对应"
                     f"问答暂缺，模型原生支持缺失值，不影响预测能不能跑，只是这几个特征"
                     f"这次没提供信息量）")
    lines.append(_ML_CAVEAT)
    return "\n".join(lines)


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


def _row_label(row: dict) -> str:
    return f"{row['date']}-{row['imu']} [{row['root_label']}]"


def _date_label(row: dict) -> str:
    """日期下拉框用的选项文案——带上root，因为同一天可能在不同根目录
    (majority/majority_syn)下都有数据，不带root会分不清选的是哪个模型
    跑出来的结果。"""
    return f"{row['date']} [{row['root_label']}]"


def _parse_bool(value) -> bool:
    """CSV里读出来的都是字符串，"False"这种非空字符串直接bool()会误判成
    True，要按内容判断。"""
    return str(value).strip().lower() in ("true", "1", "是")


def scan_imu_roots(roots_str: str):
    """扫描输入框里逗号分隔的每个根目录，找里面每个日期目录下的
    imu_daily_scratch_stats.csv（IMU_STATS=1产出的那份，每天一份、
    落在各自的日期目录里）。只有真的存在这份文件的天才会列出来——
    没跑过统计的天不会被误当成"有数据"。

    返回(所有行的列表, 下拉选项更新, 状态提示)。目录不存在/一个统计
    文件都没有的根目录会跳过，不影响其它根目录，状态提示里写清楚跳过
    了哪些；单个日期的文件读取出错只跳过那一天，不影响同一个root下的
    其它天。"""
    roots = [r.strip() for r in (roots_str or "").split(",") if r.strip()]
    if not roots:
        return [], gr.update(choices=[], value=None), "❌ 请先填至少一个根目录"

    all_rows = []
    skipped = []
    for root in roots:
        if not os.path.isdir(root):
            skipped.append(f"{root}（目录不存在）")
            continue
        # 每天一份，在各自的日期目录下——不是root下一个总的文件
        stats_csvs = sorted(glob.glob(os.path.join(root, "*", STATS_CSV_NAME)))
        if not stats_csvs:
            skipped.append(f"{root}（没有找到任何{STATS_CSV_NAME}，可能还没跑IMU_STATS=1）")
            continue
        root_label = os.path.basename(root.rstrip("/\\")) or root
        for stats_csv in stats_csvs:
            try:
                with open(stats_csv, encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
            except Exception as e:
                skipped.append(f"{stats_csv}（读取出错：{e}）")
                continue
            for row in rows:
                row = dict(row)
                row["root"] = root
                row["root_label"] = root_label
                all_rows.append(row)

    if not all_rows:
        msg = "❌ 没有扫描到任何天/机位的数据"
        if skipped:
            msg += "；" + "、".join(skipped)
        return [], gr.update(choices=[], value=None), msg

    date_labels = sorted({_date_label(r) for r in all_rows})
    status = f"✅ 扫描到{len(all_rows)}条(天,机位)数据，共{len(date_labels)}个日期"
    if skipped:
        status += "；跳过：" + "、".join(skipped)
    return all_rows, gr.update(choices=date_labels, value=date_labels[-1]), status


def update_imu_choices(rows: list, date_label: str):
    """日期下拉框选好之后，机位下拉框只列这天实际有数据的机位——不用
    在几十个"日期-机位"组合里翻，先缩小到一天，再挑机位。"""
    imus = sorted({r["imu"] for r in rows if _date_label(r) == date_label})
    return gr.update(choices=imus, value=imus[0] if imus else None)


def default_dog_for_imu(imu: str):
    """机位选好后，「对应狗狗」自动带出IMU_DOG_DEFAULT_MAP里的默认值——
    只是省得每次手动选，不是权威对应关系，下拉框本身随时可以手动改
    （IMU1历史上对应过不同的狗，尤其要留意核对一下默认值对不对）。
    映射表里查不到就留空，不强行猜一个。"""
    return gr.update(value=IMU_DOG_DEFAULT_MAP.get(imu))


def preview_stats_row(rows: list, date_label: str, imu: str) -> str:
    if not rows or not date_label or not imu:
        return ""
    match = next((r for r in rows if _date_label(r) == date_label and r["imu"] == imu), None)
    if not match:
        return ""
    flag = match.get("data_quality_flag", "")
    wear = match.get("valid_wear_hours", "")
    # 佩戴不足的天单独警示——这种天抓挠次数天然偏低，不是狗不痒了，是没
    # 多少数据可看，直接拿来定C档位会低估
    warn = ("\n\n> ⚠️ 这天佩戴时长不足12小时（数据不完整），抓挠次数天然会偏低，"
            "不建议直接拿这天的数据定C档位"
            if flag and flag != "good" else "")

    # 没有历史基线时不显示"基线0次"这种误导性数字——0次的意思是"基线确实
    # 是0"，跟"还没有基线"完全是两回事，后者不能拿来算相对倍数
    n_base = int(float(match.get("n_baseline_days", 0) or 0))
    if n_base > 0:
        baseline_cell = (f"{match['baseline_count']}次 / "
                        f"{match['baseline_duration_min']}分钟（{n_base}天历史）")
        no_baseline_warn = ""
    else:
        baseline_cell = "— 还没有基线"
        no_baseline_warn = ("\n\n> ⚠️ 这天没有可用的历史基线（第一天，或之前的天佩戴时长"
                           "都不达标）。应用到「C值计算」时会自动取消勾选「已经有个人"
                           "基线」，「变化幅度」这一项不计分，C值上限只有70分")
    return (
        f"| 字段 | 值 |\n|---|---|\n"
        f"| 佩戴时长 | {wear}小时（{flag}） |\n"
        f"| 今日次数/时长 | {match['event_count']}次 / {match['total_duration_min']}分钟 |\n"
        f"| 最长单次抓挠 | {match['max_event_duration_sec']}秒 |\n"
        f"| 聚集时段数 | {match['cluster_count']} |\n"
        f"| 夜间事件数 | {match['night_event_count']} |\n"
        f"| 普通抓挠ZN（没打断睡眠的） | {match['zn']}次 |\n"
        f"| 打断睡眠ZD | {match['zd']}次 |\n"
        f"| 长时间抓挠 | {match['long_scratch']} |\n"
        f"| 基线次数/时长（自动估算，仅供参考） | {baseline_cell} |\n"
        f"| 持续天数（自动估算，仅供参考） | {match['persistence_days']} |\n"
        f"{warn}{no_baseline_warn}"
    )


def apply_stats_to_c_calc(rows: list, date_label: str, imu: str, dog_name_val: str):
    """把选中的(天,机位)统计行填进「C值计算」标签的各个输入框，同时把
    这个日期同步进「填写问答」标签的填表日期——用户要求"选了8-19，问答
    那边的日期也自动一起选择"，不用同一个日期在两个标签各选一遍。

    dog_name_val是「导入IMU统计数据」标签里「对应狗狗」下拉框当前选的值
    （默认来自IMU_DOG_DEFAULT_MAP，但用户可能已经手动改过，尤其IMU1），
    一起同步进「填写问答」标签的狗狗名字，不用再手动选一遍。"""
    if not rows or not date_label or not imu:
        return (None, None, None, None, None, None, None, None, False, True,
                gr.update(), gr.update(), "❌ 请先扫描根目录，并选好日期和机位")
    match = next((r for r in rows if _date_label(r) == date_label and r["imu"] == imu), None)
    if not match:
        return (None, None, None, None, None, None, None, None, False, True,
                gr.update(), gr.update(), f"❌ 没找到 {date_label}-{imu} 对应的数据")

    # n_baseline_days==0 表示这天没有可用的历史基线（第一天，或者之前的天
    # 佩戴时长都不达标被排除掉了）——自动帮用户取消「已经有个人基线」的
    # 勾选，避免把统计脚本填的baseline_count=0当成"基线真的是0次"去算，
    # 那样会凭空造出一个巨大的相对倍数和假红旗
    n_baseline_days = int(float(match.get("n_baseline_days", 0) or 0))
    has_baseline_val = n_baseline_days > 0

    iso_date = _to_iso_date(match["date"])
    dog_note = f"，狗狗名字已同步为「{dog_name_val}」" if dog_name_val else "（未选「对应狗狗」，狗狗名字没有同步，麻烦去「填写问答」标签手动选一下）"
    status = (f"✅ 已把 {match['date']}（{match['imu']}）的统计数据填进「C值计算」标签，"
             f"「填写问答」标签的填表日期已同步为 {iso_date}{dog_note}")
    if not has_baseline_val:
        status += ("\n\n⚠️ 这天没有可用的历史基线（第一天，或之前的天佩戴时长都不达标），"
                  "「已经有个人基线」已自动取消勾选，「变化幅度」这一项不计分——"
                  "C值上限只有70分，会偏低，别直接拿去跟有基线的天比")
    if match.get("data_quality_flag") and match["data_quality_flag"] != "good":
        status += (f"\n\n⚠️ 注意：这天佩戴时长只有{match.get('valid_wear_hours', '?')}小时"
                  f"（不足12小时），抓挠次数天然会偏低，算出来的C值会低估，"
                  f"不建议直接拿这天定档")
    dog_update = gr.update(value=dog_name_val) if dog_name_val else gr.update()
    return (
        float(match["baseline_count"]), float(match["baseline_duration_min"]),
        float(match["event_count"]), float(match["total_duration_min"]),
        float(match["cluster_count"]), float(match["persistence_days"]),
        float(match["zn"]), float(match["zd"]), _parse_bool(match["long_scratch"]),
        has_baseline_val,
        gr.update(value=iso_date), dog_update, status,
    )


def delete_record(row_idx):
    """删除选中的那一行。row_idx是None（还没在表格里点选任何行）时
    什么都不做，直接返回当前记录，避免误触发删除第0行。"""
    rows = load_records()
    if row_idx is None or not (0 <= row_idx < len(rows)):
        return rows
    del rows[row_idx]
    _write_all(rows)
    return rows


def build_app():
    with gr.Blocks(title="狗狗皮肤问答（PM原版打分）") as demo:
        with gr.Tabs() as main_tabs:
            with gr.Tab("填写问答"):
                gr.Markdown(
                    "# 狗狗皮肤问答表\n"
                    "填表说明：设备检测到抓挠水平比平时高，麻烦花1-2分钟观察并填写下面的问题，"
                    "帮助更准确判断是否需要就医。"
                )

                with gr.Row():
                    # interactive显式写True——这几个组件没被当作任何事件回调的
                    # input参数用，Gradio会按"纯展示用途"自动判成不可交互(灰掉/
                    # 选不了)，不显式声明的话下拉框会变成只读的，之前"选不了"
                    # 就是这个问题
                    dog_name = gr.Dropdown(
                        DOG_NAME_OPTIONS,
                        label="狗狗名字", interactive=True,
                    )
                    fill_date = gr.DateTime(label="填表日期", include_time=False, type="string",
                                            interactive=True)
                    filler = gr.Dropdown(["周蕾", "刘雪飞"], label="填写人", interactive=True)

                gr.Markdown("## 一、前置问题")
                has_hair_loss = gr.Radio(
                    ["是", "否"],
                    label="1. 您家宠物身上是否有毛发稀疏或出现没有毛的情况？",
                )

                gr.Markdown("## 二、皮肤色泽评估")
                color = gr.Radio(
                    [t for t, _ in SKIN_COLOR_OPTIONS],
                    label="2. 您拨开宠物毛发看皮肤时，皮肤颜色是什么样的？",
                )

                gr.Markdown("## 三、体味评估")
                odor = gr.Radio(
                    [t for t, _ in ODOR_OPTIONS],
                    label="3. 宠物身上有没有明显异味？",
                )

                gr.Markdown("## 四、皮肤损伤评估")
                lesion = gr.Radio(
                    [t for t, _ in LESION_OPTIONS],
                    label="4. 宠物的皮肤是否完整？",
                )

                gr.Markdown("## 五、毛发质量评估")
                # 默认隐藏(visible=False)——前置问题选"否"时这两题根本不适用，
                # 不该让用户看到一道"不用回答"的题还留在页面上；选"是"时才显示
                hair_spot = gr.Radio(
                    [t for t, _ in HAIR_SPOT_OPTIONS],
                    label="5. 宠物身上没有毛或毛发稀疏的地方是如何分布的？",
                    visible=False,
                )
                hair_diameter = gr.Radio(
                    [t for t, _ in HAIR_DIAMETER_OPTIONS],
                    label="6. 最大的一块秃毛区域大概有多大？",
                    visible=False,
                )
                coat = gr.Radio(
                    [t for t, _ in COAT_QUALITY_OPTIONS],
                    label="7. 宠物整体毛发状态看起来怎么样？",
                )

                gr.Markdown("## 结果")
                with gr.Row():
                    total_score = gr.Number(label="问答分数", precision=1)
                breakdown_md = gr.Markdown()

                calc_btn = gr.Button("计算分数", variant="primary")
                inputs = [has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat]
                calc_btn.click(fn=compute_score, inputs=inputs, outputs=[total_score, breakdown_md])
                # 选项一变就自动重新算一次，不用每次都手动点按钮
                for inp in inputs:
                    inp.change(fn=compute_score, inputs=inputs, outputs=[total_score, breakdown_md])

                def toggle_hair_questions(choice):
                    """前置问题选"是"才显示5/6两题，选"否"（或还没选）直接隐藏——
                    不是显示出来但标"不用填"，是真的不出现在页面上。切回"否"时
                    顺便清空已选的答案，避免用户先选了"是"填了答案、又改成"否"，
                    答案还留在(已隐藏的)组件里造成困惑。"""
                    show = choice == "是"
                    return gr.update(visible=show, value=None), gr.update(visible=show, value=None)

                has_hair_loss.change(
                    fn=toggle_hair_questions, inputs=[has_hair_loss], outputs=[hair_spot, hair_diameter],
                )

                gr.Markdown("## 保存")
                confirm_overwrite = gr.Checkbox(
                    label="确认覆盖已有的同名记录",
                    info="狗狗名字+填表日期+填写人都相同的记录已存在时，默认不会保存，"
                         "勾选这个再点保存才会真的覆盖旧记录",
                )
                save_btn = gr.Button("保存记录")
                save_status = gr.Markdown()

            with gr.Tab("导入IMU统计数据"):
                gr.Markdown(
                    "# 导入IMU统计数据\n"
                    "填好推理结果根目录（逗号分隔可以填多个），点「扫描」——会去每个根"
                    "目录下的**每个日期目录**里找`imu_daily_scratch_stats.csv`"
                    "（`run_review_bins_all_days.sh`加`IMU_STATS=1`那次批处理跑完之后"
                    "产出的，每天一份、存在各自的日期目录下，所以跑多天不会互相覆盖；"
                    "也可以单独跑`src/imu_scratch_daily_stats.py`）。只有真的存在这份"
                    "文件的天才会列进「日期」下拉框——没跑过统计的天不会被误当成「有数据」。"
                    "先选日期，「机位」下拉框会自动只列这天实际有数据的机位，机位选好后"
                    "「对应狗狗」会带出一个默认值（IMU1→比熊-BB、IMU2→金毛-巴利、"
                    "IMU3→中华田园犬-露露、IMU4→马尔济斯-小满），**这只是默认值，"
                    "不是写死的规则**——尤其IMU1，历史数据里对应的狗不一定一直是"
                    "同一只（早期可能是金毛-巴利），不确定的话务必手动核对/改一下"
                    "再点「应用」。选好之后点「应用」，「C值计算」标签的统计数字、"
                    "「填写问答」标签的填表日期和狗狗名字会一起自动填上，不用自己"
                    "一个个数字去数、也不用来回切标签选两遍。\n\n"
                    "⚠️ 基线次数/时长、持续天数这两组是自动估算的参考值（用同一个机位"
                    "别的日子的数据粗略估的，不是PM文档要求的严格21天基线），选好之后先去"
                    "「C值计算」标签核对/调整这两组数字，再看最终C值。"
                )
                with gr.Row():
                    imu_stats_roots = gr.Textbox(
                        label="推理结果根目录（逗号分隔可填多个）",
                        value=DEFAULT_IMU_STATS_ROOTS,
                        placeholder="比如 infer_result_majority_syn, infer_result_majority",
                    )
                    scan_stats_btn = gr.Button("扫描")
                scan_status_md = gr.Markdown()
                stats_rows_state = gr.State(value=[])

                with gr.Row():
                    stats_date_select = gr.Dropdown(label="日期", interactive=True)
                    stats_imu_select = gr.Dropdown(label="机位（IMU）", interactive=True)
                    stats_dog_select = gr.Dropdown(
                        DOG_NAME_OPTIONS, label="对应狗狗", interactive=True,
                        info="自动带出默认值，不确定就手动核对/改一下（尤其IMU1）",
                    )
                stats_preview_md = gr.Markdown()

                apply_stats_btn = gr.Button("应用到「C值计算」标签", variant="primary")
                apply_status_md = gr.Markdown()

                scan_stats_btn.click(
                    fn=scan_imu_roots, inputs=[imu_stats_roots],
                    outputs=[stats_rows_state, stats_date_select, scan_status_md],
                )
                # 选好日期之后，机位下拉框只列这天实际有数据的机位——先按天
                # 缩小范围，不用在"日期-机位"的长列表里翻几十条
                stats_date_select.change(
                    fn=update_imu_choices, inputs=[stats_rows_state, stats_date_select],
                    outputs=[stats_imu_select],
                )
                for trig in (stats_date_select, stats_imu_select):
                    trig.change(
                        fn=preview_stats_row,
                        inputs=[stats_rows_state, stats_date_select, stats_imu_select],
                        outputs=[stats_preview_md],
                    )
                # 机位选好后，「对应狗狗」自动带出默认映射——只是默认值，用户
                # 选好之后仍然可以自己改（尤其IMU1历史上对应过不同的狗）
                stats_imu_select.change(
                    fn=default_dog_for_imu, inputs=[stats_imu_select], outputs=[stats_dog_select],
                )

            with gr.Tab("C值计算"):
                gr.Markdown(
                    "# IMU行为严重度（C值）计算器\n"
                    "按PM文档的规则(变化幅度/聚集程度/持续程度/中断影响四项)手动"
                    "算C值，跟`skin_health/code/scratch_burden.py`的SBS引擎逐条"
                    "核对过、规则完全一致（见`pm_rules_verification.md`）。这里用"
                    "的是结构化统计量输入，不需要接原始抓挠事件时间戳。"
                )

                gr.Markdown(
                    "## 一、变化幅度（0-30分）\n"
                    "这一项是拿今天跟这只狗**自己的历史基线**比，所以第一天/历史"
                    "数据不够的时候没法算——这种情况把下面的「已经有个人基线」"
                    "取消勾选，这一项就不计分、也不会触发红旗。\n\n"
                    "⚠️ 不要把「没有基线」当成「基线是0次」填进去：那样今天抓26次"
                    "会被算成涨了8.67倍，凭空造出一个>3倍的红旗直接判C2，其实根本"
                    "没有可比的历史数据。"
                )
                has_baseline = gr.Checkbox(
                    label="已经有个人基线（取消勾选=还没有基线，变化幅度这一项不计分）",
                    value=True,
                )
                with gr.Row():
                    baseline_count = gr.Number(label="基线次数（次/24h）", minimum=0)
                    baseline_duration_min = gr.Number(label="基线总时长（分钟/24h）", minimum=0)
                with gr.Row():
                    today_count = gr.Number(label="今日次数", minimum=0)
                    today_duration_min = gr.Number(label="今日总时长（分钟）", minimum=0)

                gr.Markdown("## 二、聚集程度（0-20分）")
                cluster_count = gr.Number(
                    label="今日聚集时段数",
                    info="1小时内抓挠事件数≥5次记为1个聚集时段", minimum=0,
                )

                gr.Markdown("## 三、持续程度（0-20分）")
                persistence_days = gr.Number(
                    label="连续天数",
                    info="过去7天内，变化幅度分连续≥10分的天数（0/1/2/3+）", minimum=0,
                )

                gr.Markdown(
                    "## 四、正常行为影响/中断（0-30分）\n"
                    "这一项看的是「抓挠有没有影响到狗的正常作息」，不只是抓了多少次。\n\n"
                    "打分规则（从重到轻依次判，命中就不看后面的，2026-08跟兽医"
                    "同事重新讨论后的版本）：\n"
                    "- `ZD≥3` **或** 有单次抓挠≥60秒 → **30分（红旗，直接判C2，"
                    "满足一项即可，不用同时满足）**\n"
                    "- `ZD<3`（即ZD是1或2） → 20分\n"
                    "- `ZN>5` → 10分\n"
                    "- 都不满足 → 0分\n\n"
                    "⚠️ 注意ZD优先级高于ZN：只要ZD≥1，不管ZN是5还是50都不影响得分。"
                )
                with gr.Row():
                    zn = gr.Number(
                        label="ZN：普通抓挠次数（没打断睡眠的）", minimum=0,
                        info="当天总抓挠次数 减去 ZD。只在ZD==0时才影响得分（>5给10分）",
                    )
                    zd = gr.Number(
                        label="ZD：打断睡眠的抓挠次数", minimum=0,
                        info="夜间(22:00-06:00)抓挠打断睡眠的次数；相邻两次间隔<5分钟"
                             "算同一次没重新睡着，合并成1次",
                    )
                long_scratch = gr.Checkbox(label="今日是否出现长时间抓挠事件（连续/高聚集抓挠达到60秒）")

                gr.Markdown("## 结果")
                with gr.Row():
                    c_score_output = gr.Number(label="C值", precision=0)
                    c_tier_output = gr.Textbox(label="C档位", interactive=False)
                c_breakdown_md = gr.Markdown()

                c_calc_btn = gr.Button("计算C值", variant="primary")
                c_inputs = [baseline_count, baseline_duration_min, today_count, today_duration_min,
                           cluster_count, persistence_days, zn, zd, long_scratch, has_baseline]
                c_outputs = [c_score_output, c_tier_output, c_breakdown_md]
                c_calc_btn.click(fn=compute_c_score, inputs=c_inputs, outputs=c_outputs)
                for inp in c_inputs:
                    inp.change(fn=compute_c_score, inputs=c_inputs, outputs=c_outputs)

                # 「导入IMU统计数据」标签选好(天,机位)后点"应用"，直接把这几个
                # 输入框填上，同时把「填写问答」标签的填表日期也同步过去。
                # has_baseline也一起带过来——统计CSV里的n_baseline_days==0就
                # 表示这天没有可用的历史基线，自动帮用户取消勾选，不用他自己
                # 去判断"这天到底有没有基线"
                apply_stats_btn.click(
                    fn=apply_stats_to_c_calc,
                    inputs=[stats_rows_state, stats_date_select, stats_imu_select, stats_dog_select],
                    outputs=[baseline_count, baseline_duration_min, today_count, today_duration_min,
                            cluster_count, persistence_days, zn, zd, long_scratch, has_baseline,
                            fill_date, dog_name, apply_status_md],
                )

            with gr.Tab("S总分"):
                gr.Markdown(
                    "# S总分（综合严重度）\n"
                    "S = C值×40% + 皮肤状态组×35% + 毛发状态组×25%。\n"
                    "皮肤/毛发两组的答案直接读取「填写问答」标签里已经选好的选项，"
                    "不用重新填一遍；C值可以手动填在下面，也可以先去「C值计算」"
                    "标签算好，这里会自动同步过来。"
                )
                s_c_value = gr.Number(
                    label="C值（0-100）",
                    info="留空按0算；也可以直接去「C值计算」标签算，这里会自动同步",
                )
                # 不显示在页面上——跟着s_c_value一起从"C值计算"标签同步过来的
                # 真实档位(含红旗判定)，S总分要用这个来判"C是不是C2"，不能拿
                # c_value自己按30/50阈值重新粗算(那样会漏掉红旗)。用户没去
                # "C值计算"标签算过、直接手动填C值时这个是空的，compute_s_total
                # 会退回简单阈值判断当兜底
                s_c_tier_hint = gr.Textbox(visible=False)
                gr.Markdown("## 结果")
                with gr.Row():
                    s_total_output = gr.Number(label="S总分", precision=2)
                    s_c_tier_output = gr.Textbox(label="C档位", interactive=False)
                    s_tier_output = gr.Textbox(label="S档位", interactive=False)
                s_breakdown_md = gr.Markdown()

                s_calc_btn = gr.Button("计算S总分", variant="primary")
                s_inputs = [s_c_value, s_c_tier_hint, has_hair_loss, color, odor, lesion,
                           hair_spot, hair_diameter, coat]
                s_outputs = [s_total_output, s_c_tier_output, s_tier_output, s_breakdown_md]
                s_calc_btn.click(fn=compute_s_total, inputs=s_inputs, outputs=s_outputs)
                for inp in s_inputs:
                    inp.change(fn=compute_s_total, inputs=s_inputs, outputs=s_outputs)

                # C值计算标签算完之后，自动把C值和真实档位(含红旗判定)一起
                # 同步进这个标签——不用手动抄数字过来，两个标签的C值和档位
                # 保持一致，不会出现S总分这边自己按阈值重新粗算出跟"C值计算"
                # 标签不一样的档位
                c_score_output.change(
                    fn=lambda v, t: (v, t), inputs=[c_score_output, c_tier_output],
                    outputs=[s_c_value, s_c_tier_hint],
                )

            with gr.Tab("ML版对比"):
                gr.Markdown(
                    "# ML版（算法组训练出来的RF模型）\n"
                    "`skin_health/`目录是算法组自己设计的方案——不用PM的固定加权公式，"
                    "权重是从数据训练出来的（模型A：IMU行为特征→C0/C1/C2；模型B：模型A"
                    "的特征+输出概率+问答特征(可选)→S0/S1/S2）。这个标签用**同一份"
                    "真实IMU数据**跑一遍模型A/B，可以跟前面「C值计算」「S总分」两个"
                    "标签的PM版结果对照，看两套方案在同一天差多少。\n\n"
                    "⚠️ **重要**：这两个模型目前只在`skin_health/data/rf_synthetic/`下的"
                    "**合成数据**上训练过，没有真实兽医标签校准过，预测结果**不代表"
                    "真实准确率**——训练脚本自己的报告也写得很清楚这一点。这里只是"
                    "把训练管道接上真实数据跑一下、看个大概方向，不能当诊断依据。\n\n"
                    "模型文件不存在时需要先在服务器上跑（只用跑一次，跑完这两个"
                    "`.joblib`文件就在仓库里，不用每次都重新训练）：\n"
                    "```\ncd skin_health/code\n"
                    "python train_rf_model_a.py --data_dir ../data/rf_synthetic\n"
                    "python gen_model_b_training_data.py --data_dir ../data/rf_synthetic\n"
                    "python train_rf_model_b.py --data_dir ../data/rf_synthetic\n```"
                )
                with gr.Row():
                    ml_roots = gr.Textbox(
                        label="推理结果根目录（逗号分隔可填多个）",
                        value=DEFAULT_IMU_STATS_ROOTS,
                        placeholder="比如 infer_result_majority_syn, infer_result_majority",
                    )
                    ml_scan_btn = gr.Button("扫描")
                ml_scan_status_md = gr.Markdown()
                ml_rows_state = gr.State(value=[])

                with gr.Row():
                    ml_date_select = gr.Dropdown(label="日期", interactive=True)
                    ml_imu_select = gr.Dropdown(label="机位（IMU）", interactive=True)
                    ml_dog_select = gr.Dropdown(
                        DOG_NAME_OPTIONS, label="对应狗狗", interactive=True,
                        info="自动带出默认值，不确定就手动核对/改一下（尤其IMU1）",
                    )
                ml_preview_md = gr.Markdown()

                gr.Markdown("### ① 先用模型A预测C档位")
                ml_predict_c_btn = gr.Button("预测C档位（模型A）", variant="primary")
                ml_c_result_md = gr.Markdown()

                gr.Markdown(
                    "### ② 问答（可选）\n"
                    "看完C档位，想让S档位预测更准的话，可以去「填写问答」标签填一下——"
                    "点下面这个按钮会自动把当前选的日期/狗狗名字带过去，不用重新选。"
                    "不想填也可以直接跳到第③步，模型B没有问答照样能预测。"
                )
                ml_goto_q_btn = gr.Button("去「填写问答」标签填问卷（可选）")

                gr.Markdown("### ③ 用模型B预测S档位（问答填不填都能跑）")
                ml_predict_s_btn = gr.Button("预测S档位（模型B）", variant="primary")
                ml_s_result_md = gr.Markdown()

                ml_scan_btn.click(
                    fn=ml_scan_roots, inputs=[ml_roots],
                    outputs=[ml_rows_state, ml_date_select, ml_scan_status_md],
                )
                # 跟「导入IMU统计数据」标签同一套交互：先选日期（下拉框只列
                # 真的有数据的天），机位下拉框自动收窄到这天实际有数据的
                # 那几个IMU，机位选好后「对应狗狗」自动带出默认值——复用
                # update_imu_choices()/default_dog_for_imu()这两个已有函数，
                # 不是照着UI外观重新写一遍逻辑
                ml_date_select.change(
                    fn=update_imu_choices, inputs=[ml_rows_state, ml_date_select],
                    outputs=[ml_imu_select],
                )
                ml_imu_select.change(
                    fn=default_dog_for_imu, inputs=[ml_imu_select], outputs=[ml_dog_select],
                )
                for trig in (ml_date_select, ml_imu_select, ml_dog_select):
                    trig.change(
                        fn=ml_preview, inputs=[ml_rows_state, ml_date_select, ml_imu_select, ml_dog_select],
                        outputs=[ml_preview_md],
                    )
                ml_predict_c_btn.click(
                    fn=ml_predict_c,
                    inputs=[ml_rows_state, ml_date_select, ml_imu_select, ml_dog_select],
                    outputs=[ml_c_result_md],
                )
                ml_goto_q_btn.click(
                    fn=ml_goto_questionnaire,
                    inputs=[ml_rows_state, ml_date_select, ml_imu_select, ml_dog_select],
                    outputs=[fill_date, dog_name, main_tabs],
                )
                ml_predict_s_btn.click(
                    fn=ml_predict_s,
                    inputs=[ml_rows_state, ml_date_select, ml_imu_select, ml_dog_select,
                           has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat],
                    outputs=[ml_s_result_md],
                )

            with gr.Tab("历史记录"):
                gr.Markdown("## 历史记录")
                with gr.Row():
                    refresh_btn = gr.Button("刷新")
                    delete_btn = gr.Button("删除选中行")
                # value传函数本身(不是load_records()调用后的结果)——数据本来就
                # 已经实时存在records.csv里，服务重启/刷新页面都不会丢；之前
                # "刷新页面历史记录就没了"的问题出在这里：value=load_records()
                # 只在demo启动那一刻调用了一次，之后每个新打开的页面/每次刷新
                # 都复用那个启动时刻的旧快照，不会重新读文件。传函数引用，
                # Gradio会在每次有人打开/刷新页面时重新调用一次，读到最新内容
                history_table = gr.Dataframe(
                    headers=RECORD_COLUMNS, value=load_records, interactive=False, wrap=True,
                )
                export_btn = gr.DownloadButton("导出为CSV")
                selected_row_idx = gr.State(value=None)

                refresh_btn.click(fn=load_records, outputs=[history_table])
                export_btn.click(fn=export_records_csv, outputs=[export_btn])
                def _on_select_row(evt: gr.SelectData):
                    return evt.index[0]

                history_table.select(fn=_on_select_row, outputs=[selected_row_idx])
                delete_btn.click(
                    fn=delete_record, inputs=[selected_row_idx], outputs=[history_table],
                )

        save_inputs = [dog_name, fill_date, filler, has_hair_loss, color, odor, lesion,
                       hair_spot, hair_diameter, coat, total_score, confirm_overwrite]
        save_btn.click(
            fn=save_record, inputs=save_inputs,
            outputs=[save_status, history_table, confirm_overwrite],
        )

    return demo


def main():
    port = 7860
    lan_ip = get_lan_ip()
    print(f"局域网访问地址: http://{lan_ip}:{port}  (把这个链接发给同一局域网内的其他设备)")
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
