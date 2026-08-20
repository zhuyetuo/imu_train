"""
PM版皮肤问答打分页面——一比一复刻PM原始Wardyn题库的固定加权公式，不用
机器学习，每个选项对应她CSV里写死的分值，选完直接加总出"问答分数"。

跟skin_health/那套RF方案是两个完全独立的东西，不要混——那边是算法组
自己设计的、权重靠训练学出来的方案；这个目录专门给PM的原始方案一个
独立的、可以直接用起来的页面，方便对照/过渡期两边同时用。

分值表来源：PM提供的Wardyn题库CSV（PFSY01-06），字段名/选项文案跟
questionnaire_paper_form.md保持一致（那份是纯离线纸质表，这个是同一套
题目的在线版，方便自动算分）。

用法：
    python pm_skin_scoring/code/questionnaire_app.py
    浏览器打开 http://localhost:7860
"""
import csv
import math
import os
import socket
from datetime import datetime

import gradio as gr


def _round_half_up(value: float, ndigits: int) -> float:
    """标准四舍五入，不用Python内置round()——它是banker's rounding
    (round-half-to-even)，round(74.25, 1)算出来是74.2不是74.25，之前
    在skin_health/合成数据标签生成那边就踩过同一个坑(见
    rf_synthetic_validation_findings.md)，这里用floor(x+0.5)方式避开。"""
    factor = 10 ** ndigits
    return math.floor(value * factor + 0.5) / factor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RECORDS_CSV = os.path.join(DATA_DIR, "records.csv")
RECORD_COLUMNS = [
    "狗狗名字", "填表日期", "填写人", "C值", "C档位",
    "有无毛发稀疏", "皮肤颜色", "体味", "皮损", "秃毛分布", "秃毛面积", "整体毛质",
    "问答分数", "S总分", "S档位", "保存时间",
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


# PM完整总分公式（见pm_skin_scoring/docs/pm_rules_verification.md）：
#   S = C值(0-100) × 40%  +  皮肤状态组原始分小计 × 35%  +  毛发状态组原始分小计 × 25%
# 皮肤状态组=皮肤颜色+体味+皮损，毛发状态组=秃毛部位+秃毛面积+整体毛质，
# 组内原始分直接相加、不归一化，再乘以组权重。C值来自IMU行为严重度评分
# （skin_health/code/scratch_burden.py的SBS引擎，已逐条核对跟PM这份文档
# 的变化幅度/聚集程度/持续程度/中断影响四项评分规则完全一致），这个
# 问答页面本身不计算C值，需要手动填入（比如从Wardyn后台/SBS跑批结果里
# 抄一个过来）。
C_WEIGHT = 0.40
SKIN_GROUP_WEIGHT = 0.35
HAIR_GROUP_WEIGHT = 0.25


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


def s_tier_of(s_value):
    """⚠️ PM原文档里这三档的边界写的是"0≤S<12+"/"12+≤S<20+"/"20+≤S≤70"，
    "+"号具体含义不确定（可能是待定/近似值，不是精确数字），而且"≤70"
    这个上限跟她自己举的满分示例(74.25分)对不上，大概率是文档还没最终
    定稿。这里按字面数字(12/20)取整数边界实现，S2不设上限(实际会被公式
    本身的最大值74.25自然限制住)——这个分档结果目前只能当参考，正式
    上线前必须找PM确认这三个边界的准确数值。"""
    if s_value is None:
        return ""
    if s_value >= 20:
        return "S2"
    if s_value >= 12:
        return "S1"
    return "S0"


def compute_score(c_value, has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat):
    """前置问题选"否"时，秃毛分布/秃毛面积这两题不需要回答，按0分计（不是
    "扣分"，是这两题在这只狗身上根本不适用，PM原表里这种情况也是记0分，
    不是留空导致的缺失）。整体毛质题不受前置问题限制，始终参与计分。"""
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
    skin_group_score = skin_group_raw * SKIN_GROUP_WEIGHT
    hair_group_score = hair_group_raw * HAIR_GROUP_WEIGHT
    questionnaire_total = _round_half_up(skin_group_score + hair_group_score, 2)

    c = c_value if c_value is not None else 0
    c_score = c * C_WEIGHT
    # 保留2位小数，不是1位——PM文档举的满分例子是精确到分的"74.25"，
    # 权重0.35/0.25乘上整数原始分本来就可能产生2位小数，四舍五入到1位
    # 会丢精度、跟她原文的数字对不上
    total = _round_half_up(c_score + skin_group_score + hair_group_score, 2)

    c_tier = c_tier_of(c_value)
    s_tier = s_tier_of(total)
    c_line = (f"| **C值({c} × {C_WEIGHT:.0%}) [{c_tier}]** | **{c_score:.1f}** |\n"
             if c_value is not None else
             "| ⚠️ 还没填C值，下面只按问答部分算(未乘40%的C值项) | — |\n")

    breakdown = (
        f"| 题目 | 原始分 |\n|---|---|\n"
        f"{c_line}"
        f"| 皮肤颜色 | {s_color} |\n"
        f"| 体味 | {s_odor} |\n"
        f"| 皮损 | {s_lesion} |\n"
        f"| **皮肤状态组小计 × {SKIN_GROUP_WEIGHT:.0%}** | **{skin_group_raw} × {SKIN_GROUP_WEIGHT:.0%} = {skin_group_score:.2f}** |\n"
        f"| 秃毛分布 | {s_spot}{'（未评估，计0分）' if has_hair_loss != '是' else ''} |\n"
        f"| 秃毛面积 | {s_diameter}{'（未评估，计0分）' if has_hair_loss != '是' else ''} |\n"
        f"| 整体毛质 | {s_coat} |\n"
        f"| **毛发状态组小计 × {HAIR_GROUP_WEIGHT:.0%}** | **{hair_group_raw} × {HAIR_GROUP_WEIGHT:.0%} = {hair_group_score:.2f}** |\n"
        f"| 问答部分小计(皮肤组+毛发组，不含C值) | {questionnaire_total} |\n"
        f"| **S总分 = C×40%+皮肤组×35%+毛发组×25%** | **{total}**{f'　→　**{s_tier}**' if s_tier else ''} |\n"
        f"\n> ⚠️ S档位边界(S0/S1/S2)数值来自PM文档但标注了"
        f"「+」号、含义不确定，正式上线前需要跟PM确认准确边界，目前仅供参考。"
    )
    return questionnaire_total, total, c_tier, s_tier, breakdown


# ── 历史记录：本地CSV持久化 ────────────────────────────────────────────────

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


def save_record(dog_name, fill_date, filler, c_value, has_hair_loss, color, odor, lesion,
                hair_spot, hair_diameter, coat, questionnaire_score, s_total_score,
                c_tier, s_tier, confirm_overwrite):
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

    new_row = [dog_name, fill_date_str, filler,
               c_value if c_value is not None else "", c_tier or "",
               has_hair_loss or "", _letter_of(color), _letter_of(odor), _letter_of(lesion),
               _letter_of(hair_spot), _letter_of(hair_diameter), _letter_of(coat),
               questionnaire_score, s_total_score, s_tier or "",
               datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

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
        with gr.Tabs():
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
                        ["比熊-BB", "金毛-巴利", "中华田园犬-露露", "马尔济斯-小满"],
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

                gr.Markdown(
                    "## 六、IMU行为严重度（C值）\n"
                    "这个页面只负责问答部分，C值来自IMU数据分析结果（Wardyn/SBS引擎），"
                    "需要手动填入——不填也能看问答部分的分数，只是S总分会先按C=0算。"
                )
                c_value = gr.Number(label="C值（0-100）", minimum=0, maximum=100)

                gr.Markdown("## 结果")
                with gr.Row():
                    questionnaire_score = gr.Number(label="问答分数（皮肤组+毛发组加权分，不含C值）", precision=2)
                    s_total_score = gr.Number(label="S总分（含C值×40%）", precision=2)
                with gr.Row():
                    c_tier_box = gr.Textbox(label="C档位", interactive=False)
                    s_tier_box = gr.Textbox(label="S档位（⚠️边界待PM确认，仅供参考）", interactive=False)
                breakdown_md = gr.Markdown()

                calc_btn = gr.Button("计算分数", variant="primary")
                inputs = [c_value, has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat]
                calc_outputs = [questionnaire_score, s_total_score, c_tier_box, s_tier_box, breakdown_md]
                calc_btn.click(fn=compute_score, inputs=inputs, outputs=calc_outputs)
                # 选项一变就自动重新算一次，不用每次都手动点按钮
                for inp in inputs:
                    inp.change(fn=compute_score, inputs=inputs, outputs=calc_outputs)

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

        save_inputs = [dog_name, fill_date, filler, c_value, has_hair_loss, color, odor, lesion,
                       hair_spot, hair_diameter, coat, questionnaire_score, s_total_score,
                       c_tier_box, s_tier_box, confirm_overwrite]
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
