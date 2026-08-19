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
import gradio as gr

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


def compute_score(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat):
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

    total = s_color + s_odor + s_lesion + s_spot + s_diameter + s_coat

    breakdown = (
        f"| 题目 | 得分 |\n|---|---|\n"
        f"| 皮肤颜色 | {s_color} |\n"
        f"| 体味 | {s_odor} |\n"
        f"| 皮损 | {s_lesion} |\n"
        f"| 秃毛分布 | {s_spot}{'（未评估，计0分）' if has_hair_loss != '是' else ''} |\n"
        f"| 秃毛面积 | {s_diameter}{'（未评估，计0分）' if has_hair_loss != '是' else ''} |\n"
        f"| 整体毛质 | {s_coat} |\n"
        f"| **问答分数合计** | **{total}** |\n"
    )
    return total, breakdown


def build_app():
    with gr.Blocks(title="狗狗皮肤问答（PM原版打分）") as demo:
        gr.Markdown(
            "# 狗狗皮肤问答表（PM原版一比一复刻）\n"
            "填表说明：设备检测到抓挠水平比平时高，麻烦花1-2分钟观察并填写下面的问题，"
            "帮助更准确判断是否需要就医。选项分值来自PM原始Wardyn题库，选完自动算出"
            "「问答分数」，可以直接填进飞书表格对应的列。\n\n"
            "> 这是PM原方案的固定加权公式，跟`skin_health/`目录下算法组自己设计的RF方案"
            "是两套独立的东西，不要混淆。"
        )

        with gr.Row():
            dog_name = gr.Textbox(label="狗狗名字")
            fill_date = gr.Textbox(label="填表日期", placeholder="比如 2026-08-19")

        gr.Markdown("## 一、前置问题")
        has_hair_loss = gr.Radio(
            ["是", "否"],
            label="1. 您家宠物身上是否有毛发稀疏或出现没有毛的情况？",
            info="选「是」才需要继续填写下面的秃毛分布/秃毛面积两题；选「否」这两题按0分计",
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
        hair_spot = gr.Radio(
            [t for t, _ in HAIR_SPOT_OPTIONS],
            label="5. 宠物身上没有毛或毛发稀疏的地方是如何分布的？（前置问题选「是」才需要填）",
        )
        hair_diameter = gr.Radio(
            [t for t, _ in HAIR_DIAMETER_OPTIONS],
            label="6. 最大的一块秃毛区域大概有多大？（前置问题选「是」才需要填）",
        )
        coat = gr.Radio(
            [t for t, _ in COAT_QUALITY_OPTIONS],
            label="7. 宠物整体毛发状态看起来怎么样？（这题始终要填）",
        )

        gr.Markdown("## 结果")
        with gr.Row():
            total_score = gr.Number(label="问答分数（0819飞书表格「问答分数」列）", precision=0)
        breakdown_md = gr.Markdown()

        calc_btn = gr.Button("计算分数", variant="primary")
        inputs = [has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat]
        calc_btn.click(fn=compute_score, inputs=inputs, outputs=[total_score, breakdown_md])
        # 选项一变就自动重新算一次，不用每次都手动点按钮——飞书表格截图里
        # 提到的诉求是"自动计算分数"，按钮留着方便手动强制刷新，两种方式都支持
        for inp in inputs:
            inp.change(fn=compute_score, inputs=inputs, outputs=[total_score, breakdown_md])

    return demo


def main():
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
