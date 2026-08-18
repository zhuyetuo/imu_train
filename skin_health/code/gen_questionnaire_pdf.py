"""
生成纸质版皮肤问答表（PDF），内容对应 docs/questionnaire_page_design.md。
纯离线表格，不是网页——兽医/主人手填后拍照或扫描存档，不需要联网。

用法：
    python3 skin_health/code/gen_questionnaire_pdf.py \
        --out skin_health/docs/questionnaire_paper_form.pdf
"""
import argparse

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (HRFlowable, PageBreak, Paragraph, SimpleDocTemplate,
                                 Spacer, Table, TableStyle)

FONT = "STSong-Light"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("TitleCN", parent=styles["Title"], fontName=FONT, fontSize=18, leading=22)
H2 = ParagraphStyle("H2CN", parent=styles["Heading2"], fontName=FONT, fontSize=13, leading=17,
                    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1a4d8f"))
BODY = ParagraphStyle("BodyCN", parent=styles["Normal"], fontName=FONT, fontSize=10.5, leading=15)
NOTE = ParagraphStyle("NoteCN", parent=styles["Normal"], fontName=FONT, fontSize=9, leading=13,
                      textColor=colors.HexColor("#666666"))
QUESTION = ParagraphStyle("QCN", parent=styles["Normal"], fontName=FONT, fontSize=11, leading=16,
                          spaceBefore=8, spaceAfter=3, fontWeight="bold")
OPTION = ParagraphStyle("OptCN", parent=styles["Normal"], fontName=FONT, fontSize=10.5, leading=18,
                        leftIndent=14)

CHECKBOX = "&#9633;"  # □


def option_line(text):
    return Paragraph(f"{CHECKBOX}&nbsp;&nbsp;{text}", OPTION)


def build_story():
    story = []
    story.append(Paragraph("狗狗皮肤问答表", TITLE))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "填表说明：设备检测到抓挠水平比平时高，麻烦花1-2分钟观察并填写下面的问题，"
        "帮助更准确判断是否需要就医。没有把握的问题可以留空，不强制全部填写。",
        NOTE))
    story.append(Spacer(1, 8))

    # ── 基本信息 ──
    info_table = Table([
        ["狗狗名字：", "", "填表日期：", ""],
        ["填表人：", "", "触发原因：  □C1需要关注   □C2建议兽医检查", ""],
    ], colWidths=[65 * mm, 55 * mm, 40 * mm, 25 * mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (1, 0), (1, 0), 0.6, colors.black),
        ("LINEBELOW", (3, 0), (3, 0), 0.6, colors.black),
        ("SPAN", (2, 1), (3, 1)),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cccccc")))

    # ── 前置引导问题 ──
    story.append(Paragraph("一、前置问题", H2))
    story.append(Paragraph("1. 过去24小时内，狗狗有抓挠、舔舐或啃咬自己的行为吗？", QUESTION))
    story.append(option_line("是（请继续填写「二、抓挠行为」）"))
    story.append(option_line("否（「二、抓挠行为」可跳过，直接填「三」开始的部分）"))
    story.append(Paragraph("2. 狗狗身上有没有毛发稀疏或没有毛的情况？", QUESTION))
    story.append(option_line("是（请继续填写「六、毛发状态」里的秃毛部位/秃毛面积两题）"))
    story.append(option_line("否（这两题可以跳过，只填「整体毛发状态」这一题）"))

    # ── 二、抓挠行为自报 ──
    story.append(Paragraph("二、抓挠行为（前置问题1选「是」才需要填）", H2))
    story.append(Paragraph("3. 过去24小时内，狗狗抓挠、舔舐、啃咬自己的次数大概有多少？", QUESTION))
    for t in ["基本不抓，少于5次",
             "偶尔抓/舔/咬，大概5-15次",
             "会时不时抓、舔、咬，大概15-30次",
             "抓挠、舔咬比较频繁，大概30-50次，还可能会蹭地、蹭家具",
             "非常频繁，超过50次；狗狗看起来很烦躁，甚至不吃饭、不睡觉或体重明显下降"]:
        story.append(option_line(t))
    story.append(Paragraph("4. 过去24小时内，狗狗抓挠、舔舐、啃咬自己的总时间大概有多久？", QUESTION))
    for t in ["基本没有抓挠、舔咬",
             "偶尔出现，总共不到1小时，不影响正常生活",
             "总共大约1-3小时；会抓会舔，但吃饭、玩耍时能停下来，睡觉基本正常",
             "总共大约3-6小时；吃饭或玩耍时也会被痒打断，睡觉也受影响",
             "超过6小时；几乎一直在抓、舔、咬或摩擦，睡觉明显睡不好"]:
        story.append(option_line(t))

    story.append(PageBreak())

    # ── 三、皮肤颜色 ──
    story.append(Paragraph("三、皮肤外观", H2))
    story.append(Paragraph("5. 拨开狗狗的毛发看皮肤时，颜色是什么样的？（可多选：发红程度选1项，"
                           "颜色/色素异常如果有再多选）", QUESTION))
    for t in ["粉粉的、肉色的，看起来比较正常",
             "有一点点发红",
             "明显鲜红，但还没有破皮",
             "皮肤表面有黑色油油的东西，可以擦下来",
             "皮肤变黑、变褐色或变成淡褐色，像是颜色沉下去了，擦不掉"]:
        story.append(option_line(t))

    # ── 四、异味 ──
    story.append(Paragraph("四、气味", H2))
    story.append(Paragraph("6. 狗狗身上有没有明显异味？", QUESTION))
    for t in ["没有什么异味",
             "只有凑近闻，才能闻到一点油脂味、潮湿味",
             "离它大概30-50cm，就能闻到比较明显的臭味",
             "一进屋或者离得很远，就能闻到恶臭"]:
        story.append(option_line(t))

    # ── 五、皮损 ──
    story.append(Paragraph("五、皮肤完整性", H2))
    story.append(Paragraph("7. 狗狗的皮肤是否完整？", QUESTION))
    for t in ["皮肤完整，看不出异常",
             "只是有点干，有少量细小白色皮屑",
             "皮肤上有一块块大于1厘米异常区域，或者有成片、成块的皮屑/表皮脱落",
             "有糜烂、液体、结痂、脓包、红疙瘩，或者皮肤裂开"]:
        story.append(option_line(t))

    story.append(PageBreak())

    # ── 六、毛发状态 ──
    story.append(Paragraph("六、毛发状态", H2))
    story.append(Paragraph("8. 没有毛或毛发稀疏的地方是如何分布的？（前置问题2选「是」才需要填）", QUESTION))
    for t in ["没有明显的无毛或毛发稀疏",
             "有1-2个小地方没有毛或毛发稀疏，比如爪子、耳朵边、肚子局部",
             "有3处或更多明显没有毛或毛发稀疏的地方",
             "脱毛连成一大片，不是零星小块"]:
        story.append(option_line(t))
    story.append(Paragraph("9. 最大的一块秃毛区域大概有多大？（前置问题2选「是」才需要填）", QUESTION))
    for t in ["没有脱毛区域",
             "最大的一块很小，直径不到1-2cm",
             "最大的一块大约超过2-3cm",
             "最大的一块超过3cm，面积比较明显"]:
        story.append(option_line(t))
    story.append(Paragraph("10. 整体毛发状态看起来怎么样？（这题始终要填，不受前置问题2限制）", QUESTION))
    for t in ["毛发光亮、顺滑、浓密，看起来比较健康",
             "毛发有点油、容易打结，摸起来不太清爽",
             "有小范围毛发断裂、变稀、变少",
             "大部分毛发都明显干枯、易断、毛质很差"]:
        story.append(option_line(t))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 6))
    story.append(Paragraph("十一、其他备注（选填，比如「最近洗澡了」「换粮了」「天气闷热」这类"
                           "可能影响判断的情况）", H2))
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#999999")))
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#999999")))

    return story


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="skin_health/docs/questionnaire_paper_form.pdf")
    args = ap.parse_args()

    doc = SimpleDocTemplate(args.out, pagesize=A4,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            leftMargin=20 * mm, rightMargin=20 * mm)
    doc.build(build_story())
    print(f"已生成: {args.out}")


if __name__ == "__main__":
    main()
