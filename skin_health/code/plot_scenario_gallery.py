"""
给 gen_synthetic_scratch_scenarios.py 里每个合成场景出一张图 + 保留合成数据，
配合 docs/scenario_gallery.md 使用，方便不跑代码也能直观看每个场景该有的
抓挠情况和SBS打分行为。

跟 gen_synthetic_scratch_scenarios.py 的关系：场景定义（10个 scenario_* 函数）
复用那边的，这里只多做三件事：
  1. 每个场景单独存一份合成数据（事件明细 + 每日SBS结果）成CSV，方便直接打开看
  2. 每个场景画一张图（上：每日抓挠次数/时长柱状图，下：SBS总分曲线+C0/C1/C2分档线）
  3. 生成一份 md 画廊文档，把每个场景的预期说明、图、CSV链接放在一起

用法：
    python skin_health/code/plot_scenario_gallery.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

for cand in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei"]:
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from gen_synthetic_scratch_scenarios import SCENARIOS
from scratch_burden import daily_features, run_pipeline

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "skin_health", "data", "synthetic_scenarios")
ASSET_DIR = os.path.join(_REPO_ROOT, "skin_health", "docs", "assets")
DOC_PATH = os.path.join(_REPO_ROOT, "skin_health", "docs", "scenario_gallery.md")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

_TIER_COLOR = {"C0": "#4c9a5c", "C1": "#e0a326", "C2": "#c0392b", "insufficient_data": "#9aa0a6"}


def plot_one(idx, pet_id, daily_df, result_df, expect):
    """上图：每日抓挠次数（柱状）+ 总时长（次坐标折线）；下图：SBS总分随天变化，
    C0/C1/C2三档背景色区分，红旗天用星号标出。每张图自带完整图例，不用对照文字说明才能看懂。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.3]})

    d = daily_df.sort_values("date")
    bar = ax1.bar(d["date"], d["event_count"], color="#4a7fb5", width=0.8)
    ax1b = ax1.twinx()
    line, = ax1b.plot(d["date"], d["total_duration_sec"] / 60.0, color="#c0392b",
                       marker="o", markersize=3, linewidth=1.2)
    ax1.set_ylabel("抓挠次数（蓝柱，左轴）", color="#4a7fb5")
    ax1b.set_ylabel("抓挠总时长/分钟（红线，右轴）", color="#c0392b")
    ax1.set_title("上图：每日抓挠原始情况（左轴蓝柱=当天抓挠次数，右轴红线=当天抓挠总时长）",
                  fontsize=9, loc="left")
    ax1.legend([bar, line], ["每日抓挠次数（左轴，蓝柱）", "每日抓挠总时长/分钟（右轴，红线）"],
               loc="upper left", fontsize=8, framealpha=0.9)

    r = result_df.sort_values("date")
    r_scored = r[r["tier"] != "insufficient_data"]
    ax2.axhspan(0, 30, color="#4c9a5c", alpha=0.08)
    ax2.axhspan(30, 50, color="#e0a326", alpha=0.10)
    ax2.axhspan(50, 100, color="#c0392b", alpha=0.08)
    ax2.axhline(30, color="#e0a326", linewidth=0.8, linestyle="--")
    ax2.axhline(50, color="#c0392b", linewidth=0.8, linestyle="--")
    colors = [_TIER_COLOR.get(t, "#9aa0a6") for t in r_scored["tier"]]
    ax2.scatter(r_scored["date"], r_scored["total"], c=colors, s=28, zorder=3)
    ax2.plot(r_scored["date"], r_scored["total"], color="#555", linewidth=0.8, zorder=2)
    flagged = r_scored[r_scored["red_flags"].apply(lambda x: bool(x))]
    if len(flagged):
        ax2.scatter(flagged["date"], flagged["total"], marker="*", s=180,
                    facecolors="none", edgecolors="black", linewidths=1.2, zorder=4)
    bootstrap = r_scored[r_scored["bootstrap_mode"]]
    if len(bootstrap):
        ax2.scatter(bootstrap["date"], bootstrap["total"], marker="s", s=70,
                    facecolors="none", edgecolors="#555", linewidths=1.0, zorder=4)
    ax2.set_ylabel("SBS总分（0~100，越高越需要关注）")
    ax2.set_ylim(-2, 100)
    ax2.set_title("下图：每日SBS皮肤评级总分（灰线连接每天分数，圆点颜色=当天分档）",
                  fontsize=9, loc="left")
    legend_handles = [
        Line2D([0], [0], marker="o", color="#555", markerfacecolor=_TIER_COLOR["C0"],
               markersize=8, linewidth=0, label="C0 正常（绿色圆点，总分<30）"),
        Line2D([0], [0], marker="o", color="#555", markerfacecolor=_TIER_COLOR["C1"],
               markersize=8, linewidth=0, label="C1 需要关注（黄色圆点，30≤总分<50）"),
        Line2D([0], [0], marker="o", color="#555", markerfacecolor=_TIER_COLOR["C2"],
               markersize=8, linewidth=0, label="C2 建议兽医检查（红色圆点，总分≥50）"),
        Line2D([0], [0], marker="*", color="w", markeredgecolor="black", markerfacecolor="none",
               markersize=14, label="黑色星号=当天触发红旗（异常聚集/长时间抓挠/影响睡眠等）"),
        Line2D([0], [0], marker="s", color="w", markeredgecolor="#555", markerfacecolor="none",
               markersize=10, label="空心方块=处于引导期（历史不足21天，靠绝对阈值兜底评分，不是跟自己比）"),
    ]
    ax2.legend(handles=legend_handles, loc="upper left", fontsize=7.5, framealpha=0.9)
    ax2.set_xlabel(f"日期（横轴每一格=1天，共{len(d)}天：前21天是基线观察窗口，"
                   "后14天是评估期，同一天上下两张图对齐）")
    fig.suptitle(f"场景{idx}/{len(SCENARIOS)}：{pet_id}\n预期：{expect}",
                 fontsize=11, x=0.01, ha="left", y=0.995)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    out_path = os.path.join(ASSET_DIR, f"scenario_{pet_id}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return os.path.relpath(out_path, os.path.dirname(DOC_PATH))


def summarize_key_days(result_df):
    """挑几天有代表性的行（tier非C0，或触发红旗，或bootstrap_mode）放进md表格，
    不把35天全列出来（太长），完整数据看CSV。"""
    r = result_df[result_df["tier"] != "insufficient_data"].copy()
    interesting = r[(r["tier"] != "C0") | (r["red_flags"].apply(bool)) | (r["bootstrap_mode"])]
    if interesting.empty:
        interesting = r.tail(3)
    return interesting.sort_values("date").head(8)


def main():
    n = len(SCENARIOS)
    doc_lines = [
        "# 皮肤评级合成场景画廊",
        "",
        f"> 一共 {n} 个合成场景，每个对应 `gen_synthetic_scratch_scenarios.py` 里的一个 "
        "`scenario_*` 函数，目的是验证 **SBS 打分机制本身**（不是抓挠事件识别准确率）——"
        "也就是给定一组已知的抓挠事件，SBS引擎算出来的评级是否符合预期，用来在改动打分公式后"
        "快速回归测试。下面每个场景的标题都带编号（场景1/10、场景2/10……），跟目录一一对应。",
        "",
        "## 名词解释（看不懂图例的先看这里）",
        "",
        "- **SBS（Scratch Burden Score，抓挠负担分）**：算法给每只狗每天算出来的0~100分，"
        "分数越高代表当天的抓挠行为越偏离正常，由四个子项（变化幅度/聚集程度/持续程度/"
        "正常行为中断）加总得到，达到不同门槛对应下面的C0/C1/C2三档。",
        "- **C0 正常 / C1 需要关注 / C2 建议兽医检查**：SBS总分 <30 是C0，30~50是C1，≥50是C2，"
        "对应下图绿/黄/红三种背景色。",
        "- **红旗（red flag）**：不管总分多少，只要触发了预定义的严重信号（比如单次连续抓挠"
        "超过1分钟、1小时内聚集抓挠≥3次、影响夜间睡眠等），就在图上标黑色星号，代表这天"
        "有值得直接关注的具体行为，不是单纯靠算分数判断。",
        "- **引导期（bootstrap_mode）**：这只狗历史数据还不满21天，没法算出\"跟自己比\"的"
        "个人基线，这期间改用文献（Whistle FIT）给的固定秒数阈值兜底评分，图上标空心方块，"
        "跟基线建立后的正常评分是两套逻辑，不能直接比较分数高低。",
        "",
        "- **横轴（日期）**：每一格代表**一整天**（不是一天内的时间段），从合成数据第1天"
        "（2026-06-01）到第35天，前21天是给算法建立个人基线用的观察窗口，后14天是真正"
        "评估/触发问题的观察期，具体每个场景的\"异常\"通常安排在后14天里发生。",
        "",
        "每张图分两部分：**上图**是当天实际发生的抓挠情况（原始数据，蓝柱=次数，红线=总时长），"
        "**下图**是算法根据上图数据算出来的SBS评分结果（total/tier），两张图上下对齐、"
        "同一天可以直接对照看\"当天发生了什么 → 算法打了多少分\"。每张图内部都带完整图例，"
        "不需要再回来看这段文字。",
        "",
    ]

    for idx, fn in enumerate(SCENARIOS, start=1):
        pet_id, events, wear, expect = fn()
        events_df = pd.DataFrame(events)
        wear_df = pd.DataFrame(wear)
        daily_df = daily_features(events_df, wear_df)
        result_df = run_pipeline(events_df, wear_df)

        events_csv = os.path.join(DATA_DIR, f"{pet_id}_events.csv")
        daily_csv = os.path.join(DATA_DIR, f"{pet_id}_daily_result.csv")
        events_df.to_csv(events_csv, index=False)
        result_df.to_csv(daily_csv, index=False)

        img_rel = plot_one(idx, pet_id, daily_df, result_df, expect)
        events_csv_rel = os.path.relpath(events_csv, os.path.dirname(DOC_PATH))
        daily_csv_rel = os.path.relpath(daily_csv, os.path.dirname(DOC_PATH))

        key_days = summarize_key_days(result_df)
        table_lines = [
            "| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |",
            "|---|---|---|---|---|---|",
        ]
        daily_lookup = daily_df.set_index("date")["event_count"].to_dict()
        for _, row in key_days.iterrows():
            flags = ", ".join(row["red_flags"]) if row["red_flags"] else "-"
            table_lines.append(
                f"| {row['date']} | {int(daily_lookup.get(row['date'], 0))} | {row['total']:.0f} | "
                f"{row['tier']} | {flags} | {'是' if row['bootstrap_mode'] else '否'} |"
            )

        doc_lines += [
            f"## 场景{idx}/{n}：{pet_id}",
            "",
            f"**预期**：{expect}",
            "",
            f"![{pet_id}]({img_rel})",
            "",
            "关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：",
            "",
            *table_lines,
            "",
            f"原始数据：[抓挠事件明细]({events_csv_rel}) | [每日SBS结果]({daily_csv_rel})",
            "",
            "---",
            "",
        ]

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines))
    print(f"已生成 {DOC_PATH}，共 {len(SCENARIOS)} 个场景，图片在 {ASSET_DIR}，CSV在 {DATA_DIR}")


if __name__ == "__main__":
    main()
