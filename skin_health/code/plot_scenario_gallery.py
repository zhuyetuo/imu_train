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

from gen_synthetic_scratch_scenarios import SCENARIOS
from scratch_burden import daily_features, run_pipeline

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_REPO_ROOT, "skin_health", "data", "synthetic_scenarios")
ASSET_DIR = os.path.join(_REPO_ROOT, "skin_health", "docs", "assets")
DOC_PATH = os.path.join(_REPO_ROOT, "skin_health", "docs", "scenario_gallery.md")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSET_DIR, exist_ok=True)

_TIER_COLOR = {"C0": "#4c9a5c", "C1": "#e0a326", "C2": "#c0392b", "insufficient_data": "#9aa0a6"}


def plot_one(pet_id, daily_df, result_df, expect):
    """上图：每日抓挠次数（柱状）+ 总时长（次坐标折线）；下图：SBS总分随天变化，
    C0/C1/C2三档背景色区分，红旗天用星号标出。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.3]})

    d = daily_df.sort_values("date")
    ax1.bar(d["date"], d["event_count"], color="#4a7fb5", width=0.8, label="抓挠次数")
    ax1b = ax1.twinx()
    ax1b.plot(d["date"], d["total_duration_sec"] / 60.0, color="#c0392b",
              marker="o", markersize=3, linewidth=1.2, label="总时长(分钟)")
    ax1.set_ylabel("抓挠次数", color="#4a7fb5")
    ax1b.set_ylabel("总时长(分钟)", color="#c0392b")
    ax1.set_title(f"{pet_id}  |  {expect}", fontsize=10, loc="left", wrap=True)

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
                    facecolors="none", edgecolors="black", linewidths=1.2,
                    zorder=4, label="触发红旗")
    bootstrap = r_scored[r_scored["bootstrap_mode"]]
    if len(bootstrap):
        ax2.scatter(bootstrap["date"], bootstrap["total"], marker="s", s=70,
                    facecolors="none", edgecolors="#555", linewidths=1.0,
                    zorder=4, label="引导期(bootstrap)")
    ax2.set_ylabel("SBS总分")
    ax2.set_ylim(-2, 100)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

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
    doc_lines = [
        "# 皮肤评级合成场景画廊",
        "",
        "> 每个场景对应 `gen_synthetic_scratch_scenarios.py` 里的一个 `scenario_*` 函数，"
        "用来验证 SBS 打分机制（不是抓挠事件识别准确率）。这份文档配图+关键天数表格，"
        "不用跑代码就能直观看每个场景的抓挠情况和评分结果；完整逐天数据在对应的CSV里。",
        "",
        "图例：上图蓝柱=每日抓挠次数，红线=每日总时长(分钟)；下图散点=SBS总分，"
        "背景绿/黄/红对应C0/C1/C2分档，黑色星号=当天触发红旗，空心方块=处于引导期"
        "(bootstrap_mode，历史不足21天，靠绝对阈值兜底评分)。",
        "",
    ]

    for fn in SCENARIOS:
        pet_id, events, wear, expect = fn()
        events_df = pd.DataFrame(events)
        wear_df = pd.DataFrame(wear)
        daily_df = daily_features(events_df, wear_df)
        result_df = run_pipeline(events_df, wear_df)

        events_csv = os.path.join(DATA_DIR, f"{pet_id}_events.csv")
        daily_csv = os.path.join(DATA_DIR, f"{pet_id}_daily_result.csv")
        events_df.to_csv(events_csv, index=False)
        result_df.to_csv(daily_csv, index=False)

        img_rel = plot_one(pet_id, daily_df, result_df, expect)
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
            f"## {pet_id}",
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
