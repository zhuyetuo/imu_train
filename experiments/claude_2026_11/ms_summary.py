"""把 queue/ 下每个跑完的 res_*.json 汇总成 summary.md（队列每跑完一个就重写一次）"""
import glob
import json
import os
import time

D = os.path.dirname(os.path.abspath(__file__))
items = [l.split("|") for l in open(os.path.join(D, "queue.txt"), encoding="utf-8").read().splitlines() if l.strip() and not l.startswith("#")]
status = {}
if os.path.exists(os.path.join(D, "status.txt")):
    for l in open(os.path.join(D, "status.txt"), encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) >= 3:
            status[p[1]] = (p[0], p[2])


def f(x, pct=False):
    if x is None:
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x}"


rows = []
for tag, desc, _args in items:
    st = status.get(tag, ("", "排队"))[1]
    p = os.path.join(D, f"res_{tag}.json")
    if not os.path.exists(p):
        rows.append(f"| {tag} | {desc} | {st} |" + " |" * 10)
        continue
    r = json.load(open(p, encoding="utf-8"))
    s, h = r["own"].get("抓挠", {}), r["own"].get("甩头/抖身", {})
    sa = r["all"].get("抓挠", {})
    rows.append(f"| {tag} | {desc} | {st}，{r['seconds'] // 60} 分 | {r['n_params']:,} | "
                f"{f(s.get('recall_hit'), 1)} | {f(s.get('recall_iou50'), 1)} | {f(s.get('precision'), 1)} | {f(s.get('fp_per_h_other'))} | "
                f"{f(sa.get('fp_per_h_other'))} | {f(s.get('onset_abs_err_s'))} | "
                f"{f(h.get('recall_hit'), 1)} | {f(h.get('recall_iou50'), 1)} | {f(h.get('precision'), 1)} |")

head = ("| 编号 | 实验 | 状态 | 参数量 | 抓挠命中 | 抓挠 IoU≥0.5 | 抓挠查准 | 抓挠误报/h（自己数据） | 抓挠误报/h（含公开） | 起点误差 s | "
        "甩头命中 | 甩头 IoU≥0.5 | 甩头查准 |\n|" + "---|" * 13)
txt = (f"# 多尺度 / 逐点输出 实验队列（自动生成，{time.strftime('%Y-%m-%d %H:%M')}）\n\n"
       "数据：pool_20261116_all + Vehkaoja 2022 + Marcato 2023（跟 #41 / train37 同一份 spec），6 轴 20 Hz，"
       "按（天, 项圈）分组 5 折，稳定版 v2 后处理后按片段算。命中 = 人标片段里模型报了；IoU≥0.5 = 起止也对得上；"
       "误报/h = 人标成别的动作的时间里每小时报几次（“自己数据”只算我们自己的项圈数据）。\n"
       "参照：现在线上 #41（6 轴 8 s CNN）抓挠命中 91.8%、查准 94.0%、误报 4.0/h（含公开）、甩头 IoU≥0.5 3.6%。\n\n"
       + head + "\n" + "\n".join(rows) + "\n")
open(os.path.join(D, "summary.md"), "w", encoding="utf-8").write(txt)
print(txt)
