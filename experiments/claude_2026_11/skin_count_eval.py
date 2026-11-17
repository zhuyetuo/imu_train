"""皮肤评估要的是「每天抓了几次、多久」。按任务比：同一段时间里，人工确认的抓挠段数 / 时长 vs
  老模型 = 平台当时的 AI 预标注（data_labeled_ai 里的 JSON，皮肤评估 AI 版用的就是它）
  新模型 = q01（6 轴 8 s CNN，跟 train37 同一份配置）5 折交叉验证里没见过这个文件的那一折，稳定版 v2 / 精准版
只用训练池里的任务（有交叉验证预测），时间范围取交叉验证推理的那段（人标的范围）。再按（日期, IMU）汇总，看每天的次数偏多少。
用法：python skin_count_eval.py reviewed_tasks.json res_q01_pred.pkl"""
import json
import os
import pickle
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

sys.path[:0] = ["/home/toky/imu_train/src/ml", "/home/toky/imu_train", "/home/toky/claude_exp/multiscale"]
import train_v2 as T  # noqa: E402
from label_service import postprocess as pp  # noqa: E402
from boundary_eval import envelope, params, refine_precise, shift_segs  # noqa: E402

NAS = "/home/toky/ai_data"


def merge(spans):
    out = []
    for a, b in sorted(spans):
        if out and a < out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def clip(spans, lo, hi):
    return [[max(a, lo), min(b, hi)] for a, b in spans if min(b, hi) > max(a, lo)]


def main(tasks_path, pkl):
    tasks = json.load(open(tasks_path, encoding="utf-8"))
    by_csv = {os.path.join(NAS, t["csv"]): t for t in tasks}
    d = pickle.load(open(pkl, "rb"))
    classes, W, S, hz = d["classes"], d["W"], d["S"], d["hz"]
    sp = params()
    targets = [c for c in T.EVENT_EVAL if c in classes]
    rows = []
    for f in d["files"]:
        t = by_csv.get(f["csv"])
        if t is None or f["dataset"] != "pool_20261116_all":
            continue
        lo, hi = f["starts"][0] / hz, f["starts"][-1] / hz + W / hz
        human = merge(clip([[a / 1000, b / 1000] for a, b in t["human"]], lo, hi))
        wins = [{"ts": pp._fmt_ts(f["t0"] + timedelta(seconds=float(s) / hz)), "label": classes[int(q.argmax())],
                 "conf": float(q.max()), "probs": {c: float(v) for c, v in zip(classes, q)}} for s, q in zip(f["starts"], f["P"])]
        segs = pp.stabilize(wins, classes, targets, W / hz, S / hz, "segment", sp, algo="viterbi")
        sec = lambda x: ((pp._parse_ts(x["start_ts"]) - f["t0"]).total_seconds(), (pp._parse_ts(x["end_ts"]) - f["t0"]).total_seconds())
        new = merge(clip([list(sec(x)) for x in segs.get("抓挠", [])], lo, hi))
        shift_segs(segs.get("抓挠", []), 0.75, -1.7)
        newp = merge(clip([list(sec(x)) for x in segs.get("抓挠", [])], lo, hi))
        old = None
        p = os.path.join(NAS, t["ai_json"])
        if os.path.exists(p):
            try:
                aj = json.load(open(p, encoding="utf-8"))
                o = []
                for lab, ss in (aj.get("segments") or {}).items():
                    if lab.startswith("抓挠"):
                        o += [[(pp._parse_ts(x["start_ts"]) - f["t0"]).total_seconds(), (pp._parse_ts(x["end_ts"]) - f["t0"]).total_seconds()] for x in ss]
                old = merge(clip(o, lo, hi))
            except Exception:  # noqa: BLE001
                old = None
        imu = t["sample_code"].split("_")[-1].upper()
        rows.append({"date": t["date"], "imu": imu, "hours": (hi - lo) / 3600,
                     "n_h": len(human), "n_new": len(new), "n_old": None if old is None else len(old),
                     "d_h": sum(b - a for a, b in human), "d_new": sum(b - a for a, b in new), "d_newp": sum(b - a for a, b in newp),
                     "d_old": None if old is None else sum(b - a for a, b in old)})
    rows = [r for r in rows if r["n_old"] is not None]
    print(f"任务 {len(rows)} 个，{sum(r['hours'] for r in rows):.0f} 小时；人工抓挠 {sum(r['n_h'] for r in rows)} 段")
    print(f"  老模型 {sum(r['n_old'] for r in rows)} 段，新模型 {sum(r['n_new'] for r in rows)} 段")
    day = defaultdict(lambda: defaultdict(float))
    for r in rows:
        k = (r["date"], r["imu"])
        for x in ("n_h", "n_new", "n_old", "d_h", "d_new", "d_newp", "d_old"):
            day[k][x] += r[x]
    days = [v for v in day.values() if v["n_h"] > 0]
    print(f"（日期, IMU）{len(day)} 个，有人工抓挠的 {len(days)} 个")
    for name, kn, kd in (("老模型（平台当时的 AI 版）", "n_old", "d_old"), ("新模型 v2", "n_new", "d_new"), ("新模型 精准版", "n_new", "d_newp")):
        rn = [v[kn] / v["n_h"] for v in days]
        rd = [v[kd] / v["d_h"] for v in days if v["d_h"] > 0]
        within = sum(1 for x in rn if 0.67 <= x <= 1.5) / len(rn)
        print(f"  {name:<16} 每天次数 / 人工：中位 {st.median(rn):.2f}（四分位 {st.quantiles(rn, n=4)[0]:.2f} ~ {st.quantiles(rn, n=4)[2]:.2f}），"
              f"在 ±50% 以内的天 {within:.0%}；时长 / 人工 中位 {st.median(rd):.2f}")
    json.dump({"rows": rows}, open("skin_count_eval.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
