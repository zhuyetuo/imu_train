"""新模型（未标时间当负例，全部空档）交叉验证里「未标时间」的抓挠预报 → 抽查池 /tmp/unl_pool.json。
只要平台数据池的文件（老数据 / 公开数据在平台上没有任务）。带上样本编号、场地、段均概率、时长"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/toky/claude_exp/evt")
sys.path[:0] = ["/home/toky/claude_exp/imu_pages/src/ml", "/home/toky/claude_exp/imu_pages/src/data"]
import evt_eval as E  # noqa: E402
import train_v2 as T  # noqa: E402

D = E.load("/home/toky/claude_exp/evt/gapall_seg.pkl.gz")
spec = json.load(open("/home/toky/claude_exp/evt/spec_gapall.json"))
main = spec["datasets"][0]
segs = T.collect_segments([main], spec["nas_root"], spec["label_remap"], spec["classes"], spec.get("fine_labels") or [])
code = {}
for s in segs:
    code.setdefault(s["csv"], s["sample_code"])
k = D["classes"].index(E.SCR)
hz, W = D["hz"], D["W"]
pool = []
for f in D["files"]:
    if f["csv"] not in code:
        continue
    pred = E.predict_file(f, {})
    gt = [(a, b) for a, b, c, *_ in f["segs"] if c == E.SCR]
    lab = [(a, b) for a, b, *_ in f["segs"]]
    st = f["starts"] / hz + W / hz / 2
    P = f["P"].astype(np.float32)
    for p in pred:
        if any(E._ov(p, g) > 0 for g in lab):
            continue
        m = (st >= p[0]) & (st <= p[1])
        pm = float(P[m, k].mean()) if m.any() else None
        dg = min([max(0, g[0] - p[1], p[0] - g[1]) for g in gt], default=None)
        pool.append({"sample_code": code[f["csv"]], "site": f["site"], "start": round(p[0], 2), "end": round(p[1], 2),
                     "p": pm, "dist": dg, "csv": f["csv"]})
print("未标时间预报（数据池文件）：", len(pool), {s: sum(1 for q in pool if q["site"] == s) for s in {q["site"] for q in pool}})
L = np.array([q["end"] - q["start"] for q in pool]); Pm = np.array([q["p"] or 0 for q in pool])
print(f"时长中位 {np.median(L):.1f}s，段均概率中位 {np.median(Pm):.2f}")
json.dump(pool, open("/tmp/unl_pool.json", "w", encoding="utf-8"), ensure_ascii=False)
