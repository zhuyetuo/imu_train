"""「未标时间」里的预报是什么：离最近的人标抓挠多远、离最近的任何人标多远、多长、平均概率、物理特征；跟对的 / 确定误报对比"""
import sys

import numpy as np

sys.path.insert(0, "/home/toky/claude_exp/evt")
import evt_eval as E  # noqa: E402

D = E.load(sys.argv[1])
k = D["classes"].index(E.SCR)
hz, W = D["hz"], D["W"]
rows = {"tp": [], "fp": [], "unl": []}
for f in D["files"]:
    pred = E.predict_file(f, {})
    gt = [(a, b) for a, b, c, *_ in f["segs"] if c == E.SCR]
    lab = [(a, b) for a, b, *_ in f["segs"]]
    other = [(a, b) for a, b, c, *_ in f["segs"] if c != E.SCR]
    st = f["starts"] / hz + W / hz / 2
    P = f["P"].astype(np.float32)
    ph = f["phys"].astype(np.float32)
    for p in pred:
        if any(E._ov(p, g) >= min(E.OV_S, (p[1] - p[0]) / 2) for g in gt):
            kind = "tp"
        elif any(E._ov(p, o) > 0 for o in other):
            kind = "fp"
        else:
            kind = "unl"
        dg = min([max(0, g[0] - p[1], p[0] - g[1]) for g in gt], default=1e9)
        dl = min([max(0, g[0] - p[1], p[0] - g[1]) for g in lab], default=1e9)
        m = (st >= p[0]) & (st <= p[1])
        if not m.any():
            m = np.abs(st - (p[0] + p[1]) / 2) <= W / hz / 2
        rows[kind].append((dg, dl, p[1] - p[0], float(P[m, k].mean()), *np.nanmean(ph[m], 0), f["site"]))
for kind, r in rows.items():
    a = np.array([x[:7] for x in r], float)
    print(f"== {kind}: {len(r)} 个")
    print(f"   时长 中位 {np.median(a[:,2]):.1f}s；平均抓挠概率 中位 {np.median(a[:,3]):.2f}；模长 {np.nanmedian(a[:,4]):.2f}；陀螺仪 std 中位 {np.nanmedian(a[:,5]):.1f}；节律占比 中位 {np.nanmedian(a[:,6]):.2f}")
    if kind == "unl":
        for lim in (2, 5, 10, 30, 60):
            print(f"   离最近人标抓挠 ≤{lim}s：{(a[:,0] <= lim).mean():.1%}；离最近任何人标 ≤{lim}s：{(a[:,1] <= lim).mean():.1%}")
        print("   平均概率分布：", np.percentile(a[:,3], [10, 25, 50, 75, 90]).round(2))
