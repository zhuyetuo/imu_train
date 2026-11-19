import json, collections, numpy as np, os
base = os.path.expanduser("~/claude_exp/oof_v19neg/w8_6ax.npz")
Z = np.load(base); segs = json.load(open(base + ".segs.json"))
P, y, seg = Z["proba"], Z["y"], Z["seg"]; C = list(Z["classes"]); S = float(Z["stride_s"]); W = float(Z["window_s"])
ACT, SCR = C.index("活动"), C.index("抓挠")
pred = P.argmax(1)
# 窗口在片段里的序号 → 窗口起点（片段起点 + k × 步长）
k = np.zeros(len(seg), int); last = -1; c = 0
for i, s in enumerate(seg):
    c = c + 1 if s == last else 0; last = s; k[i] = c
scr_by_csv = collections.defaultdict(list)
for s in segs:
    if s["coarse"] == "抓挠" and s["start_ms"] is not None:
        scr_by_csv[s["csv"]].append((s["start_ms"] / 1000, s["end_ms"] / 1000))
def dist(i):
    s = segs[seg[i]]
    if s["start_ms"] is None: return None
    w0 = s["start_ms"] / 1000 + k[i] * S; w1 = w0 + W
    ds = [max(0, a - w1, w0 - b) for a, b in scr_by_csv.get(s["csv"], [])]
    return min(ds) if ds else None
def bucket(d):
    if d is None: return "文件里没有抓挠标注"
    if d == 0: return "窗口跟抓挠标注重叠"
    if d <= 5: return "离抓挠 ≤5 秒"
    if d <= 30: return "离抓挠 5–30 秒"
    if d <= 120: return "离抓挠 30 秒–2 分钟"
    return "离抓挠 >2 分钟"
def report(mask, title):
    idx = np.flatnonzero(mask); print(f"\n##### {title}：{len(idx)} 窗")
    ds = collections.Counter(segs[seg[i]]["dataset"] for i in idx)
    print("按数据集：", dict(ds.most_common()))
    b = collections.Counter(bucket(dist(i)) for i in idx)
    print("离最近抓挠标注：", {k_: f"{v} ({v/len(idx)*100:.0f}%)" for k_, v in b.most_common()})
    conf = P[idx, SCR]
    print("判成抓挠的置信度：", {f"≥{t}": int((conf >= t).sum()) for t in (0.5, 0.7, 0.9)})
    return idx
fp = report((y == ACT) & (pred == SCR), "真实「活动」被判成抓挠")
# 连续的误判窗合成一段，列最长的
runs = []; cur = None
for i in fp:
    if cur and seg[i] == cur["seg"] and k[i] == cur["k1"] + 1:
        cur["k1"] = k[i]; cur["n"] += 1; cur["p"] += P[i, SCR]
    else:
        cur = {"seg": seg[i], "k0": k[i], "k1": k[i], "n": 1, "p": P[i, SCR], "i": i}; runs.append(cur)
print(f"\n合成 {len(runs)} 段连续误报；长度分布：", {f"{a}-{b}秒": sum(a <= r['n'] + 7 <= b for r in runs) for a, b in ((8, 10), (11, 20), (21, 60), (61, 10**6))})
ftask = collections.Counter(segs[r["seg"]]["sample_code"] for r in runs)
print("误报段最多的文件：")
for code, n in ftask.most_common(12):
    nw = sum(r["n"] for r in runs if segs[r["seg"]]["sample_code"] == code)
    print(f"   {code} [{segs[next(r['seg'] for r in runs if segs[r['seg']]['sample_code']==code)]['dataset']}]: {n} 段 / {nw} 窗")
print("\n最长的 20 段（文件, 片段内起止 mm:ss, 长度, 平均置信度, 离抓挠）:")
out = []
for r in sorted(runs, key=lambda r: -r["n"])[:20]:
    s = segs[r["seg"]]; t0 = s["start_ms"] / 1000 + r["k0"] * S; t1 = s["start_ms"] / 1000 + r["k1"] * S + W
    f = lambda t: f"{int(t//60):02d}:{int(t%60):02d}"
    d = dist(r["i"])
    print(f"   {s['sample_code']} {f(t0)}–{f(t1)} {t1-t0:.0f}s p={r['p']/r['n']:.2f} 离抓挠={'无' if d is None else f'{d:.0f}s'} [{s['dataset']}] 标签={s['leaf']}")
    out.append(dict(sample_code=s["sample_code"], dataset=s["dataset"], csv=s["csv"], t0=t0, t1=t1, p=float(r["p"]/r["n"])))
json.dump(out, open(os.path.expanduser("~/claude_exp/oof_v19neg/fp_top20.json"), "w"), ensure_ascii=False, indent=1)
report((y == SCR) & (pred == ACT), "真实「抓挠」被判成活动（漏报）")
