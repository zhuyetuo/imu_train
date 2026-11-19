import json, collections, numpy as np, os
base = os.path.expanduser("~/claude_exp/oof_v19neg/w8_6ax.npz")
Z = np.load(base); segs = json.load(open(base + ".segs.json"))
P, y, seg = Z["proba"], Z["y"], Z["seg"]; C = list(Z["classes"]); S = float(Z["stride_s"]); W = float(Z["window_s"])
ACT, SCR = C.index("活动"), C.index("抓挠")
pred = P.argmax(1)
k = np.zeros(len(seg), int); last = -1; c = 0
for i, s in enumerate(seg):
    c = c + 1 if s == last else 0; last = s; k[i] = c
scr = collections.defaultdict(list)
for s in segs:
    if s["coarse"] == "抓挠" and s["start_ms"] is not None:
        scr[s["csv"]].append((s["start_ms"] / 1000, s["end_ms"] / 1000))
fp = np.flatnonzero((y == ACT) & (pred == SCR))
# 按文件把误报窗的时间段收集起来，间隔 ≤5 秒的合成一段
by_file = collections.defaultdict(list)
for i in fp:
    s = segs[seg[i]]
    if s["dataset"] not in ("B_reviewed_20261118", "pool_20261118_all") or s["start_ms"] is None:
        continue
    t0 = s["start_ms"] / 1000 + k[i] * S
    by_file[(s["sample_code"], s["csv"], s["dataset"])].append((t0, t0 + W, float(P[i, SCR])))
runs = []
for (code, csv, ds), ws in by_file.items():
    ws.sort()
    cur = None
    for a, b, p in ws:
        if cur and a <= cur["end"] + 5:
            cur["end"] = max(cur["end"], b); cur["ps"].append(p)
        else:
            cur = {"sample_code": code, "csv": csv, "dataset": ds, "start": a, "end": b, "ps": [p]}; runs.append(cur)
for r in runs:
    r["p"] = float(np.mean(r["ps"])); r["n"] = len(r.pop("ps"))
    ds_ = [max(0, a - r["end"], r["start"] - b) for a, b in scr.get(r["csv"], [])]
    r["dist"] = min(ds_) if ds_ else None
    r.pop("csv")
cand = [r for r in runs if (r["dist"] is None or r["dist"] > 5) and r["p"] >= 0.6]
cand.sort(key=lambda r: (-r["p"] * min(r["n"], 20)))
# 每个文件最多 3 段，铺开到更多的狗 / 天
per, pool = collections.Counter(), []
for r in cand:
    if per[r["sample_code"]] >= 3: continue
    per[r["sample_code"]] += 1; pool.append(r)
print("合成后的误报段", len(runs), "；离抓挠 >5 秒且平均把握 ≥0.6：", len(cand), "；每文件最多 3 段后：", len(pool))
print("按数据集：", collections.Counter(r["dataset"] for r in pool[:150]))
print("前 150 段：文件数", len({r["sample_code"] for r in pool[:150]}), "，平均把握", round(np.mean([r["p"] for r in pool[:150]]), 2),
      "，长度中位", np.median([r["end"] - r["start"] for r in pool[:150]]), "秒")
json.dump(pool, open("/tmp/act_scr_pool.json", "w"), ensure_ascii=False)
