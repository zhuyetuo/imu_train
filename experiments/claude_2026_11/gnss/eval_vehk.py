"""在 Vehkaoja 2022（颈部 = 项圈，100 Hz → 20 Hz）上验证移动检测：逐秒对 Behavior_1。"""
import sys, collections, itertools, json, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from locomotion import steps_and_loco, DEFAULT
SRC = os.path.expanduser("~/ai_data/public_datasets/vehkaoja2022/extracted/csv/DogMoveData.csv")
cache = os.path.expanduser("~/claude_exp/gnss/vehk_neck20.npz")
if not os.path.exists(cache):
    cols = ["DogID", "TestNum", "ANeck_x", "ANeck_y", "ANeck_z", "Behavior_1"]
    parts = []
    for ch in pd.read_csv(SRC, usecols=cols, chunksize=2_000_000):
        parts.append(ch)
    df = pd.concat(parts)
    acc, lab, key = [], [], []
    for (dog, tn), g in df.groupby(["DogID", "TestNum"], sort=False):
        n = len(g) // 5 * 5
        a = g[["ANeck_x", "ANeck_y", "ANeck_z"]].to_numpy()[:n].reshape(-1, 5, 3).mean(1)
        b = g["Behavior_1"].to_numpy()[:n][::5]
        acc.append(a.astype(np.float32)); lab.append(b.astype(str)); key.append(f"{dog}_{tn}")
    np.savez(cache, acc=np.array(acc, dtype=object), lab=np.array(lab, dtype=object), key=np.array(key), allow_pickle=True)
Z = np.load(cache, allow_pickle=True)
ACC, LAB = Z["acc"], Z["lab"]
LOCO = {"Walking", "Trotting", "Galloping", "Pacing"}
SUS = "--sus" in sys.argv

def run(p):
    tot = collections.Counter(); hit = collections.Counter(); cads = collections.defaultdict(list)
    for a, l in zip(ACC, LAB):
        _, loco, cad = steps_and_loco(np.asarray(a, float), p)
        if SUS:  # 持续移动：最近 15 秒里至少 10 秒是走/跑
            c = np.convolve(loco.astype(int), np.ones(15, int))[:len(loco)]
            loco = c >= 10
        sec = np.asarray(l)[np.arange(len(loco)) * 20 + 10]
        for s, (b, f) in enumerate(zip(sec, loco)):
            if b in ("<undefined>", "Synchronization", "nan", "Extra_Synchronization"): continue
            tot[b] += 1; hit[b] += int(f)
            if b in LOCO: cads[b].append(cad[s])
    tp = sum(hit[b] for b in LOCO); pos = sum(tot[b] for b in LOCO)
    fp = sum(hit[b] for b in tot if b not in LOCO); neg = sum(tot[b] for b in tot if b not in LOCO)
    return tp / pos, fp / neg, tot, hit, cads

if "--grid" in sys.argv:
    best = []
    for pk, cv, ms in itertools.product([0.04, 0.06, 0.08, 0.1], [0.35, 0.45, 0.6], [4, 5, 6]):
        r, f, *_ = run({"peak_g": pk, "max_cv": cv, "min_steps": ms})
        best.append((r - 2 * f, pk, cv, ms, r, f))
    for b in sorted(best, reverse=True)[:8]: print("score %.3f peak %.2f cv %.2f min_steps %d | 召回 %.3f 误报 %.3f" % b)
else:
    p = json.loads(next((a for a in sys.argv[1:] if a.startswith("{")), "{}"))
    r, f, tot, hit, cads = run(p)
    print(f"参数 {dict(DEFAULT, **p)}")
    print(f"走/跑 逐秒召回 {r:.3f}；其它行为被当成走/跑 {f:.3f}")
    for b, n in sorted(tot.items(), key=lambda x: -x[1]):
        extra = f"  步频中位 {np.median(cads[b]):.2f}/s" if b in cads else ""
        print(f"  {b:<16} {n:>7} 秒  判成走/跑 {hit[b] / n:6.1%}{extra}")
