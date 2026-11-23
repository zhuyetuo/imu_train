"""同 evt_cmp.py，但只在**几个模型共有的池子文件**上比（老数据 / 公开数据集的文件只标了抓挠或根本没抓挠，
混进来会让「未标」和误报虚高；8 s 窗口的模型又会丢掉太短的文件）。用法：python evt_cmp2.py base v1 v2b v3"""
import gzip
import json
import os
import pickle
import sys

import evt_eval as E

HERE = os.path.dirname(os.path.abspath(__file__))
MEANS = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
names = sys.argv[1:]
csvs = {}
for n in names:
    with gzip.open(os.path.join(HERE, f"{n}_seg.pkl.gz"), "rb") as f:
        d = pickle.load(f)
    csvs[n] = {x["csv"] for x in d["files"] if str(x.get("dataset") or "").startswith("pool_")}
    print(n, "池子文件", len(csvs[n]), "；数据集", sorted({str(x.get('dataset')) for x in d["files"]})[:8], flush=True)
common = set.intersection(*csvs.values())
print("共有的池子文件", len(common), flush=True)
out = open(os.path.join(HERE, "cmp2.jsonl"), "w", encoding="utf-8")
res = {}
for n in names:
    E.load(os.path.join(HERE, f"{n}_seg.pkl.gz"))
    E.D["files"] = [x for x in E.D["files"] if x["csv"] in common]
    print(f"== {n}: {len(E.D['files'])} 个文件，窗口 {E.D['W'] / E.D['hz']:.0f}s 步长 {E.D['S'] / E.D['hz']:.1f}s", flush=True)
    res[n] = []
    for mm in MEANS:
        m, what, sites = E.evaluate({"event_min_mean": mm})
        res[n].append((mm, m))
        out.write(json.dumps({"model": n, "event_min_mean": mm, **m, "fp_what": what.most_common(6),
                              "sites": sites}, ensure_ascii=False) + "\n")
        print(f"  ≥{mm:.2f}: 召回 {m['recall_loose']:.3f} 确定查准 {m['prec_known']:.3f} 全部查准 {m['prec_all']:.3f} "
              f"预测 {m['n_pred']} 对 {m['tp']} 确定误报 {m['fp']} 未标 {m['unlabeled']} 人标抓挠 {m['n_gt']} 严格 R/P {m['recall_iou50']:.3f}/{m['prec_iou50']:.3f}"
              f"  误报 {what.most_common(3)}", flush=True)
out.close()


def at_recall(rows, r):
    ok = [x for x in rows if x[1]["recall_loose"] >= r]
    return max(ok, key=lambda x: x[1]["prec_known"]) if ok else max(rows, key=lambda x: x[1]["recall_loose"])


print("\n相同召回下比（召回 ≥ 目标的档里确定查准最高的）：")
for r in (0.80, 0.85, 0.88):
    print(f"  召回 ≥ {r}:")
    for n, rows in res.items():
        mm, m = at_recall(rows, r)
        print(f"    {n:5s} ≥{mm:.2f}  召回 {m['recall_loose']:.3f} 确定查准 {m['prec_known']:.3f} 全部查准 {m['prec_all']:.3f} 确定误报 {m['fp']} 未标 {m['unlabeled']}")
