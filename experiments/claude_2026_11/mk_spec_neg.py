"""从 v18 spec 生成 v19neg：舔/啃、蹭、进食/饮水 拆成单独的负类（不再并进「活动」）"""
import json, collections
import sys
tag = sys.argv[1] if len(sys.argv) > 1 else "w8_6ax"   # w8_6ax / w8_3ax / w8s_6ax / w8s_3ax
src = f"experiments/claude_2026_11/specs/train_v2_cnn_{tag}_v18.json"
spec = json.load(open(src))
NEG = [("舔/啃", ("舔", "啃")), ("蹭", ("蹭",)), ("进食/饮水", ("进食", "饮水"))]
def neg_of(lbl):
    for cls, pre in NEG:
        if lbl.startswith(pre): return cls
    return None
remap = dict(spec["label_remap"])
mins = collections.Counter()
for ds in spec["datasets"]:
    for t in json.load(open(ds["export_json"])):
        for a in t.get("annotations") or []:
            for r in a.get("result") or []:
                v = r.get("value") or {}
                for l in v.get("timeserieslabels") or []:
                    c = neg_of(l)
                    if c: remap[l] = c
                ch = v.get("timeserieslabels") or []
                if ch and "start_ms" in v:
                    c = next((remap[x] for x in reversed(ch) if x in remap), None)
                    if c in [n for n, _ in NEG]: mins[(c, ch[-1])] += (v["end_ms"] - v["start_ms"]) / 60000
for k in list(remap):
    c = neg_of(k)
    if c: remap[k] = c
spec["label_remap"] = remap
spec["classes"] = spec["classes"] + [n for n, _ in NEG]
spec["fine_labels"] = []
spec["run_name"] = f"pool_20261118_B__cnn_{tag}_v19neg"
out = f"experiments/claude_2026_11/specs/train_v2_cnn_{tag}_v19neg.json"
json.dump(spec, open(out, "w"), ensure_ascii=False, indent=1)
print(out, spec["classes"])
tot = collections.Counter()
for (c, l), m in sorted(mins.items()): tot[c] += m
for c, m in tot.items(): print(f"  {c}: {m:.1f} 分钟")
for (c, l), m in sorted(mins.items(), key=lambda x: -x[1])[:25]: print(f"    {c} <- {l}: {m:.1f}")
