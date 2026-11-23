"""几个模型的交叉验证导出放在同一把尺子上比抓挠事件级：同一套后处理（平台稳定版 v2），扫段内平均概率下限，
再在相同召回下比查准。口径见 evt_eval.py。用法：python evt_cmp.py base v1 v2b v3"""
import json
import os
import sys

import evt_eval as E

HERE = os.path.dirname(os.path.abspath(__file__))
MEANS = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
out = open(os.path.join(HERE, "cmp.jsonl"), "a", encoding="utf-8")
res = {}
for name in sys.argv[1:]:
    path = os.path.join(HERE, f"{name}_seg.pkl.gz")
    if not os.path.exists(path):
        print(name, "没有导出，跳过", flush=True)
        continue
    E.load(path)
    print(f"== {name}: {len(E.D['files'])} 个文件，窗口 {E.D['W'] / E.D['hz']:.0f}s 步长 {E.D['S'] / E.D['hz']:.1f}s", flush=True)
    res[name] = []
    for mm in MEANS:
        m, what, _ = E.evaluate({"event_min_mean": mm})
        res[name].append((mm, m))
        out.write(json.dumps({"model": name, "event_min_mean": mm, **m, "fp_what": what.most_common(6)}, ensure_ascii=False) + "\n")
        out.flush()
        print(f"  平均概率≥{mm:.2f}: 召回 {m['recall_loose']:.3f} 确定查准 {m['prec_known']:.3f} 全部查准 {m['prec_all']:.3f} "
              f"预测 {m['n_pred']} 确定误报 {m['fp']} 未标 {m['unlabeled']}  严格 R/P {m['recall_iou50']:.3f}/{m['prec_iou50']:.3f}"
              f"  误报落在 {what.most_common(4)}", flush=True)


def at_recall(rows, r):
    """召回 ≥ r 的里面查准最高的那一档（没有就取召回最高的）"""
    ok = [x for x in rows if x[1]["recall_loose"] >= r]
    return max(ok, key=lambda x: x[1]["prec_known"]) if ok else max(rows, key=lambda x: x[1]["recall_loose"])


print("\n相同召回下比（每个模型取召回 ≥ 目标里确定查准最高的一档）：")
for r in (0.80, 0.85, 0.88):
    print(f"  召回 ≥ {r}:")
    for name, rows in res.items():
        mm, m = at_recall(rows, r)
        print(f"    {name:5s} 平均概率≥{mm:.2f}  召回 {m['recall_loose']:.3f} 确定查准 {m['prec_known']:.3f} 全部查准 {m['prec_all']:.3f} 确定误报 {m['fp']} 未标 {m['unlabeled']}")
