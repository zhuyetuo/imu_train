"""第二轮：按段长 + 段内平均概率 + 陀螺仪波动卡（第一轮看出真抓挠长、概率高；误报短、概率低）。
「全部查准」把未标时间的预报都当错（保守下限），「确定查准」只算人看过的时间（上限）"""
import itertools
import json
import sys

sys.path.insert(0, "/home/toky/claude_exp/evt")
import evt_eval as E  # noqa: E402

E.load(sys.argv[1])
out = sys.argv[2]
grid = {"scale": [1.0, 0.7], "mean_min": [0, 0.6, 0.7, 0.75, 0.8], "len_min": [0, 3, 4, 5, 6],
        "gstd_min": [0, 30, 45], "event_enter": [0.5, 0.65]}
keys = list(grid)
with open(out, "w", encoding="utf-8") as fo:
    for vals in itertools.product(*grid.values()):
        opt = dict(zip(keys, vals))
        m, what, sites = E.evaluate(opt)
        fo.write(json.dumps({"opt": opt, **m, "fp_what": dict(what.most_common(6)),
                             "sites": {k: {x: v[x] for x in ("prec_all", "prec_known", "recall_loose")} for k, v in sites.items()}},
                            ensure_ascii=False) + "\n")
        fo.flush()
print("done")
