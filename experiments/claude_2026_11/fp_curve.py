"""同样误报下比命中：抓挠概率乘一个系数 k（再归一化）→ 稳定版 v2 → 片段级命中 / 误报，扫一串 k 画出命中-误报曲线，
再插值出误报 = 2 / 3 / 4 / 5 次/h 时各模型的命中率。只算自己的项圈数据。
用法：python fp_curve.py out.json res_q01_pred.pkl res_q05_pred.pkl ..."""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np

sys.path[:0] = ["/home/toky/imu_train/src/ml", "/home/toky/imu_train/src/data", "/home/toky/imu_train"]
import train_v2 as T  # noqa: E402
from label_service import postprocess as pp  # noqa: E402

OWN = "pool_20261116_all"
KS = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0, 4.5]


def one(args):
    pkl, k = args
    from boundary_eval import params
    d = pickle.load(open(pkl, "rb"))
    classes, W, S, hz = d["classes"], d["W"], d["S"], d["hz"]
    ci = classes.index("抓挠")
    sp = params()
    targets = [c for c in T.EVENT_EVAL if c in classes]
    out = []
    for f in d["files"]:
        if f["dataset"] != OWN:
            continue
        P = f["P"].astype(np.float64).copy()
        P[:, ci] *= k
        P /= P.sum(1, keepdims=True)
        wins = [{"ts": pp._fmt_ts(f["t0"] + timedelta(seconds=float(s) / hz)), "label": classes[int(q.argmax())],
                 "conf": float(q.max()), "probs": {c: float(v) for c, v in zip(classes, q)}} for s, q in zip(f["starts"], P)]
        segs = pp.stabilize(wins, classes, targets, W / hz, S / hz, "segment", sp, algo="viterbi")
        pred = {c: [((pp._parse_ts(x["start_ts"]) - f["t0"]).total_seconds(), (pp._parse_ts(x["end_ts"]) - f["t0"]).total_seconds())
                    for x in segs.get(c, [])] for c in targets}
        out.append({"gt": f["gt"], "other": f["other"], "pred": pred})
    m = T.segment_metrics(out, classes)["抓挠"]
    return pkl, k, m["recall_hit"], m["fp_per_h_other"], m["precision"]


def main(out_path, pkls):
    sys.path.insert(0, ".")
    jobs = [(p, k) for p in pkls for k in KS]
    rows = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
            print(f"{r[0]:<22} k={r[1]:<5} 命中 {r[2]:.1%}  误报 {r[3]:.2f}/h  查准 {r[4]:.1%}", flush=True)
    res = {}
    for p in pkls:
        pts = sorted([(fp, hit, k) for pp_, k, hit, fp, _ in rows if pp_ == p])
        fps, hits = [x[0] for x in pts], [x[1] for x in pts]
        res[p] = {"curve": [{"k": k, "hit": h, "fp_h": f} for f, h, k in pts],
                  "hit_at_fp": {str(t): round(float(np.interp(t, fps, hits)), 4) for t in (2, 3, 4, 5, 6)}}
    print("\n同样误报（次/h，只算自己数据）下的抓挠命中：")
    print(f"{'模型':<22}" + "".join(f"{'误报' + str(t):>10}" for t in (2, 3, 4, 5, 6)))
    for p, r in res.items():
        print(f"{p:<22}" + "".join(f"{r['hit_at_fp'][str(t)]:>10.1%}" for t in (2, 3, 4, 5, 6)))
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
