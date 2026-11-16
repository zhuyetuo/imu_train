"""Event-level scratch comparison on continuous predictions (compare3 output dir).
Each method → stable v2 postprocess (viterbi, platform params) → scratch segments → vs human scratch segments.

Pool files (grouped-CV, out-of-fold):
  cv_recall_hit : human scratch overlapped by a predicted segment
  cv_recall_iou : one-to-one, IoU >= 0.5
  cv_FP_other_h : predicted segments that overlap time labeled as something else and no human scratch,
                  per hour of time labeled as something else (pool is sparse, so this is the only FP we can judge there)
External files (densely labeled, models trained on the whole pool, never seen):
  ext_recall_hit, ext_FP_h (predicted segments inside the labeled span overlapping no human scratch, per hour of span),
  ext_prec (share of predicted segments in span that overlap a human scratch)
Sweeps a scratch-probability scale so methods can be compared at the same false-alarm level.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import numpy as np

REPO = os.path.expanduser("~/imu_train")
sys.path[:0] = [REPO]
G = {}


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def ov(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


def _init(d, method, geom):
    from label_service import postprocess as pp, config
    G.update(pp=pp, geom=geom, meta=json.load(open(os.path.join(d, "files.json"), encoding="utf-8")),
             Z=np.load(os.path.join(d, f"cont_{method}.npz")),
             sp=pp.StableParams(event_labels=tuple(config.STABLE_EVENT_LABELS), smooth_windows=config.STABLE_SMOOTH_WINDOWS,
                                min_state_s=config.STABLE_MIN_STATE_S, event_enter=config.STABLE_EVENT_ENTER,
                                event_stay=config.STABLE_EVENT_STAY, event_gap_s=config.STABLE_EVENT_GAP_S,
                                shake_absorb_s=config.STABLE_SHAKE_ABSORB_S, event_min_windows=config.STABLE_EVENT_MIN_WINDOWS,
                                event_min_mean=config.STABLE_EVENT_MIN_MEAN, event_single_conf=config.STABLE_EVENT_SINGLE_CONF,
                                spectral_min=config.STABLE_SPECTRAL_MIN, viterbi_switch=config.STABLE_VITERBI_SWITCH,
                                notworn_min_s=config.STABLE_NOTWORN_MIN_S))


def one(args):
    i, scales = args
    pp, g, f, Z = G["pp"], G["geom"], G["meta"]["files"][i], G["Z"]
    if f"s{i}" not in Z.files: return i, None
    st, P = Z[f"s{i}"], Z[f"p{i}"].astype(np.float64)
    classes = g["classes"]; k = classes.index("抓挠"); hz, W, S = g["hz"], g["W"], g["S"]
    t0 = datetime.strptime(f["t0"], "%Y-%m-%d %H:%M:%S.%f")
    ts = [pp._fmt_ts(t0 + timedelta(seconds=s / hz)) for s in st]
    res = {}
    for sc in scales:
        Q = P.copy(); Q[:, k] *= sc; Q /= Q.sum(1, keepdims=True)
        wins = [{"ts": t, "label": classes[int(q.argmax())], "conf": float(q.max()),
                 "probs": {c: float(v) for c, v in zip(classes, q)}} for t, q in zip(ts, Q)]
        out = pp.stabilize(wins, classes, ["抓挠"], W / hz, S / hz, g["label_mode"], G["sp"], algo="viterbi")
        res[sc] = [((pp._parse_ts(s["start_ts"]) - t0).total_seconds(), (pp._parse_ts(s["end_ts"]) - t0).total_seconds())
                   for s in out.get("抓挠", [])]
    return i, res


def match_iou(gt, pred):
    pairs = sorted([(iou(g_, p), a, b) for a, g_ in enumerate(gt) for b, p in enumerate(pred) if iou(g_, p) >= 0.5], reverse=True)
    ug, up, n = set(), set(), 0
    for _, a, b in pairs:
        if a not in ug and b not in up: ug.add(a); up.add(b); n += 1
    return n


def main(a):
    meta = json.load(open(os.path.join(a.dir, "files.json"), encoding="utf-8"))
    rows = {}
    for m in a.methods.split(","):
        if m == "stable":
            geom = json.load(open(os.path.join(a.dir, "stable_geom.json"), encoding="utf-8"))
        else:
            geom = {"hz": meta["hz"], "W": meta["W"], "S": meta["S"], "label_mode": "segment", "classes": meta["classes"]}
        scales = [float(x) for x in a.scales.split(",")]
        preds = {}
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.dir, m, geom)) as ex:
            for i, r in ex.map(one, [(i, scales) for i in range(len(meta["files"]))], chunksize=8):
                if r is not None: preds[i] = r
        for sc in scales:
            c = dict(cv_hit=0, cv_iou=0, cv_gt=0, cv_fp=0, cv_other_h=0.0, ex_hit=0, ex_gt=0, ex_fp=0, ex_in=0, ex_tp=0, ex_h=0.0)
            for i, f in enumerate(meta["files"]):
                if i not in preds: continue
                gt, pr = [tuple(x) for x in f["gt"]], preds[i][sc]
                hit = sum(any(ov(g_, p) for p in pr) for g_ in gt)
                if f.get("external"):
                    c["ex_hit"] += hit; c["ex_gt"] += len(gt)
                    lo, hi = f["span"]; c["ex_h"] += (hi - lo) / 3600
                    for p in pr:
                        if p[1] < lo or p[0] > hi: continue
                        c["ex_in"] += 1
                        if any(ov(p, g_) for g_ in gt): c["ex_tp"] += 1
                        else: c["ex_fp"] += 1
                else:
                    c["cv_hit"] += hit; c["cv_gt"] += len(gt); c["cv_iou"] += match_iou(gt, pr)
                    other = [tuple(x) for x in f.get("labeled_other", [])]
                    c["cv_other_h"] += sum(b - a_ for a_, b in other) / 3600
                    c["cv_fp"] += sum(1 for p in pr if any(ov(p, o) for o in other) and not any(ov(p, g_) for g_ in gt))
            r = {"cv_recall_hit": round(c["cv_hit"] / max(c["cv_gt"], 1), 3), "cv_recall_iou": round(c["cv_iou"] / max(c["cv_gt"], 1), 3),
                 "cv_FP_other_h": round(c["cv_fp"] / max(c["cv_other_h"], 1e-9), 2), "cv_gt": c["cv_gt"], "cv_other_h": round(c["cv_other_h"], 1)}
            if c["ex_h"] > 0:
                r.update(ext_recall_hit=round(c["ex_hit"] / max(c["ex_gt"], 1), 3), ext_FP_h=round(c["ex_fp"] / c["ex_h"], 2),
                         ext_prec=round(c["ex_tp"] / max(c["ex_in"], 1), 3), ext_gt=c["ex_gt"], ext_h=round(c["ex_h"], 1))
            rows[(m, sc)] = r
            print(m, sc, json.dumps(r, ensure_ascii=False), flush=True)
    json.dump({f"{m}@{sc}": v for (m, sc), v in rows.items()}, open(os.path.join(a.dir, "events.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dir", required=True)
    p.add_argument("--methods", default="stable,rule,rf,xgb,cnn"); p.add_argument("--scales", default="0.5,0.75,1,1.5,2.5")
    p.add_argument("--workers", type=int, default=20)
    main(p.parse_args())
