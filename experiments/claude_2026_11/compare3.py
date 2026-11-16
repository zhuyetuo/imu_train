"""Rules vs ML (RF, XGB) vs DL (int8-shaped 1D-CNN), 6-axis or acc-only, same data, same grouped CV.

Outputs (in --out):
  window_metrics.json            window-level CV metrics per method
  cont_<method>.npz              continuous out-of-fold probabilities for every file (within its labeled span)
  files.json                     per file: csv, dataset, group, t0, labeled span, GT scratch segments, labeled intervals
Then run eval_events.py on that folder (postprocess with stable v2 + event matching + FP/hour sweep).
"""
import argparse, json, os, sys, time, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = os.path.expanduser("~/imu_train")
sys.path[:0] = [os.path.join(REPO, "src", "ml"), os.path.join(REPO, "src", "data"), REPO]
import train_v2 as T  # noqa: E402

CLASSES = ["未佩戴", "静止/休息", "活动", "抓挠", "甩头/抖身"]
FINE = ["舔", "啃", "蹭", "饮水", "进食", "吠叫", "甩身体"]          # look like scratch; RF trains them separately
STILL_NAMES = {"静止/休息", "睡觉", "睡眠", "卧", "坐", "站立"}
SCALE = {"抓挠": 1.0, "甩头/抖身": 1.0}                              # no output calibration: swept later in eval
DATASETS = [("pool_20261116_all", None)]          # train + grouped CV: the whole data pool
EXTERNAL = []                                      # no external test set: train and validate on the pool only


def coarse_name(n):
    if n.startswith("抓挠"): return "抓挠"
    if n.startswith("甩头/抖身"): return "甩头/抖身"
    if n == "未佩戴": return "未佩戴"
    if n.split("-")[0] in STILL_NAMES or n in STILL_NAMES: return "静止/休息"
    return "活动"


def build_remap(datasets, nas):
    names = set()
    for name, _ in datasets:
        for t in json.load(open(f"{nas}/data_train/{name}/merged_tmp.json", encoding="utf-8")):
            for a in t.get("annotations") or []:
                for r in a.get("result") or []:
                    names |= set((r.get("value") or {}).get("timeserieslabels") or [])
    return {n: coarse_name(n) for n in names}


# ── rules ──────────────────────────────────────────────────────────────────
def rule_feats(Xm, hz):
    """per window: motion (|a| std), scratch band ratio (3-8 Hz / 0.5-hz/2 of detrended |a|), mean jerk"""
    mag = Xm[:, :, 2].astype(np.float64)
    mot = mag.std(1)
    d = mag - mag.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(d * np.hanning(d.shape[1])[None], axis=1)) ** 2
    f = np.fft.rfftfreq(d.shape[1], 1 / hz)
    band = P[:, (f >= 3) & (f <= 8)].sum(1); tot = P[:, (f >= 0.5)].sum(1) + 1e-12
    jerk = Xm[:, :, -1].mean(1)
    return np.stack([mot, band / tot, jerk], 1)


class Rule:
    """thresholds grid-searched on the training fold (window-level scratch F1); output 'probs'"""
    def fit(self, R, y):
        k = CLASSES.index("抓挠")
        best = (-1, None)
        for lo in (0.02, 0.04, 0.06, 0.1):
            for hi in (0.8, 1.5, 3.0):
                for r in (0.3, 0.4, 0.5, 0.6, 0.7):
                    pred = (R[:, 0] >= lo) & (R[:, 0] <= hi) & (R[:, 1] >= r)
                    tp = (pred & (y == k)).sum(); p = tp / max(pred.sum(), 1); rc = tp / max((y == k).sum(), 1)
                    f1 = 2 * p * rc / max(p + rc, 1e-9)
                    if f1 > best[0]: best = (f1, (lo, hi, r))
        self.lo, self.hi, self.r = best[1]
        # still / not-worn / shake thresholds: fixed physical rules
        return self

    def proba(self, R):
        mot, ratio = R[:, 0], R[:, 1]
        n = len(R); P = np.full((n, len(CLASSES)), 0.02)
        s = np.clip(0.5 + (ratio - self.r) * 2.5, 0, 1) * ((mot >= self.lo) & (mot <= self.hi))
        P[:, CLASSES.index("抓挠")] = 0.05 + 0.9 * s
        P[:, CLASSES.index("未佩戴")] += 0.9 * (mot < 0.0008)
        P[:, CLASSES.index("静止/休息")] += 0.9 * ((mot >= 0.0008) & (mot < 0.005))
        P[:, CLASSES.index("甩头/抖身")] += 0.9 * (mot > 1.5)
        P[:, CLASSES.index("活动")] += 0.9 * ((mot >= 0.005) & (mot <= 1.5)) * (1 - s)
        return P / P.sum(1, keepdims=True)


# ── CNN (same structure algo_tinyml exports as int8) ───────────────────────
CNN_INIT = None   # {"conv": state_dict, "mu", "sd"} from ssl_pretrain.py (--cnn_init)
CNN_FILTERS, CNN_KERNEL = (32, 64, 128), 5   # --cnn_filters / --cnn_kernel (same structure algo_tinyml exports)


def cnn_fit(X, y, epochs=30):
    import torch, torch.nn as nn
    dev = "cuda"; C = X.shape[2]; K = len(CLASSES)
    mu = X.reshape(-1, C).mean(0); sd = X.reshape(-1, C).std(0) + 1e-6
    L, ci = [], C
    for co in CNN_FILTERS:
        L += [nn.Conv1d(ci, co, CNN_KERNEL, padding=CNN_KERNEL // 2), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2)]; ci = co
    net = nn.Sequential(*L, nn.Flatten(), nn.Linear(ci * (X.shape[1] // 8), K)).to(dev)
    if CNN_INIT is not None:
        # same layer order as models.cnn.CNN.conv → keys "0.weight", "1.running_mean", ... line up
        missing, unexpected = net.load_state_dict(CNN_INIT["conv"], strict=False)
        assert not unexpected, unexpected
        mu, sd = np.asarray(CNN_INIT["mu"], np.float32), np.asarray(CNN_INIT["sd"], np.float32)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32).permute(0, 2, 1).to(dev)
    yt = torch.tensor(y, dtype=torch.long).to(dev)
    cnt = np.bincount(y, minlength=K).astype(float)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(np.sqrt(len(y) / (K * np.maximum(cnt, 1))), dtype=torch.float32, device=dev), label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3); bs = 512; nb = (len(y) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * nb)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(len(y), device=dev)
        for b in range(nb):
            ix = perm[b * bs:(b + 1) * bs]; xb = Xt[ix]
            xb = xb * (1 + 0.1 * torch.randn(len(ix), 1, 1, device=dev)) + 0.02 * torch.randn_like(xb)
            xb = torch.roll(xb, int(np.random.randint(-8, 9)), 2)
            loss = lossf(net(xb), yt[ix]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()

    def proba(Xe):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xe), 8192):
                xb = torch.tensor((Xe[i:i + 8192] - mu) / sd, dtype=torch.float32).permute(0, 2, 1).to(dev)
                out.append(torch.softmax(net(xb), 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, K))
    return proba


CACHE = os.path.expanduser("~/claude_exp/tl_cache")


def _cached_timeline(args):
    """load_timeline with a local-disk cache (NAS reads dominate run time). Returns (path, tl or None)."""
    import hashlib
    path, hz, axes, decl = args
    key = hashlib.md5(f"{path}|{hz}|{axes}|{decl}".encode()).hexdigest()
    f = os.path.join(CACHE, key + ".npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        if z["ok"]:
            return path, {"x": z["x"], "bad": z["bad"], "t0": z["t0"].item(), "src": float(z["src"])}
        return path, None
    try:
        tl = T.load_timeline(path, hz, decl, axes, False) if os.path.exists(path) else None
    except Exception:
        tl = None
    os.makedirs(CACHE, exist_ok=True)
    if tl is None:
        np.savez(f, ok=False)
    else:
        np.savez(f, ok=True, x=tl["x"], bad=tl["bad"], t0=np.array(tl["t0"], dtype=object), src=tl["src"])
    return path, tl


def build_tls(by_csv, hz, axes, workers):
    """timelines for all csvs (cached) + per-sample segment index arrays (same rule as train_v2._build_one)"""
    tls = {}
    args = [(p, hz, axes, ss[0].get("source_hz")) for p, ss in by_csv.items()]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, tl in ex.map(_cached_timeline, args, chunksize=4):
            if tl is None:
                continue
            n = len(tl["x"]); lab = np.full(n, -1, np.int32)
            for s in by_csv[p]:
                if s["start_ms"] is None: continue
                a0, b0 = max(int(np.ceil(s["start_ms"] / 1000 * hz)), 0), min(int(np.floor(s["end_ms"] / 1000 * hz)), n)
                if b0 > a0: lab[a0:b0] = s["idx"]
            tls[p] = dict(tl, seg=lab)
    return tls


def main(a):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    t0 = time.time(); hz = a.hz; W = a.window; S = int(round(hz)); axes = a.axes
    nas = os.path.expanduser("~/ai_data")
    remap = build_remap(DATASETS + EXTERNAL, nas)
    dsets = [{"name": n, "export_json": f"{nas}/data_train/{n}/merged_tmp.json", "source_hz": h} for n, h in DATASETS]
    segs = T.collect_segments(dsets, nas, remap, CLASSES, FINE)
    seen, uniq = set(), []
    for s in segs:                                  # same segment exported by several datasets → keep once
        key = (s["csv"], s["start_ms"], s["end_ms"], s["coarse"])
        if key not in seen: seen.add(key); uniq.append(s)
    segs = uniq
    for i, s in enumerate(segs): s["idx"] = i
    names = list(CLASSES) + sorted({s["name"] for s in segs} - set(CLASSES))
    back = np.array([CLASSES.index(n) if n in CLASSES else CLASSES.index(next(s["coarse"] for s in segs if s["name"] == n)) for n in names])
    by_csv = {}
    for s in segs: by_csv.setdefault(s["csv"], []).append(s)
    tls = build_tls(by_csv, hz, axes, a.workers)
    print(f"segments {len(segs)} csv {len(tls)}/{len(by_csv)} names {names} {time.time()-t0:.0f}s", flush=True)

    # training labels may come from another export of the same files (e.g. the auto-relabeled dataset);
    # evaluation (window metrics + continuous GT) always uses DATASETS (the original human labels)
    trs, ttls = segs, tls
    if a.train_ds:
        tsegs = T.collect_segments([{"name": a.train_ds, "export_json": f"{nas}/data_train/{a.train_ds}/merged_tmp.json",
                                     "source_hz": None}], nas, build_remap([(a.train_ds, None)], nas), CLASSES, FINE)
        seen2, trs = set(), []
        for s in tsegs:
            key = (s["csv"], s["start_ms"], s["end_ms"], s["coarse"])
            if key not in seen2 and s["csv"] in tls: seen2.add(key); trs.append(s)
        for i, s in enumerate(trs): s["idx"] = i
        ttls = {}
        for p, tl in tls.items():
            n = len(tl["x"]); lab = np.full(n, -1, np.int32)
            for s in trs:
                if s["csv"] != p or s["start_ms"] is None: continue
                a0, b0 = max(int(np.ceil(s["start_ms"] / 1000 * hz)), 0), min(int(np.floor(s["end_ms"] / 1000 * hz)), n)
                if b0 > a0: lab[a0:b0] = s["idx"]
            ttls[p] = dict(tl, seg=lab)
        extra = sorted({s["name"] for s in trs} - set(names))
        names = names + extra
        back = np.array([CLASSES.index(n) if n in CLASSES else CLASSES.index(next(s["coarse"] for s in segs + trs if s["name"] == n)) for n in names])
        print(f"train labels from {a.train_ds}: {len(trs)} segments", flush=True)

    Xtr_raw, seg_tr = T.make_windows(ttls, trs, W, S, short_shifts=2)
    Xev_raw, seg_ev = T.make_windows(tls, segs, W, S, short_shifts=0)
    Xtr, F_tr = T.featurize(Xtr_raw, hz); Xev, F_ev = T.featurize(Xev_raw, hz); del Xtr_raw, Xev_raw
    fmask = T.gravity_level_mask(Xtr.shape[2])[0]
    F_tr, F_ev = T.blank_features(F_tr, fmask), T.blank_features(F_ev, fmask)
    ni = {n: i for i, n in enumerate(names)}
    yf_tr = np.array([ni[trs[s]["name"]] for s in seg_tr]); y_ev = np.array([CLASSES.index(segs[s]["coarse"]) for s in seg_ev])
    pl_tr = np.array([trs[s]["name"] in (T.STILL, T.ACTIVE) for s in seg_tr]); pl_ev = np.array([segs[s]["name"] in (T.STILL, T.ACTIVE) for s in seg_ev])
    yf_tr, _ = T.still_by_motion(yf_tr, T.motion_of(Xtr), pl_tr, ni[T.STILL], ni[T.ACTIVE], 0.005)
    y_ev, _ = T.still_by_motion(y_ev, T.motion_of(Xev), pl_ev, CLASSES.index(T.STILL), CLASSES.index(T.ACTIVE), 0.005)
    yc_tr = back[yf_tr]
    g_tr = np.array([T._group_of(trs[s]["sample_code"]) for s in seg_tr]); g_ev = np.array([T._group_of(segs[s]["sample_code"]) for s in seg_ev])
    R_tr, R_ev = rule_feats(Xtr, hz), rule_feats(Xev, hz)
    print(f"windows train {len(yc_tr)} eval {len(y_ev)} groups {len(set(g_ev))} {time.time()-t0:.0f}s", flush=True)

    # files for continuous eval: windows inside each task's labeled span
    files = []
    for p, tl in tls.items():
        ss = [s for s in by_csv[p] if s["start_ms"] is not None]
        if not ss: continue
        lo = min(s["start_ms"] for s in ss) / 1000; hi = max(s["end_ms"] for s in ss) / 1000
        files.append({"csv": p, "dataset": ss[0]["dataset"], "group": T._group_of(ss[0]["sample_code"]),
                      "t0": tl["t0"].strftime("%Y-%m-%d %H:%M:%S.%f"), "span": [lo, hi],
                      "gt": sorted([(s["start_ms"] / 1000, s["end_ms"] / 1000) for s in ss if s["coarse"] == "抓挠"]),
                      "labeled": sorted([(s["start_ms"] / 1000, s["end_ms"] / 1000) for s in ss]),
                      "labeled_other": sorted([(s["start_ms"] / 1000, s["end_ms"] / 1000) for s in ss if s["coarse"] != "抓挠"])})

    methods = a.methods.split(",")
    P_ev = {m: np.zeros((len(y_ev), len(CLASSES))) for m in methods}
    cont = {m: {} for m in methods}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for k, (_, te) in enumerate(cv.split(F_ev, y_ev, g_ev)):
        test_g = set(g_ev[te]); tr = ~np.isin(g_tr, list(test_g))
        pred = {}
        rule = Rule().fit(R_tr[tr], yc_tr[tr])
        if "rule" in methods:
            pred["rule"] = lambda X, F, R, rule=rule: rule.proba(R)
        if "rf" in methods:
            rf = T.merge_forest(RandomForestClassifier(n_estimators=20, max_depth=8, max_features="sqrt", class_weight="balanced",
                                                       n_jobs=a.n_jobs, random_state=42).fit(F_tr[tr], yf_tr[tr]),
                                names, CLASSES, list(back), np.ones(len(CLASSES)))
            pred["rf"] = lambda X, F, R, rf=rf: rf.predict_proba(F)
        if "xgb" in methods:
            xg, xc = T.fit_xgb(F_tr[tr], yc_tr[tr], len(CLASSES), T.xgb_params({"n_estimators": 50, "max_depth": 6}), a.n_jobs)
            pred["xgb"] = lambda X, F, R, xg=xg, xc=xc: T.xgb_proba(xg, xc, F, len(CLASSES))
        if "cnn" in methods:
            cnn = cnn_fit(Xtr[tr], yc_tr[tr], a.epochs)
            pred["cnn"] = lambda X, F, R, cnn=cnn: cnn(X)
        for m in methods:
            P_ev[m][te] = pred[m](Xev[te], F_ev[te], R_ev[te])
        for f in files:
            if f["group"] not in test_g: continue
            tl = tls[f["csv"]]; x, bad = tl["x"], tl["bad"]
            a0 = max(0, int(f["span"][0] * hz) - W // 2); a1 = min(len(x) - W, int(f["span"][1] * hz))
            st = np.array([s for s in range(a0, max(a0, a1) + 1, S) if not bad[s:s + W].any()], int)
            if len(st) == 0: continue
            Xc, Fc = T.featurize(np.stack([x[s:s + W] for s in st]).astype(np.float32), hz)
            Fc = T.blank_features(Fc, fmask); Rc = rule_feats(Xc, hz)
            for m in methods:
                cont[m][f["csv"]] = (st, pred[m](Xc, Fc, Rc).astype(np.float16))
        print(f"fold {k+1}/5 done rule=({rule.lo},{rule.hi},{rule.r}) {time.time()-t0:.0f}s", flush=True)

    # external test set: train every method on ALL pool windows, predict the external files' labeled spans
    if EXTERNAL:
        ext_ds = [{"name": n, "export_json": f"{nas}/data_train/{n}/merged_tmp.json", "source_hz": h} for n, h in EXTERNAL]
        esegs = T.collect_segments(ext_ds, nas, remap, CLASSES, FINE)
        e_by = {}
        for s in esegs:
            e_by.setdefault(s["csv"], []).append(s)
        etls = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for p, tl, err in ex.map(T._build_one, [(p, [dict(s, idx=0) for s in ss], hz, axes, False) for p, ss in e_by.items()], chunksize=4):
                if tl is not None: etls[p] = tl
        allm = np.ones(len(yc_tr), bool)
        rule = Rule().fit(R_tr, yc_tr)
        rf = T.merge_forest(RandomForestClassifier(n_estimators=20, max_depth=8, max_features="sqrt", class_weight="balanced",
                                                   n_jobs=a.n_jobs, random_state=42).fit(F_tr, yf_tr),
                            names, CLASSES, list(back), np.ones(len(CLASSES)))
        xg, xc = T.fit_xgb(F_tr, yc_tr, len(CLASSES), T.xgb_params({"n_estimators": 50, "max_depth": 6}), a.n_jobs)
        cnn = cnn_fit(Xtr, yc_tr, a.epochs)
        pred = {"rule": lambda X, F, R: rule.proba(R), "rf": lambda X, F, R: rf.predict_proba(F),
                "xgb": lambda X, F, R: T.xgb_proba(xg, xc, F, len(CLASSES)), "cnn": lambda X, F, R: cnn(X)}
        n_overlap = 0
        for p, tl in etls.items():
            ss = [s for s in e_by[p] if s["start_ms"] is not None]
            if not ss: continue
            if p in tls:                    # same raw file also in the pool → trained on it, not a clean test
                n_overlap += 1
                continue
            f = {"csv": p, "dataset": ss[0]["dataset"], "group": "external", "external": True,
                 "t0": tl["t0"].strftime("%Y-%m-%d %H:%M:%S.%f"),
                 "span": [min(s["start_ms"] for s in ss) / 1000, max(s["end_ms"] for s in ss) / 1000],
                 "gt": sorted([(s["start_ms"] / 1000, s["end_ms"] / 1000) for s in ss if s["coarse"] == "抓挠"]),
                 "labeled": sorted([(s["start_ms"] / 1000, s["end_ms"] / 1000) for s in ss]),
                 "labeled_other": sorted([(s["start_ms"] / 1000, s["end_ms"] / 1000) for s in ss if s["coarse"] != "抓挠"])}
            x, bad = tl["x"], tl["bad"]
            a0 = max(0, int(f["span"][0] * hz) - W // 2); a1 = min(len(x) - W, int(f["span"][1] * hz))
            st = np.array([s for s in range(a0, max(a0, a1) + 1, S) if not bad[s:s + W].any()], int)
            if len(st) == 0: continue
            files.append(f)
            Xc, Fc = T.featurize(np.stack([x[s:s + W] for s in st]).astype(np.float32), hz)
            Fc = T.blank_features(Fc, fmask); Rc = rule_feats(Xc, hz)
            for m in methods:
                cont[m][p] = (st, pred[m](Xc, Fc, Rc).astype(np.float16))
        print(f"external {len(etls)} files ({n_overlap} skipped: also in pool) done {time.time()-t0:.0f}s", flush=True)

    os.makedirs(a.out, exist_ok=True)
    wm = {}
    k = CLASSES.index("抓挠"); ds_ev = np.array([segs[s]["dataset"] for s in seg_ev])
    for m in methods:
        pr = P_ev[m].argmax(1)
        tp = ((pr == k) & (y_ev == k)).sum(); P = tp / max((pr == k).sum(), 1); Rc = tp / max((y_ev == k).sum(), 1)
        wm[m] = {"acc": round(float((pr == y_ev).mean()), 4), "macro_f1": round(float(f1_score(y_ev, pr, average="macro")), 4),
                 "f1": {c: round(float(v), 3) for c, v in zip(CLASSES, f1_score(y_ev, pr, labels=range(len(CLASSES)), average=None, zero_division=0))},
                 "scratch_window": {"P": round(float(P), 3), "R": round(float(Rc), 3)},
                 "per_dataset_acc": {d: round(float((pr[ds_ev == d] == y_ev[ds_ev == d]).mean()), 4) for d in sorted(set(ds_ev))}}
        np.savez_compressed(os.path.join(a.out, f"cont_{m}.npz"),
                            **{f"s{i}": cont[m][f["csv"]][0] for i, f in enumerate(files) if f["csv"] in cont[m]},
                            **{f"p{i}": cont[m][f["csv"]][1] for i, f in enumerate(files) if f["csv"] in cont[m]})
    json.dump(wm, open(os.path.join(a.out, "window_metrics.json"), "w"), ensure_ascii=False, indent=1)
    json.dump({"hz": hz, "W": W, "S": S, "axes": axes, "classes": CLASSES, "files": files},
              open(os.path.join(a.out, "files.json"), "w"), ensure_ascii=False)
    print(json.dumps(wm, ensure_ascii=False, indent=1)); print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--axes", type=int, default=6); p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--window", type=int, default=80); p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=12); p.add_argument("--n_jobs", type=int, default=24)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train_ds", default="", help="take training labels from this export (eval stays on DATASETS)")
    p.add_argument("--methods", default="rule,rf,xgb,cnn")
    p.add_argument("--cnn_init", default="", help="ssl_pretrain.py checkpoint to initialize the CNN conv stack")
    p.add_argument("--cnn_filters", default="32,64,128"); p.add_argument("--cnn_kernel", type=int, default=5)
    args = p.parse_args()
    CNN_FILTERS = tuple(int(v) for v in args.cnn_filters.split(",")); CNN_KERNEL = args.cnn_kernel
    if args.cnn_init:
        import torch
        CNN_INIT = torch.load(args.cnn_init, map_location="cpu", weights_only=False)
        print("cnn init from", args.cnn_init, {k: CNN_INIT[k] for k in ("cfg", "n_windows", "n_files")}, flush=True)
    main(args)
