"""抓挠的事件级评估：模型在**连续数据**上推理 → 稳定版 v2 后处理出片段 → 跟人标的片段一一配对。

为什么要有这个：train_v2 里「抓挠按次数 查全 / 查准」是拿**人已经切好起止**的片段问模型
「这段是不是抓挠」，等于先把边界告诉了模型，偏乐观。这里不告诉它边界：
  1. 跟 train_v2 一样按（天, 项圈）分组 5 折，每折训一个模型（同一套特征、同样的开关）；
  2. 被留出的那些 CSV 整条时间线按步长滑窗推理（不只是标了的那几段），
     逐窗口概率交给平台同一份 label_service/postprocess.stabilize（默认 viterbi = 稳定版 v2）；
  3. 出来的抓挠片段跟人标的抓挠片段一对一配对（贪心，IoU 从大到小），算
       查全 = 配上的人标片段 / 人标片段数，查准 = 配上的预测片段 / 预测片段数，
     以及配上的那些开始 / 结束时间差了几秒、每个文件次数差多少。

配对规则给两档：IoU ≥ 0.5（重叠够多），以及「开始、结束都在 ±tol 秒内」（边界准）。
误报只算落在**标注过的时间里**的预测片段（跟任何人标片段都不重叠、但也不在任何标注里的，
单独计成「落在没标的地方」——那里没标不等于没抓，冤枉不得）。

用法（spec 跟 train_v2 同一种 JSON，model=xgb / rf）：
    python src/eval/scratch_event_cv.py --spec tmp/train_v2_xgb_edge.json --out tmp/scratch_events.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path[:0] = [os.path.join(_REPO, "src", "ml"), os.path.join(_REPO, "src", "data"), _REPO]

import train_v2 as T  # noqa: E402

SCRATCH = "抓挠"


def _stable_params(pp):
    from label_service import config
    return pp.StableParams(
        event_labels=tuple(config.STABLE_EVENT_LABELS), smooth_windows=config.STABLE_SMOOTH_WINDOWS,
        min_state_s=config.STABLE_MIN_STATE_S, event_enter=config.STABLE_EVENT_ENTER,
        event_stay=config.STABLE_EVENT_STAY, event_gap_s=config.STABLE_EVENT_GAP_S,
        shake_absorb_s=config.STABLE_SHAKE_ABSORB_S, event_min_windows=config.STABLE_EVENT_MIN_WINDOWS,
        event_min_mean=config.STABLE_EVENT_MIN_MEAN, event_single_conf=config.STABLE_EVENT_SINGLE_CONF,
        spectral_min=config.STABLE_SPECTRAL_MIN, viterbi_switch=config.STABLE_VITERBI_SWITCH,
        notworn_min_s=config.STABLE_NOTWORN_MIN_S)


def continuous_windows(tl: dict, W: int, S: int):
    """整条时间线按步长切窗口（跳过含掉包的），返回 (窗口 [N,W,C], 起点下标 [N])"""
    x, bad = tl["x"], tl["bad"]
    starts = [st for st in range(0, len(x) - W + 1, S) if not bad[st:st + W].any()]
    if not starts:
        return np.zeros((0, W, x.shape[1]), np.float32), np.zeros(0, int)
    return np.stack([x[st:st + W] for st in starts]).astype(np.float32), np.asarray(starts)


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(gt, pred, rule, tol):
    """贪心一对一：候选对按 IoU 从大到小，各自没被占用就配上。rule=iou / bound"""
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            v = iou(g, p)
            if v <= 0:
                continue
            ok = v >= 0.5 if rule == "iou" else (abs(g[0] - p[0]) <= tol and abs(g[1] - p[1]) <= tol)
            if ok:
                pairs.append((v, i, j))
    used_g, used_p, out = set(), set(), []
    for v, i, j in sorted(pairs, reverse=True):
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            out.append((i, j))
    return out


def main(a):
    from sklearn.model_selection import StratifiedGroupKFold
    from label_service import postprocess as pp

    spec = json.load(open(a.spec, encoding="utf-8"))
    hz = float(spec["hz"])
    axes = 6 if int(spec.get("axes") or 3) == 6 else 3
    classes = list(spec["classes"])
    kind = str(spec.get("model") or "rf")
    fine = [] if kind == "xgb" else (spec.get("fine_labels") if spec.get("fine_labels") is not None else T.DEFAULT_FINE)
    scale = np.array([float((spec.get("class_scale") or {}).get(c, 1.0)) for c in classes])
    W = int(spec["window"])
    S = max(1, int(round(float(spec.get("stride_s") or 1.0) * hz)))
    params = spec.get("model_params") or {}
    drop_level = bool(spec.get("drop_gravity_level"))
    still_g = float(spec.get("still_motion_g") or 0) or None
    t0 = time.time()

    segs = T.collect_segments(spec["datasets"], spec["nas_root"], spec.get("label_remap") or {}, classes, fine)
    for i, s in enumerate(segs):
        s["idx"] = i
    names = list(classes) + sorted({s["name"] for s in segs} - set(classes))
    back = np.array([classes.index(n) if n in classes else
                     classes.index(next(s["coarse"] for s in segs if s["name"] == n)) for n in names])
    by_csv = {}
    for s in segs:
        by_csv.setdefault(s["csv"], []).append(s)
    tls = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for p, tl, err in ex.map(T._build_one, [(p, ss, hz, axes, bool(spec.get("upsample"))) for p, ss in by_csv.items()], chunksize=4):
            if tl is not None:
                tls[p] = tl
    print(f"▶ {len(segs)} 段，{len(tls)} 个 CSV，{time.time() - t0:.0f}s", flush=True)

    # 训练窗口（跟 train_v2 完全一样）
    Xtr_raw, seg_tr = T.make_windows(tls, segs, W, S, short_shifts=2)
    Xev_raw, seg_ev = T.make_windows(tls, segs, W, S, short_shifts=0)
    Xtr_m, F_tr = T.featurize(Xtr_raw, hz)
    Xev_m, _ = T.featurize(Xev_raw, hz)
    fmask = T.gravity_level_mask(Xtr_m.shape[2])[0] if drop_level else None
    if fmask is not None:
        F_tr = T.blank_features(F_tr, fmask)
    name_idx = {n: i for i, n in enumerate(names)}
    yf_tr = np.array([name_idx[segs[s]["name"]] for s in seg_tr])
    y_ev = np.array([classes.index(segs[s]["coarse"]) for s in seg_ev])
    if still_g:
        plain = np.array([segs[s]["name"] in (T.STILL, T.ACTIVE) for s in seg_tr])
        yf_tr, _ = T.still_by_motion(yf_tr, T.motion_of(Xtr_m), plain, name_idx[T.STILL], name_idx[T.ACTIVE], still_g)
    g_tr = np.array([T._group_of(segs[s]["sample_code"]) for s in seg_tr])
    g_ev = np.array([T._group_of(segs[s]["sample_code"]) for s in seg_ev])
    csv_group = {p: T._group_of(ss[0]["sample_code"]) for p, ss in by_csv.items()}

    sp = _stable_params(pp)
    cv = StratifiedGroupKFold(n_splits=min(5, len(set(g_ev))), shuffle=True, random_state=0)
    per_file = []
    for k, (_, te) in enumerate(cv.split(np.zeros(len(y_ev)), y_ev, g_ev)):
        test_g = set(g_ev[te])
        tr = ~np.isin(g_tr, list(test_g))
        if kind == "xgb":
            m, mc = T.fit_xgb(F_tr[tr], back[yf_tr[tr]], len(classes), T.xgb_params(params), a.n_jobs)
            predict = lambda F, m=m, mc=mc: T.xgb_proba(m, mc, F, len(classes))  # noqa: E731
        else:
            from sklearn.ensemble import RandomForestClassifier
            rf_kw = dict(n_estimators=int(params.get("n_estimators") or 20), max_depth=params.get("max_depth", 8),
                         max_features="sqrt", class_weight="balanced", n_jobs=a.n_jobs, random_state=42)
            m = T.merge_forest(RandomForestClassifier(**rf_kw).fit(F_tr[tr], yf_tr[tr]), names, classes, list(back), np.ones(len(classes)))
            predict = m.predict_proba  # noqa: E731
        for p, tl in tls.items():
            if csv_group[p] not in test_g:
                continue
            X, starts = continuous_windows(tl, W, S)
            if len(X) == 0:
                continue
            _, F = T.featurize(X, hz)
            if fmask is not None:
                F = T.blank_features(F, fmask)
            pr = predict(F) * scale[None, :]
            pr /= pr.sum(1, keepdims=True)
            wins = [{"ts": pp._fmt_ts(tl["t0"] + timedelta(seconds=st / hz)), "label": classes[int(q.argmax())],
                     "conf": float(q.max()), "probs": {c: float(v) for c, v in zip(classes, q)}}
                    for st, q in zip(starts, pr)]
            out = pp.stabilize(wins, classes, [SCRATCH], W / hz, S / hz, "segment", sp, algo=a.algo)
            pred = [((pp._parse_ts(s_["start_ts"]) - tl["t0"]).total_seconds(),
                     (pp._parse_ts(s_["end_ts"]) - tl["t0"]).total_seconds()) for s_ in out.get(SCRATCH, [])]
            ann = [(s["start_ms"] / 1000, s["end_ms"] / 1000, s["coarse"]) for s in by_csv[p] if s["start_ms"] is not None]
            gt = [(x0, x1) for x0, x1, c in ann if c == SCRATCH]
            per_file.append({"csv": p, "fold": k, "gt": gt, "pred": pred,
                             "annotated": [(x0, x1) for x0, x1, _ in ann],
                             "timeline_s": len(tl["x"]) / hz})
        print(f"  第 {k + 1} 折完成（{time.time() - t0:.0f}s）", flush=True)

    report = {"spec": a.spec, "algo": a.algo, "model": kind, "files": len(per_file)}
    cov = [sum(x1 - x0 for x0, x1 in f["annotated"]) / max(f["timeline_s"], 1) for f in per_file]
    report["annotated_share_median"] = round(float(np.median(cov)), 3) if cov else None
    for rule in ("iou", "bound"):
        tp = n_gt = n_pred = n_unann = 0
        on, off = [], []
        for f in per_file:
            gt, pred = f["gt"], f["pred"]
            mt = match(gt, pred, rule, a.tol)
            tp += len(mt)
            n_gt += len(gt)
            matched_p = {j for _, j in mt}
            for j, p_ in enumerate(pred):
                if j in matched_p:
                    n_pred += 1
                elif any(iou(p_, z) > 0 or (z[0] <= p_[0] and p_[1] <= z[1]) for z in f["annotated"]):
                    n_pred += 1          # 落在标注过的时间里、没配上 → 误报
                else:
                    n_unann += 1         # 落在没标的地方：不算对也不算错
            for i, j in mt:
                on.append(pred[j][0] - gt[i][0])
                off.append(pred[j][1] - gt[i][1])
        rec, prec = tp / max(n_gt, 1), tp / max(n_pred, 1)
        report[rule] = {
            "recall": round(rec, 4), "precision": round(prec, 4),
            "f1": round(2 * rec * prec / max(rec + prec, 1e-9), 4),
            "gt": n_gt, "pred_in_annotated": n_pred, "matched": tp, "pred_unannotated": n_unann,
            "onset_err_s": {"median_abs": round(float(np.median(np.abs(on))), 2) if on else None,
                            "p90_abs": round(float(np.percentile(np.abs(on), 90)), 2) if on else None,
                            "median_signed": round(float(np.median(on)), 2) if on else None},
            "offset_err_s": {"median_abs": round(float(np.median(np.abs(off))), 2) if off else None,
                             "p90_abs": round(float(np.percentile(np.abs(off), 90)), 2) if off else None,
                             "median_signed": round(float(np.median(off)), 2) if off else None},
        }
    cnt = [len(f["pred"]) - len(f["gt"]) for f in per_file if f["gt"]]
    report["count_diff_per_file"] = {"median": float(np.median(cnt)) if cnt else None,
                                     "mean_abs": round(float(np.mean(np.abs(cnt))), 2) if cnt else None}
    for min_s in (2.0, 3.0):
        tp = n_gt = 0
        for f in per_file:
            gt = [g for g in f["gt"] if g[1] - g[0] >= min_s]
            tp += len(match(gt, f["pred"], "iou", a.tol))
            n_gt += len(gt)
        report[f"iou_recall_gt_ge_{int(min_s)}s"] = round(tp / max(n_gt, 1), 4)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"report": report, "per_file": per_file}, f, ensure_ascii=False, indent=1)
    print(f"▶ 用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--algo", default="viterbi", help="viterbi（稳定版 v2）/ stable / 加 _noshake")
    p.add_argument("--tol", type=float, default=2.0, help="边界档：开始、结束各允许差几秒")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n_jobs", type=int, default=16)
    main(p.parse_args())
