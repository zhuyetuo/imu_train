"""抓挠事件级：离线评估 + 扫后处理参数。读 train_v2 的 TRAIN_V2_DUMP_SEG 导出。

口径（用户要的：起止差一些、长短差一些都能接受，关键是别把不是抓挠的报成抓挠）：
  - 预测事件「对」：跟人标抓挠重叠 ≥ OV_S 秒
  - 「确定的误报」：不对，且跟人标的其他动作有重叠（那段时间人看过，说不是抓挠）
  - 「未标时间」：不对，也不碰任何人标片段（没人看过，不知道对错），单独数
  - 确定查准 = 对 / (对 + 确定误报)；宽松查全 = 被任何预测重叠 ≥ OV_S 秒的人标抓挠 / 全部人标抓挠
  - 另报严格口径（同类 + IoU≥0.5 + 一对一）方便跟训练记录对照
用法：python evt_eval.py dump.pkl.gz [--sweep]
"""
import gzip
import itertools
import json
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np

REPO = os.path.expanduser("~/imu_train")
sys.path.insert(0, REPO)
from label_service import config as cfg, postprocess as pp  # noqa: E402

OV_S = 0.5
SCR = "抓挠"
D = None


def load(path):
    global D
    with gzip.open(path, "rb") as f:
        D = pickle.load(f)
    return D


def _ov(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def base_params(**kw):
    p = dict(event_labels=tuple(cfg.STABLE_EVENT_LABELS), smooth_windows=cfg.STABLE_SMOOTH_WINDOWS,
             min_state_s=cfg.STABLE_MIN_STATE_S, event_enter=cfg.STABLE_EVENT_ENTER, event_stay=cfg.STABLE_EVENT_STAY,
             event_gap_s=cfg.STABLE_EVENT_GAP_S, shake_absorb_s=cfg.STABLE_SHAKE_ABSORB_S,
             event_min_windows=cfg.STABLE_EVENT_MIN_WINDOWS, event_min_mean=cfg.STABLE_EVENT_MIN_MEAN,
             event_single_conf=cfg.STABLE_EVENT_SINGLE_CONF, spectral_min=cfg.STABLE_SPECTRAL_MIN,
             viterbi_switch=cfg.STABLE_VITERBI_SWITCH, notworn_min_s=cfg.STABLE_NOTWORN_MIN_S)
    p.update({k: v for k, v in kw.items() if k in p})
    return pp.StableParams(**p)


def predict_file(f, opt):
    """一个文件：概率（可缩放抓挠）→ 稳定版 viterbi → 抓挠事件 [(s, e)]，再按物理门限过滤"""
    classes, W, S, hz = D["classes"], D["W"], D["S"], D["hz"]
    k = classes.index(SCR)
    P = f["P"].astype(np.float32)
    if opt.get("scale", 1.0) != 1.0:
        P = P.copy()
        P[:, k] *= opt["scale"]
        P /= P.sum(1, keepdims=True)
    t0 = datetime.fromisoformat(f["t0"])
    wins = [{"ts": pp._fmt_ts(t0 + timedelta(seconds=float(s) / hz)), "label": classes[int(q.argmax())],
             "conf": float(q.max()), "probs": {c: float(v) for c, v in zip(classes, q)}} for s, q in zip(f["starts"], P)]
    sp = base_params(**opt)
    segs = pp.stabilize(wins, classes, [SCR], W / hz, S / hz, "segment", sp, algo="viterbi")
    ev = [((pp._parse_ts(x["start_ts"]) - t0).total_seconds(), (pp._parse_ts(x["end_ts"]) - t0).total_seconds())
          for x in segs.get(SCR, [])]
    ph = f["phys"].astype(np.float32)
    st = f["starts"] / hz
    out = []
    for a, b in ev:
        # 事件覆盖到的窗口（窗口中心落在事件里）
        m = ((st + W / hz / 2) >= a) & ((st + W / hz / 2) <= b)
        if not m.any():
            m = (st <= b) & (st + W / hz >= a)
        if m.any():
            mp = np.nanmean(ph[m], axis=0) if np.isfinite(ph[m]).any() else ph[m].mean(0)
            pm = float(P[m, k].mean())
            if opt.get("gstd_min") and np.isfinite(mp[1]) and mp[1] < opt["gstd_min"]:
                continue
            if opt.get("rhy_min") and mp[2] < opt["rhy_min"]:
                continue
            if opt.get("amag_max") and mp[0] > opt["amag_max"]:
                continue
            if opt.get("mean_min") and pm < opt["mean_min"]:
                continue
            if opt.get("len_min") and (b - a) < opt["len_min"]:
                continue
        out.append((a, b))
    return out


def score_file(f, pred):
    gt = [(a, b) for a, b, c, *_ in f["segs"] if c == SCR]
    other = [(a, b, c, n) for a, b, c, n, *_ in f["segs"] if c != SCR]
    r = Counter()
    r["n_gt"] = len(gt)
    r["n_pred"] = len(pred)
    r["gt_hit"] = sum(any(_ov(g, p) >= min(OV_S, (g[1] - g[0]) / 2) for p in pred) for g in gt)
    fp_what = Counter()
    for p in pred:
        if any(_ov(p, g) >= min(OV_S, (p[1] - p[0]) / 2) for g in gt):
            r["tp"] += 1
        else:
            ov = [(o[3] or o[2], _ov(p, o)) for o in other if _ov(p, o) > 0]
            if ov:
                r["fp"] += 1
                fp_what[max(ov, key=lambda x: x[1])[0]] += 1
            else:
                r["unl"] += 1
    # 严格口径
    pairs = sorted(((_ov(g, p) / max(1e-9, (max(g[1], p[1]) - min(g[0], p[0]))), i, j) for i, g in enumerate(gt)
                    for j, p in enumerate(pred)), reverse=True)
    ug, up = set(), set()
    for iou, i, j in pairs:
        if iou >= 0.5 and i not in ug and j not in up:
            ug.add(i); up.add(j)
    r["iou50"] = len(ug)
    r["other_s"] = sum(b - a for a, b, *_ in other)
    return r, fp_what


def _job(args):
    idx, opt = args
    tot, what, sites = Counter(), Counter(), {}
    for i in idx:
        f = D["files"][i]
        r, w = score_file(f, predict_file(f, opt))
        tot.update(r); what.update(w)
        sites.setdefault(f["site"] or "public", Counter()).update(r)
    return tot, what, sites


def evaluate(opt, workers=24):
    n = len(D["files"])
    chunks = [list(range(i, n, workers)) for i in range(workers)]
    tot, what, sites = Counter(), Counter(), {}
    with ProcessPoolExecutor(workers) as ex:
        for t, w, s in ex.map(_job, [(c, opt) for c in chunks]):
            tot.update(t); what.update(w)
            for k, v in s.items():
                sites.setdefault(k, Counter()).update(v)
    return summarize(tot), what, {k: summarize(v) for k, v in sites.items()}


def summarize(t):
    tp, fp, unl = t["tp"], t["fp"], t["unl"]
    return {"n_gt": t["n_gt"], "n_pred": t["n_pred"], "tp": tp, "fp": fp, "unlabeled": unl,
            "prec_known": round(tp / max(tp + fp, 1), 4), "prec_all": round(tp / max(t["n_pred"], 1), 4),
            "recall_loose": round(t["gt_hit"] / max(t["n_gt"], 1), 4),
            "fp_per_h_other": round(fp / max(t["other_s"] / 3600, 1e-9), 2),
            "recall_iou50": round(t["iou50"] / max(t["n_gt"], 1), 4), "prec_iou50": round(t["iou50"] / max(t["n_pred"], 1), 4)}


if __name__ == "__main__":
    load(sys.argv[1])
    print(f"{len(D['files'])} 个文件，类别 {D['classes']}，窗口 {D['W']}、步长 {D['S']}、{D['hz']} Hz", flush=True)
    base, what, sites = evaluate({})
    print("基线（平台现在的稳定版参数）：", json.dumps(base, ensure_ascii=False))
    print("确定的误报落在哪些动作上：", what.most_common(12))
    for k, v in sorted(sites.items()):
        print(f"  场地 {k}: {json.dumps(v, ensure_ascii=False)}")
    if "--sweep" in sys.argv:
        out = os.path.splitext(os.path.splitext(sys.argv[1])[0])[0] + "_sweep.jsonl"
        grid = {"scale": [1.0, 0.7, 0.5], "event_enter": [0.5, 0.65, 0.8], "event_min_mean": [0.45, 0.6, 0.7],
                "event_single_conf": [0.85, 0.95], "gstd_min": [0, 20, 40], "rhy_min": [0, 0.3, 0.45]}
        keys = list(grid)
        with open(out, "w", encoding="utf-8") as fo:
            for vals in itertools.product(*grid.values()):
                opt = dict(zip(keys, vals))
                m, _, _ = evaluate(opt)
                fo.write(json.dumps({"opt": opt, **m}, ensure_ascii=False) + "\n"); fo.flush()
                print(opt, m["prec_known"], m["recall_loose"], m["fp"], m["unlabeled"], flush=True)
