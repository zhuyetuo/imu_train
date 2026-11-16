"""Automatic relabeling of a platform dataset export, using out-of-fold scratch probabilities (compare3 output: models that never
saw the file). The original export is NOT modified: writes a new dataset dir + relabel_log.csv.

Rules (conservative — drop doubtful segments rather than flip them):
  R1  scratch segment whose per-second scratch probability never reaches LOW (<0.15) over >= 2 s     → dropped (doubtful)
  R2  non-scratch segment containing a run of >= RUN_S seconds with scratch probability > HIGH (0.85) → dropped (doubtful)
  R3  scratch segment boundaries from the SIGNAL, not the model (so model blur is not written into labels):
      3–8 Hz energy envelope of |a| (1 s FFT, 0.25 s hop); start/end moved to where the envelope crosses
      RATIO × (segment median) within ±MAX_SHIFT s; unchanged if there is no clear crossing
Probability = mean of the methods given (default cnn + xgb), per second, from windows centered in that second.

Usage: python relabel.py --cmp ~/claude_exp/pool3 --src pool_20261116_all --dst pool_20261116_relabel_v1
"""
import argparse, csv, json, os, shutil
import numpy as np

NAS = os.path.expanduser("~/ai_data")


def per_second(cmp_dir, methods):
    meta = json.load(open(os.path.join(cmp_dir, "files.json"), encoding="utf-8"))
    hz, W = meta["hz"], meta["W"]
    Zs = [np.load(os.path.join(cmp_dir, f"cont_{m}.npz")) for m in methods]
    k = meta["classes"].index("抓挠")
    out = {}
    for i, f in enumerate(meta["files"]):
        if f"s{i}" not in Zs[0].files:
            continue
        st = Zs[0][f"s{i}"]
        p = np.mean([Z[f"p{i}"][:, k].astype(np.float32) for Z in Zs], 0)
        c = np.floor((st + W / 2) / hz).astype(int)
        n = c.max() + 1
        acc = np.zeros(n); cnt = np.zeros(n)
        np.add.at(acc, c, p); np.add.at(cnt, c, 1)
        sec = np.full(n, np.nan); sec[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
        out[f["csv"]] = sec
    return out


def envelope(path, hz=20.0):
    """3–8 Hz band energy of |a| every 0.25 s (1 s Hann FFT). Returns (times_s, energy) or None."""
    import sys
    sys.path[:0] = [os.path.expanduser("~/imu_train/src/ml"), os.path.expanduser("~/imu_train/src/data")]
    import train_v2 as T
    tl = T.load_timeline(path, hz, None, 3, False)
    if tl is None:
        return None
    mag = np.linalg.norm(tl["x"].astype(np.float64), axis=1)
    n, hop = int(hz), int(hz / 4)
    win = np.hanning(n); f = np.fft.rfftfreq(n, 1 / hz); band = (f >= 3) & (f <= 8)
    starts = np.arange(0, len(mag) - n, hop)
    if len(starts) == 0:
        return None
    seg = np.stack([mag[s:s + n] - mag[s:s + n].mean() for s in starts])
    e = (np.abs(np.fft.rfft(seg * win, axis=1)) ** 2)[:, band].sum(1)
    return (starts + n / 2) / hz, e


def refine(env, s0, s1, ratio, max_shift):
    t, e = env
    inside = (t >= s0) & (t <= s1)
    if inside.sum() < 2:
        return s0, s1
    thr = ratio * np.median(e[inside])
    def edge(t0, rising):
        m = (t >= t0 - max_shift) & (t <= t0 + max_shift)
        idx = [i for i in np.flatnonzero(m) if i > 0]
        if rising:
            cand = [i for i in idx if e[i - 1] < thr <= e[i]]
        else:
            cand = [i for i in idx if e[i - 1] >= thr > e[i]]
        if not cand:
            return t0
        return float(t[min(cand, key=lambda i: abs(t[i] - t0))])
    return edge(s0, True), edge(s1, False)


def crossing(sec, t, direction, max_shift):
    """nearest second within ±max_shift where p crosses 0.5 in the given direction ('up' for start, 'down' for end)"""
    best = None
    for d in range(-max_shift, max_shift + 1):
        j = int(t) + d
        if j <= 0 or j >= len(sec) or np.isnan(sec[j]) or np.isnan(sec[j - 1]):
            continue
        up = sec[j - 1] < 0.5 <= sec[j]; down = sec[j - 1] >= 0.5 > sec[j]
        if (direction == "up" and up) or (direction == "down" and down):
            if best is None or abs(d) < abs(best - int(t)):
                best = j
    return best


def main(a):
    probs = per_second(a.cmp, a.methods.split(","))
    src = os.path.join(NAS, "data_train", a.src); dst = os.path.join(NAS, "data_train", a.dst)
    tasks = json.load(open(os.path.join(src, "merged_tmp.json"), encoding="utf-8"))
    log, n_seen = [], 0
    envs = {}
    for t in tasks:
        rel = (t.get("data") or {}).get("csv")
        path = rel if os.path.isabs(rel or "") else os.path.join(NAS, rel or "")
        sec = probs.get(path)
        has_scr = any(((r.get("value") or {}).get("timeserieslabels") or [""])[0].startswith("抓挠")
                      for ann in t.get("annotations") or [] for r in ann.get("result") or [])
        env = envs.setdefault(path, envelope(path) if (has_scr and os.path.exists(path)) else None)
        for ann in t.get("annotations") or []:
            keep = []
            for r in ann.get("result") or []:
                v = r.get("value") or {}
                chain = v.get("timeserieslabels") or []
                if not chain or v.get("start_ms") is None or sec is None:
                    keep.append(r); continue
                n_seen += 1
                s0, s1 = v["start_ms"] / 1000, v["end_ms"] / 1000
                a0, a1 = int(np.floor(s0)), int(np.ceil(s1))
                seg = sec[max(a0, 0):min(a1, len(sec))]
                seg = seg[~np.isnan(seg)]
                is_scr = chain[0].startswith("抓挠")
                row = {"task": t.get("id"), "csv": rel, "label": " > ".join(chain), "start_s": round(s0, 2), "end_s": round(s1, 2)}
                if len(seg) == 0:
                    keep.append(r); continue
                if is_scr and s1 - s0 >= 2 and seg.max() < a.low:
                    log.append({**row, "action": "drop", "rule": "R1 标抓挠但整段模型抓挠概率都很低", "p_max": round(float(seg.max()), 3)})
                    continue
                if not is_scr:
                    run = best = 0
                    for v_ in seg:
                        run = run + 1 if v_ > a.high else 0; best = max(best, run)
                    if best >= a.run_s:
                        log.append({**row, "action": "drop", "rule": f"R2 标{chain[0]}但其中连续 {best}s 模型很确定是抓挠",
                                    "p_max": round(float(seg.max()), 3)})
                        continue
                if is_scr and env is not None:
                    n0, n1 = refine(env, s0, s1, a.ratio, a.max_shift)
                    if n1 - n0 >= 1.0 and (abs(n0 - s0) >= 0.5 or abs(n1 - s1) >= 0.5):
                        v = dict(v); v["start_ms"] = int(n0 * 1000); v["end_ms"] = int(n1 * 1000)
                        r = dict(r); r["value"] = v
                        log.append({**row, "action": "boundary", "rule": "R3 起止对齐到 3-8Hz 节律能量明显起落的位置（只看信号）",
                                    "new_start_s": round(n0, 2), "new_end_s": round(n1, 2)})
                keep.append(r)
            ann["result"] = keep
    os.makedirs(dst, exist_ok=True)
    json.dump(tasks, open(os.path.join(dst, "merged_tmp.json"), "w", encoding="utf-8"), ensure_ascii=False)
    meta = json.load(open(os.path.join(src, "meta.json"), encoding="utf-8"))
    meta.update(name=a.dst, relabel={"from": a.src, "by": "claude relabel.py", "cmp": a.cmp, "methods": a.methods,
                                     "low": a.low, "high": a.high, "run_s": a.run_s, "max_shift": a.max_shift, "ratio": a.ratio,
                                     "n_segments_seen": n_seen,
                                     "n_drop": sum(1 for x in log if x["action"] == "drop"),
                                     "n_boundary": sum(1 for x in log if x["action"] == "boundary")})
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    keys = ["task", "csv", "label", "start_s", "end_s", "action", "rule", "p_max", "new_start_s", "new_end_s"]
    with open(os.path.join(dst, "relabel_log.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(log)
    print(json.dumps(meta["relabel"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cmp", required=True); p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--methods", default="cnn,xgb"); p.add_argument("--low", type=float, default=0.15)
    p.add_argument("--high", type=float, default=0.85); p.add_argument("--run_s", type=int, default=3)
    p.add_argument("--max_shift", type=float, default=1.5); p.add_argument("--ratio", type=float, default=0.3)
    main(p.parse_args())
