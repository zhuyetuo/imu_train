"""精准起止：在交叉验证留出文件的逐窗口概率上（multiscale_exp.py 存的 *_pred.pkl），比几种定起止的办法。

  v2            稳定版 v2，原样
  v2+微调        平台现在推理时做的：起止各 ±1 s 内按陀螺仪能量找（refine_boundaries，ratio 0.3）
  精准版 r=…     起点在 [-2 s, +3 s]、终点在 [-4 s, +2 s] 里找（v2 的终点平均偏晚，往里多找一些）；
                能量先做 0.3 s 平滑，连续 0.3 s 高于「片段内能量中位数 × r」才算在动；
                没有陀螺仪（3 轴）就用加速度模长的能量

能量包络跟平台 pool.py 同一个算法：模长去均值、平方、按 0.1 s 分箱。
用法：python boundary_eval.py res_q01_pred.pkl spec.json
"""
import json
import pickle
import sys
from datetime import timedelta

import numpy as np

sys.path[:0] = ["/home/toky/imu_train/src/ml", "/home/toky/imu_train/src/data", "/home/toky/imu_train"]
import train_v2 as T  # noqa: E402
from label_service import config as C, postprocess as pp  # noqa: E402

OWN = "pool_20261116_all"


def params():
    return pp.StableParams(event_labels=tuple(C.STABLE_EVENT_LABELS), smooth_windows=C.STABLE_SMOOTH_WINDOWS,
                           min_state_s=C.STABLE_MIN_STATE_S, event_enter=C.STABLE_EVENT_ENTER,
                           event_stay=C.STABLE_EVENT_STAY, event_gap_s=C.STABLE_EVENT_GAP_S,
                           shake_absorb_s=C.STABLE_SHAKE_ABSORB_S, event_min_windows=C.STABLE_EVENT_MIN_WINDOWS,
                           event_min_mean=C.STABLE_EVENT_MIN_MEAN, event_single_conf=C.STABLE_EVENT_SINGLE_CONF,
                           spectral_min=C.STABLE_SPECTRAL_MIN, viterbi_switch=C.STABLE_VITERBI_SWITCH,
                           notworn_min_s=C.STABLE_NOTWORN_MIN_S)


def envelope(x, hz, t0, axes, band=None):
    mag = np.linalg.norm(x[:, 3:6] if axes == 6 else x[:, :3], axis=1)
    if band is not None:
        # 只留抓挠节律那一段频率（后腿往复 3–9 Hz）：转身、走动这些低频大动作不算「在抓」
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, [band[0], min(band[1], hz / 2 * 0.95)], btype="band", fs=hz, output="sos")
        mag = sosfiltfilt(sos, mag - mag.mean()) if len(mag) > 30 else mag - mag.mean()
    step = max(1, int(round(hz / 10)))
    sq = (mag - mag.mean()) ** 2
    nb = len(sq) // step
    env = sq[:nb * step].reshape(nb, step).mean(1)
    return {"t0": pp._fmt_ts(t0), "hz": hz / step, "energy": env.tolist()}


def refine_precise(segs, env, ratio, pre=2.0, start_in=3.0, end_in=4.0, end_out=2.0, smooth=3, sustain=3):
    t0, hz = pp._parse_ts(env["t0"]), env["hz"]
    e = np.convolve(np.asarray(env["energy"]), np.ones(smooth) / smooth, mode="same")
    n = len(e)
    for seg in segs:
        s, en = pp._parse_ts(seg["start_ts"]), pp._parse_ts(seg["end_ts"])
        si, ei = int(round((s - t0).total_seconds() * hz)), int(round((en - t0).total_seconds() * hz))
        core = e[max(0, si):min(n, ei)]
        if len(core) == 0:
            continue
        thr = float(np.median(core)) * ratio
        on = e >= thr
        lo, hi = max(0, si - int(pre * hz)), min(n - sustain, si + int(start_in * hz))
        new_s = next((i for i in range(lo, hi + 1) if on[i:i + sustain].all()), None)
        lo, hi = max(sustain - 1, ei - int(end_in * hz)), min(n - 1, ei + int(end_out * hz))
        new_e = next((i for i in range(hi, lo - 1, -1) if on[i - sustain + 1:i + 1].all()), None)
        if new_s is not None and new_e is not None and (new_e - new_s) / hz >= 0.5:
            seg["start_ts"] = pp._fmt_ts(t0 + timedelta(seconds=new_s / hz))
            seg["end_ts"] = pp._fmt_ts(t0 + timedelta(seconds=(new_e + 1) / hz))


def shift_segs(segs, ds, de):
    for seg in segs:
        s, e = pp._parse_ts(seg["start_ts"]), pp._parse_ts(seg["end_ts"])
        s2, e2 = s + timedelta(seconds=ds), e + timedelta(seconds=de)
        if (e2 - s2).total_seconds() < 0.5:   # 太短的片段别收没了：保留中间 0.5 s
            mid = s + (e - s) / 2
            s2, e2 = mid - timedelta(seconds=0.25), mid + timedelta(seconds=0.25)
        seg["start_ts"], seg["end_ts"] = pp._fmt_ts(s2), pp._fmt_ts(e2)


def part_of(f):
    import zlib
    return zlib.crc32(T._group_of(f["csv"].split("/")[-1]).encode()) % 5


def signed_errors(files, c):
    on, off = [], []
    for f in files:
        for g in f["gt"].get(c, []):
            best = max(f["pred"].get(c, []), key=lambda p: T._iou(g, p), default=None)
            if best is not None and T._iou(g, best) >= 0.5:
                on.append(best[0] - g[0]); off.append(best[1] - g[1])
    md = lambda v: round(float(np.median(v)), 2) if v else None
    return md(on), md(off)


def main(pkl, spec_path):
    d = pickle.load(open(pkl, "rb"))
    spec = json.load(open(spec_path, encoding="utf-8"))
    src_hz = {ds["name"]: ds.get("source_hz") for ds in spec["datasets"]}
    classes, W, S, hz, axes = d["classes"], d["W"], d["S"], d["hz"], d["axes"]
    sp = params()
    targets = [c for c in T.EVENT_EVAL if c in classes]
    # 偏差校正：先拿 v2 原样跑一遍，按「（天, 项圈）组」分 5 份，每份用另外 4 份量出来的
    # 起点 / 终点系统偏差（中位数）来校正——不偷看自己
    def wins_of(f):
        return [{"ts": pp._fmt_ts(f["t0"] + timedelta(seconds=float(s) / hz)), "label": classes[int(q.argmax())],
                 "conf": float(q.max()), "probs": {c: float(v) for c, v in zip(classes, q)}} for s, q in zip(f["starts"], f["P"])]

    def to_sec(segs, f):
        return {c: [((pp._parse_ts(x["start_ts"]) - f["t0"]).total_seconds(), (pp._parse_ts(x["end_ts"]) - f["t0"]).total_seconds())
                    for x in segs.get(c, [])] for c in targets}

    base = []
    for f in d["files"]:
        segs = pp.stabilize(wins_of(f), classes, targets, W / hz, S / hz, "segment", sp, algo="viterbi")
        base.append({"dataset": f["dataset"], "gt": f["gt"], "other": f["other"], "pred": to_sec(segs, f), "part": part_of(f)})
    bias = {}
    for k in range(5):
        others = [b for b in base if b["part"] != k and b["dataset"] == OWN]
        bias[k] = {c: tuple(-(v or 0.0) for v in signed_errors(others, c)) for c in targets}
    print("偏差校正量（每份不同，第 1 份）：", bias[0], flush=True)

    variants = ([("v2", None), ("v2+微调(平台现在)", "old")]
                + [(f"总能量 r={r}", ("tot", r)) for r in (0.5, 0.8)]
                + [("偏差校正", ("bias",))]
                + [(f"抓挠频段 r={r}", ("band", r)) for r in (0.3, 0.5, 0.8)]
                + [(f"校正+频段 r={r}", ("bias_band", r)) for r in (0.3, 0.5, 0.8)])
    out = {v: [] for v, _ in variants}
    for i, f in enumerate(d["files"]):
        tl = T.load_timeline(f["csv"], hz, src_hz.get(f["dataset"]), axes)
        env = envelope(tl["x"], hz, tl["t0"], axes) if tl is not None else None
        envb = envelope(tl["x"], hz, tl["t0"], axes, band=(3, 9)) if tl is not None else None
        k = part_of(f)
        wins = wins_of(f)
        for name, how in variants:
            segs = pp.stabilize(wins, classes, targets, W / hz, S / hz, "segment", sp, algo="viterbi")
            if env is not None and how is not None:
                for c in targets:
                    ss = segs.get(c) or []
                    if how == "old":
                        pp.refine_boundaries(ss, env, sp)
                    elif how[0] == "tot":
                        refine_precise(ss, env, how[1])
                    elif how[0] == "bias":
                        shift_segs(ss, *bias[k][c])
                    elif how[0] == "band":
                        refine_precise(ss, envb, how[1])
                    elif how[0] == "bias_band":
                        shift_segs(ss, *bias[k][c])
                        refine_precise(ss, envb, how[1], pre=1.0, start_in=2.0, end_in=2.0, end_out=1.0)
            pred = to_sec(segs, f)
            out[name].append({"dataset": f["dataset"], "gt": f["gt"], "other": f["other"], "pred": pred})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(d['files'])}", flush=True)
    res = {}
    print(f"\n{'后处理':<18}{'类别':<8}{'命中':>7}{'起止对齐':>9}{'查准':>7}{'误报/h':>8}{'起点误差':>9}{'终点误差':>9}{'起点偏(中位)':>12}{'终点偏(中位)':>12}")
    for name, _ in variants:
        own = [x for x in out[name] if x["dataset"] == OWN]
        m = T.segment_metrics(own, classes)
        res[name] = m
        for c in targets:
            r = m[c]
            so, se = signed_errors(own, c)
            r["onset_signed_median_s"], r["offset_signed_median_s"] = so, se
            print(f"{name:<18}{c:<8}{r['recall_hit']:>7.1%}{r['recall_iou50']:>9.1%}{r['precision']:>7.1%}{r['fp_per_h_other']:>8}"
                  f"{r['onset_abs_err_s']:>9}{r['offset_abs_err_s']:>9}{str(so):>12}{str(se):>12}")
    json.dump(res, open(pkl.replace("_pred.pkl", "_boundary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
