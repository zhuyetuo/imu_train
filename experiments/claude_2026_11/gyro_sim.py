"""陀螺仪按需开的离线模拟：3 轴模型常开当触发器，抓挠概率 ≥ θ 时开陀螺仪 H 秒（再触发就续期）。
陀螺仪开机后要攒满一个窗口（8 s）的陀螺仪数据，6 轴窗口才能用：窗口 [s, s+W) 整个落在开机时段里才用 6 轴概率，否则用 3 轴。
用 q01（6 轴）和 q12（3 轴）的 5 折留出预测（同样的文件、同样的窗口起点），走稳定版 v2，只算自己的数据。
输出：命中 / 查准 / 误报，以及陀螺仪开机时间占比、每小时触发次数（分「人标抓挠的时间」和「别的时间」）。
用法：python gyro_sim.py res_q01_pred.pkl res_q12_pred.pkl out.json"""
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np

sys.path[:0] = ["/home/toky/imu_train/src/ml", "/home/toky/imu_train/src/data", "/home/toky/imu_train", "."]
import train_v2 as T  # noqa: E402
from label_service import postprocess as pp  # noqa: E402

OWN = "pool_20261116_all"
D6 = D3 = None


def load(p6, p3):
    global D6, D3
    D6, D3 = pickle.load(open(p6, "rb")), pickle.load(open(p3, "rb"))


def pairs():
    by3 = {(f["csv"], tuple(f["starts"][:3])): f for f in D3["files"] if f["dataset"] == OWN}
    out = []
    for f6 in D6["files"]:
        if f6["dataset"] != OWN:
            continue
        f3 = by3.get((f6["csv"], tuple(f6["starts"][:3])))
        if f3 is not None and len(f3["starts"]) == len(f6["starts"]) and np.array_equal(f3["starts"], f6["starts"]):
            out.append((f6, f3))
    return out


def run(args):
    mode, theta, hold_s = args
    from boundary_eval import params
    classes, W, S, hz = D6["classes"], D6["W"], D6["S"], D6["hz"]
    ci = classes.index("抓挠")
    sp = params()
    targets = [c for c in T.EVENT_EVAL if c in classes]
    res, on_s, tot_s, trig = [], 0.0, 0.0, 0
    on_scr_s, on_oth_s = 0.0, 0.0
    for f6, f3 in pairs():
        st = f6["starts"].astype(np.int64)
        t = st / hz                                 # 窗口起点（秒）
        P3, P6 = f3["P"].astype(np.float64), f6["P"].astype(np.float64)
        if mode == "3ax":
            P, on = P3, np.zeros(len(t), bool)
        elif mode == "6ax":
            P, on = P6, np.ones(len(t), bool)
        else:
            # 触发：窗口结束时（t + W/hz）3 轴给出的抓挠概率 ≥ θ → 从那一刻起开 hold_s 秒
            win_s = W / hz
            on_until = -1e9
            spans = []
            for i in range(len(t)):
                now = t[i] + win_s
                if P3[i, ci] >= theta:
                    if now > on_until:
                        spans.append([now, now + hold_s]); trig += 1
                    else:
                        spans[-1][1] = now + hold_s
                    on_until = now + hold_s
            spans = [(a, b) for a, b in spans]
            on = np.zeros(len(t), bool)
            for a, b in spans:
                on |= (t >= a) & (t + win_s <= b)
            P = np.where(on[:, None], P6, P3)
            # 开机时长（只算标注范围里），按人标的抓挠 / 别的分开
            lo, hi = t[0], t[-1] + win_s
            gt = f6["gt"].get("抓挠", [])
            for a, b in spans:
                a, b = max(a, lo), min(b, hi)
                if b <= a:
                    continue
                on_s += b - a
                inter = sum(max(0.0, min(b, g1) - max(a, g0)) for g0, g1 in gt)
                on_scr_s += inter
                on_oth_s += (b - a) - inter
        tot_s += t[-1] + W / hz - t[0]
        wins = [{"ts": pp._fmt_ts(f6["t0"] + timedelta(seconds=float(s) / hz)), "label": classes[int(q.argmax())],
                 "conf": float(q.max()), "probs": {c: float(v) for c, v in zip(classes, q)}} for s, q in zip(st, P)]
        segs = pp.stabilize(wins, classes, targets, W / hz, S / hz, "segment", sp, algo="viterbi")
        pred = {c: [((pp._parse_ts(x["start_ts"]) - f6["t0"]).total_seconds(), (pp._parse_ts(x["end_ts"]) - f6["t0"]).total_seconds())
                    for x in segs.get(c, [])] for c in targets}
        res.append({"gt": f6["gt"], "other": f6["other"], "pred": pred})
    m = T.segment_metrics(res, classes)["抓挠"]
    scr_h = sum(e - s for f6, _ in pairs() for s, e in f6["gt"].get("抓挠", [])) / 3600
    oth_h = m["other_hours"]
    return {"mode": mode, "theta": theta, "hold_s": hold_s, "hit": m["recall_hit"], "iou50": m["recall_iou50"],
            "precision": m["precision"], "fp_h": m["fp_per_h_other"],
            "gyro_on_frac": round(on_s / max(tot_s, 1e-9), 4) if mode == "mix" else (1.0 if mode == "6ax" else 0.0),
            "gyro_on_per_h_other": round(on_oth_s / max(oth_h * 3600, 1e-9), 4),
            "triggers": trig, "labeled_hours": round(tot_s / 3600, 1), "scratch_hours": round(scr_h, 2), "other_hours": oth_h}


if __name__ == "__main__":
    p6, p3, out = sys.argv[1:4]
    load(p6, p3)
    print("配对文件", len(pairs()), flush=True)
    jobs = [("3ax", 0, 0), ("6ax", 0, 0)] + [("mix", th, h) for th in (0.1, 0.2, 0.3, 0.5) for h in (20, 40, 60, 120)]
    with ProcessPoolExecutor(max_workers=12, initializer=load, initargs=(p6, p3)) as ex:
        rows = list(ex.map(run, jobs))
    for r in rows:
        print(f"{r['mode']:<4} θ={r['theta']:<4} 开{r['hold_s']:>4}s  命中 {r['hit']:.1%}  查准 {r['precision']:.1%}  误报 {r['fp_h']:>5}/h  "
              f"陀螺仪开机占标注时间 {r['gyro_on_frac']:.1%}  非抓挠时间里开机占 {r['gyro_on_per_h_other']:.1%}  触发 {r['triggers']} 次", flush=True)
    print("标注总时长", rows[0]["labeled_hours"], "h，其中人标抓挠", rows[0]["scratch_hours"], "h，别的动作", rows[0]["other_hours"], "h")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
