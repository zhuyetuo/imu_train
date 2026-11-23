"""label_service 容器里跑：直接走 worker 的推理，看 v2 每段抓挠 bout 的平均概率 / gstd / arhythm，哪道门槛删的。"""
import sys

sys.path.insert(0, "/app")
from label_service import app as A, config, pool, postprocess as pp  # noqa: E402
from label_service import jobs  # noqa: E402

path = next(p for jid, p, _m in jobs.done_models() if jid == 60)
print("模型", path)
pool._init_worker(path)
b = pool._bundle
FILES = [("data_raw/2026_11_18_bolangwang/multicam_20261118_140600903_cam3_imu14_raw.csv", 50.0),
         ("data_raw/2026_11_18_yingpeng/multicam_20261118_210000179_cam1_imu6_raw.csv", 220.6),
         ("data_raw/2026_11_18_bolangwang/multicam_20261118_220000223_cam1_imu10_raw.csv", 49.5),
         ("data_raw/2026_11_18_bolangwang/multicam_20261118_200000174_cam5_imu18_raw.csv", 60.8)]
P = A._stable_params()
geom = (b["window_s"], b["stride_s"], b["label_mode"])
print("几何", geom, "类别", b["classes"])
for rel, hz in FILES:
    r = pool._infer_one(config.NAS_ROOT + "/" + rel, hz)
    W = r["windows"]
    tl = config.target_labels_for(b["classes"])
    v2 = pp.stabilize(W, b["classes"], tl, *geom, P, algo="viterbi")
    print("==", rel.split("/")[-1], "窗口", len(W))
    for lab, segs in v2.items():
        if not lab.startswith("抓挠"):
            continue
        for s in segs:
            a, e = pp._parse_ts(s["start_ts"]), pp._parse_ts(s["end_ts"])
            ws = [w for w in W if a <= pp._parse_ts(w["ts"]) < e]
            def m(k):
                v = [w.get(k) for w in ws if w.get(k) is not None]
                return round(sum(v) / len(v), 3) if v else None
            pr = [(w.get("probs") or {}).get(lab, 0) for w in ws]
            print(f"  {lab} {a.time()} {(e - a).total_seconds():.1f}s 段内窗口{len(ws)} 抓挠概率均值 {sum(pr)/max(1,len(pr)):.2f} "
                  f"max {max(pr) if pr else 0:.2f} gstd {m('gstd')} arhythm {m('arhythm')} amag {m('amag')} missing {m('missing')}")
    for name, q in [("只收概率", pp.replace(P, scratch_min_mean=0.7)), ("只收陀螺仪", pp.replace(P, gyro_std_min=20.0)),
                    ("只收节律", pp.replace(P, acc_rhythm_min=0.45))]:
        o = pp.stabilize(W, b["classes"], tl, *geom, q, algo="viterbi")
        print("   ", name, "剩", sum(len(v) for k, v in o.items() if k.startswith("抓挠")))
