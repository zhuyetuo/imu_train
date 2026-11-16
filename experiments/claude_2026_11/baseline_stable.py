"""Stable v2 (the production RF, models/stable_v2_rf) on the same files/spans as compare3 — not trained on any of them
(it was trained on 2026-08 data), so this is a held-out baseline. Saves cont_stable.npz + stable_geom.json in --dir."""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = os.path.expanduser("~/imu_train")
sys.path[:0] = [os.path.join(REPO, "src", "ml"), os.path.join(REPO, "src", "data"), REPO]
import train_v2 as T  # noqa: E402

HZ, W, S = 16.0, 16, 8
DECL = {"pool_20261107_052727": 50.0}
MODEL = None


def _init():
    global MODEL
    import joblib
    MODEL = joblib.load(os.path.join(REPO, "models/stable_v2_rf/ml_rf.pkl"))


def one(args):
    i, f = args
    from gravity_align import map_channels
    from features_vec import extract_features_vec
    tl = T.load_timeline(f["csv"], HZ, DECL.get(f["dataset"]), 6, False)
    if tl is None: return i, None, None
    x, bad = tl["x"], tl["bad"]
    a0 = max(0, int(f["span"][0] * HZ) - W // 2); a1 = min(len(x) - W, int(f["span"][1] * HZ))
    st = np.array([s for s in range(a0, max(a0, a1) + 1, S) if not bad[s:s + W].any()], int)
    if len(st) == 0: return i, None, None
    Xm = map_channels(np.stack([x[s:s + W] for s in st]).astype(np.float32), "tilt", True)
    F = np.nan_to_num(extract_features_vec(Xm, 16))
    return i, st, MODEL.predict_proba(F).astype(np.float16)


def main(a):
    meta = json.load(open(os.path.join(a.dir, "files.json"), encoding="utf-8"))
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for i, st, p in ex.map(one, list(enumerate(meta["files"])), chunksize=4):
            if st is not None: out[f"s{i}"] = st; out[f"p{i}"] = p
    np.savez_compressed(os.path.join(a.dir, "cont_stable.npz"), **out)
    json.dump({"hz": HZ, "W": W, "S": S, "label_mode": "majority", "classes": ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]},
              open(os.path.join(a.dir, "stable_geom.json"), "w"), ensure_ascii=False)
    print("files", len(out) // 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--dir", required=True); p.add_argument("--workers", type=int, default=16)
    main(p.parse_args())
