"""Unlabeled windows for pseudo-labeling: never-labeled NAS raw files, excluding (day, collar) groups of the given dataset.
Saves inv-channel windows (float16) of one window length. Usage:
  python make_unlabeled.py --window 120 --n_files 4000 --max_windows 400000 --out queue/unl_w120.npz"""
import argparse, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssl_pretrain as S  # group_of_path / file discovery helpers live there
T = S.T
HZ = 20.0


def load(args):
    path, W, axes = args
    try:
        tl = T.load_timeline(path, HZ, None, axes, False)
    except Exception:
        return None
    if tl is None:
        return None
    x, bad = tl["x"], tl["bad"]
    st = [s for s in range(0, len(x) - W, W // 2) if not bad[s:s + W].any()]
    if not st:
        return None
    from gravity_align import map_channels
    return map_channels(np.stack([x[s:s + W] for s in st]).astype(np.float32), "inv", True).astype(np.float16)


def main(a):
    import json
    t0 = time.time(); NAS = S.NAS
    excl, used = set(), set()
    for t in json.load(open(f"{NAS}/data_train/{a.exclude_ds}/merged_tmp.json", encoding="utf-8")):
        c = (t.get("data") or {}).get("csv")
        if c: excl.add(S.group_of_path(c))
    for d in os.listdir(f"{NAS}/data_train"):
        p = f"{NAS}/data_train/{d}/merged_tmp.json"
        if os.path.exists(p):
            for t in json.load(open(p, encoding="utf-8")):
                c = (t.get("data") or {}).get("csv")
                if c: used.add(os.path.basename(c))
    cand = [os.path.join(r, f) for r, _, fs in os.walk(f"{NAS}/data_raw") for f in fs
            if f.endswith("_raw.csv") and "imu" in f and f not in used and S.group_of_path(f) not in excl]
    random.Random(1).shuffle(cand); cand = cand[:a.n_files]
    chunks, n = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for X in ex.map(load, [(p, a.window, 3) for p in cand], chunksize=2):
            if X is not None and len(X):
                chunks.append(X); n += len(X)
    X = np.concatenate(chunks)
    if len(X) > a.max_windows:
        X = X[np.random.default_rng(0).choice(len(X), a.max_windows, replace=False)]
    np.savez(a.out, X=X)
    print(f"saved {a.out} {X.shape} from {len(cand)} files {time.time()-t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=120); p.add_argument("--n_files", type=int, default=4000)
    p.add_argument("--max_windows", type=int, default=400000); p.add_argument("--exclude_ds", default="pool_20261116_all")
    p.add_argument("--workers", type=int, default=12); p.add_argument("--out", required=True)
    main(p.parse_args())
