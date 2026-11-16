"""Self-supervised pretraining (SimCLR-style contrastive) of the *edge CNN's conv stack* on never-labeled NAS raw data.
The pretrained encoder has exactly the structure of imu_train src/dl/models/cnn.py, so fine-tuning keeps the int8 edge model
unchanged. Files whose (day, collar) group appears in the given dataset are excluded (no leakage into CV test folds).

Usage: python ssl_pretrain.py --exclude_ds pool_20261116_all --n_files 1500 --axes 3 --out ssl_acc3.pt
"""
import argparse, json, os, random, re, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = os.path.expanduser("~/imu_train")
sys.path[:0] = [os.path.join(REPO, "src", "ml"), os.path.join(REPO, "src", "data"), os.path.join(REPO, "src", "dl")]
import train_v2 as T  # noqa: E402

NAS = os.path.expanduser("~/ai_data")
HZ, W, STEP = 20.0, 80, 40


def group_of_path(p):
    b = os.path.basename(p)
    imu = re.search(r"imu\d+", b, re.I); day = re.search(r"_(\d{8})_", b)
    return f"{day.group(1) if day else '?'}_{imu.group(0).lower() if imu else b}"


def load_windows(args):
    path, axes = args
    try:
        tl = T.load_timeline(path, HZ, None, axes, False)
    except Exception:
        return None
    if tl is None:
        return None
    x, bad = tl["x"], tl["bad"]
    st = [s for s in range(0, len(x) - W, STEP) if not bad[s:s + W].any()]
    if not st:
        return None
    X = np.stack([x[s:s + W] for s in st]).astype(np.float32)
    from gravity_align import map_channels
    Xm = map_channels(X, "inv", True)
    mot = Xm[:, :, 2].std(1)
    keep = mot >= 0.005                     # only moving windows: still / not-worn need no representation learning
    return Xm[keep].astype(np.float16)


def augment(x, torch):
    """x: [B, C, T] normalized. Two random views."""
    B, C, Tn = x.shape
    s = 1 + 0.2 * torch.randn(B, 1, 1, device=x.device)
    x = x * s + 0.05 * torch.randn_like(x)
    sh = int(np.random.randint(-Tn // 4, Tn // 4 + 1))
    x = torch.roll(x, sh, 2)
    # random crop + resize back (time warp)
    L = int(Tn * np.random.uniform(0.7, 1.0)); a = np.random.randint(0, Tn - L + 1)
    x = torch.nn.functional.interpolate(x[:, :, a:a + L], size=Tn, mode="linear", align_corners=False)
    # random channel dropout (one channel zeroed for 20% of samples)
    m = (torch.rand(B, device=x.device) < 0.2)
    if m.any():
        c = int(np.random.randint(0, C)); x[m, c] = 0
    return x


def main(a):
    import importlib.util
    import torch, torch.nn as nn
    # load src/dl/models/cnn.py by path: src/ml/models (the ML package) shadows the name "models"
    spec = importlib.util.spec_from_file_location("edge_cnn", os.path.join(REPO, "src", "dl", "models", "cnn.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    CNN = mod.CNN
    t0 = time.time()
    excl = set()
    for t in json.load(open(f"{NAS}/data_train/{a.exclude_ds}/merged_tmp.json", encoding="utf-8")):
        c = (t.get("data") or {}).get("csv")
        if c: excl.add(group_of_path(c))
    used = set()
    for d in os.listdir(f"{NAS}/data_train"):
        p = f"{NAS}/data_train/{d}/merged_tmp.json"
        if os.path.exists(p):
            for t in json.load(open(p, encoding="utf-8")):
                c = (t.get("data") or {}).get("csv")
                if c: used.add(os.path.basename(c))
    cand = []
    for root, _, fs in os.walk(f"{NAS}/data_raw"):
        for f in fs:
            if f.endswith("_raw.csv") and "imu" in f and f not in used and group_of_path(f) not in excl:
                cand.append(os.path.join(root, f))
    random.Random(0).shuffle(cand)
    cand = cand[:a.n_files]
    print(f"unlabeled candidate files {len(cand)} (excluded groups {len(excl)}) {time.time()-t0:.0f}s", flush=True)
    chunks = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, X in enumerate(ex.map(load_windows, [(p, a.axes) for p in cand], chunksize=2)):
            if X is not None and len(X): chunks.append(X)
            if (i + 1) % 200 == 0: print(f"  read {i+1}/{len(cand)} windows {sum(len(c) for c in chunks)} {time.time()-t0:.0f}s", flush=True)
    X = np.concatenate(chunks).astype(np.float32); del chunks
    C = X.shape[2]
    mu = X.reshape(-1, C).mean(0); sd = X.reshape(-1, C).std(0); sd[sd < 1e-6] = 1
    print(f"pretraining windows {len(X)} x {X.shape[1:]} {time.time()-t0:.0f}s", flush=True)
    dev = "cuda"
    cfg = {"filters": [int(v) for v in a.filters.split(",")], "kernel_size": a.kernel, "dropout": 0.2}
    net = CNN(C, W, 5, cfg).to(dev)
    feat_dim = net.fc.in_features
    proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, 128)).to(dev)
    opt = torch.optim.AdamW(list(net.conv.parameters()) + list(proj.parameters()), 1e-3, weight_decay=1e-4)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float16).permute(0, 2, 1).contiguous()
    bs = a.batch; nb = len(Xt) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * nb)
    tau = 0.2
    for ep in range(a.epochs):
        net.train(); proj.train(); perm = torch.randperm(len(Xt)); tot = 0.0
        for b in range(nb):
            xb = Xt[perm[b * bs:(b + 1) * bs]].to(dev).float()
            v1, v2 = augment(xb.clone(), torch), augment(xb.clone(), torch)
            z = torch.nn.functional.normalize(proj(torch.cat([net.conv(v1).flatten(1), net.conv(v2).flatten(1)])), dim=1)
            sim = z @ z.t() / tau
            n = len(xb)
            sim.fill_diagonal_(-1e9)
            target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(dev)
            loss = nn.functional.cross_entropy(sim, target)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item()
        print(f"  epoch {ep+1}/{a.epochs} loss {tot/max(nb,1):.4f} {time.time()-t0:.0f}s", flush=True)
    torch.save({"conv": net.conv.state_dict(), "mu": mu, "sd": sd, "cfg": cfg, "axes": a.axes, "n_windows": len(X),
                "n_files": len(cand)}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exclude_ds", default="pool_20261116_all"); p.add_argument("--n_files", type=int, default=1500)
    p.add_argument("--axes", type=int, default=3); p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=12); p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=1024); p.add_argument("--filters", default="32,64,128")
    p.add_argument("--kernel", type=int, default=5)
    main(p.parse_args())
