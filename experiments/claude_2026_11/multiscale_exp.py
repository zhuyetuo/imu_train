"""多尺度 / 逐点输出 对照实验（2026-11-18）。

跟 train_v2 用同一份数据、同一套「天 × 项圈」分组 5 折、同一个稳定版 v2 后处理和片段级评估，只换模型：
  cnn        现在的端侧 1D-CNN（8 s 窗口，[64,128,128] k9）——基线，在同一个脚本里重跑一遍
  dual       双窗口 CNN：8 s 一路 + 中间 2 s 一路并行，最后拼起来判（大小窗口并行）
  filternet  FilterNet 式逐点输出：全分辨率卷积 + 两路带步长卷积（多尺度）+ 双向 LSTM，每 0.4 s 一个结果；
             训练用 25.6 s 一段、逐点标签；推理整段切块重叠一半、Hanning 加权拼回，
             再按每个 8 s 窗口中间 1 s 的平均概率喂给同一个后处理（后处理把窗口当成它中间那 1 s）

误报 / 小时另外只在自己的数据上算一遍（公开数据没有抓挠标注，混进去会把分母撑大）。

用法：python multiscale_exp.py --spec <train_v2 spec> --model filternet --out res.json [--folds 1] [--epochs 30]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.expanduser("~/imu_train")
sys.path[:0] = [os.path.join(REPO, "src", "ml"), os.path.join(REPO, "src", "data"), os.path.join(REPO, "src", "dl")]
import train_v2 as T  # noqa: E402

OWN = "pool_20261116_all"


# ── 模型 ────────────────────────────────────────────────────────────────────

def conv_stack(nn, cin, filters, k, drop):
    layers = []
    for c in filters:
        layers += [nn.Conv1d(cin, c, k, padding=k // 2), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(drop)]
        cin = c
    return nn.Sequential(*layers), cin


def make_dual(C, W, n_classes, short=40):
    import torch.nn as nn

    class Dual(nn.Module):
        def __init__(self):
            super().__init__()
            self.short = short
            self.long, cl = conv_stack(nn, C, [64, 128, 128], 9, 0.2)
            self.sh, cs = conv_stack(nn, C, [64, 128], 5, 0.2)
            self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(cl * (W // 8) + cs * (short // 4), n_classes))

        def forward(self, x):
            a = (x.shape[2] - self.short) // 2
            return self.fc(__import__("torch").cat([self.long(x).flatten(1), self.sh(x[:, :, a:a + self.short]).flatten(1)], 1))
    return Dual()


def _cnn_cls():
    # 直接按文件路径加载：别的地方也有叫 models 的包，按包名导入会撞
    import importlib.util
    sp = importlib.util.spec_from_file_location("imu_cnn", os.path.join(REPO, "src", "dl", "models", "cnn.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m.CNN


def make_cnn(C, W, n_classes):
    return _cnn_cls()(C, W, n_classes, dict(filters=[64, 128, 128], kernel_size=9, dropout=0.2))


def make_small(C, W, n_classes):
    """GR5513 预算（模型 ≤128 KB）候选：通道减半"""
    return _cnn_cls()(C, W, n_classes, dict(filters=[32, 64, 64], kernel_size=9, dropout=0.2))


def make_tcn(C, W, n_classes, f=64, k=5):
    """膨胀卷积（TCN）：感受野覆盖整个窗口，残差，全局平均池化——看长上下文、参数少"""
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, ci, d):
            super().__init__()
            self.c = nn.Sequential(nn.Conv1d(ci, f, k, padding=d * (k // 2), dilation=d), nn.BatchNorm1d(f), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Conv1d(f, f, k, padding=d * (k // 2), dilation=d), nn.BatchNorm1d(f))
            self.skip = nn.Conv1d(ci, f, 1) if ci != f else nn.Identity()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.c(x) + self.skip(x))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(*[Block(C if i == 0 else f, 2 ** i) for i in range(6)])
            self.fc = nn.Linear(2 * f, n_classes)

        def forward(self, x):
            h = self.blocks(x)
            return self.fc(__import__("torch").cat([h.mean(2), h.amax(2)], 1))
    return TCN()


WINDOW_MODELS = {"cnn": make_cnn, "dual": make_dual, "small": make_small, "tcn": make_tcn}


def make_filternet(C, n_classes, f=96, hid=128):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def cbr(ci, co, k, s=1):
        return nn.Sequential(nn.Conv1d(ci, co, k, stride=s, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU())

    class FilterNet(nn.Module):
        """输入 [B, C, T]，输出 [B, n_classes, T/8]（20 Hz 下每 0.4 s 一个）"""
        def __init__(self):
            super().__init__()
            self.a = nn.Sequential(cbr(C, f, 5), cbr(f, f, 5))                       # 全分辨率
            self.b = nn.Sequential(cbr(f, f, 5, 2), cbr(f, f, 5, 2), cbr(f, f, 5, 2))  # T/8（0.4 s）
            self.c = nn.Sequential(cbr(f, f, 5, 2), cbr(f, f, 5, 2))                   # T/32（1.6 s，更大尺度）
            self.e = nn.Sequential(cbr(3 * f, hid, 1), nn.Dropout(0.2))
            self.lstm = nn.LSTM(hid, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
            self.g = nn.Conv1d(2 * hid, n_classes, 1)

        def forward(self, x):
            a = self.a(x)
            b = self.b(a)
            n = b.shape[2]
            c = F.interpolate(self.c(b), size=n, mode="linear", align_corners=False)
            a8 = F.adaptive_avg_pool1d(a, n)
            h = self.e(torch.cat([a8, b, c], 1)).permute(0, 2, 1)
            h, _ = self.lstm(h)
            return self.g(h.permute(0, 2, 1))
    return FilterNet()


# ── 窗口模型的训练（照 train_v2.fit_cnn，只换网络）──────────────────────────────

def fit_window(make, X, y, n_classes, epochs, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    C = X.shape[2]
    mu, sd = X.reshape(-1, C).mean(0), X.reshape(-1, C).std(0)
    sd[sd < 1e-6] = 1.0
    net = make(C, X.shape[1], n_classes).cuda()
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32).permute(0, 2, 1).contiguous()
    yt = torch.tensor(y, dtype=torch.long)
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    w = np.sqrt(len(y) / (n_classes * np.maximum(cnt, 1)))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).cuda(), label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    bs = 512
    nb = (len(y) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=max(1, epochs * nb))
    shift = max(1, X.shape[1] // 10)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(y))
        for b in range(nb):
            ix = perm[b * bs:(b + 1) * bs]
            xb = Xt[ix].cuda()
            xb = xb * (1 + 0.1 * torch.randn(len(ix), 1, 1, device="cuda")) + 0.02 * torch.randn_like(xb)
            xb = torch.roll(xb, int(rng.integers(-shift, shift + 1)), 2)
            loss = lossf(net(xb), yt[ix].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    net.eval()

    def proba(Xm):
        out = []
        with torch.no_grad():
            for i in range(0, len(Xm), 4096):
                xb = torch.tensor((Xm[i:i + 4096] - mu) / sd, dtype=torch.float32).permute(0, 2, 1).cuda()
                out.append(torch.softmax(net(xb), 1).cpu().numpy())
        return np.concatenate(out)
    return proba, int(sum(p.numel() for p in net.parameters()))


# ── 逐点模型：数据、训练、推理 ────────────────────────────────────────────────

L_CHUNK, HOP_TRAIN = 512, 128


def sample_labels(tl, cls_of_seg, plain_of_seg, classes, still_g, W):
    """逐点标签（归并后的类别，-1 = 没标 / 掉包）；静止 / 活动 按 8 s 内 |a| 标准差重分（跟窗口版同一个阈值）"""
    sid = tl["seg"]
    lab = np.full(len(sid), -1, np.int64)
    m = sid >= 0
    lab[m] = cls_of_seg[sid[m]]
    if still_g:
        import pandas as pd
        mag = np.linalg.norm(tl["x"][:, :3], axis=1)
        mot = pd.Series(mag).rolling(W, center=True, min_periods=W // 2).std().to_numpy()
        plain = np.zeros(len(sid), bool)
        plain[m] = plain_of_seg[sid[m]]
        i_st, i_ac = classes.index(T.STILL), classes.index(T.ACTIVE)
        lab[plain & (lab == i_ac) & (mot < still_g)] = i_st
        lab[plain & (lab == i_st) & (mot >= still_g)] = i_ac
    lab[tl["bad"]] = -1
    return lab


def train_chunks(tls_lab):
    """训练块：25.6 s 一块、步长 6.4 s，块里至少 1/4 有标签。文件比一块短就用边缘值补齐（补的地方不算损失）"""
    X, Y = [], []
    for x, lab in tls_lab:
        n = len(x)
        if n < L_CHUNK:
            pad = L_CHUNK - n
            x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
            lab = np.r_[lab, np.full(pad, -1)]
            n = L_CHUNK
        for s in range(0, n - L_CHUNK + 1, HOP_TRAIN):
            yl = lab[s:s + L_CHUNK]
            if (yl >= 0).mean() >= 0.25:
                X.append(x[s:s + L_CHUNK])
                Y.append(yl)
    X = np.stack(X).astype(np.float32)
    Xi = __import__("gravity_align").map_channels(X, "inv", True).astype(np.float32)
    return Xi, np.stack(Y)


def fit_dense(X, Y, n_classes, epochs, seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(seed)
    C = X.shape[2]
    mu, sd = X.reshape(-1, C).mean(0), X.reshape(-1, C).std(0)
    sd[sd < 1e-6] = 1.0
    net = make_filternet(C, n_classes).cuda()
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32).permute(0, 2, 1).contiguous()
    Yt = torch.tensor(Y, dtype=torch.long)
    yv = Y[Y >= 0]
    cnt = np.bincount(yv, minlength=n_classes).astype(float)
    w = np.sqrt(len(yv) / (n_classes * np.maximum(cnt, 1)))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).cuda(), label_smoothing=0.05, ignore_index=-1)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    bs = 64
    nb = (len(X) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=max(1, epochs * nb))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for b in range(nb):
            ix = perm[b * bs:(b + 1) * bs]
            xb = Xt[ix].cuda()
            xb = xb * (1 + 0.1 * torch.randn(len(ix), 1, 1, device="cuda")) + 0.02 * torch.randn_like(xb)
            out = F.interpolate(net(xb), size=xb.shape[2], mode="linear", align_corners=False)
            loss = lossf(out, Yt[ix].cuda())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
    net.eval()

    def dense(x):
        """整段 [n, C] 原始 → 逐点概率 [n, K]：切 25.6 s 一块、重叠一半、Hanning 加权拼回"""
        n = len(x)
        half = L_CHUNK // 2
        xp = np.pad(x, ((half, half + L_CHUNK), (0, 0)), mode="edge")
        starts = list(range(0, n + half, half))
        blocks = np.stack([xp[s:s + L_CHUNK] for s in starts]).astype(np.float32)
        bi = __import__("gravity_align").map_channels(blocks, "inv", True).astype(np.float32)
        acc = np.zeros((len(xp), n_classes))
        wsum = np.zeros(len(xp))
        hw = np.hanning(L_CHUNK) + 1e-3
        with torch.no_grad():
            for i in range(0, len(bi), 256):
                xb = torch.tensor((bi[i:i + 256] - mu) / sd, dtype=torch.float32).permute(0, 2, 1).cuda()
                p = torch.softmax(F.interpolate(net(xb), size=L_CHUNK, mode="linear", align_corners=False), 1)
                p = p.permute(0, 2, 1).cpu().numpy()
                for j, s in enumerate(starts[i:i + 256]):
                    acc[s:s + L_CHUNK] += p[j] * hw[:, None]
                    wsum[s:s + L_CHUNK] += hw
        P = acc / np.maximum(wsum, 1e-9)[:, None]
        return P[half:half + n]
    return dense, int(sum(p.numel() for p in net.parameters()))


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main(a):
    from sklearn.model_selection import StratifiedGroupKFold
    from concurrent.futures import ProcessPoolExecutor

    spec = json.load(open(a.spec, encoding="utf-8"))
    hz, axes, classes = float(spec["hz"]), int(spec.get("axes") or 3), list(spec["classes"])
    remap, W = spec.get("label_remap") or {}, int(a.window or spec["window"])
    S = max(1, int(round(float(spec.get("stride_s") or 1.0) * hz)))
    still_g = float(spec.get("still_motion_g") or 0) or None
    fine = [x for x in (a.fine or "").split(",") if x]
    t0 = time.time()
    segs = T.collect_segments(spec["datasets"], spec["nas_root"], remap, classes, fine)
    for i, s in enumerate(segs):
        s["idx"] = i
    # 细类单独训（进食 / 饮水 / 舔 / 啃 / 蹭 …），输出时概率加回它归并到的那一类
    names = list(classes) + sorted({s["name"] for s in segs} - set(classes))
    name_idx = {n: i for i, n in enumerate(names)}
    back_arr = np.array([classes.index(n) if n in classes else
                         classes.index(next(s["coarse"] for s in segs if s["name"] == n)) for n in names])
    nK = len(names)
    print(f"▶ 训练类别 {names}", flush=True)

    def merge(P):
        out = np.zeros((len(P), len(classes)))
        np.add.at(out.T, back_arr, P.T)
        return out
    by_csv: dict = {}
    for s in segs:
        by_csv.setdefault(s["csv"], []).append(s)
    tls = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for p, tl, err in ex.map(T._build_one, [(p, ss, hz, axes, False) for p, ss in by_csv.items()], chunksize=4):
            if tl is not None and not err:
                tls[p] = tl
    print(f"▶ {a.model}  片段 {len(segs)}  CSV {len(tls)}  {time.time() - t0:.0f}s", flush=True)

    Xtr_raw, seg_tr = T.make_windows(tls, segs, W, S, short_shifts=2)
    Xev_raw, seg_ev = T.make_windows(tls, segs, W, S, short_shifts=0)
    Xtr_m, _ = T.featurize(Xtr_raw, hz)
    y_tr = np.array([name_idx[segs[s]["name"]] for s in seg_tr])
    y_ev = np.array([classes.index(segs[s]["coarse"]) for s in seg_ev])
    if still_g:
        Xev_m, _ = T.featurize(Xev_raw, hz)
        plain_tr = np.array([segs[s]["name"] in (T.STILL, T.ACTIVE) for s in seg_tr])
        plain_ev = np.array([segs[s]["name"] in (T.STILL, T.ACTIVE) for s in seg_ev])
        y_tr, _ = T.still_by_motion(y_tr, T.motion_of(Xtr_m), plain_tr, name_idx[T.STILL], name_idx[T.ACTIVE], still_g)
        y_ev, _ = T.still_by_motion(y_ev, T.motion_of(Xev_m), plain_ev, classes.index(T.STILL), classes.index(T.ACTIVE), still_g)
    g_tr = np.array([T._group_of(segs[s]["sample_code"]) for s in seg_tr])
    g_ev = np.array([T._group_of(segs[s]["sample_code"]) for s in seg_ev])
    gate = T.fit_motion_gate(T.motion_of(Xtr_m), back_arr[y_tr], classes, still_g)
    print(f"▶ 窗口 训练 {len(y_tr)} 验证 {len(y_ev)}；动作门 {gate}  {time.time() - t0:.0f}s", flush=True)

    if a.model == "filternet":
        cls_of_seg = np.array([name_idx[s["name"]] for s in segs])
        plain_of_seg = np.array([s["name"] in (T.STILL, T.ACTIVE) for s in segs])
        lab_of = {p: sample_labels(tl, cls_of_seg, plain_of_seg, classes, still_g, W) for p, tl in tls.items()}

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    seg_files, n_params = [], None
    for k, (_, te) in enumerate(cv.split(np.zeros(len(y_ev)), y_ev, g_ev)):
        if k >= a.folds:
            break
        test_g = set(g_ev[te])
        if a.model == "filternet":
            tr_files = [p for p in tls if T._group_of(by_csv[p][0]["sample_code"]) not in test_g]
            Xc, Yc = train_chunks([(tls[p]["x"], lab_of[p]) for p in tr_files])
            print(f"  折 {k + 1}：训练块 {len(Xc)}（{len(Xc) * L_CHUNK / hz / 3600:.1f} h，含重叠）", flush=True)
            dense, n_params = fit_dense(Xc, Yc, nK, a.epochs, seed=k)
        else:
            tr = ~np.isin(g_tr, list(test_g))
            proba, n_params = fit_window(WINDOW_MODELS[a.model], Xtr_m[tr], y_tr[tr], nK, a.epochs, seed=k)
        for p_csv, tl in tls.items():
            ss = by_csv.get(p_csv) or []
            if not ss or T._group_of(ss[0]["sample_code"]) not in test_g:
                continue
            cw = T.labeled_span_windows(tl, ss, W, S, hz)
            if cw is None:
                continue
            Xc_, st_c, _ = cw
            Xm_c, _ = T.featurize(Xc_, hz)
            if a.model == "filternet":
                lo = max(0, int(st_c[0]))
                hi = min(len(tl["x"]), int(st_c[-1]) + W)
                Pd = dense(tl["x"][lo:hi])
                c0 = W // 2 - S // 2
                P = merge(np.stack([Pd[s - lo + c0: s - lo + c0 + S].mean(0) for s in st_c]))
            else:
                P = merge(proba(Xm_c))
            P = T.apply_motion_gate(P / P.sum(1, keepdims=True), T.motion_of(Xm_c), gate)
            pred = T.postprocess_segments(P, st_c, tl["t0"], classes, W, S, hz, REPO)
            hs = [(s["start_ms"] / 1000, s["end_ms"] / 1000, s["coarse"]) for s in ss if s.get("start_ms") is not None]
            seg_files.append({"dataset": ss[0]["dataset"], "csv": p_csv, "P": P.astype(np.float32), "starts": st_c,
                              "t0": tl["t0"],
                              "gt": {c: [(x0, x1) for x0, x1, cc in hs if cc == c] for c in T.EVENT_EVAL},
                              "other": {c: [(x0, x1) for x0, x1, cc in hs if cc != c] for c in T.EVENT_EVAL},
                              "pred": pred})
        print(f"  折 {k + 1}/5 完成  {time.time() - t0:.0f}s", flush=True)
    res = {"model": a.model, "epochs": a.epochs, "folds_run": min(a.folds, 5), "n_params": n_params,
           "window": W, "fine": fine,
           "all": T.segment_metrics(seg_files, classes),
           "own": T.segment_metrics([f for f in seg_files if f["dataset"] == OWN], classes),
           "seconds": round(time.time() - t0)}
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    # 逐文件的窗口概率留着：换后处理（比如精准起止）不用重训，直接在这上面重算
    import pickle
    with open(os.path.splitext(a.out)[0] + "_pred.pkl", "wb") as fp:
        pickle.dump({"classes": classes, "W": W, "S": S, "hz": hz, "axes": axes, "files": seg_files}, fp)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--model", choices=["cnn", "dual", "small", "tcn", "filternet"], required=True)
    p.add_argument("--window", type=int, default=0, help="窗口点数，默认用 spec 里的")
    p.add_argument("--fine", default="", help="单独训、再并回去的细类，逗号分隔")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", required=True)
    main(p.parse_args())
