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


def make_valid(C, W, n_classes, filters=(64, 128, 128), k=9):
    """卷积不补零（valid）：窗口里每个输出只看真实数据，滑动时中间层可以缓存复用（流式增量计算）。
    步长要是 2^层数 个采样点的整数倍（20 Hz、3 层池化 → 0.4 s / 0.8 s），池化才对得齐"""
    import torch.nn as nn
    layers, cin, t = [], C, W
    for c in filters:
        layers += [nn.Conv1d(cin, c, k), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2)]
        cin, t = c, (t - k + 1) // 2

    class V(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(cin * t, n_classes)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))
    return V()


def make_valid_small(C, W, n_classes):
    return make_valid(C, W, n_classes, (32, 64, 64))


def make_dsc_small(C, W, n_classes, k=9):
    """深度可分离卷积：每层 = 逐通道卷积（k×C）+ 1×1 混通道，参数和乘加大约是普通卷积的 1/k"""
    import torch.nn as nn

    def block(ci, co):
        return [nn.Conv1d(ci, ci, k, padding=k // 2, groups=ci), nn.Conv1d(ci, co, 1), nn.BatchNorm1d(co), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(0.2)]

    class D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(C, 32, k, padding=k // 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                                      *block(32, 64), *block(64, 96))
            self.fc = nn.Linear(96 * (W // 8), n_classes)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))
    return D()


WINDOW_MODELS = {"cnn": make_cnn, "dual": make_dual, "small": make_small, "tcn": make_tcn,
                 "valid": make_valid, "valid_small": make_valid_small, "dsc_small": make_dsc_small}


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




def _safe_build(args):
    """读一个 CSV；格式不对（比如没有 timestamp 列）的跳过并记原因，别让一个文件拖垮整批"""
    try:
        return T._build_one(args)
    except Exception as e:  # noqa: BLE001
        return args[0], None, f"{type(e).__name__}: {str(e)[:120]}"



TL_CACHE = os.path.expanduser("~/claude_exp/tlcache")
TL_VER = "v2-zero1"   # 改了 load_timeline（比如全 0 行当缺失）就换这个，旧缓存自动作废


def _timeline_cached(csv_path, hz, declared, axes, upsample):
    import hashlib
    st = os.stat(csv_path)
    key = hashlib.md5(f"{csv_path}|{st.st_mtime_ns}|{st.st_size}|{hz}|{declared}|{axes}|{upsample}|{TL_VER}".encode()).hexdigest()
    p = os.path.join(TL_CACHE, key[:2], key + ".npz")
    if os.path.exists(p):
        try:
            z = np.load(p, allow_pickle=False)
            if int(z["none"]):
                return None
            from datetime import datetime as _dt
            return {"x": z["x"], "bad": z["bad"], "t0": _dt.fromisoformat(str(z["t0"])), "src": float(z["src"])}
        except Exception:  # noqa: BLE001 坏了就重算
            pass
    tl = T.load_timeline(csv_path, hz, declared, axes, upsample)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".{os.getpid()}.tmp.npz"
    if tl is None:
        np.savez(tmp, none=1)
    else:
        np.savez(tmp, none=0, x=tl["x"], bad=tl["bad"], t0=tl["t0"].isoformat(), src=tl["src"])
    os.replace(tmp, p)
    return tl


def _build_cached(args):
    """跟 train_v2._build_one 一样，只是时间线走缓存；人工标签每次按当前数据集重新打"""
    csv_path, segs, hz, axes, upsample = args
    try:
        if not os.path.isfile(csv_path):
            return csv_path, None, f"CSV 不存在：{csv_path}"
        tl = _timeline_cached(csv_path, hz, segs[0].get("source_hz"), axes, upsample)
        if tl is None:
            return csv_path, None, None
        n = len(tl["x"])
        lab = np.full(n, -1, np.int32)
        for s_ in segs:
            a_ms, b_ms = s_["start_ms"], s_["end_ms"]
            if a_ms is None or b_ms is None:
                a, b = T._parse_dt(s_["start"]), T._parse_dt(s_["end"])
                if a is None or b is None:
                    continue
                a_ms = (a - tl["t0"]).total_seconds() * 1000
                b_ms = (b - tl["t0"]).total_seconds() * 1000
            a, b = max(int(np.ceil(a_ms / 1000 * hz)), 0), min(int(np.floor(b_ms / 1000 * hz)), n)
            if b > a:
                lab[a:b] = s_["idx"]
        tl = dict(tl)
        tl["seg"] = lab
        return csv_path, tl, None
    except Exception as e:  # noqa: BLE001
        return csv_path, None, f"{type(e).__name__}: {str(e)[:120]}"


def _fz(X, hz):
    """窗口模型只要方向无关变换后的原始窗口；手工特征（给 XGB / RF 的）不算"""
    from gravity_align import map_channels
    return map_channels(X, "inv", True), None

# ── 留出任务上的评估 ─────────────────────────────────────────────────────────

def eval_heldout(a, spec, hz, axes, W, S, classes, remap, Xtr_m, y_tr, nK, gate, merge):
    """全部训练数据训一个，在 --heldout 的任务上算片段级命中 / 误报"""
    from concurrent.futures import ProcessPoolExecutor
    proba, n_params = fit_window(WINDOW_MODELS[a.model], Xtr_m, y_tr, nK, a.epochs, seed=0)
    ds = {"name": "heldout", "export_json": a.heldout, "source_hz": None}
    segs = T.collect_segments([ds], spec["nas_root"], remap, classes, [])
    by_csv: dict = {}
    for i, s_ in enumerate(segs):
        s_["idx"] = i
        by_csv.setdefault(s_["csv"], []).append(s_)
    files = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for p_csv, tl, err in ex.map(_build_cached, [(p, ss, hz, axes, False) for p, ss in by_csv.items()], chunksize=4):
            if tl is None or err:
                continue
            ss = by_csv[p_csv]
            cw = T.labeled_span_windows(tl, ss, W, S, hz)
            if cw is None:
                continue
            Xc_, st_c, _ = cw
            if Xc_.ndim != 3 or Xc_.shape[1] != W:
                continue          # 标注起点贴着文件尾，切出来不够一个窗口
            Xm_c, _ = _fz(Xc_, hz)
            P = merge(proba(Xm_c))
            P = T.apply_motion_gate(P / P.sum(1, keepdims=True), T.motion_of(Xm_c), gate)
            pred = T.postprocess_segments(P, st_c, tl["t0"], classes, W, S, hz, REPO)
            hs = [(x["start_ms"] / 1000, x["end_ms"] / 1000, x["coarse"]) for x in ss if x.get("start_ms") is not None]
            files.append({"gt": {c: [(x0, x1) for x0, x1, cc in hs if cc == c] for c in T.EVENT_EVAL},
                          "other": {c: [(x0, x1) for x0, x1, cc in hs if cc != c] for c in T.EVENT_EVAL}, "pred": pred})
    m = T.segment_metrics(files, classes)
    print(f"▶ 留出任务 {len(files)} 个：抓挠 {m['抓挠']}", flush=True)
    return {"files": len(files), "metrics": m, "n_params": n_params}


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
        for p, tl, err in ex.map(_build_cached, [(p, ss, hz, axes, False) for p, ss in by_csv.items()], chunksize=4):
            if tl is not None and not err:
                tls[p] = tl
    print(f"▶ {a.model}  片段 {len(segs)}  CSV {len(tls)}  {time.time() - t0:.0f}s", flush=True)

    Xtr_raw, seg_tr = T.make_windows(tls, segs, W, S, short_shifts=2)
    Xev_raw, seg_ev = T.make_windows(tls, segs, W, S, short_shifts=0)
    Xtr_m, _ = _fz(Xtr_raw, hz)
    y_tr = np.array([name_idx[segs[s]["name"]] for s in seg_tr])
    y_ev = np.array([classes.index(segs[s]["coarse"]) for s in seg_ev])
    if still_g:
        Xev_m, _ = _fz(Xev_raw, hz)
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
            Xm_c, _ = _fz(Xc_, hz)
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
    heldout = None
    if a.heldout:
        heldout = eval_heldout(a, spec, hz, axes, W, S, classes, remap, Xtr_m, y_tr, nK, gate, merge)
        n_params = heldout.pop("n_params")
    res = {"model": a.model, "epochs": a.epochs, "folds_run": min(a.folds, 5), "n_params": n_params, "heldout": heldout,
           "window": W, "fine": fine,
           "all": T.segment_metrics(seg_files, classes) if seg_files else None,
           "own": T.segment_metrics([f for f in seg_files if f["dataset"] == OWN], classes) if seg_files else None,
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
    p.add_argument("--model", choices=["cnn", "dual", "small", "tcn", "filternet", "valid", "valid_small", "dsc_small"], required=True)
    p.add_argument("--window", type=int, default=0, help="窗口点数，默认用 spec 里的")
    p.add_argument("--fine", default="", help="单独训、再并回去的细类，逗号分隔")
    p.add_argument("--heldout", default="", help="留出任务的导出 JSON：全量训一个模型，在这些任务上算命中 / 误报")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--out", required=True)
    main(p.parse_args())
