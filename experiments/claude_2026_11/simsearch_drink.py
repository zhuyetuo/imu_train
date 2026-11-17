"""实验：用已有的「饮水」片段当样例，按 IMU 相似度在别的狗的整小时数据里找饮水，量准不准。

留一只狗：要找的是这只狗的饮水，样例只用别的狗的（平台上真实用法：拿已标的去找没标的）。
数据：平台上 119 个有饮水片段的任务（15 只狗，226 段饮水，全部人看过），每个任务整小时。
窗口：中心每 1 s 一个；手工特征用 4 s 窗口，CNN 用 8 s 窗口（模型就是 8 s 训的），中心对齐。
窗口标签：中心落在人看过的饮水片段里 = 饮水；落在别的人看过的片段里 = 已知非饮水；其余 = 没标（算非饮水，偏悲观）。

三种排序比：
  feat  ：193 维手工特征（去掉重力绝对值那几维），z-score 后余弦相似度，对样例取 top-5 平均
  emb   ：负例挖掘 CNN（训练记录 #43）倒数第二层，余弦相似度，top-5 平均
  prob  ：负例挖掘 CNN 直接给的「饮水」概率（就是现在负例候选的来源）
指标（每个任务 = 一小时）：
  P@10    ：分数最高的 10 个窗口（彼此隔开 ≥4 s，不重复计同一段）里真是饮水的比例
  段命中@10：这个小时里人标的饮水片段，有多少段被这 10 个窗口碰到
  AP      ：整小时所有窗口按分数排，饮水的平均精度
"""
import importlib.util
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

WT = os.path.expanduser("~/claude_exp/imu_train_wt")
NAS = os.path.expanduser("~/ai_data")
sys.path[:0] = [os.path.join(WT, "src", "ml"), os.path.join(WT, "src", "data")]
_s = importlib.util.spec_from_file_location("tv2", os.path.join(WT, "src", "ml", "train_v2.py"))
tv = importlib.util.module_from_spec(_s)
_s.loader.exec_module(tv)

HZ, W4, W8 = 20.0, 80, 160
TARGET = "饮水"
MODEL = os.path.join(WT, "results", "processed_pool_20261116_all__cnn_mine_6ax", "20hz_v2", "cnn")
OUT = os.path.expanduser("~/claude_exp/simsearch")


def load_task(t):
    tl = tv.load_timeline(os.path.join(NAS, t["csv"]), HZ, t.get("hz"), axes=6)
    if tl is None:
        return None
    x, bad = tl["x"], tl["bad"]
    n = len(x)
    centers = np.arange(W8 // 2, n - W8 // 2, int(HZ))          # 每秒一个中心，8 s 窗口装得下
    ok = np.array([not bad[c - W8 // 2:c + W8 // 2].any() for c in centers])
    centers = centers[ok]
    if len(centers) == 0:
        return None
    X4 = np.stack([x[c - W4 // 2:c + W4 // 2] for c in centers]).astype(np.float32)
    X8 = np.stack([x[c - W8 // 2:c + W8 // 2] for c in centers]).astype(np.float32)
    Xm4, F = tv.featurize(X4, HZ)
    Xm8 = tv.map_channels(X8, "inv", True) if hasattr(tv, "map_channels") else None
    if Xm8 is None:
        from gravity_align import map_channels
        Xm8 = map_channels(X8, "inv", True)
    t_ms = centers / HZ * 1000
    lab = np.zeros(len(centers), np.int8)                     # 0 没标 / 1 饮水 / -1 已知别的
    seg_id = np.full(len(centers), -1)
    drinks = [i for i in t["items"] if i["label"] == TARGET and i["human"]]
    for i in t["items"]:
        if not i["human"] or i["label"] == TARGET:
            continue
        lab[(t_ms >= i["start_ms"]) & (t_ms < i["end_ms"])] = -1
    for k, i in enumerate(drinks):
        m = (t_ms >= i["start_ms"]) & (t_ms < i["end_ms"])
        lab[m] = 1
        seg_id[m] = k
    # 只有 1 s 长的饮水，中心可能一个都落不进：退一步按最近的中心算
    for k, i in enumerate(drinks):
        if not (seg_id == k).any():
            j = int(np.argmin(np.abs(t_ms - (i["start_ms"] + i["end_ms"]) / 2)))
            lab[j], seg_id[j] = 1, k
    return dict(task=t["task_id"], dog=_dog(t["sample_code"]), F=F, Xm8=Xm8, lab=lab, seg=seg_id,
                n_seg=len(drinks), mot=Xm4[:, :, 2].std(axis=1))


def _dog(code):
    import re
    m = re.search(r"imu\d+", code or "")
    return m.group(0) if m else code


def cnn_outputs(tasks):
    import torch
    meta = json.load(open(os.path.join(MODEL, "dl_cnn_best.json"), encoding="utf-8"))
    classes = meta["classes"] if isinstance(meta["classes"], list) else eval(meta["classes"])
    spec = importlib.util.spec_from_file_location("dlcnn", os.path.join(WT, "src", "dl", "models", "cnn.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    net = mod.CNN(int(meta["n_channels"]), int(meta["window_size"]), len(classes), meta["model_cfg"])
    net.load_state_dict(torch.load(os.path.join(MODEL, "dl_cnn_best.pt"), map_location="cpu", weights_only=True))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()
    mu, sd = np.asarray(meta["ch_mean"], np.float32), np.asarray(meta["ch_std"], np.float32)
    k = classes.index(TARGET)
    for t in tasks:
        embs, probs = [], []
        with torch.no_grad():
            for i in range(0, len(t["Xm8"]), 4096):
                xb = torch.tensor((t["Xm8"][i:i + 4096] - mu) / sd, dtype=torch.float32).permute(0, 2, 1).to(dev)
                h = net.conv(xb).flatten(1)
                probs.append(torch.softmax(net.fc(h), 1)[:, k].cpu().numpy())
                embs.append(h.cpu().numpy())
        t["emb"] = np.concatenate(embs)
        t["prob"] = np.concatenate(probs)
        del t["Xm8"]


def knn_score(db, queries, k=5):
    """db 每一行跟 queries 的余弦相似度，取最高 k 个的平均"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.tensor(queries, device=dev)
    q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
    out = []
    for i in range(0, len(db), 8192):
        d = torch.tensor(db[i:i + 8192], device=dev)
        d = d / (d.norm(dim=1, keepdim=True) + 1e-8)
        s = d @ q.T
        out.append(s.topk(min(k, s.shape[1]), dim=1).values.mean(1).cpu().numpy())
    return np.concatenate(out)


def top_spread(score, k=10, gap=4):
    """分数最高的 k 个窗口，彼此中心隔开 ≥ gap 个（秒）——同一段饮水别重复算 10 次"""
    order = np.argsort(-score)
    picked = []
    for i in order:
        if all(abs(i - j) >= gap for j in picked):
            picked.append(i)
            if len(picked) == k:
                break
    return np.array(picked)


def ap(score, y):
    order = np.argsort(-score)
    yy = y[order]
    if yy.sum() == 0:
        return np.nan
    hits = np.cumsum(yy)
    return float((hits[yy == 1] / (np.flatnonzero(yy == 1) + 1)).mean())


def main():
    items = json.load(open(os.path.join(OUT, "drink_items.json"), encoding="utf-8"))
    with Pool(16) as pool:
        tasks = [t for t in pool.map(load_task, items) if t is not None]
    print(f"载入 {len(tasks)} / {len(items)} 个任务（6 轴；没陀螺仪的跳过），窗口 {sum(len(t['lab']) for t in tasks)}，"
          f"饮水窗口 {sum((t['lab'] == 1).sum() for t in tasks)}，狗 {len({t['dog'] for t in tasks})}", flush=True)
    cnn_outputs(tasks)
    # 手工特征：去掉重力绝对值那几维，全体 z-score
    allF = np.concatenate([t["F"] for t in tasks])
    mask, _ = tv.gravity_level_mask(8)                     # 6 轴 inv = 8 路通道
    keep = ~mask
    mu, sd = allF[:, keep].mean(0), allF[:, keep].std(0) + 1e-6
    for t in tasks:
        t["Fz"] = np.nan_to_num((t["F"][:, keep] - mu) / sd).astype(np.float32)
    res = defaultdict(lambda: defaultdict(list))
    dogs = sorted({t["dog"] for t in tasks})
    for dog in dogs:
        mine = [t for t in tasks if t["dog"] == dog]
        others = [t for t in tasks if t["dog"] != dog]
        qF = np.concatenate([t["Fz"][t["lab"] == 1] for t in others])
        qE = np.concatenate([t["emb"][t["lab"] == 1] for t in others])
        for t in mine:
            moving = t["mot"] >= 0.005          # 静止的窗口不可能是在喝水，三种方法都先去掉
            scores = {
                "feat": np.where(moving, knn_score(t["Fz"], qF), -9),
                "emb": np.where(moving, knn_score(t["emb"], qE), -9),
                "prob": np.where(moving, t["prob"], -9),
                "random": np.where(moving, np.random.default_rng(t["task"]).random(len(t["lab"])), -9),
            }
            y = (t["lab"] == 1).astype(int)
            for name, sc in scores.items():
                top = top_spread(sc, 10)
                res[name]["p10"].append(float(y[top].mean()))
                if t["n_seg"]:
                    res[name]["seg"].append(len(set(t["seg"][top][t["seg"][top] >= 0])) / t["n_seg"])
                res[name]["ap"].append(ap(sc, y))
                res[name]["known"].append(float((t["lab"][top] != 0).mean()))
    print(f"\n留一只狗（{len(dogs)} 只），每个任务 = 一小时；基准：饮水窗口占比 "
          f"{np.mean([(t['lab'] == 1).mean() for t in tasks]):.2%}")
    print(f"{'方法':<8}{'P@10':>8}{'段命中@10':>11}{'AP':>8}{'前10里有人标过的':>14}")
    names = {"feat": "手工特征相似", "emb": "CNN特征相似", "prob": "负例模型概率", "random": "随机"}
    out = {}
    for k in ("feat", "emb", "prob", "random"):
        r = res[k]
        out[k] = {m: float(np.nanmean(v)) for m, v in r.items()}
        print(f"{names[k]:<8}{out[k]['p10']:>8.2%}{out[k]['seg']:>11.2%}{out[k]['ap']:>8.3f}{out[k]['known']:>14.2%}")
    json.dump(out, open(os.path.join(OUT, "result_drink.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
