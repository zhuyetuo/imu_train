"""新版训练流程（v2）：直接从 NAS 上的原始 CSV + 标注平台导出的 JSON 训随机森林。

跟 train_custom.sh 那条路的区别（都是 2026-11 用 #23 / #27 的数据做对照实验定下来的，
按「天 + 项圈」分组交叉验证，同一只狗同一天不会同时出现在训练和验证里）：

1. 按真实时间切窗口。
   老流程把一个任务里不连续的几段标注首尾拼成一条再滑窗，会切出跨两段（甚至跨两个类别）
   的假窗口，短动作（抓挠、甩头）吃亏最大。这里每个 CSV 还原成一条时间线：
   CSV 的 timestamp 是蓝牙包的到达时间（4~5 个点同一个时间），按包首点插值还原每个点的时间，
   再插到严格的源采样率网格上，用 resample_training_match（跟服务端推理同一个降采样）降到目标采样率。
   窗口只在一段标注内部滑，不跨掉包（>0.3s 没数据）。
2. 比窗口短的片段：以片段中心取窗口，前后用真实的原始数据补；训练时再左右挪几次做增强。
3. 细分训练、合并输出：进食 / 饮水 / 甩身体这种被归并进「活动」、但动作上像抓挠的，训练时
   各自单独成类（模型被迫学会区分它们和抓挠），训完把这几类的概率并回归并后的类别。
   随机森林可以把合并直接做进叶子里，存出来就是普通的 5 类模型，推理 / 端侧导出都不用改。
4. 输出校准：某几类的概率乘一个系数再取最大（比如抓挠 ×2.5；系数乘在每棵树的叶子上、每棵树再归一化，
   比乘在平均概率上弱，#27 的数据上 2.5 ≈ 乘在平均概率上的 1.5）。老数据里抓挠被标成活动的不少，
   模型学得偏保守、次数少报；乘回来让查全≈查准，次数不偏。同样做进叶子里。
5. 指标用分组交叉验证算（比按片段随机切更接近换一只狗、换一天的真实表现）。

输入：--spec 一个 JSON（label_service/jobs.py 写的），见 _load_spec。
输出：跟 train.py 同一套约定，平台 / 推理 / 端侧导出都认：
  results/processed_<run>/<hz>hz_v2/rf/ml_rf.pkl + ml_rf.json
  data/processed_<run>/<hz>hz/{train,val,test}.npz（端侧导出拿 val 算端侧 F1、挑 golden）
最后打印「纯标注: <模型路径>」，label_service 从日志里认这一行。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "data")]

DEFAULT_FINE = ["进食", "饮水", "甩身体"]


# ── 时间线 ──────────────────────────────────────────────────────────────────

def _nominal_rate(rate: float) -> float:
    """实测点率（49.6 / 15.9 …）对到标称采样率"""
    for r in (100.0, 62.5, 50.0, 31.25, 25.0, 16.0, 15.625, 10.0):
        if abs(rate - r) / r < 0.08:
            return r
    return round(rate, 3)


GYRO_COLS = (("gyro_x", "gyro_y", "gyro_z"), ("gyr_x", "gyr_y", "gyr_z"))


def load_timeline(csv_path: str, hz: float, declared: float | None = None, axes: int = 3,
                  upsample: bool = False) -> dict | None:
    """一个 CSV → 目标采样率上的 3 轴加速度 + 无效点标记。源采样率比目标低的不往上插，返回 None"""
    import pandas as pd
    from resample_training_match import resample_training_match

    head = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns
    cols = ["acc_x", "acc_y", "acc_z"]
    if axes == 6:
        g = next((c for c in GYRO_COLS if all(x in head for x in c)), None)
        if g is None:
            return None  # 要 6 轴但这份没有陀螺仪
        cols += list(g)
    d = pd.read_csv(csv_path, usecols=["timestamp", *cols], encoding="utf-8-sig")
    d = d.dropna()
    # 采集阶段蓝牙不稳：断了的那几帧采集端写「六轴全 0」占位（录制不中断、还能重连）。真实佩戴时重力总落在
    # 某个轴上，加速度三轴不可能同时精确为 0——这些行就是没有数据。以前当真信号插进窗口，带着一串 0 的窗口
    # 被标成那段时间人标的类别（活动 / 抓挠……），等于掺噪声（2026-11-18 抽 60 个训练文件：1.5% 的行是全 0，
    # 连续段中位 0.6 s、最长 11 分钟）。去掉之后跟 dropna 一样：留下的时间缺口 > 0.3 s 的被下面标成掉包，窗口不用。
    # 产品是板上实时跑的，不存在这种占位，训练里也不该有
    zero = (d[["acc_x", "acc_y", "acc_z"]] == 0).all(axis=1)
    if zero.any():
        d = d[~zero]
    if len(d) < 10:
        return None
    dt = pd.to_datetime(d.timestamp)
    t0 = dt.iloc[0]
    ts = (dt - t0).dt.total_seconds().to_numpy()
    acc = d[cols].to_numpy(np.float64)
    first = np.r_[True, np.diff(ts) > 0]
    ai = np.flatnonzero(first)
    at = ts[ai]
    rate = (len(ts) - 1) / max(ts[-1], 1e-6)
    t = np.interp(np.arange(len(ts)), np.r_[ai, len(ts)], np.r_[at, at[-1] + (len(ts) - ai[-1]) / rate])
    # 源采样率优先用平台上声明的（每份数据集各自的「原始采样率」）；掉包多的文件实测点率会偏低，猜不准
    src = float(declared) if declared else _nominal_rate(rate)
    if src < hz <= src * 1.1:
        src = hz  # 15.9Hz 实测的老 16Hz 数据当 15.625 用，不算往上插
    if hz > src + 1e-6:
        if not upsample:
            return None
        # 往上插：线性插到目标采样率。注意插出来的数据在 src/2 以上没有任何信号（16Hz → 25Hz 的话
        # 8~12.5Hz 那段是空的），跟真 25Hz 的数据不一样——要不要用，看实测
        src = hz
    grid = np.arange(0, t[-1], 1 / src)
    xs = np.stack([np.interp(grid, t, acc[:, k]) for k in range(acc.shape[1])], 1)
    idx = np.searchsorted(t, grid)
    near = np.minimum(np.abs(t[np.clip(idx, 0, len(t) - 1)] - grid), np.abs(t[np.clip(idx - 1, 0, len(t) - 1)] - grid))
    bad_src = near > max(0.3, 3 / src)
    x = resample_training_match(xs, src, hz) if abs(hz - src) > 1e-6 else xs
    n = len(x)
    bad = bad_src[np.clip(np.rint(np.arange(n) / hz * src).astype(int), 0, len(bad_src) - 1)]
    return dict(x=x.astype(np.float32), bad=bad, t0=t0.to_pydatetime(), src=src)


# ── 标签 ────────────────────────────────────────────────────────────────────

def coarse_of(chain: list[str], remap: dict, classes: list[str]) -> str | None:
    """整条类别链从细到粗找：先看归并表，再看是不是训练类别本身"""
    for name in reversed(chain):
        if name in remap:
            v = remap[name]
            return v if v in classes else None
        if name in classes:
            return name
    return None


def collect_segments(datasets: list[dict], nas_root: str, remap: dict, classes: list[str], fine: list[str]):
    """所有数据集的片段：(csv 绝对路径, start_ms, end_ms, 训练类别名, 归并后类别名, sample_code, 数据集名)"""
    from datetime import datetime

    segs = []
    for ds in datasets:
        with open(ds["export_json"], encoding="utf-8") as f:
            tasks = json.load(f)
        for t in tasks:
            data = t.get("data") or {}
            rel = data.get("csv")
            if not rel:
                continue
            path = rel if os.path.isabs(rel) else os.path.join(nas_root, rel)
            for ann in t.get("annotations") or []:
                for r in ann.get("result") or []:
                    v = r.get("value") or {}
                    chain = v.get("timeserieslabels") or []
                    if not chain:
                        continue
                    coarse = coarse_of(chain, remap, classes)
                    if coarse is None or coarse in (ds.get("exclude_classes") or []):
                        continue
                    leaf = chain[-1]
                    train_name = next((c for c in reversed(chain) if c in fine), None) or coarse
                    segs.append(dict(csv=path, start_ms=v.get("start_ms"), end_ms=v.get("end_ms"),
                                     start=v.get("start"), end=v.get("end"), name=train_name, coarse=coarse,
                                     leaf=leaf, sample_code=data.get("sample_code") or os.path.basename(path),
                                     dataset=ds["name"], source_hz=ds.get("source_hz")))
    return segs


def _parse_dt(s):
    from datetime import datetime
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except (TypeError, ValueError):
            continue
    return None


TL_MARGIN_S = 60.0          # 时间线只留标注范围前后这么多秒（要比任何窗口都长）
TL_CACHE_VER = "v2-zero1"   # load_timeline 改了（比如全 0 行当缺失）就换，旧缓存自动作废


def _timeline_cached(csv_path, hz, declared, axes, upsample):
    """TRAIN_V2_TL_CACHE=<目录> 时：每个 CSV 读一次、重采样一次存成 .npz（键 = 路径 + 修改时间 + 大小 + 参数 + 版本），
    以后几秒载入。同一批数据反复训（对比实验、正式训练）时省掉每次几分钟的读文件。没设就跟原来一样现读"""
    cache = os.environ.get("TRAIN_V2_TL_CACHE")
    if not cache:
        return load_timeline(csv_path, hz, declared, axes, upsample)
    import hashlib
    from datetime import datetime as _dt
    st = os.stat(csv_path)
    key = hashlib.md5(f"{csv_path}|{st.st_mtime_ns}|{st.st_size}|{hz}|{declared}|{axes}|{upsample}|{TL_CACHE_VER}".encode()).hexdigest()
    path = os.path.join(cache, key[:2], key + ".npz")
    if os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if int(z["none"]):
                return None
            return {"x": z["x"], "bad": z["bad"], "t0": _dt.fromisoformat(str(z["t0"])), "src": float(z["src"])}
        except Exception:  # noqa: BLE001 坏了就重读
            pass
    tl = load_timeline(csv_path, hz, declared, axes, upsample)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp.npz"
    if tl is None:
        np.savez(tmp, none=1)
    else:
        np.savez(tmp, none=0, x=tl["x"], bad=tl["bad"], t0=tl["t0"].isoformat(), src=tl["src"])
    os.replace(tmp, path)
    return tl


def _build_one(args):
    """读不了的（格式不对，比如采集程序版本不同、没有 timestamp 列）跳过并记原因：一个文件出错不该拖垮整批"""
    try:
        return _build_one_inner(args)
    except Exception as e:  # noqa: BLE001
        return args[0], None, f"读不了 {os.path.basename(args[0])}：{type(e).__name__}: {str(e)[:120]}"


def _build_one_inner(args):
    csv_path, segs, hz, axes, upsample = args
    if not os.path.isfile(csv_path):
        return csv_path, None, f"CSV 不存在：{csv_path}"
    tl = _timeline_cached(csv_path, hz, segs[0].get("source_hz"), axes, upsample)
    if tl is None:
        return csv_path, None, None
    n = len(tl["x"])
    lab = np.full(n, -1, np.int32)
    for s in segs:
        a_ms, b_ms = s["start_ms"], s["end_ms"]
        if a_ms is None or b_ms is None:
            # 老导出没有相对毫秒：拿绝对时间减 CSV 首行时间
            a, b = _parse_dt(s["start"]), _parse_dt(s["end"])
            if a is None or b is None:
                continue
            a_ms = (a - tl["t0"]).total_seconds() * 1000
            b_ms = (b - tl["t0"]).total_seconds() * 1000
        a, b = max(int(np.ceil(a_ms / 1000 * hz)), 0), min(int(np.floor(b_ms / 1000 * hz)), n)
        if b > a:
            lab[a:b] = s["idx"]
    # 只留标注范围（前后各留 TL_MARGIN_S 秒）：整天的录制只标了一部分时，省大量内存。原始下标 = 裁后下标 + off
    nz = np.flatnonzero(lab >= 0)
    off = 0
    if len(nz):
        lo_i, hi_i = int(nz[0]), int(nz[-1]) + 1
        ms = [(s["start_ms"], s["end_ms"]) for s in segs if s.get("start_ms") is not None and s.get("end_ms") is not None]
        if ms:
            lo_i = min(lo_i, int(min(a for a, _ in ms) / 1000 * hz))
            hi_i = max(hi_i, int(np.ceil(max(b for _, b in ms) / 1000 * hz)))
        m = int(TL_MARGIN_S * hz)
        off, end = max(0, lo_i - m), min(n, hi_i + m)
        if off > 0 or end < n:
            tl["x"], tl["bad"], lab = tl["x"][off:end].copy(), tl["bad"][off:end].copy(), lab[off:end].copy()
    tl["seg"] = lab
    tl["off"] = off
    return csv_path, tl, None


# ── 窗口 / 特征 ─────────────────────────────────────────────────────────────

def _map_inv_chunked(X, chunk: int = 20000):
    """map_channels(X, "inv") 的省内存版：结果一样，分块写进预先分配的 float32 数组，峰值不再是结果的 3 倍"""
    from gravity_align import map_channels
    if X.ndim != 3 or len(X) <= chunk:
        return map_channels(X, "inv", True)
    first = map_channels(X[:chunk], "inv", True)
    out = np.empty((len(X),) + first.shape[1:], np.float32)
    out[:chunk] = first
    for i in range(chunk, len(X), chunk):
        out[i:i + chunk] = map_channels(X[i:i + chunk], "inv", True)
    return out


def make_windows_cnn(tls: dict, W: int, S: int, short_shifts: int, chunk: int = 20000):
    """CNN 用：切窗口的同时做 inv 通道变换，写进一个预先分配的 float32 数组。
    前 n_ev 个 = make_windows(short_shifts=0) 的窗口（同样顺序，当验证用，直接切片不复制），
    后面是短片段左右挪出来的增强窗口。返回 (Xm, seg_of, n_ev)"""
    from gravity_align import map_channels
    keys = list(tls)
    ev, ex = [], []
    for ki, csv_path in enumerate(keys):
        tl = tls[csv_path]
        x, sid, bad = tl["x"], tl["seg"], tl["bad"]
        n = len(x)
        for s in np.unique(sid[sid >= 0]):
            idx = np.flatnonzero(sid == s)
            a, b = idx[0], idx[-1] + 1
            L = b - a
            extra = []
            if L >= W:
                base = list(range(a, b - W + 1, S))
            else:
                mid = (a + b) // 2 - W // 2
                base = [mid]
                room = (W - L) // 2
                for k in range(1, short_shifts + 1):
                    off = int(round(room * k / (short_shifts + 1)))
                    extra += [mid - off, mid + off]
            for dst, starts in ((ev, base), (ex, extra)):
                for st in starts:
                    if st < 0 or st + W > n or bad[st:st + W].any():
                        continue
                    dst.append((ki, st, int(s)))
    allw = ev + ex
    seg_of = np.asarray([w[2] for w in allw])
    if not allw:
        return np.zeros((0, W, 0), np.float32), seg_of, 0
    out = None
    for i in range(0, len(allw), chunk):
        blk = np.stack([tls[keys[k]]["x"][st:st + W] for k, st, _ in allw[i:i + chunk]]).astype(np.float32)
        m = map_channels(blk, "inv", True)
        if out is None:
            out = np.empty((len(allw),) + m.shape[1:], np.float32)
        out[i:i + len(m)] = m
    return out, seg_of, len(ev)


WIN_CACHE_VER = "w1"   # make_windows_cnn / map_channels 改了就换


def make_windows_cnn_cached(tls: dict, W: int, S: int, short_shifts: int, hz: float, axes: int):
    """make_windows_cnn 加硬盘缓存（TRAIN_V2_WIN_CACHE=<目录> 才开）。返回的数组是只读内存映射"""
    cache = os.environ.get("TRAIN_V2_WIN_CACHE")
    if not cache:
        return make_windows_cnn(tls, W, S, short_shifts)
    import hashlib
    h = hashlib.md5(f"{WIN_CACHE_VER}|{TL_CACHE_VER}|{TL_MARGIN_S}|{W}|{S}|{short_shifts}|{hz}|{axes}".encode())
    for pth, tl in tls.items():
        st = os.stat(pth)
        h.update(f"{pth}|{st.st_mtime_ns}|{st.st_size}|{tl.get('off', 0)}|{len(tl['x'])}".encode())
        h.update(np.ascontiguousarray(tl["seg"]).tobytes())
    d = os.path.join(cache, h.hexdigest())
    fx, fs, fn = (os.path.join(d, n) for n in ("X.npy", "seg.npy", "n_ev.txt"))
    if not os.path.exists(fn):
        Xm, seg_of, n_ev = make_windows_cnn(tls, W, S, short_shifts)
        os.makedirs(d, exist_ok=True)
        np.save(fx + ".tmp.npy", Xm)
        np.save(fs + ".tmp.npy", seg_of)
        del Xm
        os.replace(fx + ".tmp.npy", fx)
        os.replace(fs + ".tmp.npy", fs)
        with open(fn, "w") as f:
            f.write(str(n_ev))
        print(f"▶ 窗口存进硬盘缓存 {d}", flush=True)
    else:
        print(f"▶ 窗口从硬盘缓存载入（内存映射）{d}", flush=True)
    return np.load(fx, mmap_mode="r"), np.load(fs), int(open(fn).read())


def make_windows(tls: dict, segs: list[dict], W: int, S: int, short_shifts: int):
    X, seg_of = [], []
    for csv_path, tl in tls.items():
        x, sid, bad = tl["x"], tl["seg"], tl["bad"]
        n = len(x)
        ids = np.unique(sid[sid >= 0])
        for s in ids:
            idx = np.flatnonzero(sid == s)
            a, b = idx[0], idx[-1] + 1
            L = b - a
            if L >= W:
                starts = list(range(a, b - W + 1, S))
            else:
                mid = (a + b) // 2 - W // 2
                starts = [mid]
                room = (W - L) // 2
                for k in range(1, short_shifts + 1):
                    off = int(round(room * k / (short_shifts + 1)))
                    starts += [mid - off, mid + off]
            for st in starts:
                if st < 0 or st + W > n or bad[st:st + W].any():
                    continue
                X.append(x[st:st + W])
                seg_of.append(int(s))
    return np.asarray(X, np.float32), np.asarray(seg_of)


def featurize(X_raw, hz):
    from gravity_align import map_channels
    from features_vec import extract_features_vec

    Xm = map_channels(X_raw, "inv", True)
    return Xm, extract_features_vec(Xm, hz).astype(np.float32)


# 跟「这个项圈的加速度计怎么校准」有关、跟狗在干什么无关的特征：重力那一维的绝对大小。
# inv 通道里 acc_x = 竖直分量、acc_z = 模长，静止时两者都 ≈ 1g，但每个项圈的读数不一样
# （pool_20261107 上静止时 0.967 ~ 1.016g）。模型会学成「读数是多少 → 哪一类」：
# #29 的分组交叉验证里 imu13 摘下来静放的 12 段 94% 被判成活动，SHAP 看主要就是这几维。
# 去掉之后未佩戴 F1 0.95 → 0.99，抓挠也略好（见 ml_rf.json 的 masked_features）。
GRAVITY_LEVEL_FEATURES = r"^(acc_x|acc_z|acc_mag)_(mean|min|max|rms)$|^sma_acc$"


def gravity_level_mask(n_channels: int) -> tuple[np.ndarray, list[str]]:
    """哪几列是重力绝对大小。返回 (bool 掩码, 特征名)"""
    from features import feature_names

    names = feature_names(int(n_channels))
    m = np.array([bool(re.search(GRAVITY_LEVEL_FEATURES, n)) for n in names])
    return m, [n for n, k in zip(names, m) if k]


def blank_features(F: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """训练时把这几列置成常数 0：树永远不会在常数列上分裂，所以模型照样吃完整的特征向量，
    推理 / 端侧一行不用改（它们算出来的真实值传进来也用不上）。"""
    if mask.shape[0] != F.shape[1]:
        raise ValueError(f"特征维度对不上：掩码 {mask.shape[0]} 维，特征 {F.shape[1]} 维")
    F = F.copy()
    F[:, mask] = 0.0
    return F


# 「静止/休息」和「活动」按 IMU 动作强度分，不按人看视频的感觉分。
# pool_20261107 上 4 秒内 |a| 标准差 0.002~0.01g 那一档，人标的活动 8775 窗、静止 5745 窗，
# 289 段活动里 54 段大半时间狗根本没动（站着、坐着看）——项圈上看两者是同一种信号，
# 模型怎么训都分不开，整体准确率卡在 85% 上下。这里把两类的界线定成「身体动没动」：
# 只改标成这两类本身的窗口（进食 / 饮水 / 舔 这些细类不动），低于阈值算静止，高于算活动。
STILL, ACTIVE = "静止/休息", "活动"


def motion_of(Xm: np.ndarray) -> np.ndarray:
    """每个窗口的动作强度：inv 通道里加速度模长（第 2 列）在窗口内的标准差，单位 g"""
    return Xm[:, :, 2].std(axis=1)


def still_by_motion(y: np.ndarray, motion: np.ndarray, plain: np.ndarray, i_still: int, i_active: int,
                    thr: float) -> tuple[np.ndarray, dict]:
    """plain：这个窗口所在片段的训练类别就是 静止 或 活动 本身（不是细类）"""
    y = y.copy()
    to_still = plain & (y == i_active) & (motion < thr)
    to_active = plain & (y == i_still) & (motion >= thr)
    y[to_still], y[to_active] = i_still, i_active
    return y, {"活动→静止": int(to_still.sum()), "静止→活动": int(to_active.sum()),
               "两类窗口": int((plain & np.isin(y, [i_still, i_active])).sum())}


def forest_uses(model, mask: np.ndarray) -> list[int]:
    """森林里实际用到了哪些被屏蔽的列（应该是空的）"""
    idx = np.flatnonzero(mask)
    used = set()
    for est in model.estimators_:
        f = est.tree_.feature
        used |= set(int(x) for x in f[f >= 0])
    return sorted(used & set(int(i) for i in idx))


# ── 合并 + 校准做进树里 ─────────────────────────────────────────────────────

# ── XGBoost（端侧主力：algo_tinyml 的 GBDT 紧凑编码，同样大小下比随机森林准） ──
# 跟随机森林两处不同：
# 1. 直接训归并后的类别（不细分）。GBDT 多分类每轮每类一棵树、输出 softmax，没法像森林那样
#    把细类的概率在叶子里并回去；端上要的是归并后的那几类。
# 2. 输出校准做进第 k 类第一棵树的叶子：softmax 前给第 k 类的分数加 log(系数)，
#    跟「概率乘系数再归一化」完全等价。存出来就是普通的 XGBClassifier，端侧导出不用改。
XGB_DEFAULTS = dict(n_estimators=50, max_depth=6, learning_rate=0.2, subsample=0.8, colsample_bytree=0.8)


def xgb_params(params: dict, server: bool = False) -> dict:
    """server=True：服务端版本，不上板，深度不限（端侧紧凑编码的 127 节点限制管不着它）"""
    kw = dict(XGB_DEFAULTS)
    for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        if params.get(k) not in (None, ""):
            kw[k] = type(XGB_DEFAULTS[k])(params[k])
    if kw["max_depth"] > 6 and not server:
        # 端侧紧凑编码右孩子偏移只有 7 位：一棵树不能超过 127 个节点，深 6 正好 127
        sys.exit(f"XGBoost 深度最多 6（端侧紧凑编码一棵树 ≤127 个节点），给的是 {kw['max_depth']}")
    return kw


def fit_xgb(F: np.ndarray, y: np.ndarray, n_classes: int, kw: dict, n_jobs: int):
    """按类别均衡加权。返回 (模型, 这个模型认识的类别下标)——某一折里缺了某一类时 XGBoost
    要求标签连续，所以先压缩成连续的再训"""
    import xgboost as xgb

    cls, cnt = np.unique(y, return_counts=True)
    w = (len(y) / (len(cls) * cnt))[np.searchsorted(cls, y)]
    lut = np.full(n_classes, -1)
    lut[cls] = np.arange(len(cls))
    m = xgb.XGBClassifier(**kw, tree_method="hist", n_jobs=n_jobs, random_state=42)
    m.fit(F, lut[y], sample_weight=w)
    return m, cls


def xgb_proba(m, cls, F: np.ndarray, n_classes: int) -> np.ndarray:
    p = np.zeros((len(F), n_classes))
    p[:, cls] = m.predict_proba(F)
    return p


def bake_class_scale(model, scale: np.ndarray, check_x: np.ndarray | None = None):
    """第 k 类的第一棵树（树 k）每片叶子加 log(scale[k])。做完用 check_x 验一下：
    新概率 ∝ 旧概率 × scale（逐类的对数比应该处处等于 log scale 再差一个常数）"""
    if np.allclose(scale, 1.0):
        return model
    b = model.get_booster()
    raw = json.loads(bytes(b.save_raw("json")))
    gb = raw["learner"]["gradient_booster"]["model"]
    before = model.predict_proba(check_x) if check_x is not None else None
    for k, s in enumerate(scale):
        if s == 1.0:
            continue
        if int(gb["tree_info"][k]) != k:
            sys.exit(f"XGBoost 的树 {k} 不属于第 {k} 类（tree_info={gb['tree_info'][k]}），校准没法做进叶子")
        t, add = gb["trees"][k], float(np.log(s))
        for i, lc in enumerate(t["left_children"]):
            if lc == -1:
                t["split_conditions"][i] += add
                t["base_weights"][i] += add
    b.load_model(bytearray(json.dumps(raw).encode()))
    if before is not None:
        after = model.predict_proba(check_x)
        d = np.log(np.maximum(after, 1e-12)) - np.log(np.maximum(before, 1e-12)) - np.log(scale)[None, :]
        if np.abs(d - d.mean(1, keepdims=True)).max() > 1e-3:
            sys.exit("输出校准做进 XGBoost 叶子之后，概率跟「乘系数再归一化」对不上")
    return model


def xgb_nodes_and_features(model) -> tuple[int, set]:
    """总节点数 + 树里用到的特征下标"""
    n, used = 0, set()
    for d in model.get_booster().get_dump(dump_format="json"):
        stack = [json.loads(d)]
        while stack:
            nd = stack.pop()
            n += 1
            if "split" in nd:
                used.add(int(str(nd["split"]).lstrip("f")))
                stack += nd.get("children", [])
    return n, used


# ── 1D-CNN（端侧 int8：algo_tinyml 的 tm_prep + tm_runtime，结构就是 src/dl/models/cnn.py） ──
# [Conv1d(pad k//2) + BN + ReLU + MaxPool2 + Dropout] × 3 → flatten → Linear。直接吃原始窗口（inv 通道），
# 不算 193 维特征、不做 FFT；int8 约 30 KB。跟 XGBoost 一样直接训归并后的类别。
#
# 动作门（motion_gate）：int8 量化之后，静止（|a| 抖动 ~0.005g）和未佩戴（~0.0005g）只差一两个 LSB，
# CNN 根本看不见——#pool_20261116 交叉验证里 CNN 的静止 F1 只有 0.32。所以这两类不交给 CNN：
# 窗口内 |a| 标准差 < notworn_g → 未佩戴，< still_g → 静止（still_g 就是 still_motion_g，静止本来就是这么定义的），
# 其余窗口里这两类的概率清零、CNN 在剩下的类别里选。门限 notworn_g 在训练数据上挑（未佩戴 vs 静止分得最开的那个）。
# 平台推理（infer_csv_scratch._load_dl_model）和端侧（algo_tinyml host / 固件）用同一个门，参数存在 dl_cnn_best.json。
# 默认 [64,128,128]、k=7：数据池全部、只用加速度、分组 5 折比过 [32,64,128] k5 / 加宽 / k7 / 两两集成，
# 这个单模型跟集成一样好（相同误报下抓挠检测率 +2~3 点）；int8 约 180 KB，GR5515 的 1 MB flash 放得下
CNN_DEFAULTS = dict(filters=[64, 128, 128], kernel_size=7, dropout=0.2)
NOTWORN = "未佩戴"


def cnn_params(params: dict) -> tuple[dict, int]:
    cfg = dict(CNN_DEFAULTS)
    f = params.get("filters")
    if f:
        cfg["filters"] = [int(x) for x in (f.split(",") if isinstance(f, str) else f)]
    if params.get("kernel_size"):
        cfg["kernel_size"] = int(params["kernel_size"])
    if params.get("dropout") not in (None, ""):
        cfg["dropout"] = float(params["dropout"])
    if params.get("qat_epochs") not in (None, "", 0, "0"):
        cfg["qat_epochs"] = int(params["qat_epochs"])
    return cfg, int(params.get("epochs") or 20)


# ── 量化感知训练（QAT）────────────────────────────────────────────────────────
# 板上跑的是 algo_tinyml 训练后量化（tinyml/net.py quantize）出来的 int8：BN 折进卷积，
# 权重逐输出通道对称量化（max|w|/127），输入和每层输出按分位数范围做非对称 int8（256 格）。
# 直接训出来的 float 模型对这一步很敏感：2026-11-17 那版换个分位数，int8 跟 float 的判决一致率
# 就在 52%~89% 之间跳。QAT 在 float 训完之后再微调几轮，前向里按**同一套方案**模拟量化
# （直通估计器传梯度），让模型学会在 int8 下也判得对。存下来的还是普通的 state_dict，
# 导出照旧走训练后量化，不改端侧任何代码。
QAT_PCT = 99.9   # 激活范围的分位数，跟导出时最常选中的那档一致


def _fq_weight(w):
    """逐输出通道对称 int8（跟 net.quantize 的 w_scale 一样），直通估计器"""
    s = w.detach().abs().flatten(1).max(1).values.clamp_min(1e-12) / 127.0
    s = s.view(-1, *([1] * (w.dim() - 1)))
    return w + ((w / s).round().clamp(-127, 127) * s - w).detach()


def _fq_act(x, lo: float, hi: float):
    """非对称 int8（跟 net._affine 一样：0 要能精确表示），直通估计器"""
    import torch

    lo, hi = min(lo, 0.0), max(hi, 0.0)
    sc = max((hi - lo) / 255.0, 1e-8)
    zp = int(np.clip(round(-128 - lo / sc), -128, 127))
    q = (torch.round(x / sc) + zp).clamp(-128, 127)
    return x + ((q - zp) * sc - x).detach()


def _qat_forward(net, x, ranges: dict, update: bool):
    """跟 tinyml 的 FloatNet → QNet 同一个计算图：BN 用滑动统计量折进卷积 → relu → 量化 → 池化"""
    import torch
    import torch.nn.functional as F

    def rng(key, t):
        if update:
            flat = t.detach().flatten()
            if flat.numel() > 200_000:
                flat = flat[torch.randint(0, flat.numel(), (200_000,), device=flat.device)]
            tail = (100.0 - QAT_PCT) / 200.0
            lo, hi = torch.quantile(flat.float(), torch.tensor([tail, 1 - tail], device=flat.device)).tolist()
            if key in ranges:
                lo, hi = 0.9 * ranges[key][0] + 0.1 * lo, 0.9 * ranges[key][1] + 0.1 * hi
            ranges[key] = (lo, hi)
        return ranges[key]

    x = _fq_act(x, *rng("in", x))
    mods = list(net.conv)
    for i in range(0, len(mods), 5):
        conv, bn, drop = mods[i], mods[i + 1], mods[i + 4]
        inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * inv.view(-1, 1, 1)
        b = (conv.bias - bn.running_mean) * inv + bn.bias
        x = F.relu(F.conv1d(x, _fq_weight(w), b, padding=conv.padding[0]))
        x = _fq_act(x, *rng(f"c{i}", x))
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, drop.p, net.training)
    out = F.linear(x.flatten(1), _fq_weight(net.fc.weight), net.fc.bias)
    return _fq_act(out, *rng("fc", out))


def _qat_finetune(net, Xt, yt, lossf, dev, epochs: int, rng, shift: int):
    """float 训完之后接着微调：BN 统计量冻住（板上用的就是它们），只更新权重和 BN 的 gamma / beta"""
    import torch

    ranges: dict = {}
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-3)
    bs = 512
    nb = (len(yt) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs * nb))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(yt), device=Xt.device)
        for b in range(nb):
            ix = perm[b * bs:(b + 1) * bs]
            xb = Xt[ix].to(dev)
            xb = xb * (1 + 0.1 * torch.randn(len(ix), 1, 1, device=dev)) + 0.02 * torch.randn_like(xb)
            xb = torch.roll(xb, int(rng.integers(-shift, shift + 1)), 2)
            loss = lossf(_qat_forward(net, xb, ranges, update=True), yt[ix].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    return ranges


def _channel_stats(X: np.ndarray, idx: np.ndarray, chunk: int = 50000):
    """按块算每个通道的均值 / 标准差（float64 累加），不复制整个 X"""
    C = X.shape[2]
    s1, s2, n = np.zeros(C), np.zeros(C), 0
    for i in range(0, len(idx), chunk):
        b = X[idx[i:i + chunk]].reshape(-1, C).astype(np.float64)
        s1 += b.sum(0)
        s2 += (b * b).sum(0)
        n += len(b)
    mu = s1 / max(n, 1)
    sd = np.sqrt(np.maximum(s2 / max(n, 1) - mu * mu, 0))
    sd[sd < 1e-6] = 1.0
    return mu, sd


def _to_device_normalized(X: np.ndarray, idx: np.ndarray, mu, sd, dev, chunk: int = 50000):
    """标准化 + 转成 [N, C, T]，分块直接写进显存。显存放不下（留 4 GB 余量）就放内存"""
    import torch

    N, T, C = len(idx), X.shape[1], X.shape[2]
    need = N * T * C * 4
    where = dev
    if dev == "cuda":
        free, _total = torch.cuda.mem_get_info()
        if need > free - 4 * 1024 ** 3:
            where = "cpu"
            print(f"  ⚠ 显存不够放训练数据（要 {need / 1024 ** 3:.1f} GB，空 {free / 1024 ** 3:.1f} GB），放内存", flush=True)
    Xt = torch.empty((N, C, T), dtype=torch.float32, device=where)
    mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)
    for i in range(0, N, chunk):
        b = (X[idx[i:i + chunk]] - mu32) / sd32
        Xt[i:i + len(b)] = torch.from_numpy(np.ascontiguousarray(b.transpose(0, 2, 1))).to(where)
    return Xt


def fit_cnn(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: dict, epochs: int, seed: int = 0, idx=None):
    """X: [N, T, C] 原始窗口（inv 通道），y: 归并后的类别（对应 X[idx]）。idx 是训练用的那部分窗口（交叉验证的训练折），
    省掉先切一份副本。训练数据整份放显存（放不下才放内存）。返回 (网络（cpu、eval）, ch_mean, ch_std, 每轮曲线)"""
    import torch
    import torch.nn as nn

    sys.path.insert(0, os.path.join(_HERE, "..", "dl"))
    from models.cnn import CNN

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx = np.arange(len(X)) if idx is None else np.asarray(idx)
    C = X.shape[2]
    mu, sd = _channel_stats(X, idx)
    net = CNN(C, X.shape[1], n_classes, cfg).to(dev)
    Xt = _to_device_normalized(X, idx, mu, sd, dev)
    yt = torch.tensor(y, dtype=torch.long, device=Xt.device)
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    w = np.sqrt(len(y) / (n_classes * np.maximum(cnt, 1)))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev), label_smoothing=0.05)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    bs = 512
    nb = (len(y) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=max(1, epochs * nb))
    shift = max(1, X.shape[1] // 10)
    # 每轮的平均 loss / 训练准确率，平台「训练曲线」画这个：不额外跑前向，
    # 直接拿训练时每个 batch 的统计累加，几乎不花时间
    hist: list[dict] = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(y), device=Xt.device)
        tot, hit, seen = 0.0, 0, 0
        for b in range(nb):
            ix = perm[b * bs:(b + 1) * bs]
            xb = Xt[ix].to(dev)
            # 增强：幅度缩放、小噪声、整体平移（窗口里动作出现的位置不固定）
            xb = xb * (1 + 0.1 * torch.randn(len(ix), 1, 1, device=dev)) + 0.02 * torch.randn_like(xb)
            xb = torch.roll(xb, int(rng.integers(-shift, shift + 1)), 2)
            logits = net(xb)
            loss = lossf(logits, yt[ix].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            tot += float(loss) * len(ix)
            hit += int((logits.argmax(1) == yt[ix]).sum())
            seen += len(ix)
        hist.append({"epoch": ep + 1, "loss": round(tot / max(1, seen), 4),
                     "train_acc": round(hit / max(1, seen), 4)})
    if int(cfg.get("qat_epochs") or 0) > 0:
        _qat_finetune(net, Xt, yt, lossf, dev, int(cfg["qat_epochs"]), rng, shift)
    del Xt, yt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return net.cpu().eval(), mu.astype(np.float64), sd.astype(np.float64), hist


def cnn_proba(net, mu, sd, X: np.ndarray) -> np.ndarray:
    import torch

    out = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(dev)
    try:
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.tensor((X[i:i + 8192] - mu) / sd, dtype=torch.float32).permute(0, 2, 1).to(dev)
                out.append(torch.softmax(net(xb), 1).cpu().numpy())
    finally:
        net.cpu()
    return np.concatenate(out) if out else np.zeros((0, net.fc.out_features))


def bake_cnn_scale(net, scale: np.ndarray):
    """输出校准做进最后一层的偏置：logit 加 log(系数) ≡ 概率乘系数再归一化"""
    import torch

    with torch.no_grad():
        net.fc.bias += torch.tensor(np.log(scale), dtype=net.fc.bias.dtype)
    return net


def fit_motion_gate(mot: np.ndarray, y: np.ndarray, classes: list[str], still_g: float | None) -> dict | None:
    """未佩戴 / 静止 这两类按 |a| 抖动判。notworn_g 在训练窗口上挑：两类之间分得最准的那个阈值"""
    i_nw = classes.index(NOTWORN) if NOTWORN in classes else None
    i_st = classes.index(STILL) if (STILL in classes and still_g) else None
    if i_nw is None and i_st is None:
        return None
    best = (None, 0.0)
    if i_nw is not None:
        hi = still_g or 0.005
        nw, st = mot[y == i_nw], (mot[y == i_st] if i_st is not None else np.zeros(0))
        best = (-1.0, 0.0008)
        for t in np.geomspace(1e-4, hi, 60):
            acc = ((nw < t).sum() + (st >= t).sum()) / max(len(nw) + len(st), 1)
            if acc > best[0]:
                best = (float(acc), float(t))
    return {"channel": 2, "notworn_g": round(best[1], 6), "still_g": still_g if i_st is not None else None,
            "notworn": i_nw, "still": i_st,
            "train_acc_notworn_vs_still": round(best[0], 4) if best[0] is not None else None}


def apply_motion_gate(P: np.ndarray, mot: np.ndarray, gate: dict | None) -> np.ndarray:
    if not gate:
        return P
    P = P.copy()
    i_nw, i_st = gate.get("notworn"), gate.get("still")
    low = (mot < gate["notworn_g"]) if i_nw is not None else np.zeros(len(P), bool)
    still = ((mot < gate["still_g"]) & ~low) if (i_st is not None and gate.get("still_g")) else np.zeros(len(P), bool)
    moving = ~low & ~still
    for i in (i_nw, i_st if gate.get("still_g") else None):
        if i is not None:
            P[moving, i] = 0.0
    P[moving] /= np.maximum(P[moving].sum(1, keepdims=True), 1e-12)
    for m, i in ((low, i_nw), (still, i_st)):
        if i is not None and m.any():
            P[m] = 0.0
            P[m, i] = 1.0
    return P


# ── 片段级评估（经过后处理之后）────────────────────────────────────────────
# 窗口对了不代表片段对：片段还要过稳定版 v2 的后处理（viterbi 解码、合并间隔、最短窗口数 / 平均概率门槛、
# 抓挠吞并甩身体……），这一步会改片段的个数和起止。用户最后看到的是片段，所以按片段评：
# 交叉验证里每个留出文件，在它的标注范围内按步长连续推理（没见过这个文件的模型）→ 平台同一份
# label_service/postprocess.stabilize（viterbi）→ 出来的片段跟人标的同类片段配对。
EVENT_EVAL = ("抓挠", "甩头/抖身")


def labeled_span_windows(tl: dict, segs_of_file: list[dict], W: int, S: int, hz: float):
    """这个文件人标过的范围 [最早开始, 最晚结束] 里按步长切窗口（跳过掉包）。返回 (窗口, 起点下标, (lo, hi) 秒)"""
    ss = [s for s in segs_of_file if s.get("start_ms") is not None]
    if not ss:
        return None
    lo, hi = min(s["start_ms"] for s in ss) / 1000, max(s["end_ms"] for s in ss) / 1000
    x, bad = tl["x"], tl["bad"]
    off = int(tl.get("off", 0))     # 时间线裁过：下标都按原始文件算，取数时减 off
    N = off + len(x)
    if len(x) < W:
        return None  # 整个文件比一个窗口还短（公开数据集里有几秒长的录制）：切不出完整窗口
    a0, a1 = max(off, int(lo * hz) - W // 2), min(N - W, int(hi * hz))
    a0 = min(a0, N - W)        # 标注起点贴着文件尾：别切出不够一个窗口的半截
    st = [s for s in range(a0, max(a0, a1) + 1, S) if not bad[s - off:s - off + W].any()]
    if not st:
        return None
    return np.stack([x[s - off:s - off + W] for s in st]).astype(np.float32), np.asarray(st), (lo, hi)


def _iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def _ov(a, b):
    return min(a[1], b[1]) - max(a[0], b[0]) > 0


# 事件查全诊断：真值段按持续时间分桶，看漏的是短段还是长段
LEN_BUCKETS = ((0.0, 3.0, "lt3"), (3.0, 6.0, "3to6"), (6.0, 10.0, "6to10"), (10.0, None, "ge10"))


def _len_bucket(d: float) -> str:
    for lo, hi, name in LEN_BUCKETS:
        if d >= lo and (hi is None or d < hi):
            return name
    return LEN_BUCKETS[0][2]


def segment_metrics(per_file: list[dict], classes: list[str]) -> dict:
    """per_file: [{site, gt: {类: [(s,e)]}, pred: {类: [(s,e)]}, other: {类: [(s,e)] 标成别的类的时间}}]

    事件级（跟 pyannote.metrics 的 Detection 一个思路）：
      - 同类 + IoU ≥ 0.5 才算命中，一对一匹配（按 IoU 从大到小贪心，和 pyannote 的匹配顺序一致）
      - recall_iou50 = 命中的真值段 / 真值段；precision_iou50 = 命中的预测段 / 预测段
      - 真值一段被拆成几段、或只盖住一部分（IoU 不够），只算一次命中，其余预测段算误报
    """
    out = {}
    for c in EVENT_EVAL:
        if c not in classes:
            continue
        n_gt = hit = n_iou = fp = n_pred_judged = n_pred = 0
        other_h = 0.0
        on, off, cnt = [], [], []
        by_len = {name: {"n_gt": 0, "hit": 0, "iou50": 0} for _, _, name in LEN_BUCKETS}
        frag, cover = [], []        # 每个有任何重叠的真值段：被几个预测段盖住、盖住的比例
        sites: dict[str, dict[str, int]] = {}
        for f in per_file:
            gt, pr, other = f["gt"].get(c, []), f["pred"].get(c, []), f["other"].get(c, [])
            st = sites.setdefault(str(f.get("site") or ""), {"n_gt": 0, "n_pred": 0, "matched": 0})
            n_gt += len(gt)
            n_pred += len(pr)
            st["n_gt"] += len(gt)
            st["n_pred"] += len(pr)
            hit += sum(any(_ov(g, p) for p in pr) for g in gt)
            pairs = sorted(((_iou(g, p), i, j) for i, g in enumerate(gt) for j, p in enumerate(pr) if _iou(g, p) >= 0.5), reverse=True)
            ug, up = set(), set()
            for _, i, j in pairs:
                if i not in ug and j not in up:
                    ug.add(i); up.add(j)
                    on.append(pr[j][0] - gt[i][0]); off.append(pr[j][1] - gt[i][1])
            n_iou += len(ug)
            st["matched"] += len(ug)
            # 诊断：漏的是短段还是长段；命中段被切碎成几块、盖住多少
            for i, g in enumerate(gt):
                b = by_len[_len_bucket(g[1] - g[0])]
                b["n_gt"] += 1
                if i in ug:
                    b["iou50"] += 1
                ovs = sorted(p for p in pr if _ov(g, p))
                if not ovs:
                    continue
                b["hit"] += 1
                frag.append(len(ovs))
                cur0, cur1, cov = ovs[0][0], ovs[0][1], 0.0
                for p in ovs[1:]:
                    if p[0] <= cur1:
                        cur1 = max(cur1, p[1])
                    else:
                        cov += cur1 - cur0; cur0, cur1 = p[0], p[1]
                cov += cur1 - cur0
                cover.append(min(1.0, cov / max(g[1] - g[0], 1e-6)))
            other_h += sum(e - s for s, e in other) / 3600
            for p in pr:
                tp = any(_ov(p, g) for g in gt)
                in_other = any(_ov(p, o) for o in other)
                if tp or in_other:           # 落在没标的时间里的不算对也不算错（那里没标不等于没有）
                    n_pred_judged += 1
                    fp += (not tp) and in_other
            if gt:
                cnt.append(len(pr) - len(gt))
        med = (lambda v: round(float(np.median(np.abs(v))), 2) if v else None)
        recall_all = round(n_iou / max(n_gt, 1), 4)
        precision_all = round(n_iou / n_pred, 4) if n_pred else None

        def _site_row(s: dict[str, int]) -> dict:
            rec = round(s["matched"] / s["n_gt"], 3) if s["n_gt"] else None
            pre = round(s["matched"] / s["n_pred"], 3) if s["n_pred"] else None
            return {"n_gt": s["n_gt"], "n_pred": s["n_pred"], "matched": s["matched"],
                    "recall": rec, "precision": pre, "f1": _prf(rec, pre)}

        out[c] = {"n_gt": n_gt, "n_pred": n_pred, "recall_hit": round(hit / max(n_gt, 1), 4), "recall_iou50": recall_all,
                  "precision": round(1 - fp / max(n_pred_judged, 1), 4), "precision_iou50": precision_all,
                  "f1_iou50": _prf(recall_all, precision_all) if precision_all is not None else None,
                  "fp_per_h_other": round(fp / max(other_h, 1e-9), 2),
                  "other_hours": round(other_h, 1), "onset_abs_err_s": med(on), "offset_abs_err_s": med(off),
                  "count_diff_mean_abs": round(float(np.mean(np.abs(cnt))), 2) if cnt else None,
                  # 按场地拆（佛山 / 影棚 / 深圳），跟 per_site 的 key 对齐；public_datasets 是空 key，不展示
                  "per_site": {k: _site_row(v) for k, v in sites.items() if v["n_gt"] or v["n_pred"]},
                  # 诊断：真值段按时长分桶（<3s / 3–6s / 6–10s / ≥10s）。
                  # iou50_given_hit 低 = 盖到了但边界/时长差太多（后处理问题）；hit_rate 低 = 整段没报（模型问题）
                  "by_len": {k: {"n_gt": v["n_gt"], "hit": v["hit"], "iou50": v["iou50"],
                                 "hit_rate": round(v["hit"] / v["n_gt"], 3),
                                 "iou50_rate": round(v["iou50"] / v["n_gt"], 3),
                                 "iou50_given_hit": round(v["iou50"] / v["hit"], 3) if v["hit"] else None}
                             for k, v in by_len.items() if v["n_gt"]},
                  # 有重叠的真值段：被几个预测段盖住（碎片）、合起来盖住的比例（中位数）
                  "frag": {"n": len(frag),
                           "preds_per_hit_median": round(float(np.median(frag)), 2) if frag else None,
                           "cover_median": round(float(np.median(cover)), 3) if cover else None}}
    return out


def add_gap_negatives(segs: list[dict], cfg: dict, scratch: str = "抓挠", neg: str = "活动") -> int:
    """每个文件标注范围内的空档 → 「活动」伪片段（pseudo=True）。返回加了多少段"""
    m_scr = float(cfg.get("margin_scr", 30)) * 1000
    m_any = float(cfg.get("margin_any", 2)) * 1000
    min_len = float(cfg.get("min_len", 8)) * 1000
    ratio = float(cfg.get("max_ratio", 1.0))
    by: dict[str, list[dict]] = {}
    for x in segs:
        if x.get("start_ms") is not None and x.get("end_ms") is not None:
            by.setdefault(x["csv"], []).append(x)
    add = []
    for csv, ss in by.items():
        lo, hi = min(x["start_ms"] for x in ss), max(x["end_ms"] for x in ss)
        block = sorted([(x["start_ms"] - m_any, x["end_ms"] + m_any) for x in ss] +
                       [(x["start_ms"] - m_scr, x["end_ms"] + m_scr) for x in ss if x["coarse"] == scratch])
        gaps, cur = [], lo
        for a, b in block:
            if a > cur:
                gaps.append((cur, min(a, hi)))
            cur = max(cur, b)
        if cur < hi:
            gaps.append((cur, hi))
        gaps = [(a, b) for a, b in gaps if b - a >= min_len]
        budget = ratio * sum(x["end_ms"] - x["start_ms"] for x in ss)
        ref = ss[0]
        for a, b in gaps:
            if budget <= 0:
                break
            b = min(b, a + budget)
            if b - a < min_len:
                break
            budget -= b - a
            add.append({**{k: ref.get(k) for k in ("csv", "sample_code", "dataset", "source_hz")},
                        "start_ms": a, "end_ms": b, "start": None, "end": None,
                        "name": neg, "coarse": neg, "leaf": "未标时间", "pseudo": True})
    segs.extend(add)
    return len(add)


def window_phys(Xraw: np.ndarray, hz: float) -> np.ndarray:
    """每窗物理特征（跟 label_service/pool.py 的质量门限同口径）：[模长均值, 陀螺仪模长 std, 加速度模长 2–10 Hz 节律占比]。
    3 轴数据陀螺仪那一列是 NaN"""
    n, T = Xraw.shape[0], Xraw.shape[1]
    am = np.linalg.norm(Xraw[:, :, :3], axis=2)
    out = np.full((n, 3), np.nan, np.float32)
    out[:, 0] = am.mean(1)
    if Xraw.shape[2] >= 6:
        out[:, 1] = np.linalg.norm(Xraw[:, :, 3:6], axis=2).std(1)
    a = am - am.mean(1, keepdims=True)
    pw = np.abs(np.fft.rfft(a * np.hanning(T)[None, :], axis=1)) ** 2
    f = np.fft.rfftfreq(T, 1 / hz)
    tot = pw[:, (f >= 0.5) & (f <= 15)].sum(1)
    out[:, 2] = np.divide(pw[:, (f >= 2) & (f <= 10)].sum(1), tot, out=np.zeros_like(tot), where=tot > 0)
    return out


def postprocess_segments(P: np.ndarray, starts: np.ndarray, t0, classes: list[str], W: int, S: int, hz: float, repo: str):
    """一个文件的逐窗口概率 → 稳定版 v2（viterbi，平台 STABLE_* 参数）→ {类: [(开始秒, 结束秒)]}"""
    from datetime import timedelta

    if repo not in sys.path:
        sys.path.insert(0, repo)
    from label_service import config as ls_cfg, postprocess as pp

    sp = pp.StableParams(event_labels=tuple(ls_cfg.STABLE_EVENT_LABELS), smooth_windows=ls_cfg.STABLE_SMOOTH_WINDOWS,
                         min_state_s=ls_cfg.STABLE_MIN_STATE_S, event_enter=ls_cfg.STABLE_EVENT_ENTER,
                         event_stay=ls_cfg.STABLE_EVENT_STAY, event_gap_s=ls_cfg.STABLE_EVENT_GAP_S,
                         shake_absorb_s=ls_cfg.STABLE_SHAKE_ABSORB_S, event_min_windows=ls_cfg.STABLE_EVENT_MIN_WINDOWS,
                         event_min_mean=ls_cfg.STABLE_EVENT_MIN_MEAN, event_single_conf=ls_cfg.STABLE_EVENT_SINGLE_CONF,
                         spectral_min=ls_cfg.STABLE_SPECTRAL_MIN, viterbi_switch=ls_cfg.STABLE_VITERBI_SWITCH,
                         notworn_min_s=ls_cfg.STABLE_NOTWORN_MIN_S)
    wins = [{"ts": pp._fmt_ts(t0 + timedelta(seconds=float(s) / hz)), "label": classes[int(q.argmax())], "conf": float(q.max()),
             "probs": {c: float(v) for c, v in zip(classes, q)}} for s, q in zip(starts, P)]
    targets = [c for c in EVENT_EVAL if c in classes]
    segs = pp.stabilize(wins, classes, targets, W / hz, S / hz, "segment", sp, algo="viterbi")
    return {c: [((pp._parse_ts(x["start_ts"]) - t0).total_seconds(), (pp._parse_ts(x["end_ts"]) - t0).total_seconds())
                for x in segs.get(c, [])] for c in targets}


def merge_forest(model, fine_names: list[str], coarse_names: list[str], back: list[int], scale: np.ndarray):
    """训练用的细类森林 → 归并后类别的森林：叶子里细类的概率加到它归并到的那一类，
    再乘校准系数、每片叶子重新归一化。存出来就是普通的 sklearn 随机森林"""
    from sklearn.tree._tree import Tree

    K = len(coarse_names)
    # 行数按**森林实际见过的类别**，不是全部细类：某一折的训练数据里恰好没有某个细类（比如饮水）时，
    # 叶子里的概率列就少一列，按全部细类开矩阵会维度对不上（交叉验证当场崩）
    M = np.zeros((len(model.classes_), K))
    for i, c in enumerate(model.classes_):
        M[i, back[int(c)]] = 1.0
    for est in model.estimators_:
        tr = est.tree_
        st = tr.__getstate__()
        v = st["values"][:, 0, :]
        v = v / np.maximum(v.sum(1, keepdims=True), 1e-12)
        nv = (v @ M) * scale[None, :]
        nv = nv / np.maximum(nv.sum(1, keepdims=True), 1e-12)
        new = Tree(tr.n_features, np.array([K], dtype=np.intp), 1)
        st = dict(st)
        st["values"] = np.ascontiguousarray(nv[:, None, :], dtype=np.float64)
        new.__setstate__(st)
        est.tree_ = new
        est.n_classes_ = K
        est.classes_ = np.arange(K)
        if hasattr(est, "n_outputs_"):
            est.n_outputs_ = 1
    model.n_classes_ = K
    model.classes_ = np.arange(K)
    return model


# ── 主流程 ──────────────────────────────────────────────────────────────────

def _group_of(sample_code: str) -> str:
    imu = re.search(r"imu\d+", sample_code or "", re.I)
    day = re.search(r"_(\d{8})_", sample_code or "")
    return f"{day.group(1) if day else '?'}_{imu.group(0).lower() if imu else sample_code}"


# 抓挠按场地分开看。训练数据混了公司影棚 / 深圳雷诺 / 佛山博朗旺几个场地，
# 同一个模型在不同场地的实际表现差得很远；合在一起的 P/R/F1 会被样本多的
# 那个场地盖住，看不出「佛山很准、影棚几乎判不出来」这种情况。
# 场地从 CSV 路径认：采集端从 2026-10 起把场地写进日期目录
# （2026_11_16_bolangwang / _yingpeng），深圳那段是 _gouchang。
# 7~9 月的老目录还没有后缀，那时只在影棚采，按影棚算；公开数据集不是场地数据，不算。
SITE_NAMES = {"bolangwang": "佛山博朗旺", "yingpeng": "公司影棚", "gouchang": "深圳雷诺"}


def _site_of_csv(p: str) -> str | None:
    low = (p or "").lower()
    if "bolangwang" in low or "博朗旺" in low:
        return "bolangwang"
    if "gouchang" in low or "雷诺" in low:
        return "gouchang"
    if "yingpeng" in low or "影棚" in low:
        return "yingpeng"
    if "public_datasets" in low:
        return None
    return "yingpeng"


def _prf(rec: float | None, pre: float | None) -> float | None:
    """查全 / 查准 → F1；两边都缺（这一类在这个场地没有样本）就是 None"""
    if rec is None or pre is None or rec + pre <= 0:
        return None
    return round(2 * rec * pre / (rec + pre), 3)


def main(a):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from diagnose import confusion as diag_confusion

    with open(a.spec, encoding="utf-8") as f:
        spec = json.load(f)
    hz = float(spec["hz"])
    axes = 6 if int(spec.get("axes") or 3) == 6 else 3
    classes = list(spec["classes"])
    remap = spec.get("label_remap") or {}
    fine = [x for x in (spec.get("fine_labels") if spec.get("fine_labels") is not None else DEFAULT_FINE)]
    scale_cfg = spec.get("class_scale") or {}
    scale = np.array([float(scale_cfg.get(c, 1.0)) for c in classes])
    W = int(spec["window"])
    S = max(1, int(round(float(spec.get("stride_s") or 1.0) * hz)))
    params = spec.get("model_params") or {}
    rf_kw = dict(n_estimators=int(params.get("n_estimators") or 20), max_depth=params.get("max_depth", 8),
                 min_samples_leaf=int(params.get("min_samples_leaf") or 1),
                 max_features=params.get("max_features") or "sqrt", class_weight="balanced",
                 n_jobs=int(a.n_jobs), random_state=42)
    if rf_kw["max_depth"] is not None:
        rf_kw["max_depth"] = int(rf_kw["max_depth"])
    kind = str(spec.get("model") or "rf")
    if kind not in ("rf", "xgb", "cnn"):
        sys.exit(f"新版流程的 model 只能是 rf / xgb / cnn，给的是 {kind!r}")
    # target：edge = 要导到板上（默认）；server = 只在服务端跑，放开端侧的体积限制
    target = str(spec.get("target") or "edge")
    xgb_kw = xgb_params(params, server=target == "server") if kind == "xgb" else None
    cnn_cfg, cnn_epochs = cnn_params(params) if kind == "cnn" else (None, 0)
    if kind in ("xgb", "cnn"):
        fine = []          # GBDT / CNN 直接训归并后的类别，见 XGB_DEFAULTS 上面的说明
    run = spec["run_name"]
    drop_level = bool(spec.get("drop_gravity_level"))
    still_g = float(spec.get("still_motion_g") or 0) or None
    t0 = time.time()
    print("=" * 60)
    print(f"  新版训练流程（v2）  {axes} 轴  采样率 {hz:g}Hz  窗口 {W} 点（{W / hz:.2f}s）  步长 {S} 点（{S / hz:.2f}s）")
    print(f"  类别：{classes}")
    print(f"  细分训练、合并输出：{fine or '不用'}")
    print(f"  输出校准：{ {c: s for c, s in zip(classes, scale) if s != 1} or '不用'}")
    print(f"  XGBoost：{xgb_kw}" if kind == "xgb" else
          f"  1D-CNN：{cnn_cfg}，{cnn_epochs} 轮" if kind == "cnn" else f"  随机森林：{rf_kw}")
    print("=" * 60, flush=True)

    # 1. 片段
    segs = collect_segments(spec["datasets"], spec["nas_root"], remap, classes, fine)
    if not segs:
        sys.exit("没有可训练的片段：检查类别归并表")
    if spec.get("gap_negatives"):
        n_gap = add_gap_negatives(segs, spec["gap_negatives"])
        print(f"▶ 未标时间当负例：加了 {n_gap} 段「活动」伪片段（{spec['gap_negatives']}）", flush=True)
    for i, s in enumerate(segs):
        s["idx"] = i
    names = list(classes) + sorted({s["name"] for s in segs} - set(classes))
    back = [classes.index(n) if n in classes else None for n in names]
    for i, n in enumerate(names):
        if back[i] is None:
            # 细类并回哪一类：看它在片段里归并成了什么
            back[i] = classes.index(next(s["coarse"] for s in segs if s["name"] == n))
    print(f"▶ 片段 {len(segs)} 段，训练类别 {names}", flush=True)

    # 2. 时间线（多进程读 CSV）
    by_csv: dict[str, list] = {}
    for s in segs:
        by_csv.setdefault(s["csv"], []).append(s)
    tls, skipped, missing = {}, 0, []
    with ProcessPoolExecutor(max_workers=max(1, int(a.workers))) as ex:
        for i, (p, tl, err) in enumerate(ex.map(_build_one, [(p, ss, hz, axes, bool(spec.get("upsample"))) for p, ss in by_csv.items()], chunksize=4)):
            if err:
                missing.append(err)
            elif tl is None:
                skipped += 1
            else:
                tls[p] = tl
            if (i + 1) % 50 == 0:
                print(f"  读 CSV {i + 1}/{len(by_csv)}", flush=True)
    print(f"▶ CSV {len(tls)} 个可用" + (f"，{skipped} 个跳过（采样率低于 {hz:g}Hz{'、或没有陀螺仪' if axes == 6 else ''}）" if skipped else "")
          + (f"，{len(missing)} 个找不到" if missing else ""), flush=True)
    for m in missing[:5]:
        print(f"  ⚠ {m}")

    # 3. 窗口 + 特征：训练用的（短片段左右挪）和算指标用的（不挪）
    if kind == "cnn":
        # CNN 直接吃方向无关变换后的原始窗口，193 维手工特征（给 XGB / RF 的）用不上——百万级窗口时这一步要好几分钟。
        # 边切边变换，验证窗口 = 训练数组前 n_ev 个（切片，不复制）：几百万窗口时省一半以上内存
        Xtr_m, seg_tr, n_ev = make_windows_cnn_cached(tls, W, S, 2, hz, axes)
        Xev_m, seg_ev = Xtr_m[:n_ev], seg_tr[:n_ev]
        F_tr = F_ev = None
        print(f"▶ 窗口数组 {Xtr_m.nbytes / 1024 ** 3:.1f} GB（验证窗口共用前 {n_ev} 个）", flush=True)
    else:
        Xtr_raw, seg_tr = make_windows(tls, segs, W, S, short_shifts=2)
        Xev_raw, seg_ev = make_windows(tls, segs, W, S, short_shifts=0)
        Xtr_m, F_tr = featurize(Xtr_raw, hz)
        Xev_m, F_ev = featurize(Xev_raw, hz)
    masked = []
    if drop_level:
        fmask, masked = gravity_level_mask(Xev_m.shape[2])
        if F_tr is not None:
            F_tr, F_ev = blank_features(F_tr, fmask), blank_features(F_ev, fmask)
        print(f"▶ 不用重力绝对大小（跟项圈校准有关）：屏蔽 {len(masked)} 维 {masked}", flush=True)
    name_idx = {n: i for i, n in enumerate(names)}
    back_arr = np.asarray(back)
    yf_tr = np.array([name_idx[segs[s]["name"]] for s in seg_tr])
    y_ev = np.array([classes.index(segs[s]["coarse"]) for s in seg_ev])
    still_info = None
    if still_g:
        if STILL not in classes or ACTIVE not in classes:
            sys.exit(f"按动作强度分静止 / 活动要求类别里同时有「{STILL}」和「{ACTIVE}」")
        plain_tr = np.array([segs[s]["name"] in (STILL, ACTIVE) for s in seg_tr])
        plain_ev = np.array([segs[s]["name"] in (STILL, ACTIVE) for s in seg_ev])
        yf_tr, _ = still_by_motion(yf_tr, motion_of(Xtr_m), plain_tr, name_idx[STILL], name_idx[ACTIVE], still_g)
        y_ev, still_info = still_by_motion(y_ev, motion_of(Xev_m), plain_ev, classes.index(STILL), classes.index(ACTIVE), still_g)
        print(f"▶ 静止 / 活动按动作强度分（4 秒内 |a| 标准差 < {still_g:g}g 算静止）：{still_info}", flush=True)
    g_tr = np.array([_group_of(segs[s]["sample_code"]) for s in seg_tr])
    g_ev = np.array([_group_of(segs[s]["sample_code"]) for s in seg_ev])
    print(f"▶ 窗口：训练 {len(yf_tr)}（含短片段增强），验证 {len(y_ev)}；{len(set(g_ev))} 组（天 × 项圈）", flush=True)
    for c in range(len(classes)):
        print(f"  {classes[c]:<8} {int((y_ev == c).sum()):>7} 窗")

    gate = None
    if kind == "cnn":
        mot_tr, mot_ev = motion_of(Xtr_m), motion_of(Xev_m)
        gate = fit_motion_gate(mot_tr, back_arr[yf_tr], classes, still_g)
        print(f"▶ 动作门：{gate}", flush=True)

    # 4. 分组交叉验证 → 指标
    n_groups = len(set(g_ev))
    folds = min(5, n_groups)
    proba = np.zeros((len(y_ev), len(classes)))
    seg_files: list[dict] = []
    holdout = None
    # 每折的成绩 + CNN 的逐轮曲线，平台「训练曲线」看收敛过程
    curves: dict = {"folds": []}
    if folds >= 2:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
        seg_dump = [] if os.environ.get("TRAIN_V2_DUMP_SEG") else None
        # CNN 不算手工特征（F_ev 是 None）：分折只看标签和分组，X 用个占位
        for k, (_, te) in enumerate(cv.split(F_ev if F_ev is not None else np.zeros((len(y_ev), 1)), y_ev, g_ev)):
            test_g = set(g_ev[te])
            tr = ~np.isin(g_tr, list(test_g))
            fold_hist = None
            def _feats(Xraw):
                if kind == "cnn":
                    return _map_inv_chunked(Xraw), None
                Xm_, F_ = featurize(Xraw, hz)
                return Xm_, (blank_features(F_, fmask) if drop_level else F_)
            if kind == "xgb":
                m, mc = fit_xgb(F_tr[tr], back_arr[yf_tr[tr]], len(classes), xgb_kw, int(a.n_jobs))
                def _pred(Xm_, F_, m=m, mc=mc):
                    p_ = xgb_proba(m, mc, F_, len(classes)) * scale[None, :]
                    return p_ / p_.sum(1, keepdims=True)
            elif kind == "cnn":
                net, mu, sd, fold_hist = fit_cnn(Xtr_m, back_arr[yf_tr[tr]], len(classes), cnn_cfg, cnn_epochs, seed=k,
                                                 idx=np.flatnonzero(tr))
                def _pred(Xm_, F_, net=net, mu=mu, sd=sd):
                    p_ = cnn_proba(net, mu, sd, Xm_) * scale[None, :]
                    return apply_motion_gate(p_ / p_.sum(1, keepdims=True), motion_of(Xm_), gate)
            else:
                m = merge_forest(RandomForestClassifier(**rf_kw).fit(F_tr[tr], yf_tr[tr]), names, classes, back, scale)
                def _pred(Xm_, F_, m=m):
                    return m.predict_proba(F_)
            proba[te] = _pred(Xev_m[te], F_ev[te] if F_ev is not None else None)
            if len(te):
                pred_te = proba[te].argmax(1)
                curves["folds"].append({
                    "fold": k + 1,
                    "accuracy": round(float((pred_te == y_ev[te]).mean()), 4),
                    "macro_f1": round(float(f1_score(y_ev[te], pred_te, average="macro", zero_division=0)), 4),
                    **({"history": fold_hist} if fold_hist else {}),
                })
            # 片段级（后处理之后）：这一折留出的文件，在标注范围内连续推理
            for p_csv, tl in tls.items():
                ss = by_csv.get(p_csv) or []
                if not ss or _group_of(ss[0]["sample_code"]) not in test_g:
                    continue
                cw = labeled_span_windows(tl, ss, W, S, hz)
                if cw is None:
                    continue
                Xc, st_c, _span = cw
                Xm_c, F_c = _feats(Xc)
                P_c = _pred(Xm_c, F_c)
                # 代码根目录（label_service 在这里），不是 --repo（产物目录，测试里是临时目录）
                pred_seg = postprocess_segments(P_c, st_c, tl["t0"], classes, W, S, hz,
                                                os.path.abspath(os.path.join(_HERE, "..", "..")))
                if seg_dump is not None:
                    seg_dump.append({"csv": p_csv, "site": _site_of_csv(p_csv) or "", "dataset": ss[0].get("dataset"),
                                     "fold": k, "t0": tl["t0"].isoformat(), "starts": np.asarray(st_c, np.int32),
                                     "P": P_c.astype(np.float16), "phys": window_phys(Xc, hz).astype(np.float16),
                                     "segs": [(s_["start_ms"] / 1000, s_["end_ms"] / 1000, s_["coarse"], s_.get("name"), s_.get("leaf"))
                                              for s_ in ss if s_.get("start_ms") is not None and not s_.get("pseudo")]})
                hs = [(s["start_ms"] / 1000, s["end_ms"] / 1000, s["coarse"]) for s in ss
                      if s.get("start_ms") is not None and not s.get("pseudo")]
                seg_files.append({"site": _site_of_csv(p_csv) or "",
                                  "gt": {c: [(x0, x1) for x0, x1, cc in hs if cc == c] for c in EVENT_EVAL},
                                  "other": {c: [(x0, x1) for x0, x1, cc in hs if cc != c] for c in EVENT_EVAL},
                                  "pred": pred_seg})
            if holdout is None:
                holdout = te
            print(f"  交叉验证 {k + 1}/{folds}", flush=True)
        pred = proba.argmax(1)
        if seg_dump is not None:
            import gzip
            import pickle
            with gzip.open(os.environ["TRAIN_V2_DUMP_SEG"], "wb") as f:
                pickle.dump({"classes": classes, "W": W, "S": S, "hz": hz, "files": seg_dump}, f)
            print(f"▶ 交叉验证连续推理存到 {os.environ['TRAIN_V2_DUMP_SEG']}（{len(seg_dump)} 个文件）", flush=True)
        # TRAIN_V2_DUMP_OOF=<文件.npz>：把交叉验证里每个验证窗口的预测存下来（查误报 / 漏报用）。
        # seg 是窗口所属片段在 segs 里的下标，片段信息（csv / 起止 / 标签 / 数据集）一起存成 json
        dump = os.environ.get("TRAIN_V2_DUMP_OOF")
        if dump:
            np.savez_compressed(dump, proba=proba.astype(np.float32), y=y_ev, seg=np.asarray(seg_ev),
                                classes=np.array(classes), stride_s=S / hz, window_s=W / hz)
            with open(dump + ".segs.json", "w", encoding="utf-8") as f:
                json.dump([{k: s_.get(k) for k in ("csv", "start_ms", "end_ms", "name", "coarse", "leaf", "sample_code", "dataset")}
                           for s_ in segs], f, ensure_ascii=False)
            print(f"▶ 交叉验证逐窗预测存到 {dump}", flush=True)
    else:
        print("  ⚠ 只有一组数据，没法分组交叉验证，指标留空")
        pred = None

    # 5. 全量训练
    if kind == "xgb":
        model, mc = fit_xgb(F_tr, back_arr[yf_tr], len(classes), xgb_kw, int(a.n_jobs))
        if len(mc) != len(classes):
            sys.exit(f"全部数据里缺类别 {[classes[i] for i in set(range(len(classes))) - set(mc.tolist())]}，"
                     "XGBoost 导到端侧要求每一类都有")
        model = bake_class_scale(model, scale, F_ev[:2000])
        n_nodes, used = xgb_nodes_and_features(model)
        n_trees = len(model.get_booster().get_dump())
        print(f"▶ 全量训练完成：{n_trees} 棵树（{xgb_kw['n_estimators']} 轮 × {len(classes)} 类），{n_nodes} 个节点", flush=True)
        bad = sorted(used & set(np.flatnonzero(fmask).tolist())) if drop_level else []
    elif kind == "cnn":
        yc = back_arr[yf_tr]
        if len(set(yc.tolist())) != len(classes):
            sys.exit(f"全部数据里缺类别 {[classes[i] for i in set(range(len(classes))) - set(yc.tolist())]}")
        model, ch_mean, ch_std, final_hist = fit_cnn(Xtr_m, yc, len(classes), cnn_cfg, cnn_epochs, seed=0)
        curves["final"] = final_hist
        model = bake_cnn_scale(model, scale)
        n_nodes = int(sum(p.numel() for p in model.parameters()))
        print(f"▶ 全量训练完成：1D-CNN {cnn_cfg}，{n_nodes} 个参数（int8 ≈ {n_nodes / 1024:.0f} KB）", flush=True)
        bad = []
    else:
        model = RandomForestClassifier(**rf_kw).fit(F_tr, yf_tr)
        model = merge_forest(model, names, classes, back, scale)
        n_nodes = int(sum(e.tree_.node_count for e in model.estimators_))
        print(f"▶ 全量训练完成：{len(model.estimators_)} 棵树，{n_nodes} 个节点", flush=True)
        bad = forest_uses(model, fmask) if drop_level else []
    if bad:
        sys.exit(f"屏蔽的特征被树用到了（不应该发生）：列 {bad}")

    # 6. 存
    hz_tag = f"{int(hz) if float(hz).is_integer() else hz}"
    proc_dir = os.path.join(a.repo, "data", f"processed_{run}", f"{hz_tag}hz")
    out_dir = os.path.join(a.repo, a.results_dir, f"processed_{run}", f"{hz_tag}hz_v2", kind)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    meta = {"hz": hz, "window_size": W, "stride": S, "n_channels": str(Xev_m.shape[2]), "classes": str(classes),
            "split_strategy": "group_day_imu", "label_mode": "segment", "gravity_aligned": "True",
            "channel_mode": "inv", "dataset": "custom", "rot_aug": "0"}
    ho = holdout if holdout is not None else np.arange(len(y_ev))
    for split in ("train", "val", "test"):
        np.savez_compressed(os.path.join(proc_dir, f"{split}.npz"), X=Xev_m[ho], y=y_ev[ho],
                            y_seq=np.repeat(y_ev[ho][:, None], W, 1), **meta)

    result = {
        "hz": hz, "model": kind, "pipeline": "v2",
        "classes": classes, "gravity_aligned": True, "channel_mode": "inv",
        "window_size": W, "stride": S, "window_s": round(W / hz, 3), "stride_s": round(S / hz, 3),
        "label_mode": "segment", "n_channels": int(Xev_m.shape[2]),
        "fine_labels": fine, "train_classes": names, "class_scale": {c: float(s) for c, s in zip(classes, scale)},
        **({"xgb": xgb_kw} if kind == "xgb" else {"cnn": {**cnn_cfg, "epochs": cnn_epochs}} if kind == "cnn"
           else {"rf": {k: v for k, v in rf_kw.items() if k != "n_jobs"}}),
        "n_nodes": n_nodes, "motion_gate": gate, "target": target,
        # 训练时置成常数的特征（树里一次都不用）；推理照常算完整的特征向量
        "drop_gravity_level": drop_level, "masked_features": masked,
        # 静止 / 活动按动作强度重新分（窗口级指标按改过的标签算；片段级指标仍按人标的片段）
        "still_motion_g": still_g, "still_by_motion": still_info,
        "cv": f"按（天, 项圈）分组 {folds} 折" if pred is not None else None,
        # 逐折成绩（准确率 / macro-F1）；CNN 还有每折、全量训练的逐轮 loss 与训练准确率
        "curves": curves,
        # 主数据集（标注平台上点「用它训练」的那份，一般标得最干净）：列表上的各类 F1 默认看它
        "main_dataset": spec["datasets"][0]["name"],
    }
    if pred is not None:
        present = sorted(set(y_ev))
        rep = classification_report(y_ev, pred, labels=present, target_names=[classes[i] for i in present],
                                    zero_division=0, output_dict=True)
        result["accuracy"] = float((pred == y_ev).mean())
        result["macro_f1"] = float(f1_score(y_ev, pred, average="macro"))
        result["per_class"] = {classes[i]: {m: round(rep[classes[i]][m], 4) for m in ("precision", "recall", "f1-score")}
                               for i in present}
        text, cm = diag_confusion(y_ev, pred, list(range(len(classes))), classes)
        result["confusion"] = cm
        # 片段级（一段的窗口概率平均取最大）：皮肤评估数的是次数，看这个
        su = np.unique(seg_ev)
        sy = np.array([classes.index(segs[s]["coarse"]) for s in su])
        sp = np.array([proba[seg_ev == s].mean(0).argmax() for s in su])
        result["segment_macro_f1"] = round(float(f1_score(sy, sp, average="macro")), 4)
        # 片段级（经过稳定版 v2 后处理）：用户最后看到的是这个
        result["postprocess_segments"] = segment_metrics(seg_files, classes)
        print(f"  片段级（后处理后）：{result['postprocess_segments']}", flush=True)
        for ev, evr in result["postprocess_segments"].items():
            print(f"  [{ev}] 按时长分桶：", flush=True)
            for k, b in evr.get("by_len", {}).items():
                print(f"    {k:<5} 人标 {b['n_gt']:>4} 段  整段报出 {b['hit_rate']:.3f}"
                      f"  IoU≥0.5 {b['iou50_rate']:.3f}  报出段里 IoU≥0.5 {b['iou50_given_hit']}", flush=True)
            fg = evr.get("frag") or {}
            if fg.get("n"):
                print(f"    报出的段被切成 {fg['preds_per_hit_median']} 个预测段（中位），"
                      f"合起来盖住 {fg['cover_median']:.1%}（中位）", flush=True)
        if "抓挠" in classes:
            k = classes.index("抓挠")
            rec = round(float((sp[sy == k] == k).mean()), 3) if (sy == k).any() else None
            pre = round(float((sy[sp == k] == k).mean()), 3) if (sp == k).any() else None
            result["scratch_segments"] = {"recall": rec, "precision": pre, "f1": _prf(rec, pre)}
            # 事件级（后处理之后、IoU ≥ 0.5 一对一）：比上面那套宽松口径严格，多报/漏报/切碎都扣分
            pe = (result.get("postprocess_segments") or {}).get("抓挠")
            if pe:
                result["scratch_events"] = {"alpha": 0.5, "recall": pe.get("recall_iou50"),
                                            "precision": pe.get("precision_iou50"), "f1": pe.get("f1_iou50"),
                                            "n_gt": pe.get("n_gt"), "n_pred": pe.get("n_pred")}
        # 分数据集看：老数据标得粗（整夜一段的睡觉、抓挠间隙标成活动），混在一起的总分会被拉低
        ds_of = np.array([segs[s]["dataset"] for s in seg_ev])
        result["per_dataset"] = {}
        for d in sorted(set(ds_of)):
            mk = ds_of == d
            pr_ = sorted(set(y_ev[mk]))
            rep_d = classification_report(y_ev[mk], pred[mk], labels=pr_, target_names=[classes[i] for i in pr_],
                                          zero_division=0, output_dict=True)
            # 片段级（按次数）也分数据集：皮肤评估看的是这个
            sm = np.array([segs[x]["dataset"] == d for x in su])
            seg_d = {"segment_macro_f1": round(float(f1_score(sy[sm], sp[sm], labels=sorted(set(sy[sm])), average="macro")), 4)}
            if "抓挠" in classes:
                k = classes.index("抓挠")
                t_, p_ = sy[sm], sp[sm]
                seg_d["scratch_segments"] = {"recall": round(float((p_[t_ == k] == k).mean()), 3) if (t_ == k).any() else None,
                                             "precision": round(float((t_[p_ == k] == k).mean()), 3) if (p_ == k).any() else None}
            _, cm_d = diag_confusion(y_ev[mk], pred[mk], list(range(len(classes))), classes)
            result["per_dataset"][d] = {"macro_f1": round(float(f1_score(y_ev[mk], pred[mk], labels=pr_, average="macro")), 4),
                                        "n": int(mk.sum()),
                                        "f1": {classes[i]: round(rep_d[classes[i]]["f1-score"], 3) for i in pr_},
                                        "per_class": {classes[i]: {m: round(rep_d[classes[i]][m], 4) for m in ("precision", "recall", "f1-score")}
                                                      for i in pr_},
                                        # 交叉验证（没训练过的数据）的混淆矩阵，按窗口；诊断页能切数据集看
                                        "confusion": {"labels": cm_d["labels"], "matrix": cm_d["matrix"]},
                                        **seg_d}
        print("\n  ── 分数据集（老数据标得粗会拉低总分；看效果以标得最干净的那份为准）──")
        for d, r in result["per_dataset"].items():
            sc = r.get("scratch_segments") or {}
            print(f"  {d:<28} 窗口 macro {r['macro_f1']:.3f}  按片段 macro {r['segment_macro_f1']:.3f}"
                  + (f"  抓挠按次数 查全 {sc['recall']} 查准 {sc['precision']}" if sc.get("recall") is not None else "")
                  + f"  {r['n']:>7} 窗")
            print("      " + "  ".join(f"{k} {v:.2f}" for k, v in r["f1"].items()))
        # 抓挠按场地：佛山 / 影棚 / 深圳（分开看，别被样本多的场地盖住）
        site_ev = np.array([_site_of_csv(segs[s]["csv"]) or "" for s in seg_ev])
        site_su = np.array([_site_of_csv(segs[x]["csv"]) or "" for x in su])
        result["per_site"] = {}
        for key, cn in SITE_NAMES.items():
            mk, sm = site_ev == key, site_su == key
            if not mk.any():
                continue
            pr_ = sorted(set(y_ev[mk]))
            rep_s = classification_report(y_ev[mk], pred[mk], labels=pr_, target_names=[classes[i] for i in pr_],
                                          zero_division=0, output_dict=True)
            row = {"name": cn, "n_windows": int(mk.sum()), "n_segments": int(sm.sum()),
                   "macro_f1": round(float(f1_score(y_ev[mk], pred[mk], labels=pr_, average="macro")), 4),
                   "per_class": {classes[i]: {m: round(rep_s[classes[i]][m], 4) for m in ("precision", "recall", "f1-score")}
                                 for i in pr_}}
            if "抓挠" in classes and sm.any():
                k = classes.index("抓挠")
                t_, p_ = sy[sm], sp[sm]
                rec = round(float((p_[t_ == k] == k).mean()), 3) if (t_ == k).any() else None
                pre = round(float((t_[p_ == k] == k).mean()), 3) if (p_ == k).any() else None
                row["scratch_segments"] = {"recall": rec, "precision": pre, "f1": _prf(rec, pre),
                                           "n_gt": int((t_ == k).sum()), "n_pred": int((p_ == k).sum())}
                # 事件级也按场地拆（IoU ≥ 0.5 一对一）：这个更接近“现场到底抓没抓到”
                ev_site = ((result.get("postprocess_segments") or {}).get("抓挠") or {}).get("per_site", {}).get(key)
                if ev_site:
                    row["scratch_events"] = ev_site
            result["per_site"][key] = row
        print("\n  ── 抓挠按场地（按片段；合计看上面）──")
        for key, r in result["per_site"].items():
            sc = r.get("scratch_segments")
            if sc is None:
                continue
            print(f"  {r['name']:<8} 宽松 查全 {sc['recall']}  查准 {sc['precision']}  F1 {sc['f1']}"
                  f"  人标 {sc['n_gt']} 段 / 模型报 {sc['n_pred']} 段")
            ev = r.get("scratch_events")
            if ev:
                print(f"      按事件(IoU≥0.5) 查全 {ev['recall']}  查准 {ev['precision']}  F1 {ev['f1']}"
                      f"  命中 {ev['matched']}/{ev['n_gt']} 段")
        print("\n" + classification_report(y_ev, pred, labels=present, target_names=[classes[i] for i in present],
                                           zero_division=0, digits=3))
        print(text)
        print(f"  （以上是全部数据集合在一起）片段级 macro F1 {result['segment_macro_f1']}" +
              (f"；抓挠宽松 查全 {result['scratch_segments']['recall']} 查准 {result['scratch_segments']['precision']}"
               f" F1 {result['scratch_segments']['f1']}"
               if "scratch_segments" in result else "") +
              (f"；按事件(IoU≥0.5) 查全 {result['scratch_events']['recall']} 查准 {result['scratch_events']['precision']}"
               f" F1 {result['scratch_events']['f1']}"
               if "scratch_events" in result else ""))
    if kind == "cnn":
        # 跟 src/dl/train.py 同一套产物：dl_cnn_best.pt（state_dict）+ dl_cnn_best.json（推理元数据）
        # + dl_cnn.json（指标）。平台推理（infer_csv_scratch._load_dl_model）和端侧导出（algo_tinyml
        # export_train.export_cnn）都认这一套
        import torch
        path = os.path.join(out_dir, "dl_cnn_best.pt")
        torch.save(model.state_dict(), path)
        infer_meta = {"model": "cnn", "hz": hz, "window_size": W, "stride": S, "n_channels": int(Xev_m.shape[2]),
                      "classes": classes, "gravity_aligned": True, "channel_mode": "inv", "label_mode": "segment",
                      "m2m": False, "ch_mean": ch_mean.tolist(), "ch_std": ch_std.tolist(), "model_cfg": cnn_cfg,
                      "remap": None, "motion_gate": gate, "pipeline": "v2"}
        with open(os.path.join(out_dir, "dl_cnn_best.json"), "w", encoding="utf-8") as f:
            json.dump(infer_meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "dl_cnn.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n▶ 完成，用时 {time.time() - t0:.0f}s")
        print(f"纯标注: {os.path.relpath(path, a.repo)}")
        return
    with open(os.path.join(out_dir, f"ml_{kind}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    import joblib
    path = os.path.join(out_dir, f"ml_{kind}.pkl")
    joblib.dump(model, path)
    print(f"\n▶ 完成，用时 {time.time() - t0:.0f}s")
    print(f"纯标注: {os.path.relpath(path, a.repo)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--repo", default=os.path.abspath(os.path.join(_HERE, "..", "..")))
    p.add_argument("--results_dir", default="results")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n_jobs", type=int, default=-1)
    main(p.parse_args())
