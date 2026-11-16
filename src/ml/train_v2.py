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


def _build_one(args):
    csv_path, segs, hz, axes, upsample = args
    if not os.path.isfile(csv_path):
        return csv_path, None, f"CSV 不存在：{csv_path}"
    tl = load_timeline(csv_path, hz, segs[0].get("source_hz"), axes, upsample)
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
    tl["seg"] = lab
    return csv_path, tl, None


# ── 窗口 / 特征 ─────────────────────────────────────────────────────────────

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


def xgb_params(params: dict) -> dict:
    kw = dict(XGB_DEFAULTS)
    for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"):
        if params.get(k) not in (None, ""):
            kw[k] = type(XGB_DEFAULTS[k])(params[k])
    if kw["max_depth"] > 6:
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
    if kind not in ("rf", "xgb"):
        sys.exit(f"新版流程的 model 只能是 rf / xgb，给的是 {kind!r}")
    xgb_kw = xgb_params(params) if kind == "xgb" else None
    if kind == "xgb":
        fine = []          # GBDT 直接训归并后的类别，见 XGB_DEFAULTS 上面的说明
    run = spec["run_name"]
    drop_level = bool(spec.get("drop_gravity_level"))
    still_g = float(spec.get("still_motion_g") or 0) or None
    t0 = time.time()
    print("=" * 60)
    print(f"  新版训练流程（v2）  {axes} 轴  采样率 {hz:g}Hz  窗口 {W} 点（{W / hz:.2f}s）  步长 {S} 点（{S / hz:.2f}s）")
    print(f"  类别：{classes}")
    print(f"  细分训练、合并输出：{fine or '不用'}")
    print(f"  输出校准：{ {c: s for c, s in zip(classes, scale) if s != 1} or '不用'}")
    print(f"  XGBoost：{xgb_kw}" if kind == "xgb" else f"  随机森林：{rf_kw}")
    print("=" * 60, flush=True)

    # 1. 片段
    segs = collect_segments(spec["datasets"], spec["nas_root"], remap, classes, fine)
    if not segs:
        sys.exit("没有可训练的片段：检查类别归并表")
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
    Xtr_raw, seg_tr = make_windows(tls, segs, W, S, short_shifts=2)
    Xev_raw, seg_ev = make_windows(tls, segs, W, S, short_shifts=0)
    Xtr_m, F_tr = featurize(Xtr_raw, hz)
    Xev_m, F_ev = featurize(Xev_raw, hz)
    masked = []
    if drop_level:
        fmask, masked = gravity_level_mask(Xev_m.shape[2])
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

    # 4. 分组交叉验证 → 指标
    n_groups = len(set(g_ev))
    folds = min(5, n_groups)
    proba = np.zeros((len(y_ev), len(classes)))
    holdout = None
    if folds >= 2:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
        for k, (_, te) in enumerate(cv.split(F_ev, y_ev, g_ev)):
            test_g = set(g_ev[te])
            tr = ~np.isin(g_tr, list(test_g))
            if kind == "xgb":
                m, mc = fit_xgb(F_tr[tr], back_arr[yf_tr[tr]], len(classes), xgb_kw, int(a.n_jobs))
                p_ = xgb_proba(m, mc, F_ev[te], len(classes)) * scale[None, :]
                proba[te] = p_ / p_.sum(1, keepdims=True)
            else:
                m = RandomForestClassifier(**rf_kw).fit(F_tr[tr], yf_tr[tr])
                m = merge_forest(m, names, classes, back, scale)
                proba[te] = m.predict_proba(F_ev[te])
            if holdout is None:
                holdout = te
            print(f"  交叉验证 {k + 1}/{folds}", flush=True)
        pred = proba.argmax(1)
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
        **({"xgb": xgb_kw} if kind == "xgb" else {"rf": {k: v for k, v in rf_kw.items() if k != "n_jobs"}}),
        "n_nodes": n_nodes,
        # 训练时置成常数的特征（树里一次都不用）；推理照常算完整的特征向量
        "drop_gravity_level": drop_level, "masked_features": masked,
        # 静止 / 活动按动作强度重新分（窗口级指标按改过的标签算；片段级指标仍按人标的片段）
        "still_motion_g": still_g, "still_by_motion": still_info,
        "cv": f"按（天, 项圈）分组 {folds} 折" if pred is not None else None,
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
        if "抓挠" in classes:
            k = classes.index("抓挠")
            result["scratch_segments"] = {"recall": round(float((sp[sy == k] == k).mean()), 3) if (sy == k).any() else None,
                                          "precision": round(float((sy[sp == k] == k).mean()), 3) if (sp == k).any() else None}
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
        print("\n" + classification_report(y_ev, pred, labels=present, target_names=[classes[i] for i in present],
                                           zero_division=0, digits=3))
        print(text)
        print(f"  （以上是全部数据集合在一起）片段级 macro F1 {result['segment_macro_f1']}" +
              (f"；抓挠按段 查全 {result['scratch_segments']['recall']} 查准 {result['scratch_segments']['precision']}"
               if "scratch_segments" in result else ""))
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
