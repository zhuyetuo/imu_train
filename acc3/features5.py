"""5 通道（acc 三轴 + pitch/roll）的特征提取，**113 维**。

通道约定：
    0:3  acc_x, acc_y, acc_z   （重力对齐后）
    3:5  pitch, roll           （对齐前算的姿态角，由 acc 三轴派生）

## 为什么要新写这一份

`src/ml/features.py` 的 `_extract_one` 里，全局特征、模长特征、jerk 特征
全都挂在 `window.shape[1] >= 6` 底下。喂 5 通道进去这一整块直接跳过，
只剩 95 维——而少掉的那 34 维（acc 三轴模长 19、acc jerk 11、
SMA + 三轴相关 4）**只需要加速计就能算**。那个门槛存在的原因是
"gyro 的模长/SMA 要用到 3:6 列"，不是"acc 的算不出来"。

拿 95 维的结果去回答"砍掉陀螺仪影响多大"，会把**实现上少算的**
和**传感器上少的**混在一起，得到一个偏低的数。

## 一行新公式都没有

11 个时域统计、8 个频域统计、SMA、相关系数、模长、jerk——全部**直接调用
`src/ml/features.py` 里的那几个函数**，一个都没有重写。这里只是换了一个
拼接顺序（跳过不存在的 gyro）。

重写的话迟早跟训练侧分家，而分家的表现是"离线指标好、上服务就不对"。

## 跟 193 维的关系：113 维是它的一个**子集**，逐位相同

把 8 通道那 193 维里所有名字带 `gyr` 的维度删掉，剩下正好 113 维，
**顺序跟这里拼出来的完全一致，数值逐位相同**。
`acc3/selfcheck5.py` 会拿仓库那份代码当参照验一遍。

这个性质有用处：推理时可以照常算 193 维，然后按下标取那 113 列——
于是 5 通道的模型不需要一条新的预处理链就能上线。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "src", "data"),
           os.path.join(_REPO, "src", "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# **全部来自仓库那份**，一个都不重写
from features import (  # noqa: E402
    FREQ_FEAT_NAMES,
    TIME_FEAT_NAMES,
    _cross_axis_corr,
    _freq_stats_1d,
    _magnitude,
    _sma,
    _time_stats_1d,
)

N_CHANNELS = 5
CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "pitch", "roll"]


def extract_one(window: np.ndarray, hz: int) -> np.ndarray:
    """window: (T, 5) → (113,)

    拼接顺序跟 8 通道那份保持"删掉 gyr 之后"的同一个顺序：
        时域(5 个通道) → 频域(只有 acc 三轴) → SMA+相关 → acc 模长 → acc jerk

    频域为什么跳过 pitch/roll：跟仓库那份同一个理由——姿态角不是振荡信号，
    对它做 Welch 得到的主频/谱熵没有物理含义。
    **这里必须跟 8 通道那份保持一致**，否则 113 维就不再是 193 维的子集，
    上面说的那个"推理时按下标取列"的办法就不成立了。
    """
    w = np.asarray(window, np.float32)
    if w.ndim != 2 or w.shape[1] != N_CHANNELS:
        raise ValueError(f"要 (T, {N_CHANNELS})，给的是 {w.shape}")
    acc = w[:, 0:3]

    feats = []
    for ch in range(N_CHANNELS):            # 时域：5 个通道都算
        feats.extend(_time_stats_1d(w[:, ch]))
    for ch in range(3):                     # 频域：只有 acc 三轴
        feats.extend(_freq_stats_1d(w[:, ch], hz))
    feats.append(_sma(acc))
    feats.extend(_cross_axis_corr(acc))
    mag = _magnitude(acc)
    feats.extend(_time_stats_1d(mag))
    feats.extend(_freq_stats_1d(mag, hz))
    jerk_mag = _magnitude(np.diff(acc, axis=0) * hz)
    feats.extend(_time_stats_1d(jerk_mag))
    return np.asarray(feats, np.float32)


def feature_names() -> list:
    names = []
    for ch in CHANNEL_NAMES:
        names.extend(f"{ch}_{f}" for f in TIME_FEAT_NAMES)
    for ch in CHANNEL_NAMES[:3]:
        names.extend(f"{ch}_{f}" for f in FREQ_FEAT_NAMES)
    names.append("sma_acc")
    names.extend(["corr_acc_xy", "corr_acc_yz", "corr_acc_xz"])
    names.extend(f"acc_mag_{f}" for f in TIME_FEAT_NAMES)
    names.extend(f"acc_mag_{f}" for f in FREQ_FEAT_NAMES)
    names.extend(f"acc_jerk_mag_{f}" for f in TIME_FEAT_NAMES)
    return names


N_FEATURES = len(feature_names())      # 113


def gyro_free_indices() -> np.ndarray:
    """8 通道那 193 维里，**不含陀螺仪**的那 113 个下标。

    推理时用：照常算 193 维，然后 `feats[:, idx]`，就能喂给 5 通道的模型。
    这样 5 通道模型不需要一条新的预处理链。

    靠特征名筛，不写死下标——写死的话仓库那边加一个特征，这里的下标
    会**整体错位**，而错位之后每一维都对到别的特征上，模型照样给得出结果。
    """
    from features import feature_names as names8

    idx = [i for i, n in enumerate(names8(8)) if "gyr" not in n]
    if len(idx) != N_FEATURES:
        raise RuntimeError(
            f"8 通道那份去掉陀螺仪剩 {len(idx)} 维，这里是 {N_FEATURES} 维，"
            "两边的特征集合已经不一致了。改过 src/ml/features.py 的话，"
            "这个文件要跟着改（先跑 acc3/selfcheck5.py 看差在哪）。")
    return np.asarray(idx, np.int64)


def extract_features(X: np.ndarray, hz: int, show_progress: bool = True,
                     workers: int = 1) -> np.ndarray:
    """跟 `src/ml/features.py:extract_features` **同签名**——train.py 是按这个
    签名调的，参数名/顺序不一样的话注入之后会在运行时炸。

    workers 走 joblib 多进程，跟仓库那份同一个理由：特征提取是纯 CPU、
    窗口之间无状态，而 GIL 会卡住纯 Python 循环。
    """
    X = np.asarray(X)
    if len(X) == 0:
        return np.zeros((0, N_FEATURES), np.float32)
    if X.ndim != 3 or X.shape[2] != N_CHANNELS:
        raise ValueError(
            f"要 (N, T, {N_CHANNELS})，给的是 {X.shape}。"
            "5 通道的 npz 用 acc3/make_acc3_npz.py 生成。")

    if workers == 1 or len(X) < 64:
        out = np.stack([extract_one(X[i], hz) for i in range(len(X))])
    else:
        from joblib import Parallel, delayed
        n = os.cpu_count() or 1 if workers < 0 else workers
        chunks = np.array_split(np.arange(len(X)), max(1, n))
        parts = Parallel(n_jobs=n)(
            delayed(_chunk)(X[c], hz) for c in chunks if len(c))
        out = np.concatenate(parts) if parts else np.zeros((0, N_FEATURES), np.float32)
    if show_progress:
        print(f"[acc3/features5] {len(out)} 个窗口 → {out.shape[1]} 维")
    return out.astype(np.float32)


def _chunk(Xc, hz):
    return np.stack([extract_one(Xc[i], hz) for i in range(len(Xc))])
