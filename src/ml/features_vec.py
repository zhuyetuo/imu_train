"""
手工特征的向量化实现：一次算完所有窗口，替代 features.py 里逐窗口的 Python 循环。

为什么值得做：推理时一小时数据 ≈ 1800 个窗口，原来是 for 循环里一个窗口一个
窗口地算，每个窗口内部还要按通道再循环——单是 Python 解释器的开销就比真正的
数值计算贵。这里把窗口维度整个交给 numpy/scipy（它们本来就支持 axis 参数），
循环次数从「窗口数 × 通道数」降到「通道数」这个量级。

**输出必须跟 features.py 完全一致**，否则用旧特征训出来的模型直接废掉。
所以：
  - 每一项都照着原实现逐字翻译，包括那些看着可以"顺手优化"的地方
    （比如 std>1e-8 才算偏度、psd 归一化时的 +1e-8）
  - 峰值计数不能用「比左右邻居都大」这种简化写法：scipy.find_peaks 对
    平台（连续相等的值）会算一个峰，而掉数据被 ffill 填出来的正好就是平台。
    这里用「上一个非零差分为正、下一个非零差分为负」来判，跟 scipy 等价
  - tests/test_features_vec.py 拿随机数据 + 各种边界情况逐元素比对

保留原实现不动：这版先并行跑、比对一致了再切换（config.FEATURES_VECTORIZED）。
"""

import numpy as np
from scipy import signal, stats

from features import FREQ_BANDS


def _time_stats_batch(x: np.ndarray) -> list:
    """
    x: (N, W) —— N 条一维信号一起算，返回 11 个 (N,) 数组，顺序跟
    features._time_stats_1d 完全一致。
    """
    n, w = x.shape
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    rms = np.sqrt((x**2).mean(axis=1))
    q1, q3 = np.percentile(x, [25, 75], axis=1)

    # 偏度/峰度：原实现在 std<=1e-8 时直接给 0（常数信号算出来是 nan）
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = stats.skew(x, axis=1)
        kurt = stats.kurtosis(x, axis=1)
    flat = std <= 1e-8
    skew = np.where(flat, 0.0, skew)
    kurt = np.where(flat, 0.0, kurt)

    # 均值穿越率：穿越窗口自身均值的次数
    centered = x - mean[:, None]
    sgn = np.sign(centered)
    mcr = (np.diff(sgn, axis=1) != 0).sum(axis=1)

    return [mean, std, mn, mx, mx - mn, rms, skew, kurt, mcr.astype(np.float64), q3 - q1, _peak_count_batch(x)]


def _peak_count_batch(x: np.ndarray) -> np.ndarray:
    """
    局部极大值个数，跟 scipy.signal.find_peaks(x) 的默认行为一致（含平台处理）。

    不能简化成「比左右都大」：连续相等的一段（平台）scipy 也算一个峰，而掉数据
    被前值填充出来的恰好就是平台，简化写法会漏掉。判据是「上一个非零差分为正，
    下一个非零差分为负」——升上去、平一段、再降下来，算一个峰。
    """
    n, w = x.shape
    if w < 3:
        return np.zeros(n, dtype=np.float64)
    d = np.sign(np.diff(x, axis=1))  # (N, W-1)

    # 把 0 用「前一个非零」填掉：升→平 记作还在升，降→平 记作还在降
    nz = d != 0
    idx = np.where(nz, np.arange(d.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    prev = np.take_along_axis(d, idx, axis=1)
    # 开头就是平台（前面没有非零差分）时保持 0，不然会凭空造出一个峰
    lead = ~np.maximum.accumulate(nz, axis=1)
    prev = np.where(lead, 0, prev)

    # 峰 = 前面在升(prev>0) 且 这一步在降(d<0)
    return ((prev[:, :-1] > 0) & (d[:, 1:] < 0)).sum(axis=1).astype(np.float64)


def _freq_stats_batch(x: np.ndarray, hz: int) -> list:
    """x: (N, W) → 4 个统计量 + 4 个分频段占比，共 8 个 (N,) 数组。"""
    freqs, psd = signal.welch(x, fs=hz, nperseg=min(x.shape[1], 32), axis=1)
    psd_norm = psd / (psd.sum(axis=1, keepdims=True) + 1e-8)
    spec_mean = (freqs[None, :] * psd_norm).sum(axis=1)
    spec_std = np.sqrt((((freqs[None, :] - spec_mean[:, None]) ** 2) * psd_norm).sum(axis=1))
    peak_freq = freqs[np.argmax(psd, axis=1)]
    entropy = -(psd_norm * np.log(psd_norm + 1e-8)).sum(axis=1)
    out = [spec_mean, spec_std, peak_freq, entropy]
    nyq = hz / 2.0
    for lo_frac, hi_frac in FREQ_BANDS:
        mask = (freqs >= lo_frac * nyq) & (freqs < hi_frac * nyq)
        out.append(psd_norm[:, mask].sum(axis=1))
    return out


def _corr_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐窗口的相关系数；任一路是常数（std<=1e-8）时按原实现给 0。"""
    sa, sb = a.std(axis=1), b.std(axis=1)
    ca = a - a.mean(axis=1, keepdims=True)
    cb = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((ca**2).sum(axis=1) * (cb**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (ca * cb).sum(axis=1) / denom
    c = np.where(np.isfinite(c), c, 0.0)
    return np.where((sa > 1e-8) & (sb > 1e-8), c, 0.0)


def extract_features_vec(X: np.ndarray, hz: int) -> np.ndarray:
    """
    X: (N, window_size, n_channels)，返回 (N, n_features)。

    输出跟 features.extract_features 完全一致——拼接顺序也一样：
      时域(全通道) → 频域(前6通道) → 全局 → acc/gyro模长 → acc-jerk
    """
    if len(X) == 0:
        from features import _feature_dim

        n_ch = X.shape[2] if X.ndim == 3 else 6
        window_size = X.shape[1] if X.ndim == 3 else 10
        return np.empty((0, _feature_dim(window_size, n_ch, hz)), dtype=np.float32)

    # 不要顺手升成 float64：原实现是拿 float32 的窗口切片算的，升精度会让
    # 峰度这类高阶统计量在第 4 位小数上跟旧特征对不上——模型是用旧特征训的，
    # 一致性比精度重要
    X = np.asarray(X)
    n, w, c = X.shape
    cols: list[np.ndarray] = []

    # 时域：全部通道
    for ch in range(c):
        cols.extend(_time_stats_batch(X[:, :, ch]))
    # 频域：只有 acc+gyro 这 6 路（姿态角不是振荡信号）
    for ch in range(min(6, c)):
        cols.extend(_freq_stats_batch(X[:, :, ch], hz))

    if c >= 6:
        acc, gyro = X[:, :, 0:3], X[:, :, 3:6]
        # 全局：SMA + 三轴两两相关
        cols.append(np.abs(acc).sum(axis=2).mean(axis=1))
        cols.append(np.abs(gyro).sum(axis=2).mean(axis=1))
        for triplet in (acc, gyro):
            for i, j in ((0, 1), (1, 2), (0, 2)):
                cols.append(_corr_batch(triplet[:, :, i], triplet[:, :, j]))
        # 模长：acc、gyro 各一份时域+频域
        for triplet in (acc, gyro):
            mag = np.sqrt((triplet**2).sum(axis=2))
            cols.extend(_time_stats_batch(mag))
            cols.extend(_freq_stats_batch(mag, hz))
        # Jerk：加速度的导数的模长，只取时域
        jerk = np.diff(acc, axis=1) * hz
        cols.extend(_time_stats_batch(np.sqrt((jerk**2).sum(axis=2))))

    return np.stack(cols, axis=1).astype(np.float32)
