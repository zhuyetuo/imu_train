"""
手工特征提取（时域 + 频域 + 跨轴 + 姿态），供 ML 模型使用。

输入通道约定（固定顺序）：
  0:3  acc_x, acc_y, acc_z   （重力对齐后）
  3:6  gyr_x, gyr_y, gyr_z   （重力对齐后）
  6:8  pitch, roll           （原始未对齐姿态角，若上游未附加则跳过）

输入: X (N, window_size, n_channels)，n_channels 为 6 或 8
输出: X_feat (N, n_features)
详见 docs/features.md。
"""

import numpy as np
from scipy import stats, signal

# 频段边界，按 Nyquist 频率的比例给出，兼容不同采样率
FREQ_BANDS = [(0.0, 0.125), (0.125, 0.375), (0.375, 0.75), (0.75, 1.0)]


def _time_features(window: np.ndarray) -> np.ndarray:
    """window: (window_size, n_channels) → 1D 特征向量，逐通道提取"""
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        feats.extend([
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            np.max(x) - np.min(x),
            np.sqrt(np.mean(x ** 2)),                   # RMS
            stats.skew(x) if np.std(x) > 1e-8 else 0.0,
            stats.kurtosis(x) if np.std(x) > 1e-8 else 0.0,
            np.sum(np.diff(np.sign(x)) != 0),           # zero crossing rate
        ])
    return np.array(feats, dtype=np.float32)


def _freq_features(window: np.ndarray, hz: int, n_ch: int) -> np.ndarray:
    """频域特征：只对前 n_ch 个通道（acc+gyro）提取，姿态角通道跳过（非振荡信号）"""
    feats = []
    for ch in range(min(n_ch, window.shape[1])):
        x = window[:, ch]
        freqs, psd = signal.welch(x, fs=hz, nperseg=min(len(x), 32))
        psd_norm = psd / (psd.sum() + 1e-8)
        spec_mean = np.sum(freqs * psd_norm)
        feats.extend([
            spec_mean,                                                     # 频谱均值
            np.sqrt(np.sum((freqs - spec_mean) ** 2 * psd_norm)),          # 频谱标准差
            freqs[np.argmax(psd)],                                        # 主频
            -np.sum(psd_norm * np.log(psd_norm + 1e-8)),                  # 频谱熵
        ])
        nyq = hz / 2.0
        for lo_frac, hi_frac in FREQ_BANDS:
            mask = (freqs >= lo_frac * nyq) & (freqs < hi_frac * nyq)
            feats.append(float(psd_norm[mask].sum()))                     # 分频段能量占比
    return np.array(feats, dtype=np.float32)


def _sma(triplet: np.ndarray) -> float:
    """信号幅值面积：三轴绝对值之和的均值，衡量整体运动能量（不区分方向）"""
    return float(np.mean(np.sum(np.abs(triplet), axis=1)))


def _cross_axis_corr(triplet: np.ndarray) -> list:
    """三轴两两相关系数，捕捉不同轴之间的协同运动模式"""
    feats = []
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        xi, xj = triplet[:, i], triplet[:, j]
        if np.std(xi) > 1e-8 and np.std(xj) > 1e-8:
            c = float(np.corrcoef(xi, xj)[0, 1])
            c = 0.0 if np.isnan(c) else c
        else:
            c = 0.0
        feats.append(c)
    return feats


def _global_features(window: np.ndarray) -> np.ndarray:
    """跨通道的全局特征：SMA + 三轴相关系数，加速度与角速度各一份"""
    feats = []
    acc  = window[:, 0:3]
    gyro = window[:, 3:6]
    feats.append(_sma(acc))
    feats.append(_sma(gyro))
    feats.extend(_cross_axis_corr(acc))
    feats.extend(_cross_axis_corr(gyro))
    return np.array(feats, dtype=np.float32)


def _feature_dim(window_size: int, n_channels: int, hz: int) -> int:
    dummy = np.zeros((window_size, n_channels), dtype=np.float32)
    return len(_extract_one(dummy, hz))


def _extract_one(window: np.ndarray, hz: int) -> np.ndarray:
    parts = [_time_features(window), _freq_features(window, hz, n_ch=6)]
    if window.shape[1] >= 6:
        parts.append(_global_features(window))
    return np.concatenate(parts)


def extract_features(X: np.ndarray, hz: int, show_progress: bool = True) -> np.ndarray:
    """
    X: (N, window_size, n_channels)，n_channels 为 6（acc+gyro）或 8（+pitch/roll）
    返回: (N, n_features)，若 X 为空则返回 shape (0, n_features)
    """
    if len(X) == 0:
        n_ch = X.shape[2] if X.ndim == 3 else 6
        window_size = X.shape[1] if X.ndim == 3 else 10
        n_feat = _feature_dim(window_size, n_ch, hz)
        return np.empty((0, n_feat), dtype=np.float32)
    features = []
    it = range(len(X))
    if show_progress and len(X) > 10:
        from tqdm import tqdm
        it = tqdm(it, desc="提取特征", unit="窗口")
    for i in it:
        features.append(_extract_one(X[i], hz))
    return np.stack(features)
