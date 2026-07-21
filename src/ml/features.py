"""
手工特征提取（时域 + 频域），供 ML 模型使用。
输入: X (N, window_size, n_channels)
输出: X_feat (N, n_features)
"""

import numpy as np
from scipy import stats, signal


def _time_features(window: np.ndarray) -> np.ndarray:
    """window: (window_size, n_channels) → 1D 特征向量"""
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


def _freq_features(window: np.ndarray, hz: int) -> np.ndarray:
    """频域特征"""
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        freqs, psd = signal.welch(x, fs=hz, nperseg=min(len(x), 32))
        psd_norm = psd / (psd.sum() + 1e-8)
        feats.extend([
            np.sum(freqs * psd_norm),                    # 频谱均值
            np.sqrt(np.sum((freqs - np.sum(freqs * psd_norm)) ** 2 * psd_norm)),  # 频谱标准差
            freqs[np.argmax(psd)],                       # 主频
            -np.sum(psd_norm * np.log(psd_norm + 1e-8)), # 频谱熵
        ])
    return np.array(feats, dtype=np.float32)


def extract_features(X: np.ndarray, hz: int, show_progress: bool = True) -> np.ndarray:
    """
    X: (N, window_size, n_channels)
    返回: (N, n_features)，若 X 为空则返回 shape (0, n_features)
    """
    if len(X) == 0:
        n_ch = X.shape[2] if X.ndim == 3 else 6
        dummy = np.zeros((1, X.shape[1] if X.ndim == 3 else 10, n_ch), dtype=np.float32)
        n_feat = len(np.concatenate([_time_features(dummy[0]), _freq_features(dummy[0], hz)]))
        return np.empty((0, n_feat), dtype=np.float32)
    features = []
    it = range(len(X))
    if show_progress and len(X) > 10:
        from tqdm import tqdm
        it = tqdm(it, desc="提取特征", unit="窗口")
    for i in it:
        features.append(np.concatenate([_time_features(X[i]), _freq_features(X[i], hz)]))
    return np.stack(features)
