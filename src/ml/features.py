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
            stats.skew(x),
            stats.kurtosis(x),
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


def extract_features(X: np.ndarray, hz: int) -> np.ndarray:
    """
    X: (N, window_size, n_channels)
    返回: (N, n_features)
    """
    from tqdm import tqdm
    features = []
    for i in tqdm(range(len(X)), desc="提取特征", unit="窗口"):
        t_feat = _time_features(X[i])
        f_feat = _freq_features(X[i], hz)
        features.append(np.concatenate([t_feat, f_feat]))
    return np.stack(features)
