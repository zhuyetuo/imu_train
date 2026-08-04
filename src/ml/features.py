"""
手工特征提取（时域 + 频域 + 跨轴 + 姿态 + 模长 + Jerk），供 ML 模型使用。

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


def _time_stats_1d(x: np.ndarray) -> list:
    """单个一维信号的10个时域统计量，供逐通道特征和模长/Jerk特征共用"""
    std = np.std(x)
    q1, q3 = np.percentile(x, [25, 75])
    return [
        np.mean(x),
        std,
        np.min(x),
        np.max(x),
        np.max(x) - np.min(x),                      # 极差
        np.sqrt(np.mean(x ** 2)),                    # RMS
        stats.skew(x) if std > 1e-8 else 0.0,
        stats.kurtosis(x) if std > 1e-8 else 0.0,
        np.sum(np.diff(np.sign(x)) != 0),            # 过零率
        q3 - q1,                                      # IQR，比std更抗突发尖峰噪声
    ]


def _freq_stats_1d(x: np.ndarray, hz: int) -> list:
    """单个一维信号的频域统计量：4个统计量 + 4个分频段能量占比"""
    freqs, psd = signal.welch(x, fs=hz, nperseg=min(len(x), 32))
    psd_norm = psd / (psd.sum() + 1e-8)
    spec_mean = np.sum(freqs * psd_norm)
    feats = [
        spec_mean,                                                # 频谱均值
        np.sqrt(np.sum((freqs - spec_mean) ** 2 * psd_norm)),     # 频谱标准差
        freqs[np.argmax(psd)],                                    # 主频
        -np.sum(psd_norm * np.log(psd_norm + 1e-8)),              # 频谱熵
    ]
    nyq = hz / 2.0
    for lo_frac, hi_frac in FREQ_BANDS:
        mask = (freqs >= lo_frac * nyq) & (freqs < hi_frac * nyq)
        feats.append(float(psd_norm[mask].sum()))                 # 分频段能量占比
    return feats


def _time_features(window: np.ndarray) -> np.ndarray:
    """window: (window_size, n_channels) → 1D 特征向量，逐通道提取"""
    feats = []
    for ch in range(window.shape[1]):
        feats.extend(_time_stats_1d(window[:, ch]))
    return np.array(feats, dtype=np.float32)


def _freq_features(window: np.ndarray, hz: int, n_ch: int) -> np.ndarray:
    """频域特征：只对前 n_ch 个通道（acc+gyro）提取，姿态角通道跳过（非振荡信号）"""
    feats = []
    for ch in range(min(n_ch, window.shape[1])):
        feats.extend(_freq_stats_1d(window[:, ch], hz))
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


def _magnitude(triplet: np.ndarray) -> np.ndarray:
    """三轴合成模长 sqrt(x²+y²+z²)，旋转不变，抗项圈朝向漂移"""
    return np.sqrt(np.sum(triplet ** 2, axis=1))


def _magnitude_features(window: np.ndarray, hz: int) -> np.ndarray:
    """acc模长、gyro模长各自的时域+频域特征（各10+8=18维）"""
    feats = []
    for triplet in (window[:, 0:3], window[:, 3:6]):
        mag = _magnitude(triplet)
        feats.extend(_time_stats_1d(mag))
        feats.extend(_freq_stats_1d(mag, hz))
    return np.array(feats, dtype=np.float32)


def _jerk_features(window: np.ndarray, hz: int) -> np.ndarray:
    """
    加速度的加加速度（Jerk）模长的时域统计量，捕捉动作的"突然性"。
    平缓行走 jerk 小；惊吓反应、甩头、抓挠回位等瞬时动作 jerk 会有明显尖峰，
    这是均值/标准差/峰度都容易漏掉的维度。
    只对加速度算（角加速度可类推，暂不加，避免维度膨胀）。
    """
    acc = window[:, 0:3]
    jerk = np.diff(acc, axis=0) * hz  # 近似导数：Δacc / Δt
    jerk_mag = _magnitude(jerk)
    return np.array(_time_stats_1d(jerk_mag), dtype=np.float32)


def _feature_dim(window_size: int, n_channels: int, hz: int) -> int:
    dummy = np.zeros((window_size, n_channels), dtype=np.float32)
    return len(_extract_one(dummy, hz))


def _extract_one(window: np.ndarray, hz: int) -> np.ndarray:
    parts = [_time_features(window), _freq_features(window, hz, n_ch=6)]
    if window.shape[1] >= 6:
        parts.append(_global_features(window))
        parts.append(_magnitude_features(window, hz))
        parts.append(_jerk_features(window, hz))
    return np.concatenate(parts)


CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "pitch", "roll"]

TIME_FEAT_NAMES = ["mean", "std", "min", "max", "range", "rms", "skew", "kurtosis", "zcr", "iqr"]
FREQ_FEAT_NAMES = ["spec_mean", "spec_std", "peak_freq", "spec_entropy",
                    "band_energy_0", "band_energy_1", "band_energy_2", "band_energy_3"]
GLOBAL_FEAT_NAMES = ["sma_acc", "sma_gyro",
                      "corr_acc_xy", "corr_acc_yz", "corr_acc_xz",
                      "corr_gyro_xy", "corr_gyro_yz", "corr_gyro_xz"]
MAG_FEAT_NAMES = [f"acc_mag_{f}" for f in TIME_FEAT_NAMES] + [f"acc_mag_{f}" for f in FREQ_FEAT_NAMES] \
                + [f"gyro_mag_{f}" for f in TIME_FEAT_NAMES] + [f"gyro_mag_{f}" for f in FREQ_FEAT_NAMES]
JERK_FEAT_NAMES = [f"acc_jerk_mag_{f}" for f in TIME_FEAT_NAMES]


def feature_names(n_channels: int) -> list:
    """特征名列表，顺序必须与 _extract_one 的拼接顺序完全一致：
    时域(全部通道) → 频域(仅前6通道 acc+gyro) → 全局(SMA+轴间相关)
    → acc/gyro模长(时域+频域) → acc-jerk模长(时域)，后三组仅当 n_channels>=6"""
    names = []
    for ch in range(n_channels):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
        names.extend(f"{ch_name}_{feat}" for feat in TIME_FEAT_NAMES)
    for ch in range(min(6, n_channels)):
        ch_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"ch{ch}"
        names.extend(f"{ch_name}_{feat}" for feat in FREQ_FEAT_NAMES)
    if n_channels >= 6:
        names.extend(GLOBAL_FEAT_NAMES)
        names.extend(MAG_FEAT_NAMES)
        names.extend(JERK_FEAT_NAMES)
    return names


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
