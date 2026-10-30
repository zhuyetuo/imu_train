"""
Per-window gravity alignment.

Estimates the gravity vector as the mean of the accelerometer signal over
the window (low-frequency ≈ gravity), then rotates the entire window so
gravity points along +Z. The same rotation is applied to the gyroscope.

This makes features rotation-invariant around the yaw axis (up/down
direction remains fixed), which handles collars mounted at different angles
while preserving the directional information needed to distinguish lying
vs standing vs trotting.
"""

import numpy as np


def gravity_align(window: np.ndarray) -> np.ndarray:
    """
    Args:
        window: (T, C) array where first 3 cols are acc, next 3 are gyr.
                Extra channels beyond 6 are passed through unchanged.
    Returns:
        (T, C) array with acc and gyr rotated so gravity → +Z.
    """
    acc = window[:, :3]
    gyr = window[:, 3:6] if window.shape[1] >= 6 else None

    g_est = acc.mean(axis=0)
    g_norm = np.linalg.norm(g_est)
    if g_norm < 1e-6:
        return window  # no gravity signal, skip

    g_unit = g_est / g_norm
    ref = np.array([0.0, 0.0, 1.0])

    dot = float(np.clip(np.dot(g_unit, ref), -1.0, 1.0))

    if dot > 0.9999:
        return window  # already aligned

    if dot < -0.9999:
        # 180° flip around X axis
        R = np.diag(np.array([1.0, -1.0, -1.0]))
    else:
        axis = np.cross(g_unit, ref)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        # Rodrigues' rotation formula
        K = np.array([
            [0.0,      -axis[2],  axis[1]],
            [axis[2],   0.0,     -axis[0]],
            [-axis[1],  axis[0],  0.0    ],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    out = window.copy()
    out[:, :3] = (R @ acc.T).T
    if gyr is not None:
        out[:, 3:6] = (R @ gyr.T).T
    return out


def gravity_align_batch(X: np.ndarray) -> np.ndarray:
    """Apply gravity_align to every window in X (N, T, C)."""
    if len(X) == 0:
        return X
    return np.stack([gravity_align(w) for w in X])


def raw_tilt(acc: np.ndarray) -> np.ndarray:
    """
    Per-sample pitch/roll (radians) from RAW, un-rotated accelerometer.

    Must be computed BEFORE gravity_align: that function rotates each
    window so its mean acc vector points to +Z, which forces every
    window's average tilt to ~0 and destroys absolute-posture information
    (lying vs sitting vs standing). This function is meant to be called
    on the original per-window acc slice, with the result concatenated
    as extra channels alongside the (separately) gravity-aligned acc/gyro.

    Args:
        acc: (T, 3) raw accelerometer.
    Returns:
        (T, 2) float32 array of [pitch, roll].
    """
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    pitch = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))
    roll  = np.arctan2(ay, az)
    return np.stack([pitch, roll], axis=1).astype(np.float32)


def append_raw_tilt_batch(X: np.ndarray) -> np.ndarray:
    """X: (N, T, C>=3) raw (pre-alignment) window batch, acc in cols 0:3.
    Returns X with 2 extra trailing channels [pitch, roll] appended."""
    if len(X) == 0:
        n_ch = X.shape[2] if X.ndim == 3 else 6
        return np.empty((0, X.shape[1] if X.ndim == 3 else 0, n_ch + 2), dtype=np.float32)
    tilt = np.stack([raw_tilt(w[:, :3]) for w in X], axis=0)
    return np.concatenate([X, tilt], axis=2).astype(np.float32)


# ── 方向无关通道（channel_mode="inv"）────────────────────────────────────────
# 上面那套（channel_mode="tilt"）只对一半：重力对齐消掉了「哪个轴朝下」，但
#   · 绕重力轴转了多少（扣环在左还是右）没消掉，对齐后的 x/y 跟着佩戴方向变；
#   · pitch/roll 是对齐**前**按原始轴算的，佩戴方向一变整个变，倒着戴时 roll 停在
#     ±180° 附近，Y 轴一抖就在 +π/−π 之间跳，特征上像剧烈运动。
# 用户怎么戴都不限制，所以输入里只留跟佩戴方向无关的量。g = 窗口内加速度均值（重力方向），
# u = g/|g|。下面每一路对设备的任意固定旋转都**严格不变**（u 跟着设备一起转，点积/模长不变）：
#   v     = a·u                  竖直分量（含 1g 重力）
#   h     = |a − v·u|            水平分量的模
#   |a|   = 加速度模
#   [gyro] ωv = ω·u、ωh = |ω − ωv·u|、|ω|
#   θ     = atan2(h, v)          偏离平均重力方向的角度（替代 pitch/roll，不会在 ±π 跳）
#   jerk  = |a_t − a_{t−1}|      第一个样本记 0
# 通道数和排布跟 tilt 一样（3 轴 5 路、6 轴 8 路，前面是「传感器」通道、最后两路是角度类），
# 特征提取（时域全部通道 / 频域前 n_sensor 路）一行不用改；端侧 C 见 algo_tinyml core/tm_imu.c。
CHANNEL_MODES = ("tilt", "inv")


def invariant_channels(window: np.ndarray) -> np.ndarray:
    """window: (T, C>=3) 原始 acc(+gyro)。返回 (T, n_sensor+2) float32，见上面。"""
    acc = window[:, :3].astype(np.float64)
    g = acc.mean(axis=0)
    gn = np.linalg.norm(g)
    u = g / gn if gn >= 1e-6 else np.array([0.0, 0.0, 1.0])

    def split(x):
        vert = x @ u
        mag = np.sqrt(np.sum(x * x, axis=1))
        hor = np.sqrt(np.maximum(mag * mag - vert * vert, 0.0))
        return vert, hor, mag

    v, h, m = split(acc)
    cols = [v, h, m]
    if window.shape[1] >= 6:
        cols += list(split(window[:, 3:6].astype(np.float64)))
    theta = np.arctan2(h, v)
    jerk = np.zeros(len(acc))
    if len(acc) > 1:
        jerk[1:] = np.sqrt(np.sum(np.diff(acc, axis=0) ** 2, axis=1))
    cols += [theta, jerk]
    return np.stack(cols, axis=1).astype(np.float32)


def map_channels(X: np.ndarray, channel_mode: str = "tilt", use_gravity_align: bool = True) -> np.ndarray:
    """原始窗口 (N, T, C) → 模型输入 (N, T, n_sensor+2)。训练、推理、端侧对照都走这一个函数，
    别在调用处自己拼（以前每处各拼一遍，3 轴时有两处取错了列）。"""
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f"channel_mode 只能是 {CHANNEL_MODES}，收到 {channel_mode!r}")
    if X.ndim != 3:
        return X
    if channel_mode == "inv":
        n_out = (6 if X.shape[2] >= 6 else 3) + 2
        if len(X) == 0:
            return np.empty((0, X.shape[1], n_out), dtype=np.float32)
        return np.stack([invariant_channels(w) for w in X]).astype(np.float32)
    tilt = append_raw_tilt_batch(X)[:, :, -2:]   # 原始（未对齐）姿态角，须在重力对齐前算
    if use_gravity_align:
        X = gravity_align_batch(X)
    return np.concatenate([X, tilt], axis=2).astype(np.float32)


def channel_mode_of(meta: dict) -> str:
    """模型 / 预处理 meta 里的 channel_mode；老模型没有这一项，就是 tilt。"""
    m = str((meta or {}).get("channel_mode") or "tilt")
    return m if m in CHANNEL_MODES else "tilt"


# ── 旋转增强 ─────────────────────────────────────────────────────────────────
def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """均匀分布的 3D 旋转（随机单位四元数）。"""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def rotation_augment(X: np.ndarray, y: np.ndarray, copies: int, seed: int = 0,
                     drift_deg: float = 15.0, y_seq: np.ndarray | None = None):
    """训练集旋转增强：每个原始窗口再出 copies 份，每份
      1. 整体乘一个均匀随机的 3D 旋转（模拟任意佩戴方向）；
      2. 再叠一个窗口内缓慢变化的小旋转（0 → ±drift_deg，模拟项圈在脖子上滑动）。
    acc 和 gyro 用同一个旋转。只给训练集用——验证/测试集保持原样，指标才跟别的版本可比。
    返回 (X_aug, y_aug, y_seq_aug)，原始窗口在前。"""
    if copies <= 0 or len(X) == 0 or X.ndim != 3:
        return X, y, y_seq
    rng = np.random.default_rng(seed)
    n, t, c = X.shape
    outs = [X.astype(np.float32)]
    frac = np.linspace(0.0, 1.0, t)
    for _ in range(copies):
        Xa = np.empty_like(X, dtype=np.float32)
        for i in range(n):
            R0 = _random_rotation(rng)
            axis = rng.normal(size=3)
            amax = np.deg2rad(rng.uniform(-drift_deg, drift_deg))
            w = X[i].astype(np.float64)
            out = np.empty((t, c))
            out[:, 3:] = w[:, 3:] if c > 3 else 0
            for k in range(t):
                R = _axis_angle(axis, amax * frac[k]) @ R0
                out[k, :3] = R @ w[k, :3]
                if c >= 6:
                    out[k, 3:6] = R @ w[k, 3:6]
            Xa[i] = out
        outs.append(Xa)
    X_out = np.concatenate(outs)
    y_out = np.concatenate([y] * (copies + 1))
    ys_out = np.concatenate([y_seq] * (copies + 1)) if y_seq is not None and len(y_seq) else y_seq
    return X_out, y_out, ys_out
