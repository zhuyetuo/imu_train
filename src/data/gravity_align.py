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
