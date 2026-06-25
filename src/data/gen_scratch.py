"""
生成抓挠行为的伪 IMU 数据，注入训练集。

真实场景建模：
  - 狗坐着/站着，后腿抬起抓颈部项圈区域
  - 产生 3-6Hz 节律性振荡（每秒3-6次抓挠动作）
  - 加速度：横向(X/Y)有规律波动，Z 保持重力分量
  - 陀螺仪：对应角速度振荡 ~10-25 deg/s
  - 每次抓挠 2-8 秒，间隔短暂静止
  - 叠加真实噪声和个体差异

用法:
  python src/data/gen_scratch.py --out data/synthetic/scratch_50hz.npz
  python src/data/gen_scratch.py --out data/synthetic/scratch_50hz.npz --n_windows 2000 --seed 42
"""

import argparse
import os
import numpy as np


def gen_scratch_window(rng: np.random.Generator, window_size: int, hz: int) -> np.ndarray:
    """生成一个抓挠窗口，shape=(window_size, 6)，列=[AX,AY,AZ,GX,GY,GZ]。"""
    t = np.arange(window_size) / hz

    # ── 基础姿态：坐姿，重力主要在 Z 轴，略有倾斜 ──────────────────────────
    # 单位：m/s²，重力 ~9.81
    tilt_x = rng.uniform(-0.8, 0.8)    # 项圈左右倾斜
    tilt_y = rng.uniform(-0.5, 0.5)    # 项圈前后倾斜
    base_ax = tilt_x
    base_ay = tilt_y
    base_az = np.sqrt(max(0, 9.81**2 - tilt_x**2 - tilt_y**2))  # 重力分量

    # ── 抓挠振荡：节律性，3-6Hz，可能中途短暂停顿 ──────────────────────────
    scratch_hz   = rng.uniform(3.0, 6.0)    # 抓挠频率
    amp_x        = rng.uniform(0.8, 2.5)    # 横向振幅（主方向）
    amp_y        = rng.uniform(0.3, 1.2)    # 纵向振幅（次方向）
    phase_x      = rng.uniform(0, 2 * np.pi)
    phase_y      = phase_x + rng.uniform(0.2, 0.8)  # 轻微相位差

    # 抓挠强度包络：模拟中途停顿（0.1-0.3 的概率有短暂停顿）
    envelope = np.ones(window_size)
    if rng.random() < 0.3:
        pause_start = rng.integers(window_size // 4, window_size * 3 // 4)
        pause_len   = rng.integers(int(hz * 0.3), int(hz * 1.0))
        pause_end   = min(window_size, pause_start + pause_len)
        envelope[pause_start:pause_end] = rng.uniform(0.0, 0.15)  # 近乎静止

    osc_x = amp_x * np.sin(2 * np.pi * scratch_hz * t + phase_x) * envelope
    osc_y = amp_y * np.sin(2 * np.pi * scratch_hz * t + phase_y) * envelope
    # Z 轴因整体躯干轻微晃动，有小幅次谐波
    osc_z = rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * scratch_hz * 0.5 * t) * envelope

    ax = base_ax + osc_x
    ay = base_ay + osc_y
    az = base_az + osc_z

    # ── 陀螺仪：与加速度对应的角速度振荡，单位 deg/s ───────────────────────
    gyro_scale_x = rng.uniform(8.0, 20.0)   # deg/s per m/s² (近似)
    gyro_scale_y = rng.uniform(5.0, 15.0)
    gx = gyro_scale_x * np.sin(2 * np.pi * scratch_hz * t + phase_x + 0.3) * envelope
    gy = gyro_scale_y * np.sin(2 * np.pi * scratch_hz * t + phase_y + 0.3) * envelope
    gz = rng.uniform(2.0, 8.0) * np.sin(2 * np.pi * scratch_hz * 0.5 * t) * envelope

    # ── 传感器噪声 ──────────────────────────────────────────────────────────
    noise_acc  = rng.normal(0, 0.12, (window_size, 3))
    noise_gyro = rng.normal(0, 0.8,  (window_size, 3))

    window = np.column_stack([ax, ay, az, gx, gy, gz]) + \
             np.column_stack([noise_acc, noise_gyro])
    return window.astype(np.float32)


def generate(n_windows: int, window_size: int, hz: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    windows = np.stack([gen_scratch_window(rng, window_size, hz)
                        for _ in range(n_windows)])
    return windows  # (n_windows, window_size, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",        default="data/synthetic/scratch_50hz.npz")
    parser.add_argument("--n_windows",  type=int, default=1500,
                        help="生成的抓挠窗口数（每个窗口2秒，1500个≈50分钟数据）")
    parser.add_argument("--window_size",type=int, default=100,
                        help="每窗口帧数（默认100帧@50Hz=2秒）")
    parser.add_argument("--hz",         type=int, default=50)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    print(f"[gen_scratch] 生成 {args.n_windows} 个抓挠窗口 "
          f"({args.window_size}帧@{args.hz}Hz = {args.window_size/args.hz:.1f}秒/窗口)")

    windows = generate(args.n_windows, args.window_size, args.hz, args.seed)
    print(f"[gen_scratch] shape={windows.shape}  "
          f"acc范围=[{windows[:,:,:3].min():.2f}, {windows[:,:,:3].max():.2f}]  "
          f"gyro范围=[{windows[:,: ,3:].min():.2f}, {windows[:,:,3:].max():.2f}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=windows)
    print(f"[gen_scratch] 已保存至 {args.out}")


if __name__ == "__main__":
    main()
