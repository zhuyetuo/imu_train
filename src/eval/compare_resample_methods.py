"""
对比两套降采样算法在同一份合成信号上的差异：
  1. training_match：witmotion_imu/imu_camera_sync.py::resample_raw_imu 实际用的方法
     （滑动平均低通 + np.interp 线性插值）——训练数据（raw_custom 那批16Hz CSV）是这样生成的
  2. poly：infer_csv_scratch.py::downsample() 现在用的方法（scipy.signal.resample_poly）
     ——推理时对非16Hz原生数据（比如TF的50Hz）现场降采样用的是这个

验证：从同一段"高采样率原始信号"分别用两种方法降到16Hz，量化两者差多少，
不是单纯理论推导，实测给个数字。

用法：
    python src/eval/compare_resample_methods.py
"""
import os
import sys

import numpy as np
from scipy.signal import resample_poly
from math import gcd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from resample_training_match import resample_training_match  # noqa: E402


def resample_poly_method(data, source_hz, target_hz):
    """复刻 infer_csv_scratch.py::downsample() 的逻辑。"""
    if source_hz == target_hz:
        return data
    g = gcd(int(source_hz), int(target_hz))
    up, down = int(target_hz) // g, int(source_hz) // g
    if up == 1:
        return data[::down]
    return resample_poly(data, up, down, axis=0).astype(np.float32)


def make_test_signal(source_hz, duration_s=10.0):
    """模拟一段带有典型狗行为频率成分(步态~2-3Hz + 抓挠~4-8Hz)的合成信号，
    加一点高频噪声，用来看两种方法处理"该保留的低频信号"和"该滤掉的高频噪声"效果差多少。
    """
    n = int(duration_s * source_hz)
    t = np.arange(n) / source_hz
    rng = np.random.default_rng(0)
    gait = 0.3 * np.sin(2 * np.pi * 2.5 * t)          # 步态，2.5Hz，应该被两种方法都保留
    scratch = 0.15 * np.sin(2 * np.pi * 6.0 * t)       # 抓挠动作频率，6Hz，接近16Hz降采样后8Hz奈奎斯特的边界
    noise = rng.normal(0, 0.05, n)                     # 高频噪声，应该被两种方法都滤掉大半
    sig = 1.0 + gait + scratch + noise                 # 叠加在重力上
    return np.stack([sig, sig * 0.3, sig * 0.5], axis=1)  # 凑成3轴，制造一点轴间差异


def main():
    for source_hz in [50, 100]:
        print(f"\n{'='*70}\n源采样率: {source_hz}Hz → 目标: 16Hz\n{'='*70}")
        data = make_test_signal(source_hz)

        out_train = resample_training_match(data, source_hz, 16)
        out_poly = resample_poly_method(data, source_hz, 16)

        n = min(len(out_train), len(out_poly))
        out_train, out_poly = out_train[:n], out_poly[:n]

        diff = out_train - out_poly
        rmse = np.sqrt(np.mean(diff ** 2))
        rel = rmse / np.std(out_poly)

        print(f"  training_match 输出行数: {len(resample_training_match(data, source_hz, 16))}")
        print(f"  poly 方法输出行数: {len(resample_poly_method(data, source_hz, 16))}")
        print(f"  两种方法输出信号的 RMSE: {rmse:.4f}  (相对于poly方法信号标准差的比例: {rel*100:.1f}%)")

        # 看一下高频噪声抑制效果差多少：目标输出信号里，8Hz以上频段残留能量对比
        from numpy.fft import rfft, rfftfreq
        for name, out in [("training_match", out_train), ("poly", out_poly)]:
            spec = np.abs(rfft(out[:, 0] - out[:, 0].mean()))
            freqs = rfftfreq(len(out), d=1 / 16)
            high_freq_energy = spec[freqs > 6.5].sum()
            total_energy = spec.sum()
            print(f"  [{name}] >6.5Hz 频段能量占比: {high_freq_energy/total_energy*100:.1f}%"
                  f"（越低说明抗混叠滤波效果越好）")


if __name__ == "__main__":
    main()
