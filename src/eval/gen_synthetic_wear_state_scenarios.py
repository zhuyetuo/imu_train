"""
生成合成原始 IMU 数据，验证 wear_state.py 里的松动检测 + 未佩戴/静置(心跳呼吸微振动)检测机制。

跟 gen_synthetic_scratch_scenarios.py 是同一个思路：验证的是检测逻辑本身对不对，
不是验证真机信噪比够不够（那个必须用真实数据才能回答，见 docs/wear_state_detection.md）。

用法：
    python src/eval/gen_synthetic_wear_state_scenarios.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from wear_state import (  # noqa: E402
    classify_wear_state, detect_loose_window, aggregate_loose_events,
)

HZ = 50  # 合成用较高采样率，模拟"未降采样原始流"，见文档里关于采样率的讨论
rng = np.random.default_rng(42)


def _gravity_baseline():
    return np.array([0.0, 0.0, 1.0])  # g，单位近似（不追求物理量纲精确，只看相对结构）


def gen_walking_segment(duration_s, hz=HZ, gait_hz=2.0, wobble_std=0.0, drift_per_sec=0.0, noise=0.02):
    """模拟稳定行走：垂直加速度周期性起伏(步态) + 项圈朝向漂移(wobble_std/drift_per_sec 越大越松)。"""
    n = int(duration_s * hz)
    t = np.arange(n) / hz
    g = _gravity_baseline()
    acc = np.tile(g, (n, 1)).astype(float)

    gait = 0.3 * np.sin(2 * np.pi * gait_hz * t)
    acc[:, 2] += gait  # 步态起伏主要体现在垂直方向

    # 项圈朝向漂移：随时间累积的低频摆动，drift_per_sec 越大代表越松，摆动幅度越来越大
    drift = drift_per_sec * t
    acc[:, 0] += drift * np.sin(2 * np.pi * 0.3 * t) + rng.normal(0, wobble_std, n)
    acc[:, 1] += drift * np.cos(2 * np.pi * 0.3 * t) + rng.normal(0, wobble_std, n)

    acc += rng.normal(0, noise, (n, 3))
    return acc


def gen_resting_worn_segment(duration_s, hz=HZ, resp_hz=0.3, hr_hz=1.5,
                              resp_amp=0.015, hr_amp=0.008, noise=0.003):
    """模拟佩戴中静息/睡眠：宏观几乎不动，但叠加呼吸+心率引起的微幅周期性振动。

    这里的振幅/底噪是为了让检测机制本身可演示而选的合成参数，不代表真机的真实信噪比——
    真机的心跳/呼吸信号相对传感器底噪到底有多大，必须用真实设备+真实狗验证，见文档 §3。
    """
    n = int(duration_s * hz)
    t = np.arange(n) / hz
    g = _gravity_baseline()
    acc = np.tile(g, (n, 1)).astype(float)
    bio = resp_amp * np.sin(2 * np.pi * resp_hz * t) + hr_amp * np.sin(2 * np.pi * hr_hz * t)
    acc[:, 2] += bio
    acc += rng.normal(0, noise, (n, 3))
    return acc


def gen_off_body_segment(duration_s, hz=HZ, noise=0.003):
    """模拟未佩戴/静置(放桌上/充电)：宏观不动，且没有生物周期信号，只有传感器底噪。"""
    n = int(duration_s * hz)
    g = _gravity_baseline()
    acc = np.tile(g, (n, 1)).astype(float)
    acc += rng.normal(0, noise, (n, 3))
    return acc


def gen_loosening_walk_scenario():
    """场景1：一段30分钟的稳定行走，项圈松动程度随时间从0线性增加到明显松动。"""
    segs, windows = [], []
    window_s = 5.0
    total_min = 30
    for i in range(int(total_min * 60 / window_s)):
        frac = i / (total_min * 60 / window_s)  # 0→1，模拟越走越松
        drift = frac * 0.08  # 到后期明显超过阈值
        seg = gen_walking_segment(window_s, drift_per_sec=drift, wobble_std=0.01 + frac * 0.03)
        segs.append(seg)
        windows.append((i * window_s, seg, True))  # is_stable_gait=True
    return "loosening_walk", windows, "松动程度随时间递增，前期不应触发，后期应稳定检测到松动"


def gen_stable_collar_walk_scenario():
    """场景2：全程佩戴稳固的行走，对照组，不应触发任何松动。"""
    window_s = 5.0
    total_min = 30
    windows = []
    for i in range(int(total_min * 60 / window_s)):
        seg = gen_walking_segment(window_s, drift_per_sec=0.0, wobble_std=0.01)
        windows.append((i * window_s, seg, True))
    return "stable_collar_walk", windows, "全程佩戴稳固，不应触发任何松动事件"


def gen_worn_sleep_scenario():
    """场景3：长时间佩戴中睡眠，宏观静止但有心跳/呼吸微振动，应判定 worn_resting。"""
    window_s = 30.0
    total_min = 60
    windows = []
    for i in range(int(total_min * 60 / window_s)):
        seg = gen_resting_worn_segment(window_s)
        windows.append((i * window_s, seg, False))
    return "worn_sleep", windows, "全程应判定 worn_resting（有生物信号），不应被误判成 off_body"


def gen_off_body_scenario():
    """场景4：设备被摘下放置/充电，宏观静止且无生物信号，应判定 off_body_or_static。"""
    window_s = 30.0
    total_min = 60
    windows = []
    for i in range(int(total_min * 60 / window_s)):
        seg = gen_off_body_segment(window_s)
        windows.append((i * window_s, seg, False))
    return "off_body_static", windows, "全程应判定 off_body_or_static（无生物信号）"


def gen_mixed_day_scenario():
    """场景5：一天里活动→睡眠→摘下充电→重新佩戴活动的完整周期，验证状态切换。"""
    windows = []
    t0 = 0.0
    window_s = 30.0

    def add(gen_fn, minutes, is_gait=False, **kw):
        nonlocal t0
        n_win = int(minutes * 60 / window_s)
        for _ in range(n_win):
            seg = gen_fn(window_s, **kw) if gen_fn is not gen_walking_segment else \
                gen_walking_segment(window_s, **kw)
            windows.append((t0, seg, is_gait))
            t0 += window_s

    add(gen_resting_worn_segment, 20)          # 早上还在睡
    add(gen_walking_segment, 15, is_gait=True)  # 起来活动
    add(gen_off_body_segment, 40)               # 摘下充电
    add(gen_walking_segment, 15, is_gait=True)  # 重新佩戴活动
    add(gen_resting_worn_segment, 20)           # 晚上睡觉
    return "mixed_day", windows, "应观察到 worn_resting→worn_active→off_body_or_static→worn_active→worn_resting 的正确切换"


SCENARIOS = [
    gen_loosening_walk_scenario,
    gen_stable_collar_walk_scenario,
    gen_worn_sleep_scenario,
    gen_off_body_scenario,
    gen_mixed_day_scenario,
]


def run_scenario(name, windows, expect):
    print(f"\n### {name}")
    print(f"预期: {expect}")

    loose_flags = []
    wear_results = []
    for t, seg, is_gait in windows:
        is_loose, drift = detect_loose_window(seg, HZ, is_gait)
        if is_gait:
            loose_flags.append((t, is_loose))
        state, detail = classify_wear_state(seg, HZ, is_stable_gait=is_gait)
        wear_results.append((t, state, detail))

    events = aggregate_loose_events(loose_flags, window_sec=5.0)
    print(f"松动事件数: {len(events)}  {events[:3]}{'...' if len(events) > 3 else ''}")

    # 打印 wear_state 的分段摘要（合并连续相同状态）
    summary = []
    for t, state, detail in wear_results:
        if summary and summary[-1][1] == state:
            summary[-1] = (summary[-1][0], state, summary[-1][2] + 1)
        else:
            summary.append((t, state, 1))
    print("wear_state 分段摘要 (起始秒, 状态, 连续窗口数):")
    for row in summary:
        print(f"  {row}")


def main():
    for fn in SCENARIOS:
        name, windows, expect = fn()
        run_scenario(name, windows, expect)


if __name__ == "__main__":
    main()
