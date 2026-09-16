"""「甩身体和抓挠分不开，是不是 16Hz 太低？提到 25Hz 就好了？」

不讲道理，造两个已知频率的信号跑一遍：

    python -m pytest diagnose/test_resolution.py -q

实测（抓挠 5.0Hz、甩身体 4.5Hz，60 个噪声种子）：

    配置        分辨率    抓挠主频    甩身体主频     结论
    16Hz/1s     1.000    [5.0]      [4.0, 5.0]    重合
    25Hz/1s     1.000    [5.0]      [4.0, 5.0]    重合  ← 跟 16Hz 一模一样
    50Hz/1s     1.563    [4.69]     [4.69]        重合，分辨率反而更差
    16Hz/2s     0.500    [5.0]      [4.5]         **稳定分开**
    25Hz/2s     0.781    [4.69]     [4.69]        重合

三条结论，都在下面钉住了：

1. **分辨率 = fs / nperseg**，而 `nperseg = min(窗口点数, 32)`。
   没被截断时 `窗口点数 = fs × 窗口秒数`，fs 约掉 → **分辨率 = 1/窗口秒数**，
   跟采样率无关。
2. 所以 16Hz→25Hz 在 1 秒窗口下**分辨率纹丝不动**，两类照样重合。
   4.5 跟 5.0 只差 0.5Hz，不到一格，落在哪一格由噪声决定——**不稳定**。
3. 采样率不是没用：它决定**能看到多高的频率**（Nyquist）。9Hz 在 16Hz 下
   会混叠，25Hz 下看得到。所以诊断脚本里还有一条「多少窗口贴着 Nyquist」。

顺带一个坑：`min(x, 32)` 那个上限让 **50Hz/1s 的分辨率(1.56Hz)比
16Hz/1s(1.0Hz) 还差**，25Hz/2s 也只有 0.78Hz 而不是 0.5Hz。
这个上限写死在 `src/ml/features.py:_freq_stats_1d` 里。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp

F_SCRATCH, F_SHAKE = 5.0, 4.5      # 抓挠 / 甩身体的典型往复频率
NPERSEG_CAP = 32                   # 跟 src/ml/features.py 的 min(len(x), 32) 一致
SEEDS = range(60)


def _dominant(freq_hz, fs, win_s, seed=0):
    """造一个 freq_hz 的正弦（带噪），返回 Welch 主频。跟诊断脚本同一套参数。"""
    rng = np.random.default_rng(seed)
    n = int(round(fs * win_s))
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * freq_hz * t) + rng.normal(0, 0.1, n)
    freqs, psd = sp.welch(x, fs=fs, nperseg=min(n, NPERSEG_CAP))
    return float(freqs[int(np.argmax(psd))])


def _spread(freq_hz, fs, win_s):
    """多个噪声种子下主频落在哪几格。只有一格 = 稳定。"""
    return sorted({_dominant(freq_hz, fs, win_s, s) for s in SEEDS})


def _resolution(fs, win_s):
    return fs / min(int(round(fs * win_s)), NPERSEG_CAP)


# ── 分辨率公式 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fs", [16, 25])
def test_resolution_is_one_over_window_seconds(fs):
    """没撞上 nperseg 上限时，1 秒窗口 → 1Hz，**跟采样率无关**。"""
    assert _resolution(fs, 1.0) == pytest.approx(1.0)


def test_longer_window_is_what_buys_resolution():
    assert _resolution(16, 2.0) == pytest.approx(0.5)


def test_the_nperseg_cap_can_make_a_higher_rate_worse():
    """**50Hz/1 秒的分辨率比 16Hz/1 秒还差。**

    nperseg 被 min(len(x), 32) 截断之后，分辨率变成 fs/32——采样率越高
    反而越粗。这个上限写死在 src/ml/features.py:_freq_stats_1d 里。
    想当然地"提采样率总没坏处"在这里是错的。
    """
    assert _resolution(50, 1.0) > _resolution(16, 1.0)
    assert _resolution(50, 1.0) == pytest.approx(50 / 32)
    # 2 秒窗口同理：25Hz 有 50 点，被截到 32，拿不到 0.5Hz
    assert _resolution(25, 2.0) == pytest.approx(25 / 32)


# ── 实际能不能分开 ────────────────────────────────────────────────────────


def test_16hz_one_second_cannot_separate_them():
    """16Hz + 1 秒：甩身体的主频在 4 和 5 之间跳，跟抓挠重合。"""
    a, b = _spread(F_SCRATCH, 16, 1.0), _spread(F_SHAKE, 16, 1.0)
    assert set(a) & set(b), f"这一档本该重合，却得到 {a} vs {b}"


def test_raising_the_sample_rate_does_not_help():
    """**这条是整件事的重点：25Hz 跟 16Hz 一模一样，照样分不开。**

    提采样率抬的是 Nyquist，不是分辨率。1 秒窗口下两者都是 1Hz 一格，
    而 4.5 跟 5.0 只差半格。
    """
    assert _resolution(25, 1.0) == _resolution(16, 1.0)
    a, b = _spread(F_SCRATCH, 25, 1.0), _spread(F_SHAKE, 25, 1.0)
    assert set(a) & set(b), f"25Hz 下居然分开了（{a} vs {b}）——这条结论要重新检查"


def test_the_shake_frequency_is_unstable_at_one_hz_resolution():
    """4.5Hz 卡在格子边上，**落到哪一格由噪声决定**。

    这比"分不开"更糟：同一个动作，有时报 4Hz 有时报 5Hz，
    模型拿到的这一维特征本身就是抖的。
    """
    for fs in (16, 25):
        assert len(_spread(F_SHAKE, fs, 1.0)) > 1, \
            f"{fs}Hz 下 4.5Hz 本该不稳定"


def test_a_longer_window_does_help():
    """同样 16Hz，窗口加到 2 秒：两类各自稳定在一格，而且分开。"""
    a, b = _spread(F_SCRATCH, 16, 2.0), _spread(F_SHAKE, 16, 2.0)
    assert a == [F_SCRATCH], f"抓挠应该稳定在 5.0，实际 {a}"
    assert b == [F_SHAKE], f"甩身体应该稳定在 4.5，实际 {b}"


def test_higher_frequencies_do_need_a_higher_sample_rate():
    """采样率**不是没用**，它决定能看到多高的频率。

    9Hz 超过 16Hz 采样的 Nyquist（8Hz），会混叠成别的频率；25Hz 看得到。
    诊断脚本里"多少窗口贴着 Nyquist"那一条盯的就是这个。
    """
    assert _dominant(9.0, 16, 1.0) != pytest.approx(9.0, abs=1.0), "16Hz 下 9Hz 本该混叠"
    assert _dominant(9.0, 25, 1.0) == pytest.approx(9.0, abs=1.0)


# ── 诊断脚本里那段话跟代码一致 ────────────────────────────────────────────


def test_the_script_says_resolution_not_sample_rate():
    """脚本里那段解释别哪天被改反了。"""
    import os

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "shake_vs_scratch.py")
    with open(p, encoding="utf-8") as f:
        src = f.read()
    assert "频率分辨率由窗口时长决定，不由采样率决定" in src
    assert "res = hz / nperseg" in src
