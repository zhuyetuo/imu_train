"""疑似进食 / 饮水 / 吠叫候选（behaviors.py）：合成数据上看规则分不分得开。

    python -m pytest label_service/test_behaviors.py -q

**这里验证的是规则按设计工作，不是它在真狗身上准**——这几类还没有标注数据，
阈值是按动作的物理特点给的初值。合成的形状照着那几条物理特点造：
舔水规整 4 Hz、吃饭 1.2 Hz 慢往复、舔身体 2.5 Hz（跟 test_grooming 同一段），
叫一声是几十毫秒的尖脉冲、声与声之间安静。
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
G = importlib.import_module("label_service.grooming")
B = importlib.import_module("label_service.behaviors")

HZ = 50.0
T0 = pd.Timestamp("2026-09-17 10:00:00")


def _rot(tilt_deg):
    t = np.radians(tilt_deg)
    return np.array([0.0, np.sin(t), np.cos(t)], dtype=np.float32)


def _base(total_s, seed=0):
    rng = np.random.default_rng(seed)
    n = int(total_s * HZ)
    t = np.arange(n) / HZ
    acc = np.tile(_rot(0.0), (n, 1)) + rng.normal(0, 0.005, (n, 3)).astype(np.float32)
    ts = pd.Series(T0 + pd.to_timedelta(t, unit="s"))
    return rng, t, acc, np.ones(n, bool), ts


def _put(acc, t, rng, s0, s1, tilt, amp, f, noise=0.005):
    a, b = int(s0 * HZ), int(s1 * HZ)
    osc = amp * np.sin(2 * np.pi * f * t[a:b])
    acc[a:b] = _rot(tilt) + osc[:, None] * np.array([0.3, 1.0, 0.2], np.float32) + rng.normal(0, noise, (b - a, 3))


def _sec(c, key):
    return (pd.Timestamp(c[key]) - T0).total_seconds()


def _classify(acc, ts, valid):
    return G.mine(acc, ts, valid, HZ, classify=lambda a, h, m, d: B.classify_posture_bout(a, h, m, d))


def test_posture_bouts_are_split_into_drink_eat_lick():
    rng, t, acc, valid, ts = _base(900)
    _put(acc, t, rng, 120, 145, tilt=50, amp=0.06, f=4.0)    # 舔水：规整 4 Hz
    _put(acc, t, rng, 300, 345, tilt=55, amp=0.15, f=1.2)    # 吃饭：1.2 Hz 慢往复
    _put(acc, t, rng, 600, 620, tilt=50, amp=0.08, f=2.5)    # 舔身体：2.5 Hz
    cands = _classify(acc, ts, valid)
    got = sorted((round(_sec(c, "start_ts")), c["label"], c["reason"]) for c in cands)
    labels = [g[1] for g in got]
    assert labels == ["饮水", "进食", "舔身体"], got
    assert [g[2] for g in got] == ["drinking", "eating", "grooming"]
    for (s, _l, _r), want in zip(got, (120, 300, 600)):
        assert abs(s - want) <= 2


def test_split_does_not_add_or_drop_bouts():
    """分类只换类别：段的个数和起止跟不分类时一模一样（舔的召回不受影响）。"""
    rng, t, acc, valid, ts = _base(900)
    _put(acc, t, rng, 120, 145, tilt=50, amp=0.06, f=4.0)
    _put(acc, t, rng, 300, 345, tilt=55, amp=0.15, f=1.2)
    _put(acc, t, rng, 600, 620, tilt=50, amp=0.08, f=2.5)
    plain = G.mine(acc, ts, valid, HZ)
    split = _classify(acc, ts, valid)
    assert [(c["start_ts"], c["end_ts"]) for c in plain] == [(c["start_ts"], c["end_ts"]) for c in split]


def _bark(acc, t0_s, n, gap_s, height=0.5):
    for k in range(n):
        i = int((t0_s + k * gap_s) * HZ)
        acc[i] += np.array([0.1, 0.3, 1.0], np.float32) * height       # 一个采样点的尖峰
        acc[i + 1] += np.array([0.05, 0.15, 0.5], np.float32) * height


def test_a_run_of_barks_is_found():
    rng, t, acc, valid, ts = _base(300)
    _bark(acc, 100.0, n=5, gap_s=0.6)
    cands = B.mine_barks(acc, ts, valid, HZ)
    assert len(cands) == 1, cands
    c = cands[0]
    assert c["label"] == "吠叫" and c["reason"] == "barking" and c["n_barks"] == 5
    assert 99 <= _sec(c, "start_ts") <= 100.0 and 102.4 <= _sec(c, "end_ts") <= 103.5


def test_single_spike_or_two_is_not_a_bark():
    """碰一下、蹭一下也是一个尖峰；至少三声才算一串。"""
    rng, t, acc, valid, ts = _base(300)
    _bark(acc, 50.0, n=1, gap_s=0.6)
    _bark(acc, 150.0, n=2, gap_s=0.6)
    assert B.mine_barks(acc, ts, valid, HZ) == []


def test_walking_and_shaking_are_not_barks():
    """走路落脚、甩身体都有一串峰，但峰之间不安静。"""
    rng, t, acc, valid, ts = _base(300)
    _put(acc, t, rng, 50, 80, tilt=40, amp=0.8, f=2.0, noise=0.05)    # 走 / 小跑
    _put(acc, t, rng, 150, 154, tilt=10, amp=1.2, f=5.0, noise=0.05)  # 甩身体
    # 走路再加上落脚的冲击
    for k in range(60):
        i = int((50 + k * 0.5) * HZ)
        acc[i] += np.array([0, 0.2, 0.6], np.float32)
    assert B.mine_barks(acc, ts, valid, HZ) == []


def test_barks_in_missing_data_are_ignored():
    rng, t, acc, valid, ts = _base(300)
    _bark(acc, 100.0, n=5, gap_s=0.6)
    valid[int(99 * HZ):int(104 * HZ)] = False
    assert B.mine_barks(acc, ts, valid, HZ) == []
