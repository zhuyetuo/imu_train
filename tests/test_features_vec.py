"""
向量化特征必须跟逐窗口的老实现输出完全一致——模型是用老特征训出来的，
差一点点就等于换了一套输入。这里拿随机数据和几种容易出岔子的形状逐元素比对。

跑法：pytest tests/test_features_vec.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "ml"))

from features import _extract_one  # noqa: E402
from features_vec import extract_features_vec  # noqa: E402


def _reference(X, hz):
    """老实现：逐窗口。不走 extract_features 是因为它现在默认转发到向量化版本。"""
    if len(X) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([_extract_one(X[i], hz) for i in range(len(X))])


def _same(X, hz):
    a, b = _reference(X, hz), extract_features_vec(X, hz)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_ch", [6, 8])
def test_random(n_ch):
    rng = np.random.default_rng(0)
    _same(rng.normal(size=(40, 100, n_ch)).astype(np.float32), 50)


def test_constant_window():
    """常数信号：偏度/峰度/相关系数都要走 std<=1e-8 那条分支，算出来该是 0 不是 nan"""
    _same(np.zeros((5, 100, 6), dtype=np.float32), 50)


def test_plateau():
    """掉数据被前值填充出来的平台：峰值计数最容易在这里跟 scipy 对不上"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 100, 6)).astype(np.float32)
    X[:, 30:60, :] = X[:, 30:31, :]
    _same(X, 50)


def test_staircase():
    """整段都是平台的阶梯信号，把峰值判据推到极端"""
    rng = np.random.default_rng(2)
    step = np.repeat(rng.integers(0, 5, size=(10, 20)), 5, axis=1).astype(np.float32)
    _same(np.repeat(step[:, :, None], 6, axis=2), 50)


@pytest.mark.parametrize("hz", [25, 50, 100])
def test_sine_various_hz(hz):
    """采样率影响 welch 的分频段边界，几个常见值都过一遍"""
    rng = np.random.default_rng(3)
    t = np.arange(128) / hz
    X = (np.sin(2 * np.pi * 6 * t)[None, :, None] + rng.normal(scale=0.1, size=(15, 128, 6))).astype(np.float32)
    _same(X, hz)


def test_empty():
    out = extract_features_vec(np.empty((0, 100, 6), dtype=np.float32), 50)
    assert out.shape[0] == 0
