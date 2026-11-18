"""采集阶段蓝牙断了写的「六轴全 0」占位行：读时间线时当缺失，跨过它的窗口不用"""
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "ml"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))
import train_v2 as T  # noqa: E402


def _csv(tmp_path, zero_from_s=None, zero_to_s=None, hz=50, secs=20):
    n = hz * secs
    t0 = datetime(2026, 11, 18, 10, 0, 0)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": [(t0 + timedelta(seconds=i / hz)).strftime("%Y-%m-%d %H:%M:%S.%f") for i in range(n)],
        "acc_x": 0.02 * rng.standard_normal(n), "acc_y": 0.02 * rng.standard_normal(n), "acc_z": 1 + 0.02 * rng.standard_normal(n),
        "gyro_x": rng.standard_normal(n), "gyro_y": rng.standard_normal(n), "gyro_z": rng.standard_normal(n),
    })
    if zero_from_s is not None:
        sl = slice(int(zero_from_s * hz), int(zero_to_s * hz))
        df.loc[sl, ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]] = 0.0
    p = tmp_path / "x_raw.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_全0占位那段标成掉包(tmp_path):
    tl = T.load_timeline(_csv(tmp_path, 8, 10), 20.0, 50, axes=6)
    bad = tl["bad"]
    assert bad[int(8.5 * 20):int(9.5 * 20)].all()          # 全 0 那 2 秒是掉包
    assert not bad[: int(7 * 20)].any() and not bad[int(11 * 20):].any()
    assert (np.abs(tl["x"][~bad, 2]) > 0.5).all()           # 留下的都是真数据（竖直方向有重力）


def test_没有全0的文件不受影响(tmp_path):
    tl = T.load_timeline(_csv(tmp_path), 20.0, 50, axes=6)
    assert not tl["bad"].any()
