"""训练的降采样必须跟推理一模一样（2026-09-23：单数据集训练没降采样，50Hz 当 16Hz 用）。"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "data"))

from resample_csv_hz import resample_df  # noqa: E402
from resample_training_match import resample_training_match  # noqa: E402

SH = (ROOT / "train_custom.sh").read_text()


def test_single_dataset_branch_resamples():
    # 没有额外批次时也要按 source_hz 降采样
    assert re.search(r'EXTRA_DATES\[@\]\} -eq 0 && -n "\$SOURCE_HZ"', SH)
    assert "resample_csv_hz.py" in SH and '--method "$RESAMPLE_METHOD"' in SH


def test_default_method_matches_inference():
    from label_service import config
    m = re.search(r'^RESAMPLE_METHOD="(\w+)"', SH, re.M)
    assert m and m.group(1) == config.RESAMPLE_METHOD


def test_build_command_always_passes_source_hz():
    from label_service import config, jobs
    cmd = jobs.build_command({"date": "d"}, "rf", None, 1)
    assert cmd[cmd.index("--source_hz") + 1] == str(config.DEVICE_HZ)
    assert cmd[cmd.index("--resample_method") + 1] == config.RESAMPLE_METHOD


def test_resample_df_equals_inference_path():
    rng = np.random.default_rng(0)
    n = 500
    cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    data = rng.normal(size=(n, 6))
    df = pd.DataFrame(data, columns=cols)
    df.insert(0, "label", ["a"] * 250 + ["b"] * 250)
    df.insert(0, "record_id", "r1")
    out = resample_df(df, 50, 16)
    ref = resample_training_match(data, 50, 16)
    np.testing.assert_allclose(out[cols].to_numpy(), ref, rtol=1e-5, atol=1e-6)
    assert abs(len(out) - n * 16 / 50) <= 2
    assert set(out["label"]) == {"a", "b"}


def test_synthesize_segment_resampled():
    from synthesize_scratch import _seg_hz, _to_hz
    ts = pd.Series(pd.date_range("2026-01-01", periods=100, freq="20ms"))
    assert abs(_seg_hz(ts) - 50) < 0.5
    seg = np.ones((100, 3), dtype=np.float32)
    assert abs(len(_to_hz(seg, 50, 16, "training_match")) - 32) <= 1
    assert len(_to_hz(seg, 15.8, 16, "training_match")) == 100  # 差不到 10% 不动


def test_qmi8658_native_31_25hz():
    """QMI8658 没有 16 / 25 / 50 Hz 档位，原生是 31.25（15.625 = 二分频）。训练、推理都要认小数采样率：
    重采样比例是整数比、窗口点数是整数、推理侧换回点数跟训练一致。"""
    from hzutil import n_points, parse_hz, rate_ratio
    from preprocess import downsample

    assert parse_hz("31.25") == 31.25 and parse_hz("16") == 16 and isinstance(parse_hz("16"), int)
    assert rate_ratio(50, 31.25) == (5, 8) and rate_ratio(50, 15.625) == (5, 16) and rate_ratio(50, 16) == (8, 25)
    x, y = downsample(np.random.randn(800, 6), np.array(["a"] * 800), 50, 31.25)
    assert len(x) == 500 and len(y) == 500
    # 训练 64 点，meta 存 window_s=round(64/31.25, 3)=2.048，推理按它换回来还是 64
    assert n_points(round(64 / 31.25, 3), 31.25) == 64 and n_points(round(32 / 31.25, 3), 31.25) == 32
    df = pd.DataFrame({"record_id": 1, "label": "a", "timestamp": 0,
                       **{c: np.random.randn(500) for c in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]}})
    out = resample_df(df, 50, 31.25)
    assert abs(len(out) - 312) <= 1


def test_window_points_integer_at_31_25hz():
    """2 秒 / 1 秒在 31.25 Hz 下是 62.5 / 31.25 点：没指定窗口时自动用 64 / 32 点（2.048 / 1.024 秒）"""
    from label_service import jobs

    assert jobs.window_args({"hz": 16}) == []
    assert jobs.window_args({"hz": 31.25}) == ["--window_s", "2.048", "--stride_s", "1.024"]
    assert jobs.window_args({"hz": 15.625}) == ["--window_s", "2.048", "--stride_s", "1.024"]
    cmd = jobs.build_command({"date": "d", "hz": 31.25, "source_hz": 50}, "rf", None, 9)
    assert cmd[cmd.index("--hz") + 1] == "31.25" and cmd[cmd.index("--window_s") + 1] == "2.048"
