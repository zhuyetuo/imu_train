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
