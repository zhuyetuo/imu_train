"""设备调试页直传原始采样跑模型：写出来的 CSV 走同一条推理链能读，时间按 UTC 往返不丢不偏。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from label_service import app as A  # noqa: E402


def test_samples_csv_roundtrip_through_loader():
    from infer_csv_scratch import load_csv

    t0 = 1_793_671_800_123
    req = A.InferSamplesRequest(t_ms=[t0, t0 + 32, t0 + 64], acc=[[0.1, 0.2, 1.0]] * 3, hz=31.25)
    path = A._write_samples_csv(req)
    try:
        acc, gyro, ts, valid, null_ratio = load_csv(path)
        assert acc.shape == (3, 3) and abs(acc[0][2] - 1.0) < 1e-6 and gyro.sum() == 0 and null_ratio == 0
        back = [A._utc_str_to_ms(x.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]) for x in ts]
        assert back == [t0, t0 + 32, t0 + 64]
    finally:
        os.remove(path)


def test_length_mismatch_rejected():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        A._write_samples_csv(A.InferSamplesRequest(t_ms=[1, 2], acc=[[0, 0, 1]], hz=31.25))
