"""
推理进程池：跟 run_review_bins_all_days.sh 的 WORKERS=-1 一样按文件并行——
每个 worker 进程启动时各自加载一次模型（RF 模型跨进程 pickle 一次几十 MB，
不能每个请求都传），之后一个请求一个文件丢给空闲进程跑。RF 本身预测很快，
慢的是特征提取/重采样这些纯 CPU 的 Python 循环，单进程吃不满多核，进程池
才能把 14900K 这种 24 核跑起来。

worker 里跑的 _infer_one 跟单进程版一模一样：让 infer_file 在临时目录按它
自己的格式落 *_infer.json 再读回来，返回的就是命令行产出的同一份东西。
"""

import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from label_service import config

_bundle: dict = {}


def _init_worker(model_path: str) -> None:
    sys.path.insert(0, os.path.join(config.REPO_ROOT, "src"))
    from label_service.logging_setup import worker_setup_logging
    from label_service.model_loader import load_model_bundle
    worker_setup_logging()
    _bundle.update(load_model_bundle(model_path))
    _bundle["model_path"] = model_path


def _spectral_ratio_per_window(full_path: str, windows: list[dict], window_s: float) -> None:
    """
    给每个窗口算陀螺仪 4–8 Hz 能量占 1–15 Hz 的比例，写进 window["spec"]。抓挠是
    后腿高频往复，这个频段有明显峰；模型之外的独立证据，稳定版可按它过滤误报
    （config.STABLE_SPECTRAL_MIN）。算不出来（没陀螺仪/没时间戳）就全 None。
    """
    try:
        from infer_csv_scratch import load_csv
        _acc, gyro, ts, _mask, _null = load_csv(full_path)
        if gyro is None or ts is None or len(gyro) == 0:
            return
        hz = config.DEVICE_HZ
        n_samp = int(window_s * hz)
        ts_vals = ts.values.astype("datetime64[ns]")
        mag = np.linalg.norm(gyro, axis=1).astype(np.float32)
        freqs = np.fft.rfftfreq(n_samp, d=1.0 / hz)
        band = (freqs >= 4) & (freqs <= 8)
        wide = (freqs >= 1) & (freqs <= 15)
        hann = np.hanning(n_samp).astype(np.float32)
        for w in windows:
            w["spec"] = None
            t = w.get("ts")
            if not t:
                continue
            try:
                t0 = np.datetime64(t.replace(" ", "T"))
            except ValueError:
                continue
            i0 = int(np.searchsorted(ts_vals, t0))
            seg = mag[i0:i0 + n_samp]
            if len(seg) < n_samp // 2:
                continue
            seg = seg - seg.mean()
            if len(seg) < n_samp:
                seg = np.pad(seg, (0, n_samp - len(seg)))
            pw = np.abs(np.fft.rfft(seg * hann)) ** 2
            tot = float(pw[wide].sum())
            w["spec"] = round(float(pw[band].sum()) / tot, 3) if tot > 0 else None
    except Exception:  # noqa: BLE001 频谱只是附加信息，算不出来不影响推理
        for w in windows:
            w.setdefault("spec", None)


def _infer_one(full_path: str) -> dict:
    from infer_csv_scratch import infer_file  # worker 进程里 src/ 已在 sys.path
    b = _bundle
    model_hz = b["hz"]
    window_size = int(b["window_s"] * model_hz)
    stride = int(b["stride_s"] * model_hz)
    with tempfile.TemporaryDirectory(prefix="label_infer_") as tmp:
        infer_file(
            full_path, b["model"], b["classes"], window_size, stride,
            config.DEVICE_HZ, model_hz, b["gravity_aligned"],
            quiet=True, scratch_only=True, label_mode=b["label_mode"], output_dir=tmp,
            resample_method=config.RESAMPLE_METHOD,
            target_labels=config.TARGET_LABELS, is_dl=b["is_dl"],
        )
        stem = os.path.splitext(os.path.basename(full_path))[0]
        segments, windows, n_windows = {}, [], 0
        for label in config.TARGET_LABELS:
            p = os.path.join(tmp, label, "_infer", f"{stem}_infer.json")
            if not os.path.exists(p):
                segments[label] = []
                continue
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            segments[label] = data["scratch_segments"]
            windows, n_windows = data["windows"], data["n_windows"]  # 各类别文件里这两项相同
    if windows:
        _spectral_ratio_per_window(full_path, windows, b["window_s"])
    return {"segments": segments, "windows": windows, "n_windows": n_windows}


def create_pool(model_path: str) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=config.INFER_WORKERS,
        initializer=_init_worker,
        initargs=(model_path,),
    )
