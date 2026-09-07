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


def _memoize_load_csv() -> None:
    """
    同一个文件在一次推理里要读两遍 CSV：infer_file 自己读一遍算特征，
    _spectral_ratio_per_window 再读一遍算频谱。一小时 50Hz 的数据是 18 万行，
    pandas 解析一遍就大几百毫秒。这里给 load_csv 套一个只存最近一个文件的缓存，
    第二次直接命中；换文件就自动淘汰，不会把内存堆起来。
    """
    import infer_csv_scratch as m

    if getattr(m, "_lru_wrapped", False):
        return
    orig = m.load_csv
    cache: dict = {}

    def cached(path):
        if cache.get("path") != path:
            cache.clear()
            cache["path"], cache["val"] = path, orig(path)
        return cache["val"]

    m.load_csv = cached
    m._lru_wrapped = True


def _init_worker(model_path: str) -> None:
    sys.path.insert(0, os.path.join(config.REPO_ROOT, "src"))
    from label_service.logging_setup import worker_setup_logging
    from label_service.model_loader import load_model_bundle
    worker_setup_logging()
    _bundle.update(load_model_bundle(model_path))
    _bundle["model_path"] = model_path
    _memoize_load_csv()


def _spectral_ratio_per_window(full_path: str, windows: list[dict], window_s: float) -> dict | None:
    """
    给每个窗口算陀螺仪 4–8 Hz 能量占 1–15 Hz 的比例，写进 window["spec"]。抓挠是
    后腿高频往复，这个频段有明显峰；模型之外的独立证据，稳定版可按它过滤误报
    （config.STABLE_SPECTRAL_MIN）。算不出来（没陀螺仪/没时间戳）就全 None。
    同时返回 10 Hz 的陀螺仪能量包络 {t0, hz, energy}，给边界微调用（postprocess.refine_boundaries）。
    """
    envelope = None
    try:
        from infer_csv_scratch import load_csv
        _acc, gyro, ts, _mask, _null = load_csv(full_path)
        if gyro is None or ts is None or len(gyro) == 0:
            return None
        hz = config.DEVICE_HZ
        n_samp = int(window_s * hz)
        ts_vals = ts.values.astype("datetime64[ns]")
        mag = np.linalg.norm(gyro, axis=1).astype(np.float32)
        # 能量包络：去均值后的幅值平方按 0.1s 分箱求均值
        step = max(1, hz // 10)
        dm = mag - float(np.nanmean(mag))
        sq = np.nan_to_num(dm * dm)
        n_bins = len(sq) // step
        if n_bins > 0:
            env = sq[: n_bins * step].reshape(n_bins, step).mean(axis=1)
            envelope = {"t0": ts.iloc[0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "hz": hz / step,
                        "energy": [round(float(v), 4) for v in env]}
        freqs = np.fft.rfftfreq(n_samp, d=1.0 / hz)
        band = (freqs >= 4) & (freqs <= 8)
        wide = (freqs >= 1) & (freqs <= 15)
        hann = np.hanning(n_samp).astype(np.float32)

        # 所有窗口一次性堆成矩阵做批量 FFT——逐窗口在 Python 里循环几千次
        # rfft，光解释器开销就比 FFT 本身还贵
        starts, idx_of = [], []
        for k, w in enumerate(windows):
            w["spec"] = None
            t = w.get("ts")
            if not t:
                continue
            try:
                t0 = np.datetime64(t.replace(" ", "T"))
            except ValueError:
                continue
            i0 = int(np.searchsorted(ts_vals, t0))
            if len(mag) - i0 < n_samp // 2:
                continue
            starts.append(i0)
            idx_of.append(k)
        if starts:
            padded = np.zeros(len(mag) + n_samp, dtype=np.float32)
            padded[: len(mag)] = mag
            offsets = np.asarray(starts)[:, None] + np.arange(n_samp)[None, :]
            segs = padded[offsets]
            segs = segs - segs.mean(axis=1, keepdims=True)
            pw = np.abs(np.fft.rfft(segs * hann, axis=1)) ** 2
            tot = pw[:, wide].sum(axis=1)
            ratio = np.divide(pw[:, band].sum(axis=1), tot, out=np.zeros_like(tot), where=tot > 0)
            for k, r, tt in zip(idx_of, ratio, tot):
                windows[k]["spec"] = round(float(r), 3) if tt > 0 else None
    except Exception:  # noqa: BLE001 频谱只是附加信息，算不出来不影响推理
        for w in windows:
            w.setdefault("spec", None)
    return envelope


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
    envelope = _spectral_ratio_per_window(full_path, windows, b["window_s"]) if windows else None
    return {"segments": segments, "windows": windows, "n_windows": n_windows, "envelope": envelope}


def create_pool(model_path: str) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=config.INFER_WORKERS,
        initializer=_init_worker,
        initargs=(model_path,),
    )
