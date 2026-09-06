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

from label_service import config

_bundle: dict = {}


def _init_worker(model_path: str) -> None:
    sys.path.insert(0, os.path.join(config.REPO_ROOT, "src"))
    from label_service.model_loader import load_model_bundle
    _bundle.update(load_model_bundle(model_path))
    _bundle["model_path"] = model_path


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
    return {"segments": segments, "windows": windows, "n_windows": n_windows}


def create_pool(model_path: str) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=config.INFER_WORKERS,
        initializer=_init_worker,
        initargs=(model_path,),
    )
