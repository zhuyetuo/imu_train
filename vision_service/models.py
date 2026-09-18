"""本地模型的集中管理：列状态、加载 / 卸载、一键测试、调用计数。

算法机上的模型越来越多（狗检测 YOLO、SAM、SigLIP 向量、姿态 RTMPose……），平台「模型服务」
页从这里拿一张表，每个模型：在不在、跑在哪、权重在哪、出了什么错、调了多少次、平均多久。

调用计数在进程内存里（重启归零）：每个模型的推理入口调一下 record()，记次数 / 帧数 / 耗时。
"""

from __future__ import annotations

import time

from . import config, dog, embed, pose, sam
from . import meter as _meter

_started = time.time()


def meter(key: str) -> dict:
    return _meter.get(key)


def record(key: str, ms: float, frames: int = 1, ok: bool = True) -> None:
    _meter.record(key, ms, frames, ok)


def reset_meter() -> None:
    _meter.reset()


def _cuda_free() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ── 每个模型：轻量状态（不触发加载）/ 加载 / 卸载 / 测试 ──────────────

def _dog_status() -> dict:
    return {"available": dog._model is not None, "error": dog._load_error, "device": dog._device_used,
            "weights": config.DOG_WEIGHTS, "warm": dog._warm}


def _dog_unload() -> None:
    with dog._lock:
        dog._model = None
        dog._warm = False
        dog._load_error = None
    _cuda_free()


def _dog_test() -> dict:
    import numpy as np

    dog._load(force=True)
    if dog._model is None:
        raise RuntimeError(dog._load_error or "没加载")
    img = np.random.default_rng(1).integers(0, 256, size=(640, 640, 3), dtype="uint8")
    t0 = time.monotonic()
    boxes = dog.detect(img, 0.1)
    return {"latency_ms": int((time.monotonic() - t0) * 1000), "detail": f"随机图跑一次，框到 {len(boxes)} 个（随机图有没有框都正常）"}


def _sam_status() -> dict:
    return {"available": sam._model is not None, "error": sam._load_error, "device": sam._device_used,
            "weights": config.SAM_CHECKPOINT, "warm": sam._warm}


def _sam_unload() -> None:
    with sam._lock:
        sam._model = None
        sam._warm = False
        sam._load_error = None
    _cuda_free()


def _sam_test() -> dict:
    t0 = time.monotonic()
    r = sam.warmup()
    if not r.get("warm"):
        raise RuntimeError(r.get("error") or "预热失败")
    return {"latency_ms": int((time.monotonic() - t0) * 1000), "detail": "随机图点一下出了掩码"}


def _embed_status() -> dict:
    return {"available": embed._model is not None, "error": embed._load_error, "device": embed._device,
            "weights": config.EMBED_MODEL, "loading": embed._loading, "progress": embed.download_progress()}


def _embed_unload() -> None:
    with embed._lock:
        embed._model = None
        embed._processor = None
        embed._load_error = None
    _cuda_free()


def _embed_test() -> dict:
    import io

    import numpy as np

    embed._load(force=True)
    if embed._model is None:
        raise RuntimeError(embed._load_error or "没加载")
    from PIL import Image

    img = Image.fromarray(np.random.default_rng(2).integers(0, 256, size=(224, 224, 3), dtype="uint8"))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    t0 = time.monotonic()
    v = embed._default_encoder.encode_images([buf.getvalue()])
    tv = embed._default_encoder.encode_text(["a dog"])
    return {"latency_ms": int((time.monotonic() - t0) * 1000),
            "detail": f"图向量 {v.shape[1]} 维、文本向量 {tv.shape[1]} 维，各算了一次"}


def _pose_status() -> dict:
    st = pose.status()
    err = st["error"]
    if st["available"] and st["device"] == "cpu" and (config.POSE_DEVICE or "").startswith("cuda"):
        err = "在 CPU 上跑：onnxruntime 没有 CUDA provider（重跑 ./up.sh deploy 会装 onnxruntime-gpu）"
    return {"available": st["available"], "error": err, "device": st["device"], "weights": config.POSE_ONNX}


def _pose_unload() -> None:
    with pose._lock:
        pose._model = None
        pose._loaded = False
        pose._error = None


def _pose_test() -> dict:
    import numpy as np

    if not pose.available():
        raise RuntimeError(pose.status()["error"] or "没加载")
    img = np.random.default_rng(3).integers(0, 256, size=(480, 640, 3), dtype="uint8")
    t0 = time.monotonic()
    r = pose.keypoints(img, [{"bbox": [0.2, 0.2, 0.6, 0.6], "conf": 0.9}])
    n = int((r[1] >= pose.MIN_KP_SCORE).sum()) if r is not None else 0
    return {"latency_ms": int((time.monotonic() - t0) * 1000), "detail": f"随机图跑一次，{n}/17 个点过阈值（随机图接近 0 正常）"}


REGISTRY: dict[str, dict] = {
    "dog": {"name": "狗检测（YOLO）", "purpose": "画面里有没有狗、狗在哪；建索引 / 找片段 / 找相似都先用它框狗",
            "status": _dog_status, "load": lambda: dog.warmup(), "unload": _dog_unload, "test": _dog_test},
    "sam": {"name": "SAM 分割", "purpose": "视觉标注页点一下出掩码",
            "status": _sam_status, "load": lambda: sam.warmup(), "unload": _sam_unload, "test": _sam_test},
    "embed": {"name": "画面向量（SigLIP）", "purpose": "画面向量索引：以图搜图 / 一句话搜",
              "status": _embed_status, "load": lambda: embed.warmup(), "unload": _embed_unload, "test": _embed_test},
    "pose": {"name": "姿态关键点（RTMPose AP-10K）", "purpose": "以图搜图的第二路信号：鼻子够到了哪只爪",
             "status": _pose_status, "load": lambda: {"warm": pose.available(), "error": pose.status()["error"]},
             "unload": _pose_unload, "test": _pose_test},
}


def list_models() -> list[dict]:
    out = []
    for key, spec in REGISTRY.items():
        st = spec["status"]()
        out.append({"key": key, "name": spec["name"], "purpose": spec["purpose"], **st, "meter": meter(key)})
    return out


def act(key: str, action: str) -> dict:
    spec = REGISTRY.get(key)
    if spec is None:
        raise KeyError(key)
    if action == "load":
        r = spec["load"]() or {}
        st = spec["status"]()
        return {"ok": bool(st.get("available")), "error": st.get("error") or r.get("error"), "status": st}
    if action == "unload":
        spec["unload"]()
        return {"ok": True, "error": None, "status": spec["status"]()}
    if action == "test":
        t0 = time.monotonic()
        try:
            with _meter.paused():      # 调试测试不进统计
                r = spec["test"]()
            return {"ok": True, "error": None, "latency_ms": r.get("latency_ms", int((time.monotonic() - t0) * 1000)),
                    "detail": r.get("detail"), "status": spec["status"]()}
        except Exception as e:  # noqa: BLE001 测试就是要把错误原样带回给人看
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "latency_ms": int((time.monotonic() - t0) * 1000),
                    "detail": None, "status": spec["status"]()}
    raise ValueError(action)


def overview() -> dict:
    gpu = None
    try:
        gpu = dog.cuda_report()
    except Exception:  # noqa: BLE001
        pass
    return {"started_at": _started, "uptime_s": int(time.time() - _started), "gpu": gpu, "models": list_models()}


__all__ = ["record", "meter", "reset_meter", "list_models", "act", "overview", "REGISTRY"]
