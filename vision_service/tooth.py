"""牙齿 / 口腔照片 YOLO 检测——从 label_service 挪到视觉服务，跟狗检测、SAM 这些放一起上 GPU。

接口跟 label_service 那边的 /api/v1/tooth/detect 一模一样（path 相对素材库根、conf、
with_image、top_k；返回 detections / class_names / 带框图），label_infra 只换了个地址。
参数也跟 tooth_health/code/web_app.py 一致：conf=0.5、imgsz=960、results[0].plot() 画框。

权重放在 models/vision/tooth/best.pt（TOOTH_WEIGHTS 可改）。第一次加载时这里还没有、
而老地方（tooth_health/data/runs/tooth_detect/weights/best.pt，label_service 用的那份）有，
就**复制**一份过来——老服务那份原样留着，不动它。
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import threading
import time

from . import config, gpumem

_logger = logging.getLogger("vision.tooth")

_model = None
_lock = threading.RLock()
_load_error: str | None = None
_device_used: str | None = None
_copied_from: str | None = None

WEIGHTS = os.environ.get("TOOTH_WEIGHTS") or os.path.join(config.HERE, "..", "models", "vision", "tooth", "best.pt")
OLD_WEIGHTS = os.path.join(config.REPO_ROOT, "tooth_health", "data", "runs", "tooth_detect", "weights", "best.pt")
CONF = float(os.environ.get("TOOTH_CONF", "0.5"))
IMGSZ = int(os.environ.get("TOOTH_IMGSZ", "960"))


def _ensure_weights() -> str:
    """权重不在就从老地方复制过来（不移动、不改老的）"""
    global _copied_from
    p = os.path.abspath(WEIGHTS)
    if os.path.isfile(p):
        return p
    if os.path.isfile(OLD_WEIGHTS):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copy2(OLD_WEIGHTS, p)
        _copied_from = OLD_WEIGHTS
        _logger.info("牙齿模型权重从 %s 复制到 %s", OLD_WEIGHTS, p)
        return p
    raise FileNotFoundError(f"牙齿模型权重不存在：{p}（老地方 {OLD_WEIGHTS} 也没有；"
                            f"把训练好的 best.pt 放到 models/vision/tooth/，或设 TOOTH_WEIGHTS）")


def _pick_device() -> str:
    if config.SAM_DEVICE != "cuda":
        return config.SAM_DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def load():
    global _model, _load_error, _device_used
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        try:
            path = _ensure_weights()
            from ultralytics import YOLO

            m = YOLO(path)
            _device_used = _pick_device()
            try:
                with gpumem.track("牙齿 yolo", _device_used):
                    m.to(_device_used)
            except Exception as e:  # noqa: BLE001
                _logger.warning("牙齿模型搬到 %s 失败，留在 CPU：%s", _device_used, e)
                _device_used = "cpu"
            _model = m
            _load_error = None
            return _model
        except Exception as e:  # noqa: BLE001
            _load_error = f"{type(e).__name__}: {e}"
            raise


def unload() -> None:
    global _model, _device_used
    with _lock:
        _model = None
        _device_used = None


def class_names() -> list[str]:
    names = load().names
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def detect(full_path: str, conf: float | None = None, imgsz: int | None = None, with_image: bool = True,
           top_k: int | None = 1) -> dict:
    """top_k=1（默认）：每张图只留置信度最高的一个框（一张口腔照片就是一张嘴）；
    agnostic_nms 让同一位置不会同时冒出「正常」和「异常」两个框。top_k=0 不限。"""
    import cv2

    from . import meter

    model = load()
    t0 = time.monotonic()
    kwargs = dict(conf=conf or CONF, imgsz=imgsz or IMGSZ, verbose=False, agnostic_nms=True, device=_device_used)
    if top_k:
        kwargs["max_det"] = top_k
    try:
        r = model.predict(full_path, **kwargs)[0]
    except Exception:
        meter.record("tooth", (time.monotonic() - t0) * 1000, 1, ok=False)
        raise
    meter.record("tooth", (time.monotonic() - t0) * 1000, 1)
    names = model.names
    dets = []
    if r.boxes is not None and len(r.boxes):
        for cls_id, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            dets.append({"class_id": int(cls_id), "class_name": names[int(cls_id)], "confidence": round(float(cf), 4),
                         "box": [round(v, 1) for v in xyxy]})
    h, w = r.orig_shape
    out = {"width": int(w), "height": int(h), "detections": dets, "class_names": class_names(),
           "conf": conf or CONF, "imgsz": imgsz or IMGSZ, "top_k": top_k or 0, "annotated_jpeg_b64": None}
    if with_image:
        ok, buf = cv2.imencode(".jpg", r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out["annotated_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return out


def status() -> dict:
    p = os.path.abspath(WEIGHTS)
    have = os.path.isfile(p) or os.path.isfile(OLD_WEIGHTS)
    return {
        "available": _model is not None or (have and _load_error is None),
        "loaded": _model is not None, "warm": _model is not None,
        "weights": p, "device": _device_used, "copied_from": _copied_from,
        "error": _load_error if _load_error else (None if have else f"权重不存在：{p}"),
        "material_root": config.MATERIAL_ROOT, "conf": CONF, "imgsz": IMGSZ,
    }
