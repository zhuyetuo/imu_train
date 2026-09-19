"""把狗从画面里抠出来（实例分割），背景涂成中性灰，再去算画面向量。

为什么：狗只占画面一角，裁出来的那一块里一大半是花砖地、门框、笼子。SigLIP 的向量
里这些"共同背景"占的分量比狗的姿态还大——去均值能压一部分，但压不干净：
狗挪到另一块地砖上，背景就变了，分数跟着乱。把背景涂掉，向量里剩下的就只有狗。

用 YOLO 的分割版权重（跟检测同一家，COCO 类别），不用 SAM：SAM 一帧几百毫秒，
建索引一路视频几千帧扛不住；YOLO-seg 一帧几十毫秒，能跟检测一起走。

模型不在 / 加载失败 → 退回不抠（跟以前一样），status 里报出来。
"""

from __future__ import annotations

import logging
import os
import threading
import time

from . import config

_logger = logging.getLogger("vision_service.segmask")

_model = None
_load_error: str | None = None
_lock = threading.RLock()
_last_try = 0.0
_RETRY_AFTER_S = 60.0
_device_used: str | None = None
_classes: list[int] = []

#: 背景涂成什么颜色（BGR）。114 灰是 YOLO 系列 letterbox 的填充色，模型对它最"无感"
BG_COLOR = (114, 114, 114)


def _load(force: bool = False):
    global _model, _load_error, _last_try, _device_used, _classes
    if _model is not None or not config.EMBED_MASK_BG:
        return
    if _load_error is not None and not force and (time.monotonic() - _last_try) < _RETRY_AFTER_S:
        return
    with _lock:
        if _model is not None:
            return
        _last_try = time.monotonic()
        _load_error = None
        try:
            from ultralytics import YOLO
        except ImportError as e:
            _load_error = f"没装 ultralytics：{e}"
            return
        try:
            p = config.SEG_WEIGHTS
            os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
            m = YOLO(p)
            from . import dog

            _classes = dog._resolve_classes(m)
            _device_used = dog._pick_device()
            try:
                m.to(_device_used)
            except Exception as e:  # noqa: BLE001
                _logger.warning("把分割模型搬到 %s 失败，留在 CPU 上：%s", _device_used, e)
                _device_used = "cpu"
            _model = m
        except Exception as e:  # noqa: BLE001
            _load_error = (f"加载失败：{type(e).__name__}: {e}（权重 {config.SEG_WEIGHTS}；"
                           f"这台机器下不动的话手动把 .pt 放过去，或 EMBED_MASK_BG=0 关掉抠图）")


def available() -> bool:
    _load()
    return _model is not None


def status() -> dict:
    _load()
    return {"enabled": bool(config.EMBED_MASK_BG), "available": _model is not None,
            "weights": config.SEG_WEIGHTS, "device": _device_used, "error": _load_error}


def dog_mask(frame, boxes: list[dict] | None = None, conf: float = 0.25):
    """整帧上狗的掩码（HxW uint8，1 = 狗）。几只狗都算进去；boxes 给了就只要跟检测框
    有重叠的实例（分割模型偶尔会把沙发也当熊）。没模型 / 一个实例都没有 → None。"""
    import numpy as np

    if not available():
        return None
    from . import meter

    h, w = frame.shape[:2]
    with _lock, meter.timed("seg"):
        res = _model.predict(frame, verbose=False, conf=conf, imgsz=config.DETECT_IMGSZ,
                             classes=list(_classes) or None, agnostic_nms=True, retina_masks=True,
                             device=_device_used or "cpu")
    mask = np.zeros((h, w), dtype="uint8")
    found = False
    for r in res:
        m = getattr(r, "masks", None)
        if m is None or getattr(m, "data", None) is None:
            continue
        data = m.data.cpu().numpy() if hasattr(m.data, "cpu") else np.asarray(m.data)
        xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.asarray(r.boxes.xyxy)
        for inst, bb in zip(data, xyxy):
            if boxes and not _overlaps_any(bb, boxes, w, h):
                continue
            inst = np.asarray(inst)
            if inst.shape != (h, w):
                import cv2
                inst = cv2.resize(inst.astype("float32"), (w, h), interpolation=cv2.INTER_LINEAR)
            mask |= (inst > 0.5).astype("uint8")
            found = True
    return mask if found else None


def _overlaps_any(bb, boxes: list[dict], w: int, h: int) -> bool:
    x1, y1, x2, y2 = (float(v) for v in bb)
    for b in boxes:
        bx, by, bw, bh = b["bbox"]
        ix = max(0.0, min(x2, (bx + bw) * w) - max(x1, bx * w))
        iy = max(0.0, min(y2, (by + bh) * h) - max(y1, by * h))
        small = min((x2 - x1) * (y2 - y1), bw * w * bh * h)
        if small > 0 and ix * iy / small >= 0.3:
            return True
    return False


def apply(frame, mask, feather_px: int = 3):
    """背景涂灰。边缘留一圈羽化，免得抠出来的狗带着锯齿边（向量对硬边很敏感）。"""
    import cv2
    import numpy as np

    m = mask.astype("float32")
    if feather_px > 0:
        m = cv2.dilate(m, np.ones((feather_px * 2 + 1,) * 2, dtype="uint8"))
        m = cv2.GaussianBlur(m, (feather_px * 2 + 1,) * 2, 0)
    m = m[..., None]
    bg = np.empty_like(frame)
    bg[:] = BG_COLOR
    return (frame.astype("float32") * m + bg.astype("float32") * (1 - m)).astype("uint8")


def masked_crop(frame, boxes: list[dict], crop_fn, max_side: int = 512):
    """建索引 / 查询共用：抠狗 → 按检测框裁 → 缩到 max_side。抠不到就返回 None（调用方用原图）。"""
    import cv2

    mask = dog_mask(frame, boxes)
    if mask is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = crop_fn(boxes, w, h)
    img = apply(frame[y1:y2, x1:x2], mask[y1:y2, x1:x2])
    if not img.size:
        return None
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    return img
