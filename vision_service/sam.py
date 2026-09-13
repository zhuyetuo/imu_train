"""
SAM 2.1 交互式分割：人在图上点一下，出一个掩膜 → 转成框和多边形还给标注平台。

为什么是 SAM 2.1 而不是 SAM 3：2/2.1 的代码和权重是 Apache-2.0，可商用无 copyleft；
SAM 3 走 Meta 自定的许可，衍生分发要沿用同一许可。而 SAM 3 的增量是"文本概念提示"
——"第 108 号牙"没有任何语义能跟邻牙区分，对这个场景零增益。

这个服务只做标注加速，永远不进生产推理：SAM 是交互式的、必须有提示、单张几十到
几百毫秒，做不了无人值守的批量。批量是检测模型的事。

模型是懒加载的：没装 sam2 或者没有权重时，/status 如实说不可用，/segment 返回 503，
平台那边把按钮置灰，其它功能一概不受影响。
"""

import os
import threading

import numpy as np

from . import config

_model = None
_load_error: str | None = None
# 单卡上并发跑 SAM 只会买到显存峰值翻倍和碎片化，本来就串行，索性排队
_lock = threading.Lock()


def _load():
    """真正加载模型。装不上/没权重都不抛到调用方，记下原因让 /status 去说。"""
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        _load_error = f"没装 sam2 或 torch：{e}"
        return
    if not os.path.isfile(config.SAM_CHECKPOINT):
        _load_error = f"权重不存在：{config.SAM_CHECKPOINT}"
        return
    try:
        from sam2.build_sam import build_sam2

        device = config.SAM_DEVICE
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model = build_sam2(config.SAM_MODEL_CFG, config.SAM_CHECKPOINT, device=device)
        _model = SAM2ImagePredictor(model)
    except Exception as e:  # noqa: BLE001 - 加载失败的原因五花八门，全都要能报出来
        _load_error = f"加载失败：{type(e).__name__}: {e}"


def status() -> dict:
    _load()
    return {
        "available": _model is not None,
        "checkpoint": config.SAM_CHECKPOINT,
        "device": config.SAM_DEVICE,
        "error": _load_error,
    }


def mask_to_shapes(mask: np.ndarray) -> dict | None:
    """掩膜 → 归一化的框 + 多边形。

    框是必给的（训练走检测口径）；多边形是顺手的（掩膜质心比框心准，牙齿倾斜时
    框心明显偏，几何推号直接吃这个精度；将来要升分割也不用重标）。

    只取面积最大的那一块连通域：SAM 偶尔会在反光处additionally 多吐一小片，
    那一小片跟着进去就是脏数据。
    """
    m = np.asarray(mask).astype(np.uint8)
    if m.ndim != 2 or not m.any():
        return None
    h, w = m.shape

    ys, xs = np.where(m > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = [x0 / w, y0 / h, (x1 - x0 + 1) / w, (y1 - y0 + 1) / h]

    polygon = None
    try:
        import cv2

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            # 抽稀到几十个点：原始轮廓动辄上千点，存库和传输都没必要，
            # 而牙齿这种凸形目标抽稀之后形状几乎不变
            eps = 0.004 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            if len(approx) >= 3:
                polygon = [[float(px) / w, float(py) / h] for px, py in approx]
    except ImportError:
        pass

    return {
        "bbox": [round(v, 6) for v in bbox],
        "polygon": [[round(px, 6), round(py, 6)] for px, py in polygon] if polygon else None,
        "area_ratio": round(float(m.sum()) / (w * h), 6),
    }


def segment(image_path: str, points: list[dict], box: list[float] | None = None) -> dict:
    """按提示分割。

    points：[{x, y, label}]，x/y 是**归一化**的 0-1（前端拿到的图是缩放过的，
    传像素坐标就得两边都知道原图尺寸，迟早错一次）；label 1=正点 0=负点。
    box：可选的框提示，同样归一化。
    """
    _load()
    if _model is None:
        raise RuntimeError(_load_error or "模型没加载")

    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"读不出这张图：{image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    pt = np.array([[p["x"] * w, p["y"] * h] for p in points], dtype=np.float32) if points else None
    lb = np.array([int(p.get("label", 1)) for p in points], dtype=np.int32) if points else None
    bx = np.array([box[0] * w, box[1] * h, (box[0] + box[2]) * w, (box[1] + box[3]) * h], dtype=np.float32) if box else None
    if pt is None and bx is None:
        raise ValueError("至少给一个点或一个框")

    with _lock:
        _model.set_image(img)
        masks, scores, _ = _model.predict(
            point_coords=pt, point_labels=lb, box=bx,
            multimask_output=True,  # 一个点是有歧义的（牙面/整颗牙/一排牙），让它出三个再挑
        )

    best = int(np.argmax(scores))
    shapes = mask_to_shapes(masks[best])
    if shapes is None:
        raise ValueError("没分割出东西来，换个位置再点一下")
    return {**shapes, "score": float(scores[best]), "width": w, "height": h}
