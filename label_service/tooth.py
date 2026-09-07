"""
牙齿/口腔照片 YOLO 检测——跟 tooth_health/code/web_app.py 的图片检测走同一套参数
（conf=0.5, imgsz=960，results[0].plot() 画框），只是把结果以 JSON + 带框图 base64
返回，给 label_infra 的「牙齿识别」页面用。

权重 best.pt 不在仓库里（tooth_health/data 整个 gitignore），服务启动时不存在就
把这一组接口标成不可用（/health 里 tooth.available=false），不影响 IMU 推理。
模型在主进程里懒加载一次（YOLO 对象不能跨进程 pickle，也不像特征提取那么吃 CPU）。
"""

import base64
import logging
import os
import threading

from label_service import config

log = logging.getLogger("label_service.tooth")

_model = None
_model_lock = threading.Lock()
_load_error: str | None = None


def weights_available() -> bool:
    return os.path.isfile(config.TOOTH_WEIGHTS)


def _get_model():
    global _model, _load_error
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not weights_available():
            _load_error = f"牙齿模型权重不存在: {config.TOOTH_WEIGHTS}（先跑 tooth_health/code/train_yolo26.py，或设 TOOTH_WEIGHTS）"
            raise FileNotFoundError(_load_error)
        try:
            from ultralytics import YOLO
        except ImportError as e:
            _load_error = "缺少 ultralytics（pip install ultralytics）"
            raise RuntimeError(_load_error) from e
        log.info("加载牙齿 YOLO 模型 %s", config.TOOTH_WEIGHTS)
        _model = YOLO(config.TOOTH_WEIGHTS)
        _load_error = None
        return _model


def class_names() -> list[str]:
    m = _get_model()
    names = m.names
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def resolve_image_path(relative_path: str) -> str:
    """path 可以相对 MATERIAL_ROOT（口腔验证/2026-09-02-ok/Bali/xxx.jpg）或相对 NAS_ROOT；
    不接受绝对路径和 ..，防止跳出这两个根目录。"""
    if os.path.isabs(relative_path) or ".." in relative_path.replace("\\", "/").split("/"):
        raise ValueError(f"非法路径（必须是 MATERIAL_ROOT 或 NAS_ROOT 下的相对路径）: {relative_path}")
    for root in (config.MATERIAL_ROOT, config.NAS_ROOT):
        full = os.path.join(root, relative_path)
        if os.path.isfile(full):
            return full
    raise FileNotFoundError(f"文件不存在（在 {config.MATERIAL_ROOT} 和 {config.NAS_ROOT} 下都没找到）: {relative_path}")


def detect(full_path: str, conf: float | None = None, imgsz: int | None = None, with_image: bool = True) -> dict:
    import cv2

    model = _get_model()
    results = model.predict(full_path, conf=conf or config.TOOTH_CONF, imgsz=imgsz or config.TOOTH_IMGSZ, verbose=False)
    r = results[0]
    names = model.names
    dets = []
    if r.boxes is not None and len(r.boxes):
        for cls_id, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            dets.append({
                "class_id": int(cls_id),
                "class_name": names[int(cls_id)],
                "confidence": round(float(cf), 4),
                "box": [round(v, 1) for v in xyxy],  # x1, y1, x2, y2 像素坐标（原图尺寸）
            })
    h, w = r.orig_shape
    out = {
        "width": int(w), "height": int(h),
        "detections": dets,
        "class_names": class_names(),
        "conf": conf or config.TOOTH_CONF, "imgsz": imgsz or config.TOOTH_IMGSZ,
        "annotated_jpeg_b64": None,
    }
    if with_image:
        ok, buf = cv2.imencode(".jpg", r.plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            out["annotated_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return out


def status() -> dict:
    return {
        "available": weights_available() and _load_error is None,
        "weights": config.TOOTH_WEIGHTS,
        "loaded": _model is not None,
        "error": _load_error if not weights_available() or _load_error else None,
        "material_root": config.MATERIAL_ROOT,
        "conf": config.TOOTH_CONF,
        "imgsz": config.TOOTH_IMGSZ,
    }
