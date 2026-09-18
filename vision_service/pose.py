"""狗的姿态关键点（第二路检索信号）。

以图搜图只靠 SigLIP 向量分不清"头贴前左爪"和"头贴前右爪"——它看的是整体长相。
关键点直接量"鼻子到哪只爪多近"，正好对应部位标签。做法：

  帧 → 狗框（dog.detect） → RTMPose（AP-10K 17 个四足动物关键点） → 一个几十维的姿态向量
  建索引时每帧存一份；搜索时 分数 = (1-w)·画面相似 + w·姿态相似，w 可调（默认 0.5）。

模型：RTMPose-m AP-10K，ONNX 一个文件，rtmlib + onnxruntime 跑（不用 mmpose 那一大套）。
没装 rtmlib / 没权重时这一路自动关：索引里不存姿态，搜索只用画面，status 里如实说。

AP-10K 关键点顺序：
  0 左眼 1 右眼 2 鼻子 3 脖子 4 尾根
  5 左肩 6 左肘 7 左前爪   8 右肩 9 右肘 10 右前爪
  11 左髋 12 左膝 13 左后爪  14 右髋 15 右膝 16 右后爪
"""

from __future__ import annotations

import logging
import os
import threading

from . import config

_logger = logging.getLogger("vision_service.pose")

K = 17
NOSE, NECK, TAIL = 2, 3, 4
PAWS = (7, 10, 13, 16)          # 左前 右前 左后 右后
# 姿态向量 = 17 个点的归一化坐标(34) + 鼻子到四爪/尾根/脖子的距离(6) + 关键点可见位(17)
DIM = K * 2 + 6 + K
MIN_KP_SCORE = 0.3

_lock = threading.Lock()
_model = None
_error: str | None = None
_loaded = False


def onnx_path() -> str | None:
    p = config.POSE_ONNX
    return p if p and os.path.isfile(p) else None


def _load():
    global _model, _error, _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        _loaded = True
        p = onnx_path()
        if not p:
            _error = "没有姿态模型权重（POSE_ONNX 没指向一个存在的 .onnx；./vision_service/get_pose_weights.sh 会下，下不到看它打印的办法）"
            return
        try:
            from rtmlib import RTMPose
        except ImportError:
            _error = "没装 rtmlib（pip install rtmlib onnxruntime-gpu）"
            return
        try:
            dev = "cuda" if (config.POSE_DEVICE or "").startswith("cuda") else "cpu"
            _model = RTMPose(onnx_model=p, model_input_size=(config.POSE_INPUT, config.POSE_INPUT),
                             backend="onnxruntime", device=dev)
            _logger.info("姿态模型已加载：%s（%s）", p, dev)
        except Exception as e:  # noqa: BLE001
            _error = f"姿态模型加载失败：{type(e).__name__}: {e}"


def status() -> dict:
    return {"available": _model is not None, "error": _error if _model is None else None,
            "onnx": config.POSE_ONNX, "dim": DIM, "weight": config.POSE_W}


def available() -> bool:
    _load()
    return _model is not None


def keypoints(frame, boxes: list[dict]):
    """帧 + 狗框（归一化 xywh）→ (kps (17,2) 像素, scores (17,))；用面积最大的那只。没狗返回 None。
    调用方保证 available()。"""
    import numpy as np

    if not boxes:
        return None
    h, w = frame.shape[:2]
    b = max(boxes, key=lambda x: x["bbox"][2] * x["bbox"][3])
    x, y, bw, bh = b["bbox"]
    xyxy = np.array([[x * w, y * h, (x + bw) * w, (y + bh) * h]], dtype="float32")
    from . import meter

    with _lock, meter.timed("pose"):
        kps, scores = _model(frame, bboxes=xyxy)
    kps = np.asarray(kps, dtype="float32")
    scores = np.asarray(scores, dtype="float32")
    if kps.ndim != 3 or kps.shape[0] == 0:
        return None
    return kps[0], scores[0], (x * w, y * h, (x + bw) * w, (y + bh) * h)


def descriptor(kps, scores, box_xyxy) -> "list[float]":
    """关键点 → 姿态向量（长度 DIM，L2 归一化）。纯函数，测试直接喂假点。

    坐标按狗框归一：减框中心、除框长边——狗在画面哪里、离镜头多远都不影响；
    距离按体长（脖子到尾根）归一，量的是"鼻子够到了哪个部位"；
    看不见的点坐标记 0、可见位记 0，距离缺一头就记 0。
    """
    import numpy as np

    kps = np.asarray(kps, dtype="float32").reshape(K, 2)
    sc = np.asarray(scores, dtype="float32").reshape(K)
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 1.0)
    vis = (sc >= MIN_KP_SCORE).astype("float32")
    norm = ((kps - np.array([cx, cy], dtype="float32")) / side) * vis[:, None]
    body = float(np.linalg.norm(kps[NECK] - kps[TAIL])) if vis[NECK] and vis[TAIL] else side * 0.6
    body = max(body, 1e-3)

    def dist(a: int, b: int) -> float:
        if not (vis[a] and vis[b]):
            return 0.0
        return float(np.linalg.norm(kps[a] - kps[b]) / body)

    dists = [dist(NOSE, p) for p in PAWS] + [dist(NOSE, TAIL), dist(NOSE, NECK)]
    v = np.concatenate([norm.reshape(-1), np.array(dists, dtype="float32"), vis]).astype("float32")
    n = float(np.linalg.norm(v))
    return (v / n).tolist() if n > 0 else v.tolist()


def frame_descriptor(frame, boxes: list[dict]):
    """一帧 → 姿态向量；不可用 / 没狗 / 没测到点返回 None。"""
    if not available():
        return None
    try:
        r = keypoints(frame, boxes)
    except Exception as e:  # noqa: BLE001 一帧算不出来别让整条索引白跑
        _logger.warning("姿态推理失败：%s", e)
        return None
    if r is None:
        return None
    kps, sc, box = r
    return descriptor(kps, sc, box)


__all__ = ["K", "DIM", "status", "available", "keypoints", "descriptor", "frame_descriptor", "onnx_path"]
