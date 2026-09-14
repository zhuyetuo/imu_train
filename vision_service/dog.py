"""
画面里有没有狗、有几只。

这是"用画面去交叉验证 IMU"这条线的第一步，也是最便宜的一步：**不训练任何模型**，
用 COCO 预训练权重里现成的 `dog` 类就够了。要回答的问题只有一个——

    这段视频里到底有没有狗？

有了它，一整段没狗的片段在平台上可以直接标出来，人不用点进去看波形才发现这半
小时狗根本不在画面里。后面「是不是那只狗」「画面抓挠对不对得上 IMU」都建在
这个之上。

── 两个不能想当然的地方 ──────────────────────────────────────────────

**一、不能按帧号换算时间。** 我们的视频是 VFR（采集端用的是
`_FfmpegVfrSink`，每个 tick 写一帧，帧间隔不固定），`帧号 / fps` 算出来的时间
跟真实时间对不上，越往后差得越多。而这个结果是要按时间戳跟 IMU 片段对齐的，
差几秒就对到别的行为上去了。所以时间一律读 `CAP_PROP_POS_MSEC`（容器里的 PTS，
就是真实时刻）。

**二、不能用 seek。** `CAP_PROP_POS_FRAMES` 在 VFR 上是近似的，而且对长视频
每次 seek 都要回到最近的关键帧重解。改成顺序 `grab()`——grab 只取不解码，很便宜，
只在跨过采样点时才 `retrieve()` 真正解一帧。这跟采集端读摄像头是同一个路数
（见 imu_camera_sync_multicam.py 的 _reader_loop），那边也是因为解码才是贵的。

── 降级 ──────────────────────────────────────────────────────────────

跟 SAM 一样：没装 torch/ultralytics 或者没有权重，/status 如实说不可用，
扫描接口返回 503，平台那边把这一项当"还没扫"处理，其它功能一概不受影响。
"""

import os
import threading
import time

from . import config

# COCO 里 dog 的类别号。写死是有意的：ultralytics 的 names 表里 16 就是 dog，
# 这个映射属于 COCO 数据集定义，不会变。按名字找反而脆——不同权重的 names
# 可能是英文也可能被人改过。
COCO_DOG_CLASS = 16

_model = None
_load_error: str | None = None
# 一张卡上并发跑检测只会买到显存峰值翻倍。扫描本来就是后台任务，排队就行。
_lock = threading.RLock()
_last_try = 0.0
_RETRY_AFTER_S = 60.0
_warm = False


def _load(force: bool = False):
    global _model, _load_error, _last_try
    if _model is not None:
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
            _model = YOLO(config.DOG_WEIGHTS)
        except Exception as e:  # noqa: BLE001 权重不存在/下载失败/torch 版本不对，全都要能报出来
            _load_error = f"加载失败：{type(e).__name__}: {e}"


def status() -> dict:
    _load()
    return {
        "available": _model is not None,
        "weights": config.DOG_WEIGHTS,
        "device": config.SAM_DEVICE,
        "error": _load_error,
        "warm": _warm,
    }


def warmup() -> dict:
    """跟 SAM 那边同一个道理：首刀要付"加载 + 第一次前向"两笔钱，挪到启动时。"""
    global _warm
    _load(force=True)
    if _model is None:
        return {"warm": False, "error": _load_error}
    try:
        import numpy as np

        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(640, 640, 3), dtype="uint8")
        with _lock:
            _model.predict(img, verbose=False, device=config.SAM_DEVICE)
    except Exception as e:  # noqa: BLE001 预热失败不该把服务带倒
        return {"warm": False, "error": f"预热推理失败：{type(e).__name__}: {e}"}
    _warm = True
    return {"warm": True, "error": None}


def scan_video(path: str, every_sec: float = 5.0, conf: float = 0.35, max_frames: int = 1200) -> dict:
    """按时间采样跑狗检测。

    every_sec：多少秒看一眼。5 秒是够的——要回答的是"这段有没有狗"，不是
    "狗每一秒在哪"。一小时的视频就是 720 个采样点。
    max_frames：上限，防止有人传个 every_sec=0.01 把显卡占一下午。
    """
    _load()
    if _model is None:
        raise RuntimeError(_load_error or "模型没加载")

    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"打不开这个视频：{path}")
    try:
        frames = []
        next_t = 0.0
        last_ms = 0.0
        n_grabbed = 0
        while len(frames) < max_frames:
            # grab 只取不解码，便宜；只有跨过采样点时才 retrieve
            if not cap.grab():
                break
            n_grabbed += 1
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            # 有些容器最后几帧读不出 PTS（返回 0）。别把它当成"回到了 0 秒"——
            # 那会让采样点整个乱掉。读不出就沿用上一次的时间，跳过这一帧
            if ms <= 0 and n_grabbed > 1:
                continue
            last_ms = ms
            if ms / 1000.0 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            next_t = ms / 1000.0 + every_sec
            with _lock:
                res = _model.predict(frame, verbose=False, conf=conf, classes=[COCO_DOG_CLASS],
                                     device=config.SAM_DEVICE)
            h, w = frame.shape[:2]
            boxes = []
            for r in res:
                for b in getattr(r, "boxes", []):
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                    boxes.append({
                        "bbox": [round(x1 / w, 4), round(y1 / h, 4),
                                 round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)],
                        "conf": round(float(b.conf[0]), 3),
                    })
            frames.append({"t": round(ms / 1000.0, 2), "n_dogs": len(boxes), "boxes": boxes})
    finally:
        cap.release()

    return {"duration_sec": round(last_ms / 1000.0, 2), "every_sec": every_sec,
            "conf": conf, **summarize(frames), "frames": frames}


def summarize(frames: list[dict]) -> dict:
    """采样结果 → 一句能放进列表里的话。

    抽出来单独一个纯函数，是因为这几个数直接决定平台上"这段要不要人看"，
    算错了不会报错、只会让人白看或者漏看一整段。
    """
    n = len(frames)
    if not n:
        # 一帧都没采到（视频是坏的 / 长度为 0）。**不能**返回 no_dog_ratio=1，
        # 那等于告诉人"这段确认没狗"——实际是"没看成"，两者要分得开
        return {"sampled": 0, "frames_with_dog": 0, "no_dog_ratio": None,
                "max_dogs": 0, "verdict": "unknown"}
    with_dog = sum(1 for f in frames if f["n_dogs"] > 0)
    ratio = (n - with_dog) / n
    max_dogs = max(f["n_dogs"] for f in frames)
    if with_dog == 0:
        verdict = "no_dog"          # 整段没看见狗：这段可以直接不用看
    elif ratio >= 0.8:
        verdict = "mostly_empty"    # 绝大部分时间画面里没狗
    else:
        verdict = "has_dog"
    return {"sampled": n, "frames_with_dog": with_dog,
            "no_dog_ratio": round(ratio, 3), "max_dogs": max_dogs, "verdict": verdict}
