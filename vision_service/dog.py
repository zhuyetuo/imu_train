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

import logging
import os
import threading
import time

from . import config

_logger = logging.getLogger("vision_service.dog")

# COCO 里 dog 的类别号。**按名字从模型自己的 names 表里查**，查不到才退回 16。
#
# 一开始是写死 16 的，理由是"这属于 COCO 数据集定义，不会变"。这个理由站不住：
# 16 是不是 dog 取决于**这份权重**的 names 表，不取决于 COCO 规范——换个家族、
# 换个自训权重、或者哪天官方调了顺序，16 就是别的东西了。而这种错不会报任何
# 错，只会让「有没有狗」整个失真：画面里明明有狗被判成没狗，人跳过一整段素材。
#
# 查 names 就没这个问题：权重里写的是什么就是什么。查不到 dog 这个名字时退回
# 16 并记一条日志——总比直接不干活强，但要让人知道是在猜。
_DOG_FALLBACK_CLASS = 16
_dog_class: int | None = None


def _resolve_dog_class(model) -> int:
    """从权重自己的 names 表里找 dog 的类别号。"""
    names = getattr(model, "names", None) or {}
    try:
        items = names.items() if hasattr(names, "items") else enumerate(names)
        for k, v in items:
            if str(v).strip().lower() == "dog":
                return int(k)
    except Exception:  # noqa: BLE001 names 的形状各版本不一样，查不动就退回
        pass
    _logger.warning(
        "这份权重（%s）的 names 表里找不到 dog，退回用类别号 %d。"
        "如果它不是 COCO 80 类的模型，检测结果会是错的类别。",
        config.DOG_WEIGHTS, _DOG_FALLBACK_CLASS,
    )
    return _DOG_FALLBACK_CLASS

_model = None
_load_error: str | None = None
# 一张卡上并发跑检测只会买到显存峰值翻倍。扫描本来就是后台任务，排队就行。
_lock = threading.RLock()
_last_try = 0.0
_RETRY_AFTER_S = 60.0
_warm = False
#: 实际跑在哪个设备上（从模型里读，不是配置值）
_device_used: str | None = None


def _weights_path() -> str:
    """权重文件：配置的路径不存在、但同名文件躺在仓库根 / vision_service 目录（老版本 ultralytics 下到
    当前目录的），挪到配置的位置；都没有就把目录建好，ultralytics 会按文件名下到那里。"""
    import shutil

    p = config.DOG_WEIGHTS
    if os.path.isfile(p):
        return p
    name = os.path.basename(p)
    here = os.path.dirname(os.path.abspath(__file__))
    for old in (os.path.join(here, name), os.path.join(here, "..", name), os.path.join(os.getcwd(), name)):
        if os.path.isfile(old) and os.path.abspath(old) != os.path.abspath(p):
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
            shutil.move(old, p)
            _logger.info("把检测权重从 %s 挪到 %s", old, p)
            return p
    os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
    return p


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
            _model = YOLO(_weights_path())
            globals()["_dog_class"] = _resolve_dog_class(_model)
            # **加载时就搬上卡**，不要靠 predict(device=...) 每次搬。
            #
            # ultralytics 的 YOLO(权重) 是加载到 CPU 的，predict(device="cuda")
            # 才把权重挪过去——而且是每次调用都挪一趟。SAM 那边是
            # build_sam2(..., device=device)，加载时就在卡上。两边不一样，
            # 表现出来就是"SAM 能用 CUDA、YOLO 不行"：
            #   - 读 next(model.parameters()).device 永远是 cpu（加载时的状态）
            #   - 每采一帧就搬一次几十 MB 的权重，白烧带宽
            globals()["_device_used"] = _pick_device()
            try:
                _model.to(_device_used)
            except Exception as e:  # noqa: BLE001 搬不过去就留在 CPU 上，但要说出来
                _logger.warning("把检测模型搬到 %s 失败，留在 CPU 上：%s", _device_used, e)
                globals()["_device_used"] = "cpu"
        except Exception as e:  # noqa: BLE001 权重不存在/下载失败/torch 版本不对，全都要能报出来
            # 把换法一起写进错误里：最常见的失败是这台机器下不了权重（离线/防火墙），
            # 而这种时候人需要知道的是"换成哪个、怎么换"，不是一句"加载失败"
            _load_error = (
                f"加载失败：{type(e).__name__}: {e}"
                f"（权重 {config.DOG_WEIGHTS}。名字解析不出来多半是 ultralytics 太旧——"
                f"先试 pip install -U ultralytics；这台机器下不动权重的话，把 .pt 手动放到 "
                f"vision_service/weights/ 并设 DOG_WEIGHTS 指过去）"
            )


def _pick_device() -> str:
    """想要 cuda 就先确认 cuda 真的能用，不能用就退回 cpu——**并且让 status 报出来**。

    跟 SAM 那边同一个判断。不做这一步的话，cuda 用不了时 ultralytics 会在每次
    predict 里抛错或自己退回，两种都不会有人看见。
    """
    want = config.SAM_DEVICE
    if want != "cuda":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def status() -> dict:
    _load()
    return {
        "available": _model is not None,
        # 把**实际加载的**那个也报出来：DOG_WEIGHTS 改了但服务没重启的话，
        # 只看配置值会以为换成功了
        "weights": config.DOG_WEIGHTS,
        "loaded_weights": getattr(getattr(_model, "ckpt_path", None), "__str__", lambda: None)()
        if _model is not None else None,
        # 实际设备从模型身上读，读不到才退回配置值——配置值只代表"想要哪个"
        "device": _device_used or config.SAM_DEVICE,
        "cuda": cuda_report(),
        "error": _load_error,
        "warm": _warm,
        # 报出来：万一退回了兜底值，人看 status 就该看得见，而不是等结果不对才查
        "dog_class": _dog_class,
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
            _model.predict(img, verbose=False, device=_device_used or "cpu")
    except Exception as e:  # noqa: BLE001 预热失败不该把服务带倒
        return {"warm": False, "error": f"预热推理失败：{type(e).__name__}: {e}"}
    _warm = True
    return {"warm": True, "error": None}


def detect_batch(frames: list, conf: float = 0.35) -> list[list[dict]]:
    """一批帧一起过模型（GPU 上一批 16~32 张比一张张送快好几倍）。返回每帧的框。"""
    if not frames:
        return []
    # 半精度：5090 上 x 模型快近一倍，框的差别在小数点后。CPU 上 half 不支持，自动不用
    half = bool(config.DETECT_HALF and (_device_used or "cpu") != "cpu")
    from . import meter

    with _lock, meter.timed("dog", frames=len(frames)):
        res = _model.predict(list(frames), verbose=False, conf=conf, half=half,
                             classes=[_dog_class if _dog_class is not None else _DOG_FALLBACK_CLASS],
                             device=_device_used or "cpu")
    out = []
    for frame, r in zip(frames, res):
        h, w = frame.shape[:2]
        boxes = []
        for b in getattr(r, "boxes", []):
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            boxes.append({
                "bbox": [round(x1 / w, 4), round(y1 / h, 4),
                         round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)],
                "conf": round(float(b.conf[0]), 3),
            })
        out.append(boxes)
    return out


def detect(frame, conf: float = 0.35) -> list[dict]:
    """一帧里的狗框，归一化 [x, y, w, h]。scan_video 和 seek（找片段）共用。

    调用方保证模型已加载（先 _load()）。锁在这里拿：一张卡上并发只会买到
    显存峰值翻倍。
    """
    from . import meter

    with _lock, meter.timed("dog"):
        res = _model.predict(frame, verbose=False, conf=conf,
                             classes=[_dog_class if _dog_class is not None else _DOG_FALLBACK_CLASS],
                             device=_device_used or "cpu")
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
    return boxes


def scan_video(path: str, every_sec: float = 5.0, conf: float = 0.35, max_frames: int = 1200,
               batch: int = 32) -> dict:
    """按时间采样跑狗检测。

    every_sec：多少秒看一眼。5 秒是够的——要回答的是"这段有没有狗"，不是
    "狗每一秒在哪"。一小时的视频就是 720 个采样点。
    max_frames：上限，防止有人传个 every_sec=0.01 把显卡占一下午。

    快在两处（一小时 720p 从十几秒到两三秒）：
      1. 解码走 ffmpeg（多线程，有卡时 NVDEC），不再 cv2 逐帧 grab 九万次；
         ffmpeg 不在 / 起不来就退回 cv2 那条老路，结果一样只是慢
      2. 检测按 batch 张一起送 GPU，不是一张张送
    """
    _load()
    if _model is None:
        raise RuntimeError(_load_error or "模型没加载")

    from . import seek

    frames: list[dict] = []
    pending: list[tuple[float, object]] = []

    def flush():
        if not pending:
            return
        results = detect_batch([f for _t, f in pending], conf)
        for (t, _f), boxes in zip(pending, results):
            frames.append({"t": round(t, 2), "n_dogs": len(boxes), "boxes": boxes})
        pending.clear()

    last_t = 0.0
    # 有 ffmpeg 走 ffmpeg（先 NVDEC 再软解），没有退回 cv2，见 seek.iter_frames
    it = seek.iter_frames(path, every_sec)
    for t, frame in it:
        if len(frames) + len(pending) >= max_frames:
            break
        last_t = t
        pending.append((t, frame))
        if len(pending) >= batch:
            flush()
    flush()
    # cv2 那条路给的 t 是 PTS，最后一帧的时间才是真实时长；ffmpeg 那条路是等间隔抽的，
    # 时长按最后一个采样点算，最多差半个间隔
    frames.sort(key=lambda f: f["t"])
    return {"duration_sec": round(last_t, 2), "every_sec": every_sec,
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


def cuda_report() -> dict:
    """CUDA 到底能不能用、为什么不能。

    这一块单独抽出来，是因为原来 /status 里的 `device` 报的是**配置值**
    （config.SAM_DEVICE），不是实际用的那个。而 SAM 加载时有一句
    "cuda 不可用就退回 cpu"——于是配置写着 cuda、实际跑在 CPU 上、
    status 还理直气壮地报 cuda。慢十几倍，一点提示都没有。

    状态接口宁可多报几个字段，也不能报一个"看起来对"的值。
    """
    out = {"wanted": config.SAM_DEVICE, "cuda_available": None,
           "torch": None, "cuda_build": None, "gpu": None, "why": None}
    try:
        import torch
    except ImportError as e:
        out["why"] = f"没装 torch：{e}"
        return out
    out["torch"] = torch.__version__
    # torch.version.cuda 是 None = 装的是 CPU 版的轮子。这是最常见的原因，
    # 而且从版本号上看不出来（2.x.y 和 2.x.y+cpu 有时都显示成 2.x.y）
    out["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001
        out["why"] = f"torch.cuda.is_available() 出错：{type(e).__name__}: {e}"
        return out
    if out["cuda_available"]:
        try:
            out["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
        return out
    if not out["cuda_build"]:
        out["why"] = ("装的是 CPU 版 torch（torch.version.cuda 是 None）。"
                      "按机器上的 CUDA 版本重装 GPU 版："
                      "pip install torch --index-url https://download.pytorch.org/whl/cu121 之类")
    else:
        out["why"] = (f"torch 是带 CUDA {out['cuda_build']} 的版本，但 torch.cuda.is_available() 是 False——"
                      "多半是驱动版本对不上、或者进程看不到 GPU（容器没给 --gpus）")
    return out
