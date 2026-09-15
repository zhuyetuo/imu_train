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
import time

import numpy as np

from . import config

_model = None
_load_error: str | None = None
# 单卡上并发跑 SAM 只会买到显存峰值翻倍和碎片化，本来就串行，索性排队。
# 加载也走这把锁：不然两个并发的首请求会各自加载一份模型，显存直接翻倍。
_lock = threading.RLock()
# 上次尝试加载的时间。失败原因不能永久缓存——最常见的失败是"权重还没下完"，
# 下好之后不该还要重启进程才能用。
_last_try = 0.0
_RETRY_AFTER_S = 60.0
# 预热过没有：加载完 + 空跑过一次推理。跟 available 分开报——
# available 说的是"模型在不在"，warm 说的是"第一刀还要不要等十几秒"
_warm = False
_warm_seconds: float | None = None
#: 实际加载到哪个设备上。跟 config.SAM_DEVICE 分开——后者只是想要哪个
_device_used: str | None = None


def _load(force: bool = False):
    """真正加载模型。装不上/没权重都不抛到调用方，记下原因让 /status 去说。"""
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
        _load_locked()


def _load_locked():
    global _model, _load_error
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
        globals()["_device_used"] = device
    except Exception as e:  # noqa: BLE001 - 加载失败的原因五花八门，全都要能报出来
        _load_error = f"加载失败：{type(e).__name__}: {e}"


def status() -> dict:
    # 权重文件出现了就立刻重试一次，不等退避——运维刚把权重拷进去，
    # 下一件事一定是刷这个接口看好没好
    _load(force=os.path.isfile(config.SAM_CHECKPOINT) and _model is None)
    return {
        "available": _model is not None,
        "checkpoint": config.SAM_CHECKPOINT,
        # device 报**实际**用的那个，不是配置值。配置写 cuda、实际退回 cpu 的时候，
        # 报配置值等于让状态接口撒谎——而"慢十几倍"这件事没有任何别的提示
        "device": _device_used or config.SAM_DEVICE,
        "cuda": cuda_report(),
        "error": _load_error,
        # 平台据此区分「模型坏了」和「还在预热」：前者置灰按钮并显示原因，
        # 后者显示「模型加载中」。以前只有 available，这两种情况长得一模一样
        "warm": _warm,
        "warm_seconds": _warm_seconds,
    }


def warmup() -> dict:
    """加载权重 + 空跑一次推理，把首次调用的开销挪到启动时。

    首刀慢是两笔钱叠在一起，光 _load() 只付掉第一笔：
      1. build_sam2 + 权重搬上显存——几秒
      2. **第一次前向**：CUDA context、kernel 编译、cudnn autotune——同样几秒
    所以这里必须真跑一次 predict，不能只加载完就算数。

    空跑用随机噪声而不是全零图：全零的话某些算子会走到退化分支，预热不到
    真实路径，第一刀照样慢。

    不抛异常：预热失败就是没预热，服务照常起，退回懒加载那条路。
    """
    global _warm, _warm_seconds
    t0 = time.monotonic()
    _load(force=True)
    if _model is None:
        return {"warm": False, "error": _load_error}
    try:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8)
        with _lock:
            _model.set_image(np.asarray(img))
            _model.predict(
                point_coords=np.array([[512.0, 512.0]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
    except Exception as e:  # noqa: BLE001 预热失败不该把服务带倒
        return {"warm": False, "error": f"预热推理失败：{type(e).__name__}: {e}"}
    _warm = True
    _warm_seconds = round(time.monotonic() - t0, 1)
    return {"warm": True, "error": None, "warm_seconds": _warm_seconds}


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


def pick_mask(shapes: list[dict], scores: list[float], has_box: bool, prefer: str = "auto") -> int:
    """三个候选里挑哪一个。

    ── 为什么不能用 argmax(score) ────────────────────────────────────────

    2026-09-15 拿真实牙齿照片实测，四张里三张是错的，而且错得很自信：

        点在嘴下方的毛 → 切出**整个狗头**，score 0.97
        点在脸颊       → 切出前景一个物件，score 0.96
        点在嘴角       → 切出整个口鼻部（鼻子+嘴唇+牙一起），score 0.44
        点正好在牙上   → 一颗牙，score 0.84

    score 是"SAM 有多确信这是**一个物体**"，不是"这是不是你要的那个"。切整个
    狗头它当然确信——那本来就是个完整、边界清楚的物体。而 argmax(score) 恰好
    就是在挑"最完整的那个物体"，对切单颗牙来说方向是反的。

    SAM 的三个掩膜大致是 子部件 / 部件 / 整体 三档。标牙齿永远要最细那一档。

    ── 挑法 ──────────────────────────────────────────────────────────────

    auto（默认）：
      给了框  → 用 argmax(score)。框本身已经把歧义消掉了，这时候 score 是可信的。
      只给点  → 挑**面积最小**的那个非退化掩膜。点提示的歧义永远是
                "这颗牙 / 这排牙 / 整个嘴 / 整张脸"，而要的永远是最小那个。
    score：老行为（argmax）。留着是为了能一键对比，以及万一别的场景要用。
    """
    if prefer == "score" or (prefer == "auto" and has_box):
        return int(max(range(len(scores)), key=lambda i: scores[i]))
    # 面积为 0 的（mask_to_shapes 返回 None）不参与挑选
    usable = [i for i, sh in enumerate(shapes) if sh and sh["area_ratio"] > 0]
    if not usable:
        return int(max(range(len(scores)), key=lambda i: scores[i]))
    return min(usable, key=lambda i: shapes[i]["area_ratio"])


def segment(image_path: str, points: list[dict], box: list[float] | None = None,
            prefer: str = "auto") -> dict:
    """按提示分割。

    points：[{x, y, label}]，x/y 是**归一化**的 0-1（前端拿到的图是缩放过的，
    传像素坐标就得两边都知道原图尺寸，迟早错一次）；label 1=正点 0=负点。
    box：可选的框提示，同样归一化。
    prefer：三个候选怎么挑，见 pick_mask。
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

    # 三个都转出来再挑。原来是先 argmax 再转，于是另外两个候选**根本看不到**——
    # 而实测发现对的那个经常就在没被选中的里面
    cand = [mask_to_shapes(m) for m in masks]
    best = pick_mask(cand, [float(x) for x in scores], has_box=bx is not None, prefer=prefer)
    shapes = cand[best]
    if shapes is None:
        raise ValueError("没分割出东西来，换个位置再点一下")
    return {
        **shapes,
        "score": float(scores[best]),
        "chosen": best,
        # 把三个候选的大小和分数都带出来：挑得对不对，只有把没被选中的那两个
        # 也摆出来才判得了。前端也可以据此给个"换一个"的按钮
        "candidates": [
            {"area_ratio": (c["area_ratio"] if c else 0.0), "score": float(sc)}
            for c, sc in zip(cand, scores)
        ],
        "width": w, "height": h,
    }


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
