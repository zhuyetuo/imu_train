"""本地模型的集中管理：列状态、加载 / 卸载、一键测试、调用计数。

算法机上的模型越来越多（狗检测 YOLO、SAM、SigLIP 向量、姿态 RTMPose……），平台「模型服务」
页从这里拿一张表，每个模型：在不在、跑在哪、权重在哪、出了什么错、调了多少次、平均多久。

调用计数在进程内存里（重启归零）：每个模型的推理入口调一下 record()，记次数 / 帧数 / 耗时。
"""

from __future__ import annotations

import time

from . import config, dog, embed, lowlight, pose, sam, segmask, vllm_manager
from . import meter as _meter

_started = time.time()
# 每个模型最近一次「测试」的结果：{at, ok, latency_ms, detail, error}（进程内存）
_last_test: dict[str, dict] = {}


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


def _vllm_status() -> dict:
    st = vllm_manager.status()
    if st["ready"]:
        err = None
    elif st["running"]:
        sp = st.get("startup_progress") or {}
        err = f"模型加载中：{sp.get('stage', '')}（{sp.get('pct', 0)}%，已用 {sp.get('elapsed_s', 0)} 秒；7B 一般一两分钟）"
    elif st["downloading"]:
        pr = st.get("download_progress") or {}
        if pr.get("total_mb"):
            eta = f"，预计还要 {pr['eta_s'] // 60} 分 {pr['eta_s'] % 60} 秒" if pr.get("eta_s") is not None else ""
            err = (f"正在下权重：{pr['pct']}%（{pr['done_mb'] / 1000:.2f} / {pr['total_mb'] / 1000:.2f} GB，"
                   f"{pr['speed_mbps']} MB/s{eta}）")
        else:
            err = f"正在下权重：已下 {pr.get('done_mb', 0) / 1000:.2f} GB，{pr.get('speed_mbps', 0)} MB/s；" + "；".join(st["download_log"][-1:])
    elif st.get("pulling"):
        err = f"正在拉镜像 {st.get('image')}（十几 GB）：" + "；".join(st.get("pull_log") or [])
    elif st.get("exited"):
        errs = st.get("log_errors") or []
        err = f"vllm {'容器' if st.get('backend') == 'docker' else '进程'}退出了（code {st.get('exit_code')}）：" + (errs[0] if errs else "看日志") + "。点「日志」看全部"
    else:
        if st.get("backend") == "docker":
            not_installed = ("docker 不可用（没装或 daemon 没起）" if not st.get("docker_available") else
                             (st.get("pull_error") or f"镜像 {st.get('image')} 还没拉（点启动会拉；或 ./up.sh deploy 时拉）"))
        else:
            not_installed = "没装 vllm（VLLM_BACKEND=process 时重跑 ./up.sh deploy -g 会自动装）"
        err = st["error"] or st["download_error"] or (
            not_installed if not st["installed"] else
            f"权重还没齐（缺 {', '.join(st.get('missing_files') or [])}；点启动会接着下）" if not st["weights_ready"] else "没启动")
    return {"available": bool(st["ready"]), "error": err, "device": "cuda" if st["running"] else None,
            "weights": st["local_dir"], "warm": st["ready"], "loading": bool(st["running"] and not st["ready"]) or st["downloading"],
            "progress": st.get("download_progress"),
            "startup": st.get("startup_progress"),
            "vllm": {k: st[k] for k in ("installed", "model", "weights_ready", "downloading", "running", "pid", "port",
                                        "port_open", "ready", "uptime_s", "log_tail", "download_log", "download_error",
                                        "exited", "exit_code", "log_errors", "backend", "image", "pulling", "pull_log")}}


def _vllm_load() -> dict:
    r = vllm_manager.start()
    return {"warm": r.get("ok"), "error": r.get("error")}


def _vllm_unload() -> None:
    r = vllm_manager.stop()
    if not r.get("ok"):
        raise RuntimeError(r.get("error") or "停不掉")


def _seg_status() -> dict:
    st = segmask.status()
    err = st["error"]
    if not st["enabled"]:
        err = "EMBED_MASK_BG=0 关着：建索引 / 找相似不抠狗"
    return {"available": st["available"], "error": err, "device": st["device"], "weights": config.SEG_WEIGHTS,
            "loading": st["loading"]}


def _seg_load() -> dict:
    segmask._load(force=True)
    return {"warm": segmask._model is not None, "error": segmask._load_error}


def _seg_unload() -> None:
    with segmask._lock:
        segmask._model = None
        segmask._load_error = None
    _cuda_free()


def _seg_test() -> dict:
    import numpy as np

    segmask._load(force=True)
    if segmask._model is None:
        raise RuntimeError(segmask._load_error or "没加载")
    img = np.random.default_rng(4).integers(0, 256, size=(640, 640, 3), dtype="uint8")
    t0 = time.monotonic()
    m = segmask.dog_mask(img, None)
    return {"latency_ms": int((time.monotonic() - t0) * 1000),
            "detail": "随机图跑一次，" + ("抠到了一块" if m is not None else "没抠到东西（随机图没狗，正常）")}


def _lowlight_status() -> dict:
    """夜视增强模型（Retinexformer）。**没配也不算坏**——按片段的「夜视」照样
    能给"只拉伸"和"多帧堆栈"两张，那两张还不编造像素。所以这里报的是
    "配没配"，而不是"能不能用"。
    """
    import os

    w, repo = config.LOWLIGHT_WEIGHTS, config.LOWLIGHT_REPO
    missing = []
    if not w:
        missing.append("LOWLIGHT_WEIGHTS（.pth 的绝对路径）")
    elif not os.path.isfile(w):
        missing.append(f"权重文件不在：{w}")
    if not repo:
        missing.append("LOWLIGHT_REPO（git clone 的 Retinexformer 目录）")
    elif not os.path.isdir(repo):
        missing.append(f"仓库目录不在：{repo}")
    loaded = bool(lowlight._model_cache)
    return {
        "available": loaded,
        "device": "cuda" if loaded else None,
        "weights": w or "（没配）",
        "error": ("没配：" + "；".join(missing) +
                  "。不配也能用——「夜视」里的「只拉伸」和「多帧堆栈」照常出，"
                  "而且那两张不编造像素") if missing else (None if loaded else "已配好，第一次用时加载"),
    }


def _lowlight_load() -> dict:
    lowlight._load_model(config.LOWLIGHT_WEIGHTS, config.LOWLIGHT_REPO)
    return {"warm": True}


def _lowlight_unload() -> None:
    lowlight._model_cache.clear()
    _cuda_free()


def _lowlight_test() -> dict:
    import numpy as np

    # 拿一张"跟夜间素材一样暗"的图跑：亮度挤在 16~38 这 22 级里
    img = np.random.default_rng(7).integers(16, 39, size=(256, 256, 3), dtype="uint8")
    t0 = time.monotonic()
    out = lowlight.run_model(img)
    return {"latency_ms": int((time.monotonic() - t0) * 1000),
            "detail": f"256x256 暗图跑一次，输出均值 {float(out.mean()):.0f}/255"}


REGISTRY: dict[str, dict] = {
    "dog": {"name": "狗检测（YOLO）", "purpose": "画面里有没有狗、狗在哪；建索引 / 找片段 / 找相似都先用它框狗",
            "status": _dog_status, "load": lambda: dog.warmup(), "unload": _dog_unload, "test": _dog_test},
    "sam": {"name": "SAM 分割", "purpose": "视觉标注页点一下出掩码",
            "status": _sam_status, "load": lambda: sam.warmup(), "unload": _sam_unload, "test": _sam_test},
    "embed": {"name": "画面向量（SigLIP）", "purpose": "画面向量索引：以图搜图 / 一句话搜",
              "status": _embed_status, "load": lambda: embed.warmup(), "unload": _embed_unload, "test": _embed_test},
    "seg": {"name": "抠狗分割（YOLO-seg）", "purpose": "建索引 / 找相似算向量前把狗抠出来、背景涂灰，向量里只剩狗；没加载就退回按框裁（有背景噪音）",
            "status": _seg_status, "load": _seg_load, "unload": _seg_unload, "test": _seg_test},
    "pose": {"name": "姿态关键点（RTMPose AP-10K）", "purpose": "以图搜图的第二路信号：鼻子够到了哪只爪",
             "status": _pose_status, "load": lambda: {"warm": pose.available(), "error": pose.status()["error"]},
             "unload": _pose_unload, "test": _pose_test},
    "vllm": {"name": "本地大模型（vLLM）", "purpose": "「画面找片段」的本地视觉大模型，OpenAI 兼容口，docker 跑；「大模型 API」页里「本地服务」那一行连的就是它",
             "status": _vllm_status, "load": _vllm_load, "unload": _vllm_unload, "test": vllm_manager.test},
    "lowlight": {"name": "夜视增强（Retinexformer）",
                 "purpose": "夜里那几路黑得看不出狗在干嘛时，按片段增强。**可选**：不配也能用——「夜视」里的「只拉伸」和「多帧堆栈」照常出，而且那两张不编造像素；模型这张好看但不能当证据",
                 "status": _lowlight_status, "load": _lowlight_load,
                 "unload": _lowlight_unload, "test": _lowlight_test},
}


def list_models() -> list[dict]:
    out = []
    for key, spec in REGISTRY.items():
        st = spec["status"]()
        out.append({"key": key, "name": spec["name"], "purpose": spec["purpose"], **st, "meter": meter(key),
                    "last_test": _last_test.get(key)})
    return out


def act(key: str, action: str) -> dict:
    spec = REGISTRY.get(key)
    if spec is None:
        raise KeyError(key)
    # 加载/卸载抛出来的异常**要接住并原样带回**，跟「测试」一个待遇。
    # 原来只有测试接了，加载没接：一抛就是个光秃秃的 500，界面上只显示
    # 「视觉服务返回 500」，而真正有用的那句（权重跟网络结构对不上、缺依赖、
    # 文件不在）全丢了——人只能猜，或者去翻服务日志。
    if action == "load":
        try:
            r = spec["load"]() or {}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}",
                    "status": spec["status"]()}
        st = spec["status"]()
        # vLLM 是"拉起来了但还在加载"：进程起来就算 ok，ready 由状态刷新去反映
        ok = bool(st.get("available")) or (key == "vllm" and bool(r.get("warm")))
        return {"ok": ok, "error": (r.get("error") or st.get("error")) if not ok else None, "status": st}
    if action == "unload":
        try:
            spec["unload"]()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}",
                    "status": spec["status"]()}
        return {"ok": True, "error": None, "status": spec["status"]()}
    if action == "test":
        t0 = time.monotonic()
        try:
            with _meter.paused():      # 调试测试不进统计
                r = spec["test"]()
            out = {"ok": True, "error": None, "latency_ms": r.get("latency_ms", int((time.monotonic() - t0) * 1000)),
                   "detail": r.get("detail")}
        except Exception as e:  # noqa: BLE001 测试就是要把错误原样带回给人看
            out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "latency_ms": int((time.monotonic() - t0) * 1000),
                   "detail": None}
        _last_test[key] = {"at": time.time(), **out}
        return {**out, "status": spec["status"]()}
    raise ValueError(action)


def overview() -> dict:
    gpu = None
    try:
        gpu = dog.cuda_report()
    except Exception:  # noqa: BLE001
        pass
    return {"started_at": _started, "uptime_s": int(time.time() - _started), "gpu": gpu, "models": list_models()}


__all__ = ["record", "meter", "reset_meter", "list_models", "act", "overview", "REGISTRY"]
