"""
vision_service：给标注平台用的 GPU 视觉服务。目前只有 SAM 2.1 交互式分割。

跟现有服务的关系：
  - algo_service 是线上服务，一行不碰；
  - label_service（IMU 推理，纯 CPU、进程池）也不碰，只是端口错开（它 8383，这里 8385）；
  - 平台调不通这个服务时，SAM 按钮置灰，其它功能一概不受影响——所以这里挂了
    不是事故，是降级。

路径安全：请求里只传相对 MATERIAL_ROOT 的路径，realpath 之后必须仍在它之内。
"""

import contextlib
import logging
import os
import threading

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, gpumem, dog, embed, llm as llmmod, lowlight, models, sam, seek

_logger = logging.getLogger("vision_service")


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warmup_on_startup()
    yield
    # 视觉服务退出时把自己拉起来的 vLLM 一起停掉，别留孤儿占显存
    from . import vllm_manager

    vllm_manager.shutdown_on_exit()


app = FastAPI(title="vision_service", version="0.1.0", lifespan=_lifespan)


def _warmup_on_startup():
    """启动就把模型加载好、空跑一次，别让第一个标注员替所有人等。

    首刀慢是两笔钱叠在一起：权重搬上显存几秒，**第一次前向**（CUDA context、
    kernel 编译、cudnn autotune）又是几秒。懒加载的话这十几秒结结实实落在
    第一次点「SAM 辅助」的人头上，而且界面上只是个转圈，看不出在干什么，
    多半会被当成卡死了再点几下。

    放后台线程，不阻塞启动：
      - 端口立刻就能连上，/status 立刻能答「还在预热」而不是超时；
      - 预热要是卡住了（比如权重在 NAS 上、网又慢），服务不会跟着起不来。
    线程里走的是 sam 模块那把 RLock，预热没完时进来的请求会排在它后面，
    不会加载出第二份模型。
    """
    if not config.WARMUP:
        _logger.info("VISION_WARMUP=0，跳过预热，第一次调用时才加载模型")
        return

    def run():
        r = sam.warmup()
        if r.get("warm"):
            _logger.info("SAM 预热完成，耗时 %.1fs，之后每刀都是热的", r.get("warm_seconds") or 0.0)
        else:
            _logger.warning("SAM 预热没成：%s（不影响启动，退回第一次调用时加载）", r.get("error"))
        # 狗检测单独预热一次。串行不并行：两个模型同时往一张卡上搬，显存峰值叠加，
        # 8G 的卡上很容易就 OOM——而预热 OOM 会让两个都用不了，比慢几秒糟得多
        d = dog.warmup()
        if d.get("warm"):
            _logger.info("画面狗检测预热完成")
        else:
            _logger.warning("画面狗检测预热没成：%s（不影响启动和 SAM）", d.get("error"))
        e = embed.warmup()
        # 姿态关键点也一起预热（有权重才会加载；没有就记一句）
        from . import pose

        if pose.available():
            _logger.info("姿态模型已加载（%s）", pose.status().get("device"))
        else:
            _logger.info("姿态模型没加载：%s", pose.status().get("error"))
        from . import segmask

        segmask.available()          # 触发加载（第一次下权重）；status 本身不加载
        ms = segmask.status()
        if ms["available"]:
            _logger.info("分割模型已加载（%s），建索引 / 查询会先把狗抠出来", ms.get("device"))
        elif ms["enabled"]:
            _logger.warning("分割模型没加载，索引不抠狗：%s", ms.get("error"))
        if e.get("warm"):
            _logger.info("画面向量模型预热完成")
        else:
            _logger.warning("画面向量模型预热没成：%s（不影响别的功能）", e.get("error"))

    threading.Thread(target=run, name="vision-warmup", daemon=True).start()


def _resolve_under(root_dir: str, rel_path: str) -> str:
    """相对某个根的路径 → 绝对路径，realpath 必须仍在那个根之内。

    跟 label_service/tooth.py 一个思路：只认相对路径，`..` 穿越在这里被挡住。
    照片和视频是两棵不同的树（素材库 vs 采集 NAS），所以根要能传进来——
    但**只能是配置里那两个**，不接受调用方随便给一个根，不然沙箱就没了。
    """
    root = os.path.realpath(root_dir)
    full = os.path.realpath(os.path.join(root, rel_path))
    if full != root and not full.startswith(root + os.sep):
        raise HTTPException(422, "非法路径")
    if not os.path.isfile(full):
        raise HTTPException(422, f"文件不存在: {rel_path}")
    return full


def _resolve(rel_path: str) -> str:
    return _resolve_under(config.MATERIAL_ROOT, rel_path)


class Point(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="归一化横坐标")
    y: float = Field(..., ge=0.0, le=1.0, description="归一化纵坐标")
    label: int = Field(1, description="1=正点（要这块），0=负点（不要这块）")


class SegmentIn(BaseModel):
    path: str = Field(..., description="相对 MATERIAL_ROOT 的路径，比如 口腔验证/2026-09-01-ok/巴利/a.jpg")
    points: list[Point] = Field(default_factory=list, max_length=32)
    box: list[float] | None = Field(None, min_length=4, max_length=4, description="可选框提示 [x,y,w,h] 归一化")
    # auto：给了框用 score 挑，只给点挑最小的那个（点提示的歧义永远是
    # "这颗牙/这排牙/整个嘴"，要的永远是最小那个）。score = 老行为，留着能对比
    prefer: str = Field("auto", pattern="^(auto|score)$")
    # gingiva：牙龈专用修整——SAM 的掩膜里只留粉红色那部分（去掉连带的牙和嘴唇），多边形抽稀更细
    refine: str | None = Field(None, pattern="^(gingiva)$")
    # 这张图上已经标好的别的东西的轮廓（归一化 [[x,y],...]，框给四个角）：从结果里挖掉。标牙龈时传牙
    exclude: list[list[list[float]]] = Field(default_factory=list, max_length=200)


@app.get("/health")
def health():
    return {"ok": True, "sam": sam.status()}


@app.get("/api/v1/sam/status")
def sam_status():
    return sam.status()


@app.post("/api/v1/sam/segment")
def sam_segment(body: SegmentIn):
    full = _resolve(body.path)
    st = sam.status()
    if not st["available"]:
        # 503 而不是 500：平台据此把按钮置灰并提示原因，而不是弹一个红叉
        raise HTTPException(503, st["error"] or "SAM 模型不可用")
    try:
        return sam.segment(full, [p.model_dump() for p in body.points], body.box, prefer=body.prefer, refine=body.refine,
                           exclude=body.exclude or None)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"分割失败: {type(e).__name__}: {e}") from e


class DogScanIn(BaseModel):
    path: str = Field(..., description="相对 VIDEO_ROOT 的视频路径")
    every_sec: float = Field(5.0, ge=0.2, le=60.0, description="多少秒看一眼")
    conf: float = Field(0.35, ge=0.05, le=0.95)


@app.get("/api/v1/dog/status")
def dog_status():
    return dog.status()


@app.post("/api/v1/dog/scan")
def dog_scan(body: DogScanIn):
    """这段视频里有没有狗、有几只。

    一小时的视频按 5 秒采样是 720 个点，跑完几十秒——所以调用方要当成后台任务，
    别挂在一个用户点击上等。
    """
    full = _resolve_under(config.VIDEO_ROOT, body.path)
    st = dog.status()
    if not st["available"]:
        # 503 而不是 500：平台据此当成"还没扫"，不是"扫出来没狗"。这两个的
        # 后果完全相反——后者会让人直接跳过一整段真有狗的素材
        raise HTTPException(503, st["error"] or "画面狗检测不可用")
    try:
        return dog.scan_video(full, every_sec=body.every_sec, conf=body.conf)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"扫描失败: {type(e).__name__}: {e}") from e


class SeekLabelIn(BaseModel):
    name: str = Field(..., max_length=50)
    description: str = Field("", max_length=300)
    # 部位上限 60：平台上了解剖学层级标签之后，「啃」一类的子孙能到 37 条（区域 → 部位 → 左右）。
    # 原来卡 12，平台一送 24 条整批 422，一个任务都跑不了。送多少由平台决定（它有粒度选项），
    # 这里只兜一个不会让 prompt 长到离谱的上限
    parts: list[str] = Field(default_factory=list, max_length=60)


class LlmIn(BaseModel):
    """用哪家、哪个模型、key。平台「大模型 API」页存的，请求时带过来；不带就退回环境变量。"""
    provider: str = Field(..., pattern="^(anthropic|openai|doubao|gemini|local)$")
    model: str = Field(..., max_length=100)
    api_key: str = Field("", max_length=500)
    base_url: str | None = Field(None, max_length=300)
    price_in: float = Field(0.0, ge=0)
    price_out: float = Field(0.0, ge=0)


class SeekIn(BaseModel):
    path: str = Field(..., description="相对 VIDEO_ROOT 的视频路径")
    llm: LlmIn | None = Field(None, description="不带 = 用环境变量里的 Claude key")
    labels: list[SeekLabelIn] = Field(..., min_length=1, max_length=24)
    every_sec: float = Field(1.0, ge=0.5, le=5.0)
    clip_s: float = Field(6.0, ge=2.0, le=20.0)
    stride_s: float = Field(3.0, ge=1.0, le=20.0)
    n_frames: int = Field(6, ge=2, le=12)
    max_clips: int = Field(120, ge=1, le=2000, description="一个视频最多送多少段去问模型（控花费）")
    min_dog_frac: float = Field(0.8, ge=0.0, le=1.0)
    motion_min: float = Field(0.02, ge=0.0, le=1.0)
    motion_max: float = Field(1.0, ge=0.0, le=1.0)
    min_conf: float = Field(0.5, ge=0.0, le=1.0)
    start_s: float = Field(0.0, ge=0.0)
    end_s: float | None = Field(None, ge=0.0)
    conf: float = Field(0.35, ge=0.05, le=0.95, description="狗检测阈值")
    dry_run: bool = Field(False, description="只做本地筛选、不调 API，看会送多少段")


@app.get("/api/v1/seek/status")
def seek_status():
    return seek.status()


@app.post("/api/v1/seek")
def seek_run(body: SeekIn):
    """用画面找片段：本地筛出"有狗且在动"的几秒窗，抽帧问视觉大模型，返回像的片段。

    一小时视频：本地筛选一两分钟（狗检测每秒一帧），然后按 max_clips 送去问，
    每段一两秒、几段并行——调用方要当后台任务。dry_run 不花钱，先看送多少段。
    """
    full = _resolve_under(config.VIDEO_ROOT, body.path)
    if not dog.status()["available"]:
        raise HTTPException(503, dog.status()["error"] or "画面狗检测不可用")
    llm = None
    if body.llm is not None:
        try:
            llm = llmmod.from_dict(body.llm.model_dump())
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if not body.dry_run and not llm.api_key and llm.provider != "local":
            raise HTTPException(422, f"{llm.provider} 没配 API key")
    elif not body.dry_run:
        st = seek.status()
        if not st["available"]:
            raise HTTPException(503, st["error"] or "视觉大模型不可用")
    labels = [seek.Label(name=l.name, description=l.description, parts=l.parts) for l in body.labels]
    try:
        return seek.seek_video(
            full, labels, every_sec=body.every_sec, clip_s=body.clip_s, stride_s=body.stride_s,
            n_frames=body.n_frames, max_clips=body.max_clips, min_dog_frac=body.min_dog_frac,
            motion_min=body.motion_min, motion_max=body.motion_max, min_conf=body.min_conf,
            start_s=body.start_s, end_s=body.end_s, dry_run=body.dry_run, conf=body.conf, llm=llm,
            rel_path=body.path,
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"找片段失败: {type(e).__name__}: {e}") from e


# ── 画面向量索引 ──────────────────────────────────────────────────────

class EmbedBuildIn(BaseModel):
    path: str = Field(..., description="相对 VIDEO_ROOT 的视频路径")
    every_sec: float = Field(1.0, ge=0.5, le=5.0)
    force: bool = False
    conf: float = Field(0.35, ge=0.05, le=0.95)
    # fine = 每秒一帧（慢、全）；fast = 只解关键帧（快、稀，这批素材约 12 秒一帧）。
    # 已经有精档的路再要快档会原样返回，不降级
    mode: str | None = Field(None, pattern="^(fine|fast)$")


class EmbedIndexedIn(BaseModel):
    paths: list[str] = Field(..., max_length=5000)


class EmbedRef(BaseModel):
    path: str
    t: float = Field(..., ge=0)


class EmbedSearchIn(BaseModel):
    """text 和 ref 二选一：一句英文，或"某视频第几秒那一帧"。"""
    text: str | None = Field(None, max_length=300)
    ref: EmbedRef | None = None
    paths: list[str] = Field(..., min_length=1, max_length=5000, description="在哪些视频的索引里找")
    top_k: int = Field(50, ge=1, le=2000)
    min_score: float = Field(0.0, ge=-1, le=1)
    gap_s: float = Field(3.0, ge=0, le=60)
    exclude_self_s: float = Field(10.0, ge=0, le=600, description="以图搜图时把样例前后这么多秒排掉")
    center: bool = Field(True, description="减掉所有帧的平均向量再比（去掉同狗同房同地板的共同背景）")
    pose_w: float | None = Field(None, ge=0, le=1, description="姿态相似占多少（0 只看画面，1 只看姿态）；不传用 POSE_W")
    # 部位是几何硬条件（鼻子够到哪只爪），不是相似度。SigLIP 分不清左前爪和右前爪——
    # 它看整体长相；这一条按关键点距离直接卡。认不出的部位名不筛，结果里 part_used 会说
    part: str | None = Field(None, max_length=40, description="只要鼻子够到这个部位的帧，比如 后爪 / 后右爪 / 尾根")
    part_near_max: float | None = Field(None, gt=0, le=3, description="多近算够到（体长倍数）；不传用 POSE_PART_NEAR_MAX")


def _embed_ready():
    st = embed.status()
    if not st["available"]:
        raise HTTPException(503, st["error"] or "画面向量模型不可用")
    if not dog.status()["available"]:
        raise HTTPException(503, dog.status()["error"] or "画面狗检测不可用")


@app.get("/api/v1/embed/status")
def embed_status():
    return embed.status()


@app.post("/api/v1/embed/build")
def embed_build(body: EmbedBuildIn):
    """给一路视频建索引。一小时视频：狗检测每秒一帧一两分钟，编码几十秒。调用方当后台任务。"""
    full = _resolve_under(config.VIDEO_ROOT, body.path)
    _embed_ready()
    try:
        return embed.build(body.path, full, every_sec=body.every_sec, force=body.force,
                           conf=body.conf, mode=body.mode)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"建索引失败: {type(e).__name__}: {e}") from e


@app.post("/api/v1/embed/indexed")
def embed_indexed(body: EmbedIndexedIn):
    """这些视频哪些已经有索引。"""
    return {p: embed.has_index(p) for p in body.paths}


class EmbedPreviewIn(BaseModel):
    path: str
    t: float = Field(..., ge=0)
    conf: float = Field(0.35, ge=0.05, le=0.95)


@app.post("/api/v1/embed/preview")
def embed_preview(body: EmbedPreviewIn):
    """以图搜图之前给人看一眼：这一帧框到了哪几只狗、拿哪一块去搜。只要狗检测模型，不要索引。"""
    full = _resolve_under(config.VIDEO_ROOT, body.path)
    st = dog.status()
    if not st.get("available"):
        raise HTTPException(503, st.get("error") or "狗检测模型不可用")
    try:
        return embed.frame_preview(full, body.t, conf=body.conf)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"取帧失败: {type(e).__name__}: {e}") from e


@app.get("/api/v1/embed/thumb")
def embed_thumb(path: str, t: float, crop: bool = True, max_side: int = 320, view: str | None = None):
    """某视频某一秒的缩略图。view = mask（抠掉背景的那块，拿去比的就是它）/ raw（那块原图）/
    pose（那块画上关键点骨架）/ box（整帧带检测框）；不给 view 时按 crop 老规矩。给平台"先看命中"用。"""
    from fastapi.responses import Response

    full = _resolve_under(config.VIDEO_ROOT, path)
    st = dog.status()
    if not st.get("available"):
        raise HTTPException(503, st.get("error") or "狗检测模型不可用")
    try:
        if view is not None and view not in ("mask", "raw", "pose", "box"):
            raise HTTPException(422, "view 只能是 mask / raw / pose / box")
        data = embed.frame_thumb(full, max(0.0, t), crop=crop, max_side=max(64, min(1920, max_side)), view=view)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"取帧失败: {type(e).__name__}: {e}") from e
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.post("/api/v1/embed/search")
def embed_search(body: EmbedSearchIn):
    if (body.text is None) == (body.ref is None):
        raise HTTPException(422, "text 和 ref 要且只要给一个")
    _embed_ready()
    try:
        if body.ref is not None:
            full = _resolve_under(config.VIDEO_ROOT, body.ref.path)
            q = embed.frame_query(full, body.ref.t)
            exclude = (body.ref.path, body.ref.t - body.exclude_self_s, body.ref.t + body.exclude_self_s)
            r = embed.search(q["vec"], body.paths, top_k=body.top_k, min_score=body.min_score,
                             gap_s=body.gap_s, exclude=exclude, center=body.center,
                             pose_vec=q.get("pose"), pose_w=body.pose_w,
                             part=body.part, part_near_max=body.part_near_max)
            r["query"] = {"kind": "frame", "has_dog": q["has_dog"], "t": q["t"], "has_pose": q.get("pose") is not None}
        else:
            r = embed.search(embed.text_query(body.text), body.paths, top_k=body.top_k,
                             min_score=body.min_score, gap_s=body.gap_s, center=body.center,
                             part=body.part, part_near_max=body.part_near_max, is_text=True)
            r["query"] = {"kind": "text", "text": body.text}
        return r
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"搜索失败: {type(e).__name__}: {e}") from e


@app.get("/api/v1/decode")
def decode_settings():
    """解码这几个旋钮**实际生效的值**，外加 .env 找没找到。

    「我改了 .env 为什么没变」是这类配置最常见的坑：文件路径不对、服务没重启、
    环境里已经有同名变量压过了。猜是猜不出来的——把生效值直接打出来，
    跟 ps 里看到的 ffmpeg 命令一对就知道。
    """
    import os

    from . import seek

    n = 6
    picks = [seek.pick_hwaccel() for _ in range(n)]
    return {
        "env_file": config.ENV_FILE,
        "env_file_exists": os.path.isfile(config.ENV_FILE),
        "decode_ffmpeg": config.DECODE_FFMPEG,
        "decode_hwaccel": config.DECODE_HWACCEL,
        "decode_gpu_filter": config.DECODE_GPU_FILTER,
        "decode_cpu_share": config.DECODE_CPU_SHARE,
        "decode_cpu_threads": config.DECODE_CPU_THREADS,
        # 接下来 6 路会怎么分（True=NVDEC）。这一行最直接：
        # 全是 True 就说明 cpu_share 没生效，别再猜了
        "next_6_streams_nvdec": picks,
        "detect_imgsz": config.DETECT_IMGSZ,
        "detect_batch": config.DETECT_BATCH,
        "embed_batch": config.EMBED_BATCH,
        "pose_on": bool(config.POSE_ONNX and os.path.isfile(config.POSE_ONNX)),
        "note": ("cpu_share=0 = 全走 NVDEC。5090 只有 2 个 NVDEC 引擎，"
                 "并发开到十几路全走它只会排队，不会更快——要么把并发压回 6，"
                 "要么让一部分路走 CPU 软解（DECODE_CPU_SHARE=0.5）。"),
    }


@app.get("/api/v1/embed/spent")
def embed_spent(n: int = 30):
    """最近 n 份索引各步各花了多少秒。建索引慢的时候先看这个，别凭感觉调旋钮。"""
    return embed.spent_summary(n)


@app.post("/api/v1/lowlight")
def lowlight_clip(body: dict = Body(...)):
    """夜里那几秒黑得看不见狗在干嘛——把那一刻捞出来看清楚。

    {path, t, window_s?, fill?, model?} → 三张 base64 JPEG：
      raw      原样（先证明"原片就是这样"，不是平台把画面弄黑了）
      stretch  只拉伸（不编造，但噪声照样放大）
      stacked  前后几秒对齐后平均，再拉伸（软件能做到的上限，同样不编造）

    **stack_info 里的 gain 是判据**：放大到 10 倍还是一片噪点，说明这一路夜间
    根本没拍到东西——该去补红外补光，不是接着调算法，更不是上模型。
    模型（Retinexformer 之类）能把噪声画成看起来合理的画面，而人正是拿这张图
    去确认「这是不是抓挠」的。
    """
    import base64

    rel = str(body.get("path") or "")
    if not rel:
        raise HTTPException(status_code=422, detail="要给 path")
    # 走跟别的接口同一套沙箱解析：只认 VIDEO_ROOT 底下的相对路径，`..` 穿越被挡住
    full = _resolve_under(config.VIDEO_ROOT, rel)
    try:
        r = lowlight.enhance_clip(
            full, float(body.get("t") or 0.0),
            window_s=float(body.get("window_s") or 2.0),
            fill=float(body.get("fill") or 1.0),
            model=(body.get("model") or None),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    out = {k: v for k, v in r.items() if not isinstance(v, (bytes, bytearray))}
    for k in ("raw", "stretch", "stacked", "model"):
        if isinstance(r.get(k), (bytes, bytearray)):
            out[k] = base64.b64encode(r[k]).decode()
    return out


@app.get("/api/v1/gpu")
def gpu_report():
    """显存被谁占了：逐个模型 + torch 缓存 + 非 torch 那部分。

    nvitop 只看得到「这个进程 27 GiB」，拆不开是检测、分割、向量还是姿态，
    在容器里更看不出来。这一个接口把它拆开。
    """
    return gpumem.report()


@app.post("/api/v1/gpu/release")
def gpu_release():
    """把 torch 缓存着、已经不用的显存还给系统。不动任何模型。"""
    return gpumem.release()


@app.get("/api/v1/models")
def models_list():
    """本地模型一张表：在不在、跑在哪、权重、错误、调用计数。不触发加载。"""
    return models.overview()


class ModelActionIn(BaseModel):
    action: str = Field(..., pattern="^(load|unload|test)$")


@app.post("/api/v1/models/{key}")
def models_act(key: str, body: ModelActionIn):
    """加载（含预热）/ 卸载（释放显存）/ 测试（跑一次最小推理，回耗时和结果）。"""
    try:
        return models.act(key, body.action)
    except KeyError:
        raise HTTPException(404, f"没有这个模型：{key}") from None


@app.get("/api/v1/models/vllm/log")
def models_vllm_log(n: int = 300):
    """vLLM 这次启动的日志（最后 n 行）+ 挑出来的报错行。"""
    from . import vllm_manager

    return {"lines": vllm_manager.read_log(max(20, min(3000, n))), "errors": vllm_manager.log_errors(30)}


@app.post("/api/v1/models/meter/reset")
def models_meter_reset():
    models.reset_meter()
    return {"ok": True}


class LlmTestIn(BaseModel):
    llm: LlmIn


@app.post("/api/v1/llm/test")
def llm_test(body: LlmTestIn):
    """key 对不对、模型名对不对：发一句最短的话看回不回。不带图，几乎不花钱。"""
    try:
        llm = llmmod.from_dict(body.llm.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return llmmod.ping(llm)


def main():
    import uvicorn

    os.makedirs(config.LOG_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)


if __name__ == "__main__":
    main()
