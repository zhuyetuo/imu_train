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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, sam

_logger = logging.getLogger("vision_service")


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warmup_on_startup()
    yield


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

    threading.Thread(target=run, name="sam-warmup", daemon=True).start()


def _resolve(rel_path: str) -> str:
    """相对素材库的路径 → 绝对路径，realpath 必须仍在素材库之内。

    跟 label_service/tooth.py 一个思路：只认相对路径，`..` 穿越在这里被挡住。
    """
    root = os.path.realpath(config.MATERIAL_ROOT)
    full = os.path.realpath(os.path.join(root, rel_path))
    if full != root and not full.startswith(root + os.sep):
        raise HTTPException(422, "非法路径")
    if not os.path.isfile(full):
        raise HTTPException(422, f"文件不存在: {rel_path}")
    return full


class Point(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="归一化横坐标")
    y: float = Field(..., ge=0.0, le=1.0, description="归一化纵坐标")
    label: int = Field(1, description="1=正点（要这块），0=负点（不要这块）")


class SegmentIn(BaseModel):
    path: str = Field(..., description="相对 MATERIAL_ROOT 的路径，比如 口腔验证/2026-09-01-ok/巴利/a.jpg")
    points: list[Point] = Field(default_factory=list, max_length=32)
    box: list[float] | None = Field(None, min_length=4, max_length=4, description="可选框提示 [x,y,w,h] 归一化")


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
        return sam.segment(full, [p.model_dump() for p in body.points], body.box)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"分割失败: {type(e).__name__}: {e}") from e


def main():
    import uvicorn

    os.makedirs(config.LOG_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)


if __name__ == "__main__":
    main()
