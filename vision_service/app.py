"""
vision_service：给标注平台用的 GPU 视觉服务。目前只有 SAM 2.1 交互式分割。

跟现有服务的关系：
  - algo_service 是线上服务，一行不碰；
  - label_service（IMU 推理，纯 CPU、进程池）也不碰，只是端口错开（它 8383，这里 8385）；
  - 平台调不通这个服务时，SAM 按钮置灰，其它功能一概不受影响——所以这里挂了
    不是事故，是降级。

路径安全：请求里只传相对 MATERIAL_ROOT 的路径，realpath 之后必须仍在它之内。
"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, sam

app = FastAPI(title="vision_service", version="0.1.0")


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
