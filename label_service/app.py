"""
数据标注平台（label_infra）用的 AI 服务：推理 + 训练，跑在 imu_train 仓库里，
跟命令行用的是同一份代码——

  POST /api/v1/label/infer           同步。传 NAS 相对路径的 IMU CSV，走
                                      src/infer_csv_scratch.infer_file()，跟
                                      run_review_bins_all_days.sh 底层调的是同一个函数、
                                      同一套参数（模型/DEVICE_HZ/RESAMPLE_METHOD/TARGET_LABELS
                                      都从环境变量读，见 config.py），出来的片段跟
                                      infer_result_majority/ 下的 *_infer.json 一致
  POST /api/v1/label/infer_batch     一次传一批路径，进程池并行跑（几百个样本批量预标注用这个，
                                      一个请求就能把 CPU 吃满），逐个返回成功结果或错误信息
  POST /api/v1/label/train            提交训练任务，立刻返回 job_id，后台跑 train_custom.sh
  GET  /api/v1/label/train/{job_id}   轮询训练任务状态
  GET  /health

启动：  bash label_service/run.sh   （或者直接 uvicorn label_service.app:app --port 8383）
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from label_service import config, jobs, pool

sys.path.insert(0, os.path.join(config.REPO_ROOT, "src"))
from label_service.model_loader import load_model_bundle  # noqa: E402

_bundle: dict = {}      # 主进程只留元数据（health 用），真正推理在 pool 的 worker 进程里
_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    model_path = config.resolve_model_path()
    _bundle.update(load_model_bundle(model_path))
    _bundle["model_path"] = model_path
    _bundle.pop("model", None)
    print(f"[label_service] 模型: {model_path}  device_hz={config.DEVICE_HZ}  "
          f"resample={config.RESAMPLE_METHOD}  target_labels={config.TARGET_LABELS}  "
          f"nas_root={config.NAS_ROOT}  infer_workers={config.INFER_WORKERS}")
    missing = [t for t in config.TARGET_LABELS if t not in _bundle["classes"]]
    if missing:
        print(f"[label_service][警告] TARGET_LABELS 里这些类别模型没有: {missing}  模型类别: {_bundle['classes']}")
    _pool = pool.create_pool(model_path)
    yield
    _pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="imu_train label_service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_path": _bundle.get("model_path"),
        "classes": _bundle.get("classes"),
        "model_hz": _bundle.get("hz"),
        "device_hz": config.DEVICE_HZ,
        "resample_method": config.RESAMPLE_METHOD,
        "target_labels": config.TARGET_LABELS,
        "nas_root": config.NAS_ROOT,
        "infer_workers": config.INFER_WORKERS,
    }


# ── /infer ──────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    path: str = Field(..., description="NAS_ROOT 下的相对路径，指向一份 IMU CSV")
    sample_id: int | None = Field(None, description="label_infra 的 sample.id，仅用于回显关联")


class Segment(BaseModel):
    start_ts: str
    end_ts: str
    conf_max: float
    conf_mean: float
    n_windows: int


class WindowOut(BaseModel):
    ts: str | None
    label: str
    conf: float
    probs: dict[str, float]


class InferResponse(BaseModel):
    sample_id: int | None
    path: str
    model_path: str
    classes: list[str]
    n_windows: int
    windows: list[WindowOut]
    segments: dict[str, list[Segment]]   # {label: [片段]}，跟 *_infer.json 的 scratch_segments 一致


def _resolve_nas_path(relative_path: str) -> str:
    if os.path.isabs(relative_path) or ".." in relative_path.split("/"):
        raise HTTPException(422, f"非法路径（必须是 NAS_ROOT 下的相对路径）: {relative_path}")
    full = os.path.join(config.NAS_ROOT, relative_path)
    if not os.path.isfile(full):
        raise HTTPException(422, f"文件不存在: {full}")
    return full


async def _infer_in_pool(full_path: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, pool._infer_one, full_path)


@app.post("/api/v1/label/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    full_path = _resolve_nas_path(req.path)
    try:
        result = await _infer_in_pool(full_path)
    except Exception as e:  # noqa: BLE001 把底层错误原样带给调用方，方便排查
        raise HTTPException(500, f"推理失败: {type(e).__name__}: {e}") from e
    return InferResponse(
        sample_id=req.sample_id, path=req.path, model_path=_bundle["model_path"],
        classes=_bundle["classes"], **result,
    )


class InferBatchItem(BaseModel):
    path: str
    sample_id: int | None = None


class InferBatchRequest(BaseModel):
    items: list[InferBatchItem] = Field(..., min_length=1)


class InferBatchResult(BaseModel):
    sample_id: int | None
    path: str
    ok: bool
    error: str | None = None
    result: InferResponse | None = None


@app.post("/api/v1/label/infer_batch", response_model=list[InferBatchResult])
async def infer_batch(req: InferBatchRequest):
    """一批文件同时丢进进程池并行跑，单个文件失败不影响其它的，逐项带回
    ok/error。几百个样本一次发一个请求就行，不用调用方自己控制并发。"""
    async def _one(item: InferBatchItem) -> InferBatchResult:
        try:
            full_path = _resolve_nas_path(item.path)
            result = await _infer_in_pool(full_path)
            return InferBatchResult(sample_id=item.sample_id, path=item.path, ok=True, result=InferResponse(
                sample_id=item.sample_id, path=item.path, model_path=_bundle["model_path"],
                classes=_bundle["classes"], **result))
        except HTTPException as e:
            return InferBatchResult(sample_id=item.sample_id, path=item.path, ok=False, error=str(e.detail))
        except Exception as e:  # noqa: BLE001
            return InferBatchResult(sample_id=item.sample_id, path=item.path, ok=False, error=f"{type(e).__name__}: {e}")
    return await asyncio.gather(*(_one(i) for i in req.items))


# ── /train ──────────────────────────────────────────────────────────────

class DatasetSpec(BaseModel):
    date: str = Field(..., description="跟 train_custom.sh --date 一致，如 2026_8_20")
    extra_date: list[str] = Field(default_factory=list, description="跟 --extra_date 一致，格式 DATE:HZ")
    missing_strategy: str | None = Field(None, description="none/drop/ffill/drop_window")
    skip_syn: bool = Field(False, description="只训练方案A，跳过合成数据")
    feat_workers: int | None = Field(None, description="特征提取并行数，-1=全部CPU")


class TrainRequest(BaseModel):
    dataset: DatasetSpec
    model_type: str = Field("rf", description="跟 train_custom.sh --model 一致")
    tag: str | None = None


@app.post("/api/v1/label/train")
async def submit_train(req: TrainRequest):
    job = jobs.create_job(req.dataset.model_dump(), req.model_type, req.tag)
    asyncio.create_task(jobs.run_job(job["job_id"]))
    return job


@app.get("/api/v1/label/train/{job_id}")
async def get_train_status(job_id: int):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"训练任务 #{job_id} 不存在")
    return job
