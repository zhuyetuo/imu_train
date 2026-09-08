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
  POST /api/v1/tooth/detect           牙齿/口腔照片 YOLO 检测（label_infra 牙齿识别页用），见 tooth.py
  GET  /api/v1/tooth/status
  /api/v1/skin/*                      皮肤评估：PM 规则（问答分/C值/S总分）、IMU 日统计扫描、
                                      ML 模型 A/B，见 skin.py
  GET  /health

启动：  bash label_service/run.sh   （或者直接 uvicorn label_service.app:app --port 8383）
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from label_service import config, jobs, pool, postprocess, skin, tooth
from label_service import queue as infer_queue
from label_service.logging_setup import setup_logging

setup_logging()
log = logging.getLogger("label_service")

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
    log.info("启动 模型=%s classes=%s model_hz=%s device_hz=%s resample=%s target_labels=%s nas_root=%s "
             "infer_workers=%s log_dir=%s", model_path, _bundle["classes"], _bundle["hz"], config.DEVICE_HZ,
             config.RESAMPLE_METHOD, config.TARGET_LABELS, config.NAS_ROOT, config.INFER_WORKERS, config.LOG_DIR)
    missing = [t for t in config.TARGET_LABELS if t not in _bundle["classes"]]
    if missing:
        log.warning("TARGET_LABELS 里这些类别模型没有: %s  模型类别: %s", missing, _bundle["classes"])
    _pool = pool.create_pool(model_path)
    infer_queue.setup(config.INFER_WORKERS, config.INFER_RESERVE)
    yield
    log.info("关闭")
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
        "tooth": tooth.status(),
        "skin_ml": skin.ml_status(),
    }


# ── /infer ──────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    path: str = Field(..., description="NAS_ROOT 下的相对路径，指向一份 IMU CSV")
    sample_id: int | None = Field(None, description="label_infra 的 sample.id，仅用于回显关联")
    mode: Literal["raw", "stable", "viterbi"] = Field("raw", description="raw=调试版（模型逐窗口原始输出）；stable=稳定版（滞回+间隙合并+过滤）；viterbi=稳定版 v2（动态规划解码），见 postprocess.py")


class Segment(BaseModel):
    start_ts: str
    end_ts: str
    conf_max: float
    conf_mean: float
    n_windows: int
    spec: float | None = None   # 陀螺仪 4–8 Hz 能量占比（稳定版才有）


class WindowOut(BaseModel):
    ts: str | None
    label: str
    conf: float
    probs: dict[str, float]
    spec: float | None = None


class Candidate(Segment):
    reason: str   # low_conf=模型低置信 / spectral=频谱像抓挠但模型没判


class InferResponse(BaseModel):
    sample_id: int | None
    path: str
    mode: str = "raw"
    # 疑似抓挠候选（稳定版/v2 才有）：不进正式片段，给人工审核找漏检
    candidates: list[Candidate] = []
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


def _stable_params() -> postprocess.StableParams:
    return postprocess.StableParams(
        event_labels=tuple(config.STABLE_EVENT_LABELS),
        smooth_windows=config.STABLE_SMOOTH_WINDOWS,
        min_state_s=config.STABLE_MIN_STATE_S,
        event_enter=config.STABLE_EVENT_ENTER,
        event_stay=config.STABLE_EVENT_STAY,
        event_gap_s=config.STABLE_EVENT_GAP_S,
        shake_absorb_s=config.STABLE_SHAKE_ABSORB_S,
        event_min_windows=config.STABLE_EVENT_MIN_WINDOWS,
        event_min_mean=config.STABLE_EVENT_MIN_MEAN,
        event_single_conf=config.STABLE_EVENT_SINGLE_CONF,
        spectral_min=config.STABLE_SPECTRAL_MIN,
        viterbi_switch=config.STABLE_VITERBI_SWITCH,
        cand_enter=config.CAND_ENTER,
        cand_stay=config.CAND_STAY,
        cand_min_windows=config.CAND_MIN_WINDOWS,
        cand_min_mean=config.CAND_MIN_MEAN,
        cand_spec_min=config.CAND_SPEC_MIN,
        cand_max=config.CAND_MAX,
        refine_margin_s=config.REFINE_MARGIN_S,
        refine_ratio=config.REFINE_RATIO,
    )


async def _infer_in_pool(full_path: str, mode: str = "raw", priority: str = infer_queue.PRIORITY_BATCH) -> dict:
    loop = asyncio.get_running_loop()
    t0 = time.time()
    try:
        # 批量的要抢槽位排队，交互式的直接进（见 queue.py）
        async with infer_queue.slot(priority):
            result = await loop.run_in_executor(_pool, pool._infer_one, full_path)
    except Exception:
        log.exception("推理失败 %s (%.1fs)", full_path, time.time() - t0)
        raise
    envelope = result.pop("envelope", None)
    result["candidates"] = []
    if mode in ("stable", "viterbi"):
        # 同一次推理的逐窗口结果做后处理，模型不用再跑一遍
        params = _stable_params()
        geom = (_bundle["window_s"], _bundle["stride_s"], _bundle["label_mode"])
        result["segments"] = postprocess.stabilize(
            result["windows"], _bundle["classes"], config.TARGET_LABELS, *geom, params, algo=mode,
        )
        cands = postprocess.scratch_candidates(result["windows"], result["segments"], *geom, params)
        # 按置信度从高到低裁到上限。裁掉多少要记下来——一小时几百条人根本审不过来，
        # 但也不能让人以为"就这么点"，日志里说清楚
        if len(cands) > config.CAND_MAX:
            log.info("疑似抓挠候选 %d 条，只保留置信度最高的 %d 条（%s）",
                     len(cands), config.CAND_MAX, os.path.basename(full_path))
            cands = cands[: config.CAND_MAX]
        result["candidates"] = cands
        if config.REFINE_ENABLED:
            for lab in config.STABLE_EVENT_LABELS:
                postprocess.refine_boundaries(result["segments"].get(lab) or [], envelope, params)
            postprocess.refine_boundaries(result["candidates"], envelope, params)
    result["mode"] = mode
    counts = {k: len(v) for k, v in result["segments"].items() if v}
    log.info("推理完成[%s] %s  %.1fs  windows=%d  segments=%s  candidates=%d", mode,
             os.path.relpath(full_path, config.NAS_ROOT), time.time() - t0, result["n_windows"], counts,
             len(result["candidates"]))
    return result


@app.post("/api/v1/label/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    full_path = _resolve_nas_path(req.path)
    try:
        result = await _infer_in_pool(full_path, req.mode, infer_queue.PRIORITY_INTERACTIVE)
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
    mode: Literal["raw", "stable", "viterbi"] = "raw"


class InferBatchResult(BaseModel):
    sample_id: int | None
    path: str
    ok: bool
    error: str | None = None
    result: InferResponse | None = None


@app.get("/api/v1/label/queue")
async def queue_status():
    """当前排队情况：几个在算、几个在等、平均一个文件多久、预计多久消化完。"""
    return infer_queue.stats()


@app.post("/api/v1/label/infer_batch", response_model=list[InferBatchResult])
async def infer_batch(req: InferBatchRequest):
    """一批文件同时丢进进程池并行跑，单个文件失败不影响其它的，逐项带回
    ok/error。几百个样本一次发一个请求就行，不用调用方自己控制并发。"""
    log.info("批量推理开始 %d 个  队列=%s", len(req.items), infer_queue.stats())
    t0 = time.time()
    async def _one(item: InferBatchItem) -> InferBatchResult:
        try:
            full_path = _resolve_nas_path(item.path)
            result = await _infer_in_pool(full_path, req.mode)
            return InferBatchResult(sample_id=item.sample_id, path=item.path, ok=True, result=InferResponse(
                sample_id=item.sample_id, path=item.path, model_path=_bundle["model_path"],
                classes=_bundle["classes"], **result))
        except HTTPException as e:
            return InferBatchResult(sample_id=item.sample_id, path=item.path, ok=False, error=str(e.detail))
        except Exception as e:  # noqa: BLE001
            return InferBatchResult(sample_id=item.sample_id, path=item.path, ok=False, error=f"{type(e).__name__}: {e}")
    results = await asyncio.gather(*(_one(i) for i in req.items))
    n_ok = sum(1 for r in results if r.ok)
    log.info("批量推理结束 %d 个  成功 %d 失败 %d  %.1fs  队列=%s", len(results), n_ok, len(results) - n_ok,
             time.time() - t0, infer_queue.stats())
    return results


# ── /train ──────────────────────────────────────────────────────────────

class DatasetSpec(BaseModel):
    date: str = Field(..., description="跟 train_custom.sh --date 一致，如 2026_8_20")
    extra_date: list[str] = Field(default_factory=list, description="跟 --extra_date 一致，格式 DATE:HZ")
    missing_strategy: str | None = Field(None, description="none/drop/ffill/drop_window")
    skip_syn: bool = Field(False, description="只训练方案A，跳过合成数据")
    feat_workers: int | None = Field(None, description="特征提取并行数，-1=全部CPU")
    source_hz: int | None = Field(None, description="主批次真实采样率（NAS 原始 50Hz 数据必须传 50）")
    hz: int | None = Field(None, description="训练目标采样率，默认脚本里的 16")
    export_json: str | None = Field(None, description="label_infra 导出的 Label Studio 格式 JSON 在 NAS_ROOT 下的相对路径；"
                                                    "csv 字段是 NAS_ROOT 下的相对路径，服务会整理成 data/raw_custom/<date>/merged_tmp.json")
    clean: bool = Field(False, description="--clean 重新生成缓存（换了导出数据时要传）")


class TrainRequest(BaseModel):
    dataset: DatasetSpec
    model_type: str = Field("rf", description="跟 train_custom.sh --model 一致")
    tag: str | None = None


@app.post("/api/v1/label/train")
async def submit_train(req: TrainRequest):
    job = jobs.create_job(req.dataset.model_dump(), req.model_type, req.tag)
    log.info("训练任务 #%d 提交: %s", job["job_id"], job["command"])
    asyncio.create_task(jobs.run_job(job["job_id"]))
    return job


class ModelSwitchRequest(BaseModel):
    model_path: str = Field(..., description="训练产出的 .pkl，绝对路径或相对 imu_train 仓库")


@app.post("/api/v1/label/model/switch")
async def switch_model(req: ModelSwitchRequest):
    """
    切换当前推理用的模型：重新加载元数据、重建进程池（worker 启动时各自加载模型）。
    正在跑的推理会被取消，调用方（label_infra 训练页「启用此模型」）自己挑空闲时候点。
    只改运行时，不改 config/LABEL_MODEL——重启服务会回到配置里的模型，想固定就把
    LABEL_MODEL 环境变量改成这个路径。
    """
    global _pool
    path = req.model_path if os.path.isabs(req.model_path) else os.path.join(config.REPO_ROOT, req.model_path)
    if not os.path.isfile(path):
        raise HTTPException(422, f"模型文件不存在: {path}")
    try:
        bundle = await asyncio.to_thread(load_model_bundle, path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"模型加载失败: {type(e).__name__}: {e}") from e
    bundle.pop("model", None)
    old = _pool
    _pool = pool.create_pool(path)
    _bundle.clear()
    _bundle.update(bundle)
    _bundle["model_path"] = path
    if old is not None:
        old.shutdown(wait=False, cancel_futures=True)
    log.info("已切换模型: %s classes=%s", path, _bundle["classes"])
    return {"model_path": path, "classes": _bundle["classes"], "hz": _bundle["hz"]}


@app.get("/api/v1/label/train/{job_id}")
async def get_train_status(job_id: int):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"训练任务 #{job_id} 不存在")
    return job


# ── /tooth ──────────────────────────────────────────────────────────────

class ToothDetectRequest(BaseModel):
    path: str = Field(..., description="相对 MATERIAL_ROOT（如 口腔验证/2026-09-02-ok/Bali/x.jpg）或相对 NAS_ROOT 的图片路径")
    conf: float | None = Field(None, ge=0.0, le=0.95, description="置信度阈值，缺省用服务配置（0.5）；传 0 = 全部检出都返回（内部按 0.001）")
    with_image: bool = Field(True, description="是否返回带检测框的 JPEG（base64）")
    top_k: int = Field(1, ge=0, le=50, description="每张图最多保留几个框（按置信度），默认 1；0 = 不限")


@app.get("/api/v1/tooth/status")
async def tooth_status():
    return tooth.status()


@app.post("/api/v1/tooth/detect")
async def tooth_detect(req: ToothDetectRequest):
    try:
        full_path = tooth.resolve_image_path(req.path)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))
    t0 = time.time()
    try:
        # YOLO 单张几十毫秒到几百毫秒，丢线程池别卡住事件循环；模型在主进程懒加载一次
        conf = None if req.conf is None else max(req.conf, 0.001)  # ultralytics conf=0 会被当成默认值，用 0.001 表示"全部"
        result = await asyncio.to_thread(tooth.detect, full_path, conf, None, req.with_image, req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("牙齿检测失败 %s", full_path)
        raise HTTPException(500, f"检测失败: {type(e).__name__}: {e}") from e
    log.info("牙齿检测 %s  %.2fs  %d 个框 %s", req.path, time.time() - t0, len(result["detections"]),
             [(d["class_name"], d["confidence"]) for d in result["detections"]])
    return {"path": req.path, **result}


# ── /skin ───────────────────────────────────────────────────────────────

class QuestionnaireIn(BaseModel):
    has_hair_loss: str | None = None
    color: str | None = None
    odor: str | None = None
    lesion: str | None = None
    hair_spot: str | None = None
    hair_diameter: str | None = None
    coat: str | None = None


class CScoreIn(BaseModel):
    baseline_count: float | None = 0
    baseline_duration_min: float | None = 0
    today_count: float | None = 0
    today_duration_min: float | None = 0
    cluster_count: float | None = 0
    persistence_days: float | None = 0
    zn: float | None = 0
    zd: float | None = 0
    long_scratch: bool = False
    has_baseline: bool = True


class STotalIn(QuestionnaireIn):
    c_value: float | None = None
    c_tier_hint: str | None = None


class RootsIn(BaseModel):
    roots: str = Field(..., description="逗号分隔，绝对路径或相对 imu_train 仓库根目录")
    target_label: str = "抓挠"


class MlSelectIn(BaseModel):
    rows: list[dict]
    date_label: str
    imu: str
    dog_name: str | None = None
    answers: QuestionnaireIn | None = None


@app.get("/api/v1/skin/options")
async def skin_options():
    return skin.options()


@app.post("/api/v1/skin/questionnaire-score")
async def skin_questionnaire_score(q: QuestionnaireIn):
    return skin.questionnaire_score(q.has_hair_loss, q.color, q.odor, q.lesion, q.hair_spot, q.hair_diameter, q.coat)


@app.post("/api/v1/skin/c-score")
async def skin_c_score(c: CScoreIn):
    return skin.c_score(c.baseline_count, c.baseline_duration_min, c.today_count, c.today_duration_min,
                        c.cluster_count, c.persistence_days, c.zn, c.zd, c.long_scratch, c.has_baseline)


@app.post("/api/v1/skin/s-total")
async def skin_s_total(s: STotalIn):
    return skin.s_total(s.c_value, s.c_tier_hint or "", s.has_hair_loss, s.color, s.odor, s.lesion,
                        s.hair_spot, s.hair_diameter, s.coat)


@app.post("/api/v1/skin/stats/scan")
async def skin_stats_scan(body: RootsIn):
    return await asyncio.to_thread(skin.scan_stats, body.roots, body.target_label)


class EventsIn(BaseModel):
    rows: list[dict] = Field(..., description="[{date, imu, events:[[start_ts,end_ts],...], wear_seconds}]")
    target_label: str = "抓挠"


@app.post("/api/v1/skin/stats/from-events")
async def skin_stats_from_events(body: EventsIn):
    """label_infra 把标注平台里的抓挠片段（AI 版 / 人工版）聚合成按天按 IMU 的事件列表
    发过来，这里按 imu_daily_scratch_stats.py 的口径算成跟 stats.csv 一样的行。"""
    return await asyncio.to_thread(skin.stats_from_events, body.rows, body.target_label)


@app.post("/api/v1/skin/stats/to-c-inputs")
async def skin_stats_to_c(row: dict):
    return skin.stats_to_c_inputs(row)


@app.post("/api/v1/skin/ml/scan")
async def skin_ml_scan(body: RootsIn):
    return await asyncio.to_thread(skin.ml_scan, body.roots)


@app.post("/api/v1/skin/ml/preview")
async def skin_ml_preview(body: MlSelectIn):
    return await asyncio.to_thread(skin.ml_preview, body.rows, body.date_label, body.imu, body.dog_name)


@app.post("/api/v1/skin/ml/predict-c")
async def skin_ml_predict_c(body: MlSelectIn):
    return await asyncio.to_thread(skin.ml_predict, body.rows, body.date_label, body.imu, body.dog_name, "c", None)


@app.post("/api/v1/skin/ml/predict-s")
async def skin_ml_predict_s(body: MlSelectIn):
    ans = body.answers.model_dump() if body.answers else None
    return await asyncio.to_thread(skin.ml_predict, body.rows, body.date_label, body.imu, body.dog_name, "s", ans)


class WeeklyRowsIn(BaseModel):
    rows: list[list] = Field(..., description="每行 36 列（WEEKLY_REPORT_COLUMNS 顺序），短行自动补空")


@app.post("/api/v1/skin/weekly/recompute")
async def skin_weekly_recompute(body: WeeklyRowsIn):
    """对比/误差分析列重算（模型vs人工 次数/时长差、人工vs兽医1 差、兽医1vs兽医2 档位一致性），
    规则同 questionnaire_app.recompute_weekly_errors"""
    from label_service import skin_rules as R
    return {"rows": R.recompute_weekly_errors([list(r) for r in body.rows])}


@app.post("/api/v1/skin/weekly/defaults")
async def skin_weekly_defaults(body: WeeklyRowsIn):
    """按审核链条（模型→人工→兽医1→兽医2）给空格子垫默认值，规则同 _apply_weekly_defaults"""
    from label_service import skin_rules as R
    out = []
    for r in body.rows:
        row = list(r) + [""] * (len(R.WEEKLY_REPORT_COLUMNS) - len(r))
        out.append(R._apply_weekly_defaults(row))
    return {"rows": out}
