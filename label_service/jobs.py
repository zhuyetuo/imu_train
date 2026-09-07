"""
训练任务：/train 提交后立刻返回 job_id，后台起子进程跑仓库根目录的
train_custom.sh（跟命令行手动跑完全是同一个脚本、同一套参数），状态存成
JOBS_DIR/{job_id}.json，stdout/stderr 落 JOBS_DIR/{job_id}.log，方便出问题时
直接看日志。没有用数据库——这个服务就是给标注平台调的单机工具，文件够用，
也方便命令行 cat 一眼看到。

训练完不会自动把新模型切成 /infer 正在用的那个：LABEL_MODEL 是启动时定的，
要换模型改环境变量重启服务，训练和"上线"分开。
"""

import asyncio
import json
import logging
import os
import re
import time

from label_service import config

log = logging.getLogger("label_service.jobs")

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# train_custom.sh 跑完会自己打印"模型路径:"下面的"纯标注: xxx.pkl"/"带合成: xxx.pkl"，
# 从这里解析比自己重新拼一遍 DATASET_TAG/HZ/MODEL_TYPE 的目录规则可靠
_MODEL_PATH_RE = re.compile(r"^\s*(纯标注|带合成):\s*(\S+\.pkl)\s*$", re.MULTILINE)


def _job_path(job_id: int) -> str:
    return os.path.join(config.JOBS_DIR, f"{job_id}.json")


def _log_path(job_id: int) -> str:
    return os.path.join(config.JOBS_DIR, f"{job_id}.log")


def _save(job: dict) -> None:
    os.makedirs(config.JOBS_DIR, exist_ok=True)
    tmp = _job_path(job["job_id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _job_path(job["job_id"]))


def get_job(job_id: int) -> dict | None:
    p = _job_path(job_id)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _next_job_id() -> int:
    os.makedirs(config.JOBS_DIR, exist_ok=True)
    ids = [int(n[:-5]) for n in os.listdir(config.JOBS_DIR) if n.endswith(".json") and n[:-5].isdigit()]
    return (max(ids) + 1) if ids else 1


def prepare_export(dataset_spec: dict) -> None:
    """
    label_infra 导出的数据集：NAS 上一份 Label Studio 格式 JSON，csv 字段是 NAS_ROOT
    下的相对路径。train_custom.sh 只认 data/raw_custom/<date>/merged_tmp.json + 固定的
    data/raw_wit/ 当 CSV 目录（按文件名找），所以这里把 JSON 抄过去、csv 改成文件名，
    并把 NAS 上的 CSV 软链进 data/raw_wit/（文件名带日期时间，不会撞）。
    """
    export_json = dataset_spec.get("export_json")
    if not export_json:
        return
    src = os.path.join(config.NAS_ROOT, export_json)
    if not os.path.isfile(src):
        raise RuntimeError(f"导出的数据集 JSON 不存在: {src}")
    with open(src, encoding="utf-8") as f:
        tasks = json.load(f)
    csv_dir = os.path.join(config.REPO_ROOT, "data", "raw_wit")
    os.makedirs(csv_dir, exist_ok=True)
    for t in tasks:
        rel = (t.get("data") or {}).get("csv")
        if not rel:
            continue
        full = rel if os.path.isabs(rel) else os.path.join(config.NAS_ROOT, rel)
        if not os.path.isfile(full):
            raise RuntimeError(f"CSV 不存在: {full}")
        link = os.path.join(csv_dir, os.path.basename(full))
        if os.path.islink(link):
            if os.readlink(link) != full:
                os.remove(link)
                os.symlink(full, link)
        elif not os.path.exists(link):
            os.symlink(full, link)
        t["data"]["csv"] = os.path.basename(full)
    data_dir = os.path.join(config.REPO_ROOT, "data", "raw_custom", dataset_spec["date"])
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "merged_tmp.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)
    log.info("数据集 %s 已整理: %d 个任务 → %s", dataset_spec["date"], len(tasks), data_dir)


def build_command(dataset_spec: dict, model_type: str, tag: str | None) -> list[str]:
    cmd = ["bash", "train_custom.sh", "--date", dataset_spec["date"]]
    if dataset_spec.get("source_hz"):
        cmd += ["--source_hz", str(dataset_spec["source_hz"])]
    if dataset_spec.get("hz"):
        cmd += ["--hz", str(dataset_spec["hz"])]
    if dataset_spec.get("clean"):
        cmd += ["--clean"]
    for extra in dataset_spec.get("extra_date", []):
        cmd += ["--extra_date", extra]
    if dataset_spec.get("missing_strategy"):
        cmd += ["--missing_strategy", dataset_spec["missing_strategy"]]
    if model_type:
        cmd += ["--model", model_type]
    if tag:
        cmd += ["--tag", tag]
    if dataset_spec.get("skip_syn"):
        cmd += ["--skip_syn"]
    if dataset_spec.get("feat_workers"):
        cmd += ["--feat_workers", str(dataset_spec["feat_workers"])]
    return cmd


def parse_model_path(stdout: str) -> str | None:
    matches = dict(_MODEL_PATH_RE.findall(stdout))
    return matches.get("带合成") or matches.get("纯标注")


def _load_metrics(pkl_path: str) -> dict:
    json_path = os.path.splitext(pkl_path)[0] + ".json"
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001 元数据读不出来不该让整个任务判失败
        return {}


def create_job(dataset_spec: dict, model_type: str, tag: str | None) -> dict:
    job = {
        "job_id": _next_job_id(),
        "status": STATUS_QUEUED,
        "dataset_spec": dataset_spec,
        "model_type": model_type,
        "tag": tag,
        "command": " ".join(build_command(dataset_spec, model_type, tag)),
        "model_version": None,
        "model_path": None,
        "metrics": None,
        "error": None,
        "log_path": None,
        "created_at": int(time.time()),
        "started_at": None,
        "finished_at": None,
    }
    _save(job)
    return job


async def run_job(job_id: int) -> None:
    job = get_job(job_id)
    job["status"] = STATUS_RUNNING
    job["started_at"] = int(time.time())
    job["log_path"] = _log_path(job_id)
    _save(job)
    log.info("训练任务 #%d 开始，训练输出见 %s", job_id, _log_path(job_id))

    cmd = build_command(job["dataset_spec"], job["model_type"], job["tag"])
    try:
        await asyncio.to_thread(prepare_export, job["dataset_spec"])
        with open(_log_path(job_id), "wb") as log_f:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=config.REPO_ROOT, stdout=log_f, stderr=asyncio.subprocess.STDOUT,
            )
            await proc.wait()
        with open(_log_path(job_id), encoding="utf-8", errors="replace") as f:
            stdout = f.read()

        if proc.returncode != 0:
            raise RuntimeError(f"train_custom.sh 退出码 {proc.returncode}，详见日志 {_log_path(job_id)}，末尾:\n{stdout[-2000:]}")

        model_path = parse_model_path(stdout)
        if not model_path:
            raise RuntimeError(f"训练结束但没能从输出里解析出模型路径，详见日志 {_log_path(job_id)}")
        if not os.path.isabs(model_path):
            model_path = os.path.join(config.REPO_ROOT, model_path)

        job["status"] = STATUS_DONE
        job["model_path"] = model_path
        job["model_version"] = job["tag"] or f"{job['model_type']}_{job['created_at']}"
        job["metrics"] = _load_metrics(model_path)
        log.info("训练任务 #%d 完成 %.0fs 模型=%s", job_id, time.time() - job["started_at"], model_path)
    except Exception as e:  # noqa: BLE001 后台任务异常不能让服务进程崩，落盘状态即可
        job["status"] = STATUS_FAILED
        job["error"] = str(e)[:4000]
        log.error("训练任务 #%d 失败: %s", job_id, str(e)[:500])
    job["finished_at"] = int(time.time())
    _save(job)
