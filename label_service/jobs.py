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


def apply_label_remap(tasks: list, remap: dict) -> int:
    """按 {原名: 新名} 改写标注的类别名，返回改了多少段。

    ## 为什么要有这一步

    训练那边取的是整条链的**第 0 个**（labels[0]，见
    src/data/labelstudio_to_custom.py），所以「抓挠-肩胸」进训练时本来就是
    「抓挠」。但人想做的归并不止这一种：

      - 某个细类太少（肩胸 1 段 9 秒）→ 并进兄弟类别里
      - 某个大类不是要训的目标（舔、甩头/抖身）→ 折进「活动」当负样本

    第二种尤其要紧：configs/remap_custom_3class.yaml 用的还是旧模板的名字
    （舔身体 / 甩身体 / 蹭擦身体），现在模板发出来的是「舔」「甩头/抖身」
    「蹭」——对不上，apply_remap() 就把这些样本**静默丢掉**了。

    ## 为什么放在训练时、不放在导出时

    导出时改的话，那份数据集就永远是改过的，想换一种归并得重导一遍；而归并
    方案是要反复试的（这次把舔当负样本，下次单独训它）。放这里，同一份数据集
    能喂给不同的归并跑。

    ## 匹配规则：先叶子后根

    一段的链是 [抓挠, 抓挠-躯干]。先拿最后一级去查表（能精确到细类），查不到
    再拿第一级（整个大类一起搬）。命中就把整条链换成 [新名]——留着旧的父级
    没有意义，训练只看第 0 个。

    链已经正好是 [新名] 才算没事干；只比第 0 个是不够的，那样「抓挠-头颈耳
    → 抓挠」会被当成没变，二级标签留在链上不收。
    """
    if not remap:
        return 0
    n = 0
    for t in tasks:
        for ann in t.get("annotations") or []:
            for seg in ann.get("result") or []:
                v = seg.get("value") or {}
                labels = v.get("timeserieslabels") or []
                if not labels:
                    continue
                target = remap.get(labels[-1]) or remap.get(labels[0])
                # 已经正好就是这一个名字才算没事干。**不能只比第 0 个**：
                # 把「抓挠-头颈耳」并成「抓挠」时第 0 个本来就是抓挠，那样判
                # 就会跳过，二级标签留在链上没被收掉，人在界面上明明选了合并
                if not target or labels == [target]:
                    continue
                v["timeserieslabels"] = [target]
                n += 1
    return n


def prepare_export(dataset_spec: dict) -> None:
    """把 label_infra 导出的数据集整理成 train_custom.sh 认的样子。

    NAS 上是一份 Label Studio 格式 JSON，csv 字段是 NAS_ROOT 下的相对路径。
    train_custom.sh 只认 data/raw_custom/<date>/merged_tmp.json + 固定的
    data/raw_wit/ 当 CSV 目录（按文件名找），所以这里把 JSON 抄过去、csv 改成
    文件名，并把 NAS 上的 CSV 软链进 data/raw_wit/（文件名带日期时间，不会撞）。

    **主数据集和一起训练的那几份走同一条路。** 以前只整理主的，额外批次得事先
    自己躺在 data/raw_custom/ 下——界面上根本没法选，多数据集训练等于用不了。
    """
    _prepare_one(dataset_spec.get("export_json"), dataset_spec["date"],
                 dataset_spec.get("label_remap") or {})
    for extra in dataset_spec.get("extra_datasets") or []:
        _prepare_one(extra.get("export_json"), extra["date"],
                     dataset_spec.get("label_remap") or {})


def _prepare_one(export_json: str | None, date: str, label_remap: dict) -> None:
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
    n_remapped = apply_label_remap(tasks, label_remap)
    data_dir = os.path.join(config.REPO_ROOT, "data", "raw_custom", date)
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "merged_tmp.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)
    log.info("数据集 %s 已整理: %d 个任务 → %s%s", date, len(tasks), data_dir,
             f"（按类别映射改写了 {n_remapped} 段）" if n_remapped else "")


def build_command(dataset_spec: dict, model_type: str, tag: str | None) -> list[str]:
    cmd = ["bash", "train_custom.sh", "--date", dataset_spec["date"]]
    if dataset_spec.get("source_hz"):
        cmd += ["--source_hz", str(dataset_spec["source_hz"])]
    if dataset_spec.get("hz"):
        cmd += ["--hz", str(dataset_spec["hz"])]
    if dataset_spec.get("clean"):
        cmd += ["--clean"]
    # 一起训练的那几份。界面选的走 extra_datasets（带各自的 json 和采样率），
    # extra_date 是老的手写形式（DATE:HZ，数据已经在 data/raw_custom 下），留着
    for extra in dataset_spec.get("extra_datasets") or []:
        hz = extra.get("source_hz") or dataset_spec.get("source_hz") or 50
        cmd += ["--extra_date", f"{extra['date']}:{hz}"]
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
