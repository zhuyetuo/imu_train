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


def run_date(date: str, job_id: int | None) -> str:
    """这个任务在 data/raw_custom、data/processed_*、results/ 底下用的名字。

    **每个任务一份，互不相干。** 以前直接用数据集名，于是：
      - 同一份数据集再训一次，train_custom.sh 的 --clean 先 rm -rf 掉上一版
        的模型——训出来的每一版都只活到下一次提交为止
      - 提交训练是并发跑的，两次用不同归并表的训练同时整理同一份数据集，
        后一个会在前一个读到一半时把 merged_tmp.json 改写掉
    带上任务号之后这两件事都不会发生，删某一版也只删它自己那一摊。

    job_id 为 None（命令行手动跑、老调用方）时就是原来的数据集名。
    """
    return f"{date}__job{job_id}" if job_id is not None else date


def prepare_export(dataset_spec: dict, job_id: int | None = None) -> None:
    """把 label_infra 导出的数据集整理成 train_custom.sh 认的样子。

    NAS 上是一份 Label Studio 格式 JSON，csv 字段是 NAS_ROOT 下的相对路径。
    train_custom.sh 只认 data/raw_custom/<date>/merged_tmp.json + 固定的
    data/raw_wit/ 当 CSV 目录（按文件名找），所以这里把 JSON 抄过去、csv 改成
    文件名，并把 NAS 上的 CSV 软链进 data/raw_wit/（文件名带日期时间，不会撞）。

    **主数据集和一起训练的那几份走同一条路。** 以前只整理主的，额外批次得事先
    自己躺在 data/raw_custom/ 下——界面上根本没法选，多数据集训练等于用不了。
    """
    _prepare_one(dataset_spec.get("export_json"), run_date(dataset_spec["date"], job_id),
                 dataset_spec.get("label_remap") or {})
    for extra in dataset_spec.get("extra_datasets") or []:
        # 一起训练的那几份也要按任务隔离：归并表是按任务给的，同一份老数据集
        # 被两个任务用不同的归并表同时整理，会互相改写
        _prepare_one(extra.get("export_json"), run_date(extra["date"], job_id),
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


def write_runtime_remap(dataset_spec: dict, job_id: int | None = None) -> str | None:
    """把界面那张归并表落成一份 remap 配置，返回相对仓库根的路径。没有就 None。

    ## 为什么这一步是"能不能识别二级标签"的关键

    训练取的是链的第 0 个（labels[0]）。想让「抓挠-头颈耳」成为一个独立类别，
    光在 apply_label_remap 里把链收成 ["抓挠-头颈耳"] 还不够——train.py 后面
    还要过一张 remap 表，表里没有这个名字的样本会被**直接丢掉**。

    所以训练类别就是**归并表里那些目标名**：映射到自己 = 自成一类，映射到
    「活动」= 折进活动当负样本。表由界面给，这里落成文件传给 --remap。

    只按用到的目标建类，不并进默认那张表：并进去会多出几个一个样本都没有的
    类别（比如把抓挠全拆成部位之后，「抓挠」自己就空了），指标上看着莫名其妙。

    文件名带数据集名，这样 results/ 下的目录名能对上是哪一次训的
    （train.py 的输出目录是 {hz}hz_{remap文件名}）。
    """
    remap = dataset_spec.get("label_remap") or {}
    if not remap:
        return None
    classes = list(dict.fromkeys(remap.values()))
    if not classes:
        return None
    # 按任务号起名：每一版模型都能对回当初用的是哪张表，而不是"每次提交都
    # 覆盖同一个文件"——那样训完第二版，第一版用的归并表就再也找不回来了
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", str(job_id) if job_id is not None else dataset_spec["date"])[:60]
    rel = os.path.join("configs", f"remap_ui_job{safe}.yaml" if job_id is not None else f"remap_ui_{safe}.yaml")
    path = os.path.join(config.REPO_ROOT, rel)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 由标注平台「提交训练」里的归并表生成。\n")
        if job_id is not None:
            f.write(f"# 训练任务 #{job_id}\n")
        f.write(f"# 数据集: {dataset_spec['date']}\n")
        f.write("# 左边是归并之后的类别名，右边是训练类别——映射到自己就是自成一类。\n")
        for c in classes:
            f.write(f"{c}: {c}\n")
    log.info("归并表已写入 %s：%d 个训练类别 %s", rel, len(classes), classes)
    return rel


def build_command(dataset_spec: dict, model_type: str, tag: str | None,
                  job_id: int | None = None) -> list[str]:
    cmd = ["bash", "train_custom.sh", "--date", run_date(dataset_spec["date"], job_id)]
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
        cmd += ["--extra_date", f"{run_date(extra['date'], job_id)}:{hz}"]
    for extra in dataset_spec.get("extra_date", []):
        cmd += ["--extra_date", extra]
    # 3 轴（只用加速度）：端侧没有陀螺仪时要这么训。默认 6 不传，保持原行为
    if int(dataset_spec.get("axes") or 6) == 3:
        cmd += ["--axes", "3"]
    # 界面给了归并表就用它生成的那份，没给还是默认的 3 类表
    runtime_remap = write_runtime_remap(dataset_spec, job_id)
    if runtime_remap:
        cmd += ["--remap", runtime_remap]
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
    job_id = _next_job_id()
    job = {
        "job_id": job_id,
        # 这个任务在磁盘上用的名字。删某一版时照着它清，别的版本碰不到
        "run_date": run_date(dataset_spec["date"], job_id),
        "status": STATUS_QUEUED,
        "dataset_spec": dataset_spec,
        "model_type": model_type,
        "tag": tag,
        "command": " ".join(build_command(dataset_spec, model_type, tag, job_id)),
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

    # 老任务的 json 里没有 run_date：照旧用数据集名，不去猜
    jid = job_id if job.get("run_date") else None
    cmd = build_command(job["dataset_spec"], job["model_type"], job["tag"], jid)
    try:
        await asyncio.to_thread(prepare_export, job["dataset_spec"], jid)
        with open(_log_path(job_id), "wb") as log_f:
            # PYTHONUNBUFFERED：脚本里那几个 python 步骤的输出写进文件时默认是
            # 攒满一块才落盘，网页上看日志就是半天不动、然后一下子蹦出一大截。
            # 关掉缓冲，日志才是真的"实时"
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=config.REPO_ROOT, stdout=log_f, stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
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


# ── 实时日志 ─────────────────────────────────────────────────────────────

# 一次最多回这么多字节。训练日志一跑就是几万行，一口全吐给浏览器会卡住；
# 前端按 offset 一段段接着要，每 2 秒一次，追得上
_LOG_CHUNK = 256 * 1024


def read_log(job_id: int, offset: int = 0) -> dict | None:
    """从 offset 往后读这个任务的训练日志。任务不存在返回 None。

    返回的 offset 是**下次该从哪儿接着读**。前端只管把 text 追加到屏幕上、
    把 offset 存下来下回带上——不用每次拉全量，也不会重复。

    stage 是日志里最后一行 ▶ 开头的，也就是"现在跑到哪一步了"。训练动辄
    几十分钟，只给一堆滚动的日志的话，人看不出离完还有多远。
    """
    job = get_job(job_id)
    if job is None:
        return None
    path = _log_path(job_id)
    text, size, stage = "", 0, None
    if os.path.exists(path):
        size = os.path.getsize(path)
        offset = max(0, min(int(offset), size))
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(_LOG_CHUNK)
        # 可能正好切在一个多字节汉字中间：把尾巴上不完整的那几个字节留到下一次
        cut = len(raw)
        while cut > 0:
            try:
                text = raw[:cut].decode("utf-8")
                break
            except UnicodeDecodeError:
                cut -= 1
        offset += cut
        stage = _last_stage(path)
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "offset": offset,
        "size": size,
        "text": text,
        "stage": stage,
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
    }


def _last_stage(path: str) -> str | None:
    """日志里最后一行 ▶ 开头的。只看文件末尾一段，日志再长也是常数时间。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64 * 1024))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        s = line.strip()
        if s.startswith("▶"):
            return s.lstrip("▶ ").strip()
    return None


# ── 删除某一版 ───────────────────────────────────────────────────────────


class JobBusy(Exception):
    """任务还在跑，或者它的模型正是现在推理在用的那个——不能删。"""


def _under_repo(p: str) -> bool:
    root = os.path.realpath(config.REPO_ROOT)
    rp = os.path.realpath(p)
    return rp == root or rp.startswith(root + os.sep)


def delete_job(job_id: int, active_model_path: str | None) -> dict:
    """删掉这一版训练的全部产物。返回删了哪些路径。任务不存在返回 {"deleted": []}。

    ## 删什么

    新任务（有 run_date，形如 ds_x__job12）：它在磁盘上的那一整摊——整理出来的
    数据、预处理产物、模型、合成数据、日志、它自己那张归并表。**因为每个任务
    本来就各用各的目录，删它碰不到别的版本。**

    老任务（没有 run_date）：那时候同一份数据集的各版本共用目录，按目录删会连
    别的版本一起带走。所以只删它那个模型文件所在的目录，别的不动。

    ## 不能删的

    - 还在跑的（删了它的目录，脚本还在往里写，只会留下一堆半截文件）
    - 模型正是现在推理在用的那个（删了之后下次重启服务就起不来）

    ## 为什么要这么多护栏

    这里是 rm -rf。run_date 要是空的或者不带 __job，拼出来的通配符就是
    data/processed_* ——**所有训练数据一次清光**。所以只认带 __job<本任务号>
    的名字，并且每个路径都要落在仓库里面。
    """
    import glob
    import shutil

    job = get_job(job_id)
    if job is None:
        return {"deleted": []}
    if job.get("status") in (STATUS_QUEUED, STATUS_RUNNING):
        raise JobBusy(f"训练任务 #{job_id} 还在{'排队' if job['status'] == STATUS_QUEUED else '跑'}，等它结束再删")
    mp = job.get("model_path")
    if mp and active_model_path and os.path.realpath(mp) == os.path.realpath(active_model_path):
        raise JobBusy(f"训练任务 #{job_id} 的模型正是现在推理在用的那个，先切到别的模型再删")

    targets: list[str] = []
    rd = job.get("run_date") or ""
    marker = f"__job{job_id}"
    if rd and rd.endswith(marker):
        repo = config.REPO_ROOT
        for pat in (
            f"data/raw_custom/{rd}",
            f"data/processed_{rd}_*",
            f"data/processed_{rd}",
            f"results/processed_{rd}_*",
            f"results/processed_{rd}",
            f"data/synthetic/*_{rd}_*",
            f"tmp/train_full_processed_{rd}_*",
            f"configs/remap_ui_job{job_id}.yaml",
            # 一起训练的那几份，同样带着本任务号
            f"data/raw_custom/*{marker}",
        ):
            targets += glob.glob(os.path.join(repo, pat))
    elif mp:
        # 老任务：只删模型自己那一层目录（.../rf/），它的兄弟版本不动
        targets.append(os.path.dirname(mp))

    deleted: list[str] = []
    for t in sorted(set(targets)):
        if not _under_repo(t) or os.path.realpath(t) == os.path.realpath(config.REPO_ROOT):
            log.warning("删除训练任务 #%d：跳过仓库外的路径 %s", job_id, t)
            continue
        if os.path.isdir(t) and not os.path.islink(t):
            shutil.rmtree(t, ignore_errors=True)
        elif os.path.lexists(t):
            os.remove(t)
        else:
            continue
        deleted.append(os.path.relpath(t, config.REPO_ROOT))
    for p in (_job_path(job_id), _log_path(job_id)):
        if os.path.exists(p):
            os.remove(p)
    log.info("训练任务 #%d 已删除：%s", job_id, deleted)
    return {"deleted": deleted}
