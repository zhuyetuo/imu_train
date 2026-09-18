"""本地大模型（vLLM）的起停：从「模型服务」页一键启动 / 停止 / 测试。

vLLM 是独立进程（自己吃显存，能开 OpenAI 兼容口），这里只是拉起来、盯着、停掉：
  start()  权重不在本地就先下（ModelScope → hf-mirror），再拉起 vllm 进程，端口 VLLM_PORT
  stop()   结束进程，释放显存
  status() 装没装 / 权重在不在 / 进程活没活 / 端口通不通 / 日志末尾

权重放 models/vision/llm/<模型名> 下，跟别的模型一起管。日志在 vision_service/.vllm.log。
vllm 这个包由 run.sh 在有显卡的机器上自动装（VLLM_INSTALL=0 跳过），没装 status 里说清楚。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time

from . import config

_logger = logging.getLogger("vision_service.vllm")

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_started_at: float | None = None
_last_error: str | None = None
_downloading = False
_download_error: str | None = None
_download_log: list[str] = []
# 下载进度：总大小从仓库文件列表算（拿不到就 None），已下的按本地目录体积量
_dl = {"total": 0, "started": None, "samples": []}   # samples: [(t, bytes)] 最近几个点算速度

LOG_PATH = os.path.join(config.HERE, ".vllm.log")


def _installed() -> bool:
    try:
        r = subprocess.run([sys.executable, "-c", "import vllm"], capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


_installed_cache: tuple[float, bool] | None = None


def installed() -> bool:
    """import vllm 要一两秒（它 import 一堆东西），结果缓存一分钟。"""
    global _installed_cache
    now = time.monotonic()
    if _installed_cache and now - _installed_cache[0] < 60:
        return _installed_cache[1]
    ok = _installed()
    _installed_cache = (now, ok)
    return ok


def local_dir() -> str:
    return os.path.join(config.VLLM_LOCAL_ROOT, config.VLLM_MODEL.split("/")[-1])


def missing_weight_files() -> list[str]:
    """按 model.safetensors.index.json 核对：里面引用的每个分片都得在、且没有下到一半的临时文件。
    没有 index 的小模型：至少一个 .safetensors / .bin。返回缺的文件名（空 = 齐了）。"""
    import json

    d = local_dir()
    if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "config.json")):
        return ["config.json"]
    files = set(os.listdir(d))
    # 下到一半的临时文件（huggingface_hub 是 .incomplete，modelscope 是 .tmp / ._）
    partial = [f for f in files if f.endswith((".incomplete", ".tmp")) or f.startswith("._")]
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        try:
            with open(idx, encoding="utf-8") as f:
                shards = sorted(set((json.load(f).get("weight_map") or {}).values()))
        except (OSError, ValueError):
            shards = []
        missing = [sh for sh in shards if sh not in files]
        return missing + partial
    if any(f.endswith((".safetensors", ".bin")) for f in files):
        return partial
    return ["*.safetensors"]


def weights_ready() -> bool:
    return not missing_weight_files()


def _dir_bytes(d: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _repo_total_bytes() -> int:
    """仓库里权重文件加起来多大：先问 ModelScope，再问 hf-mirror；都拿不到返回 0（只显示已下多少）。"""
    try:
        from modelscope.hub.api import HubApi

        files = HubApi().get_model_files(config.VLLM_MODEL, recursive=True)
        n = sum(int(f.get("Size") or 0) for f in files if f.get("Type") != "tree")
        if n > 0:
            return n
    except Exception:  # noqa: BLE001
        pass
    try:
        from huggingface_hub import HfApi

        info = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")).model_info(config.VLLM_MODEL, files_metadata=True)
        return sum(int(getattr(s, "size", 0) or 0) for s in info.siblings)
    except Exception:  # noqa: BLE001
        return 0


def download_progress() -> dict | None:
    """{pct, done_mb, total_mb, speed_mbps, eta_s}；没在下返回 None。"""
    if not _downloading:
        return None
    done = _dir_bytes(local_dir()) if os.path.isdir(local_dir()) else 0
    now = time.monotonic()
    _dl["samples"].append((now, done))
    _dl["samples"] = [x for x in _dl["samples"] if now - x[0] <= 30][-10:]
    speed = 0.0
    if len(_dl["samples"]) >= 2:
        (t0, b0), (t1, b1) = _dl["samples"][0], _dl["samples"][-1]
        if t1 > t0:
            speed = max(0.0, (b1 - b0) / (t1 - t0))
    total = _dl["total"]
    eta = int((total - done) / speed) if total and speed > 0 and total > done else None
    return {"pct": round(min(100.0, done / total * 100), 1) if total else None,
            "done_mb": round(done / 1e6, 1), "total_mb": round(total / 1e6, 1) if total else None,
            "speed_mbps": round(speed / 1e6, 2), "eta_s": eta,
            "elapsed_s": int(now - _dl["started"]) if _dl["started"] else 0}


def _download() -> None:
    """ModelScope 先试（国内快），不行走 hf-mirror。进度写进 _download_log 给 status 看。"""
    global _downloading, _download_error
    _downloading = True
    _download_error = None
    _download_log.clear()
    _dl["started"] = time.monotonic()
    _dl["samples"] = []
    _dl["total"] = _repo_total_bytes()
    try:
        os.makedirs(config.VLLM_LOCAL_ROOT, exist_ok=True)
        errors = []
        try:
            from modelscope import snapshot_download as ms_download

            _download_log.append(f"从 ModelScope 下 {config.VLLM_MODEL} …")
            ms_download(config.VLLM_MODEL, local_dir=local_dir())
            if weights_ready():
                _download_log.append("ModelScope 下好了")
                return
        except Exception as e:  # noqa: BLE001
            errors.append(f"ModelScope: {type(e).__name__}: {str(e)[:200]}")
        try:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from huggingface_hub import snapshot_download as hf_download

            _download_log.append(f"从 {os.environ['HF_ENDPOINT']} 下 {config.VLLM_MODEL} …")
            hf_download(config.VLLM_MODEL, local_dir=local_dir())
            if weights_ready():
                _download_log.append("hf-mirror 下好了")
                return
        except Exception as e:  # noqa: BLE001
            errors.append(f"hf-mirror: {type(e).__name__}: {str(e)[:200]}")
        _download_error = "权重没下到：" + "；".join(errors) + f"。可以在能上网的电脑上下好拷到 {local_dir()}"
    finally:
        _downloading = False


def _alive() -> bool:
    return _proc is not None and _proc.poll() is None


def _port_open() -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", config.VLLM_PORT), timeout=1):
            return True
    except OSError:
        return False


def health() -> dict:
    """GET /v1/models：口开了、模型名对得上才算通。"""
    try:
        import httpx

        r = httpx.get(f"http://127.0.0.1:{config.VLLM_PORT}/v1/models", timeout=3)
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}"}
        names = [m.get("id") for m in (r.json().get("data") or [])]
        return {"ok": True, "models": names}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}"}


def _log_tail(n: int = 30) -> list[str]:
    return read_log(n)


def read_log(n: int = 300) -> list[str]:
    """日志最后 n 行（只读这次启动之后的：从最后一个「===== 启动」分隔起）。"""
    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 400000))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("===== "):
            lines = lines[i:]
            break
    return lines[-n:]


_ERR_PAT = ("ERROR", "Error", "error:", "Exception", "CUDA out of memory", "OutOfMemory", "not supported",
            "No module", "RuntimeError", "ValueError", "Killed", "core dumped")


def log_errors(limit: int = 15) -> list[str]:
    """从这次启动的日志里把像报错的行挑出来（引擎那边的错在 API 进程堆栈之前，尾巴看不到）。"""
    out = []
    for line in read_log(2000):
        if any(p in line for p in _ERR_PAT) and "Traceback" not in line and not line.strip().endswith("^"):
            out.append(line.strip()[:300])
    # 去掉连续重复
    dedup: list[str] = []
    for l in out:
        if not dedup or dedup[-1] != l:
            dedup.append(l)
    return dedup[-limit:]


# 启动阶段的里程碑：日志里出现这句 → 走到了这个百分比。分片加载那段再按它自己报的百分比细分
_STAGES = [
    ("Loading safetensors checkpoint shards", 5, "读权重分片"),
    ("Model loading took", 40, "权重已进显存"),
    ("torch_compile_cache", 50, "编译计算图"),
    ("Dynamo bytecode transform", 55, "编译计算图"),
    ("Compiling a graph", 60, "编译计算图"),
    ("torch.compile takes", 70, "编译完成"),
    ("Capturing CUDA graphs", 75, "捕获 CUDA 图"),
    ("Graph capturing finished", 88, "CUDA 图完成"),
    ("init engine", 92, "引擎初始化"),
    ("Starting vLLM API server", 96, "起 API 服务"),
    ("Application startup complete", 100, "就绪"),
]


def startup_progress() -> dict | None:
    """进程在跑、还没 ready 时，从这次启动的日志估个进度 {pct, stage, elapsed_s}。"""
    if not _alive():
        return None
    lines = read_log(3000)
    pct, stage = 0, "拉起进程"
    for line in lines:
        for key, p, name in _STAGES:
            if key in line and p > pct:
                pct, stage = p, name
        if "Loading safetensors checkpoint shards" in line and "%" in line and pct < 40:
            try:
                sub = int(line.split("Completed")[0].strip().split()[-1].rstrip("%"))
                pct, stage = max(pct, 5 + int(sub * 0.35)), f"读权重分片 {sub}%"
            except (ValueError, IndexError):
                pass
    if _port_open() and health().get("ok"):
        pct, stage = 100, "就绪"
    return {"pct": min(100, pct), "stage": stage,
            "elapsed_s": int(time.time() - _started_at) if _started_at else 0}


def status() -> dict:
    alive = _alive()
    port = _port_open()
    return {
        "installed": installed(),
        "model": config.VLLM_MODEL,
        "local_dir": local_dir(),
        "weights_ready": weights_ready(),
        "missing_files": missing_weight_files()[:5],
        "downloading": _downloading,
        "download_error": _download_error,
        "download_log": _download_log[-5:],
        "download_progress": download_progress(),
        "startup_progress": startup_progress(),
        "running": alive,
        "pid": _proc.pid if alive else None,
        "port": config.VLLM_PORT,
        "port_open": port,
        "ready": port and health().get("ok", False),
        "started_at": _started_at,
        "uptime_s": int(time.time() - _started_at) if alive and _started_at else None,
        "error": _last_error,
        "log_tail": _log_tail(),
        # 进程死了（起过但现在不活）也要说：不然页面只看到"没启动"
        "exited": (_proc is not None and _proc.poll() is not None),
        "exit_code": (_proc.poll() if _proc is not None else None),
        "log_errors": log_errors(),
        "args": config.VLLM_ARGS,
    }


def start() -> dict:
    """权重不在就先在后台下（这次返回 downloading，下好后再点一次）；在就拉起进程。
    不等它加载完（7B 要一两分钟），status 里的 ready 变 True 才是能用。"""
    global _proc, _started_at, _last_error
    with _lock:
        _last_error = None
        if _alive():
            return {"ok": True, "note": "已经在跑", "status": status()}
        if _port_open():
            _last_error = f"端口 {config.VLLM_PORT} 已被别的进程占着（可能是手动起的 vllm）；不归这里管，先停掉它"
            return {"ok": False, "error": _last_error, "status": status()}
        if not installed():
            _last_error = "没装 vllm。重跑 ./up.sh deploy -g 会自动装（几 GB）；装完再点启动"
            return {"ok": False, "error": _last_error, "status": status()}
        if not weights_ready():
            if not _downloading:
                threading.Thread(target=_download, name="vllm-download", daemon=True).start()
            miss = missing_weight_files()
            return {"ok": False, "downloading": True,
                    "error": f"权重还没齐（缺 {', '.join(miss[:3])}{'…' if len(miss) > 3 else ''}），正在下到 {local_dir()}"
                             f"（几 GB，看状态里的进度；下过一半的会接着下）；下好后再点一次启动",
                    "status": status()}
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", local_dir(), "--served-model-name", config.VLLM_MODEL,
               "--host", "0.0.0.0", "--port", str(config.VLLM_PORT),
               "--max-model-len", str(config.VLLM_MAX_LEN),
               "--gpu-memory-utilization", str(config.VLLM_GPU_UTIL)]
        if config.VLLM_ARGS:
            cmd += config.VLLM_ARGS.split()
        try:
            log = open(LOG_PATH, "ab")
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动：{' '.join(cmd)}\n".encode())
            _proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=config.HERE,
                                     start_new_session=True)
            _started_at = time.time()
            _logger.info("vLLM 已拉起 pid=%s：%s", _proc.pid, " ".join(cmd))
        except Exception as e:  # noqa: BLE001
            _last_error = f"拉不起 vllm：{type(e).__name__}: {e}"
            return {"ok": False, "error": _last_error, "status": status()}
        return {"ok": True, "note": "已拉起，模型加载要一两分钟，状态里 ready 变 True 才能用", "status": status()}


def stop() -> dict:
    global _proc, _started_at
    with _lock:
        if not _alive():
            _proc = None
            return {"ok": True, "note": "本来就没在跑", "status": status()}
        try:
            import signal

            os.killpg(os.getpgid(_proc.pid), signal.SIGTERM)
            try:
                _proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(_proc.pid), signal.SIGKILL)
                _proc.wait(timeout=10)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"停不掉：{type(e).__name__}: {e}", "status": status()}
        _proc = None
        _started_at = None
        return {"ok": True, "status": status()}


def test() -> dict:
    """真发一句话（不带图）看回不回。"""
    if not _port_open():
        raise RuntimeError("vLLM 没在跑（端口没开）；先点启动，等 ready")
    from . import llm as llmmod

    t0 = time.monotonic()
    r = llmmod.ping(llmmod.LLM("local", config.VLLM_MODEL, "", base_url=f"http://127.0.0.1:{config.VLLM_PORT}/v1"))
    if not r["ok"]:
        raise RuntimeError(r["error"] or "没回")
    return {"latency_ms": int((time.monotonic() - t0) * 1000), "detail": f"回了「{r['reply']}」"}


def shutdown_on_exit() -> None:
    """视觉服务自己退出时把 vLLM 一起带走，别留个孤儿占着显存。"""
    if _alive():
        stop()


__all__ = ["status", "start", "stop", "test", "health", "installed", "weights_ready", "shutdown_on_exit"]
