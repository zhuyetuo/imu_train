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


def weights_ready() -> bool:
    d = local_dir()
    if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "config.json")):
        return False
    return any(f.endswith((".safetensors", ".bin")) for f in os.listdir(d))


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
    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 20000))
            lines = f.read().decode("utf-8", "replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def status() -> dict:
    alive = _alive()
    port = _port_open()
    return {
        "installed": installed(),
        "model": config.VLLM_MODEL,
        "local_dir": local_dir(),
        "weights_ready": weights_ready(),
        "downloading": _downloading,
        "download_error": _download_error,
        "download_log": _download_log[-5:],
        "download_progress": download_progress(),
        "running": alive,
        "pid": _proc.pid if alive else None,
        "port": config.VLLM_PORT,
        "port_open": port,
        "ready": port and health().get("ok", False),
        "started_at": _started_at,
        "uptime_s": int(time.time() - _started_at) if alive and _started_at else None,
        "error": _last_error,
        "log_tail": _log_tail(),
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
            return {"ok": False, "downloading": True,
                    "error": f"权重还没在本地，正在下到 {local_dir()}（几 GB，看状态里的进度）；下好后再点一次启动",
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
