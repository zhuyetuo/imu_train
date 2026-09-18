"""vLLM 起停：不真起 vllm，桩掉子进程和端口；状态机、权重下载分支、模型服务页的映射。"""

from __future__ import annotations

from vision_service import models, vllm_manager as vm


_n = [0]


def _reset(monkeypatch, tmp_path, installed=True, weights=True, port_open=False):
    # 每次一个新目录：同一个测试里前一次建的权重目录不能带到下一次
    _n[0] += 1
    tmp_path = tmp_path / f"r{_n[0]}"
    tmp_path.mkdir()
    monkeypatch.setattr(vm, "_proc", None)
    monkeypatch.setattr(vm, "_started_at", None)
    monkeypatch.setattr(vm, "_last_error", None)
    monkeypatch.setattr(vm, "_downloading", False)
    monkeypatch.setattr(vm, "_download_error", None)
    monkeypatch.setattr(vm, "installed", lambda: installed)
    monkeypatch.setattr(vm, "_port_open", lambda: port_open)
    monkeypatch.setattr(vm, "health", lambda: {"ok": port_open, "models": ["m"]})
    monkeypatch.setattr(vm.config, "VLLM_LOCAL_ROOT", str(tmp_path))
    monkeypatch.setattr(vm.config, "VLLM_MODEL", "Org/M-AWQ")
    monkeypatch.setattr(vm, "LOG_PATH", str(tmp_path / "vllm.log"))
    if weights:
        d = tmp_path / "M-AWQ"
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"0")


class _P:
    pid = 4242

    def __init__(self):
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def wait(self, timeout=None):
        self.alive = False


def test_没装_没权重_端口被占_三种起不来的原因(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path, installed=False)
    r = vm.start()
    assert r["ok"] is False and "没装 vllm" in r["error"]
    _reset(monkeypatch, tmp_path, weights=False)
    started = []
    monkeypatch.setattr(vm.threading, "Thread", lambda **kw: type("T", (), {"start": lambda self: started.append(kw["target"])})())
    r = vm.start()
    assert r["ok"] is False and r["downloading"] is True and started and vm.status()["weights_ready"] is False
    _reset(monkeypatch, tmp_path, port_open=True)
    r = vm.start()
    assert r["ok"] is False and "占" in r["error"]


def test_起停_日志_状态(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    procs = []

    def popen(cmd, **kw):
        procs.append(cmd)
        kw["stdout"].write(b"INFO loading model\n")
        return _P()

    monkeypatch.setattr(vm.subprocess, "Popen", popen)
    r = vm.start()
    assert r["ok"] is True and procs and "--served-model-name" in procs[0] and "Org/M-AWQ" in procs[0]
    assert "--port" in procs[0] and procs[0][procs[0].index("--port") + 1] == str(vm.config.VLLM_PORT)
    st = vm.status()
    assert st["running"] is True and st["pid"] == 4242 and st["ready"] is False and "loading model" in "".join(st["log_tail"])
    assert vm.start()["note"] == "已经在跑"
    killed = []
    monkeypatch.setattr(vm.os, "killpg", lambda pg, sig: killed.append(sig))
    monkeypatch.setattr(vm.os, "getpgid", lambda pid: pid)
    r = vm.stop()
    assert r["ok"] is True and killed and vm.status()["running"] is False
    assert vm.stop()["note"] == "本来就没在跑"


def test_模型服务页的映射(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    row = next(m for m in models.list_models() if m["key"] == "vllm")
    assert row["available"] is False and "没启动" in row["error"] and row["vllm"]["weights_ready"] is True
    monkeypatch.setattr(vm, "start", lambda: {"ok": True, "note": "x"})
    r = models.act("vllm", "load")
    assert r["ok"] is True
    monkeypatch.setattr(vm, "stop", lambda: {"ok": True})
    assert models.act("vllm", "unload")["ok"] is True
    r = models.act("vllm", "test")
    assert r["ok"] is False and "没在跑" in r["error"]
    # 通了
    _reset(monkeypatch, tmp_path, port_open=True)
    monkeypatch.setattr(vm.llmmod if hasattr(vm, "llmmod") else __import__("vision_service.llm", fromlist=["x"]), "ping",
                        lambda llm, **kw: {"ok": True, "reply": "OK", "error": None})
    r = models.act("vllm", "test")
    assert r["ok"] is True and "OK" in r["detail"]
