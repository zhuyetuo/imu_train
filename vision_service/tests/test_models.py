"""本地模型集中管理：一张表、加载 / 卸载 / 测试、调用计数。"""

from __future__ import annotations

import numpy as np

from vision_service import dog, embed, meter, models, pose


def test_计数器_次数帧数耗时_成败(monkeypatch):
    meter.reset()
    with meter.timed("dog", frames=16):
        pass
    meter.record("dog", 40, frames=1)
    try:
        with meter.timed("dog"):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    m = meter.get("dog")
    assert m["calls"] == 3 and m["frames"] == 18 and m["errors"] == 1 and m["max_ms"] >= 40 and m["last_at"]
    assert meter.get("nope")["calls"] == 0
    meter.reset()
    assert meter.get("dog")["calls"] == 0


def test_列表_不触发加载_每个都有状态和计数(monkeypatch):
    monkeypatch.setattr(dog, "_model", None)
    monkeypatch.setattr(dog, "_load_error", "没装 ultralytics")
    monkeypatch.setattr(embed, "_model", None)
    calls = []
    monkeypatch.setattr(dog, "_load", lambda force=False: calls.append("dog"))
    ov = models.overview()
    assert calls == []                                    # 列表不加载
    keys = [m["key"] for m in ov["models"]]
    assert keys == ["dog", "sam", "embed", "pose", "vllm"]
    d = next(m for m in ov["models"] if m["key"] == "dog")
    assert d["available"] is False and d["error"] == "没装 ultralytics" and d["meter"]["calls"] == 0
    assert "uptime_s" in ov and all("name" in m and "purpose" in m for m in ov["models"])


def test_加载卸载测试_走各模块(monkeypatch):
    meter.reset()
    monkeypatch.setattr(dog, "warmup", lambda: {"warm": True, "error": None})
    monkeypatch.setattr(dog, "_model", object())
    monkeypatch.setattr(dog, "_device_used", "cuda:0")
    r = models.act("dog", "load")
    assert r["ok"] is True and r["status"]["device"] == "cuda:0"
    # 测试：detect 被调一次
    monkeypatch.setattr(dog, "_load", lambda force=False: None)

    def fake_detect(frame, conf=0.35):
        meter.record("dog", 12)
        return [{"bbox": [0, 0, 1, 1], "conf": 0.5}]

    monkeypatch.setattr(dog, "detect", fake_detect)
    r = models.act("dog", "test")
    assert r["ok"] is True and "框到 1 个" in r["detail"] and r["latency_ms"] >= 0
    assert meter.get("dog")["calls"] == 0                 # 页面上的测试不算业务调用
    # 卸载：模型清掉、状态变不可用
    r = models.act("dog", "unload")
    assert r["ok"] is True and dog._model is None and r["status"]["available"] is False
    # 测试失败原样带回
    monkeypatch.setattr(dog, "_load_error", "坏了")
    r = models.act("dog", "test")
    assert r["ok"] is False and "坏了" in r["error"]
    # 不存在的 key
    try:
        models.act("nope", "load")
        assert False
    except KeyError:
        pass
    # 姿态：没权重时 load 报错、unload 干净
    monkeypatch.setattr(pose.config, "POSE_ONNX", "/no/x.onnx")
    monkeypatch.setattr(pose, "_loaded", False)
    monkeypatch.setattr(pose, "_model", None)
    r = models.act("pose", "load")
    assert r["ok"] is False and "权重" in r["error"]
    assert models.act("pose", "unload")["ok"] is True
    # 加载在 CPU 上、但想要 cuda：状态里提醒装 GPU 版
    monkeypatch.setattr(pose, "_model", object())
    monkeypatch.setattr(pose, "_device_used", "cpu")
    monkeypatch.setattr(pose.config, "POSE_DEVICE", "cuda")
    st = next(m for m in models.list_models() if m["key"] == "pose")
    assert st["available"] is True and st["device"] == "cpu" and "onnxruntime-gpu" in st["error"]


def test_接口(monkeypatch):
    from fastapi.testclient import TestClient

    from vision_service import app as appmod

    with TestClient(appmod.app) as tc:
        r = tc.get("/api/v1/models")
        assert r.status_code == 200 and [m["key"] for m in r.json()["models"]] == ["dog", "sam", "embed", "pose", "vllm"]
        monkeypatch.setattr(models, "act", lambda k, a: {"ok": True, "error": None, "status": {}, "k": k, "a": a})
        r = tc.post("/api/v1/models/dog", json={"action": "test"})
        assert r.status_code == 200 and r.json()["a"] == "test"
        assert tc.post("/api/v1/models/dog", json={"action": "boom"}).status_code == 422
        monkeypatch.setattr(models, "act", lambda k, a: (_ for _ in ()).throw(KeyError(k)))
        assert tc.post("/api/v1/models/nope", json={"action": "load"}).status_code == 404
        assert tc.post("/api/v1/models/meter/reset").status_code == 200


def test_检测入口有计数(monkeypatch):
    meter.reset()

    class R:
        boxes = []

    class M:
        def predict(self, *a, **k):
            return [R()] * (len(a[0]) if isinstance(a[0], list) else 1)

    monkeypatch.setattr(dog, "_model", M())
    monkeypatch.setattr(dog, "_device_used", "cpu")
    frames = [np.zeros((8, 8, 3), dtype="uint8")] * 5
    dog.detect_batch(frames)
    dog.detect(frames[0])
    m = meter.get("dog")
    assert m["calls"] == 2 and m["frames"] == 6


def test_检测权重路径_老位置的文件挪到models目录(tmp_path, monkeypatch):
    dest = tmp_path / "models" / "vision" / "yolo" / "yolo26x.pt"
    monkeypatch.setattr(dog.config, "DOG_WEIGHTS", str(dest))
    old = tmp_path / "yolo26x.pt"
    old.write_bytes(b"w")
    monkeypatch.chdir(tmp_path)
    assert dog._weights_path() == str(dest) and dest.read_bytes() == b"w" and not old.exists()
    # 没有老文件：目录建好，路径原样返回，让 ultralytics 自己下
    dest.unlink()
    assert dog._weights_path() == str(dest) and dest.parent.is_dir()
