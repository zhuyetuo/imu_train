"""
画面向量索引。

要守住的：
  1. 索引只存有狗的采样点，t 是 PTS；模型换了要重建，没换直接用旧的。
  2. 搜索：分数最高的在前，能排掉样例自己那一段，相邻命中合成段，缺索引的视频报出来。
  3. 以图搜图那一帧：有狗裁狗，没狗整帧。
  4. 没装模型时 status 如实说、接口 503。

编码器换成假的：向量由图片内容决定（同一张图永远同一个向量），这样"像不像"
可以在测试里造出来。视频用 test_seek 那套假 VideoCapture。
"""

from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest

from vision_service import dog, embed, seek

from .test_seek import _Cap  # noqa: F401  复用假视频


class FakeEncoder:
    """向量 = 图片字节的哈希撒成 8 维再归一化；bias 让指定图片彼此相似。"""

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []

    def _vec(self, seed: bytes):
        h = hashlib.sha256(seed).digest()
        v = np.frombuffer(h[: self.dim * 4], dtype="uint32").astype("float32")
        v = v / 4e9 - 0.5
        return v / np.linalg.norm(v)

    def encode_images(self, jpegs):
        self.calls.append(("img", len(jpegs)))
        return np.stack([self._vec(b) for b in jpegs]) if jpegs else np.zeros((0, self.dim), dtype="float32")

    def encode_text(self, texts):
        self.calls.append(("txt", len(texts)))
        return np.stack([self._vec(t.encode()) for t in texts])


@pytest.fixture()
def index_dir(tmp_path, monkeypatch):
    d = tmp_path / "index"
    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(d))
    monkeypatch.setattr(embed.config, "EMBED_MODEL", "fake/siglip")
    embed._cache.clear()
    return d


@pytest.fixture()
def fake_video(monkeypatch):
    import cv2 as real_cv2

    holder = {}

    class _Proxy(types.ModuleType):
        def __getattr__(self, name):
            return getattr(real_cv2, name)

    proxy = _Proxy("cv2")
    proxy.VideoCapture = lambda p: holder["cap"]
    monkeypatch.setitem(sys.modules, "cv2", proxy)
    monkeypatch.setattr(dog, "_model", object())
    monkeypatch.setattr(dog, "_load", lambda force=False: None)
    return holder


def _stub_detect(monkeypatch, fn):
    """狗检测桩：同时替换单帧和批量两个入口（sample_video 走批量）。"""
    monkeypatch.setattr(dog, "detect", fn)
    monkeypatch.setattr(dog, "detect_batch", lambda frames, conf=0.35: [fn(f, conf) for f in frames])


BOX = [{"bbox": [0.07, 0.13, 0.15, 0.15], "conf": 0.9}]


def _dog_when(monkeypatch, pred):
    calls = {"n": 0}

    def detect(frame, conf=0.35):
        calls["n"] += 1
        return BOX if pred(calls["n"]) else []
    _stub_detect(monkeypatch, detect)


# ── 建索引 ────────────────────────────────────────────────────────────

def test_建索引_只存有狗的点_t是pts_模型不变就复用(index_dir, fake_video, monkeypatch):
    fake_video["cap"] = _Cap([0, 1000, 2000, 3000, 4000, 5000])
    _dog_when(monkeypatch, lambda n: n != 3)          # 第 3 秒没狗
    enc = FakeEncoder()
    r = embed.build("d/a.mp4", "x.mp4", encoder=enc)
    assert r["cached"] is False and r["n"] == 5 and r["with_dog"] == 5 and r["sampled"] == 6
    assert enc.calls == [("img", 5)]
    d = embed.load("d/a.mp4")
    assert d["t"].tolist() == [0.0, 1.0, 3.0, 4.0, 5.0]
    assert d["emb"].shape == (5, 8) and d["box"].shape == (5, 4)
    assert d["meta"]["model"] == "fake/siglip" and d["meta"]["path"] == "d/a.mp4"
    assert embed.has_index("d/a.mp4") and not embed.has_index("d/b.mp4")
    assert np.allclose(np.linalg.norm(d["emb"], axis=1), 1.0, atol=1e-2)

    # 再建：直接用旧的，不解码不编码
    fake_video["cap"] = _Cap([0])
    r2 = embed.build("d/a.mp4", "x.mp4", encoder=enc)
    assert r2["cached"] is True and enc.calls == [("img", 5)]
    # 模型换了 → 重建
    monkeypatch.setattr(embed.config, "EMBED_MODEL", "fake/siglip-v2")
    fake_video["cap"] = _Cap([0, 1000])
    _dog_when(monkeypatch, lambda n: True)
    r3 = embed.build("d/a.mp4", "x.mp4", encoder=enc)
    assert r3["cached"] is False and r3["n"] == 2 and embed.load("d/a.mp4")["meta"]["model"] == "fake/siglip-v2"
    # force 也重建
    fake_video["cap"] = _Cap([0])
    assert embed.build("d/a.mp4", "x.mp4", force=True, encoder=enc)["n"] == 1


def test_整段没狗_索引是空的_不炸(index_dir, fake_video, monkeypatch):
    fake_video["cap"] = _Cap([0, 1000, 2000])
    _dog_when(monkeypatch, lambda n: False)
    r = embed.build("d/e.mp4", "x.mp4", encoder=FakeEncoder())
    assert r["n"] == 0 and embed.has_index("d/e.mp4")
    assert embed.search(np.ones(8), ["d/e.mp4"])["hits"] == []


# ── 搜索 ──────────────────────────────────────────────────────────────

def _save(rel, t, emb):
    import json
    import os

    os.makedirs(embed.config.EMBED_INDEX_DIR, exist_ok=True)
    np.savez(embed.index_path(rel), t=np.array(t, dtype="float32"), emb=np.array(emb, dtype="float16"),
             box=np.zeros((len(t), 4), dtype="float32"),
             meta=np.array(json.dumps({"model": "fake/siglip", "path": rel})))


def test_搜索_最像的在前_排掉自己_合段_缺索引报出来(index_dir):
    a = np.array([1, 0, 0, 0], dtype="float32")
    b = np.array([0, 1, 0, 0], dtype="float32")
    ab = (a + b) / np.sqrt(2)
    # 视频 A：10-12 秒像 a，20 秒像 b，30-31 秒一半像
    _save("A.mp4", [10, 11, 12, 20, 30, 31], [a, a, a, b, ab, ab])
    # 视频 B：5 秒像 a
    _save("B.mp4", [5], [a])
    r = embed.search(a, ["A.mp4", "B.mp4", "C.mp4"], top_k=10, gap_s=3.0)
    assert r["searched"] == 2 and r["missing"] == ["C.mp4"]
    assert [(h["path"], h["t"]) for h in r["hits"][:4]] == [("A.mp4", 10.0), ("A.mp4", 11.0), ("A.mp4", 12.0), ("B.mp4", 5.0)]
    assert all(r["hits"][i]["score"] >= r["hits"][i + 1]["score"] for i in range(len(r["hits"]) - 1))
    segs = r["segments"]
    assert segs[0] == {"path": "A.mp4", "start_s": 9.0, "end_s": 13.0, "score": 1.0, "n": 3}
    assert {(s["path"], s["start_s"]) for s in segs} >= {("B.mp4", 4.0), ("A.mp4", 29.0)}
    # min_score 把 b 那个点（分数 0）挡掉
    assert all(h["score"] > 0.5 for h in embed.search(a, ["A.mp4"], min_score=0.5)["hits"])
    # 排掉样例自己：样例在 A 的 11 秒 ±2 秒
    r2 = embed.search(a, ["A.mp4", "B.mp4"], top_k=10, exclude=("A.mp4", 9.0, 13.0))
    assert ("A.mp4", 10.0) not in [(h["path"], h["t"]) for h in r2["hits"]]
    assert r2["hits"][0] == {"path": "B.mp4", "t": 5.0, "score": 1.0}
    # top_k 截断
    assert len(embed.search(a, ["A.mp4", "B.mp4"], top_k=2)["hits"]) == 2


def test_group_hits_按视频分开_间隔超过gap就断():
    hits = [{"path": "x", "t": 1.0, "score": 0.9}, {"path": "x", "t": 2.0, "score": 0.8},
            {"path": "x", "t": 10.0, "score": 0.95}, {"path": "y", "t": 2.5, "score": 0.5}]
    segs = embed.group_hits(hits, gap_s=3.0, pad_s=0.5)
    assert segs == [{"path": "x", "start_s": 9.5, "end_s": 10.5, "score": 0.95, "n": 1},
                    {"path": "x", "start_s": 0.5, "end_s": 2.5, "score": 0.9, "n": 2},
                    {"path": "y", "start_s": 2.0, "end_s": 3.0, "score": 0.5, "n": 1}]
    assert embed.group_hits([]) == []


def test_同一张图建索引和查询_能把自己找回来(index_dir, fake_video, monkeypatch):
    """端到端：假视频 6 帧建索引，再拿第 3 秒那一帧当样例（同一张图），排掉自己后
    最像的应该是别的秒——这里所有帧长得都不一样，只验证流程通、分数区间对。"""
    fake_video["cap"] = _Cap([0, 1000, 2000, 3000, 4000, 5000], moving=lambda t: False)   # 方块不动 → 帧全一样
    _dog_when(monkeypatch, lambda n: True)

    class SameEncoder(FakeEncoder):
        """建索引时的裁图和查询时的裁图 JPEG 质量不同，字节不一样；这里只验流程，
        所以让编码器对"内容一样"的图给同一个向量（按解码后的像素均值算）。"""

        def encode_images(self, jpegs):
            import cv2
            self.calls.append(("img", len(jpegs)))
            out = []
            for b in jpegs:
                img = cv2.imdecode(np.frombuffer(b, dtype="uint8"), cv2.IMREAD_GRAYSCALE)
                m = float(img.mean())
                out.append(self._vec(str(round(m)).encode()))
            return np.stack(out)

    enc = SameEncoder()
    embed.build("d/a.mp4", "x.mp4", encoder=enc)

    class _CapSeek(_Cap):
        def set(self, prop, val):
            self.i = int(round(val / 1000.0)) - 1
            return True

        def read(self):
            self.i += 1
            return self.retrieve()

    fake_video["cap"] = _CapSeek([0, 1000, 2000, 3000, 4000, 5000], moving=lambda t: False)
    q = embed.frame_query("x.mp4", 3.0, encoder=enc)
    assert q["has_dog"] is True
    r = embed.search(q["vec"], ["d/a.mp4"], top_k=10, exclude=("d/a.mp4", 2.0, 4.0))
    assert [h["t"] for h in r["hits"]] == [0.0, 1.0, 5.0]           # 2/3/4 秒被排掉
    assert all(abs(h["score"] - 1.0) < 1e-3 for h in r["hits"])      # 帧全一样 → 分数 1


def test_frame_query_没狗就整帧(fake_video, monkeypatch):
    class _CapSeek(_Cap):
        def set(self, prop, val):
            return True

        def read(self):
            self.i = 0
            return self.retrieve()

    fake_video["cap"] = _CapSeek([0])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [])
    enc = FakeEncoder()
    q = embed.frame_query("x.mp4", 0.0, encoder=enc)
    assert q["has_dog"] is False and q["vec"].shape == (8,) and enc.calls == [("img", 1)]


def test_文本查询走文本编码器(index_dir):
    a = np.array([1, 0, 0, 0], dtype="float32")
    _save("A.mp4", [1], [a])
    enc = FakeEncoder(dim=4)
    v = embed.text_query("dog licking its paw", encoder=enc)
    assert enc.calls == [("txt", 1)] and v.shape == (4,)


# ── 状态 / 接口 ───────────────────────────────────────────────────────

def test_status_不触发加载_预热后如实说没装(monkeypatch, index_dir):
    monkeypatch.setattr(embed, "_model", None)
    monkeypatch.setattr(embed, "_load_error", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    st = embed.status()                       # 没预热：不加载，只说还没加载
    assert st["available"] is False and st["loading"] is False and "还没加载" in st["error"]
    w = embed.warmup()                        # 预热才真去加载
    assert w["warm"] is False and "transformers" in w["error"]
    st = embed.status()
    assert st["available"] is False and "transformers" in st["error"] and st["indexed_videos"] == 0


def test_接口_模型不可用503_可用时走到底(monkeypatch, tmp_path, index_dir):
    from fastapi.testclient import TestClient

    from vision_service import app as appmod

    v = tmp_path / "a.mp4"
    v.write_bytes(b"0")
    monkeypatch.setattr(appmod.config, "VIDEO_ROOT", str(tmp_path))
    with TestClient(appmod.app) as tc:
        monkeypatch.setattr(embed, "status", lambda: {"available": False, "error": "没装"})
        assert tc.post("/api/v1/embed/build", json={"path": "a.mp4"}).status_code == 503
        assert tc.post("/api/v1/embed/search", json={"text": "x", "paths": ["a.mp4"]}).status_code == 503

        monkeypatch.setattr(embed, "status", lambda: {"available": True, "error": None})
        monkeypatch.setattr(dog, "status", lambda: {"available": True, "error": None})
        monkeypatch.setattr(embed, "build", lambda rel, full, **kw: {"n": 3, "cached": False, "rel": rel, "force": kw["force"]})
        r = tc.post("/api/v1/embed/build", json={"path": "a.mp4", "force": True})
        assert r.status_code == 200 and r.json()["rel"] == "a.mp4" and r.json()["force"] is True
        assert tc.post("/api/v1/embed/build", json={"path": "../a.mp4"}).status_code == 422

        _save("a.mp4", [1, 2], [[1, 0, 0, 0], [0, 1, 0, 0]])
        assert tc.post("/api/v1/embed/indexed", json={"paths": ["a.mp4", "b.mp4"]}).json() == {"a.mp4": True, "b.mp4": False}

        monkeypatch.setattr(embed, "text_query", lambda text, **kw: np.array([1, 0, 0, 0], dtype="float32"))
        r = tc.post("/api/v1/embed/search", json={"text": "dog", "paths": ["a.mp4", "b.mp4"], "top_k": 5})
        assert r.status_code == 200
        j = r.json()
        assert j["query"]["kind"] == "text" and j["hits"][0]["t"] == 1.0 and j["missing"] == ["b.mp4"]

        monkeypatch.setattr(embed, "frame_query", lambda full, t, **kw: {"vec": np.array([0, 1, 0, 0], dtype="float32"), "has_dog": True, "t": t})
        r = tc.post("/api/v1/embed/search", json={"ref": {"path": "a.mp4", "t": 30}, "paths": ["a.mp4"], "exclude_self_s": 5})
        assert r.status_code == 200 and r.json()["hits"][0]["t"] == 2.0 and r.json()["query"]["has_dog"] is True
        # 样例自己前后 5 秒排掉：t=2 在 30±5 之外所以还在；换个 ref 让它落在里面
        r = tc.post("/api/v1/embed/search", json={"ref": {"path": "a.mp4", "t": 3}, "paths": ["a.mp4"], "exclude_self_s": 5})
        assert [h["t"] for h in r.json()["hits"]] == []
        assert tc.post("/api/v1/embed/search", json={"paths": ["a.mp4"]}).status_code == 422
        assert tc.post("/api/v1/embed/search", json={"text": "x", "ref": {"path": "a.mp4", "t": 1}, "paths": ["a.mp4"]}).status_code == 422


def test_下载进度_百分比和预计时间(monkeypatch):
    monkeypatch.setattr(embed, "_progress", {"done": 0, "total": 0, "started": None, "file": None})
    assert embed.download_progress() is None
    now = {"t": 100.0}
    monkeypatch.setattr(embed.time, "monotonic", lambda: now["t"])
    embed._progress.update({"total": 400_000_000, "started": 100.0, "file": "model.safetensors"})
    now["t"] = 110.0
    embed._progress["done"] = 100_000_000          # 10 秒下了 100MB → 10MB/s，剩 300MB → 30 秒
    p = embed.download_progress()
    assert p["pct"] == 25.0 and p["done_mb"] == 100.0 and p["total_mb"] == 400.0
    assert p["speed_mbps"] == 10.0 and p["eta_s"] == 30 and p["finished"] is False
    embed._progress["done"] = 400_000_000
    assert embed.download_progress()["finished"] is True and embed.download_progress()["eta_s"] == 0
    st = embed.status()
    assert st["progress"]["pct"] == 100.0
