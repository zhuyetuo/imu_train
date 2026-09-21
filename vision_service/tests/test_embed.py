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
    # 静止跳检默认关掉：这些测试数"第几次检测"，跳检会让次数对不上；专门测跳检的用例自己打开
    monkeypatch.setattr(seek.config, "STATIC_SKIP_THR", 0.0)
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
    assert enc.calls == [("img", 3)]                  # 5 帧有狗，但假视频里只有 3 种画面：一样的图只算一次向量
    d = embed.load("d/a.mp4")
    assert d["t"].tolist() == [0.0, 1.0, 3.0, 4.0, 5.0]
    assert d["emb"].shape == (5, 8) and d["box"].shape == (5, 4)
    assert d["meta"]["model"] == "fake/siglip" and d["meta"]["path"] == "d/a.mp4"
    assert embed.has_index("d/a.mp4") and not embed.has_index("d/b.mp4")
    assert np.allclose(np.linalg.norm(d["emb"], axis=1), 1.0, atol=1e-2)

    # 再建：直接用旧的，不解码不编码
    fake_video["cap"] = _Cap([0])
    r2 = embed.build("d/a.mp4", "x.mp4", encoder=enc)
    assert r2["cached"] is True and enc.calls == [("img", 3)]
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

def _save(rel, t, emb, raw=True):
    """raw=False 造一份**老索引**（没有原图向量那一列）：一句话搜该跳过它。"""
    import json
    import os

    os.makedirs(embed.config.EMBED_INDEX_DIR, exist_ok=True)
    cols = {"t": np.array(t, dtype="float32"), "emb": np.array(emb, dtype="float16"),
            "box": np.zeros((len(t), 4), dtype="float32"),
            "meta": np.array(json.dumps({"model": "fake/siglip", "path": rel}))}
    if raw:
        # 真实索引里这一列是"没抠背景"的那张算出来的，值跟 emb 不一样；
        # 测试里只要求"有这一列、行数对得上"，所以直接复用 emb
        cols["emb_raw"] = np.array(emb, dtype="float16")
    np.savez(embed.index_path(rel), **cols)


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
    assert r2["hits"][0] == {"path": "B.mp4", "t": 5.0, "score": 1.0, "vis_score": 1.0}
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


def test_transformers_4和5的返回都能取到向量():
    import types

    class T:                       # 装作张量
        def norm(self, **kw):
            return 1
    t = T()
    assert embed._as_tensor(t) is t                                                     # 4.x：直接是张量
    assert embed._as_tensor(types.SimpleNamespace(pooler_output=t, last_hidden_state=None)) is t   # 5.x
    assert embed._as_tensor(types.SimpleNamespace(pooler_output=None, image_embeds=t)) is t


def test_frame_preview_给人看的框和裁剪区_没狗就整帧(fake_video, monkeypatch):
    import base64

    class _CapSeek(_Cap):
        def set(self, prop, val):
            return True

        def read(self):
            self.i = 0
            return self.retrieve()

    fake_video["cap"] = _CapSeek([0])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [{"bbox": [0.4, 0.4, 0.2, 0.2], "conf": 0.9}])
    p = embed.frame_preview("x.mp4", 2.0)
    assert p["has_dog"] is True and p["t"] == 2.0 and p["boxes"][0]["bbox"] == [0.4, 0.4, 0.2, 0.2]
    x1, y1, x2, y2 = p["crop"]
    assert 0 <= x1 < 0.4 and 0.6 < x2 <= 1 and 0 <= y1 < 0.4 and 0.6 < y2 <= 1     # 裁剪区包住框、带边距
    assert base64.b64decode(p["jpeg"])[:2] == b"\xff\xd8" and p["w"] > 0 and p["h"] > 0
    _stub_detect(monkeypatch, lambda f, conf=0.35: [])
    p = embed.frame_preview("x.mp4", 0.0)
    assert p["has_dog"] is False and p["boxes"] == [] and p["crop"] == [0.0, 0.0, 1.0, 1.0]


def test_接口_preview_只要狗检测模型(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from vision_service import app as appmod

    (tmp_path / "a.mp4").write_bytes(b"0")
    monkeypatch.setattr(appmod.config, "VIDEO_ROOT", str(tmp_path))
    with TestClient(appmod.app) as tc:
        monkeypatch.setattr(dog, "status", lambda: {"available": False, "error": "没装"})
        assert tc.post("/api/v1/embed/preview", json={"path": "a.mp4", "t": 3}).status_code == 503
        monkeypatch.setattr(dog, "status", lambda: {"available": True, "error": None})
        monkeypatch.setattr(embed, "frame_preview", lambda full, t, **kw: {"t": t, "has_dog": True, "boxes": [], "crop": [0, 0, 1, 1], "jpeg": "", "w": 1, "h": 1})
        r = tc.post("/api/v1/embed/preview", json={"path": "a.mp4", "t": 3})
        assert r.status_code == 200 and r.json()["t"] == 3.0
        assert tc.post("/api/v1/embed/preview", json={"path": "../a.mp4", "t": 3}).status_code == 422
        monkeypatch.setattr(embed, "frame_preview", lambda full, t, **kw: (_ for _ in ()).throw(ValueError("读不到")))
        assert tc.post("/api/v1/embed/preview", json={"path": "a.mp4", "t": 3}).status_code == 422


def test_去背景_减均值后共同成分不再主导(index_dir):
    """所有帧都带一大坨相同的背景向量 bg，动作差别只在一个很小的分量上：
    原始余弦全部 0.99+ 分不开；减掉均值后，跟样例同动作的排前面、分数拉开。"""
    rng = np.random.default_rng(0)
    bg = np.array([10, 0, 0, 0], dtype="float32")
    lick = np.array([0, 1, 0, 0], dtype="float32")
    sleep = np.array([0, 0, 1, 0], dtype="float32")
    ts, embs = [], []
    for i in range(30):
        act = lick if i % 3 == 0 else sleep
        v = bg + act + rng.normal(0, 0.05, 4).astype("float32")
        ts.append(float(i))
        embs.append(v / np.linalg.norm(v))
    _save("V.mp4", ts, embs)
    q = bg + lick
    raw = embed.search(q, ["V.mp4"], top_k=30, center=False)
    assert raw["centered"] is False and min(h["score"] for h in raw["hits"]) > 0.98   # 分不开
    cen = embed.search(q, ["V.mp4"], top_k=30, center=True)
    assert cen["centered"] is True
    # 舔的 10 帧全在前面且分数接近 1；睡觉的残差方向相反、分数为负，被 min_score=0 截掉
    assert {h["t"] for h in cen["hits"]} == {float(i) for i in range(30) if i % 3 == 0}
    assert min(h["score"] for h in cen["hits"]) > 0.9
    cen_all = embed.search(q, ["V.mp4"], top_k=30, min_score=-1.0, center=True)
    assert len(cen_all["hits"]) == 30 and cen_all["hits"][0]["score"] - cen_all["hits"][-1]["score"] > 1.0
    # 帧太少（<20）不做去背景，免得均值本身就是噪声
    _save("W.mp4", [0.0, 1.0], [embs[0], embs[1]])
    assert embed.search(q, ["W.mp4"], center=True)["centered"] is False


def test_缩略图_狗框那一块或整帧带框(fake_video, monkeypatch):
    class _CapSeek(_Cap):
        def set(self, prop, val):
            return True

        def read(self):
            self.i = 0
            return self.retrieve()

    fake_video["cap"] = _CapSeek([0])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [{"bbox": [0.4, 0.4, 0.2, 0.2], "conf": 0.9}])
    import cv2

    crop = cv2.imdecode(np.frombuffer(embed.frame_thumb("x.mp4", 1.0, crop=True), dtype="uint8"), cv2.IMREAD_COLOR)
    full = cv2.imdecode(np.frombuffer(embed.frame_thumb("x.mp4", 1.0, crop=False), dtype="uint8"), cv2.IMREAD_COLOR)
    # 假视频帧很小，裁剪区被最小边长撑到整帧；只验两种都能出图、都不超过 320
    assert max(full.shape[:2]) <= 320 and max(crop.shape[:2]) <= 320
    _stub_detect(monkeypatch, lambda f, conf=0.35: [])
    none = cv2.imdecode(np.frombuffer(embed.frame_thumb("x.mp4", 1.0, crop=True), dtype="uint8"), cv2.IMREAD_COLOR)
    assert none.shape[:2] == full.shape[:2]                                  # 没狗就整帧


def test_接口_thumb(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from vision_service import app as appmod

    (tmp_path / "a.mp4").write_bytes(b"0")
    monkeypatch.setattr(appmod.config, "VIDEO_ROOT", str(tmp_path))
    with TestClient(appmod.app) as tc:
        monkeypatch.setattr(dog, "status", lambda: {"available": True, "error": None})
        monkeypatch.setattr(embed, "frame_thumb", lambda full, t, **kw: b"\xff\xd8jpeg" + str(kw.get("crop")).encode())
        r = tc.get("/api/v1/embed/thumb", params={"path": "a.mp4", "t": 3, "crop": "false"})
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content.endswith(b"False")
        assert tc.get("/api/v1/embed/thumb", params={"path": "../a.mp4", "t": 3}).status_code == 422


def test_缩略图_四种看法(fake_video, monkeypatch):
    """mask = 抠掉背景；raw = 原图那块；pose = 原图画骨架；box = 整帧带框。"""
    class _CapSeek(_Cap):
        def set(self, prop, val):
            return True

        def read(self):
            self.i = 0
            return self.retrieve()

    fake_video["cap"] = _CapSeek([0])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [{"bbox": [0.4, 0.4, 0.2, 0.2], "conf": 0.9}])
    monkeypatch.setattr(embed.config, "EMBED_MASK_BG", True)
    calls = []
    monkeypatch.setattr(embed.segmask, "masked_crop",
                        lambda frame, boxes, crop_fn, max_side=512: (calls.append("mask"), np.full((30, 30, 3), 114, dtype="uint8"))[1])
    monkeypatch.setattr(embed.pose, "available", lambda: True)
    monkeypatch.setattr(embed.pose, "keypoints",
                        lambda frame, boxes: (calls.append("pose"), (np.full((17, 2), 600.0), np.ones(17), (0, 0, 1, 1)))[1])
    import cv2

    def dec(b):
        return cv2.imdecode(np.frombuffer(b, dtype="uint8"), cv2.IMREAD_COLOR)

    m = dec(embed.frame_thumb("x.mp4", 1.0, view="mask"))
    assert calls == ["mask"] and m.shape[:2] == (30, 30)
    raw = dec(embed.frame_thumb("x.mp4", 1.0, view="raw"))
    assert calls == ["mask"] and raw.shape[:2] != (30, 30)            # 不抠
    p = dec(embed.frame_thumb("x.mp4", 1.0, view="pose"))
    assert calls == ["mask", "pose"] and p.shape[:2] == raw.shape[:2]
    box = dec(embed.frame_thumb("x.mp4", 1.0, view="box"))
    assert max(box.shape[:2]) <= 320
    # 老参数还认：crop=True 就是 mask
    dec(embed.frame_thumb("x.mp4", 1.0, crop=True))
    assert calls == ["mask", "pose", "mask"]


def test_画骨架_只画过阈值的点():
    img = np.zeros((100, 100, 3), dtype="uint8")
    kps = np.array([[10, 10]] * 17, dtype="float32")
    kps[2] = [50, 50]                     # 鼻子
    sc = np.zeros(17); sc[2] = 0.9
    embed.draw_pose(img, kps, sc, offset=(0, 0))
    assert img[50, 50].any() and not img[10, 10].any()


def test_建索引记下每一步花了多少秒(monkeypatch, tmp_path):
    """慢的时候要能一眼看出是解码/检测慢，还是姿态、抠狗、算向量慢——
    不然只能去算法机翻日志猜。"""
    import numpy as np

    from vision_service import embed, pose, seek, segmask

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(embed.config, "EMBED_MASK_BG", False)
    monkeypatch.setattr(segmask, "available", lambda: False)
    monkeypatch.setattr(pose, "available", lambda: True)
    monkeypatch.setattr(pose, "frame_descriptor", lambda f, b: [0.1] * pose.DIM)

    frame = np.zeros((16, 16, 3), dtype="uint8")

    def fake_sample(path, every_sec=1.0, conf=0.35, on_frame=None, on_batch=None, stats_out=None, **kw):
        recs = []
        for i in range(3):
            rec = {"t": float(i), "boxes": [{"bbox": [0, 0, 1, 1], "conf": 0.9}],
                   "jpeg": b"j%d" % i, "motion": None, "static": False}
            on_frame(rec, frame)
            recs.append(rec)
        if stats_out is not None:
            stats_out.update({"detected": 3, "skipped": 1})
        return recs
    monkeypatch.setattr(seek, "sample_video", fake_sample)

    class Enc:
        def encode_images(self, imgs):
            return np.ones((len(imgs), 4), dtype="float32")
    r = embed.build("a/b.mp4", "/x/b.mp4", encoder=Enc())
    assert set(r["spent"]) == {"pose", "seg", "embed", "scan"}
    assert r["detected"] == 3 and r["skipped"] == 1
    # 存进索引 meta，事后翻旧索引也能看
    assert embed.load("a/b.mp4")["meta"]["spent"]["pose"] >= 0


def test_一句话搜用没抠背景那一列_老索引跳过并如实说(index_dir):
    """SigLIP 的文本塔是拿自然照片训的，而索引默认存的是"狗抠出来、背景涂灰"的图——
    那种图不在它见过的分布里，文字跟它对不上，一句话搜的分永远在 0.2 上下。
    以图搜图两边都是抠图、同分布，所以那条路 0.8 都有。所以文字走 emb_raw。

    没有那一列的老索引**这次不搜它**：两个空间的分数不可比，混着排出来的名次是假的。
    宁可少搜几路、并且明说，也别给一个看着正常的假名次。
    """
    a = np.array([1, 0, 0, 0], dtype="float32")
    b = np.array([0, 1, 0, 0], dtype="float32")
    _save("new.mp4", [1.0, 2.0], [a, b])                 # 新索引：两列都有
    _save("old.mp4", [1.0], [a], raw=False)              # 老索引：只有抠图那一列

    r = embed.search(a, ["new.mp4", "old.mp4"], center=False, is_text=True)
    assert r["searched"] == 1 and r["old_index"] == 1 and r["text_space"] == "原图"
    assert [h["path"] for h in r["hits"]] == ["new.mp4", "new.mp4"]

    # 以图搜图照旧用抠图那一列，老索引一起搜（那条路本来就不挑）
    r2 = embed.search(a, ["new.mp4", "old.mp4"], center=False)
    assert r2["searched"] == 2 and r2["old_index"] == 0 and r2["text_space"] is None
    assert {h["path"] for h in r2["hits"]} == {"new.mp4", "old.mp4"}


def test_分步耗时汇总_说清慢在哪一步(index_dir, tmp_path, monkeypatch):
    """建索引慢的时候，唯一有用的问题是"慢在哪一步"——四步的代价差着数量级，
    凭感觉调错旋钮只会白慢一遍。"""
    import json
    import os

    os.makedirs(embed.config.EMBED_INDEX_DIR, exist_ok=True)

    def _idx(name, spent, n=3):
        p = os.path.join(embed.config.EMBED_INDEX_DIR, name)
        np.savez(p, t=np.zeros(n, dtype="float32"), emb=np.zeros((n, 4), dtype="float16"),
                 box=np.zeros((n, 4), dtype="float32"),
                 meta=np.array(json.dumps({"model": "m", "spent": spent})))

    _idx("a.npz", {"scan": 4.0, "pose": 12.0, "seg": 2.0, "embed": 2.0})
    _idx("b.npz", {"scan": 6.0, "pose": 18.0, "seg": 3.0, "embed": 3.0})
    # 没记 spent 的老索引：跳过，不拉低平均
    p = os.path.join(embed.config.EMBED_INDEX_DIR, "old.npz")
    np.savez(p, t=np.zeros(1, dtype="float32"), emb=np.zeros((1, 4), dtype="float16"),
             box=np.zeros((1, 4), dtype="float32"), meta=np.array(json.dumps({"model": "m"})))

    r = embed.spent_summary()
    assert r["n"] == 2 and r["total_sec"] == 50.0 and r["per_video_sec"] == 25.0
    assert r["steps"][0]["step"] == "姿态" and r["steps"][0]["pct"] == 60.0
    assert "最重的是「姿态」" in r["note"]


def test_快档只解关键帧_时间用真实PTS_不是第n帧乘每秒(monkeypatch, tmp_path):
    """关键帧的间隔是编码器定的、不均匀（实测 0 / 13.8 / 25.7 / 36.8 / 50.1），
    所以 t 不能沿用「第 n 帧 × every_sec」——那会把 13.8 秒那一帧记成 1 秒，
    跳转过去是完全不同的画面，而且错得无声无息。
    """
    import subprocess

    from vision_service import seek

    w, h = 8, 4
    frames = b"".join(bytes([i]) * (w * h * 3) for i in range(3))

    class P:
        def __init__(self):
            self.stdout = __import__("io").BytesIO(frames)
            self.stderr = __import__("io").BytesIO(b"")

        def wait(self):
            return 0

    cmds = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: (cmds.append(cmd), P())[1])
    monkeypatch.setattr(seek, "_video_size", lambda p: (w, h))
    monkeypatch.setattr(seek, "keyframe_times", lambda p: [0.0, 13.8, 25.72, 36.76])

    got = list(seek.iter_frames_keyframes("x.mp4", hwaccel=True))
    assert [t for t, _ in got] == [0.0, 13.8, 25.72]          # 按真实 PTS 配对
    cmd = cmds[0]
    # -discard nokey 要在 -i 前面：在**拆包**那一步就扔掉非关键帧，解码器根本看不到它们
    assert cmd.index("-discard") < cmd.index("-i") and cmd[cmd.index("-discard") + 1] == "nokey"
    # -vsync 0：既不补帧也不丢帧，不然解出来的帧数跟 ffprobe 数的对不上，配对就错位了
    assert cmd[cmd.index("-vsync") + 1] == "0"

    # 拿不到时间戳就报错，不拿「第 n 帧」凑一个假的出来
    monkeypatch.setattr(seek, "keyframe_times", lambda p: [])
    with pytest.raises(RuntimeError, match="关键帧时间戳"):
        list(seek.iter_frames_keyframes("x.mp4"))


def test_精档是快档的超集_不会被降级(index_dir, monkeypatch):
    """人的用法是「快档全量刷一遍找目标 → 只对要扩的那几路建精档」。
    所以已有精档时再要快档必须原样返回——把 3600 帧的索引换成 290 帧的，
    而且悄无声息，那是纯粹的数据损失。"""
    from vision_service import seek

    monkeypatch.setattr(embed.config, "EMBED_MODEL", "fake/siglip")
    monkeypatch.setattr(embed.pose, "available", lambda: False)
    monkeypatch.setattr(embed.segmask, "available", lambda: False)
    embed._cache.clear()
    seen = []
    monkeypatch.setattr(seek, "sample_video",
                        lambda *a, **kw: (seen.append(kw.get("keyframes_only")),
                                          [{"t": 1.0, "boxes": [{"bbox": [0, 0, 1, 1], "conf": 1}],
                                            "jpeg": b"x", "motion": None}])[1])

    class Enc:
        def encode_images(self, jpegs):
            return np.ones((len(jpegs), 4), dtype="float32")

    enc = Enc()
    assert embed.build("A.mp4", "/x/A.mp4", encoder=enc, mode="fine")["mode"] == "fine"
    assert seen == [False]
    r = embed.build("A.mp4", "/x/A.mp4", encoder=enc, mode="fast")
    assert r["cached"] is True and r["mode"] == "fine"        # 不降级
    assert seen == [False]                                     # 压根没再跑一遍

    # 反过来：已有快档、要精档 → 重建
    embed._cache.clear()
    assert embed.build("B.mp4", "/x/B.mp4", encoder=enc, mode="fast")["mode"] == "fast"
    assert seen[-1] is True
    assert embed.build("B.mp4", "/x/B.mp4", encoder=enc, mode="fine")["cached"] is False
