"""姿态关键点那一路：描述子是纯函数，假点就能测；索引里存不存、搜索混不混，用桩模型验。"""

from __future__ import annotations

import numpy as np
import pytest

from vision_service import embed, pose
from vision_service.tests.test_embed import fake_video, index_dir  # noqa: F401 复用那边的夹具


def _kps(**over):
    """一只"标准狗"：脖子 (100,100)，尾根 (300,100)，四爪在下面，鼻子在脖子前面。"""
    k = np.zeros((17, 2), dtype="float32")
    k[2] = (60, 110)      # 鼻子
    k[3] = (100, 100)     # 脖子
    k[4] = (300, 100)     # 尾根
    k[7] = (110, 200)     # 左前爪
    k[10] = (130, 200)    # 右前爪
    k[13] = (280, 200)    # 左后爪
    k[16] = (300, 200)    # 右后爪
    for i, v in over.items():
        k[int(i)] = v
    return k


def test_描述子_归一化_平移缩放不变_够到哪只爪分得开():
    sc = np.ones(17, dtype="float32")
    box = (40, 80, 320, 220)
    a = np.array(pose.descriptor(_kps(), sc, box))
    assert a.shape == (pose.DIM,) and abs(np.linalg.norm(a) - 1) < 1e-5
    # 整体平移 + 放大 2 倍，框也跟着 → 向量一样
    k2 = _kps() * 2 + 500
    box2 = tuple(v * 2 + 500 for v in box)
    b = np.array(pose.descriptor(k2, sc, box2))
    assert np.allclose(a, b, atol=1e-5)
    # 鼻子贴到左前爪 vs 贴到右后爪：两个向量明显不同
    lick_fl = np.array(pose.descriptor(_kps(**{"2": (110, 195)}), sc, box))
    lick_rr = np.array(pose.descriptor(_kps(**{"2": (300, 195)}), sc, box))
    assert float(lick_fl @ lick_rr) < float(lick_fl @ np.array(pose.descriptor(_kps(**{"2": (112, 190)}), sc, box)))
    # 看不见的点：坐标 0、可见位 0、距离 0
    sc2 = sc.copy()
    sc2[7] = 0.1
    c = np.array(pose.descriptor(_kps(), sc2, box))
    assert c[7 * 2] == 0 and c[7 * 2 + 1] == 0 and c[17 * 2] == 0 and c[17 * 2 + 6 + 7] == 0


def test_没模型时status如实说_可用性为假(monkeypatch):
    monkeypatch.setattr(pose.config, "POSE_ONNX", "/nonexistent/x.onnx")
    monkeypatch.setattr(pose, "_loaded", False)
    monkeypatch.setattr(pose, "_model", None)
    monkeypatch.setattr(pose, "_error", None)
    assert pose.available() is False
    st = pose.status()
    assert st["available"] is False and "权重" in st["error"]
    assert pose.frame_descriptor(np.zeros((10, 10, 3), dtype="uint8"), [{"bbox": [0, 0, 1, 1], "conf": 0.9}]) is None


def test_搜索_有姿态就按权重混_没姿态的帧只看画面(index_dir, monkeypatch):
    import json
    import os

    # 两帧画面向量一样，姿态不一样；第三帧没姿态（全 0）
    v = np.array([1, 0, 0, 0], dtype="float32")
    pa = np.zeros(pose.DIM, dtype="float32"); pa[0] = 1
    pb = np.zeros(pose.DIM, dtype="float32"); pb[1] = 1
    os.makedirs(embed.config.EMBED_INDEX_DIR, exist_ok=True)
    np.savez(embed.index_path("P.mp4"), t=np.array([1, 2, 3], dtype="float32"),
             emb=np.array([v, v, v], dtype="float16"), box=np.zeros((3, 4), dtype="float32"),
             pose=np.array([pa, pb, np.zeros(pose.DIM)], dtype="float16"),
             meta=np.array(json.dumps({"model": "fake/siglip", "path": "P.mp4", "pose": True})))
    r = embed.search(v, ["P.mp4"], top_k=10, center=False, pose_vec=pa, pose_w=0.5)
    assert r["pose_used"] is True and r["pose_w"] == 0.5
    by_t = {h["t"]: h for h in r["hits"]}
    assert by_t[1.0]["score"] == pytest.approx(1.0, abs=1e-3)         # 画面 1、姿态 1
    assert by_t[2.0]["score"] == pytest.approx(0.5, abs=1e-3)         # 画面 1、姿态 0
    assert by_t[3.0]["score"] == pytest.approx(1.0, abs=1e-3)         # 没姿态 → 只看画面
    assert by_t[1.0]["pose_score"] == pytest.approx(1.0, abs=1e-3)
    # 权重 0 / 没给样例姿态 → 不混
    r0 = embed.search(v, ["P.mp4"], top_k=10, center=False, pose_vec=pa, pose_w=0)
    assert r0["pose_used"] is False and all(h["score"] == pytest.approx(1.0, abs=1e-3) for h in r0["hits"])
    r1 = embed.search(v, ["P.mp4"], top_k=10, center=False, pose_vec=None)
    assert r1["pose_used"] is False
    # 老索引没有 pose 列也能搜
    np.savez(embed.index_path("Q.mp4"), t=np.array([1], dtype="float32"), emb=np.array([v], dtype="float16"),
             box=np.zeros((1, 4), dtype="float32"), meta=np.array(json.dumps({"model": "fake/siglip", "path": "Q.mp4"})))
    r2 = embed.search(v, ["Q.mp4"], top_k=10, center=False, pose_vec=pa, pose_w=0.5)
    assert r2["pose_used"] is False and embed.load("Q.mp4")["pose"] is None


def test_建索引_姿态可用就每帧存一条(index_dir, fake_video, monkeypatch):
    from vision_service.tests.test_embed import FakeEncoder, _Cap, _dog_when

    fake_video["cap"] = _Cap([0, 1000, 2000], moving=lambda t: True)
    _dog_when(monkeypatch, lambda n: True)
    monkeypatch.setattr(pose, "available", lambda: True)
    calls = []

    def fake_desc(frame, boxes):
        calls.append(len(boxes))
        d = [0.0] * pose.DIM
        d[len(calls) % pose.DIM] = 1.0
        return d

    monkeypatch.setattr(pose, "frame_descriptor", fake_desc)
    r = embed.build("d/p.mp4", "x.mp4", encoder=FakeEncoder())
    assert r["with_pose"] == r["with_dog"] == len(calls) > 0
    d = embed.load("d/p.mp4")
    assert d["pose"].shape == (r["n"], pose.DIM) and d["meta"]["pose"] is True
