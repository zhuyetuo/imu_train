"""几何粗筛 → 大模型判断：用索引里的框裁帧、部位选项只给相关的、统计要分开"看不清"和"没命中"。"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from vision_service import embed, partask, posepart, seek
from vision_service.pose import DIM, K, NOSE, PAWS


def _row(dists):
    v = np.zeros(DIM, dtype="float32")
    xy = np.stack([np.linspace(-0.4, 0.4, K), np.linspace(0.4, -0.4, K)], axis=1).astype("float32")
    for k in (NOSE, *PAWS):
        v[k * 2:k * 2 + 2] = xy[k]
        v[DIM - K + k] = 1.0
    v[posepart.D0: posepart.D0 + posepart.N_DIST] = dists
    return v / np.linalg.norm(v)


def _index(tmp_path, rel, ts, boxes):
    meta = {"path": rel, "pose": True, "model": embed.config.EMBED_MODEL, "masked": False}
    np.savez(embed.index_path(rel), t=np.asarray(ts, dtype="float32"),
             emb=np.zeros((len(ts), 2), dtype="float16"),
             box=np.asarray(boxes, dtype="float32"),
             pose=np.stack([_row([0.9, 0.9, 0.15, 0.9, 0.9, 0.9])] * len(ts)).astype("float16"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    embed._cache.clear()


def test_裁帧用索引里存的框_不重跑检测(tmp_path, monkeypatch):
    """框就在索引里（建索引时那一次检测的结果）。重跑一次既慢，又可能跟当初
    判断用的框不一样——拿一张跟判断依据不一致的图去问，问出来的结果没意义。"""
    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    rel = "data_raw/2026_9_14_gouchang/a_cam1_imu1_raw.mp4"
    _index(tmp_path, rel, [10, 11, 12, 13], [[0.25, 0.25, 0.75, 0.75]] * 4)

    def fake_iter(path, every, start_s=0.0, end_s=None):
        for t in (10.0, 11.0, 12.0, 13.0):
            if t >= start_s - 0.6 and (end_s is None or t <= end_s + 0.6):
                yield t, np.full((200, 200, 3), int(t) * 10, np.uint8)
    monkeypatch.setattr(seek, "iter_frames", fake_iter)
    monkeypatch.setattr(partask.seek, "iter_frames", fake_iter)

    detected = []
    monkeypatch.setattr(partask, "embed", embed)
    from vision_service import dog
    monkeypatch.setattr(dog, "detect", lambda *a, **kw: detected.append(1) or [])

    frames = partask.frames_around("/nas/a.mp4", rel, 11.5, n=4, span_s=3.0)
    assert len(frames) == 4 and not detected              # 一次检测都没跑
    img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    # 框是 0.25~0.75（100x100），四周各留 25% → 150x150
    assert img.shape[0] == 150 and img.shape[1] == 150

    assert partask.frames_around("/nas/a.mp4", "没建过索引.mp4", 11.5) == []


def test_部位选项只给相关的_不给全表():
    """候选本来就是"鼻子够到后爪"筛出来的，再让模型从 37 个部位里选等于把粗筛的
    信息扔了。左右两个都给：四爪全可见只有 58%，几何判的左右本来就不牢。"""
    ls = partask.labels_for("后爪", ["舔", "啃"])
    assert [l.name for l in ls] == ["舔", "啃"]
    assert ls[0].parts == ["后左爪", "后右爪"] and ls[0].description
    assert partask.labels_for("后右爪", ["舔"])[0].parts == ["后右爪"]


def test_问一条_取不到帧就跳过而不是硬问(tmp_path, monkeypatch):
    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    rel = "data_raw/d/a_cam1_imu1_raw.mp4"
    _index(tmp_path, rel, [10, 11], [[0.2, 0.2, 0.8, 0.8]] * 2)
    root = tmp_path / "nas"
    (root / "data_raw" / "d").mkdir(parents=True)
    (root / "data_raw" / "d" / "a_cam1_imu1_raw.mp4").write_bytes(b"0")

    hit = {"path": rel, "t": 10.0, "dist": 0.1, "slot": "后左爪"}
    # 视频不在：明说，不去问
    r = partask.ask_one(hit, "后爪", [], None, n_frames=4, span_s=3.0, video_root="/不存在")
    assert "视频不在" in r["skipped"]

    monkeypatch.setattr(partask.seek, "iter_frames", lambda *a, **kw: iter(()))
    r = partask.ask_one(hit, "后爪", [], None, n_frames=4, span_s=3.0, video_root=str(root))
    assert "取不到帧" in r["skipped"]

    # 正常问：候选本身的字段要跟着回答一起带出来，否则结果对不回是哪一条
    def fake_iter(path, every, start_s=0.0, end_s=None):
        yield 10.0, np.full((100, 100, 3), 100, np.uint8)
    monkeypatch.setattr(partask.seek, "iter_frames", fake_iter)
    monkeypatch.setattr(partask.seek, "ask", lambda *a, **kw: {
        "label": "舔", "body_part": "后左爪", "confidence": 0.8, "see": "clear",
        "desc": "侧卧口鼻接触左后肢", "note": "有往复", "usage": {"input": 9, "output": 3}})
    r = partask.ask_one(hit, "后爪", [], None, n_frames=4, span_s=3.0, video_root=str(root))
    assert r["path"] == rel and r["t"] == 10.0 and r["label"] == "舔"
    assert r["n_frames"] == 1 and r["usage"]["input"] == 9


def test_统计_看不清和没命中是两回事():
    """unclear 说的不是模型好不好，是送进去的候选好不好；命中率说的才是几何粗筛
    的真实精度。混在一起看的话，一批 none 回来不知道该改哪边。"""
    res = [
        {"path": "a.mp4", "t": 1, "see": "clear", "label": "舔", "body_part": "后左爪",
         "confidence": 0.8, "desc": "d1", "note": "n1", "usage": {"input": 100, "output": 10}},
        {"path": "b.mp4", "t": 2, "see": "clear", "label": None, "confidence": 0.0,
         "desc": "趴着睡", "note": "", "usage": {"input": 100, "output": 10}},
        {"path": "c.mp4", "t": 3, "see": "unclear", "label": None, "confidence": 0.0,
         "desc": "太暗", "note": "", "usage": {"input": 100, "output": 10}},
        {"path": "d.mp4", "t": 4, "skipped": "视频不在"},
    ]
    txt = partask.summarize(res)
    assert "问了 3/4 条（1 条取不到帧）" in txt
    assert "看不清 1（33%）" in txt and "命中   1（33%）" in txt   # 两个数分开
    assert "舔-后左爪 1" in txt and "d1" in txt and "n1" in txt    # 依据要打出来
    assert partask.summarize([])                                   # 空的不炸


def test_没key时只dry_run_但钱要估出来并列出几个档位(monkeypatch, capsys, tmp_path):
    """两件事：

    1. 没 key 就别假装问了一圈什么都没命中——那会让人以为是数据的问题。
    2. 但「值不值得花这个钱」这个决定，缺了数字根本没法做。原来没 key 时只说
       "只能 dry-run"，一个数都不给，等于把决定权还给人却不给依据。
    """
    import sys

    monkeypatch.setattr(posepart, "find", lambda *a, **kw: {
        "known": True, "hits": [{"path": "a.mp4", "t": 1.0, "dist": 0.1, "slot": "后左爪"}] * 448})
    monkeypatch.setattr(partask.llmmod, "from_env", lambda: None)
    monkeypatch.setattr(sys, "argv", ["x", "--part", "后爪", "--index-dir", str(tmp_path)])
    partask.main()
    out = capsys.readouterr().out
    assert "672,000" in out                              # 448 × 1500
    for m in partask.config.SEEK_PRICE_PER_M:
        assert m in out and "约 $" in out                # 每个档位都给价
    assert "vision_service/.env" in out and "SEEK_MODEL" in out   # 怎么配
    assert "SEEK_PROVIDER=doubao" in out                          # 非 Claude 那条路也要写出来
