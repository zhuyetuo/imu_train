"""抠狗涂背景：掩码怎么合、背景涂成灰、没模型就退回、索引 meta 记着抠没抠（不一致就重建）。"""

from __future__ import annotations

import numpy as np
import pytest

from vision_service import segmask


def test_涂背景_狗留着_背景灰():
    frame = np.full((20, 20, 3), 200, dtype="uint8")
    mask = np.zeros((20, 20), dtype="uint8")
    mask[5:15, 5:15] = 1
    out = segmask.apply(frame, mask, feather_px=0)
    assert tuple(out[10, 10]) == (200, 200, 200)
    assert tuple(out[0, 0]) == segmask.BG_COLOR
    out2 = segmask.apply(frame, mask, feather_px=2)           # 羽化：边上是过渡值，中心不变
    assert tuple(out2[10, 10]) == (200, 200, 200) and tuple(out2[0, 0]) == segmask.BG_COLOR


def test_没模型_返回None_不抛(monkeypatch):
    monkeypatch.setattr(segmask, "_model", None)
    monkeypatch.setattr(segmask, "_load_error", "没装")
    monkeypatch.setattr(segmask, "_last_try", 1e18)
    monkeypatch.setattr(segmask, "_loading", False)
    frame = np.zeros((10, 10, 3), dtype="uint8")
    assert segmask.dog_mask(frame) is None
    assert segmask.masked_crop(frame, [{"bbox": [0, 0, 1, 1], "conf": 1}], lambda b, w, h: (0, 0, w, h)) is None
    st = segmask.status()
    assert st["available"] is False and "没装" in st["error"]


class _Res:
    def __init__(self, masks, xyxy):
        class M:
            data = np.asarray(masks)
        self.masks = M() if masks else None

        class B:
            pass
        self.boxes = B()
        self.boxes.xyxy = np.asarray(xyxy, dtype="float32").reshape(-1, 4)


class _Model:
    def __init__(self, res):
        self.res = res

    def predict(self, frame, **kw):
        assert kw.get("agnostic_nms") is True and kw.get("retina_masks") is False   # 原图分辨率掩码太贵，自己 resize
        assert kw.get("half") is False            # half 会让分割头 dtype 对不上，直接崩
        return self.res


def test_掩码_只要跟检测框重叠的实例_几只狗合一起(monkeypatch):
    h = w = 10
    a = np.zeros((h, w)); a[0:3, 0:3] = 1            # 左上那只，跟检测框重叠
    b = np.zeros((h, w)); b[7:10, 7:10] = 1          # 右下：沙发被当成熊，检测框里没它
    monkeypatch.setattr(segmask, "_model", _Model([_Res([a, b], [[0, 0, 3, 3], [7, 7, 10, 10]])]))
    monkeypatch.setattr(segmask.config, "EMBED_MASK_BG", True)
    frame = np.zeros((h, w, 3), dtype="uint8")
    m = segmask.dog_mask(frame, [{"bbox": [0.0, 0.0, 0.3, 0.3], "conf": 0.9}])
    assert m[1, 1] == 1 and m[8, 8] == 0
    m2 = segmask.dog_mask(frame, None)                # 不给框：全都要
    assert m2[1, 1] == 1 and m2[8, 8] == 1
    monkeypatch.setattr(segmask, "_model", _Model([_Res([], [])]))
    assert segmask.dog_mask(frame) is None            # 一个实例都没有


def test_建索引_抠没抠记在meta_不一致就重建(monkeypatch, tmp_path):
    from vision_service import embed, seek

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(embed.config, "EMBED_MODEL", "fake/siglip")
    monkeypatch.setattr(embed.config, "EMBED_MASK_BG", True)
    monkeypatch.setattr(embed.pose, "available", lambda: False)
    embed._cache.clear()
    monkeypatch.setattr(seek, "sample_video", lambda *a, **kw: [{"t": 1.0, "boxes": [{"bbox": [0, 0, 1, 1], "conf": 1}],
                                                                  "jpeg": b"x", "motion": None}])

    class Enc:
        def encode_images(self, jpegs):
            return np.ones((len(jpegs), 4), dtype="float32")

    enc = Enc()
    monkeypatch.setattr(segmask, "available", lambda: False)
    r = embed.build("A.mp4", "/x/A.mp4", encoder=enc)
    assert r["cached"] is False and r["masked"] is False
    assert embed.build("A.mp4", "/x/A.mp4", encoder=enc)["cached"] is True
    # 分割模型上线了 → 老索引是没抠的，自动重建
    monkeypatch.setattr(segmask, "available", lambda: True)
    r2 = embed.build("A.mp4", "/x/A.mp4", encoder=enc)
    assert r2["cached"] is False and r2["masked"] is True
    assert embed.build("A.mp4", "/x/A.mp4", encoder=enc)["cached"] is True
    monkeypatch.setattr(embed.config, "EMBED_MASK_BG", False)
    assert embed.build("A.mp4", "/x/A.mp4", encoder=enc)["cached"] is False    # 关掉开关也重建


def test_建索引_抠狗按批送_不是一张张(monkeypatch, tmp_path):
    """sample_video 攒够一批才喊 on_batch；embed.build 用它一批算掩码、逐张替 JPEG。"""
    from vision_service import embed, seek

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(embed.config, "EMBED_MODEL", "fake/siglip")
    monkeypatch.setattr(embed.config, "EMBED_MASK_BG", True)
    monkeypatch.setattr(embed.pose, "available", lambda: False)
    monkeypatch.setattr(segmask, "available", lambda: True)
    embed._cache.clear()
    calls = []

    def fake_batch(frames, boxes_list, crop_fn, max_side=512):
        calls.append(len(frames))
        return [np.full((8, 8, 3), 9, dtype="uint8") for _ in frames]

    monkeypatch.setattr(segmask, "masked_crop_batch", fake_batch)
    box = [{"bbox": [0.1, 0.1, 0.5, 0.5], "conf": 1}]
    frames = [np.full((40, 40, 3), 77, dtype="uint8") for _ in range(5)]

    def fake_sample(path, every_sec=1.0, conf=0.35, on_frame=None, on_batch=None, **kw):
        recs = [{"t": float(i), "boxes": box, "jpeg": b"orig", "motion": None} for i in range(5)]
        on_batch(list(zip(recs[:3], frames[:3])))
        on_batch(list(zip(recs[3:], frames[3:])))
        return recs

    monkeypatch.setattr(seek, "sample_video", fake_sample)
    seen = []

    class Enc:
        def encode_images(self, jpegs):
            seen.extend(jpegs)
            return np.ones((len(jpegs), 4), dtype="float32")

    r = embed.build("A.mp4", "/x/A.mp4", encoder=Enc())
    assert calls == [3, 2] and r["with_mask"] == 5
    # 索引那一列全是抠完的图（b"orig" 是没抠的）
    assert all(j != b"orig" for j in seen[:1]) and r["masked"] is True
    # 一句话搜那一列用的是**没抠的**那张：文本塔拿自然照片训的，涂灰背景的抠图
    # 不在它见过的分布里。所以这两张都要算一次向量
    assert b"orig" in seen and r["raw"] is True
    d = embed.load("A.mp4")
    assert d["emb_raw"] is not None and len(d["emb_raw"]) == 5


def test_sample_video_攒批喊on_batch(monkeypatch):
    from vision_service import dog, seek
    from vision_service.tests.test_seek import _Cap

    monkeypatch.setattr(dog, "_model", object())
    monkeypatch.setattr(dog, "_load", lambda force=False: None)
    monkeypatch.setattr(seek.config, "STATIC_SKIP_THR", 0.0)
    monkeypatch.setattr(seek.config, "DECODE_FFMPEG", False)
    box = [{"bbox": [0.1, 0.1, 0.5, 0.5], "conf": 1}]
    monkeypatch.setattr(dog, "detect_batch", lambda frames, conf=0.35: [box for _ in frames])
    import cv2 as real_cv2
    import sys
    import types

    class _Proxy(types.ModuleType):
        def __getattr__(self, name):
            return getattr(real_cv2, name)
    proxy = _Proxy("cv2")
    proxy.VideoCapture = lambda p: _Cap([i * 1000 for i in range(7)])
    monkeypatch.setitem(sys.modules, "cv2", proxy)
    got = []
    out = seek.sample_video("v.mp4", every_sec=1.0, batch=3, on_batch=lambda items: got.append(len(items)))
    assert len(out) == 7 and got == [3, 3, 1]


def test_分割在裁剪块上跑_掩码按块给_框换算到块里(monkeypatch):
    """整帧 100x100，狗框在右下角；分割只看裁出来的那块，掩码尺寸跟块一样，框也换算到块坐标。"""
    seen = {}

    def fake_mask_batch(crops, boxes_list, imgsz=None):
        seen["shapes"] = [c.shape[:2] for c in crops]
        seen["boxes"] = boxes_list
        return [np.ones(c.shape[:2], dtype="uint8") for c in crops]

    monkeypatch.setattr(segmask, "dog_mask_batch", fake_mask_batch)
    frame = np.full((100, 100, 3), 200, dtype="uint8")
    boxes = [{"bbox": [0.6, 0.6, 0.2, 0.2], "conf": 0.9}]
    crop_fn = lambda b, w, h: (50, 50, 100, 100)
    out = segmask.masked_crop_batch([frame], [boxes], crop_fn, max_side=512)
    assert seen["shapes"] == [(50, 50)]
    bx = seen["boxes"][0][0]["bbox"]
    assert bx == pytest.approx([0.2, 0.2, 0.4, 0.4])      # (60-50)/50 …
    assert out[0].shape == (50, 50, 3) and tuple(out[0][25, 25]) == (200, 200, 200)
    # 抠不到：None，调用方用原图
    monkeypatch.setattr(segmask, "dog_mask_batch", lambda c, b, imgsz=None: [None] * len(c))
    assert segmask.masked_crop(frame, boxes, crop_fn) is None


def test_静止的帧沿用上一帧的抠图_不再送分割_向量只算一次(monkeypatch, tmp_path):
    from vision_service import embed, seek

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(embed.config, "EMBED_MODEL", "fake/siglip")
    monkeypatch.setattr(embed.config, "EMBED_MASK_BG", True)
    monkeypatch.setattr(embed.pose, "available", lambda: False)
    monkeypatch.setattr(segmask, "available", lambda: True)
    embed._cache.clear()
    sent = []

    def fake_batch(frames, boxes_list, crop_fn, max_side=512):
        sent.append(len(frames))
        return [np.full((8, 8, 3), 9 + 40 * len(sent) + i, dtype="uint8") for i, _ in enumerate(frames)]   # 每张不一样

    monkeypatch.setattr(segmask, "masked_crop_batch", fake_batch)
    box = [{"bbox": [0.1, 0.1, 0.5, 0.5], "conf": 1}]
    frames = [np.full((40, 40, 3), 77, dtype="uint8") for _ in range(5)]

    def fake_sample(path, every_sec=1.0, conf=0.35, on_frame=None, on_batch=None, **kw):
        # 第 0 帧真检测，1~3 静止沿用，第 4 帧又变了
        recs = [{"t": float(i), "boxes": box, "jpeg": b"orig%d" % i, "motion": None, "static": i in (1, 2, 3)}
                for i in range(5)]
        on_batch(list(zip(recs, frames)))
        return recs

    monkeypatch.setattr(seek, "sample_video", fake_sample)
    encoded = []

    class Enc:
        def encode_images(self, jpegs):
            encoded.extend(jpegs)
            return np.arange(len(jpegs) * 4, dtype="float32").reshape(len(jpegs), 4)

    r = embed.build("A.mp4", "/x/A.mp4", encoder=Enc())
    assert sent == [2] and r["with_mask"] == 2 and r["static_reused"] == 3
    # 抠图 2 个 + 没抠的那一列 2 个（静止沿用的不重复算）
    assert len(encoded) == 4                                   # 5 帧只算 2+2 个向量
    d = embed.load("A.mp4")
    assert len(d["t"]) == 5 and np.allclose(d["emb"][0], d["emb"][2]) and not np.allclose(d["emb"][0], d["emb"][4])
