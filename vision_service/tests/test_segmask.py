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
        assert kw.get("agnostic_nms") is True and kw.get("retina_masks") is True
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
