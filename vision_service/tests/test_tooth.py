"""牙齿检测挪到视觉服务：权重不在新地方时从老地方复制（老的不动），两处都没有时报清楚。"""
import os

import pytest

from vision_service import tooth


def test_weights_copied_from_old_place(tmp_path, monkeypatch):
    old = tmp_path / "old" / "best.pt"
    old.parent.mkdir()
    old.write_bytes(b"weights")
    new = tmp_path / "models" / "tooth" / "best.pt"
    monkeypatch.setattr(tooth, "WEIGHTS", str(new))
    monkeypatch.setattr(tooth, "OLD_WEIGHTS", str(old))
    assert tooth._ensure_weights() == str(new)
    assert new.read_bytes() == b"weights" and old.exists()      # 复制，不是移动
    assert tooth.status()["copied_from"] == str(old)


def test_missing_weights_explained(tmp_path, monkeypatch):
    monkeypatch.setattr(tooth, "WEIGHTS", str(tmp_path / "nope.pt"))
    monkeypatch.setattr(tooth, "OLD_WEIGHTS", str(tmp_path / "nope_old.pt"))
    with pytest.raises(FileNotFoundError, match="models/vision/tooth"):
        tooth._ensure_weights()
    st = tooth.status()
    assert st["available"] is False and "权重不存在" in st["error"]
    assert not os.path.exists(tmp_path / "nope.pt")


def test_registered_on_models_page():
    from vision_service import models
    assert "tooth" in models.REGISTRY
    assert models.REGISTRY["tooth"]["status"]()["available"] in (True, False)
