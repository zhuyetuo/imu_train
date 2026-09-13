"""
vision_service 能在没有 GPU、没装 sam2、没有权重的机器上测的那部分。

测不了的是 SAM 本身（要卡要权重），这里不假装测了。能测、也最该测的是三件事：

  1. 掩膜 → 框/多边形的换算。算错不报错，只会让平台上的框整体偏掉。
  2. 路径沙箱。这个服务直接按请求里的路径读文件，`..` 穿越必须在这里挡住。
  3. **没有 SAM 时的降级行为**。模型装不上是常态（新机器、忘了下权重），
     这时候必须返回 503 + 说清楚原因，而不是 500 或者把整个服务拖挂——
     平台那边就靠这个把按钮置灰。
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vision_service import app as app_mod
from vision_service import config, sam

client = TestClient(app_mod.app)


# ── 掩膜换算 ────────────────────────────────────────────────────────────

def test_掩膜转成归一化的框():
    m = np.zeros((100, 200), dtype=np.uint8)
    m[20:40, 50:100] = 1  # y 20-39, x 50-99
    got = sam.mask_to_shapes(m)
    # x/y/w/h 归一化：50/200, 20/100, 50/200, 20/100
    assert got["bbox"] == pytest.approx([0.25, 0.2, 0.25, 0.2], abs=1e-6)
    assert got["area_ratio"] == pytest.approx(20 * 50 / (100 * 200), abs=1e-6)


def test_空掩膜返回空而不是崩掉():
    assert sam.mask_to_shapes(np.zeros((10, 10), dtype=np.uint8)) is None


def test_多边形点都在_0_1_之间():
    m = np.zeros((80, 120), dtype=np.uint8)
    m[10:70, 20:100] = 1
    got = sam.mask_to_shapes(m)
    assert got["polygon"], "装了 cv2 就该给出多边形"
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in got["polygon"])
    assert len(got["polygon"]) >= 3


def test_只取最大的那块连通域():
    """SAM 偶尔会在反光处多吐一小片，跟着进去就是脏数据。
    多边形只取最大块；框仍然覆盖全部前景（这是 numpy 取极值的定义），
    所以这条测的是多边形不会被那一小片带跑。"""
    m = np.zeros((100, 100), dtype=np.uint8)
    m[10:60, 10:60] = 1   # 大块
    m[90:95, 90:95] = 1   # 反光那一小片
    got = sam.mask_to_shapes(m)
    xs = [p[0] for p in got["polygon"]]
    ys = [p[1] for p in got["polygon"]]
    assert max(xs) < 0.7 and max(ys) < 0.7, "多边形被那一小片带跑了"


def test_单像素掩膜也能出框():
    m = np.zeros((50, 50), dtype=np.uint8)
    m[25, 25] = 1
    got = sam.mask_to_shapes(m)
    assert got["bbox"][2] > 0 and got["bbox"][3] > 0, "宽高不能是 0，否则平台那边会判非法"


# ── 路径沙箱 ────────────────────────────────────────────────────────────

@pytest.fixture()
def material(tmp_path, monkeypatch):
    root = tmp_path / "mat"
    (root / "口腔验证").mkdir(parents=True)
    (root / "口腔验证" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (tmp_path / "secret.txt").write_text("不该被读到")
    monkeypatch.setattr(config, "MATERIAL_ROOT", str(root))
    return root


@pytest.mark.parametrize("bad", ["../secret.txt", "口腔验证/../../secret.txt", "/etc/passwd", "口腔验证/没有这个.jpg"])
def test_越界路径被挡住(material, bad):
    r = client.post("/api/v1/sam/segment", json={"path": bad, "points": [{"x": 0.5, "y": 0.5}]})
    assert r.status_code == 422, r.text


def test_合法路径能过沙箱这一关(material, monkeypatch):
    """过了沙箱之后才轮到 SAM。这里把 SAM 标成不可用，验的是"沙箱放行了"
    而不是"分割成功了"——分割要卡要权重，这台机器上测不了。"""
    monkeypatch.setattr(sam, "status", lambda: {"available": False, "error": "测试里没有模型"})
    r = client.post("/api/v1/sam/segment", json={"path": "口腔验证/a.jpg", "points": [{"x": 0.5, "y": 0.5}]})
    assert r.status_code == 503, r.text
    assert "没有模型" in r.json()["detail"]


# ── 降级 ────────────────────────────────────────────────────────────────

def test_没装_sam_时状态如实说不可用():
    """这台机器上本来就没有 sam2 和权重，正好是真实的降级场景。"""
    st = sam.status()
    assert st["available"] is False
    assert st["error"], "不可用就必须说清楚为什么，否则运维只能猜"


def test_健康检查任何时候都能回(material):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert "sam" in r.json()


def test_模型不可用时是_503_不是_500(material):
    """503 = 服务在但这个能力暂时没有，平台据此置灰按钮；
    500 会被当成 bug 弹红叉。这个区别决定了标注员看到的是「SAM 没开」
    还是「系统出错了」。"""
    r = client.post("/api/v1/sam/segment", json={"path": "口腔验证/a.jpg", "points": [{"x": 0.5, "y": 0.5}]})
    assert r.status_code == 503


def test_点的坐标必须是归一化的(material):
    r = client.post("/api/v1/sam/segment", json={"path": "口腔验证/a.jpg", "points": [{"x": 640, "y": 480}]})
    assert r.status_code == 422, "像素坐标应该被 schema 拦下来——两边对不齐迟早错一次"
