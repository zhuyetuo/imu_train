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


# ── 加载失败要能自己恢复 ────────────────────────────────────────────────
#
# 审查查出来的：_load_error 一旦缓存就是永久的。最常见的失败是"权重还没下完"，
# 下好之后还要重启进程才能用——运维不会想到这一点，只会觉得服务坏了。

def test_权重出现之后不用重启也能重试(tmp_path, monkeypatch):
    """最常见的失败是"权重还没下完"。下好之后不该还要重启进程——
    运维不会想到这一点，只会觉得服务坏了。

    这台机器上没装 torch，所以加载注定失败；这里验的是**它有没有再试一次**，
    而不是试的结果（结果要有卡有包才看得到）。"""
    ckpt = tmp_path / "还没下.pt"
    monkeypatch.setattr(config, "SAM_CHECKPOINT", str(ckpt))
    sam._model = None
    sam._load_error = None
    sam._last_try = 0.0

    calls = []
    orig = sam._load_locked
    monkeypatch.setattr(sam, "_load_locked", lambda: (calls.append(1), orig())[1])

    sam.status()
    assert len(calls) == 1
    sam.status()
    assert len(calls) == 1, "权重还没出现，不该反复重试"

    ckpt.write_bytes(b"fake")            # 权重到位
    sam.status()
    assert len(calls) == 2, "权重到位了还在念缓存的错误，只能重启进程才能恢复"


def test_退避期内不会每次请求都重试加载(tmp_path, monkeypatch):
    """反过来也不能每次请求都去试——加载失败有时要几秒，
    每个请求都试一遍会把服务拖死。"""
    monkeypatch.setattr(config, "SAM_CHECKPOINT", str(tmp_path / "没有.pt"))
    sam._model = None
    sam._load_error = None
    sam._last_try = 0.0

    calls = []
    orig = sam._load_locked
    monkeypatch.setattr(sam, "_load_locked", lambda: (calls.append(1), orig())[1])

    for _ in range(3):
        sam.status()
    assert len(calls) == 1, f"退避没生效，试了 {len(calls)} 次"


# ── 三个候选里挑哪一个 ──────────────────────────────────────────────────
#
# 2026-09-15 拿真实牙齿照片实测，argmax(score) 四张错三张，而且错得很自信：
#   点在嘴下方的毛 → 整个狗头，score 0.97
#   点在脸颊       → 前景一个物件，score 0.96
#   点在嘴角       → 整个口鼻部，score 0.44
#   点正好在牙上   → 一颗牙，score 0.84
# score 是"有多确信这是一个物体"，不是"这是不是你要的那个"。切整个狗头它当然
# 确信——那本来就是个完整、边界清楚的物体。

from vision_service.sam import pick_mask


def _sh(a):
    return {"area_ratio": a} if a is not None else None


def test_只给点时挑最小的那个():
    """SAM 的三个掩膜大致是 子部件/部件/整体。标牙齿永远要最细那一档。"""
    shapes = [_sh(0.8035), _sh(0.0643), _sh(0.0003)]
    scores = [0.97, 0.44, 0.84]
    assert pick_mask(shapes, scores, has_box=False) == 2


def test_那张切出整个狗头的_按老规则会选错():
    """把实测那一组原样钉下来：argmax(score) 会选中 0.80（整个狗头）。
    没有这条的话，以后有人把挑法改回 score 不会有测试变红。"""
    shapes = [_sh(0.8035), _sh(0.0643), _sh(0.0003)]
    scores = [0.97, 0.44, 0.84]
    assert pick_mask(shapes, scores, has_box=False, prefer="score") == 0
    assert pick_mask(shapes, scores, has_box=False) == 2


def test_给了框就用score挑():
    """框本身已经把歧义消掉了（人已经指明要哪块），这时 score 是可信的；
    再挑最小的话，会在框里挑出一个小碎片。"""
    shapes = [_sh(0.05), _sh(0.004), _sh(0.0001)]
    scores = [0.9, 0.7, 0.3]
    assert pick_mask(shapes, scores, has_box=True) == 0


def test_退化掩膜不参与挑选():
    """mask_to_shapes 对空掩膜返回 None。不排掉的话会选中一个"面积最小"的空掩膜，
    然后 segment 抛"没分割出东西来"——而其实有可用的候选。"""
    shapes = [None, _sh(0.0), _sh(0.004)]
    scores = [0.99, 0.98, 0.5]
    assert pick_mask(shapes, scores, has_box=False) == 2


def test_全都退化时退回score():
    """一个能用的都没有——这时挑谁都一样，别抛异常，交给上层去报"没分割出东西"。"""
    assert pick_mask([None, None, None], [0.1, 0.9, 0.2], has_box=False) == 1


def test_牙龈修整_只留粉红_去掉白牙和黑嘴唇():
    from vision_service.sam import refine_gingiva

    rgb = np.zeros((40, 60, 3), dtype=np.uint8)
    rgb[:, :20] = (245, 240, 230)      # 白牙
    rgb[:, 20:40] = (230, 120, 140)    # 粉红牙龈
    rgb[:, 40:] = (20, 15, 15)         # 黑嘴唇
    mask = np.ones((40, 60), dtype=np.uint8)
    out = refine_gingiva(mask, rgb)
    ys, xs = np.where(out > 0)
    assert xs.min() >= 20 and xs.max() < 40 and len(xs) > 0.8 * 40 * 20


def test_牙龈修整_颜色全过滤掉时退回原样():
    from vision_service.sam import refine_gingiva

    rgb = np.full((20, 20, 3), 250, dtype=np.uint8)     # 一片白
    mask = np.ones((20, 20), dtype=np.uint8)
    assert refine_gingiva(mask, rgb).sum() == 400


def test_挖掉已经标好的牙():
    from vision_service.sam import subtract_polygons

    mask = np.ones((100, 100), dtype=np.uint8)
    out = subtract_polygons(mask, [[[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]]], dilate_px=2)
    assert out[30, 30] == 0 and out[80, 80] == 1
    assert out[19, 30] == 0 and out[15, 30] == 1           # 往外胀了 2 像素
    assert subtract_polygons(mask, []).sum() == 10000


def test_框按最大那块算_不被碎点撑大():
    from vision_service.sam import mask_to_shapes

    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:40, 20:40] = 1        # 主体
    m[80:82, 90:92] = 1        # 远处一小片碎点
    sh = mask_to_shapes(m)
    x, y, w, h = sh["bbox"]
    assert abs(x - 0.2) < 0.02 and abs(y - 0.2) < 0.02 and abs(w - 0.2) < 0.02 and abs(h - 0.2) < 0.02
    assert all(0.19 <= px <= 0.41 and 0.19 <= py <= 0.41 for px, py in sh["polygon"])
