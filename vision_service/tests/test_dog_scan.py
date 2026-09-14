"""
画面里有没有狗。

这一步的产出会直接决定平台上"这段素材要不要人看"，所以两件事错不得：

  1. **时间戳**。结果是要按时间跟 IMU 片段对齐的，差几秒就对到别的行为上去了。
     我们的视频是 VFR（采集端 _FfmpegVfrSink 每个 tick 写一帧），按帧号换算
     必然越往后越偏——所以一律读 PTS。
  2. **"没看成"和"确认没狗"要分得开**。混了的话，一段解码失败的视频会被当成
     "确认没狗"，人直接跳过一整段真有狗的素材。

下面用一个假的 VideoCapture 驱动真的 scan_video——不需要 torch，也不需要真视频。
"""

import sys
import types

import pytest

# 注意别在模块级 stub 掉 cv2：同一次 pytest 里 SAM 那边要用真的 cv2，
# 全局换掉会把它们一起弄挂（第一版就是这么干的，SAM 4 项跟着红了）。
# dog.py 是在 scan_video 里才 import cv2 的，所以只要在 fixture 里临时换就够。
from vision_service import dog


# ── summarize：这几个数直接决定人要不要看 ───────────────────────────────

def _f(t, n):
    return {"t": t, "n_dogs": n, "boxes": [{}] * n}


def test_整段没狗():
    s = dog.summarize([_f(i * 5, 0) for i in range(20)])
    assert s["verdict"] == "no_dog" and s["no_dog_ratio"] == 1.0 and s["max_dogs"] == 0


def test_一直有狗():
    s = dog.summarize([_f(i * 5, 1) for i in range(20)])
    assert s["verdict"] == "has_dog" and s["no_dog_ratio"] == 0.0


def test_大部分时间空镜也要标出来():
    """狗场那种关笼子的场景：一小时里狗只出现几分钟。判成 has_dog 的话人
    还是得从头看到尾，等于这个功能白做。"""
    s = dog.summarize([_f(i, 1 if i < 2 else 0) for i in range(20)])
    assert s["verdict"] == "mostly_empty" and s["no_dog_ratio"] == 0.9


def test_刚过线的不算大部分空镜():
    s = dog.summarize([_f(i, 1 if i < 3 else 0) for i in range(20)])   # 空 85%
    assert s["verdict"] == "mostly_empty"
    s = dog.summarize([_f(i, 1 if i < 5 else 0) for i in range(20)])   # 空 75%
    assert s["verdict"] == "has_dog"


def test_一帧都没采到是_unknown_不是没狗():
    """视频坏了 / 长度为 0 / 解码失败，是"没看成"，不是"确认没狗"。
    返回 no_dog 的话，人会直接跳过一整段可能有狗的素材。"""
    s = dog.summarize([])
    assert s["verdict"] == "unknown"
    assert s["no_dog_ratio"] is None, "没看成就不该给出一个比例，那会被当成结论"
    assert s["sampled"] == 0


def test_最大只数是给身份那一步用的():
    """影棚一个画面里同时有四只狗。'画面里只有 1 只' 这个信息，
    下一步判'要标的那只在不在'时用得上。"""
    s = dog.summarize([_f(0, 1), _f(5, 4), _f(10, 2)])
    assert s["max_dogs"] == 4


# ── scan_video：VFR 的时间戳 ────────────────────────────────────────────

class FakeCap:
    """按给定的 PTS 序列吐帧，模拟 VFR。"""

    def __init__(self, pts_ms):
        self.pts = list(pts_ms)
        self.i = -1
        self.released = False

    def isOpened(self):
        return True

    def grab(self):
        self.i += 1
        return self.i < len(self.pts)

    def get(self, _prop):
        return self.pts[self.i] if self.i < len(self.pts) else 0.0

    def retrieve(self):
        import numpy as np
        return True, np.zeros((8, 8, 3), dtype="uint8")

    def release(self):
        self.released = True


@pytest.fixture()
def fake_cv2(monkeypatch):
    import numpy as np
    cv2 = types.ModuleType("cv2")
    cv2.CAP_PROP_POS_MSEC = 0
    cv2.VideoCapture = lambda p: cv2._cap
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(dog, "_model", object())          # 装作模型在
    monkeypatch.setattr(dog, "_load", lambda force=False: None)
    np  # noqa: B018
    return cv2


def _stub_predict(monkeypatch, per_call):
    """让"模型"按调用次序返回几只狗。"""
    calls = {"n": 0}

    class _T(list):
        """照 ultralytics 的真实形状来：boxes.xyxy[0] 是个 tensor，有 .tolist()。
        桩给个普通 list 的话，测的就是我自己想象的 API，不是真的那个。"""
        def tolist(self):
            return list(self)

    class Box:
        def __init__(self):
            self.xyxy = [_T([1.0, 1.0, 5.0, 5.0])]
            self.conf = _T([0.9])

    class Res:
        def __init__(self, k):
            self.boxes = [Box() for _ in range(k)]

    class M:
        def predict(self, *a, **kw):
            k = per_call[min(calls["n"], len(per_call) - 1)]
            calls["n"] += 1
            return [Res(k)]

    monkeypatch.setattr(dog, "_model", M())
    return calls


def test_时间戳取的是_pts_不是帧号算的(fake_cv2, monkeypatch):
    """VFR 的关键一条。这串 PTS 的间隔是不均匀的（20ms/40ms/80ms 混着），
    按 `帧号 / fps` 算的话采样点会整个偏掉。"""
    pts = [0, 20, 60, 140, 300, 620, 1260, 2540, 5100, 10220]
    fake_cv2._cap = FakeCap(pts)
    _stub_predict(monkeypatch, [1])
    r = dog.scan_video("x.mp4", every_sec=1.0)
    ts = [f["t"] for f in r["frames"]]
    # 每个采样点都必须是某个真实 PTS，而不是 0.0/1.0/2.0 这种整秒
    assert all(round(t * 1000) in pts for t in ts), ts
    assert ts == [0.0, 1.26, 2.54, 5.1, 10.22]
    assert r["duration_sec"] == 10.22, "时长也要按最后一个 PTS，不是帧数除以 fps"


def test_采样间隔是按时间不是按帧数(fake_cv2, monkeypatch):
    """前半段每 10ms 一帧、后半段每 500ms 一帧。按帧数采的话，前半段会被
    密集采样、后半段几乎不采——而后半段一样长。"""
    pts = [i * 10 for i in range(100)] + [1000 + i * 500 for i in range(10)]
    fake_cv2._cap = FakeCap(pts)
    _stub_predict(monkeypatch, [1])
    r = dog.scan_video("x.mp4", every_sec=1.0)
    ts = [f["t"] for f in r["frames"]]
    gaps = [round(b - a, 2) for a, b in zip(ts, ts[1:])]
    assert all(g >= 1.0 for g in gaps), gaps
    assert ts[-1] >= 5.0, "后半段也得采到，不能全挤在前面"


def test_读不出_pts_的帧不会把采样点带回零(fake_cv2, monkeypatch):
    """有些容器最后几帧 POS_MSEC 返回 0。当成"回到 0 秒"的话，后面每一帧
    都会越过采样点，等于把整段又密集扫一遍。"""
    pts = [0, 1000, 2000, 0, 0, 3000]
    fake_cv2._cap = FakeCap(pts)
    _stub_predict(monkeypatch, [1])
    r = dog.scan_video("x.mp4", every_sec=1.0)
    assert [f["t"] for f in r["frames"]] == [0.0, 1.0, 2.0, 3.0]


def test_框是归一化的(fake_cv2, monkeypatch):
    """跟标注那边一个口径：存归一化坐标，换显示尺寸不用重算。"""
    fake_cv2._cap = FakeCap([0])
    _stub_predict(monkeypatch, [1])
    r = dog.scan_video("x.mp4", every_sec=1.0)
    b = r["frames"][0]["boxes"][0]["bbox"]
    assert b == [0.125, 0.125, 0.5, 0.5], b   # (1,1)-(5,5) 在 8x8 上
    assert all(0.0 <= v <= 1.0 for v in b)


def test_有上限防止把卡占一下午(fake_cv2, monkeypatch):
    fake_cv2._cap = FakeCap([i * 10 for i in range(5000)])
    _stub_predict(monkeypatch, [1])
    r = dog.scan_video("x.mp4", every_sec=0.2, max_frames=10)
    assert r["sampled"] == 10


def test_扫完一定释放(fake_cv2, monkeypatch):
    """不释放的话，扫几百个样本就把文件句柄耗光了。"""
    cap = FakeCap([0, 1000])
    fake_cv2._cap = cap
    _stub_predict(monkeypatch, [1])
    dog.scan_video("x.mp4")
    assert cap.released


def test_中途出错也要释放(fake_cv2, monkeypatch):
    cap = FakeCap([0, 1000])
    fake_cv2._cap = cap

    class Boom:
        def predict(self, *a, **kw):
            raise RuntimeError("显存不够了")

    monkeypatch.setattr(dog, "_model", Boom())
    with pytest.raises(RuntimeError):
        dog.scan_video("x.mp4")
    assert cap.released, "抛异常也得 release，不然句柄泄漏"


def test_打不开的视频报得明白(fake_cv2, monkeypatch):
    class Dead(FakeCap):
        def isOpened(self):
            return False

    fake_cv2._cap = Dead([])
    with pytest.raises(ValueError, match="打不开"):
        dog.scan_video("x.mp4")


def test_模型没加载时不静默返回空(monkeypatch):
    """返回一个空结果的话，上面 summarize 会给出 unknown，看着像"视频是空的"——
    实际是模型没装。两种情况要分得开。"""
    monkeypatch.setattr(dog, "_model", None)
    monkeypatch.setattr(dog, "_load_error", "没装 ultralytics：x")
    monkeypatch.setattr(dog, "_load", lambda force=False: None)
    with pytest.raises(RuntimeError, match="ultralytics"):
        dog.scan_video("x.mp4")


# ── 权重选哪个 ──────────────────────────────────────────────────────────

def test_默认不是_nano():
    """这一步要的是召回不是速度：按时间采样，一小时才 720 帧，最大的模型也就
    几十秒跑完。而漏一只狗 = 人跳过一整段真有素材的视频。"""
    from vision_service import config
    assert "n.pt" not in config.DOG_WEIGHTS, f"{config.DOG_WEIGHTS} 是最小那档，召回不够"


def test_换权重只要改环境变量(monkeypatch):
    """代码里不该写死型号——有更新的模型时，换法应该是一个环境变量。"""
    import importlib
    from vision_service import config as c
    monkeypatch.setenv("DOG_WEIGHTS", "yolo12x.pt")
    importlib.reload(c)
    assert c.DOG_WEIGHTS == "yolo12x.pt"
    monkeypatch.delenv("DOG_WEIGHTS")
    importlib.reload(c)


def test_下不动权重时错误里要说怎么办(monkeypatch):
    """最常见的失败是这台机器下不了权重（离线/防火墙）。这时候人需要知道的是
    "换成哪个、怎么换"，不是一句"加载失败"。"""
    import sys, types
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda *a, **kw: (_ for _ in ()).throw(OSError("connection refused"))
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(dog, "_model", None)
    monkeypatch.setattr(dog, "_load_error", None)
    monkeypatch.setattr(dog, "_last_try", 0.0)
    dog._load(force=True)
    assert dog._load_error and "DOG_WEIGHTS" in dog._load_error, dog._load_error
    assert "weights/" in dog._load_error, "要告诉人权重手动放哪儿"
