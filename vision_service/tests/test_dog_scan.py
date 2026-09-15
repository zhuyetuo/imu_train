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

def test_默认权重是可配的且没写死在代码里():
    """型号是会换的（yolov8n → yolo11x → yolo26n 都发生在同一天），
    所以代码里不该出现型号字符串，只该出现"读配置"。"""
    import io as _io
    from vision_service import config
    assert config.DOG_WEIGHTS, "总得有个默认值，不然装完不配就用不了"
    # 只找**型号**（yolo 后面跟数字），不找 `from ultralytics import YOLO`——
    # 那是类名，不是型号
    import re as _re
    src = _io.open("vision_service/dog.py", encoding="utf-8").read()
    hits = _re.findall(r"yolo\s*\d+\w*", src, _re.I)
    assert not hits, f"型号不该写死在 dog.py 里（{hits}），只该读 config.DOG_WEIGHTS"


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


# ── dog 是哪个类别号：按名字查，不写死 ──────────────────────────────────
#
# 写死 16 的理由曾经是"这属于 COCO 数据集定义，不会变"。这个理由站不住：
# 16 是不是 dog 取决于**这份权重**的 names 表。换家族、换自训权重、官方调顺序，
# 16 就是别的东西了——而且不报错，只会让「有没有狗」整个失真。

class _M:
    def __init__(self, names):
        self.names = names


@pytest.mark.parametrize("names,expect", [
    ({0: "person", 16: "dog"}, 16),                    # COCO 的常规位置
    ({0: "cat", 1: "dog"}, 1),                         # 自训的小类别表
    ({0: "person", 5: "DOG"}, 5),                      # 大小写
    ({0: "person", 7: " dog "}, 7),                    # 前后空格
    (["person", "bicycle", "dog"], 2),                 # names 是 list 不是 dict
])
def test_按名字从权重里查出类别号(names, expect):
    assert dog._resolve_dog_class(_M(names)) == expect


@pytest.mark.parametrize("names", [{}, None, {0: "cat", 1: "bird"}, {0: "狗"}])
def test_查不到就退回兜底值并出声(names, caplog):
    """退回是对的（总比不干活强），但必须让人知道是在猜——
    不出声的话，一个非 COCO 权重会静默地按 16 过滤出别的类别。"""
    with caplog.at_level("WARNING"):
        assert dog._resolve_dog_class(_M(names)) == dog._DOG_FALLBACK_CLASS
    assert "dog" in caplog.text


def test_scan_用的是查出来的类别号不是写死的(fake_cv2, monkeypatch):
    """真正要守的一条：predict 收到的 classes 得跟查出来的一致。
    查归查、用归用地脱节的话，前面那些测试全是白测。"""
    got = {}

    class M:
        names = {0: "cat", 3: "dog"}

        def predict(self, *a, **kw):
            got.update(kw)
            return []

    m = M()
    monkeypatch.setattr(dog, "_model", m)
    monkeypatch.setattr(dog, "_dog_class", dog._resolve_dog_class(m))
    fake_cv2._cap = FakeCap([0])
    dog.scan_video("x.mp4")
    assert got["classes"] == [3], got


def test_类别号会在_status_里报出来(monkeypatch):
    """万一退回了兜底值，看 status 就该看得见，而不是等结果不对才去查。"""
    monkeypatch.setattr(dog, "_model", object())
    monkeypatch.setattr(dog, "_dog_class", 16)
    monkeypatch.setattr(dog, "_load", lambda force=False: None)
    assert dog.status()["dog_class"] == 16


# ── CUDA 到底用上没有 ───────────────────────────────────────────────────
#
# 原来 /status 里的 device 报的是**配置值**（config.SAM_DEVICE），不是实际用的。
# 而 SAM 加载时有一句"cuda 不可用就退回 cpu"——于是配置写着 cuda、实际跑在
# CPU 上、status 还理直气壮地报 cuda。慢十几倍，一点提示都没有。

def test_没装torch时说清楚是没装(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    r = dog.cuda_report()
    assert r["cuda_available"] is None and "没装 torch" in r["why"]


def _fake_torch(monkeypatch, cuda_build, available, name="NVIDIA GeForce RTX 4090"):
    import sys, types
    t = types.ModuleType("torch")
    t.__version__ = "2.5.1"
    t.version = types.SimpleNamespace(cuda=cuda_build)
    t.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda i: name,
    )
    monkeypatch.setitem(sys.modules, "torch", t)
    return t


def test_装成cpu版torch是最常见的原因(monkeypatch):
    """torch.version.cuda 是 None = CPU 版轮子。这是最常见的、而且从版本号上
    看不出来的原因（2.5.1 和 2.5.1+cpu 有时都显示成 2.5.1）。"""
    _fake_torch(monkeypatch, cuda_build=None, available=False)
    r = dog.cuda_report()
    assert r["cuda_available"] is False
    assert "CPU 版" in r["why"] and "download.pytorch.org" in r["why"]


def test_有cuda版但用不了_指向驱动和容器(monkeypatch):
    """torch 是 GPU 版但 is_available() False——驱动对不上、或者容器没给 --gpus。
    跟上一条的处理办法完全不同，不能笼统说一句"CUDA 不可用"。"""
    _fake_torch(monkeypatch, cuda_build="12.1", available=False)
    r = dog.cuda_report()
    assert "驱动" in r["why"] and "--gpus" in r["why"]
    assert "CPU 版" not in r["why"]


def test_能用的时候报出是哪块卡(monkeypatch):
    _fake_torch(monkeypatch, cuda_build="12.1", available=True)
    r = dog.cuda_report()
    assert r["cuda_available"] is True and r["why"] is None
    assert "4090" in r["gpu"] and r["cuda_build"] == "12.1"


def test_status_报的是实际设备不是配置值(monkeypatch):
    """这条就是那个 bug：配置 cuda、实际 cpu，status 不能还报 cuda。"""
    monkeypatch.setattr(dog, "_model", object())
    monkeypatch.setattr(dog, "_load", lambda force=False: None)
    monkeypatch.setattr(dog, "_device_used", "cpu")
    assert dog.status()["device"] == "cpu"
    monkeypatch.setattr(dog, "_device_used", None)
    assert dog.status()["device"] == "cuda", "读不到实际设备时才退回配置值"


# ── SAM 能用 CUDA 但 YOLO 不行：两边加载方式不一样 ──────────────────────
#
# ultralytics 的 YOLO(权重) 加载到 **CPU**，predict(device="cuda") 才逐次搬；
# SAM 是 build_sam2(..., device=device)，加载时就在卡上。
# 不统一的话：status 读参数永远是 cpu，而且每采一帧就搬一次几十 MB 权重。

class _FakeYOLO:
    def __init__(self, *a, **kw):
        self.names = {16: "dog"}
        self.moved_to = None

    def to(self, dev):
        self.moved_to = dev
        return self

    def predict(self, *a, **kw):
        return []


def _install_fake_yolo(monkeypatch, cuda_ok):
    import sys, types
    ul = types.ModuleType("ultralytics")
    made = {}
    def YOLO(*a, **kw):
        made["m"] = _FakeYOLO()
        return made["m"]
    ul.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ul)
    _fake_torch(monkeypatch, cuda_build="12.1", available=cuda_ok)
    monkeypatch.setattr(dog, "_model", None)
    monkeypatch.setattr(dog, "_load_error", None)
    monkeypatch.setattr(dog, "_device_used", None)
    monkeypatch.setattr(dog, "_last_try", 0.0)
    return made


def test_加载时就把模型搬上卡_不靠每次predict搬(monkeypatch):
    made = _install_fake_yolo(monkeypatch, cuda_ok=True)
    dog._load(force=True)
    assert made["m"].moved_to == "cuda", "没在加载时 .to(cuda) 的话，权重每次 predict 都要搬一趟"
    assert dog._device_used == "cuda"
    assert dog.status()["device"] == "cuda"


def test_cuda用不了就退回cpu而且报出来(monkeypatch):
    """不判断的话，ultralytics 会在每次 predict 里抛错或自己退回——两种都没人看见。"""
    _install_fake_yolo(monkeypatch, cuda_ok=False)
    dog._load(force=True)
    assert dog._device_used == "cpu"
    assert dog.status()["device"] == "cpu"
    assert dog.status()["cuda"]["cuda_available"] is False


def test_predict用的是实际设备不是配置值(fake_cv2, monkeypatch):
    """配置写 cuda、实际退回 cpu 时，还往 predict 里传 cuda 就会当场抛错。"""
    got = {}

    class M:
        names = {16: "dog"}
        def predict(self, *a, **kw):
            got.update(kw)
            return []

    monkeypatch.setattr(dog, "_model", M())
    monkeypatch.setattr(dog, "_dog_class", 16)
    monkeypatch.setattr(dog, "_device_used", "cpu")
    fake_cv2._cap = FakeCap([0])
    dog.scan_video("x.mp4")
    assert got["device"] == "cpu"
