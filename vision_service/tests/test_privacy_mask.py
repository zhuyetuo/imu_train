"""
隐私遮罩：哪几块留、时间戳一帧不挪、框外真的是黑的。

端到端那条用 PyAV 现编一段 VFR 小视频（帧间隔故意不均匀），检测器换成假的，
跑真的 render + verify，再逐帧解出来看：PTS 跟原视频一模一样、框外像素全是 0。
"""

import json
import os

import pytest

from vision_service import config, privacy_mask as pm

P = pm.MaskParams(pad=0.3, hold_s=3.0, merge_s=0.5, every_sec=0.25)


# ── 扩框 ───────────────────────────────────────────────────────────────

def test_扩框_四边各扩_并裁到画面内():
    x1, y1, x2, y2 = pm.pad_box([0.4, 0.4, 0.2, 0.2], 0.3)
    assert (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)) == (0.34, 0.34, 0.66, 0.66)
    assert pm.pad_box([0.0, 0.9, 0.2, 0.2], 0.5) == (0.0, pytest.approx(0.8), pytest.approx(0.3), 1.0)


# ── 每一刻留哪几块 ─────────────────────────────────────────────────────

def test_前后半秒内的框取并集():
    tl = pm.Timeline([(1.0, [[0.1, 0.1, 0.1, 0.1]]), (1.25, [[0.5, 0.5, 0.1, 0.1]])], P)
    rects, how = tl.rects_at(1.1)
    assert how == "det" and len(rects) == 2


def test_漏检沿用之前的框_最多三秒_之后全黑():
    tl = pm.Timeline([(0.0, [[0.1, 0.1, 0.2, 0.2]])] + [(0.25 * i, []) for i in range(1, 40)], P)
    assert tl.rects_at(2.0)[1] == "hold"
    assert tl.rects_at(3.0)[1] == "hold"
    rects, how = tl.rects_at(3.2)
    assert how == "black" and rects == []


def test_一开始就没狗_整帧全黑():
    tl = pm.Timeline([(0.25 * i, []) for i in range(10)], P)
    assert tl.rects_at(1.0) == ([], "black")


def test_多只狗各留一块():
    tl = pm.Timeline([(1.0, [[0.0, 0.0, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]])], P)
    assert len(tl.rects_at(1.0)[0]) == 2


def test_块外涂黑():
    np = pytest.importorskip("numpy")
    img = np.full((100, 200, 3), 200, dtype=np.uint8)
    out = pm.apply_mask(img, [(0.25, 0.5, 0.5, 1.0)])
    assert out[60, 60].tolist() == [200, 200, 200]
    assert out[10, 10].tolist() == [0, 0, 0] and out[60, 150].tolist() == [0, 0, 0]
    assert pm.apply_mask(img, []).max() == 0


# ── 路径 ───────────────────────────────────────────────────────────────

def test_路径不能穿越_也不能对遮罩版再遮(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "VIDEO_ROOT", str(tmp_path))
    with pytest.raises(ValueError):
        pm._paths("../etc/passwd.mp4")
    with pytest.raises(ValueError):
        pm._paths(f"{pm.MASK_DIR}/a.mp4")
    src, dst, meta = pm._paths("2026_11_11/a.mp4")
    assert dst == os.path.join(str(tmp_path), pm.MASK_DIR, "2026_11_11", "a.mp4") and meta == dst + ".json"


# ── 端到端：真编码，时间戳一帧不挪 ──────────────────────────────────────

def _make_vfr(path, pts_ms):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from fractions import Fraction

    with av.open(path, "w", format="mp4") as c:
        st = c.add_stream("libx264", rate=25)
        st.width, st.height, st.pix_fmt = 64, 48, "yuv420p"
        st.time_base = Fraction(1, 1000)
        st.codec_context.time_base = Fraction(1, 1000)
        for ms in pts_ms:
            f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), 180, dtype=np.uint8), format="rgb24")
            f.pts, f.time_base = ms, Fraction(1, 1000)
            for pkt in st.encode(f):
                c.mux(pkt)
        for pkt in st.encode(None):
            c.mux(pkt)


def test_端到端_时间戳不挪_框外全黑(monkeypatch, tmp_path):
    av = pytest.importorskip("av")
    monkeypatch.setattr(config, "VIDEO_ROOT", str(tmp_path))
    os.makedirs(tmp_path / "d", exist_ok=True)
    # 帧间隔故意不均匀（VFR）：40/40/120/40/200/...
    pts = [0, 40, 80, 200, 240, 440, 480, 520, 1000, 1040, 5200, 5240]
    _make_vfr(str(tmp_path / "d" / "a.mp4"), pts)

    def fake_detector(src, p, progress=None):
        # 0~1 秒有一只狗在左上角，之后再没检到
        return [(0.25 * i, [[0.0, 0.0, 0.25, 0.25]] if i <= 4 else []) for i in range(25)]

    info = pm.process("d/a.mp4", P, detector=fake_detector)
    assert info["ok"] and info["frames"] == len(pts)
    out = tmp_path / pm.MASK_DIR / "d" / "a.mp4"
    with av.open(str(out)) as c:
        st = c.streams.video[0]
        frames = [(round(float(f.pts * f.time_base) * 1000), f.to_ndarray(format="rgb24")) for f in c.decode(st)]
    assert [ms for ms, _ in frames] == pts, "每一帧的时间戳必须跟原视频一样"
    t0 = frames[0][1]
    assert t0[2, 2].mean() > 100, "狗框里那块要留着"
    assert t0[40, 60].max() < 30, "框外要涂黑"
    # 5.2 秒那两帧：最后一次有框是 1.0 秒，过了 3 秒 → 整帧全黑
    assert frames[-1][1].max() < 30
    assert info["ratio"]["black"] > 0
    # 说明写好了，状态是 done；原视频一改就不算数
    assert pm.read_info("d/a.mp4") is not None
    assert pm.status(["d/a.mp4"])["d/a.mp4"]["state"] == "done"
    os.utime(tmp_path / "d" / "a.mp4", (1, 1))
    assert pm.read_info("d/a.mp4") is None
    assert json.load(open(str(out) + ".json", encoding="utf-8"))["masked_rel"] == f"{pm.MASK_DIR}/d/a.mp4"


def test_对不上就不交付(monkeypatch, tmp_path):
    pytest.importorskip("av")
    monkeypatch.setattr(config, "VIDEO_ROOT", str(tmp_path))
    _make_vfr(str(tmp_path / "b.mp4"), [0, 40, 80])
    monkeypatch.setattr(pm, "verify", lambda stats, dst: "帧数对不上：测试")
    with pytest.raises(RuntimeError):
        pm.process("b.mp4", P, detector=lambda s, p, progress=None: [(0.0, [])])
    assert not os.path.exists(tmp_path / pm.MASK_DIR / "b.mp4")
    assert pm.read_info("b.mp4") is None


def test_队列_多路同时做_插队的先做(monkeypatch, tmp_path):
    import threading
    import time as _t

    monkeypatch.setattr(config, "VIDEO_ROOT", str(tmp_path))
    for n in "abcd":
        (tmp_path / f"{n}.mp4").write_bytes(b"x")
    monkeypatch.setattr(pm, "WORKERS", 2)
    gate = threading.Event()
    started: list[str] = []
    lock = threading.Lock()

    def fake_process(rel, p, progress=None, detector=None, check=None):
        with lock:
            started.append(rel)
        gate.wait(5)
        return {}

    monkeypatch.setattr(pm, "process", fake_process)
    import queue as _queue

    # 每个测试一条新队列、一批新 worker（上一个测试留下的 worker 卡在旧队列上，不来抢）
    monkeypatch.setattr(pm, "_q", _queue.PriorityQueue())
    monkeypatch.setattr(pm, "_workers", [])
    pm._state.clear()
    pm.submit(["a.mp4", "b.mp4", "c.mp4"], P)
    for _ in range(100):
        if len(started) >= 2:
            break
        _t.sleep(0.02)
    assert sorted(started) == ["a.mp4", "b.mp4"], "两路同时开工"
    # c 还在排队：d 插队，空出位置时先做 d
    pm.submit(["d.mp4"], P, front=True)
    gate.set()
    for _ in range(100):
        if len(started) >= 4:
            break
        _t.sleep(0.02)
    assert started[2] == "d.mp4" and started[3] == "c.mp4"


def test_暂停_继续_停止(monkeypatch, tmp_path):
    import threading
    import time as _t

    monkeypatch.setattr(config, "VIDEO_ROOT", str(tmp_path))
    for n in "abc":
        (tmp_path / f"{n}.mp4").write_bytes(b"x")
    monkeypatch.setattr(pm, "WORKERS", 1)
    steps = {"a.mp4": 0}
    started: list[str] = []

    def fake_process(rel, p, progress=None, detector=None, check=None):
        started.append(rel)
        for _ in range(200):
            check()
            steps[rel] = steps.get(rel, 0) + 1
            _t.sleep(0.01)
        return {}

    monkeypatch.setattr(pm, "process", fake_process)
    import queue as _queue

    # 每个测试一条新队列、一批新 worker（上一个测试留下的 worker 卡在旧队列上，不来抢）
    monkeypatch.setattr(pm, "_q", _queue.PriorityQueue())
    monkeypatch.setattr(pm, "_workers", [])
    pm._state.clear()
    pm.resume()
    pm.submit(["a.mp4", "b.mp4", "c.mp4"], P)
    _t.sleep(0.2)
    pm.pause()
    _t.sleep(0.6)
    n = steps["a.mp4"]
    _t.sleep(0.6)
    assert steps["a.mp4"] == n, "暂停了还在跑"
    assert pm.status(["a.mp4"])["a.mp4"]["state"] == "running"
    # 停止：正在做的 a 中断、排队的 b 拿掉；c 留着
    r = pm.cancel(["a.mp4", "b.mp4"])
    assert r == {"removed": 1, "stopping": 1}
    pm.resume()
    for _ in range(200):
        if "c.mp4" in started:
            break
        _t.sleep(0.02)
    assert started[:2] == ["a.mp4", "c.mp4"], "b 停掉了不该再做"
    assert pm.status(["a.mp4"])["a.mp4"]["state"] == "none"
    pm.cancel()
