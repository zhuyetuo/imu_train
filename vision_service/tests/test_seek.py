"""
画面找片段。

要守住的：
  1. 时间是 PTS 不是帧号（候选按时间跟 IMU 对，差几秒就对错动作）。
  2. 门槛真的在控花费：没狗 / 不动 / 超过上限的窗不会送去问模型。
  3. 模型答什么都不会炸：不在候选里的类别、部位乱填、非 JSON，一律当 none。
  4. dry_run 一个 API 调用都不发。
  5. 合并：相邻同类合一段，置信度取最大，部位取多数。

视频用一个假的 VideoCapture 喂真的 numpy 帧，所以裁图、缩放、JPEG 编码走的是
真 cv2；狗检测和大模型客户端换成桩。
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from vision_service import dog, llm as llmmod, seek
from vision_service.seek import Label, Window

CLAUDE = llmmod.LLM("anthropic", "m", "k", price_in=5.0, price_out=25.0)


# ── 假视频 ────────────────────────────────────────────────────────────

class _Cap:
    """PTS 列表驱动；每帧是 720p，"狗"是左上角一块会动的亮方块。"""

    def __init__(self, pts_ms, moving=None, w=1280, h=720):
        self.pts = list(pts_ms)
        self.i = -1
        self.w, self.h = w, h
        self.moving = moving if moving is not None else (lambda t: True)

    def isOpened(self):
        return True

    def grab(self):
        self.i += 1
        return self.i < len(self.pts)

    def get(self, _prop):
        return self.pts[self.i]

    def retrieve(self):
        t = self.pts[self.i] / 1000.0
        img = np.zeros((self.h, self.w, 3), dtype="uint8")
        # 动的话方块位置随时间挪；不动就固定
        dx = int(t * 20) % 80 if self.moving(t) else 0
        img[100:200, 100 + dx:250 + dx] = 200
        return True, img

    def release(self):
        pass


@pytest.fixture()
def fake_video(monkeypatch):
    import cv2 as real_cv2

    holder = {}

    class _CV2Proxy(types.ModuleType):
        def __getattr__(self, name):
            return getattr(real_cv2, name)

    proxy = _CV2Proxy("cv2")
    proxy.VideoCapture = lambda p: holder["cap"]
    monkeypatch.setitem(sys.modules, "cv2", proxy)
    monkeypatch.setattr(dog, "_model", object())
    monkeypatch.setattr(dog, "_load", lambda force=False: None)
    # 静止跳检默认关掉：这些测试数"第几次检测"，跳检会让次数对不上；专门测跳检的用例自己打开
    monkeypatch.setattr(seek.config, "STATIC_SKIP_THR", 0.0)
    return holder


def _dog_at(monkeypatch, has_dog):
    """狗检测桩：has_dog(t_sec) 决定这一帧有没有狗。"""
    def detect(frame, conf=0.35):
        # 框就是那块亮方块所在的大概位置（归一化）
        return [{"bbox": [0.07, 0.13, 0.15, 0.15], "conf": 0.9}] if has_dog(_dog_at.t) else []
    _stub_detect(monkeypatch, detect)


def _stub_detect(monkeypatch, fn):
    """狗检测桩：同时替换单帧和批量两个入口（sample_video 走批量）。"""
    monkeypatch.setattr(dog, "detect", fn)
    monkeypatch.setattr(dog, "detect_batch", lambda frames, conf=0.35: [fn(f, conf) for f in frames])


# ── crop_rect ─────────────────────────────────────────────────────────

def test_crop_rect_留边_撑到最小边_贴边():
    r = seek.crop_rect([{"bbox": [0.5, 0.5, 0.05, 0.05]}], 1000, 1000, margin=0.25, min_side=224)
    x1, y1, x2, y2 = r
    assert x2 - x1 >= 224 and y2 - y1 >= 224                  # 小狗撑到最小边
    assert x1 < 500 < x2 and y1 < 500 < y2                    # 围着狗
    r2 = seek.crop_rect([{"bbox": [0.0, 0.0, 0.05, 0.05]}], 1000, 1000)
    assert r2[0] == 0 and r2[1] == 0                          # 贴边不越界
    r3 = seek.crop_rect([{"bbox": [0.1, 0.1, 0.2, 0.2]}, {"bbox": [0.6, 0.6, 0.2, 0.2]}], 1000, 1000)
    assert r3[0] <= 100 and r3[2] >= 800                      # 多只狗取并集


# ── 采样 ──────────────────────────────────────────────────────────────

def test_采样按_pts_每秒一张_有狗才裁图(fake_video, monkeypatch):
    # PTS 不均匀：0.0 0.3 1.1 1.9 2.05 3.2 ... 按帧号算会错
    pts = [0, 300, 1100, 1900, 2050, 3200, 4010, 5500]
    fake_video["cap"] = _Cap(pts)
    seen = []

    def detect(frame, conf=0.35):
        seen.append(len(seen))
        return [{"bbox": [0.07, 0.13, 0.15, 0.15], "conf": 0.9}] if len(seen) != 3 else []
    _stub_detect(monkeypatch, detect)

    s = seek.sample_video("x.mp4", every_sec=1.0)
    # 跨过采样点才取（1.1 之后下一个点是 ≥2.1，2.05 不取；3.2 之后 ≥4.2，4.01 不取），时间是 PTS
    assert [r["t"] for r in s] == [0.0, 1.1, 3.2, 5.5]
    assert s[2]["jpeg"] is None and s[2]["boxes"] == []             # 第三次检测没狗 → 不裁
    assert all(r["jpeg"] is not None for i, r in enumerate(s) if i != 2)
    assert s[0]["motion"] is None                                    # 第一张没得比
    assert s[1]["motion"] is not None and s[1]["motion"] > 0         # 方块在动
    assert s[3]["motion"] is None                                    # 前一张没狗，断掉重新比
    import cv2
    img = cv2.imdecode(np.frombuffer(s[0]["jpeg"], dtype="uint8"), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) <= 512                                 # 缩到 max_side


def test_采样_start_end_只取区间(fake_video, monkeypatch):
    fake_video["cap"] = _Cap([i * 1000 for i in range(20)])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [])
    s = seek.sample_video("x.mp4", every_sec=1.0, start_s=5, end_s=8)
    assert [r["t"] for r in s] == [5.0, 6.0, 7.0, 8.0]


# ── 选窗 ──────────────────────────────────────────────────────────────

def _samples(n, dog_at=lambda t: True, motion=0.05):
    out = []
    for t in range(n):
        has = dog_at(t)
        out.append({"t": float(t), "boxes": [{}] if has else [],
                    "jpeg": b"x" if has else None,
                    "motion": (motion(t) if callable(motion) else motion) if has and t > 0 else None})
    return out


def test_选窗_没狗和不动的窗不送():
    s = _samples(30, dog_at=lambda t: t < 12 or t >= 24, motion=lambda t: 0.05 if t < 12 else 0.001)
    wins = seek.pick_windows(s, clip_s=6, stride_s=3, motion_min=0.02)
    starts = [w.start for w in wins]
    assert starts and all(st + 6 <= 12 + 1e-6 for st in starts), starts   # 12-24 没狗；24-30 不动
    assert all(w.dog_frac >= 0.8 for w in wins)


def test_选窗_上限按动作量取最强():
    s = _samples(60, motion=lambda t: t / 60.0)
    wins = seek.pick_windows(s, clip_s=6, stride_s=3, max_clips=3, motion_max=1.0)
    assert len(wins) == 3
    assert wins == sorted(wins, key=lambda w: w.start)              # 返回仍按时间排
    assert min(w.motion for w in wins) > 0.7                         # 挑的是动作最大的几个


def test_选窗_动得太厉害也不要():
    s = _samples(30, motion=0.9)
    assert seek.pick_windows(s, motion_max=0.5) == []


def test_选窗_空输入():
    assert seek.pick_windows([]) == []


# ── 问模型：提示词与解析 ─────────────────────────────────────────────

LABELS = [Label("舔身体", "用舌头舔自己", ["前肢爪", "后肢臀尾"]), Label("蹭身体", "在地上/墙上蹭")]


def test_提示词包含类别_描述_部位():
    system, user = seek.build_prompt(LABELS, 6, 6)
    assert "none" in system
    for s in ("舔身体", "用舌头舔自己", "前肢爪", "蹭身体", "输出格式", "label"):
        assert s in user


@pytest.mark.parametrize("text,label,part,conf", [
    ('{"label":"舔身体","body_part":"前肢爪","confidence":0.8,"note":"x"}', "舔身体", "前肢爪", 0.8),
    ('前面废话 {"label":"舔身体","body_part":"脑袋","confidence":"0.6"} 后面废话', "舔身体", None, 0.6),  # 部位不在表里
    ('{"label":"蹭身体","body_part":"前肢爪","confidence":1.7}', "蹭身体", None, 1.0),   # 别的类别的部位不算；conf 截到 1
    ('{"label":"抓挠","confidence":0.9}', None, None, 0.0),                               # 不在候选里
    ('{"label":"none","confidence":0.9}', None, None, 0.0),
    ('不是 json', None, None, 0.0),
    ('{"label": 5}', None, None, 0.0),
])
def test_解析回答_一律不炸(text, label, part, conf):
    a = seek.parse_answer(text, LABELS)
    assert (a["label"], a["body_part"]) == (label, part)
    assert a["confidence"] == pytest.approx(conf)


class _FakeClient:
    """照 anthropic SDK 的形状：client.messages.create(...) → 有 .content / .usage。"""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

        outer = self

        class _Msgs:
            def create(self, **kw):
                outer.calls.append(kw)
                text = outer.answers.pop(0) if outer.answers else '{"label":"none"}'
                if isinstance(text, Exception):
                    raise text
                blk = types.SimpleNamespace(type="text", text=text)
                usage = types.SimpleNamespace(input_tokens=1000, output_tokens=20)
                return types.SimpleNamespace(content=[blk], usage=usage)
        self.messages = _Msgs()


def test_ask_把帧和文字都送了_并返回用量():
    c = _FakeClient(['{"label":"舔身体","body_part":"前肢爪","confidence":0.7,"note":"n"}'])
    a = seek.ask([b"\xff\xd8a", b"\xff\xd8b"], LABELS, 6, CLAUDE, client=c)
    assert a["label"] == "舔身体" and a["usage"] == {"input": 1000, "output": 20}
    kw = c.calls[0]
    assert kw["model"] == "m"
    content = kw["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "image", "text"]
    assert content[0]["source"]["media_type"] == "image/jpeg"
    assert "舔身体" in content[-1]["text"] and "舔身体" in kw["system"] or "none" in kw["system"]


# ── 合并 ──────────────────────────────────────────────────────────────

def _w(s, e):
    return Window(start=s, end=e, motion=0.1, dog_frac=1.0)


def test_合并_相邻同类合一段_置信取最大_部位取多数():
    wins = [_w(0, 6), _w(3, 9), _w(6, 12), _w(30, 36), _w(33, 39)]
    ans = [
        {"label": "舔身体", "body_part": "前肢爪", "confidence": 0.6},
        {"label": "舔身体", "body_part": "前肢爪", "confidence": 0.9},
        {"label": "舔身体", "body_part": "后肢臀尾", "confidence": 0.7},
        {"label": "蹭身体", "body_part": None, "confidence": 0.8},
        {"label": None, "body_part": None, "confidence": 0.0},
    ]
    segs = seek.merge_segments(wins, ans, min_conf=0.5)
    assert [(s["start_s"], s["end_s"], s["label"]) for s in segs] == [(0, 12, "舔身体"), (30, 36, "蹭身体")]
    assert segs[0]["confidence"] == 0.9 and segs[0]["body_part"] == "前肢爪" and segs[0]["n_clips"] == 3
    assert segs[1]["body_part"] is None
    assert "_parts" not in segs[0]


def test_合并_低置信和不同类不合():
    wins = [_w(0, 6), _w(3, 9), _w(6, 12)]
    ans = [{"label": "舔身体", "body_part": None, "confidence": 0.9},
           {"label": "舔身体", "body_part": None, "confidence": 0.3},
           {"label": "蹭身体", "body_part": None, "confidence": 0.9}]
    segs = seek.merge_segments(wins, ans, min_conf=0.5)
    assert [(s["start_s"], s["end_s"], s["label"]) for s in segs] == [(0, 6, "舔身体"), (6, 12, "蹭身体")]


# ── 整条 ──────────────────────────────────────────────────────────────

def test_dry_run_不调模型_但给出会送多少段(fake_video, monkeypatch):
    fake_video["cap"] = _Cap([i * 1000 for i in range(30)])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [{"bbox": [0.07, 0.13, 0.15, 0.15], "conf": 0.9}])
    c = _FakeClient([])
    r = seek.seek_video("x.mp4", LABELS, dry_run=True, llm=CLAUDE, client=c, motion_min=0.0)
    assert r["dry_run"] and r["segments"] == [] and c.calls == []
    assert r["stats"]["clips_candidate"] == len(r["windows"]) > 0
    assert r["stats"]["clips_sent"] == 0 and r["stats"]["usage"]["est_usd"] == 0


def test_整条_送去问_合成片段_并算花费(fake_video, monkeypatch):
    fake_video["cap"] = _Cap([i * 1000 for i in range(30)])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [{"bbox": [0.07, 0.13, 0.15, 0.15], "conf": 0.9}])
    # 前三段说舔、其余 none；其中一段抛异常也不能让整条挂
    answers = ['{"label":"舔身体","body_part":"前肢爪","confidence":0.8}'] * 3 + [RuntimeError("boom")]
    c = _FakeClient(answers)
    r = seek.seek_video("x.mp4", LABELS, llm=CLAUDE, client=c, motion_min=0.0, concurrency=1, max_clips=50)
    assert r["stats"]["llm"] == {"provider": "anthropic", "model": "m"}
    assert not r["dry_run"]
    assert r["stats"]["clips_sent"] == len(c.calls) == len(r["windows"]) > 4
    assert r["stats"]["errors"] == 1 and r["stats"]["hits"] == 3
    assert r["segments"] and r["segments"][0]["label"] == "舔身体" and r["segments"][0]["body_part"] == "前肢爪"
    n = r["stats"]["clips_sent"] - 1
    assert r["stats"]["usage"] == {"input": 1000 * n, "output": 20 * n,
                                   "est_usd": round(1000 * n / 1e6 * 5 + 20 * n / 1e6 * 25, 4)}
    # 每次调用单独一条（平台落表做统计）：token / 耗时 / 成败 / 对应哪一段
    calls = r["stats"]["calls"]
    assert len(calls) == r["stats"]["clips_sent"]
    assert sum(1 for c in calls if not c["ok"]) == 1 and all("latency_ms" in c and c["latency_ms"] >= 0 for c in calls)
    good = [c for c in calls if c["ok"]]
    assert all(c["input"] == 1000 and c["output"] == 20 and c["est_usd"] == round(1000 / 1e6 * 5 + 20 / 1e6 * 25, 4) for c in good)
    assert all(c["end_s"] > c["start_s"] for c in calls)
    # 每段送的帧数不超过 n_frames，且都是 JPEG
    for kw in c.calls:
        imgs = [b for b in kw["messages"][0]["content"] if b["type"] == "image"]
        assert 1 <= len(imgs) <= 6


def test_没有窗时不问也不炸(fake_video, monkeypatch):
    fake_video["cap"] = _Cap([i * 1000 for i in range(10)])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [])
    c = _FakeClient([])
    r = seek.seek_video("x.mp4", LABELS, llm=CLAUDE, client=c)
    assert r["segments"] == [] and c.calls == [] and r["stats"]["with_dog"] == 0


def test_没带_llm_也没环境变量_真跑要报错_dry_run不用(fake_video, monkeypatch):
    fake_video["cap"] = _Cap([i * 1000 for i in range(10)])
    _stub_detect(monkeypatch, lambda f, conf=0.35: [])
    monkeypatch.setattr(seek.config, "ANTHROPIC_API_KEY", "")
    with pytest.raises(RuntimeError, match="llm"):
        seek.seek_video("x.mp4", LABELS)
    assert seek.seek_video("x.mp4", LABELS, dry_run=True)["dry_run"] is True


# ── 接口 ──────────────────────────────────────────────────────────────

def test_status_没_key_如实说(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", types.ModuleType("anthropic"))   # 装作 SDK 在
    monkeypatch.setattr(seek.config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(dog, "status", lambda: {"available": True})
    st = seek.status()
    assert st["available"] is False and "ANTHROPIC_API_KEY" in (st["error"] or "")


def test_接口_狗检测不可用给503_dry_run不需要key(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from vision_service import app as appmod

    v = tmp_path / "a.mp4"
    v.write_bytes(b"0")
    monkeypatch.setattr(appmod.config, "VIDEO_ROOT", str(tmp_path))
    body = {"path": "a.mp4", "labels": [{"name": "舔身体"}], "dry_run": True}
    with TestClient(appmod.app) as tc:
        monkeypatch.setattr(dog, "status", lambda: {"available": False, "error": "没装"})
        assert tc.post("/api/v1/seek", json=body).status_code == 503

        monkeypatch.setattr(dog, "status", lambda: {"available": True, "error": None})
        monkeypatch.setattr(seek, "seek_video", lambda *a, **kw: {"segments": [], "windows": [], "stats": {}, "dry_run": kw["dry_run"]})
        monkeypatch.setattr(seek, "status", lambda: {"available": False, "error": "没配 ANTHROPIC_API_KEY"})
        r = tc.post("/api/v1/seek", json=body)
        assert r.status_code == 200 and r.json()["dry_run"] is True           # dry_run 不要 key
        r = tc.post("/api/v1/seek", json=body | {"dry_run": False})
        assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
        # 请求里带了 llm 就不看环境变量；没 key 是 422；瞎写的提供方 422
        got = {}
        monkeypatch.setattr(seek, "seek_video", lambda *a, **kw: got.update(kw) or {"segments": [], "windows": [], "stats": {}, "dry_run": False})
        llm = {"provider": "gemini", "model": "gemini-2.5-flash", "api_key": "g", "price_in": 0.3}
        r = tc.post("/api/v1/seek", json=body | {"dry_run": False, "llm": llm})
        assert r.status_code == 200 and got["llm"].provider == "gemini" and got["llm"].price_in == 0.3
        r = tc.post("/api/v1/seek", json=body | {"dry_run": False, "llm": llm | {"api_key": ""}})
        assert r.status_code == 422
        r = tc.post("/api/v1/seek", json=body | {"llm": llm | {"provider": "baidu"}})
        assert r.status_code == 422
        assert tc.post("/api/v1/seek", json=body | {"path": "../x.mp4"}).status_code == 422


# ── 解码快路 / 批量检测 ────────────────────────────────────────────────

def test_ffmpeg_抽帧_按时间编号_两种都不行退回cv2(monkeypatch, fake_video):
    import subprocess

    w, h = 64, 32
    frames = [bytes([i]) * (w * h * 3) for i in range(3)]

    class FakeProc:
        def __init__(self, ok):
            self.stdout = __import__("io").BytesIO(b"".join(frames) if ok else b"")
            self.stderr = __import__("io").BytesIO(b"" if ok else b"cuda not available")
            self._ok = ok
        def wait(self):
            return 0 if self._ok else 1

    calls = []

    def popen(cmd, **kw):
        calls.append(cmd)
        return FakeProc(ok="-hwaccel" not in cmd)          # NVDEC 那次失败，软解成功
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(seek, "_video_size", lambda p: (w, h))
    monkeypatch.setattr(seek, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(seek.config, "DECODE_HWACCEL", True)

    got = list(seek.iter_frames("x.mp4", every_sec=2.0, start_s=10.0))
    assert [t for t, _ in got] == [10.0, 12.0, 14.0]
    assert got[1][1].shape == (h, w, 3) and int(got[1][1][0, 0, 0]) == 1
    assert len(calls) == 2 and "-hwaccel" in calls[0] and "-hwaccel" not in calls[1]
    assert "fps=1/2.0" in calls[1] and calls[1][calls[1].index("-ss") + 1] == "10.000"

    # 两种都不行 → cv2
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: FakeProc(ok=False))
    fake_video["cap"] = _Cap([0, 1000, 2000])
    got = list(seek.iter_frames("x.mp4", every_sec=1.0))
    assert [t for t, _ in got] == [0.0, 1.0, 2.0]


def test_sample_video_分批送检测(fake_video, monkeypatch):
    fake_video["cap"] = _Cap([i * 1000 for i in range(10)])
    monkeypatch.setattr(seek, "ffmpeg_available", lambda: False)
    monkeypatch.setattr(seek.config, "STATIC_SKIP_THR", 0.0)        # 只验分批
    sizes = []

    def detect_batch(frames, conf=0.35):
        sizes.append(len(frames))
        return [[{"bbox": [0.07, 0.13, 0.15, 0.15], "conf": 0.9}] for _ in frames]
    monkeypatch.setattr(dog, "detect_batch", detect_batch)
    s = seek.sample_video("x.mp4", every_sec=1.0, batch=4)
    assert len(s) == 10 and sizes == [4, 4, 2]                     # 10 帧分三批
    assert s[1]["motion"] is not None and all(r["jpeg"] for r in s)


def test_detect_batch_形状跟单帧一致(monkeypatch):
    import numpy as np

    class _T(list):
        def tolist(self):
            return list(self)

    class Box:
        xyxy = [_T([10.0, 20.0, 30.0, 60.0])]
        conf = _T([0.8])

    class Res:
        boxes = [Box()]

    class M:
        def predict(self, imgs, **kw):
            assert isinstance(imgs, list) and len(imgs) == 2
            return [Res(), Res()]
    monkeypatch.setattr(dog, "_model", M())
    f = np.zeros((100, 200, 3), dtype="uint8")
    out = dog.detect_batch([f, f])
    assert out == [[{"bbox": [0.05, 0.2, 0.1, 0.4], "conf": 0.8}]] * 2
    assert dog.detect_batch([]) == []


def test_静止跳检_画面没变沿用上次的框_动了照送(fake_video, monkeypatch):
    # 0~4 秒方块不动，5~9 秒动
    fake_video["cap"] = _Cap([i * 1000 for i in range(10)], moving=lambda t: t >= 5)
    monkeypatch.setattr(seek, "ffmpeg_available", lambda: False)
    monkeypatch.setattr(seek.config, "STATIC_SKIP_THR", 0.008)
    sent = []

    def detect_batch(frames, conf=0.35):
        sent.append(len(frames))
        return [[{"bbox": [0.07, 0.13, 0.15, 0.15], "conf": 0.9}] for _ in frames]
    monkeypatch.setattr(dog, "detect_batch", detect_batch)
    s = seek.sample_video("x.mp4", every_sec=1.0, batch=4)
    assert len(s) == 10 and all(r["boxes"] for r in s)               # 跳过的也带着框
    assert sum(sent) < 10                                             # 静止那几秒没送
    assert s[1]["motion"] == 0.0 and s[6]["motion"] > 0               # 静止时 motion 0，动了有值
    # 关掉优化：全送
    sent.clear()
    monkeypatch.setattr(seek.config, "STATIC_SKIP_THR", 0.0)
    seek.sample_video("x.mp4", every_sec=1.0, batch=4)
    fake_video["cap"] = _Cap([i * 1000 for i in range(10)], moving=lambda t: t >= 5)
    sent.clear()
    seek.sample_video("x.mp4", every_sec=1.0, batch=4)
    assert sum(sent) == 10


def test_静止跳检_狗那一小块动了也要送(fake_video, monkeypatch):
    """狗只占画面 0.5%：整帧平均差远低于阈值，但狗那块在动 → 必须送检测。"""
    import numpy as np

    class _TinyDog(_Cap):
        def retrieve(self):
            t = self.pts[self.i] / 1000.0
            img = np.zeros((self.h, self.w, 3), dtype="uint8")
            dx = int(t * 3) % 6 if t >= 3 else 0            # 3 秒后一块 20x20 的小方块在挪
            img[100:120, 100 + dx:120 + dx] = 200
            return True, img

    fake_video["cap"] = _TinyDog([i * 1000 for i in range(8)])
    monkeypatch.setattr(seek, "ffmpeg_available", lambda: False)
    monkeypatch.setattr(seek.config, "STATIC_SKIP_THR", 0.008)
    sent = []

    def detect_batch(frames, conf=0.35):
        sent.append(len(frames))
        return [[{"bbox": [0.07, 0.13, 0.03, 0.04], "conf": 0.9}] for _ in frames]
    monkeypatch.setattr(dog, "detect_batch", detect_batch)
    s = seek.sample_video("x.mp4", every_sec=1.0, batch=2)
    assert len(s) == 8
    # 0 秒送检；1、2 秒跟 0 秒一样 → 跳；3 秒起每秒都跟上一帧不同 → 每帧都送。至少 6 帧
    assert sum(sent) >= 6


# ── 走画面索引：不解码不检测，预览秒出 ───────────────────────────────

def _save_index(rel, t, emb, boxes=None):
    import json
    import os

    import numpy as np

    from vision_service import embed

    os.makedirs(embed.config.EMBED_INDEX_DIR, exist_ok=True)
    box = np.array(boxes if boxes is not None else [[0.07, 0.13, 0.22, 0.28]] * len(t), dtype="float32")
    np.savez(embed.index_path(rel), t=np.array(t, dtype="float32"), emb=np.array(emb, dtype="float16"),
             box=box, meta=np.array(json.dumps({"model": "fake/siglip", "path": rel})))


def test_从索引取样本_动作量是向量距离(tmp_path, monkeypatch):
    import numpy as np

    from vision_service import embed

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    embed._cache.clear()
    a = np.array([1, 0, 0, 0], dtype="float32")
    b = np.array([0, 1, 0, 0], dtype="float32")
    # 0~4 秒不动（向量一样），5 秒突然变，10 秒（隔了 5 秒，中间没狗）不比
    _save_index("v.mp4", [0, 1, 2, 3, 4, 5, 10], [a, a, a, a, a, b, b])
    s = seek.samples_from_index("v.mp4")
    assert [r["t"] for r in s] == [0, 1, 2, 3, 4, 5, 10]
    assert s[0]["motion"] is None and s[1]["motion"] == 0.0
    assert abs(s[5]["motion"] - np.sqrt(2) / 2) < 1e-3          # ‖a−b‖/2
    assert s[6]["motion"] is None                                # 隔太久不比
    assert s[1]["boxes"][0]["bbox"] == [0.07, 0.13, 0.15, 0.15] and s[1]["jpeg"] == b""
    assert seek.samples_from_index("nope.mp4") is None
    assert [r["t"] for r in seek.samples_from_index("v.mp4", start_s=2, end_s=5)] == [2, 3, 4, 5]


def test_seek_有索引就不解码_dry_run_秒出(tmp_path, monkeypatch):
    import numpy as np

    from vision_service import embed

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    embed._cache.clear()
    rng = np.random.default_rng(0)
    # 30 秒，每秒向量都随机 → 相邻距离大 → 都算"在动"
    vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(30, 8))]      # 真索引是归一化过的
    _save_index("v.mp4", list(range(30)), vecs)
    monkeypatch.setattr(seek, "sample_video", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("不该解码")))
    monkeypatch.setattr(seek.config, "SEEK_INDEX_MOTION_MIN", 0.06)
    c = _FakeClient([])
    r = seek.seek_video("/abs/v.mp4", LABELS, dry_run=True, llm=CLAUDE, client=c, rel_path="v.mp4")
    assert r["stats"]["from_index"] is True and r["stats"]["clips_candidate"] > 0 and c.calls == []


def test_seek_走索引_真跑时只抽选中窗的帧(tmp_path, monkeypatch, fake_video):
    import numpy as np

    from vision_service import embed

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    embed._cache.clear()
    rng = np.random.default_rng(1)
    vecs = [v / np.linalg.norm(v) for v in rng.normal(size=(12, 8))]
    _save_index("v.mp4", list(range(12)), vecs)
    monkeypatch.setattr(seek, "ffmpeg_available", lambda: False)
    # 抽帧走 cv2 假视频；检测不该被调用（框来自索引）
    _stub_detect(monkeypatch, lambda f, conf=0.35: (_ for _ in ()).throw(AssertionError("不该检测")))
    made = []
    orig = seek.iter_frames

    def spy(path, every, start_s=0.0, end_s=None):
        made.append((start_s, end_s))
        fake_video["cap"] = _Cap([int((start_s + k) * 1000) for k in range(int((end_s or 12) - start_s) + 1)])
        return orig(path, every, start_s, end_s)
    monkeypatch.setattr(seek, "iter_frames", spy)
    c = _FakeClient(['{"label":"舔身体","body_part":"前肢爪","confidence":0.9}'] * 20)
    r = seek.seek_video("/abs/v.mp4", LABELS, llm=CLAUDE, client=c, rel_path="v.mp4", concurrency=1, clip_s=6, stride_s=3)
    assert r["stats"]["from_index"] is True and r["stats"]["clips_sent"] == len(r["windows"]) > 0
    assert len(made) == len(r["windows"])                              # 每个窗只抽自己那几秒
    for kw in c.calls:
        imgs = [b for b in kw["messages"][0]["content"] if b["type"] == "image"]
        assert 1 <= len(imgs) <= 6
    assert r["segments"] and r["segments"][0]["label"] == "舔身体"
