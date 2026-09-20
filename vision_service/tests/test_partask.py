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

    seen = {}

    def fake_iter(path, every, start_s=0.0, end_s=None):
        seen.update(every=every, start_s=start_s, end_s=end_s)
        t = start_s
        while end_s is None or t <= end_s:
            yield t, np.full((200, 200, 3), 100, np.uint8)
            t += every
    monkeypatch.setattr(seek, "iter_frames", fake_iter)
    monkeypatch.setattr(partask.seek, "iter_frames", fake_iter)

    detected = []
    monkeypatch.setattr(partask, "embed", embed)
    from vision_service import dog
    monkeypatch.setattr(dog, "detect", lambda *a, **kw: detected.append(1) or [])

    frames, why = partask.frames_around("/nas/a.mp4", rel, 11.5, n=6, span_s=3.0, step_s=0.3)
    assert len(frames) == 6 and not detected and why == ""   # 一次检测都没跑
    # **按 0.3 秒解码，不是照着索引的 1 秒**：舔/啃是 2-4Hz，1 秒间隔只能采到随机相位
    assert seen["every"] == 0.3
    assert abs(seen["start_s"] - (11.5 - 0.3 * 5 / 2)) < 1e-6
    img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    # 框是 0.25~0.75（100x100），四周各留 25% → 150x150
    assert img.shape[0] == 150 and img.shape[1] == 150

    # 拿不到帧的四种原因要分得开——解法完全不同
    _f, why = partask.frames_around("/nas/a.mp4", "没建过索引.mp4", 11.5)
    assert _f == [] and "索引里没有这一路" in why
    _f, why = partask.frames_around("/nas/a.mp4", rel, 9999.0)
    assert _f == [] and "不在索引里" in why and "索引覆盖 10~13s" in why


def test_部位选项默认给全六个_只给两个的话乱猜也能蒙对一半():
    """原来只给"后左爪/后右爪"，理由是别把粗筛的信息扔了。2026-09-20 的对照组
    推翻了这个理由：只给两个选项时，乱猜有一半概率蒙对部位，于是正式组和对照组
    的命中率都是 15%——那个数完全没有意义。

    给六个之后蒙对概率降到 1/6，而且模型选的部位跟几何判的对不对得上，
    本身就成了一个可验证的信号。
    """
    ls = partask.labels_for("后爪", ["舔", "啃"])
    assert [l.name for l in ls] == ["舔", "啃"] and ls[0].description
    assert ls[0].parts == list(posepart.SLOT_NAMES)              # 四爪 + 尾根 + 颈部
    assert len(ls[0].parts) == 6 and "前左爪" in ls[0].parts
    # 想对比两种问法时还能退回去
    assert partask.labels_for("后爪", ["舔"], wide=False)[0].parts == ["后左爪", "后右爪"]
    assert partask.labels_for("后右爪", ["舔"], wide=False)[0].parts == ["后右爪"]


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
    assert "解码没给出" in r["skipped"]        # 解码问题，跟索引问题分得开

    # 正常问：候选本身的字段要跟着回答一起带出来，否则结果对不回是哪一条
    def fake_iter(path, every, start_s=0.0, end_s=None):
        yield 10.0, np.full((100, 100, 3), 100, np.uint8)
    monkeypatch.setattr(partask.seek, "iter_frames", fake_iter)
    monkeypatch.setattr(partask.seek, "ask", lambda *a, **kw: {
        "label": "舔", "body_part": "后左爪", "confidence": 0.8, "see": "clear",
        "desc": "侧卧口鼻接触左后肢", "note": "有往复", "usage": {"input": 9, "output": 3}})
    r = partask.ask_one(hit, "后爪", [], None, n_frames=4, span_s=3.0, video_root=str(root),
                        contact=False)
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


def test_probe_逐层打印实际发生了什么(tmp_path, monkeypatch, capsys):
    """取不到帧时，索引层 / ffmpeg 层 / cv2 层各有自己的失败方式，而每一层的错都被
    上一层吞掉了——CLI 里连 logger 的 warning 都看不见。2026-09-20 为这件事猜了两轮
    还没猜对，所以不猜了：把每一层的真实输出原样打出来。"""
    import subprocess
    import sys

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    rel = "data_raw/d/a_cam1_imu1_raw.mp4"
    _index(tmp_path, rel, [418, 420, 422, 424], [[0.2, 0.2, 0.8, 0.8]] * 4)
    root = tmp_path / "nas"
    (root / "data_raw" / "d").mkdir(parents=True)
    (root / "data_raw" / "d" / "a_cam1_imu1_raw.mp4").write_bytes(b"0" * 1000)

    monkeypatch.setattr(partask.seek, "_video_size", lambda p: (640, 360))

    class R:
        returncode = 1
        stdout = b""
        stderr = "moov atom not found".encode()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: R())

    class Cap:
        def isOpened(self):
            return True

        def get(self, prop):
            return 0.0

        def set(self, *a):
            return True

        def grab(self):
            return False

        def release(self):
            pass
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda p: Cap())

    txt = partask.probe({"path": rel, "t": 421.0}, str(root))
    assert "存在=True" in txt and "索引里 4 帧，覆盖 418~424s" in txt
    assert "要的时刻：[418.0, 420.0, 422.0, 424.0]" in txt
    assert "hwaccel=True" in txt and "hwaccel=False" in txt
    assert "-ss 417.500" in txt and "退出码=1" in txt
    assert "moov atom not found" in txt                 # ffmpeg 真正的抱怨要原样带出来
    assert "grab 到 0 帧" in txt

    # 视频不在：一句话说完，不往下跑
    txt2 = partask.probe({"path": "没有.mp4", "t": 1.0}, str(root))
    assert "**不存在**" in txt2 and "ffmpeg" not in txt2

    # --probe 不调 API
    monkeypatch.setattr(posepart, "find", lambda *a, **kw: {
        "known": True, "hits": [{"path": rel, "t": 421.0, "dist": 0.1, "slot": "后左爪"}]})
    monkeypatch.setattr(partask.config, "VIDEO_ROOT", str(root))
    monkeypatch.setattr(sys, "argv", ["x", "--part", "后爪", "--probe", "1",
                                      "--index-dir", str(tmp_path)])
    partask.main()
    assert "moov atom not found" in capsys.readouterr().out


def test_索引里存的框要是归一化的_不能过crop_rect():
    """crop_rect 是给像素坐标用的，最后一步 int() 取整。传 w=1,h=1 时 0.35 砍成 0、
    0.75 也砍成 0——每一行都存成 (0,0,0,0)。444 路索引从写进去那天起全是零，
    而在此之前没有任何代码读过它，所以一直没人发现（2026-09-20）。"""
    b = [{"bbox": [0.3, 0.25, 0.4, 0.5], "conf": 0.9}]
    assert seek.crop_rect(b, 1, 1, margin=0.0, min_side=0) == (0, 0, 0, 0)   # 老做法，坏的
    assert embed.norm_box(b) == (0.3, 0.25, 0.7, 0.75)                        # 新做法
    # 两只狗取并集
    b2 = b + [{"bbox": [0.1, 0.6, 0.2, 0.3], "conf": 0.8}]
    assert embed.norm_box(b2) == (0.1, 0.25, 0.7, 0.9)
    assert embed.box_ok((0.3, 0.25, 0.7, 0.75)) and not embed.box_ok((0, 0, 0, 0))
    assert not embed.box_ok((0.5, 0.5, 0.5, 0.9)) and not embed.box_ok(None)


def test_老索引的空框_当场重跑检测而不是裁出空图(tmp_path, monkeypatch):
    """重建 444 路要一个多小时，不值得为这一列重来。读的那边认出空框就自己兜底。"""
    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    rel = "data_raw/d/a_cam1_imu1_raw.mp4"
    _index(tmp_path, rel, [10, 11], [[0, 0, 0, 0]] * 2)        # 老索引：框全是零

    def fake_iter(path, every, start_s=0.0, end_s=None):
        for t in (10.0, 11.0):
            yield t, np.full((200, 200, 3), 120, np.uint8)
    monkeypatch.setattr(partask.seek, "iter_frames", fake_iter)

    from vision_service import dog
    calls = []
    monkeypatch.setattr(dog, "detect", lambda f, *a, **kw: calls.append(1) or
                        [{"bbox": [0.25, 0.25, 0.5, 0.5], "conf": 0.9}])
    frames, why = partask.frames_around("/nas/a.mp4", rel, 10.5, n=2, span_s=3.0)
    assert len(frames) == 2 and why == "" and len(calls) == 2   # 每帧重跑一次检测
    img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] == 150 and img.shape[1] == 150          # 100x100 的框 + 各 25% 边

    # 框是好的就不重跑
    _index(tmp_path, rel, [10, 11], [[0.25, 0.25, 0.75, 0.75]] * 2)
    calls.clear()
    frames, why = partask.frames_around("/nas/a.mp4", rel, 10.5, n=2, span_s=3.0)
    assert len(frames) == 2 and not calls


def _row2(dists):
    """跟 _row 一样，但显式收 6 个距离——对照组要造"离所有爪子都很远"的行。"""
    v = np.zeros(DIM, dtype="float32")
    xy = np.stack([np.linspace(-0.4, 0.4, K), np.linspace(0.4, -0.4, K)], axis=1).astype("float32")
    for k in (NOSE, *PAWS):
        v[k * 2:k * 2 + 2] = xy[k]
        v[DIM - K + k] = 1.0
    v[posepart.D0: posepart.D0 + posepart.N_DIST] = dists
    return v / np.linalg.norm(v)


def _idx2(d, name, path, rows, ts):
    meta = {"path": path, "sampled": len(rows), "with_dog": len(rows), "pose": True}
    np.savez(os.path.join(d, name), t=np.asarray(ts, dtype="float32"),
             pose=np.asarray(rows, dtype="float16"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))


def test_对照组_几何判定离爪子很远的帧(tmp_path, monkeypatch):
    """我们告诉模型"这是舔/啃候选、部位在后爪里选"，而几何筛出来的本来就全是
    "头靠近后爪"的画面——一个无脑总说「舔-后爪」的模型也能拿到很高的命中率。
    对照组问同一个问题，只是鼻子离爪子很远。"""
    d = str(tmp_path)
    near = [_row2([0.9, 0.9, 0.15, 0.9, 0.9, 0.9])] * 3          # 够到后爪
    far = [_row2([2.0, 2.1, 2.2, 2.3, 2.0, 2.0])] * 3            # 离所有爪子都 >1.5 体长
    _idx2(d, "a.npz", "data_raw/d/a_cam1_imu1_raw.mp4", near + far, range(6))
    ctl = partask.control_hits(d, "后爪", 10)
    assert len(ctl) and all(c["control"] and c["dist"] > 1.5 for c in ctl)
    assert all(c["t"] >= 3 for c in ctl)                          # 只挑远的那三帧
    assert len(partask.control_hits(d, "后爪", 1)) == 1            # 要几条给几条


def test_对照组也高时_要明说正式组那个数是假的():
    """对照组跟正式组差不多 = 模型在顺着提示词猜。这时候命中率不是"精度低"，
    是"这个数没有意义"——两者的下一步完全不同。"""
    def mk(label, control=False):
        return {"path": "a.mp4", "t": 1, "see": "clear", "label": label, "body_part": "后左爪",
                "confidence": 0.8, "desc": "d", "note": "n", "control": control,
                "usage": {"input": 1, "output": 1}}

    # 正式 4/10 命中，对照 3/10 命中 → 假的
    bad = [mk("舔") for _ in range(4)] + [mk(None) for _ in range(6)] \
        + [mk("舔", True) for _ in range(3)] + [mk(None, True) for _ in range(7)]
    txt = partask.summarize(bad)
    assert "命中   4（40%）" in txt                                # 对照组不进正式组的分母
    assert "对照组 10 条" in txt and "命中 3（30%）" in txt
    assert "是假的" in txt and "顺着提示词猜" in txt
    assert "对照组误报：d；n" in txt                                # 误报长什么样要打出来

    # 正式 7/20、对照 0/10 → 可信
    good = [mk("舔") for _ in range(7)] + [mk(None) for _ in range(13)] \
        + [mk(None, True) for _ in range(10)]
    txt2 = partask.summarize(good)
    assert "✓ 对照组明显低于正式组（0% vs 35%）" in txt2 and "可信" in txt2


def test_统计里对照组不进分母_但也要算进问了几条():
    """把对照组从分母剔掉之后忘了同步改分子，于是 40 条全问了却报成
    "问了 20/40（20 条取不到帧）"——凭空多出 20 条失败，人会去查根本不存在的问题。"""
    def mk(control=False, skipped=None):
        r = {"path": "a.mp4", "t": 1, "see": "clear", "label": None, "confidence": 0.0,
             "desc": "d", "note": "n", "control": control, "usage": {"input": 1, "output": 1}}
        if skipped:
            r["skipped"] = skipped
        return r

    txt = partask.summarize([mk() for _ in range(20)] + [mk(True) for _ in range(20)])
    assert "问了 40/40 条（0 条取不到帧）" in txt
    assert "正式 20 条、对照 20 条" in txt
    txt2 = partask.summarize([mk() for _ in range(3)] + [mk(skipped="视频不在")])
    assert "问了 3/4 条（1 条取不到帧）" in txt2 and "对照" not in txt2.split("\n")[1]


def test_自洽性_自己跟自己对不上时要明说命中率没意义():
    """2026-09-20 同样 20 条问了两次，一次 7 条命中、一次 3 条，只有 2 条重合。
    判断不稳到这个程度时，命中率是多少都没意义，调提示词也没用。"""
    def r(labels):
        return [{"path": f"{i}.mp4", "t": i, "label": l, "body_part": "后左爪" if l else None}
                for i, l in enumerate(labels)]

    # 五条里三条两轮答得不一样
    bad = partask.agreement([r(["舔", "舔", None, None, "舔"]),
                             r([None, "舔", "舔", None, None])])
    assert "每轮命中数：[3, 2]" in bad and "5 条里 2 条（40%）" in bad
    assert "命中率是多少都没意义" in bad and "没在看画面" in bad

    good = partask.agreement([r(["舔", None, None, None, "舔"]),
                              r(["舔", None, None, None, "舔"])])
    assert "5 条里 5 条（100%）" in good and "其中 2 条部位也一样" in good
    assert "没意义" not in good


def test_两组都是0时_不能说成顺着提示词猜():
    """两边都是 0 不是"模型在猜"，是**一条都没判出来**——解法完全相反：
    前者要换问法，后者要查采样间隔/放松提示词。说错了人就往错的方向改。"""
    def mk(label, control=False):
        return {"path": "a.mp4", "t": 1, "see": "clear", "label": label, "body_part": None,
                "confidence": 0.0, "desc": "d", "note": "n", "control": control,
                "usage": {"input": 1, "output": 1}}

    zero = partask.summarize([mk(None) for _ in range(15)] + [mk(None, True) for _ in range(15)])
    assert "两组都是 0" in zero and "对所有片段都答 none" in zero
    assert "2-4Hz" in zero and "--step 0.3" in zero      # 指向真正该查的地方
    assert "顺着提示词猜" not in zero

    # 两边都有命中且差不多：那才是"在猜"
    guess = partask.summarize([mk("舔") for _ in range(6)] + [mk(None) for _ in range(9)]
                              + [mk("舔", True) for _ in range(5)] + [mk(None, True) for _ in range(10)])
    assert "顺着提示词猜" in guess and "两组都是 0" not in guess


def test_全答none时_自洽率100是白送的():
    """每轮都是 0 命中，"几轮答的类别一样"必然 100%——那说明不了稳不稳，
    不点破的话会被当成"模型很稳定"。"""
    rows = [[{"path": f"{i}.mp4", "t": i, "label": None, "body_part": None} for i in range(5)]] * 2
    txt = partask.agreement(rows)
    assert "100%" in txt and "白送的" in txt and "说明不了稳不稳" in txt


def test_dump_把真正发出去的图和提示词存下来(tmp_path):
    """2026-09-20 为「一条都判不出来」改了五轮提示词和采样参数，**从来没看过一眼
    真正发出去的图**。狗在 720p 俯拍里只占 100x50 像素，裁进 384px 的格子再六格
    拼一张，舌头可能只剩几个像素——那样改什么提示词、换什么模型都没用。"""
    res = [
        {"path": "d/a.mp4", "t": 422.0, "slot": "后左爪", "dist": 0.1, "n_frames": 6,
         "_tile": b"\xff\xd8fake-jpeg", "_system": "系统提示", "_user": "用户提示",
         "_raw": '{"label":"none"}'},
        {"path": "d/b.mp4", "t": 10.0, "slot": "（对照）", "dist": 2.0, "n_frames": 6,
         "control": True, "_tile": b"\xff\xd8x", "_system": "s", "_user": "u", "_raw": "r"},
        {"path": "d/c.mp4", "t": 1.0, "skipped": "视频不在"},      # 没图的跳过
    ]
    msg = partask.dump_debug(res, str(tmp_path / "dbg"))
    names = sorted(os.listdir(tmp_path / "dbg"))
    assert names == ["00_hit.jpg", "00_hit.txt", "01_ctl.jpg", "01_ctl.txt"]
    assert (tmp_path / "dbg" / "00_hit.jpg").read_bytes() == b"\xff\xd8fake-jpeg"
    txt = (tmp_path / "dbg" / "00_hit.txt").read_text(encoding="utf-8")
    assert "d/a.mp4" in txt and "422.0s" in txt and "后左爪" in txt
    assert "系统提示" in txt and "用户提示" in txt and '{"label":"none"}' in txt
    assert "存了 2 组" in msg and "先看图" in msg


def test_ask_debug_带回真正发出去的那张图(monkeypatch):
    from vision_service import llm as llmmod

    frames = [bytes(cv2.imencode(".jpg", np.full((60, 60, 3), c, np.uint8))[1].tobytes())
              for c in (50, 150, 250)]
    sent = {}

    def fake(llm, system, user, jpegs, max_tokens=300, client=None, http=None):
        sent["jpegs"] = jpegs
        return '{"see":"clear","desc":"d","label":"none","confidence":0,"note":"n"}', {"input": 1, "output": 1}
    monkeypatch.setattr(llmmod, "chat_vision", fake)

    llm = llmmod.LLM(provider="anthropic", api_key="k", model="m")
    plain = seek.ask(frames, [seek.Label(name="舔")], 1.5, llm)
    assert "_tile" not in plain                                  # 不开 debug 不多带东西

    dbg = seek.ask(frames, [seek.Label(name="舔")], 1.5, llm, debug=True)
    assert dbg["_tile"] == sent["jpegs"][0]                      # 存的就是发出去的那一张
    assert "序号" in dbg["_user"] and dbg["_raw"].startswith("{")


def test_写明细不能吃掉答案_调试字段要剔掉(tmp_path, monkeypatch, capsys):
    """--dump 把图片 bytes 塞进结果，--out 写 JSON 时炸了，73 秒的问答全白跑、
    一个数都没看到（2026-09-20）。两条：调试字段不进 JSONL；结果先打印再写文件。"""
    import sys

    hits = [{"path": "a.mp4", "t": 1.0, "dist": 0.1, "slot": "后左爪"}]
    monkeypatch.setattr(posepart, "find", lambda *a, **kw: {"known": True, "hits": hits})
    monkeypatch.setattr(partask.llmmod, "from_env",
                        lambda: partask.llmmod.LLM(provider="anthropic", api_key="k", model="m"))
    monkeypatch.setattr(partask, "ask_one", lambda h, *a, **kw: {
        **h, "see": "clear", "contact": True, "part": "后左爪", "moving": True,
        "confidence": 0.8, "desc": "d", "usage": {"input": 1, "output": 1},
        "_tile": b"\xff\xd8bytes", "_system": "s", "_user": "u", "_raw": "r"})
    out = tmp_path / "v.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--part", "后爪", "--index-dir", str(tmp_path),
                                      "--out", str(out)])
    partask.main()

    printed = capsys.readouterr().out
    assert "口鼻贴着身体：正式 100%" in printed and "明细写到" in printed
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["part"] == "后左爪" and row["t"] == 1.0
    assert not [k for k in row if k.startswith("_")]        # 调试字段不进 JSONL

    # 写文件失败也不能吞掉结果
    monkeypatch.setattr(sys, "argv", ["x", "--part", "后爪", "--index-dir", str(tmp_path),
                                      "--out", str(tmp_path / "没有这个目录" / "v.jsonl")])
    partask.main()
    printed = capsys.readouterr().out
    assert "口鼻贴着身体：正式 100%" in printed and "明细没写成" in printed and "照样有效" in printed


def test_接触问法_一个字不提舔啃抓挠(monkeypatch):
    """「是不是在舔」画面答不了：舔是 2-4Hz、舌头只有几个像素，俯拍上看不见。
    唯一看得见的是"口鼻贴在某个部位"——而老提示词把那个明确判成"休息"，
    等于禁掉了唯一的证据。所以只问接触关系，行为那一半交给 IMU。"""
    from vision_service import llm as llmmod

    frames = [bytes(cv2.imencode(".jpg", np.full((60, 60, 3), c, np.uint8))[1].tobytes())
              for c in (50, 150)]
    seen = {}

    def fake(llm, system, user, jpegs, max_tokens=300, client=None, http=None):
        seen.update(system=system, user=user)
        return ('{"see":"clear","desc":"侧卧口鼻贴左后肢","contact":true,'
                '"part":"后左爪","moving":true,"confidence":0.8}'), {"input": 1, "output": 1}
    monkeypatch.setattr(llmmod, "chat_vision", fake)

    parts = list(posepart.SLOT_NAMES)
    r = seek.ask_contact(frames, parts, 1.5,
                         llmmod.LLM(provider="anthropic", api_key="k", model="m"))
    assert r["contact"] and r["part"] == "后左爪" and r["moving"] is True
    for word in ("舔", "啃", "抓挠"):
        assert word not in seen["system"] and word not in seen["user"]   # 不给行为上的暗示
    assert "口鼻" in seen["system"] and "接触" in seen["system"]


def test_接触问法_看不清时不采信接触():
    parts = list(posepart.SLOT_NAMES)
    r = seek.parse_contact('{"see":"unclear","desc":"太暗","contact":true,'
                           '"part":"后左爪","moving":true,"confidence":0.9}', parts)
    assert r["contact"] is False and r["part"] is None and r["see"] == "unclear"
    # 部位不在清单里：当没贴到处理，不硬塞
    r2 = seek.parse_contact('{"see":"clear","contact":true,"part":"肚子","moving":false,'
                            '"confidence":0.5}', parts)
    assert r2["contact"] is True and r2["part"] is None and r2["moving"] is False
    assert seek.parse_contact("不是 JSON", parts)["see"] == "unknown"


def test_接触问法的统计_三个数都可验证():
    """跟"命中率"不同，这三个数不依赖模型会不会判行为：接触率有干净的对照、
    部位一致率有几何当参照、在动的比例是下一步跟 IMU 取交集的底。"""
    def mk(contact, part, slot, moving=True, control=False):
        return {"path": "a.mp4", "t": 1, "see": "clear", "contact": contact, "part": part,
                "slot": slot, "dist": 0.1, "moving": moving, "confidence": 0.8, "desc": "d",
                "control": control, "usage": {"input": 1, "output": 1}}

    res = ([mk(True, "后左爪", "后左爪") for _ in range(6)]        # 部位全对
           + [mk(True, "后右爪", "后左爪") for _ in range(3)]      # 前后对、左右反
           + [mk(False, None, "后左爪") for _ in range(1)]
           + [mk(False, None, "（对照）", control=True) for _ in range(9)]
           + [mk(True, "后左爪", "（对照）", control=True)])
    txt = partask.summarize_contact(res)
    assert "正式 90%" in txt and "对照 10%" in txt and "✓ 差了 80 个点" in txt
    # 前后和左右分开数：9 条里全对 6、前后对 9、左右对 6
    assert "全对 6（67%）、前后对 9（100%）、左右对 6（67%）" in txt
    assert "样本不够下结论" in txt                                  # n=9 < 15
    assert "这几帧里在动：100%" in txt
    assert "模型说的部位：正式" in txt and "对照" in txt             # 偏好要看得见

    # 两组差不多：粗筛本身要重做
    bad = [mk(True, "后左爪", "后左爪") for _ in range(5)] + \
          [mk(True, "后左爪", "（对照）", control=True) for _ in range(5)]
    assert "⚠ 两组差不多" in partask.summarize_contact(bad)


def test_候选全是同一侧时_前后一致率没有信息量():
    """2026-09-20 我把"前后 0/7"读成了「系统性相反」，差点据此去查姿态模型。

    **那个 0 是结构性必然的**：候选是按几何判「后爪」筛出来的，几何那一边前后
    是个常数；模型又一次「后爪」都没说过。两个常数当然永远不相等，里面没有信息。
    有信息的是另一句：画面说过几次「后爪」。
    """
    def mk(part, slot):
        return {"path": "a.mp4", "t": 1, "see": "clear", "contact": True, "part": part,
                "slot": slot, "dist": 0.1, "moving": True, "confidence": 0.8, "desc": "d",
                "usage": {"input": 1, "output": 1}}

    res = ([mk("前左爪", "后左爪")] * 4 + [mk("前右爪", "后右爪")] * 2
           + [mk("前左爪", "后右爪")])
    txt = partask.summarize_contact(res)
    assert "全对 0（0%）、前后对 0（0%）、左右对 6（86%）" in txt
    assert "结构性地没有信息" in txt and "两个常数比" in txt
    assert "一次都没说过「后爪」" in txt                            # 有信息的是这一句
    assert "两边都不是真值" in txt
    assert "样本不够下结论" in txt and "p=0.06" in txt              # n=7，别当结论

    # 几何那边前后有变化时，才轮得到"系统性相反"这个说法
    mixed = ([mk("前左爪", "后左爪")] * 5 + [mk("后右爪", "前右爪")] * 5)
    t2 = partask.summarize_contact(mixed)
    assert "前后系统性对不上" in t2 and "结构性" not in t2


def test_两边几乎全不一致时_指向人工标真值():
    """两个不可靠的估计器互相对，只能说明至少有一个错，没法知道是哪个——
    因为两边都不是真值。2026-09-20 在这上面绕了一整天。"""
    def mk(part, slot):
        return {"path": "a.mp4", "t": 1, "see": "clear", "contact": True, "part": part,
                "slot": slot, "dist": 0.1, "moving": True, "confidence": 0.8, "desc": "d",
                "usage": {"input": 1, "output": 1}}

    res = [mk("前左爪", "后左爪")] * 10 + [mk("前右爪", "后左爪")] * 10
    txt = partask.summarize_contact(res)
    assert "两边几乎完全不一致" in txt and "都不是真值" in txt
    assert "--label-sheet" in txt and "二十分钟" in txt



def test_导人工标注单_和读回来算分(tmp_path, monkeypatch):
    """两个估计器互相对判不出谁对，唯一的出路是人工标一小批当真值。
    50 帧二十分钟，就能同时量出几何和画面各自的准确率。"""
    import csv

    monkeypatch.setattr(partask, "frames_around",
                        lambda *a, **kw: ([b"\xff\xd8jpg"], ""))
    res = [{"path": "d/a.mp4", "t": 1.0, "slot": "后左爪", "part": "前左爪", "contact": True},
           {"path": "d/b.mp4", "t": 2.0, "slot": "（对照）", "part": None, "control": True},
           {"path": "d/c.mp4", "t": 3.0, "skipped": "视频不在"}]
    d = str(tmp_path / "sheet")
    msg = partask.label_sheet(res, d, "/nas")
    assert "导了 2 张" in msg and "别看" in msg and "--truth" in msg   # 提醒别看着另两列填
    assert sorted(os.listdir(d)) == ["00.jpg", "01.jpg", "labels.csv"]
    rows = list(csv.DictReader(open(os.path.join(d, "labels.csv"), encoding="utf-8-sig")))
    assert rows[0]["几何"] == "后左爪" and rows[0]["画面"] == "前左爪"
    assert rows[0]["人工填这一列"] == "" and rows[1]["对照"] == "是"

    # 填完读回来：人工说前左爪 → 画面全对、几何全错
    for r in rows:
        r["人工填这一列"] = "前左爪" if not r["对照"] else "没贴到"
    with open(os.path.join(d, "labels.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    txt = partask.score_truth(os.path.join(d, "labels.csv"))
    assert "几何（姿态关键点）对 0/2（0%）" in txt      # 人工说前左爪，几何说后左爪
    assert "画面（大模型）  对 2/2（100%）" in txt      # 画面两条都对上了
    assert "对照组 1 条里人工也说「没贴到」的：1" in txt
    assert "画面明显强于几何" in txt                    # 给出该怎么办

    # 「看不清」要从分母里剔掉：人也判不了的帧，拿来给模型打分没有意义
    for r in rows:
        r["人工填这一列"] = "看不清"
    with open(os.path.join(d, "labels.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    t2 = partask.score_truth(os.path.join(d, "labels.csv"))
    assert "全是「看不清」" in t2 and "人也判不了" in t2

    # 还没填就来算：明说，别报个 0%
    for r in rows:
        r["人工填这一列"] = ""
    with open(os.path.join(d, "labels.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert "还是空的" in partask.score_truth(os.path.join(d, "labels.csv"))
