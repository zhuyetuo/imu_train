"""拿去 web 上手试的包：对照组要混进去、真值要先填、看不清要从分母里剔掉。"""

from __future__ import annotations

import csv
import os

import numpy as np

from vision_service import evalpack as ep
from vision_service import partask, posepart, seek


def _pack(tmp_path, monkeypatch, n=4, n_ctl=2):
    monkeypatch.setattr(posepart, "find", lambda *a, **kw: {
        "hits": [{"path": "d/a.mp4", "t": float(i), "dist": 0.1, "slot": "后左爪"}
                 for i in range(n)]})
    monkeypatch.setattr(partask, "control_hits", lambda *a, **kw: [
        {"path": "d/b.mp4", "t": float(i), "dist": 2.0, "slot": "（对照）", "control": True}
        for i in range(n_ctl)])
    monkeypatch.setattr(partask, "frames_around", lambda *a, **kw: ([b"\xff\xd8x"], ""))
    monkeypatch.setattr(seek, "tile_frames", lambda f, **kw: b"\xff\xd8tile")
    root = tmp_path / "nas" / "d"
    root.mkdir(parents=True)
    for f in ("a.mp4", "b.mp4"):
        (root / f).write_bytes(b"0")
    out = str(tmp_path / "pack")
    msg = ep.build(out, index_dir=str(tmp_path), part="后爪", n=n, n_control=n_ctl,
                   near_max=0.4, min_motion=0.06, video_root=str(tmp_path / "nas"))
    return out, msg


def _read(out, name):
    with open(os.path.join(out, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_打包_对照混在里面_答题卡上看不出哪些是(tmp_path, monkeypatch):
    """正式的排前面、对照的排后面的话，人翻到一半就看出规律了——那对照就白设了。"""
    out, msg = _pack(tmp_path, monkeypatch)
    assert "6 张，其中对照 2 张" in msg
    assert sorted(os.listdir(os.path.join(out, "图"))) == [f"{i:02d}.jpg" for i in range(6)]

    sheet = _read(out, "答题卡.csv")
    # 答题卡上**没有**「是对照」「几何判的」这些列：看着机器的答案填，
    # 填出来的"真值"里就掺了机器的错
    assert set(sheet[0]) == {"图", "人工填这一列", *ep.MODEL_COLS}
    key = _read(out, ep.ANSWER_KEY)
    assert sum(1 for r in key if r["是对照"]) == 2
    # 打乱过：两张对照不会正好是最后两张
    assert [r["图"] for r in key if r["是对照"]] != ["04.jpg", "05.jpg"]


def test_提示词跟线上一字不差(tmp_path, monkeypatch):
    """包里的提示词要是另写一份，试出来的结论就不代表线上。"""
    out, _ = _pack(tmp_path, monkeypatch)
    txt = open(os.path.join(out, "提示词.txt"), encoding="utf-8").read()
    sys_, user = seek.build_contact_prompt(partask.CONTACT_PARTS, 6, 0.3 * 5, tiled=True)
    assert sys_ in txt and user in txt
    assert "一次一张" in txt and "重新开一个对话" in txt
    how = open(os.path.join(out, "怎么用.md"), encoding="utf-8").read()
    assert "看不清" in how and "别硬选" in how and "对照" in how
    # 「把整个目录压缩了丢给对话框」是个会自然想到的做法，但测出来的不是单张的能力
    assert "别把整个目录压缩" in how and "互相影响" in how


def test_一次多张的版本要写明它跟线上不是同一个条件(tmp_path, monkeypatch):
    """26 个对话确实烦，所以给一份省事的。但拿它的数去跟线上比就错了——
    线上是一张图一次请求，一次给好几张时前面的答案会带着后面走。"""
    out, _ = _pack(tmp_path, monkeypatch)
    f = os.path.join(out, "提示词_一次多张（省事但打折）.txt")
    txt = open(f, encoding="utf-8").read()
    assert "不是同一个条件" in txt or "不等于线上" in txt
    assert "分别独立判断" in txt and "不要因为前一张" in txt      # 提示词里也要压一句
    assert "给领导看的那个数" in txt and "一张一张" in txt        # 该用哪份说清楚
    # 系统提示那一段照抄线上的，别另写一份
    sys_, _u = seek.build_contact_prompt(partask.CONTACT_PARTS, 6, 0.3 * 5, tiled=True)
    assert sys_ in txt


def test_算分_看不清剔出分母_对照组乱报要单独点出来(tmp_path, monkeypatch):
    out, _ = _pack(tmp_path, monkeypatch, n=4, n_ctl=2)
    key = {r["图"]: r for r in _read(out, ep.ANSWER_KEY)}
    sheet = _read(out, "答题卡.csv")
    for r in sheet:
        ctl = bool(key[r["图"]]["是对照"])
        r["人工填这一列"] = "没贴到" if ctl else "前左爪"
        r["模型A"] = "前左爪"          # 见什么都说前左爪：正式组全对，对照组全错
        r["模型B"] = "没贴到" if ctl else "前左爪"   # 真在看画面
    sheet[0]["人工填这一列"] = "看不清"   # 这一张从分母里剔掉
    with open(os.path.join(out, "答题卡.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)

    txt = ep.score(out)
    assert "「看不清」1 张" in txt and "从分母里剔掉" in txt
    assert "模型A" in txt and "模型B" in txt and "模型C" not in txt   # 空列不进表
    assert "对照组乱报" in txt
    assert "顺着提示词猜" in txt

    # 没填真值 / 目录不对：各说各的，别报个 0%
    with open(os.path.join(out, "答题卡.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows([{**r, "人工填这一列": ""} for r in sheet])
    assert "还是空的" in ep.score(out)
    assert "读不到" in ep.score(str(tmp_path / "没有这个目录"))


def test_算分给的结论按准确率分档(tmp_path, monkeypatch):
    """『换更贵的模型解决不了』和『可以拿来排序』是两个完全不同的决定。"""
    out, _ = _pack(tmp_path, monkeypatch, n=10, n_ctl=0)
    sheet = _read(out, "答题卡.csv")

    def run(n_right):
        for i, r in enumerate(sheet):
            r["人工填这一列"] = "前左爪"
            r["模型A"] = "前左爪" if i < n_right else "尾根"
        with open(os.path.join(out, "答题卡.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sheet[0]))
            w.writeheader()
            w.writerows(sheet)
        return ep.score(out)

    assert "别充" in run(2)                       # 20%，跟瞎猜差不多
    assert "别指望免掉人" in run(5)                # 50%
    assert "算一笔账再定" in run(9)                # 90%
