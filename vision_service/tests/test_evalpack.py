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


def test_吃模型回复原文_按顺序对位_不信序号():
    """手抄二十几个答案是整件事里最烦也最容易错的一步。模型偶尔会把序号写错或
    跳号，而**顺序几乎不会乱**——所以按出现顺序对位，不信序号。"""
    reply = '''好的，逐张判断如下：

1. {"see": "clear", "desc": "侧卧", "contact": true, "part": "前左爪", "moving": true, "confidence": 0.8}
2. {"see": "clear", "desc": "趴着", "contact": false, "part": null, "moving": false, "confidence": 0.7}
3. {"see": "unclear", "desc": "太暗", "contact": false, "part": null, "moving": true, "confidence": 0.3}
5. {"see": "clear", "desc": "蜷着", "contact": true, "part": "后右爪", "moving": true, "confidence": 0.9}

以上。'''
    parts, why = ep.parse_reply(reply, 4)
    assert why == ""
    # 第 4 条模型写的序号是 5，但它就是第 4 张——按顺序对位
    assert parts == ["前左爪", "没贴到", "看不清", "后右爪"]

    # contact=true 但 part 是 null：当没贴到，别把 None 当部位塞进去
    p2, _ = ep.parse_reply('{"see":"clear","contact":true,"part":null}', 1)
    assert p2 == ["没贴到"]
    # 只给了 part 没给 contact：有部位就算贴到
    p3, _ = ep.parse_reply('{"see":"clear","part":"尾根"}', 1)
    assert p3 == ["尾根"]
    # 回复里夹着别的 JSON（比如它先解释了一下格式）：不认的跳过
    p4, _ = ep.parse_reply('先说格式：{"字段":"说明"}\n{"see":"clear","contact":true,"part":"前爪"}', 1)
    assert p4 == ["前爪"]


def test_条数对不上就不填_别硬凑():
    """凑出来的对位是错的，比没有还糟——从第一条错位开始，后面全错。"""
    parts, why = ep.parse_reply('{"see":"clear","contact":true,"part":"前左爪"}', 5)
    assert parts and "解析出 1 条" in why and "5 张图" in why
    assert "没往里填" in why and "别硬凑" in why
    assert "单独问一次" in why                      # 给出补救办法


def test_填进答题卡_再填别家不会覆盖上一家(tmp_path, monkeypatch):
    import csv as _csv

    out, _ = _pack(tmp_path, monkeypatch, n=2, n_ctl=1)
    reply = "\n".join('{"see":"clear","contact":true,"part":"%s"}' % p
                      for p in ("前左爪", "后右爪", "尾根"))
    f1 = tmp_path / "a.txt"
    f1.write_text(reply, encoding="utf-8")
    msg = ep.fill(out, "豆包1.6", str(f1))
    assert "填好了 3 条" in msg and "豆包1.6" in msg
    assert "人工那一列还得人自己填" in msg          # 标尺不能让机器填

    rows = _read(out, "答题卡.csv")
    assert list(rows[0]) == ["图", "人工填这一列", "豆包1.6"]   # 占位的模型A/B/C 去掉了
    assert [r["豆包1.6"] for r in rows] == ["前左爪", "后右爪", "尾根"]

    # 再填一家：上一家那一列要留着
    f2 = tmp_path / "b.txt"
    f2.write_text('{"see":"unclear"}\n{"see":"clear","contact":false}\n'
                  '{"see":"clear","contact":true,"part":"前右爪"}', encoding="utf-8")
    ep.fill(out, "GPT-5", str(f2))
    rows = _read(out, "答题卡.csv")
    assert list(rows[0]) == ["图", "人工填这一列", "豆包1.6", "GPT-5"]
    assert [r["GPT-5"] for r in rows] == ["看不清", "没贴到", "前右爪"]
    assert rows[0]["豆包1.6"] == "前左爪"

    assert "读不到" in ep.fill(out, "x", str(tmp_path / "没有这个文件"))
