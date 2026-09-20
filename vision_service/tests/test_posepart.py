"""靠姿态几何筛部位：距离要还原尺度、测不到的不能冒充最近、认不出的部位不筛。"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from vision_service import posepart as pp
from vision_service.pose import DIM, K, NOSE, PAWS


def _row(dists, visible=(NOSE, *PAWS), coords=0.3):
    """造一行"存进索引的样子"：拼好再整条 L2 归一化（真索引就是这么存的）。"""
    v = np.zeros(DIM, dtype="float32")
    v[: K * 2] = coords
    v[pp.D0: pp.D0 + pp.N_DIST] = dists
    for k in visible:
        v[DIM - K + k] = 1.0
    return v / np.linalg.norm(v)


def test_距离要乘回尺度_不然行与行之间不可比():
    """整条向量归一化过，两行可见点数不同时 ‖v‖ 不同，存下来的距离也不同。
    还原之后必须回到同一个刻度——不还原就拿去卡阈值是错的。"""
    a = _row([0.9, 0.9, 0.25, 0.9, 0.9, 0.9], visible=(NOSE, *PAWS))
    b = _row([0.9, 0.9, 0.25, 0.9, 0.9, 0.9], visible=(NOSE, PAWS[2]))   # 只两个点可见
    assert not np.isclose(a[pp.D0 + 2], b[pp.D0 + 2])                     # 存的时候不一样
    da, _ = pp.decode(a)
    db, _ = pp.decode(b)
    assert np.isclose(da[0, 2], 0.25, atol=1e-3) and np.isclose(db[0, 2], 0.25, atol=1e-3)


def test_测不到的距离记0_不能冒充最近():
    """descriptor 里两头有一头没测到就记 0。0 是"不知道"不是"贴着"，
    当成最近的话，测不到的帧会全部冒充成命中。"""
    slot, d = pp.nearest(np.array([[0.0, 0.0, 0.5, 0.8, 0.0, 0.0]], dtype="float32"))
    assert slot[0] == 2 and np.isclose(d[0], 0.5)
    slot, d = pp.nearest(np.zeros((1, 6), dtype="float32"))              # 一个都没测到
    assert slot[0] == -1 and d[0] == 0.0


def test_要够近_还要是最近的那个():
    rows = np.stack([
        _row([0.9, 0.9, 0.20, 0.9, 0.9, 0.9]),      # 鼻子贴后左爪 → 后爪命中
        _row([0.15, 0.9, 0.30, 0.9, 0.9, 0.9]),     # 后左爪也在阈值内，但前左爪更近 → 不算后爪
        _row([0.9, 0.9, 0.95, 0.9, 0.9, 0.9]),      # 后左爪最近但太远 → 不算
    ])
    assert list(pp.match(rows, "后爪", near_max=0.6)) == [True, False, False]
    assert list(pp.match(rows, "前爪", near_max=0.6)) == [False, True, False]
    assert list(pp.match(rows, "后左爪", near_max=0.6)) == [True, False, False]
    assert list(pp.match(rows, "后右爪", near_max=0.6)) == [False, False, False]
    # 不要求"最近"时，第二行的后左爪也进来——所以默认要开着，不然一帧算进四个部位
    assert list(pp.match(rows, "后爪", near_max=0.6, require_nearest=False)) == [True, True, False]


@pytest.mark.parametrize("name,want", [
    ("后右爪", "后右爪"), ("右后爪", "后右爪"), ("后爪", "后爪"), ("前爪", "前爪"),
    ("前左爪", "前左爪"), ("后肢", "后爪"), ("趾间/爪垫", "爪"),
    ("尾根", "尾根"), ("尾/尾根", "尾根"), ("颈侧/颈下", "颈部"),
    # 鼻子到爪的距离说明不了这些，认不出来 → 不筛，而不是滤成空
    ("腰（少见）", None), ("腹股沟", None), ("耳/耳后", None), ("", None),
])
def test_部位名按关键字认_认不出的返回None(name, want):
    assert pp.part_of(name) == want


def test_认不出的部位不返回空清单而是明说(tmp_path):
    """滤成空会让人以为"索引里没有这种数据"，那是最坏的误导。"""
    r = pp.find(str(tmp_path), "腹股沟")
    assert r["known"] is False and r["hits"] == []
    assert pp.match(np.stack([_row([0.1] * 6)]), "腹股沟").tolist() == [False]


def _idx(d, name, path, rows, ts):
    meta = {"path": path, "sampled": len(rows), "with_dog": len(rows), "pose": True}
    np.savez(os.path.join(d, name), t=np.asarray(ts, dtype="float32"),
             pose=np.asarray(rows, dtype="float16"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))


def test_扫整份索引_按距离排_每路限量(tmp_path):
    d = str(tmp_path)
    near = [_row([0.9, 0.9, 0.1 + i * 0.01, 0.9, 0.9, 0.9]) for i in range(5)]
    far = [_row([0.9, 0.9, 0.95, 0.9, 0.9, 0.9])]
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/a_cam1_imu1_raw.mp4", near + far, range(6))
    _idx(d, "b.npz", "data_raw/2026_9_14_gouchang/b_cam2_imu2_raw.mp4",
         [_row([0.9, 0.9, 0.05, 0.9, 0.9, 0.9])], [42])
    r = pp.find(d, "后爪", near_max=0.6, max_per_video=2)
    assert r["known"] and r["with_dog"] == 7
    # b 那一路最近（0.05），排第一；a 那一路只留 2 个
    assert r["hits"][0]["t"] == 42.0 and len(r["hits"]) == 3
    assert [h["dist"] for h in r["hits"]] == sorted(h["dist"] for h in r["hits"])
    assert all(h["slot"] == "后左爪" for h in r["hits"])


def test_标定_给分布和各阈值的帧数_不拍脑袋定(tmp_path):
    d = str(tmp_path)
    rows = [_row([0.9, 0.9, v, 0.9, 0.9, 0.9]) for v in (0.1, 0.2, 0.35, 0.55, 0.9)]
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/a_cam1_imu1_raw.mp4", rows, range(5))
    txt = pp.calib(d, "后爪")
    assert "p50=" in txt and "≤0.6" in txt and "体长" in txt
    assert pp.calib(d, "腹股沟").startswith("不认识的部位")


def test_空索引目录和坏文件都不炸(tmp_path):
    assert pp.find(str(tmp_path), "后爪")["hits"] == []
    assert pp.find(str(tmp_path / "不存在"), "后爪")["hits"] == []
    (tmp_path / "bad.npz").write_bytes(b"not an npz")
    assert pp.find(str(tmp_path), "后爪")["hits"] == []


def test_找相似加部位条件_只留鼻子够到那只爪的帧(tmp_path, monkeypatch):
    """SigLIP 分不清「头贴前左爪」和「头贴前右爪」，所以画面分可能前者更高；
    加上部位这个几何硬条件之后，只有真够到后爪的那帧还在。"""
    from vision_service import embed

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    rows = np.stack([
        _row([0.9, 0.9, 0.15, 0.9, 0.9, 0.9]),     # 鼻子贴后左爪
        _row([0.12, 0.9, 0.9, 0.9, 0.9, 0.9]),     # 鼻子贴前左爪
        _row([0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),      # 哪儿都没够到
    ])
    emb = np.array([[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]], dtype="float32")   # 第 2 帧画面分最高
    meta = {"path": "data_raw/2026_9_14_gouchang/a_cam1_imu1_raw.mp4", "pose": True,
            "model": embed.config.EMBED_MODEL, "masked": False}
    np.savez(embed.index_path(meta["path"]), t=np.array([1.0, 2.0, 3.0], dtype="float32"),
             emb=emb.astype("float16"), box=np.zeros((3, 4), dtype="float32"),
             pose=rows.astype("float16"), meta=np.array(json.dumps(meta, ensure_ascii=False)))
    embed._cache.clear()

    q = [1.0, 0.0]
    plain = embed.search(q, [meta["path"]], center=False)
    assert plain["hits"][0]["t"] == 2.0 and plain["part_used"] is None      # 不筛时前左爪第一

    r = embed.search(q, [meta["path"]], center=False, part="后爪", part_near_max=0.6)
    assert [h["t"] for h in r["hits"]] == [1.0]                              # 只剩真够到后爪那帧
    assert r["part_used"] == "后爪" and r["part_frames"] == 1

    # 判不了的部位：不筛，但如实说 part_used=None——不能让人把没筛的当筛过的
    r2 = embed.search(q, [meta["path"]], center=False, part="腹股沟")
    assert len(r2["hits"]) == 3 and r2["part"] == "腹股沟" and r2["part_used"] is None


def _vid(tmp_path, rel, emb, rows, ts):
    from vision_service import embed

    meta = {"path": rel, "pose": True, "model": embed.config.EMBED_MODEL, "masked": False}
    np.savez(embed.index_path(rel), t=np.asarray(ts, dtype="float32"),
             emb=np.asarray(emb, dtype="float16"), box=np.zeros((len(ts), 4), dtype="float32"),
             pose=np.asarray(rows, dtype="float16"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))


def test_按视频各减各的均值_才找得到别的狗(tmp_path, monkeypatch):
    """这是"图像检索只找得到同一只狗"的根因。

    造两路：A 是黑狗（身份向量 [1,0,0]），B 是白狗（[0,1,0]）。两边各有一帧在舔后爪
    （动作向量 +[0,0,1]），其余帧是别的动作。全局减均值时，B 的"白狗"分量原封不动地
    留在每一帧里，拿 A 的舔爪帧去查，B 的帧全都远——结果永远是同一只狗。
    按视频各减各的均值之后，身份两边同时抵消，剩下的才是动作。
    """
    from vision_service import embed

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    lick, other1, other2 = [0, 0, 1.0], [0, 0.0, 0], [0, 0, 0]

    def route(ident, acts):
        v = np.array([np.array(ident, dtype="float32") * 3 + np.array(a, dtype="float32")
                      for a in acts], dtype="float32")
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # 每路 24 帧（≥20 才用自己的均值）：第 0 帧舔后爪，其余是两种别的动作轮着来
    acts = [lick] + [other1 if i % 2 else other2 for i in range(23)]
    rows = np.stack([_row([0.9, 0.9, 0.15, 0.9, 0.9, 0.9])] * 24)
    a_rel = "data_raw/2026_9_14_gouchang/a_cam1_imu1_raw.mp4"
    b_rel = "data_raw/2026_9_14_gouchang/b_cam2_imu2_raw.mp4"
    _vid(tmp_path, a_rel, route([1, 0, 0], acts), rows, range(24))     # 黑狗
    _vid(tmp_path, b_rel, route([0, 1, 0], acts), rows, range(24))     # 白狗
    embed._cache.clear()

    q = route([1, 0, 0], [lick])[0]                                    # 黑狗那张舔后爪
    ex = (a_rel, -1.0, 0.5)                                            # 把样例自己那一帧排掉

    g = embed.search(q, [a_rel, b_rel], center="global", top_k=5, exclude=ex)
    assert g["hits"][0]["path"] == a_rel                               # 老行为：还是同一只狗

    v = embed.search(q, [a_rel, b_rel], center="video", top_k=5, exclude=ex)
    assert v["center"] == "video"
    assert v["hits"][0]["path"] == b_rel and v["hits"][0]["t"] == 0.0   # 找到别的狗在舔后爪
    # 老调用方传 True 的，直接拿到改进后的行为
    assert embed.search(q, [a_rel, b_rel], center=True, top_k=5, exclude=ex)["center"] == "video"
    assert embed.search(q, [a_rel, b_rel], center=False, top_k=5, exclude=ex)["center"] == "none"


def test_帧太少的那一路退回全局均值_不被减成噪声(tmp_path, monkeypatch):
    from vision_service import embed

    monkeypatch.setattr(embed.config, "EMBED_INDEX_DIR", str(tmp_path))
    rows = np.stack([_row([0.9, 0.9, 0.15, 0.9, 0.9, 0.9])] * 25)
    big = "data_raw/2026_9_14_gouchang/big_cam1_imu1_raw.mp4"
    tiny = "data_raw/2026_9_14_gouchang/tiny_cam2_imu2_raw.mp4"
    e = np.eye(3, dtype="float32")[np.arange(25) % 3]
    _vid(tmp_path, big, e, rows, range(25))
    _vid(tmp_path, tiny, e[:3], rows[:3], range(3))                    # 只有 3 帧
    embed._cache.clear()
    r = embed.search([1.0, 0, 0], [big, tiny], center="video", top_k=50)
    assert r["center"] == "video" and any(h["path"] == tiny for h in r["hits"])
