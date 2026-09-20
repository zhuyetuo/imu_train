"""分层抽帧给标注：按场地/机位/昼夜/姿势分格轮取，优先挑现在判得最不确定的。"""

from __future__ import annotations

import json
import os

import numpy as np

from vision_service import trainset as ts
from vision_service.pose import DIM, K, NOSE, PAWS


def _row(visible, spread_scale=0.4):
    v = np.zeros(DIM, dtype="float32")
    xy = np.stack([np.linspace(-spread_scale, spread_scale, K),
                   np.linspace(spread_scale, -spread_scale, K)], axis=1).astype("float32")
    for k in visible:
        v[k * 2:k * 2 + 2] = xy[k]
        v[DIM - K + k] = 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def _idx(d, name, path, rows, ts_):
    meta = {"path": path, "pose": True}
    np.savez(os.path.join(d, name), t=np.asarray(ts_, dtype="float32"),
             pose=np.asarray(rows, dtype="float16"),
             box=np.tile([0.2, 0.2, 0.8, 0.8], (len(ts_), 1)).astype("float32"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))


def test_分格_场地机位昼夜姿势():
    """随手抽会抽到一堆白天狗场同一只狗趴着的帧——那种模型本来就会，标了学不到东西。"""
    assert ts._bucket("data_raw/2026_9_18_gouchang/multicam_20260918_030016038_cam2_imu11_raw.mp4",
                      True) == ("gouchang", "cam2", "夜", "蜷着")
    assert ts._bucket("data_raw/2026_9_14_yingpeng/multicam_20260914_140016916_cam5_imu18_raw.mp4",
                      False) == ("yingpeng", "cam5", "昼", "摊开")
    # 19 点算夜、7 点算昼（红外/彩色的分界）
    assert ts._bucket("data_raw/2026_9_14_a/multicam_20260914_190014312_cam1_raw.mp4", False)[2] == "夜"
    assert ts._bucket("data_raw/2026_9_14_a/multicam_20260914_070018354_cam1_raw.mp4", False)[2] == "昼"
    assert ts._bucket("认不出来的名字.mp4", False)[:2] == ("?", "?")     # 不炸


def test_优先挑判得最不确定的帧(tmp_path):
    """全挑最好认的等于白标——要学的正是现在测不全的那些。"""
    d = str(tmp_path)
    rows = [_row(range(K)),              # 17 个点全可见：最好认，不该优先
            _row((NOSE, PAWS[0])),       # 只两个点：最该标
            _row((NOSE, *PAWS))]
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/a_cam1_imu1_raw.mp4", rows, [1, 2, 3])
    got = ts.scan(d, max_per_video=2)
    assert [c["n_vis"] for c in got] == [2, 5]        # 按可见点数从少到多
    assert got[0]["t"] == 2.0 and got[0]["box"] == [0.2, 0.2, 0.8, 0.8]


def test_轮取_把量摊到有数据的格子上_同一路不刷屏():
    """不用"每格固定取几条"：格子数是数据决定的，固定配额会在小格子上取空、
    在大格子上砍掉。而且同一路视频里挨着的帧长得一模一样，一路最多留三条。"""
    big = [{"path": f"data_raw/2026_9_14_gouchang/v{i // 10}_cam1_imu1_raw.mp4", "t": float(i),
            "n_vis": 3, "spread": 0.1, "box": None, "bucket": ("gouchang", "cam1", "昼", "摊开")}
           for i in range(40)]
    small = [{"path": "data_raw/2026_9_14_yingpeng/x_cam3_imu3_raw.mp4", "t": 1.0,
              "n_vis": 3, "spread": 0.1, "box": None, "bucket": ("yingpeng", "cam3", "夜", "蜷着")}]
    got = ts.pick(big + small, 8)
    from collections import Counter
    c = Counter(x["bucket"] for x in got)
    assert len(c) == 2 and c[("yingpeng", "cam3", "夜", "蜷着")] == 1   # 小格子那一条没被挤掉
    # 同一路视频最多三条
    per_video = Counter(x["path"] for x in got)
    assert max(per_video.values()) <= 3
    assert len(ts.pick(big + small, 500)) <= len(big) + len(small)      # 要得比有的多也不炸
    assert ts.pick([], 5) == []


def test_导出_留一大圈边_小目标放大_manifest带分层(tmp_path, monkeypatch):
    """标关键点要看得见四肢伸出去的样子，紧贴框裁会把爪子和尾巴切掉——
    而那几个点正是最难标也最要紧的。"""
    import csv

    import cv2

    from vision_service import seek

    monkeypatch.setattr(seek, "iter_frames",
                        lambda *a, **kw: iter([(1.0, np.full((200, 200, 3), 120, np.uint8))]))
    monkeypatch.setattr(ts, "config", ts.config)
    root = tmp_path / "nas"
    (root / "data_raw" / "d").mkdir(parents=True)
    (root / "data_raw" / "d" / "a_cam1_imu1_raw.mp4").write_bytes(b"0")

    picks = [{"path": "data_raw/d/a_cam1_imu1_raw.mp4", "t": 1.0, "n_vis": 3,
              "box": [0.25, 0.25, 0.75, 0.75], "bucket": ("gouchang", "cam1", "夜", "蜷着")},
             {"path": "data_raw/d/没有.mp4", "t": 1.0, "n_vis": 3, "box": None,
              "bucket": ("yingpeng", "cam3", "昼", "摊开")}]
    out = str(tmp_path / "train")
    msg = ts.export(picks, out, str(root), max_side=640)

    assert "导了 1 张" in msg and "gouchang/夜/蜷着" in msg        # 视频不在的那条跳过
    assert "先标 50 张就停下来评一次" in msg and "45% 的帧人也判不了" in msg
    assert "0 左眼" in msg and "16 右后爪" in msg                  # 关键点顺序要写清楚
    img = cv2.imread(os.path.join(out, "0000.jpg"))
    # 框 100x100，四周各留 60% → 200x200（被原图边界截住），再放大到 640
    assert img.shape[0] == 640 and img.shape[1] == 640
    rows = list(csv.DictReader(open(os.path.join(out, "manifest.csv"), encoding="utf-8-sig")))
    assert rows[0]["场地"] == "gouchang" and rows[0]["姿势"] == "蜷着"
    assert rows[0]["现在测到几个点"] == "3"
