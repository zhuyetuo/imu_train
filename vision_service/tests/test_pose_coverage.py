"""姿态覆盖率统计：分母是「有狗的帧」不是「采样帧」；没开姿态建的索引要单独说明。"""

from __future__ import annotations

import json
import os

import numpy as np

from vision_service import pose_coverage as pc


def _idx(d, name, path, sampled, with_dog, with_pose, pose_on=True):
    meta = {"path": path, "sampled": sampled, "with_dog": with_dog,
            "with_pose": with_pose, "pose": pose_on}
    np.savez(os.path.join(d, name), t=np.zeros(1, dtype="float32"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))


def test_扫索引_解析场地机位_分母是有狗的帧(tmp_path):
    d = str(tmp_path)
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/multicam_x_cam4_imu15_raw.mp4", 3600, 1000, 800)
    _idx(d, "b.npz", "data_raw/2026_9_14_gouchang/multicam_x_cam7_raw.mp4", 3600, 500, 50)
    _idx(d, "c.npz", "data_raw/2026_9_12_yingpeng/multicam_x_cam1_imu1_raw.mp4", 3600, 2000, 1900)
    rows = pc.scan(d)
    assert {r["site"] for r in rows} == {"gouchang", "yingpeng"}
    assert {r["cam"] for r in rows} == {"cam4", "cam7", "cam1"}
    assert {r["day"] for r in rows} == {"2026_9_14", "2026_9_12"}

    txt = pc.report(rows)
    # 覆盖率 = 2750/3500 = 78.6%，不是 2750/10800
    assert "78.6%" in txt and "够用" in txt
    # 公共区那一路单独一行，跟单间机位分开
    assert "gouchang cam7" in txt and "gouchang cam4" in txt


def test_没开姿态建的索引_不算成测不到(tmp_path):
    d = str(tmp_path)
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/x_cam1_imu9_raw.mp4", 100, 100, 0, pose_on=False)
    txt = pc.report(pc.scan(d))
    assert "没开姿态" in txt and "不代表测不到" in txt


def test_覆盖率低时给的是排查方向_不是结论(tmp_path):
    d = str(tmp_path)
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/x_cam1_imu9_raw.mp4", 1000, 900, 90)
    txt = pc.report(pc.scan(d), worst=5)
    assert "偏低" in txt and "最测不到姿态的" in txt


def test_索引目录空_说清楚下一步(tmp_path):
    assert "建画面索引" in pc.report(pc.scan(str(tmp_path)))
    assert "建画面索引" in pc.report(pc.scan(str(tmp_path / "不存在")))
