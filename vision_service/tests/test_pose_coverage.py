"""关键点可见率：读 pose 向量里的可见位，不是数"有没有姿态向量"。"""

from __future__ import annotations

import json
import os

import numpy as np

from vision_service import pose_coverage as pc
from vision_service.pose import DIM, K, NOSE, PAWS


def _rows(n: int, visible: list[int]) -> np.ndarray:
    """n 帧，每帧这些关键点可见。存的时候整条向量 L2 归一化过，所以可见位不是 1。"""
    v = np.zeros((n, DIM), dtype="float32")
    for k in visible:
        v[:, DIM - K + k] = 1.0
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(norm, 1e-9)).astype("float16")


def _idx(d, name, path, sampled, rows: np.ndarray, pose_on=True):
    meta = {"path": path, "sampled": sampled, "with_dog": len(rows), "pose": pose_on}
    np.savez(os.path.join(d, name), t=np.zeros(len(rows), dtype="float32"), pose=rows,
             meta=np.array(json.dumps(meta, ensure_ascii=False)))


def test_可判部位要鼻子和爪同时可见_归一化过的可见位也认得出(tmp_path):
    d = str(tmp_path)
    # 100 帧鼻子+左前爪都在（可判部位），100 帧只有鼻子（判不了），100 帧只有爪（判不了）
    rows = np.concatenate([_rows(100, [NOSE, PAWS[0]]), _rows(100, [NOSE]), _rows(100, [PAWS[0]])])
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/multicam_x_cam4_imu15_raw.mp4", 900, rows)
    r = pc.scan(d)[0]
    assert r["with_dog"] == 300 and r["part_ok"] == 100 and r["nose_ok"] == 200
    assert r["paws4_ok"] == 0 and r["site"] == "gouchang" and r["cam"] == "cam4"

    txt = pc.report(r and pc.scan(d))
    assert " 33.3%" in txt and "能用但要挑" in txt          # 100/300


def test_给框必出点_所以不拿有没有向量当指标(tmp_path):
    """17 个点全部低于分数线：向量还在（top-down 必出点），但一个部位也判不了。
    老版本把这种算成「有姿态 100%」，等于量了个寂寞。"""
    d = str(tmp_path)
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/x_cam1_imu9_raw.mp4", 500, _rows(500, []))
    rows = pc.scan(d)
    assert rows[0]["with_dog"] == 500 and rows[0]["part_ok"] == 0
    txt = pc.report(rows, worst=5)
    assert "偏低" in txt and "最判不出部位的" in txt
    assert "恒等于 100%" in txt and "不是指标" in txt       # 报告里写明白这个坑


def test_四爪全可见和均可见数(tmp_path):
    d = str(tmp_path)
    _idx(d, "a.npz", "data_raw/2026_9_18_yingpeng/x_cam3_imu3_raw.mp4", 10,
         _rows(10, [NOSE, *PAWS]))
    txt = pc.report(pc.scan(d))
    assert "四爪全 100.0%" in txt and f"均可见  5.0/{K}" in txt
    assert "够用" in txt


def test_没开姿态建的索引_不算成测不准(tmp_path):
    d = str(tmp_path)
    _idx(d, "a.npz", "data_raw/2026_9_14_gouchang/x_cam1_imu9_raw.mp4", 100, _rows(100, []),
         pose_on=False)
    txt = pc.report(pc.scan(d))
    assert "没开姿态" in txt and "不代表测不准" in txt


def test_老索引没有pose那一列_不炸_退回meta(tmp_path):
    d = str(tmp_path)
    meta = {"path": "data_raw/2026_9_14_gouchang/x_cam1_imu9_raw.mp4", "sampled": 100,
            "with_dog": 80, "pose": True}
    np.savez(os.path.join(d, "a.npz"), t=np.zeros(1, dtype="float32"),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    r = pc.scan(d)[0]
    assert r["with_dog"] == 80 and r["part_ok"] == 0


def test_索引目录空_说清楚下一步(tmp_path):
    assert "建画面索引" in pc.report(pc.scan(str(tmp_path)))
    assert "建画面索引" in pc.report(pc.scan(str(tmp_path / "不存在")))
