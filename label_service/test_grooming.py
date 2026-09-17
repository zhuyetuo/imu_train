"""疑似舔/啃候选：只靠加速度姿态，能不能把"像在理毛"的那一段挑出来、
而且**不把睡觉、走路、掉数据挑进来**。

    python -m pytest label_service/test_grooming.py -q

合成一份 10 分钟的 50Hz 加速度：平时趴着（重力沿 +z）、中间插一段舔（头歪 50°
+ 2.5Hz 小幅节律）、一段走路（头也低，但整个身体在颠）、一段歪着头睡（姿态偏
但一动不动）、一段掉数据（六轴全 0）、一段太短的舔。只有那一段舔该出来。
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
# 走包路径 import（label_service.grooming），跟服务里一致；**不把 label_service 目录
# 本身塞进 sys.path**——那里的 queue.py 会盖掉标准库的 queue
sys.path.insert(0, os.path.dirname(_HERE))
G = importlib.import_module("label_service.grooming")

HZ = 50.0


def _rot(tilt_deg: float) -> np.ndarray:
    """把重力从 +z 绕 x 轴转 tilt 度后的单位向量。"""
    t = np.radians(tilt_deg)
    return np.array([0.0, np.sin(t), np.cos(t)], dtype=np.float32)


def synth(total_s: float = 600.0, seed: int = 0):
    """→ (acc, ts, valid, 各段的秒区间)。单位 g。"""
    rng = np.random.default_rng(seed)
    n = int(total_s * HZ)
    t = np.arange(n) / HZ
    acc = np.tile(_rot(0.0), (n, 1)) + rng.normal(0, 0.005, (n, 3)).astype(np.float32)
    valid = np.ones(n, bool)
    spans = {}

    def put(name, s0, s1, tilt, amp, f, noise=0.005):
        a, b = int(s0 * HZ), int(s1 * HZ)
        base = _rot(tilt)
        osc = amp * np.sin(2 * np.pi * f * t[a:b])
        acc[a:b] = base + osc[:, None] * np.array([0.3, 1.0, 0.2], np.float32) + rng.normal(0, noise, (b - a, 3))
        spans[name] = (s0, s1)

    put("lick", 120, 140, tilt=50, amp=0.08, f=2.5)      # 舔：头歪 50°，2.5Hz 小幅
    put("walk", 200, 220, tilt=40, amp=0.8, f=2.0)       # 走路：头也低，但颠得厉害
    put("sleep_tilted", 300, 360, tilt=60, amp=0.0, f=0, noise=0.002)   # 歪着头睡：不动
    a, b = int(400 * HZ), int(420 * HZ)                  # 掉数据：六轴全 0
    acc[a:b] = 0.0
    valid[a:b] = False
    spans["missing"] = (400, 420)
    put("lick_short", 500, 502, tilt=50, amp=0.08, f=2.5)   # 太短
    ts = pd.Series(pd.to_datetime("2026-09-17 10:00:00") + pd.to_timedelta(t, unit="s"))
    return acc, ts, valid, spans


def _sec(c, key, t0=pd.Timestamp("2026-09-17 10:00:00")):
    return (pd.Timestamp(c[key]) - t0).total_seconds()


# ── 主线：只挑出那一段舔 ──────────────────────────────────────────────────


def test_only_the_lick_bout_is_flagged():
    acc, ts, valid, spans = synth()
    cands = G.mine(acc, ts, valid, HZ)
    assert len(cands) == 1, [(c["start_ts"], c["end_ts"], c["tilt_deg"]) for c in cands]
    c = cands[0]
    s0, s1 = spans["lick"]
    assert abs(_sec(c, "start_ts") - s0) <= 2.0
    assert abs(_sec(c, "end_ts") - s1) <= 2.0
    assert c["reason"] == "grooming" and c["label"] == "舔身体"
    assert 40 <= c["tilt_deg"] <= 60


def test_walking_with_head_down_is_not_a_candidate():
    """走路时头也低、姿态也偏，但整个身体在颠——动作量过大要排掉，
    不然一天几小时的散步全成了候选。"""
    acc, ts, valid, spans = synth()
    s0, s1 = spans["walk"]
    for c in G.mine(acc, ts, valid, HZ):
        assert not (_sec(c, "start_ts") < s1 and _sec(c, "end_ts") > s0), "把走路挑进来了"


def test_sleeping_with_head_tilted_is_not_a_candidate():
    """歪着头睡：姿态偏了但一动不动。光看姿态会把一整晚睡觉挑出来。"""
    acc, ts, valid, spans = synth()
    s0, s1 = spans["sleep_tilted"]
    for c in G.mine(acc, ts, valid, HZ):
        assert not (_sec(c, "start_ts") < s1 and _sec(c, "end_ts") > s0), "把睡觉挑进来了"


def test_missing_data_is_not_a_candidate():
    """掉数据的段六轴全 0：重力方向"偏"了 90°，动作量 0。valid_mask 说没数据就没数据。"""
    acc, ts, valid, spans = synth()
    s0, s1 = spans["missing"]
    for c in G.mine(acc, ts, valid, HZ):
        assert not (_sec(c, "start_ts") < s1 and _sec(c, "end_ts") > s0)


def test_a_window_with_too_many_invalid_rows_is_not_trusted_even_if_values_look_like_licking():
    """load_csv 把缺的行 ffill 掉了，填完的值**看着完全正常**——所以不能拿数值判，
    要看 valid_mask。一个窗口里 40% 的行没数据，那段就不算，哪怕填出来的曲线像在舔。"""
    acc, ts, valid, spans = synth()
    s0, s1 = spans["lick"]
    a, b = int(s0 * HZ), int(s1 * HZ)
    valid[a:b][::5] = False          # 每 5 行缺 2 行 → 有效占比 60%
    valid[a:b][1::5] = False
    for c in G.mine(acc, ts, valid, HZ):
        assert not (_sec(c, "start_ts") < s1 and _sec(c, "end_ts") > s0), "有效行不够的窗口也当成候选了"


def test_all_zero_rows_count_as_missing_even_if_mask_says_valid():
    """别的产出源写 0 占位但 mask 是 True：真实佩戴六轴不可能同时精确为 0。

    要害不在"那段会不会成候选"（动作量 0 本来就挡住了），而在**基准**：
    占位 0 是"最安静"的窗口，占了大头就会被当成平时姿态——基准成了零向量，
    整份文件一条候选都挖不出来。"""
    acc, ts, valid, spans = synth()
    valid[:] = True
    rng = np.random.default_rng(7)
    # 60–100s：醒着、姿态没偏、有点小动作（站着看看四周）。基准要是被 0 拉成
    # 零向量，所有窗口的夹角都成了 90°，这一段就会被当成候选
    a, b = int(60 * HZ), int(100 * HZ)
    acc[a:b] = _rot(0.0) + rng.normal(0, 0.05, (b - a, 3)).astype(np.float32)
    acc[int(150 * HZ):] = 0.0        # 后 3/4 全是 0 占位（舔在 120–140s，前面）
    cands = G.mine(acc, ts, valid, HZ)
    assert len(cands) == 1, f"占位 0 把基准拉歪了: {[(c['start_ts'], c['tilt_deg']) for c in cands]}"
    s0, s1 = spans["lick"]
    assert abs(_sec(cands[0], "start_ts") - s0) <= 2.0


def test_too_short_bouts_are_dropped():
    acc, ts, valid, spans = synth()
    s0, s1 = spans["lick_short"]
    for c in G.mine(acc, ts, valid, HZ):
        assert not (_sec(c, "start_ts") < s1 and _sec(c, "end_ts") > s0), "2 秒的碎片也挑进来了"


# ── 基准是相对本次佩戴的 ──────────────────────────────────────────────────


def test_baseline_follows_the_collar_not_the_world():
    """项圈今天转了个角度：整份数据的"平时姿态"跟着转。舔的时候相对平时偏 50°
    才算，绝对方向不重要——否则项圈一滑，一整天都是候选。"""
    acc, ts, valid, spans = synth()
    # 整份数据绕 x 轴再转 35°（模拟项圈滑转）
    t = np.radians(35.0)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]], np.float32)
    acc2 = acc @ R.T
    cands = G.mine(acc2, ts, valid, HZ)
    assert len(cands) == 1
    s0, s1 = spans["lick"]
    assert abs(_sec(cands[0], "start_ts") - s0) <= 2.0


def test_baseline_ignores_walking_windows():
    """狗场白天大半时间在走，走的时候头是低的。走路窗口要是参与基准，
    "平时姿态"就成了走路姿态，舔的偏转量被吃掉一半，挖不出来。"""
    rng = np.random.default_rng(3)
    n = int(600 * HZ)
    t = np.arange(n) / HZ
    acc = np.tile(_rot(0.0), (n, 1)) + rng.normal(0, 0.005, (n, 3)).astype(np.float32)
    valid = np.ones(n, bool)

    def walk(a_s, b_s):
        a, b = int(a_s * HZ), int(b_s * HZ)
        osc = 0.8 * np.sin(2 * np.pi * 2.0 * t[a:b])
        acc[a:b] = _rot(45) + osc[:, None] * np.array([0.3, 1, 0.2], np.float32) + rng.normal(0, 0.01, (b - a, 3))

    # 每两分钟只歇 10 秒，其余都在走（头低 45°）
    for k in range(5):
        walk(k * 120, k * 120 + 110)
    # 240–260 那次歇息里舔了一下（头偏 50°，2.5Hz 小幅）
    a, b = int(240 * HZ), int(260 * HZ)
    acc[a:b] = _rot(50) + (0.08 * np.sin(2 * np.pi * 2.5 * t[a:b]))[:, None] * np.array([0.3, 1, 0.2], np.float32)
    ts = pd.Series(pd.to_datetime("2026-09-17 10:00:00") + pd.to_timedelta(t, unit="s"))
    cands = G.mine(acc, ts, valid, HZ)
    hit = [c for c in cands if _sec(c, "start_ts") < 260 and _sec(c, "end_ts") > 240]
    assert hit, "基准被走路姿态带偏，舔挖不出来了"


def test_baseline_is_local_so_sleeping_and_awake_each_get_their_own():
    """一份文件里前半夜在睡（姿态 A，一动不动），后半段醒着（姿态 B，偏 40°，
    站着看看四周那种小动作）。基准要是整份文件一个，醒着的那一整段都成候选。"""
    rng = np.random.default_rng(5)
    n = int(2400 * HZ)                       # 40 分钟：前 30 分钟睡，后 10 分钟醒
    t = np.arange(n) / HZ
    acc = np.empty((n, 3), np.float32)
    half = int(1800 * HZ)
    acc[:half] = _rot(0.0) + rng.normal(0, 0.002, (half, 3))                       # 睡：不动
    acc[half:] = _rot(40.0) + rng.normal(0, 0.03, (n - half, 3))                   # 醒：小动作
    ts = pd.Series(pd.to_datetime("2026-09-17 02:00:00") + pd.to_timedelta(t, unit="s"))
    cands = G.mine(acc, ts, np.ones(n, bool), HZ)
    # 刚醒那几分钟基准还带着睡姿，标出来一段是可以接受的；但醒着的 10 分钟
    # 不能整段都是候选
    total_flagged = sum(_sec(c, "end_ts") - _sec(c, "start_ts") for c in cands)
    assert total_flagged < 300, f"醒着的 10 分钟被标了 {total_flagged:.0f}s 候选——基准没跟着状态走"


def test_baseline_exists_when_the_whole_file_is_moving_moderately():
    """整份数据都有动作量（没有一刻完全安静）也得给出基准。"""
    acc, ts, valid, spans = synth()
    rng = np.random.default_rng(1)
    acc = acc + rng.normal(0, 0.05, acc.shape).astype(np.float32)
    p = G.GroomParams()
    starts, means, motion, vfrac, _ = G._window_stats(acc, valid, HZ, p)
    base = G.baseline_directions(means, motion, vfrac, starts, HZ, p)
    assert base is not None and base.shape == means.shape


# ── 边界情况 ──────────────────────────────────────────────────────────────


def test_no_timestamps_means_no_candidates():
    acc, _ts, valid, _ = synth()
    assert G.mine(acc, None, valid, HZ) == []


def test_empty_input():
    assert G.mine(np.zeros((0, 3)), pd.Series([], dtype="datetime64[ns]"), np.zeros(0, bool), HZ) == []


def test_works_at_16hz():
    """8-11 之前的数据是 16Hz 存的，nyquist 8Hz，节律频段 1.5–4.5 还在里面。"""
    acc, ts, valid, spans = synth()
    acc16, ts16, valid16 = acc[::3], ts.iloc[::3].reset_index(drop=True), valid[::3]
    cands = G.mine(acc16, ts16, valid16, HZ / 3)
    assert len(cands) == 1
    s0, s1 = spans["lick"]
    assert abs(_sec(cands[0], "start_ts") - s0) <= 2.0


def test_drop_overlapping_removes_candidates_on_top_of_scratch():
    """抓挠时头也会歪，姿态判据会把它当成理毛；那段已经在疑似抓挠里了，别重复。"""
    cands = [
        {"start_ts": "2026-09-17 10:02:00.000", "end_ts": "2026-09-17 10:02:20.000"},
        {"start_ts": "2026-09-17 10:05:00.000", "end_ts": "2026-09-17 10:05:10.000"},
    ]
    scratch = [{"start_ts": "2026-09-17 10:02:15.000", "end_ts": "2026-09-17 10:02:30.000"}]
    kept = G.drop_overlapping(cands, scratch)
    assert [c["start_ts"] for c in kept] == ["2026-09-17 10:05:00.000"]


def test_drop_overlapping_keeps_adjacent():
    """紧挨着不算重叠（跟平台确认时并片段的判据一致：严格 <）。"""
    cands = [{"start_ts": "2026-09-17 10:02:30.000", "end_ts": "2026-09-17 10:02:40.000"}]
    scratch = [{"start_ts": "2026-09-17 10:02:15.000", "end_ts": "2026-09-17 10:02:30.000"}]
    assert len(G.drop_overlapping(cands, scratch)) == 1


def test_ranked_by_score_then_length():
    acc, ts, valid, spans = synth()
    # 再加一段偏得更少的舔，**放在时间上更早的位置**——不排序的话它会排第一
    a, b = int(30 * HZ), int(50 * HZ)
    t = np.arange(b - a) / HZ
    acc[a:b] = _rot(30) + (0.08 * np.sin(2 * np.pi * 2.5 * t))[:, None] * np.array([0.3, 1, 0.2], np.float32)
    cands = G.mine(acc, ts, valid, HZ)
    assert len(cands) == 2
    assert cands[0]["tilt_deg"] > cands[1]["tilt_deg"]


# ── 接进服务的那一段（源码级守卫，不起服务） ───────────────────────────────


def _app_src() -> str:
    src = open(os.path.join(_HERE, "app.py"), encoding="utf-8").read()
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


def test_grooming_is_added_after_refine_and_before_missing_drop():
    """顺序有讲究：边界微调是拿陀螺仪能量对齐抓挠的，对姿态类候选没意义，所以要在
    它**之后**加进来；掉数据那一刀要在**之后**砍，不然掉数据的段会漏进候选。"""
    code = _app_src()
    i_refine = code.index('postprocess.refine_boundaries(result["candidates"]')
    i_groom = code.index('result["candidates"] = result["candidates"] + gc')
    i_holes = code.index('result["candidates"] = postprocess.drop_missing(result["candidates"], holes)')
    assert i_refine < i_groom < i_holes


def test_grooming_candidates_drop_those_overlapping_scratch():
    """抓挠时头也歪，姿态判据会把它当理毛；跟抓挠正式片段/疑似抓挠重叠的要去掉。"""
    code = _app_src()
    assert "_grooming.drop_overlapping(grooming_cands, taken)" in code


def test_candidate_schema_carries_label_and_is_backward_compatible():
    """老调用方不给 label：字段默认 None，平台那边当抓挠。"""
    src = open(os.path.join(_HERE, "app.py"), encoding="utf-8").read()
    body = src.split("class Candidate(Segment):", 1)[1].split("\nclass ", 1)[0]
    assert "label: str | None = None" in body
    assert "tilt_deg: float | None = None" in body


def test_worker_never_lets_grooming_break_inference():
    """候选是锦上添花：挖掘抛异常只能记日志，不能把整份推理拖垮。"""
    src = open(os.path.join(_HERE, "pool.py"), encoding="utf-8").read()
    body = src.split("grooming: list[dict] = []", 1)[1].split("return {", 1)[0]
    assert "except Exception" in body and "log.exception" in body
