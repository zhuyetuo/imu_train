"""抓挠吞并甩身体：那个"3 秒"以前根本没起作用。

    python -m pytest label_service/test_shake_absorb.py -q

原来的判据是**相邻窗口之间的间隔** ≤ shake_absorb_s：

    while (i0-1) in shake_idx and (zones[i0][0] - zones[i0-1][1]).total_seconds() <= absorb_s:

而 majority 模式下相邻窗口的 zone 是**首尾相接**的（见 `_zones`：
`zones[k] = (ts[k], ts[k+1])`），所以那个间隔**恒等于 0**。
于是 `0 <= absorb_s` 对任何非负值都成立，while 一路链式吞到底：

  · 参数写 3 秒，实际把一串 4.5 秒的甩身体**整段**吞掉了
  · 写 0 也关不掉（`0 <= 0` 为真）——我一度让人这么关，关不掉

现在的判据是**从抓挠边界往外扩了多少秒（累计）**，并且 `<= 0 = 关闭`
（跟 spectral_min 一个约定）。
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_pp():
    """按文件路径加载 postprocess，**不把 label_service 塞进 sys.path**。

    那个目录里有个 queue.py 会盖掉标准库的 queue——端侧服务踩过一次，
    80 个样本全军覆没。
    """
    spec = importlib.util.spec_from_file_location(
        "_pp_under_test", os.path.join(_HERE, "postprocess.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_pp_under_test"] = m      # dataclass 要在 sys.modules 找到自己
    spec.loader.exec_module(m)
    return m


pp = _load_pp()

CLASSES = ["活动", "睡觉", "抓挠", "未佩戴", "甩身体"]
T0 = dt.datetime(2026, 9, 16, 10, 0, 0)
WINDOW_S, STRIDE_S = 1.0, 0.5


def _win(i, lab, conf=0.9):
    probs = {c: (1 - conf) / (len(CLASSES) - 1) for c in CLASSES}
    probs[lab] = conf
    return {"ts": (T0 + dt.timedelta(seconds=STRIDE_S * i)).strftime(
                "%Y-%m-%d %H:%M:%S.%f")[:-3],
            "label": lab, "conf": conf, "probs": probs, "spec": None}


def _params(absorb):
    return pp.StableParams(
        event_labels=("抓挠", "甩身体"), shake_absorb_s=absorb,
        smooth_windows=1, min_state_s=0, event_gap_s=4,
        event_min_windows=1, event_min_mean=0.0, event_single_conf=0.0)


def _run(windows, absorb, algo="stable"):
    segs = pp.stabilize(windows, CLASSES, ["抓挠", "甩身体"],
                        WINDOW_S, STRIDE_S, "majority", _params(absorb), algo=algo)
    def total(lab):
        return round(sum(
            (dt.datetime.strptime(s["end_ts"], "%Y-%m-%d %H:%M:%S.%f")
             - dt.datetime.strptime(s["start_ts"], "%Y-%m-%d %H:%M:%S.%f")
             ).total_seconds() for s in segs.get(lab, [])), 1)
    return total("抓挠"), total("甩身体")


#: 2 个抓挠窗口（1.0s）+ 紧跟 8 个甩身体窗口（4.5s）。测**往后扩**那条分支
SCRATCH_THEN_SHAKE = ([_win(i, "抓挠") for i in range(2)]
                      + [_win(i, "甩身体") for i in range(2, 10)])

#: 反过来：8 个甩身体在前、2 个抓挠在后。测**往前扩**那条分支。
#
# 两个方向是两段独立的 while，**必须各测一遍**：第一版只有上面那个用例，
# 于是"把往前扩那条改回原来的 bug"这个变异活了下来——两段代码长得几乎一样，
# 只改一段看不出来。
SHAKE_THEN_SCRATCH = ([_win(i, "甩身体") for i in range(8)]
                      + [_win(i, "抓挠") for i in range(8, 10)])


def test_zero_actually_disables_it():
    """`STABLE_SHAKE_ABSORB_S=0` 必须是**关掉**。

    改之前 0 关不掉（相邻间隔恒为 0，`0 <= 0` 为真），而"我已经关掉了"
    这个误判会让人把现象归到模型头上，白查一圈。
    """
    scratch, shake = _run(SCRATCH_THEN_SHAKE, 0.0)
    assert shake == 4.5, f"甩身体应该一秒都没被吞，实际只剩 {shake}s"
    assert scratch == 1.0, f"抓挠应该还是原来那 1 秒，实际 {scratch}s"


@pytest.mark.parametrize("absorb", [-1.0, -0.5])
def test_negative_also_disables_it(absorb):
    assert _run(SCRATCH_THEN_SHAKE, absorb) == (1.0, 4.5)


# ── 往前扩（甩身体在抓挠**之前**）──────────────────────────────────────────
#
# 跟往后扩是两段独立的 while，长得几乎一样。只测一个方向的话，
# 另一个方向改回原来的 bug 也看不出来。


#: 这个方向上"一秒没被吞"时的基准时长。
#
# **跟正方向的 (1.0, 4.5) 不一样**，别照抄：每个窗口的 zone 只有一个步长宽
# （0.5s），只有**最后一个**窗口是整窗宽（1.0s，见 _zones）。
# 所以 8 个甩身体在前 = 8×0.5 = 4.0s，2 个抓挠在后 = 0.5 + 1.0 = 1.5s。
# 第一版直接照抄了正方向的数，红在这里。
BACK_BASE_SCRATCH, BACK_BASE_SHAKE = 1.5, 4.0


def test_backward_zero_actually_disables_it():
    scratch, shake = _run(SHAKE_THEN_SCRATCH, 0.0)
    assert shake == BACK_BASE_SHAKE, f"前面那些甩身体应该一秒没被吞，实际剩 {shake}s"
    assert scratch == BACK_BASE_SCRATCH


def test_backward_absorbs_at_most_the_configured_seconds():
    """往前扩也要被参数限制住——改之前这个方向同样是链式吞到底。"""
    scratch, shake = _run(SHAKE_THEN_SCRATCH, 3.0)
    assert shake > 0, f"前面 {BACK_BASE_SHAKE} 秒的甩身体被 3 秒的参数整段吞掉了"
    assert scratch <= BACK_BASE_SCRATCH + 3.0 + STRIDE_S, \
        f"抓挠涨到 {scratch}s，超过了「原本 {BACK_BASE_SCRATCH} 秒 + 最多吞 3 秒」"


def test_backward_is_monotonic():
    s3, k3 = _run(SHAKE_THEN_SCRATCH, 3.0)
    s1, k1 = _run(SHAKE_THEN_SCRATCH, 1.0)
    s0, k0 = _run(SHAKE_THEN_SCRATCH, 0.0)
    assert s3 > s1 > s0, f"{s3} / {s1} / {s0}"
    assert k3 < k1 < k0, f"{k3} / {k1} / {k0}"


def test_backward_still_absorbs_a_short_shake():
    """反方向也别把规则本来要解决的问题修坏了。"""
    ws = ([_win(i, "活动") for i in range(6)]
          + [_win(i, "甩身体") for i in range(6, 8)]     # 1.0 秒
          + [_win(i, "抓挠") for i in range(8, 12)])
    assert _run(ws, 3.0)[1] == 0.0, "紧挨着抓挠的 1 秒甩身体应该被吞"
    assert _run(ws, 0.0)[1] > 0.0


def test_it_absorbs_at_most_the_configured_seconds():
    """**参数要真的限制住吞多少。**

    改之前是链式的：只看相邻间隔，所以 3 秒的参数把 4.5 秒的甩身体
    整段吞光了——参数形同虚设。
    """
    scratch, shake = _run(SCRATCH_THEN_SHAKE, 3.0)
    assert shake > 0, "4.5 秒的甩身体被 3 秒的参数整段吞掉了——参数没起作用"
    assert scratch <= 1.0 + 3.0 + STRIDE_S, \
        f"抓挠涨到 {scratch}s，超过了「原本 1 秒 + 最多吞 3 秒」"


def test_a_smaller_value_absorbs_less():
    """单调性：调小就该吞得少。没有这条的话，参数动了但没反应也看不出来。"""
    s3, k3 = _run(SCRATCH_THEN_SHAKE, 3.0)
    s1, k1 = _run(SCRATCH_THEN_SHAKE, 1.0)
    s0, k0 = _run(SCRATCH_THEN_SHAKE, 0.0)
    assert s3 > s1 > s0, f"抓挠时长应该随参数单调递减：{s3} / {s1} / {s0}"
    assert k3 < k1 < k0, f"甩身体时长应该随参数单调递增：{k3} / {k1} / {k0}"


def test_a_short_shake_next_to_scratch_is_still_absorbed():
    """规则本来要解决的问题**别修坏了**：抓完甩一下，那一两个窗口该并进抓挠。

    只测"别吞太多"的话，把整段吞并逻辑删掉也能通过。
    """
    ws = ([_win(i, "抓挠") for i in range(4)]
          + [_win(i, "甩身体") for i in range(4, 6)]      # 1.0 秒
          + [_win(i, "活动") for i in range(6, 12)])
    _, shake_on = _run(ws, 3.0)
    _, shake_off = _run(ws, 0.0)
    assert shake_on == 0.0, f"紧挨着抓挠的 1 秒甩身体应该被吞，实际剩 {shake_on}s"
    assert shake_off > 0.0, "关掉之后它该留着"


def test_viterbi_path_too():
    """稳定版 v2（viterbi）走的是同一段吞并代码，别只在 stable 下验。"""
    assert _run(SCRATCH_THEN_SHAKE, 0.0, algo="viterbi")[1] == 4.5


def test_config_default_is_documented_as_seconds():
    """config 里那句注释跟行为一致：说的是"多少秒内"，那就得真的按秒限制。"""
    with open(os.path.join(_HERE, "config.py"), encoding="utf-8") as f:
        src = f.read()
    assert "STABLE_SHAKE_ABSORB_S" in src
    line = next(ln for ln in src.splitlines() if ln.startswith("STABLE_SHAKE_ABSORB_S"))
    assert "秒" in line
