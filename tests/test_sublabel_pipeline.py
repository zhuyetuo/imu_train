"""拆了二级标签之后，推理这一路上几处按名字硬匹配的地方会悄悄失效。

模型类别从「抓挠」变成「抓挠-头颈耳」「抓挠-躯干」之后，下面每一处都
**不报错、但什么都出不来**，而平台那边看到的只是"模型没检出来"：

  TARGET_LABELS        写死一串，新类别不在里面 → 一个片段都不输出
  STABLE_EVENT_LABELS  精确相等 → 二级抓挠掉进"状态"那套平滑，几秒的段被抹掉
  边界微调 / 舔啃去重   同样按名字取，二级的取不到
"""

import sys

import pytest

sys.path.insert(0, ".")


def _is_event(lab: str, event_labels=("抓挠", "甩身体")) -> bool:
    """跟 postprocess 里那个判据必须一致——这里是它的规格说明。"""
    return any(lab == e or lab.startswith(e + "-") for e in event_labels)


def test_二级抓挠要按事件处理_不是状态():
    """事件走滞回+最短窗口那套，状态走平滑。二级抓挠掉进状态的话，
    一段几秒的抓挠会被直接抹平，看起来就像模型没检出来。"""
    for lab in ("抓挠", "抓挠-头颈耳", "抓挠-躯干", "甩身体"):
        assert _is_event(lab), lab
    for lab in ("活动", "睡觉", "静止/休息", "未佩戴", "甩头/抖身"):
        assert not _is_event(lab), lab


def test_前缀判据在postprocess里是同一份():
    src = open("label_service/postprocess.py", encoding="utf-8").read()
    assert 'lab.startswith(e + "-")' in src, "postprocess 改回精确匹配了，二级标签会掉进状态那一套"


def test_边界微调和舔啃去重也按前缀认():
    src = open("label_service/app.py", encoding="utf-8").read()
    # 原来是 for lab in config.STABLE_EVENT_LABELS: ... segments.get(lab)
    assert "config.STABLE_EVENT_LABELS:\n                postprocess.refine_boundaries" not in src
    assert 'lab.startswith(scratch_head + "-")' in src


def test_目标类别默认跟着模型走():
    """写死一串的话，模型能输出的新类别反倒不在里面，一个片段都出不来。"""
    from label_service import config

    assert config.TARGET_LABELS == [], "默认该是空的（= 跟着模型的 classes 走）"
    src = open("label_service/pool.py", encoding="utf-8").read()
    assert 'config.TARGET_LABELS or list(b["classes"])' in src


def test_环境变量写了还是听环境变量(monkeypatch):
    """想锁死只输出某几类的场合还得留着。"""
    monkeypatch.setenv("TARGET_LABELS", "抓挠,活动")
    import importlib

    from label_service import config as c

    importlib.reload(c)
    try:
        assert c.TARGET_LABELS == ["抓挠", "活动"]
    finally:
        monkeypatch.delenv("TARGET_LABELS")
        importlib.reload(c)
