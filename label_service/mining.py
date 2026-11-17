"""负例挖掘：模型里单独训出来的「容易被判成抓挠的动作」（进食 / 饮水 / 嗅闻 / 舔 / 啃 / 蹭…），
不进正式片段，改成候选给人确认。

为什么：抓挠误报主要来自这些低头、有节律的小动作。它们在「活动」里占得太少，模型学不到跟抓挠的区别，
要单独补标注。标注最贵的是「从几十个小时里找到那几分钟」——让模型先把像的片段挑出来，
人在工作台「疑似」列表里确认 / 排除就行，走的还是疑似抓挠那条现成通道。

正式片段（草稿）里那段时间记作 merge_to（默认「活动」）：项目里的类别不变，跟以前一样。
"""
from __future__ import annotations

from label_service.postprocess import _parse_ts

REASON = "model_neg"


COVER_MAX = 0.5   # 候选有一半以上的时间被盖住才算「已经有人在看了」


def _overlaps(c: dict, spans: list[tuple]) -> bool:
    """按比例算，不是"碰到就算"：2026-11-17 一段 2 分半的饮水（模型概率 ~1.0）开头跟一段 5 秒的抓挠
    擦了 1 秒，按"碰到就丢"整段没了"""
    a, b = _parse_ts(c.get("start_ts")), _parse_ts(c.get("end_ts"))
    if a is None or b is None:
        return True
    dur = (b - a).total_seconds()
    if dur <= 0:
        return True
    cov = sum(max(0.0, (min(b, tb) - max(a, ta)).total_seconds()) for ta, tb in spans)
    return cov / dur > COVER_MAX


def _spans(items) -> list[tuple]:
    out = []
    for s in items or []:
        a, b = _parse_ts(s.get("start_ts")), _parse_ts(s.get("end_ts"))
        if a and b:
            out.append((a, b))
    return out


def mine_negatives(segments: dict[str, list[dict]], candidates: list[dict], classes: list[str],
                   labels: list[str], merge_to: str = "活动", max_per_label: int = 40,
                   scratch_head: str = "抓挠") -> tuple[dict[str, list[dict]], list[dict]]:
    """返回 (改过的 segments, 新候选)。segments 原地改：负例类别的片段挪进 merge_to。

    不给的只有两种（都按「一半以上时间被盖住」算，擦边不算）：跟抓挠重叠（正式片段含「抓挠-xx」二级、
    疑似抓挠候选——那段已经有人在看了），跟**同一类**的已有候选重叠（重复）。跟别的类的规则候选（姿态像舔、振荡像甩头…）重叠照样给：
    那些规则候选一个文件几十条、铺得很开，按"重叠就丢"的话模型判得最准的那段饮水会被它们整段挤掉
    （2026-11-17 实测：11:50–11:53 模型饮水概率 ~1.0，候选里一条都没有）。人看到两条，选对的那条确认。
    上限按类别各算，按平均置信度从高到低留。"""
    mine = [c for c in classes if c in labels]
    if not mine:
        return segments, []

    def is_scratch(name) -> bool:
        name = name or scratch_head   # 候选不带 label = 抓挠（老约定）
        return name == scratch_head or name.startswith(scratch_head + "-")

    scratch = _spans([s for lab, segs in segments.items() if is_scratch(lab) for s in (segs or [])])
    scratch += _spans([c for c in candidates if is_scratch(c.get("label"))])
    out: list[dict] = []
    moved = False
    for lab in mine:
        segs = segments.pop(lab, None) or []
        if segs:
            segments.setdefault(merge_to, []).extend(dict(s) for s in segs)
            moved = True
        taken = scratch + _spans([c for c in candidates if c.get("label") == lab])
        one = [{**s, "label": lab, "reason": REASON} for s in segs]
        one = [c for c in one if not _overlaps(c, taken)]
        one.sort(key=lambda c: c.get("conf_mean") or 0, reverse=True)
        out += one[:max_per_label]
    if moved:
        segments[merge_to].sort(key=lambda s: s.get("start_ts") or "")
    return segments, out
