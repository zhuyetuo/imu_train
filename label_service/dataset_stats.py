"""训练集统计：这一次训练到底喂了什么。

导出数据集那页有一张按标签的统计，但那是导出时的样子；真正进模型的是
**归并之后**的类别、**几份数据集合在一起**、再按窗口切。这里按训练任务算一份：
每个类别有多少段、多少秒、大约多少个窗口、来自几个任务 / 几个采集日——
后两个是"多样性"：抓挠有 1000 秒但全是一只狗一天的，模型学到的是那只狗那天。

还顺手给几句提示（占比太低、来源太单一、短段太多），界面上原样显示。
提示只是提示，不替人决定。
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import pandas as pd

from label_service import config

_DATE_RE = re.compile(r"(20\d{2})[-_.](\d{1,2})[-_.](\d{1,2})")


def _day_of(task: dict, fallback: str) -> str:
    """采集日：从 CSV 路径或样本编号里认，认不出就用数据集名。"""
    data = task.get("data") or {}
    for s in (data.get("csv") or "", data.get("sample_code") or ""):
        m = _DATE_RE.search(str(s))
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return fallback


def _window_geom() -> tuple[float, float]:
    try:
        import yaml
        with open(os.path.join(config.REPO_ROOT, "configs", "data.yaml"), encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return float(d.get("window_seconds", 2.0)), float(d.get("stride_seconds", 1.0))
    except Exception:  # noqa: BLE001
        return 2.0, 1.0


def _remap_table(dataset_spec: dict) -> dict:
    """train.py 最后过的那张表：界面给了归并表就是「目标名 → 自己」，没给就是默认 3 类表。
    表里没有的类别会被 train.py 丢掉，这里也照样记成「丢弃」。"""
    remap = dataset_spec.get("label_remap") or {}
    if remap:
        return {c: c for c in dict.fromkeys(remap.values())}
    try:
        import yaml
        with open(os.path.join(config.REPO_ROOT, "configs", "remap_custom_3class.yaml"), encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return {str(k): str(v) for k, v in d.items() if not str(k).startswith("#")}
    except Exception:  # noqa: BLE001
        return {}


def _seconds(seg: dict) -> float | None:
    val = seg.get("value") or {}
    try:
        t0, t1 = pd.to_datetime(val.get("start")), pd.to_datetime(val.get("end"))
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(t0) or pd.isna(t1):
        return None
    return max(0.0, (t1 - t0).total_seconds())


def compute(dataset_spec: dict, run_dates: list[str], window_s: float | None = None,
            stride_s: float | None = None) -> dict:
    """run_dates：data/raw_custom/<名字>/merged_tmp.json 那些名字（主 + 一起训练的）。
    merged_tmp.json 已经按归并表改写过类别（prepare_export 干的），这里再过一遍训练那张表。"""
    w_s, s_s = _window_geom()
    window_s = window_s or w_s
    stride_s = stride_s or s_s
    table = _remap_table(dataset_spec)

    per: dict[str, dict] = defaultdict(lambda: {"segments": 0, "seconds": 0.0, "windows": 0, "short": 0,
                                                "tasks": set(), "days": set()})
    dropped: dict[str, dict] = defaultdict(lambda: {"segments": 0, "seconds": 0.0})
    days_all: set[str] = set()
    tasks_all = 0
    per_day: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for rd in run_dates:
        p = os.path.join(config.REPO_ROOT, "data", "raw_custom", rd, "merged_tmp.json")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            tasks = json.load(f)
        fallback = rd.split("__job")[0]
        for t in tasks:
            tasks_all += 1
            day = _day_of(t, fallback)
            days_all.add(day)
            for ann in t.get("annotations") or []:
                for seg in ann.get("result") or []:
                    labels = (seg.get("value") or {}).get("timeserieslabels") or []
                    if not labels:
                        continue
                    sec = _seconds(seg)
                    if sec is None:
                        continue
                    raw = str(labels[0])
                    cls = table.get(raw) if table else raw
                    if cls is None:
                        dropped[raw]["segments"] += 1
                        dropped[raw]["seconds"] += sec
                        continue
                    d = per[cls]
                    d["segments"] += 1
                    d["seconds"] += sec
                    d["windows"] += int((sec - window_s) // stride_s) + 1 if sec >= window_s else 0
                    if sec < window_s:
                        d["short"] += 1
                    d["tasks"].add(t.get("id"))
                    d["days"].add(day)
                    per_day[day][cls] += sec

    total_sec = sum(d["seconds"] for d in per.values()) or 1.0
    total_win = sum(d["windows"] for d in per.values())
    rows = []
    for cls, d in sorted(per.items(), key=lambda kv: -kv[1]["seconds"]):
        rows.append({
            "label": cls, "segments": d["segments"], "seconds": round(d["seconds"], 1),
            "windows": d["windows"], "share": round(d["seconds"] / total_sec, 4),
            "n_tasks": len(d["tasks"]), "n_days": len(d["days"]), "short_segments": d["short"],
        })
    hints: list[str] = []
    if rows:
        biggest = rows[0]
        for r in rows:
            if r["windows"] == 0:
                hints.append(f"「{r['label']}」一个窗口都切不出来（{r['segments']} 段全比 {window_s:g} 秒短），这一类等于没训")
                continue
            if r["share"] < 0.05:
                hints.append(f"「{r['label']}」只占 {r['share'] * 100:.1f}%（{r['windows']} 窗，最多的「{biggest['label']}」有 {biggest['windows']} 窗）："
                             f"少的类别容易被当噪声，先确认标注没漏，再考虑「类别均衡」把多的砍一砍")
            if r["n_days"] <= 2 and len(days_all) > 3:
                hints.append(f"「{r['label']}」只来自 {r['n_days']} 个采集日（整份数据有 {len(days_all)} 天）：模型学的可能是那几天的狗和环境，换一只狗就不认")
            if r["segments"] and r["short_segments"] / r["segments"] > 0.3:
                hints.append(f"「{r['label']}」{r['short_segments']}/{r['segments']} 段比一个窗口（{window_s:g} 秒）还短，进不了训练：要么是手滑点出来的碎段，要么该把窗口调小")
    if dropped:
        names = "、".join(f"{k}（{v['segments']} 段）" for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]['segments'])[:6])
        hints.append(f"归并表里没有、被丢掉的类别：{names}。想用就在提交训练时把它们归并进某一类")
    return {
        "window_s": window_s, "stride_s": stride_s,
        "n_tasks": tasks_all, "n_days": len(days_all),
        "total_seconds": round(total_sec, 1), "total_windows": total_win,
        "rows": rows,
        "dropped": [{"label": k, "segments": v["segments"], "seconds": round(v["seconds"], 1)}
                    for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]["segments"])],
        "per_day": [{"day": day, **{k: round(v, 1) for k, v in sorted(cls.items())}} for day, cls in sorted(per_day.items())],
        "hints": hints,
    }
