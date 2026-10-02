"""训练完回放：拿刚训好的模型把训练集自己再预测一遍，挑出跟人标的对不上的段。

对不上有三种可能，这里分不出来，但能把它们摆到人面前：
  - 人标错了（最值钱：改掉它，下一轮就干净了）
  - 模型真不行（这一类数据太少 / 太单一）
  - 边界问题（段的起止跟动作对不齐，窗口里混了别的）

按**段**比，不按窗口：一段人标的抓挠，里面的窗口多数投票得一个预测类别，
跟人标的不一样就是一行错例。另外在**没标注的时间**里，模型连着几个窗口报
事件类（抓挠 / 甩身体）的也列出来——那是"误识别"或者"漏标"，人看一眼就知道。

走的是服务里正式的推理路（同一个进程池、同样的重采样和窗口几何），
所以这里看到的就是预标注时会看到的。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter, defaultdict

import pandas as pd

from label_service import config

log = logging.getLogger("label_service.review")

KIND_MISS = "漏识别"      # 人标了事件类，模型没认出来
KIND_FALSE = "误识别"     # 模型报了事件类，人标的不是
KIND_CONFUSE = "混淆"     # 两边都不是事件类，但类别不同（活动 ↔ 睡觉）
KIND_UNLABELED = "未标注区报事件"   # 没人标的时间里模型连着报事件类


def _is_event(label: str, event_labels: list[str]) -> bool:
    return any(label == e or label.startswith(e + "-") for e in event_labels)


def _table(dataset_spec: dict) -> dict:
    from label_service.dataset_stats import _remap_table
    return _remap_table(dataset_spec)


def _load_tasks(run_dates: list[str]) -> list[tuple[str, dict]]:
    out = []
    for rd in run_dates:
        p = os.path.join(config.REPO_ROOT, "data", "raw_custom", rd, "merged_tmp.json")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            for t in json.load(f):
                out.append((rd, t))
    return out


def _csv_full_path(task: dict) -> str | None:
    name = (task.get("data") or {}).get("csv")
    if not name:
        return None
    link = os.path.join(config.REPO_ROOT, "data", "raw_wit", os.path.basename(name))
    if os.path.exists(link):
        return os.path.realpath(link)
    full = name if os.path.isabs(name) else os.path.join(config.NAS_ROOT, name)
    return full if os.path.isfile(full) else None


def _segments(task: dict, table: dict) -> list[dict]:
    """人标的段：{label, t0, t1, start_ms, end_ms}。类别过训练那张表，表里没有的跳过。"""
    segs = []
    for ann in task.get("annotations") or []:
        for seg in ann.get("result") or []:
            v = seg.get("value") or {}
            labels = v.get("timeserieslabels") or []
            if not labels or not v.get("start") or not v.get("end"):
                continue
            raw = str(labels[0])
            cls = table.get(raw) if table else raw
            if cls is None:
                continue
            try:
                t0, t1 = pd.to_datetime(v["start"]), pd.to_datetime(v["end"])
            except Exception:  # noqa: BLE001
                continue
            if pd.isna(t0) or pd.isna(t1) or t1 <= t0:
                continue
            segs.append({"label": cls, "t0": t0, "t1": t1,
                         "start_ms": v.get("start_ms"), "end_ms": v.get("end_ms")})
    segs.sort(key=lambda s: s["t0"])
    return segs


def _review_task(task: dict, result: dict, table: dict, window_s: float, stride_s: float,
                 event_labels: list[str], day: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """→ (错例行, [(人标, 预测)] 每段一对，给混淆矩阵)。"""
    wins = result.get("windows") or []
    if not wins:
        return [], []
    w_ts = pd.to_datetime([w["ts"] for w in wins])
    w_lab = [w["label"] for w in wins]
    w_probs = [w.get("probs") or {} for w in wins]
    segs = _segments(task, table)
    # CSV 起点：段的绝对时间 − 相对毫秒。拿它把窗口时间换算成 seek 用的毫秒
    csv_start = None
    for s in segs:
        if s["start_ms"] is not None:
            csv_start = s["t0"] - pd.Timedelta(milliseconds=int(s["start_ms"]))
            break
    if csv_start is None:
        csv_start = w_ts[0]
    w_ms = ((w_ts - csv_start).total_seconds() * 1000).astype(int)
    win_td = pd.Timedelta(seconds=window_s)
    # 掉数据的时间段：里面的窗口一律不算
    missing = []
    for m in result.get("missing") or []:
        try:
            missing.append((pd.to_datetime(m["start_ts"]), pd.to_datetime(m["end_ts"])))
        except Exception:  # noqa: BLE001
            pass

    def in_missing(i: int) -> bool:
        a, b = w_ts[i], w_ts[i] + win_td
        return any(a < me and b > ms for ms, me in missing)

    task_id = task.get("id")
    sample_code = (task.get("data") or {}).get("sample_code")
    rows: list[dict] = []
    pairs: list[tuple[str, str]] = []
    covered = [False] * len(wins)
    for s in segs:
        # 整个窗口落在段里的；一个都没有（段比窗口短）就退回"有重叠"
        idx = [i for i in range(len(wins)) if w_ts[i] >= s["t0"] - win_td * 0.25 and w_ts[i] + win_td <= s["t1"] + win_td * 0.25]
        if not idx:
            idx = [i for i in range(len(wins)) if w_ts[i] < s["t1"] and w_ts[i] + win_td > s["t0"]]
        for i in [i for i in range(len(wins)) if w_ts[i] < s["t1"] and w_ts[i] + win_td > s["t0"]]:
            covered[i] = True
        idx = [i for i in idx if not in_missing(i)]
        if not idx:
            continue
        votes = Counter(w_lab[i] for i in idx)
        top = max(votes.items(), key=lambda kv: (kv[1], sum(w_probs[i].get(kv[0], 0.0) for i in idx)))[0]
        p_pred = sum(w_probs[i].get(top, 0.0) for i in idx) / len(idx)
        p_human = sum(w_probs[i].get(s["label"], 0.0) for i in idx) / len(idx)
        pairs.append((s["label"], top))
        if top == s["label"]:
            continue
        h_ev, p_ev = _is_event(s["label"], event_labels), _is_event(top, event_labels)
        kind = KIND_MISS if h_ev and not p_ev else KIND_FALSE if p_ev and not h_ev else KIND_CONFUSE
        if h_ev and p_ev:
            kind = KIND_CONFUSE
        start_ms = int(s["start_ms"]) if s["start_ms"] is not None else int((s["t0"] - csv_start).total_seconds() * 1000)
        end_ms = int(s["end_ms"]) if s["end_ms"] is not None else int((s["t1"] - csv_start).total_seconds() * 1000)
        rows.append({
            "task_id": task_id, "sample_code": sample_code, "day": day, "kind": kind,
            "human": s["label"], "pred": top, "start_ms": start_ms, "end_ms": end_ms,
            "seconds": round((end_ms - start_ms) / 1000, 1),
            "conf": round(p_pred, 3), "p_human": round(p_human, 3), "n_windows": len(idx),
            "votes": {k: v for k, v in votes.most_common(4)},
        })
    # 没人标的时间里连着报事件类的
    run: list[int] = []

    def flush() -> None:
        if len(run) >= 2:
            labs = Counter(w_lab[i] for i in run)
            top = labs.most_common(1)[0][0]
            conf = sum(w_probs[i].get(top, 0.0) for i in run) / len(run)
            rows.append({
                "task_id": task_id, "sample_code": sample_code, "day": day, "kind": KIND_UNLABELED,
                "human": None, "pred": top, "start_ms": int(w_ms[run[0]]), "end_ms": int(w_ms[run[-1]] + window_s * 1000),
                "seconds": round(len(run) * stride_s + (window_s - stride_s), 1),
                "conf": round(conf, 3), "p_human": None, "n_windows": len(run), "votes": dict(labs.most_common(4)),
            })
        run.clear()

    for i in range(len(wins)):
        if not covered[i] and not in_missing(i) and _is_event(w_lab[i], event_labels):
            if run and i != run[-1] + 1:
                flush()
            run.append(i)
        else:
            flush()
    flush()
    return rows, pairs


async def run(job: dict, run_dates: list[str], infer, window_s: float, stride_s: float,
              classes: list[str], concurrency: int = 3, max_rows: int = 2000) -> dict:
    """infer(full_path, device_hz) → 推理结果（raw 模式）。按任务并发几个，结果落盘 + 返回。"""
    dataset_spec = job.get("dataset_spec") or {}
    table = _table(dataset_spec)
    event_labels = [e for e in config.STABLE_EVENT_LABELS if any(_is_event(c, [e]) for c in classes)] or \
        [c for c in classes if "抓挠" in c]
    from label_service.dataset_stats import _day_of
    hz_of_run = {dataset_spec.get("date"): dataset_spec.get("source_hz")}
    for ex in dataset_spec.get("extra_datasets") or []:
        hz_of_run[ex.get("date")] = ex.get("source_hz") or dataset_spec.get("source_hz")
    tasks = _load_tasks(run_dates)
    sem = asyncio.Semaphore(concurrency)
    all_rows: list[dict] = []
    pairs: list[tuple[str, str]] = []
    n_ok = n_fail = 0
    errors: list[str] = []

    async def one(rd: str, t: dict) -> None:
        nonlocal n_ok, n_fail
        full = _csv_full_path(t)
        if not full:
            n_fail += 1
            errors.append(f"任务 #{t.get('id')}：找不到 CSV")
            return
        data = t.get("data") or {}
        hz = data.get("sample_rate_hz") or hz_of_run.get(rd.split("__job")[0]) or config.DEVICE_HZ
        async with sem:
            try:
                res = await infer(full, float(hz))
            except Exception as e:  # noqa: BLE001 一个文件挂了别拖垮整轮
                n_fail += 1
                errors.append(f"任务 #{t.get('id')}：{type(e).__name__}: {str(e)[:120]}")
                return
        rows, prs = _review_task(t, res, table, window_s, stride_s, event_labels, _day_of(t, rd.split("__job")[0]))
        all_rows.extend(rows)
        pairs.extend(prs)
        n_ok += 1

    await asyncio.gather(*(one(rd, t) for rd, t in tasks))

    # 混淆矩阵（按段）
    cls_order = list(classes) + [c for c in sorted({a for a, _ in pairs} | {b for _, b in pairs}) if c not in classes]
    idx = {c: i for i, c in enumerate(cls_order)}
    matrix = [[0] * len(cls_order) for _ in cls_order]
    for h, p in pairs:
        matrix[idx[h]][idx[p]] += 1
    per_class = {}
    for c in cls_order:
        i = idx[c]
        n_h = sum(matrix[i])
        n_p = sum(matrix[j][i] for j in range(len(cls_order)))
        tp = matrix[i][i]
        per_class[c] = {"segments": n_h, "correct": tp,
                        "recall": round(tp / n_h, 4) if n_h else None,
                        "precision": round(tp / n_p, 4) if n_p else None}
    by_kind = Counter(r["kind"] for r in all_rows)
    order = {KIND_MISS: 0, KIND_FALSE: 1, KIND_UNLABELED: 2, KIND_CONFUSE: 3}
    all_rows.sort(key=lambda r: (order.get(r["kind"], 9), -(r["conf"] or 0)))
    out = {
        "model_tag": job.get("model_version"), "job_id": job.get("job_id"),
        "n_tasks": len(tasks), "n_tasks_ok": n_ok, "n_tasks_failed": n_fail,
        "n_segments": len(pairs), "n_wrong": sum(1 for r in all_rows if r["kind"] != KIND_UNLABELED),
        "by_kind": dict(by_kind), "event_labels": event_labels,
        "confusion": {"classes": cls_order, "matrix": matrix}, "per_class": per_class,
        "rows": all_rows[:max_rows], "n_rows_total": len(all_rows),
        "errors": errors[:20],
        "note": ("按段比：一段里的窗口多数投票当预测类别。漏识别 = 人标了事件类模型没认；"
                 "误识别 = 模型报事件类人标的不是；混淆 = 别的类别之间互认；"
                 "未标注区报事件 = 没人标的时间里模型连着报事件类，可能是误报也可能是漏标"),
    }
    return out


def review_path(job_id: int) -> str:
    from label_service.jobs import _job_path
    return os.path.join(os.path.dirname(_job_path(job_id)), f"job{job_id}_review.json")


def summary(full: dict, keep_rows: int = 300) -> dict:
    """塞进训练记录 metrics 里的那份：汇总 + 前几百条，整份走 GET /train/{id}/review。"""
    s = {k: v for k, v in full.items() if k != "rows"}
    s["rows"] = full.get("rows", [])[:keep_rows]
    return s
