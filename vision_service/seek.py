"""
用画面找片段：一句话描述要的行为，把视频里像的那几段挑出来，给人去确认。

要解决的痛点：每加一个新类别（舔、啃、蹭……）都得有人从 24 小时视频里把片段翻出来。
IMU 规则只能挑姿态特殊的（舔/啃），而且每个类别都要重新想规则。这里换一条路：

    画面里有狗 + 狗在动（YOLO 框 + 帧差）    ← 本地，便宜，把 90% 的空镜和睡觉筛掉
      → 剩下的切成几秒一段，裁出狗那一块      ← 720p 俯拍狗只占 100x50 像素，不裁模型看不清
      → 每段抽几帧问视觉大模型（Claude API）  ← 贵，只看筛剩下的
      → 相邻同类合并成片段，带类别/部位/置信度 ← 平台按视频时间写成候选，IMU 段随之落下

**大模型走 API，不在本地起**（用户拍板）。所以这一步的成本是按送出去的段数算的，
下面所有的门槛（狗占比、动作量、每个视频最多送多少段）都是在控这个数。
dry_run=True 只做本地筛选、不调 API，先看会送多少段再决定。

**时间一律读 PTS**（跟 dog.py 同一个原因：视频是 VFR，按帧号算越往后越偏）。
候选要按时间跟 IMU 对上，差几秒就对到别的动作上去了。
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import config, dog

_logger = logging.getLogger("vision_service.seek")


# ── 输入 ──────────────────────────────────────────────────────────────

@dataclass
class Label:
    name: str                       # 平台里的标签名，比如「舔身体」
    description: str = ""           # 给模型看的一句话：这个动作长什么样
    parts: list[str] = field(default_factory=list)   # 可选的部位子类，比如 前肢爪/后肢臀尾


@dataclass
class Window:
    start: float
    end: float
    motion: float                   # 窗内动作量均值（0-1）
    dog_frac: float                 # 窗内多少比例的采样点有狗
    idx: list[int] = field(default_factory=list)      # 用到的采样点下标


# ── 采样：一遍顺序解码，每 every_sec 留一张裁好的狗图 ─────────────────

def crop_rect(boxes: list[dict], w: int, h: int, margin: float = 0.25, min_side: int = 224) -> tuple[int, int, int, int]:
    """所有狗框的并集，四周留 margin，短边至少 min_side（太小的话缩放后是马赛克）。

    返回像素 (x1, y1, x2, y2)。多只狗时并集会很大，那也没办法——分不出哪只是
    哪只的时候本来就不该拿来当 IMU 的候选（平台那边按一间一狗的场地过滤）。
    """
    xs1 = [b["bbox"][0] for b in boxes]
    ys1 = [b["bbox"][1] for b in boxes]
    xs2 = [b["bbox"][0] + b["bbox"][2] for b in boxes]
    ys2 = [b["bbox"][1] + b["bbox"][3] for b in boxes]
    x1, y1, x2, y2 = min(xs1) * w, min(ys1) * h, max(xs2) * w, max(ys2) * h
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    # 撑到最小边长，围绕中心
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side_w = max(x2 - x1, min_side)
    side_h = max(y2 - y1, min_side)
    x1, x2 = cx - side_w / 2, cx + side_w / 2
    y1, y2 = cy - side_h / 2, cy + side_h / 2
    # 贴边
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2


def motion_score(prev_gray, cur_gray) -> float:
    """两张同尺寸小灰度图的平均绝对差，归一化到 0-1。"""
    import numpy as np

    d = np.abs(prev_gray.astype("int16") - cur_gray.astype("int16"))
    return float(d.mean() / 255.0)


def sample_video(path: str, every_sec: float = 1.0, conf: float = 0.35,
                 start_s: float = 0.0, end_s: float | None = None,
                 max_side: int = 512, jpeg_quality: int = 80, max_samples: int = 20000) -> list[dict]:
    """顺序过一遍视频，每 every_sec 取一帧：跑狗检测，有狗就裁出来存成 JPEG。

    返回 [{t, boxes, jpeg(bytes|None), motion(float|None)}]。motion 是跟上一个
    有狗采样点比的帧差（狗那块区域，缩到 64x64 再比，跟裁框位置无关）。
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"打不开这个视频：{path}")
    out: list[dict] = []
    prev_small = None
    try:
        next_t = start_s
        n_grabbed = 0
        while len(out) < max_samples:
            if not cap.grab():
                break
            n_grabbed += 1
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms <= 0 and n_grabbed > 1:
                continue
            t = ms / 1000.0
            if end_s is not None and t > end_s:
                break
            if t < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            next_t = t + every_sec
            boxes = dog.detect(frame, conf)
            rec = {"t": round(t, 2), "boxes": boxes, "jpeg": None, "motion": None}
            if boxes:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = crop_rect(boxes, w, h)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64))
                    if prev_small is not None:
                        rec["motion"] = motion_score(prev_small, small)
                    prev_small = small
                    scale = max_side / max(crop.shape[:2])
                    if scale < 1:
                        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
                    ok2, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                    if ok2:
                        rec["jpeg"] = bytes(np.asarray(buf).tobytes())
            else:
                prev_small = None
            out.append(rec)
    finally:
        cap.release()
    return out


# ── 选窗：纯函数，不碰视频 ────────────────────────────────────────────

def pick_windows(samples: list[dict], clip_s: float = 6.0, stride_s: float = 3.0,
                 min_dog_frac: float = 0.8, motion_min: float = 0.02, motion_max: float = 1.0,
                 max_clips: int = 120) -> list[Window]:
    """哪些时间窗值得送去问模型。

    条件：窗里 ≥ min_dog_frac 的采样点有狗（裁得出图），动作量均值在
    [motion_min, motion_max]（太低是趴着不动，太高多半是走/跑或镜头抖）。
    超过 max_clips 时按动作量取最强的那些——这是在控 API 花费。
    """
    if not samples:
        return []
    t0 = samples[0]["t"]
    t_end = samples[-1]["t"]
    wins: list[Window] = []
    start = t0
    while start < t_end:
        end = start + clip_s
        idx = [i for i, s in enumerate(samples) if start <= s["t"] < end]
        if idx:
            with_dog = [i for i in idx if samples[i]["jpeg"] is not None]
            frac = len(with_dog) / len(idx)
            motions = [samples[i]["motion"] for i in with_dog if samples[i]["motion"] is not None]
            if frac >= min_dog_frac and motions:
                m = sum(motions) / len(motions)
                if motion_min <= m <= motion_max:
                    wins.append(Window(start=round(start, 2), end=round(min(end, t_end + 1e-6), 2),
                                       motion=round(m, 4), dog_frac=round(frac, 3), idx=with_dog))
        start += stride_s
    if len(wins) > max_clips:
        wins = sorted(wins, key=lambda w: -w.motion)[:max_clips]
    wins.sort(key=lambda w: w.start)
    return wins


# ── 问模型 ────────────────────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()
_client_error: str | None = None


def _get_client():
    global _client, _client_error
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            import anthropic
        except ImportError:
            _client_error = "没装 anthropic SDK：pip install anthropic"
            return None
        if not config.ANTHROPIC_API_KEY:
            _client_error = "没配 ANTHROPIC_API_KEY"
            return None
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, max_retries=3, timeout=120.0)
        _client_error = None
        return _client


def status() -> dict:
    c = _get_client()
    return {
        "available": c is not None,
        "error": _client_error,
        "model": config.SEEK_MODEL,
        "dog": dog.status().get("available"),
    }


def build_prompt(labels: list[Label], clip_s: float, n_frames: int) -> tuple[str, str]:
    """(system, user_text)。类别和部位写进去，让模型只在这几个里选。"""
    lines = []
    for lb in labels:
        line = f"- {lb.name}"
        if lb.description:
            line += f"：{lb.description}"
        if lb.parts:
            line += f"（部位可选：{' / '.join(lb.parts)}）"
        lines.append(line)
    system = (
        "你在看一段狗舍俯拍监控里裁出来的狗。给你的是同一段视频按时间顺序抽的几帧，"
        "任务是判断这几秒里狗在做下面哪一种行为。只能从给定类别里选，都不像就答 none。"
        "要保守：拿不准就 none，宁可漏也别把普通的趴着、走动、张望判成这些行为。"
        "只输出一个 JSON 对象，不要别的文字。"
    )
    user = (
        f"这是 {clip_s:.0f} 秒里按顺序抽的 {n_frames} 帧。\n候选行为：\n" + "\n".join(lines) +
        '\n\n输出格式：{"label": "<类别名或 none>", "body_part": "<该类别的部位之一，没有就 null>", '
        '"confidence": <0到1>, "note": "<不超过20字的依据>"}'
    )
    return system, user


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_answer(text: str, labels: list[Label]) -> dict:
    """模型的回答 → {label, body_part, confidence, note}。答非所问一律当 none。"""
    m = _JSON_RE.search(text or "")
    if not m:
        return {"label": None, "body_part": None, "confidence": 0.0, "note": "无法解析"}
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return {"label": None, "body_part": None, "confidence": 0.0, "note": "无法解析"}
    names = {lb.name: lb for lb in labels}
    label = d.get("label")
    if not isinstance(label, str) or label not in names:
        return {"label": None, "body_part": None, "confidence": 0.0, "note": str(d.get("note") or "")[:40]}
    part = d.get("body_part")
    if not isinstance(part, str) or part not in names[label].parts:
        part = None
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"label": label, "body_part": part, "confidence": conf, "note": str(d.get("note") or "")[:40]}


def ask(frames: list[bytes], labels: list[Label], clip_s: float, client=None, model: str | None = None) -> dict:
    """把一段的几帧送去问。返回 parse_answer 的结果 + usage。"""
    client = client or _get_client()
    if client is None:
        raise RuntimeError(_client_error or "模型客户端不可用")
    system, user = build_prompt(labels, clip_s, len(frames))
    content: list[dict] = []
    for b in frames:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(b).decode("ascii")}})
    content.append({"type": "text", "text": user})
    resp = client.messages.create(
        model=model or config.SEEK_MODEL,
        max_tokens=300,
        system=system,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
    out = parse_answer(text, labels)
    u = getattr(resp, "usage", None)
    out["usage"] = {"input": int(getattr(u, "input_tokens", 0) or 0),
                    "output": int(getattr(u, "output_tokens", 0) or 0)}
    return out


# ── 合并 ──────────────────────────────────────────────────────────────

def merge_segments(wins: list[Window], answers: list[dict], min_conf: float = 0.5) -> list[dict]:
    """相邻/重叠、同类别的窗合成一段。置信度取最大，部位取多数。"""
    segs: list[dict] = []
    for w, a in zip(wins, answers):
        if not a.get("label") or a.get("confidence", 0.0) < min_conf:
            continue
        last = segs[-1] if segs else None
        if last and last["label"] == a["label"] and w.start <= last["end_s"] + 1e-6:
            last["end_s"] = max(last["end_s"], w.end)
            last["confidence"] = max(last["confidence"], a["confidence"])
            last["_parts"].append(a.get("body_part"))
            last["n_clips"] += 1
        else:
            segs.append({"start_s": w.start, "end_s": w.end, "label": a["label"],
                         "confidence": a["confidence"], "_parts": [a.get("body_part")],
                         "note": a.get("note") or "", "n_clips": 1})
    for s in segs:
        parts = [p for p in s.pop("_parts") if p]
        s["body_part"] = Counter(parts).most_common(1)[0][0] if parts else None
    return segs


# ── 整条流水线 ────────────────────────────────────────────────────────

def estimate_usd(input_tokens: int, output_tokens: int, model: str | None = None) -> float:
    pin, pout = config.SEEK_PRICE_PER_M.get(model or config.SEEK_MODEL, (0.0, 0.0))
    return round(input_tokens / 1e6 * pin + output_tokens / 1e6 * pout, 4)


def seek_video(path: str, labels: list[Label], *, every_sec: float = 1.0, clip_s: float = 6.0,
               stride_s: float = 3.0, n_frames: int = 6, max_clips: int = 120,
               min_dog_frac: float = 0.8, motion_min: float = 0.02, motion_max: float = 1.0,
               min_conf: float = 0.5, start_s: float = 0.0, end_s: float | None = None,
               dry_run: bool = False, conf: float = 0.35, concurrency: int | None = None,
               client=None) -> dict:
    t0 = time.monotonic()
    samples = sample_video(path, every_sec=every_sec, conf=conf, start_s=start_s, end_s=end_s)
    wins = pick_windows(samples, clip_s=clip_s, stride_s=stride_s, min_dog_frac=min_dog_frac,
                        motion_min=motion_min, motion_max=motion_max, max_clips=max_clips)
    stats = {
        "sampled": len(samples),
        "with_dog": sum(1 for s in samples if s["jpeg"] is not None),
        "clips_candidate": len(wins),
        "clips_sent": 0,
        "usage": {"input": 0, "output": 0, "est_usd": 0.0},
        "model": config.SEEK_MODEL,
        "seconds": 0.0,
    }
    if dry_run or not wins:
        stats["seconds"] = round(time.monotonic() - t0, 1)
        return {"segments": [], "windows": [w.__dict__ | {"idx": None} for w in wins], "stats": stats, "dry_run": True}

    def frames_of(w: Window) -> list[bytes]:
        idx = w.idx
        if len(idx) > n_frames:
            step = len(idx) / n_frames
            idx = [idx[int(i * step)] for i in range(n_frames)]
        return [samples[i]["jpeg"] for i in idx]

    def one(w: Window) -> dict:
        try:
            return ask(frames_of(w), labels, clip_s, client=client)
        except Exception as e:  # noqa: BLE001 一段问失败不该让整个视频白跑
            _logger.warning("问模型失败 %.1f-%.1f：%s", w.start, w.end, e)
            return {"label": None, "body_part": None, "confidence": 0.0, "note": f"失败:{type(e).__name__}",
                    "usage": {"input": 0, "output": 0}, "error": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=concurrency or config.SEEK_CONCURRENCY) as ex:
        answers = list(ex.map(one, wins))

    stats["clips_sent"] = len(wins)
    stats["errors"] = sum(1 for a in answers if a.get("error"))
    stats["usage"]["input"] = sum(a["usage"]["input"] for a in answers)
    stats["usage"]["output"] = sum(a["usage"]["output"] for a in answers)
    stats["usage"]["est_usd"] = estimate_usd(stats["usage"]["input"], stats["usage"]["output"])
    stats["hits"] = sum(1 for a in answers if a.get("label"))
    stats["seconds"] = round(time.monotonic() - t0, 1)
    segs = merge_segments(wins, answers, min_conf=min_conf)
    return {"segments": segs, "windows": [w.__dict__ | {"idx": None, "answer": {k: v for k, v in a.items() if k != "usage"}}
                                          for w, a in zip(wins, answers)],
            "stats": stats, "dry_run": False}
