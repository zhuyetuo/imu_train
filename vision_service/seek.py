"""
用画面找片段：一句话描述要的行为，把视频里像的那几段挑出来，给人去确认。

要解决的痛点：每加一个新类别（舔、啃、蹭……）都得有人从 24 小时视频里把片段翻出来。
IMU 规则只能挑姿态特殊的（舔/啃），而且每个类别都要重新想规则。这里换一条路：

    画面里有狗 + 狗在动（YOLO 框 + 帧差）    ← 本地，便宜，把 90% 的空镜和睡觉筛掉
      → 剩下的切成几秒一段，裁出狗那一块      ← 720p 俯拍狗只占 100x50 像素，不裁模型看不清
      → 每段抽几帧问视觉大模型（API：Claude / GPT / 豆包 / Gemini，见 llm.py）← 贵，只看筛剩下的
      → 相邻同类合并成片段，带类别/部位/置信度 ← 平台按视频时间写成候选，IMU 段随之落下

**大模型走 API，不在本地起**（用户拍板）。所以这一步的成本是按送出去的段数算的，
下面所有的门槛（狗占比、动作量、每个视频最多送多少段）都是在控这个数。
dry_run=True 只做本地筛选、不调 API，先看会送多少段再决定。

**时间一律读 PTS**（跟 dog.py 同一个原因：视频是 VFR，按帧号算越往后越偏）。
候选要按时间跟 IMU 对上，差几秒就对到别的动作上去了。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import config, dog, llm as llmmod

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


def _video_size(path: str) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"打不开这个视频：{path}")
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def ffmpeg_available() -> bool:
    import shutil

    return config.DECODE_FFMPEG and shutil.which("ffmpeg") is not None


def iter_frames_ffmpeg(path: str, every_sec: float, start_s: float = 0.0, end_s: float | None = None,
                       hwaccel: bool = True):
    """用 ffmpeg 解码、按时间抽帧，yield (t, BGR ndarray)。

    比 cv2 逐帧 grab 快好几倍：ffmpeg 多线程解码，有卡时还能走 NVDEC。fps 滤镜是按 PTS
    重采样的，VFR 也对得上（每一帧就是离 t=k·every_sec 最近的那一帧，误差 ≤ 半个间隔）。
    """
    import subprocess

    import numpy as np

    w, h = _video_size(path)
    if not w or not h:
        raise ValueError(f"读不到分辨率：{path}")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", path]
    if end_s is not None:
        cmd += ["-t", f"{max(0.0, end_s - start_s):.3f}"]
    cmd += ["-vf", f"fps=1/{every_sec}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 4)
    n = 0
    try:
        while True:
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            frame = np.frombuffer(buf, dtype="uint8").reshape(h, w, 3)
            yield start_s + n * every_sec, frame
            n += 1
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
        rc = proc.wait()
    if rc != 0 and n == 0:
        raise RuntimeError(f"ffmpeg 退出码 {rc}：{err[-300:]}")


def iter_frames_cv2(path: str, every_sec: float, start_s: float = 0.0, end_s: float | None = None):
    """老路：cv2 顺序 grab，只在跨过采样点时 retrieve。时间读 PTS。"""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"打不开这个视频：{path}")
    try:
        next_t = start_s
        n_grabbed = 0
        while True:
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
            yield t, frame
    finally:
        cap.release()


def iter_frames(path: str, every_sec: float, start_s: float = 0.0, end_s: float | None = None):
    """有 ffmpeg 走 ffmpeg（先试 NVDEC，不行退软解），没有走 cv2。"""
    if ffmpeg_available():
        for hw in ((True, False) if config.DECODE_HWACCEL else (False,)):
            try:
                yielded = False
                for item in iter_frames_ffmpeg(path, every_sec, start_s, end_s, hwaccel=hw):
                    yielded = True
                    yield item
                return
            except RuntimeError as e:
                if yielded:
                    raise
                _logger.warning("ffmpeg 解码（hwaccel=%s）失败，换一种：%s", hw, e)
        _logger.warning("ffmpeg 两种都不行，退回 cv2 解码：%s", path)
    yield from iter_frames_cv2(path, every_sec, start_s, end_s)


def sample_video(path: str, every_sec: float = 1.0, conf: float = 0.35,
                 start_s: float = 0.0, end_s: float | None = None,
                 max_side: int = 512, jpeg_quality: int = 80, max_samples: int = 20000,
                 batch: int | None = None, on_frame=None, on_batch=None) -> list[dict]:
    """过一遍视频，每 every_sec 取一帧：跑狗检测（一批批送 GPU），有狗就裁出来存成 JPEG。

    返回 [{t, boxes, jpeg(bytes|None), motion(float|None)}]。motion 是跟上一个
    有狗采样点比的帧差（狗那块区域，缩到 64x64 再比，跟裁框位置无关）。
    on_frame(rec, frame)：有狗的帧多做点事（建索引时算姿态关键点）——整帧不存，
    只在这一刻能拿到，算完往 rec 里塞。
    on_batch([(rec, frame), ...])：同上，但攒够一批（batch 张）一起给——要过 GPU 模型的
    活（抠狗）一张张送比一批送慢好几倍。
    """
    import cv2
    import numpy as np

    batch = batch or config.DETECT_BATCH
    out: list[dict] = []
    prev_small = None
    # 静止跳检：整帧缩小后跟上一次真送检测的那帧比，没变就沿用它的框。
    # 狗睡着、空房间时省掉大半检测；一动就照常送
    last_key = None           # 上一次真送检测的小图
    last_boxes: list[dict] = []
    skip_thr = config.STATIC_SKIP_THR
    stats = {"detected": 0, "skipped": 0}

    def frame_key(frame, boxes):
        """(整帧小图, 上次狗框那块的小图)。只看整帧不够：狗只占画面 0.5%，它舔爪子时
        整帧平均差远低于阈值，会被当成"没变"跳掉——所以狗那块单独比。"""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        whole = cv2.resize(g, (64, 36))
        region = None
        if boxes:
            h, w = g.shape[:2]
            # 不撑到 224：撑大了狗只占一角，它的动作被稀释；就按框本身（留一点边）
            x1, y1, x2, y2 = crop_rect(boxes, w, h, margin=0.25, min_side=16)
            if y2 > y1 and x2 > x1:
                region = cv2.resize(g[y1:y2, x1:x2], (64, 64))
        return whole, region

    def _pending_boxes_hint(boxes):
        # 参照帧还没送检测时不知道它的框，用上一次检测的框当区域；没有就整帧比
        return boxes

    def unchanged(key_now, key_last) -> bool:
        if motion_score(key_last[0], key_now[0]) >= skip_thr:
            return False
        if key_last[1] is not None and key_now[1] is not None:
            # 狗那块用更严的门槛（区域小，变化会被稀释；乘 2 是经验值）
            return motion_score(key_last[1], key_now[1]) < skip_thr * 2
        return key_last[1] is None and key_now[1] is None

    def finish(t: float, frame, boxes: list[dict], static: bool = False) -> None:
        nonlocal prev_small
        # static：画面跟上一次送检测的那帧没变（静止跳检沿用的框）。建索引拿它省掉抠狗 / 算向量
        rec = {"t": round(t, 2), "boxes": boxes, "jpeg": None, "motion": None, "static": static}
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
            if on_frame is not None:
                on_frame(rec, frame)
            if on_batch is not None:
                hook_buf.append((rec, frame))
                if len(hook_buf) >= batch:
                    on_batch(list(hook_buf))
                    hook_buf.clear()
        else:
            prev_small = None
        out.append(rec)

    hook_buf: list[tuple[dict, object]] = []

    def flush(pending: list[tuple[float, object]]) -> None:
        nonlocal last_key, last_boxes
        boxes_all = dog.detect_batch([f for _t, f in pending], conf)
        stats["detected"] += len(pending)
        for (t, frame), boxes in zip(pending, boxes_all):
            last_key, last_boxes = frame_key(frame, boxes), boxes
            finish(t, frame, boxes)

    pending: list[tuple[float, object]] = []
    # 参照帧 = 最近一个"决定要送检测"的帧（可能还在 pending 里没送）。跟它比没变就沿用它的框；
    # 它还没送的话先把这批送掉（批偶尔小一点，换来静止时段几乎不送检测）
    ref_key = None
    for t, frame in iter_frames(path, every_sec, start_s, end_s):
        if skip_thr > 0 and ref_key is not None:
            k = frame_key(frame, last_boxes if not pending else _pending_boxes_hint(last_boxes))
            if unchanged(k, ref_key):
                if pending:
                    flush(pending)
                    pending = []
                stats["skipped"] += 1
                finish(t, frame, list(last_boxes), static=True)      # 画面没变，框也没变
                continue
            ref_key = k
        else:
            ref_key = frame_key(frame, last_boxes)
        pending.append((t, frame))
        if len(pending) >= batch:
            flush(pending)
            pending = []
        if len(out) + len(pending) >= max_samples:
            break
    if pending:
        flush(pending)
    if on_batch is not None and hook_buf:
        on_batch(list(hook_buf))
        hook_buf.clear()
    _logger.info("采样 %s：%d 帧，送检 %d，静止跳过 %d", os.path.basename(path), len(out), stats["detected"], stats["skipped"])
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

def status() -> dict:
    """没带 llm 时退回环境变量那把 key（老部署方式）。"""
    env = llmmod.from_env()
    err = None
    if env is None:
        err = "没配 ANTHROPIC_API_KEY（也可以在平台「大模型 API」页配好，请求时带过来）"
    else:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            err = "没装 anthropic SDK：pip install anthropic"
    return {
        "available": err is None,
        "error": err,
        "model": config.SEEK_MODEL,
        "providers": list(llmmod.PROVIDERS),
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


def ask(frames: list[bytes], labels: list[Label], clip_s: float, llm: llmmod.LLM,
        client=None, http=None) -> dict:
    """把一段的几帧送去问。返回 parse_answer 的结果 + usage。"""
    system, user = build_prompt(labels, clip_s, len(frames))
    t0 = time.monotonic()
    text, usage = llmmod.chat_vision(llm, system, user, frames, max_tokens=300, client=client, http=http)
    out = parse_answer(text, labels)
    out["usage"] = usage
    out["latency_ms"] = int((time.monotonic() - t0) * 1000)
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

def samples_from_index(rel_path: str, start_s: float = 0.0, end_s: float | None = None) -> list[dict] | None:
    """建过画面索引的视频：直接从索引拿"每秒有没有狗 + 动没动"，不再解码、不再检测。

    索引里每秒一条 (t, 向量, 框)。动作量用相邻两秒向量的距离（‖e_t − e_{t−1}‖/2，0~1）：
    狗趴着不动 0.02 左右，舔/抓/走 0.1 以上。跟像素帧差不是一个刻度，所以 seek 用
    索引时的 motion_min 走 config.SEEK_INDEX_MOTION_MIN，不用调用方传的那个。
    jpeg 先留空，选中的窗再去视频里抽那几帧（frames_for_window）。
    """
    from . import embed

    import numpy as np

    d = embed.load(rel_path)
    if d is None or not len(d["t"]):
        return None
    ts, emb, boxes = d["t"], d["emb"], d["box"]
    out: list[dict] = []
    prev = None
    for i in range(len(ts)):
        t = float(ts[i])
        if t < start_s or (end_s is not None and t > end_s):
            continue
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        rec = {"t": round(t, 2), "boxes": [{"bbox": [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)], "conf": 1.0}],
               "jpeg": b"", "motion": None, "_i": i}
        if prev is not None and t - float(ts[prev]) <= 2.5:
            rec["motion"] = float(np.linalg.norm(emb[i] - emb[prev]) / 2.0)
        prev = i
        out.append(rec)
    return out


def frames_for_window(full_path: str, w: Window, samples: list[dict], n_frames: int,
                      max_side: int = 512, jpeg_quality: int = 80) -> list[bytes]:
    """只把选中窗那几秒的帧抽出来（ffmpeg -ss 定位，几十毫秒），按索引里的框裁狗。"""
    import cv2
    import numpy as np

    idx = w.idx
    if len(idx) > n_frames:
        step = len(idx) / n_frames
        idx = [idx[int(i * step)] for i in range(n_frames)]
    want = {round(samples[i]["t"], 2): samples[i]["boxes"] for i in idx}
    out: list[bytes] = []
    for t, frame in iter_frames(full_path, 1.0, start_s=w.start, end_s=w.end + 0.5):
        key = round(t, 2)
        # ffmpeg 抽出来的时间是 start + k 秒，索引里的 t 也是整秒起，取最近的
        near = min(want, key=lambda x: abs(x - key)) if want else None
        if near is None or abs(near - key) > 0.6:
            continue
        boxes = want.pop(near)
        h, wd = frame.shape[:2]
        x1, y1, x2, y2 = crop_rect(boxes, wd, h)
        crop = frame[y1:y2, x1:x2]
        if not crop.size:
            continue
        scale = max_side / max(crop.shape[:2])
        if scale < 1:
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            out.append(bytes(np.asarray(buf).tobytes()))
        if not want:
            break
    return out


def seek_video(path: str, labels: list[Label], *, every_sec: float = 1.0, clip_s: float = 6.0,
               stride_s: float = 3.0, n_frames: int = 6, max_clips: int = 120,
               min_dog_frac: float = 0.8, motion_min: float = 0.02, motion_max: float = 1.0,
               min_conf: float = 0.5, start_s: float = 0.0, end_s: float | None = None,
               dry_run: bool = False, conf: float = 0.35, concurrency: int | None = None,
               llm: llmmod.LLM | None = None, client=None, http=None,
               rel_path: str | None = None) -> dict:
    t0 = time.monotonic()
    llm = llm or llmmod.from_env()
    if llm is None and not dry_run:
        raise RuntimeError("没有可用的大模型：请求里没带 llm，环境变量也没配 ANTHROPIC_API_KEY")
    # 建过索引的视频走索引：不解码不检测，预览秒出；只有选中的窗才去抽那几帧
    samples = samples_from_index(rel_path, start_s, end_s) if rel_path else None
    from_index = samples is not None
    if from_index:
        motion_min, motion_max = config.SEEK_INDEX_MOTION_MIN, config.SEEK_INDEX_MOTION_MAX
    else:
        samples = sample_video(path, every_sec=every_sec, conf=conf, start_s=start_s, end_s=end_s)
    wins = pick_windows(samples, clip_s=clip_s, stride_s=stride_s, min_dog_frac=min_dog_frac,
                        motion_min=motion_min, motion_max=motion_max, max_clips=max_clips)
    stats = {
        "sampled": len(samples),
        "with_dog": sum(1 for s in samples if s["jpeg"] is not None),
        "clips_candidate": len(wins),
        "clips_sent": 0,
        "usage": {"input": 0, "output": 0, "est_usd": 0.0},
        "llm": llmmod.describe(llm),
        "from_index": from_index,
        "seconds": 0.0,
    }
    if dry_run or not wins:
        stats["seconds"] = round(time.monotonic() - t0, 1)
        return {"segments": [], "windows": [w.__dict__ | {"idx": None} for w in wins], "stats": stats, "dry_run": True}

    def frames_of(w: Window) -> list[bytes]:
        if from_index:
            return frames_for_window(path, w, samples, n_frames)
        idx = w.idx
        if len(idx) > n_frames:
            step = len(idx) / n_frames
            idx = [idx[int(i * step)] for i in range(n_frames)]
        return [samples[i]["jpeg"] for i in idx]

    def one(w: Window) -> dict:
        t1 = time.monotonic()
        try:
            return ask(frames_of(w), labels, clip_s, llm, client=client, http=http)
        except Exception as e:  # noqa: BLE001 一段问失败不该让整个视频白跑
            _logger.warning("问模型失败 %.1f-%.1f：%s", w.start, w.end, e)
            return {"label": None, "body_part": None, "confidence": 0.0, "note": f"失败:{type(e).__name__}",
                    "usage": {"input": 0, "output": 0}, "error": str(e)[:200],
                    "latency_ms": int((time.monotonic() - t1) * 1000)}

    with ThreadPoolExecutor(max_workers=concurrency or config.SEEK_CONCURRENCY) as ex:
        answers = list(ex.map(one, wins))

    stats["clips_sent"] = len(wins)
    stats["errors"] = sum(1 for a in answers if a.get("error"))
    stats["usage"]["input"] = sum(a["usage"]["input"] for a in answers)
    stats["usage"]["output"] = sum(a["usage"]["output"] for a in answers)
    stats["usage"]["est_usd"] = llmmod.estimate_usd(llm, stats["usage"]["input"], stats["usage"]["output"])
    stats["hits"] = sum(1 for a in answers if a.get("label"))
    stats["seconds"] = round(time.monotonic() - t0, 1)
    # 每一次调用单独记一条：平台那边存表做统计（次数 / token / 耗时）
    stats["calls"] = [{"latency_ms": int(a.get("latency_ms") or 0), "input": int(a["usage"].get("input") or 0),
                       "output": int(a["usage"].get("output") or 0),
                       "est_usd": llmmod.estimate_usd(llm, int(a["usage"].get("input") or 0), int(a["usage"].get("output") or 0)),
                       "ok": not a.get("error"), "error": a.get("error"),
                       "start_s": w.start, "end_s": w.end}
                      for w, a in zip(wins, answers)]
    segs = merge_segments(wins, answers, min_conf=min_conf)
    return {"segments": segs, "windows": [w.__dict__ | {"idx": None, "answer": {k: v for k, v in a.items() if k != "usage"}}
                                          for w, a in zip(wins, answers)],
            "stats": stats, "dry_run": False}
