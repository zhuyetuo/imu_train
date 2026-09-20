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
    if n == 0:
        # **退出码 0 但一帧都没给，也算失败**。原来只在 rc != 0 时抛，于是这一支
        # 既不抛也不产出，调用方那边直接 return——cv2 那条后路根本没机会跑，
        # 表现成"解码没给出那几帧"而毫无线索。带 -ss 的小窗口上真会发生
        # （2026-09-20 实测：partask 20 条全军覆没，建索引却一直是好的——
        # 因为建索引那条路不带 -ss）。
        raise RuntimeError(f"ffmpeg 退出码 {rc} 但一帧都没解出来"
                           f"（-ss {start_s:.1f} -t {(end_s - start_s) if end_s else -1:.1f}）："
                           f"{err[-300:] or '（stderr 是空的）'}")


def iter_frames_cv2(path: str, every_sec: float, start_s: float = 0.0, end_s: float | None = None):
    """老路：cv2 顺序 grab，只在跨过采样点时 retrieve。时间读 PTS。

    start_s 大于零时先 seek 过去：不 seek 的话要从头 grab 到那儿，一小时的视频里
    取 2548 秒那几帧得读四万多帧，几十秒。seek 的落点是关键帧、可能比要的时间早
    几秒，所以 seek 完照样按 PTS 往前找，不影响准确性。
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"打不开这个视频：{path}")
    try:
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s - 2.0) * 1000.0)
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
            yielded = False
            try:
                for item in iter_frames_ffmpeg(path, every_sec, start_s, end_s, hwaccel=hw):
                    yielded = True
                    yield item
            except RuntimeError as e:
                # 已经吐过帧再出错：那是解到一半断了，退回 cv2 从头来会给出重复的帧，
                # 不如让调用方看到错
                if yielded:
                    raise
                _logger.warning("ffmpeg 解码（hwaccel=%s）失败，换一种：%s", hw, e)
                continue
            if yielded:
                return
            # 没抛也没产出：这种在 iter_frames_ffmpeg 里已经改成抛了，留这一手兜底
            _logger.warning("ffmpeg（hwaccel=%s）没产出帧，换一种：%s", hw, path)
        _logger.warning("ffmpeg 两种都不行，退回 cv2 解码：%s", path)
    yield from iter_frames_cv2(path, every_sec, start_s, end_s)


def prefetch(it, size: int = 0):
    """把一个帧迭代器丢到后台线程里跑，边解码边让主线程算。

    为什么有它：解码是 CPU（ffmpeg）、检测/姿态/分割/向量是 GPU，原来串在一个
    循环里——GPU 算的时候 ffmpeg 停着，ffmpeg 解的时候 GPU 停着，一路视频白等掉
    其中一半。队列只放 size 帧，解码跑太前面就自己停住，不会把内存吃光。

    size <= 1 直接返回原迭代器（测试和排查时好对比）。
    """
    import queue
    import threading

    size = size or config.DECODE_PREFETCH
    if size <= 1:
        return it

    q: "queue.Queue" = queue.Queue(maxsize=size)
    done = object()
    stop = threading.Event()
    err: list[BaseException] = []

    def work():
        try:
            for item in it:
                if stop.is_set():
                    break
                q.put(item)
        except BaseException as e:  # noqa: BLE001 解码出错要带回主线程抛，不能吞在子线程里
            err.append(e)
        finally:
            it.close() if hasattr(it, "close") else None
            q.put(done)

    th = threading.Thread(target=work, name="decode-prefetch", daemon=True)
    th.start()

    def gen():
        try:
            while True:
                item = q.get()
                if item is done:
                    break
                yield item
        finally:
            # 提前 break（max_samples 到了 / 上游抛了）时，解码线程可能正卡在 q.put 上：
            # 先把队列抽干让它动起来，它才会看到 stop 并收掉 ffmpeg
            stop.set()
            while th.is_alive() or not q.empty():
                try:
                    if q.get(timeout=0.05) is done:
                        break
                except queue.Empty:
                    continue
        if err:
            raise err[0]

    return gen()


def sample_video(path: str, every_sec: float = 1.0, conf: float = 0.35,
                 start_s: float = 0.0, end_s: float | None = None,
                 max_side: int = 512, jpeg_quality: int = 80, max_samples: int = 20000,
                 batch: int | None = None, on_frame=None, on_batch=None,
                 stats_out: dict | None = None) -> list[dict]:
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
    last_boxes: list[dict] = []          # 上一次真送检测的那帧的框，静止帧沿用它
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

    def flush(pending: list[tuple[float, object, bool]]) -> None:
        """一批帧：要检测的一起送 GPU，静止的沿用它前面那个检测帧的框。

        静止帧也排在这个队列里（而不是当场处理掉）：排队时它的框还不知道——它要沿用的
        那一帧可能就在同一批里还没送检测。原来的做法是遇到静止帧就把当前批先送掉，
        狗一会儿动一会儿停的时候，批长期只有一两帧，等于白攒。这里按顺序回填，
        批始终是满的，送检的帧数一帧不多。
        """
        nonlocal last_boxes
        todo = [f for _t, f, st in pending if not st]
        boxes_all = iter(dog.detect_batch(todo, conf) if todo else [])
        stats["detected"] += len(todo)
        for t, frame, st in pending:
            if st:
                finish(t, frame, list(last_boxes), static=True)      # 画面没变，框也没变
            else:
                last_boxes = next(boxes_all)
                finish(t, frame, last_boxes)

    pending: list[tuple[float, object, bool]] = []
    # 参照帧 = 最近一个"决定要送检测"的帧。跟它比没变就沿用它的框
    ref_key = None
    for t, frame in prefetch(iter_frames(path, every_sec, start_s, end_s)):
        static = False
        if skip_thr > 0:
            # 区域比对用 last_boxes 当位置提示：一批之内它是冻住的，参照帧和当前帧裁的是
            # 同一块，比出来的差才是真的画面变化
            k = frame_key(frame, last_boxes)
            if ref_key is not None and unchanged(k, ref_key):
                static = True
                stats["skipped"] += 1
            else:
                ref_key = k
        pending.append((t, frame, static))
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
    if stats_out is not None:
        stats_out.update(stats)
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


def build_prompt(labels: list[Label], clip_s: float, n_frames: int,
                 tiled: bool = False) -> tuple[str, str]:
    """(system, user_text)。类别和部位写进去，让模型只在这几个里选。

    三件事是 2026-09-20 那轮调试逼出来的：

    1. **see 跟 label 分开**。原来"不是这些行为"和"根本看不清"都答 none，混成一个数。
       这两种的含义天差地别：前者是模型正常工作、正确拒绝；后者说明送进来的候选本身
       是垃圾（画面太小/太暗/狗被挡），要回头修出候选的那一层。混着的话，一批 none
       回来我完全不知道该改哪边。
    2. **note 要写看到了什么，不是只写结论**。20 字的「像在舔」没有任何调试价值；
       「侧卧，头转向身后，口鼻接触左后肢，四帧里口鼻位置有小幅往复」一眼就能判断
       模型是看懂了还是在猜。之前部位检索踩的坑就是——只有结论、没有依据，
       错了只能反复猜。
    3. **拼成一张带序号的图**时要告诉它序号就是时间顺序，否则它会当成几只不同的狗。
    """
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
        "**是同一只狗的连续几秒**，不是几只不同的狗。"
        "任务是判断这几秒里狗在做下面哪一种行为。只能从给定类别里选，都不像就答 none。"
        # 下面这三句是 2026-09-20 的对照组逼出来的：拿"鼻子离爪子 >1.5 体长、不可能在舔"
        # 的帧混进去问同一个问题，命中率跟正式组一模一样（都是 15%）——说明模型在顺着
        # 提示词走，答案跟画面基本无关。所以要把"多数应该答 none"这个先验说死，
        # 并且要求它指出**具体哪一帧的什么变化**，不能拿姿势当证据
        "**送进来的片段大多数其实是狗在趴着、睡觉或休息，预期答案就是 none。**"
        "判成某个行为之前先问自己：几帧之间有没有看得见的动作变化？"
        "只是'头靠近某个部位'而几帧之间姿势没变，那是休息，不是舔/啃——"
        "舔和啃的定义里都有'反复'，看不到反复就答 none。"
        "note 里要指出是第几帧到第几帧的什么变化让你这么判；指不出来就说明证据不足，答 none。"
        "看不清就照实说 see=unclear，别硬猜——画面太小、太暗、狗被挡住、只拍到局部，都算看不清。"
        "只输出一个 JSON 对象，不要别的文字。"
    )
    how = (f"这是 {clip_s:.0f} 秒里按顺序抽的 {n_frames} 帧，"
           + ("拼成了一张图，从左到右、从上到下是时间顺序，每格左上角有序号。\n"
              if tiled else "按时间先后给你。\n"))
    user = (
        how + "候选行为：\n" + "\n".join(lines) +
        '\n\n输出格式（只要这个 JSON）：\n'
        '{"see": "clear 或 unclear", '
        '"desc": "<30字内：狗什么姿势、口鼻朝哪、几帧之间有没有变化>", '
        '"label": "<类别名或 none>", "body_part": "<该类别的部位之一，没有就 null>", '
        '"confidence": <0到1>, "note": "<30字内：为什么判成这个>"}\n'
        "desc 写你实际看到的，不要写结论；note 写判断依据。看不清时 label 一律 none。"
    )
    return system, user


def build_contact_prompt(parts: list[str], n_frames: int, clip_s: float,
                        tiled: bool = False) -> tuple[str, str]:
    """只问画面答得了的那个问题：**口鼻贴着身体的哪个部位**。

    ## 为什么不问"是不是在舔"

    2026-09-20 连着几轮都栽在这上面。问「这是舔/啃/抓挠还是 none」时：

      提示词偏松 → 见什么都说"在舔"，对照组一样高，那个命中率是假的
      提示词偏严 → 全答 none。模型明明看见了（"1-2口鼻朝后爪，3-4头低下，
                   姿势有变化"），但我写着"看不到反复就答 none"，它照做

    根子上是**这个问题画面答不了**：

      1. 舔是 2-4 Hz，舌头只有几个像素，俯拍画面上根本看不见
      2. 唯一看得见的特征是"口鼻长时间贴在某个部位"——而我为了防误报，
         明确写了"头靠近某个部位但姿势没变 = 休息"，等于亲手禁掉了唯一的证据
      3. 采样 0.3 秒 = 3.3 Hz，正好在奈奎斯特频率上，"反复"照样采不到

    「是不是在理毛」**IMU 答得了**（label_service/grooming.py 那条规则测的就是
    这个动作特征）。画面该答的是另一半：**贴的是哪个部位**。两者取交集才是
    「舔-后左爪」。这是一开始定的分工，中间走丢了。

    所以这里一个字都不提舔/啃/抓挠——不给任何行为上的暗示，只问接触关系。
    """
    system = (
        "你在看一段狗舍俯拍监控里裁出来的狗。给你的是同一只狗连续几秒的几帧，"
        "不是几只不同的狗。"
        "**只回答一件事：这只狗的口鼻（嘴和鼻子那一块）有没有贴到自己身体的某个部位。**"
        "贴到 = 接触或几乎接触（中间没有明显空隙）。头只是朝那个方向、但隔着一段距离，"
        "不算贴到。"
        # 2026-09-20 实测：选项里带「颈部」时，70 条里 25 条（36%）选了它——
        # 狗趴着时口鼻本来就在自己胸颈一带，那是个**永远成立**的答案，等于给了
        # 一个不用看画面就能选的出口。只数爪子和尾根，那两个要真的够过去才成立
        "**狗趴着/蜷着时口鼻本来就挨着自己的胸口和脖子，那不算「贴到」**——"
        "只数爪子和尾根：狗得主动把头够过去才算。"
        "画面里不止一只狗、分不清是哪只的部位时，说 see=unclear。"
        "另外说一下这几帧之间狗有没有在动。"
        "不用判断它在做什么行为，也不要猜——看不清就说 see=unclear。"
        "只输出一个 JSON 对象，不要别的文字。"
    )
    how = (f"这是 {clip_s:.1f} 秒里按顺序抽的 {n_frames} 帧，"
           + ("拼成了一张图，从左到右、从上到下是时间顺序，每格左上角有序号。\n"
              if tiled else "按时间先后给你。\n"))
    user = (
        how + "部位只能从这几个里选：" + " / ".join(parts) +
        '\n\n输出格式（只要这个 JSON）：\n'
        '{"see": "clear 或 unclear", '
        '"desc": "<30字内：狗什么姿势、口鼻在哪、几帧之间变了什么>", '
        '"contact": true 或 false, '
        '"part": "<贴到的部位，没贴到就 null>", '
        '"moving": true 或 false, '
        '"confidence": <0到1>}\n'
        "desc 写你实际看到的。contact 只看口鼻和身体部位之间有没有空隙，不用管它在干什么。"
    )
    return system, user


def parse_contact(text: str, parts: list[str]) -> dict:
    """接触问法的回答 → {contact, part, moving, see, desc, confidence}。"""
    blank = {"contact": False, "part": None, "moving": None, "see": "unknown",
             "desc": "", "confidence": 0.0, "note": "无法解析"}
    m = _JSON_RE.search(text or "")
    if not m:
        return blank
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return blank
    see = d.get("see")
    see = see if see in ("clear", "unclear") else "unknown"
    part = d.get("part")
    part = part if isinstance(part, str) and part in parts else None
    contact = bool(d.get("contact")) and see != "unclear"
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    moving = d.get("moving")
    return {"contact": contact, "part": part if contact else None,
            "moving": bool(moving) if isinstance(moving, bool) else None,
            "see": see, "desc": str(d.get("desc") or "")[:80],
            "confidence": conf, "note": ""}


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_answer(text: str, labels: list[Label]) -> dict:
    """模型的回答 → {label, body_part, confidence, note, desc, see}。答非所问一律当 none。

    see 和 label 是两件事，不能合并：label=None 可能是"不是这些行为"（模型正常工作、
    正确拒绝），也可能是"根本看不清"（送进来的候选本身是垃圾）。混成一个数的话，
    一批 none 回来完全不知道该改哪边——是调提示词，还是回头修出候选的那一层。
    """
    blank = {"label": None, "body_part": None, "confidence": 0.0,
             "note": "无法解析", "desc": "", "see": "unknown"}
    m = _JSON_RE.search(text or "")
    if not m:
        return blank
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return blank
    see = d.get("see")
    see = see if see in ("clear", "unclear") else "unknown"
    desc = str(d.get("desc") or "")[:80]
    note = str(d.get("note") or "")[:80]
    names = {lb.name: lb for lb in labels}
    label = d.get("label")
    if see == "unclear" or not isinstance(label, str) or label not in names:
        # 看不清时即使给了标签也不采信：提示词里写明了看不清一律 none
        return {"label": None, "body_part": None, "confidence": 0.0,
                "note": note, "desc": desc, "see": see}
    part = d.get("body_part")
    if not isinstance(part, str) or part not in names[label].parts:
        part = None
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence"))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"label": label, "body_part": part, "confidence": conf,
            "note": note, "desc": desc, "see": see}


def tile_frames(jpegs: list[bytes], cols: int = 0, cell: int = 336) -> bytes:
    """几帧拼成一张带序号的图。

    为什么拼而不是分开发几张：分开发时模型容易把它们当成几只不同的狗（俯拍裁出来的
    狗本来就难认），而舔和啃在单帧上几乎一样、差别全在几帧之间的变化。拼成一张、
    标上序号，"这是同一只狗的连续几秒"就写在图里，不用指望模型自己记住顺序。
    顺带省钱：一张图比 N 张图的 token 少。
    """
    import cv2
    import numpy as np

    if len(jpegs) <= 1:
        return jpegs[0] if jpegs else b""
    imgs = []
    for b in jpegs:
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue                      # 解不开的那张跳过，不要因为一张坏图整段不问
        sc = cell / max(im.shape[:2])
        if sc < 1:
            im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)))
        imgs.append((b, im))
    if not imgs:
        return b""
    if len(imgs) == 1:
        return imgs[0][0]                 # 只剩一张：原样发，标个"1"没有意义
    imgs = [im for _b, im in imgs]
    cols = cols or min(len(imgs), 3)
    rows = (len(imgs) + cols - 1) // cols
    ch = max(i.shape[0] for i in imgs)
    cw = max(i.shape[1] for i in imgs)
    sheet = np.full((rows * ch, cols * cw, 3), 32, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet[r * ch:r * ch + im.shape[0], c * cw:c * cw + im.shape[1]] = im
        # 序号画在格子里：拼图上没有别的地方能放，而模型要靠它知道时间顺序
        cv2.putText(sheet, str(i + 1), (c * cw + 6, r * ch + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(sheet, str(i + 1), (c * cw + 6, r * ch + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return bytes(np.asarray(buf).tobytes()) if ok else b""


def ask_contact(frames: list[bytes], parts: list[str], clip_s: float, llm: llmmod.LLM,
                client=None, http=None, tile: bool = True, debug: bool = False) -> dict:
    """只问"口鼻贴着哪个部位"（见 build_contact_prompt）。"""
    n = len(frames)
    sheet = tile_frames(frames) if (tile and n > 1) else b""
    send = [sheet] if sheet else frames
    system, user = build_contact_prompt(parts, n, clip_s, tiled=bool(sheet))
    t0 = time.monotonic()
    text, usage = llmmod.chat_vision(llm, system, user, send, max_tokens=400, client=client, http=http)
    out = parse_contact(text, parts)
    out["usage"] = usage
    out["latency_ms"] = int((time.monotonic() - t0) * 1000)
    if debug:
        out["_tile"] = sheet or (frames[0] if frames else b"")
        out["_system"], out["_user"], out["_raw"] = system, user, text
    return out


def ask(frames: list[bytes], labels: list[Label], clip_s: float, llm: llmmod.LLM,
        client=None, http=None, tile: bool = True, debug: bool = False) -> dict:
    """把一段的几帧送去问。返回 parse_answer 的结果 + usage。

    tile=True 把几帧拼成一张带序号的图再发（见 tile_frames）；拼不出来就退回分开发。

    debug=True 多带回 _tile（真正发出去的那张图）/ _system / _user / _raw（原始回答）。
    **模型答得不对时，第一件事是看这四样，不是改提示词**——2026-09-20 为「一条都判不出来」
    改了五轮提示词和采样参数，从来没看过一眼真正发出去的图；狗在 720p 俯拍里只占
    100x50 像素，裁进 384px 的格子再六格拼一张，舌头可能只剩几个像素，那样改什么都没用。
    """
    n = len(frames)
    sheet = tile_frames(frames) if (tile and n > 1) else b""
    send = [sheet] if sheet else frames
    system, user = build_prompt(labels, clip_s, n, tiled=bool(sheet))
    t0 = time.monotonic()
    text, usage = llmmod.chat_vision(llm, system, user, send, max_tokens=400, client=client, http=http)
    out = parse_answer(text, labels)
    out["usage"] = usage
    out["latency_ms"] = int((time.monotonic() - t0) * 1000)
    if debug:
        out["_tile"] = sheet or (frames[0] if frames else b"")
        out["_system"], out["_user"], out["_raw"] = system, user, text
    return out


# ── 合并 ──────────────────────────────────────────────────────────────

def merge_segments(wins: list[Window], answers: list[dict], min_conf: float = 0.5) -> list[dict]:
    """相邻/重叠、同类别的窗合成一段。置信度取最大，部位取多数。

    desc（模型看到了什么）跟着置信度最高的那个窗走，不是第一个窗：一段里几个窗，
    最有把握的那个窗的描述才最值得给人看。它是人复核时的第一眼信息——不对的话
    不用点开视频就能排掉。
    """
    segs: list[dict] = []
    for w, a in zip(wins, answers):
        if not a.get("label") or a.get("confidence", 0.0) < min_conf:
            continue
        last = segs[-1] if segs else None
        if last and last["label"] == a["label"] and w.start <= last["end_s"] + 1e-6:
            last["end_s"] = max(last["end_s"], w.end)
            if a["confidence"] > last["confidence"]:
                last["confidence"] = a["confidence"]
                last["note"] = a.get("note") or ""
                last["desc"] = a.get("desc") or ""
            last["_parts"].append(a.get("body_part"))
            last["n_clips"] += 1
        else:
            segs.append({"start_s": w.start, "end_s": w.end, "label": a["label"],
                         "confidence": a["confidence"], "_parts": [a.get("body_part")],
                         "note": a.get("note") or "", "desc": a.get("desc") or "",
                         "n_clips": 1})
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
    n_bad_box = 0
    for i in range(len(ts)):
        t = float(ts[i])
        if t < start_s or (end_s is not None and t > end_s):
            continue
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        # 2026-09-20 之前建的索引 box 列全是 (0,0,0,0)（见 embed.norm_box）。
        # 给个空框让 frames_for_window 自己去检测，别把 (0,0,0,0) 传下去——
        # 那样裁出来是空图，整段被静默跳过，而原因离这里隔着好几层
        if not (x2 > x1 and y2 > y1):
            n_bad_box += 1
            rec = {"t": round(t, 2), "boxes": [], "jpeg": b"", "motion": None, "_i": i}
            if prev is not None and t - float(ts[prev]) <= 2.5:
                rec["motion"] = float(np.linalg.norm(emb[i] - emb[prev]) / 2.0)
            prev = i
            out.append(rec)
            continue
        rec = {"t": round(t, 2), "boxes": [{"bbox": [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)], "conf": 1.0}],
               "jpeg": b"", "motion": None, "_i": i}
        if prev is not None and t - float(ts[prev]) <= 2.5:
            rec["motion"] = float(np.linalg.norm(emb[i] - emb[prev]) / 2.0)
        prev = i
        out.append(rec)
    if n_bad_box:
        _logger.warning("%s：%d/%d 帧的框是老索引里的空框，这几帧会当场重跑检测（重建索引可以省掉）",
                        os.path.basename(rel_path), n_bad_box, len(out))
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
        if not boxes:
            boxes = dog.detect(frame)        # 老索引里没有可用的框，当场检测一次
            if not boxes:
                continue
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
                    "desc": "", "see": "unknown",
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
    # 看不清有多少：这个数说的不是模型好不好，是**送进来的候选好不好**。
    # 居高不下就该回头修出候选的那一层（裁得太小 / 夜里太暗 / 狗被挡），
    # 而不是调提示词——「不是这些行为」和「根本看不清」混成一个 none 的话，
    # 这两条路分不开
    stats["unclear"] = sum(1 for a in answers if a.get("see") == "unclear")
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
