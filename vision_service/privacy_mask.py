"""
隐私遮罩：生成一份「狗框以外全黑」的视频，给标注员看。

要解决的事：不想让标注人员看到狗以外的画面（人、房间、别的东西）。

── 为什么必须在服务器上生成一份新视频 ─────────────────────────────────

只在播放器上盖一层黑等于没做：浏览器拿到的还是原视频，开发者工具里复制地址就能
下载原版。所以这里真的把像素涂黑，另存一份；平台对标注员只给这一份。原视频不动。

── 两遍 ─────────────────────────────────────────────────────────────

1. 找狗：每 every_sec 秒取一帧跑狗检测（跟 dog.scan_video 同一个模型 / 同一套类别），
   得到一串 (t, [框...])。
2. 涂黑：逐帧解码，按这一帧的时刻算出该留哪几块（rects_at），其余涂黑，重新编码。

── 每一帧留哪几块（rects_at）────────────────────────────────────────

- 扩框：每个框四边各往外扩 pad × 框宽 / 框高（默认 0.3）。抓挠、甩头这种快动作
  爪子会跑出原框。
- 前后合并：merge_s 秒内（默认 ±0.5 秒）所有检测到的框取并集。检测是每 0.25 秒一次，
  狗在两次之间移动的那一截靠这个兜住。
- 漏检补洞：前后 merge_s 内都没框（夜里红外、趴着、被栏杆挡住，实测 10~20% 的点会漏），
  就沿用之前最后一次的框，最多 hold_s 秒（默认 3 秒）。
- 再找不到 → 整帧全黑。隐私优先：宁可黑几秒，也不把整个画面露出来。
- 多只狗：每只各留一块，其余全黑。

── 时间戳：一帧都不能挪 ────────────────────────────────────────────

采集出来的视频是 VFR（每个 tick 一帧，间隔不固定），平台按视频时间跟 IMU 对齐。
所以新视频**每一帧沿用原帧的 PTS**，不能按固定帧率重排——差几百毫秒就对到别的
动作上去了。编完会把帧数、首末帧时间跟原视频对一遍，对不上就算失败、不交付。

没有音轨：原视频里要是有声音（人说话），那本身就是隐私，不带。

── 交付 ─────────────────────────────────────────────────────────────

写到 VIDEO_ROOT/_privacy_mask/<原相对路径>，先写 .part 再改名，最后写同名 .json
（参数、统计、原文件大小和修改时间）。有 .json 且 ok、原文件没变，才算处理完。
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass

from . import config

_logger = logging.getLogger("vision_service.privacy_mask")

#: 遮罩视频放在 VIDEO_ROOT 下的这个目录里，路径结构跟原视频一样。平台那边同一个 NAS
#: 挂载点直接就能读到（"_privacy_mask/" + 原相对路径）。
MASK_DIR = os.environ.get("PRIVACY_MASK_DIR", "_privacy_mask").strip("/") or "_privacy_mask"
#: 编码器："auto" 先试 NVENC（有卡快很多），开不起来退 libx264
CODEC = os.environ.get("PRIVACY_MASK_CODEC", "auto")
#: 同时做几个视频。一个视频的时间大半花在解码 / 涂黑 / 编码上（CPU + 显卡的编解码单元），
#: 狗检测那一步几个视频轮流用 GPU（dog.py 里有锁）。5090 上 3 路 GPU 才用一半；消费卡 NVENC
#: 同时编码路数有上限（现在的驱动一般 8 路），别开太大
WORKERS = max(1, int(os.environ.get("PRIVACY_MASK_WORKERS", "3")))


@dataclass
class MaskParams:
    pad: float = 0.3          # 四边各扩框宽 / 框高的多少
    hold_s: float = 3.0       # 漏检时沿用上一次框的最长秒数，超过整帧全黑
    merge_s: float = 0.5      # 前后多少秒内的框取并集
    every_sec: float = 0.25   # 多久检测一次
    conf: float = 0.2         # 检测置信度（跟扫描一样放低：漏检比误检代价大）


# ── 纯函数：每一刻留哪几块 ──────────────────────────────────────────────


def pad_box(b: list[float], pad: float) -> tuple[float, float, float, float]:
    """归一化 [x, y, w, h] → 扩完、裁到画面内的 (x1, y1, x2, y2)。"""
    x, y, w, h = b
    x1, y1 = max(0.0, x - pad * w), max(0.0, y - pad * h)
    x2, y2 = min(1.0, x + w + pad * w), min(1.0, y + h + pad * h)
    return (x1, y1, x2, y2)


class Timeline:
    """检测结果 → 任意时刻该留的块。检测点按时间排好，查询用二分。"""

    def __init__(self, dets: list[tuple[float, list[list[float]]]], p: MaskParams):
        dets = sorted(dets, key=lambda d: d[0])
        self.p = p
        self.t = [d[0] for d in dets]
        self.boxes = [[pad_box(b, p.pad) for b in d[1]] for d in dets]
        # 有框的那些点（漏检补洞时往前找「最后一次有框」用）
        self.hit_idx = [i for i, bs in enumerate(self.boxes) if bs]
        self.hit_t = [self.t[i] for i in self.hit_idx]

    def rects_at(self, t: float) -> tuple[list[tuple[float, float, float, float]], str]:
        """返回 (要留的块, 来源)。来源：'det' 前后合并窗口里有检测 / 'hold' 沿用之前的 /
        'black' 整帧全黑。"""
        p = self.p
        lo = bisect.bisect_left(self.t, t - p.merge_s)
        hi = bisect.bisect_right(self.t, t + p.merge_s)
        near = [r for i in range(lo, hi) for r in self.boxes[i]]
        if near:
            return near, "det"
        k = bisect.bisect_right(self.hit_t, t) - 1
        if k >= 0 and t - self.hit_t[k] <= p.hold_s:
            return list(self.boxes[self.hit_idx[k]]), "hold"
        return [], "black"


def apply_mask(img, rects) -> object:
    """img：HxWx3 ndarray。块外全部涂黑，返回新数组（不改原图）。"""
    import numpy as np

    h, w = img.shape[:2]
    out = np.zeros_like(img)
    for x1, y1, x2, y2 in rects:
        a, b = int(x1 * w), int(min(w, round(x2 * w)))
        c, d = int(y1 * h), int(min(h, round(y2 * h)))
        if b > a and d > c:
            out[c:d, a:b] = img[c:d, a:b]
    return out


# ── 路径 ───────────────────────────────────────────────────────────────


def masked_rel(rel_path: str) -> str:
    return f"{MASK_DIR}/{rel_path.lstrip('/')}"


def _paths(rel_path: str) -> tuple[str, str, str]:
    """(原视频绝对路径, 遮罩视频绝对路径, 说明 json 绝对路径)。路径穿越在这里挡。"""
    root = os.path.realpath(config.VIDEO_ROOT)
    rel = rel_path.lstrip("/")
    if rel.split("/", 1)[0] == MASK_DIR:
        raise ValueError("这已经是遮罩视频了")
    src = os.path.realpath(os.path.join(root, rel))
    dst = os.path.realpath(os.path.join(root, masked_rel(rel)))
    for p in (src, dst):
        if not p.startswith(root + os.sep):
            raise ValueError("非法路径")
    return src, dst, dst + ".json"


def _src_sig(src: str) -> dict:
    st = os.stat(src)
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def read_info(rel_path: str) -> dict | None:
    """处理完的说明。没处理过 / 原视频变了 / 上次失败 → None。"""
    try:
        src, dst, meta = _paths(rel_path)
        with open(meta, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not info.get("ok") or not os.path.isfile(dst):
        return None
    try:
        if info.get("src") != _src_sig(src):
            return None
    except OSError:
        return None
    return info


# ── 两遍 ───────────────────────────────────────────────────────────────


def detect_timeline(path: str, p: MaskParams, progress=None, check=None) -> list[tuple[float, list[list[float]]]]:
    """第一遍：每 every_sec 秒一帧跑狗检测。ffmpeg 抽帧 + 一批批送 GPU，跟 scan_video 一个路数。"""
    from . import dog, seek

    out: list[tuple[float, list[list[float]]]] = []
    pending: list[tuple[float, object]] = []

    def flush():
        if not pending:
            return
        res = dog.detect_batch([f for _t, f in pending], p.conf)
        for (t, _f), boxes in zip(pending, res):
            out.append((float(t), [b["bbox"] for b in boxes]))
        pending.clear()
        if progress:
            progress(out[-1][0])

    for t, frame in seek.iter_frames(path, p.every_sec):
        if check:
            check()
        pending.append((t, frame))
        if len(pending) >= config.DETECT_BATCH:
            flush()
    flush()
    return out


class _EncoderFailed(Exception):
    def __init__(self, name, err):
        super().__init__(f"{name}: {err}")
        self.name = name


def render(src: str, dst_part: str, tl: Timeline, codec_pref: str = CODEC, progress=None, check=None) -> dict:
    """第二遍：逐帧解码 → 涂黑 → 按原 PTS 编码。返回统计。"""
    import av

    order = ["h264_nvenc", "libx264"] if codec_pref == "auto" else [codec_pref]
    last_err: Exception | None = None
    for name in order:
        try:
            return _render_with(src, dst_part, tl, name, progress, check)
        except _EncoderFailed as e:
            last_err = e
            continue
        except av.error.FFmpegError as e:
            # NVENC 能开但编到一半报错（显存 / 会话数上限）：换软编从头来
            if name != order[-1]:
                _logger.warning("用 %s 编码失败，换下一个：%s", name, e)
                last_err = e
                continue
            raise
    raise RuntimeError(f"没有能用的编码器：{last_err}")


def _render_with(src: str, dst_part: str, tl: Timeline, codec: str, progress=None, check=None) -> dict:
    import av

    stats = {"frames": 0, "det": 0, "hold": 0, "black": 0}
    first_pts_s = last_pts_s = None
    with av.open(src) as inp:
        vin = inp.streams.video[0]
        vin.thread_type = "AUTO"
        dur = float(vin.duration * vin.time_base) if vin.duration else None
        with av.open(dst_part, "w", format="mp4", options={"movflags": "+faststart"}) as out:
            from fractions import Fraction

            rate = vin.average_rate or vin.guessed_rate or Fraction(25, 1)
            try:
                vout = out.add_stream(codec, rate=rate)
                vout.width = vin.codec_context.width
                vout.height = vin.codec_context.height
                vout.pix_fmt = "yuv420p"
                vout.time_base = vin.time_base
                vout.codec_context.time_base = vin.time_base
                vout.options = ({"crf": "23", "preset": "veryfast"} if codec == "libx264"
                                else {"preset": "p4", "rc": "vbr", "cq": "25"} if codec == "h264_nvenc" else {})
                vout.codec_context.open()
            except Exception as e:  # noqa: BLE001
                raise _EncoderFailed(codec, e) from e
            last_report = 0.0
            for frame in inp.decode(vin):
                if check and stats["frames"] % 50 == 0:
                    check()
                if frame.pts is None:
                    continue
                t = float(frame.pts * frame.time_base)
                rects, how = tl.rects_at(t)
                stats[how] += 1
                stats["frames"] += 1
                img = frame.to_ndarray(format="rgb24")
                newf = av.VideoFrame.from_ndarray(apply_mask(img, rects), format="rgb24")
                newf.pts = frame.pts
                newf.time_base = frame.time_base
                for pkt in vout.encode(newf):
                    out.mux(pkt)
                if first_pts_s is None:
                    first_pts_s = t
                last_pts_s = t
                if progress and t - last_report >= 5.0:
                    last_report = t
                    progress(t, dur)
            for pkt in vout.encode(None):
                out.mux(pkt)
    stats.update(codec=codec, first_s=first_pts_s, last_s=last_pts_s, duration_s=dur)
    return stats


def probe_frames(path: str) -> tuple[int, float | None, float | None]:
    """数一遍帧：(帧数, 首帧时间, 末帧时间)。只拆包不解码，很快。"""
    import av

    n, first, last = 0, None, None
    with av.open(path) as c:
        st = c.streams.video[0]
        pts = sorted(float(p.pts * p.time_base) for p in c.demux(st) if p.pts is not None)
    if pts:
        n, first, last = len(pts), pts[0], pts[-1]
    return n, first, last


def verify(src_stats: dict, dst: str, tol_s: float = 0.002) -> str | None:
    """编出来的跟原视频对一遍：帧数一样、首末帧时间一样。对不上返回原因。"""
    n, first, last = probe_frames(dst)
    if n != src_stats["frames"]:
        return f"帧数对不上：原 {src_stats['frames']}，遮罩版 {n}"
    if first is None or abs(first - (src_stats["first_s"] or 0.0)) > tol_s:
        return f"首帧时间对不上：原 {src_stats['first_s']}，遮罩版 {first}"
    if last is None or abs(last - (src_stats["last_s"] or 0.0)) > tol_s:
        return f"末帧时间对不上：原 {src_stats['last_s']}，遮罩版 {last}"
    return None


def _duration(path: str) -> float | None:
    """视频时长（秒），给进度条用；读不到就 None（进度条停在 0，不影响处理）。"""
    try:
        import av

        with av.open(path) as c:
            st = c.streams.video[0]
            if st.duration:
                return float(st.duration * st.time_base)
            if c.duration:
                return c.duration / 1_000_000
    except Exception:  # noqa: BLE001
        pass
    return None


class Cancelled(Exception):
    """停止了：这个视频不做了，半成品删掉。"""


def process(rel_path: str, p: MaskParams, progress=None, detector=None, check=None) -> dict:
    """处理一个视频，写好遮罩版和说明，返回说明。detector 给测试用（替掉真模型）。"""
    src, dst, meta = _paths(rel_path)
    if not os.path.isfile(src):
        raise ValueError(f"文件不存在：{rel_path}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t0 = time.time()

    def _p(stage, frac):
        if progress:
            progress(stage, max(0.0, min(1.0, frac)))

    sig = _src_sig(src)
    dur = _duration(src)
    if detector is not None:
        dets = detector(src, p, progress=lambda t: _p("detect", (t / dur) if dur else 0))
    else:
        dets = detect_timeline(src, p, progress=lambda t: _p("detect", (t / dur) if dur else 0), check=check)
    tl = Timeline(dets, p)
    part = dst + ".part"
    try:
        stats = render(src, part, tl, progress=lambda t, dur: _p("render", (t / dur) if dur else 0), check=check)
    except Cancelled:
        try:
            os.remove(part)
        except OSError:
            pass
        raise
    err = verify(stats, part)
    if err:
        try:
            os.remove(part)
        except OSError:
            pass
        raise RuntimeError(err)
    os.replace(part, dst)
    n = max(1, stats["frames"])
    info = {
        "ok": True, "src_rel": rel_path, "masked_rel": masked_rel(rel_path), "src": sig,
        "params": asdict(p), "codec": stats["codec"], "frames": stats["frames"],
        "detections": len(dets), "detections_with_dog": sum(1 for _t, b in dets if b),
        # 每帧是怎么留的：det 前后有检测 / hold 漏检沿用之前 / black 整帧全黑
        "ratio": {k: round(stats[k] / n, 4) for k in ("det", "hold", "black")},
        "first_s": stats["first_s"], "last_s": stats["last_s"],
        "elapsed_s": round(time.time() - t0, 1), "done_at": int(time.time()),
    }
    with open(meta + ".part", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)
    os.replace(meta + ".part", meta)
    return info


# ── 后台队列：WORKERS 个同时做；「先试这个任务」的插到最前面 ──────────────


_q: "queue.PriorityQueue[tuple[int, int, str, MaskParams]]" = queue.PriorityQueue()
_lock = threading.Lock()
_state: dict[str, dict] = {}     # rel_path -> {state, stage, progress, error}
_seq: dict[str, int] = {}        # rel_path -> 最新那次入队的序号（插队时旧的那条作废）
_counter = 0
_workers: list[threading.Thread] = []
# 暂停：正在做的停在原地（不丢进度），排队的不开工；继续就接着做
_paused = threading.Event()
# 停止：这些视频正在做的马上中断（半成品删掉），排队的拿掉
_cancel: set[str] = set()


def _checker(rel: str):
    def check():
        while True:
            with _lock:
                if rel in _cancel:
                    _cancel.discard(rel)
                    raise Cancelled(rel)
            if not _paused.is_set():
                return
            time.sleep(0.5)
    return check


def _run(q):
    while True:
        _prio, seq, rel, p = q.get()
        while _paused.is_set():
            time.sleep(0.5)
        try:
            with _lock:
                # 插队时同一个视频会再入队一次，旧的那条到这里就跳过
                if _seq.get(rel) != seq or _state.get(rel, {}).get("state") != "queued":
                    continue
                _state[rel] = {"state": "running", "stage": "detect", "progress": 0.0}

            def prog(stage, frac, _rel=rel):
                with _lock:
                    _state[_rel] = {"state": "running", "stage": stage, "progress": round(frac, 3)}

            try:
                process(rel, p, progress=prog, check=_checker(rel))
                with _lock:
                    _state.pop(rel, None)
            except Cancelled:
                with _lock:
                    _state.pop(rel, None)
                _logger.info("遮罩已停止：%s", rel)
            except Exception as e:  # noqa: BLE001 一个视频坏了不能把队列带死
                _logger.exception("遮罩失败：%s", rel)
                with _lock:
                    _state[rel] = {"state": "failed", "error": f"{type(e).__name__}: {e}"}
        finally:
            q.task_done()


def _ensure_workers_locked() -> None:
    _workers[:] = [w for w in _workers if w.is_alive()]
    while len(_workers) < WORKERS:
        w = threading.Thread(target=_run, args=(_q,), name=f"privacy-mask-{len(_workers)}", daemon=True)
        w.start()
        _workers.append(w)


def submit(paths: list[str], p: MaskParams, force: bool = False, front: bool = False) -> dict:
    """排队。已经处理好的（且原视频没变）跳过，除非 force。front：插到最前面（先试一个任务）。"""
    global _counter
    queued, skipped = 0, 0
    with _lock:
        _ensure_workers_locked()
        for rel in dict.fromkeys(paths):
            _paths(rel)  # 路径不对直接报出去
            cur = _state.get(rel, {}).get("state")
            if cur == "running" or (cur == "queued" and not front):
                skipped += 1
                continue
            if cur != "queued" and not force and read_info(rel) is not None:
                skipped += 1
                continue
            _counter += 1
            _seq[rel] = _counter
            _state[rel] = {"state": "queued"}
            _q.put((0 if front else 1, _counter, rel, p))
            queued += 1
    return {"queued": queued, "skipped": skipped, "pending": _q.qsize()}


def pause() -> dict:
    _paused.set()
    return {"paused": True}


def resume() -> dict:
    _paused.clear()
    return {"paused": False}


def cancel(paths: list[str] | None = None) -> dict:
    """停止：给了 paths 只停这些，不给全停。排队的拿掉，正在做的中断。已经做好的不动。"""
    removed = stopping = 0
    with _lock:
        targets = list(_state) if paths is None else list(dict.fromkeys(paths))
        for rel in targets:
            st = _state.get(rel, {}).get("state")
            if st == "queued":
                _state.pop(rel, None)
                _seq.pop(rel, None)       # 队列里那条到 worker 手上会被跳过
                removed += 1
            elif st == "running":
                _cancel.add(rel)
                stopping += 1
    return {"removed": removed, "stopping": stopping}


def status(paths: list[str]) -> dict[str, dict]:
    """每个视频现在的状态：done（带说明）/ queued / running（进度）/ failed / none。"""
    out: dict[str, dict] = {}
    for rel in dict.fromkeys(paths):
        with _lock:
            cur = dict(_state.get(rel) or {})
        if cur.get("state") in ("queued", "running"):
            out[rel] = cur
            continue
        info = read_info(rel)
        if info is not None:
            out[rel] = {"state": "done", "masked_rel": info["masked_rel"], "ratio": info.get("ratio"),
                        "frames": info.get("frames"), "done_at": info.get("done_at"),
                        "params": info.get("params")}
        elif cur.get("state") == "failed":
            out[rel] = cur
        else:
            out[rel] = {"state": "none"}
    return out


def available() -> dict:
    try:
        import av  # noqa: F401
    except ImportError:
        return {"available": False, "error": "没装 av（pip install av）"}
    from . import dog

    st = dog.status()
    if not st.get("available"):
        return {"available": False, "error": st.get("error") or "狗检测模型不可用"}
    return {"available": True, "error": None, "pending": _q.qsize(), "workers": WORKERS, "paused": _paused.is_set()}
