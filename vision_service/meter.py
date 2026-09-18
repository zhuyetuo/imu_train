"""本地模型的调用计数（进程内存，重启归零）。每个模型的推理入口调一下 record()。"""

from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_meter: dict[str, dict] = {}


def record(key: str, ms: float, frames: int = 1, ok: bool = True) -> None:
    """一次推理记一笔。frames：这一次处理了几张（批量检测一次几十张）。"""
    with _lock:
        m = _meter.setdefault(key, {"calls": 0, "frames": 0, "total_ms": 0.0, "max_ms": 0.0, "errors": 0, "last_at": None})
        m["calls"] += 1
        m["frames"] += max(1, int(frames))
        m["total_ms"] += float(ms)
        m["max_ms"] = max(m["max_ms"], float(ms))
        if not ok:
            m["errors"] += 1
        m["last_at"] = time.time()


def get(key: str) -> dict:
    with _lock:
        m = dict(_meter.get(key) or {"calls": 0, "frames": 0, "total_ms": 0.0, "max_ms": 0.0, "errors": 0, "last_at": None})
    m["avg_ms"] = round(m["total_ms"] / m["calls"]) if m["calls"] else 0
    m["avg_ms_per_frame"] = round(m["total_ms"] / m["frames"], 1) if m["frames"] else 0
    m["total_ms"] = round(m["total_ms"])
    m["max_ms"] = round(m["max_ms"])
    return m


def reset() -> None:
    with _lock:
        _meter.clear()


class timed:
    """with timed("dog", frames=n): ... 自动记耗时和成败。"""

    def __init__(self, key: str, frames: int = 1):
        self.key, self.frames = key, frames

    def __enter__(self):
        self.t0 = time.monotonic()
        return self

    def __exit__(self, et, ev, tb):
        record(self.key, (time.monotonic() - self.t0) * 1000, frames=self.frames, ok=et is None)
        return False
