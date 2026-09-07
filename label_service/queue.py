"""
推理排队。进程池本身就会排队（提交多了就等空闲 worker），但那是"先来先服务"的
一锅粥：项目批量预标注一次丢几十个文件进来，标注员在工作台点一次「AI预标注」就得
排在几十个后面等好几分钟，体验很差。

这里加两件事：

1. 分档限流——批量（batch）最多占 总worker数-RESERVE 个槽位，剩下的永远留给交互式
   单次推理（interactive）。批量慢一点没人在乎，工作台点一下要马上有反应。
2. 队列状态——谁在跑、排了多少、平均一个文件多久，/queue 能查，日志也会记。

只管"同时有多少文件在算"，不改推理本身。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

PRIORITY_INTERACTIVE = "interactive"
PRIORITY_BATCH = "batch"


@dataclass
class QueueStats:
    workers: int = 0
    batch_slots: int = 0
    running: int = 0            # 正在算的文件数
    waiting: int = 0            # 在排队等槽位的文件数
    running_batch: int = 0
    running_interactive: int = 0
    done: int = 0
    failed: int = 0
    # 最近 50 个文件的耗时，用来估平均速度和排队时间
    recent_sec: list[float] = field(default_factory=list)

    @property
    def avg_sec(self) -> float | None:
        return sum(self.recent_sec) / len(self.recent_sec) if self.recent_sec else None

    def to_dict(self) -> dict:
        avg = self.avg_sec
        eta = None
        if avg and self.workers:
            # 排队的文件按 worker 数并行消化
            eta = round((self.waiting + self.running) / self.workers * avg, 1)
        return {
            "workers": self.workers,
            "batch_slots": self.batch_slots,
            "running": self.running,
            "running_batch": self.running_batch,
            "running_interactive": self.running_interactive,
            "waiting": self.waiting,
            "done": self.done,
            "failed": self.failed,
            "avg_sec_per_file": round(avg, 2) if avg else None,
            "eta_sec": eta,
        }


_stats = QueueStats()
_batch_sem: asyncio.Semaphore | None = None


def setup(workers: int, reserve: int) -> None:
    """服务启动时调一次。reserve = 给交互式留几个槽位（至少留 1）。"""
    global _batch_sem
    reserve = max(1, min(reserve, workers - 1)) if workers > 1 else 0
    slots = max(1, workers - reserve)
    _stats.workers = workers
    _stats.batch_slots = slots
    _batch_sem = asyncio.Semaphore(slots)


def stats() -> dict:
    return _stats.to_dict()


class slot:
    """
    async with queue.slot(priority): ... 　批量要抢信号量，交互式直接进。
    只统计不限流的那一档也会计入 running，/queue 看到的是真实并发。
    """

    def __init__(self, priority: str = PRIORITY_BATCH):
        self.priority = priority
        self._held = False
        self._t0 = 0.0

    async def __aenter__(self):
        batch = self.priority == PRIORITY_BATCH and _batch_sem is not None
        if batch:
            _stats.waiting += 1
            try:
                await _batch_sem.acquire()
            finally:
                _stats.waiting -= 1
            self._held = True
        _stats.running += 1
        if self.priority == PRIORITY_BATCH:
            _stats.running_batch += 1
        else:
            _stats.running_interactive += 1
        self._t0 = time.time()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        _stats.running -= 1
        if self.priority == PRIORITY_BATCH:
            _stats.running_batch -= 1
        else:
            _stats.running_interactive -= 1
        if exc_type is None:
            _stats.done += 1
            _stats.recent_sec.append(time.time() - self._t0)
            if len(_stats.recent_sec) > 50:
                del _stats.recent_sec[: len(_stats.recent_sec) - 50]
        else:
            _stats.failed += 1
        if self._held and _batch_sem is not None:
            _batch_sem.release()
        return False
