"""显存到底被谁占了。

## 要解决什么

`nvitop` 只能看到「vision_service 这个进程占了 27 GiB」。进程里同时驻着检测、
分割、向量、姿态四个模型，还可能有 vLLM——**到底哪个吃掉的，看不出来**。
在容器里更看不出来：宿主机上连进程名都对不上。

于是「显存不够」这件事只能靠猜，而猜出来的结论通常是错的：最常见的两个真相是

  1. **torch 的缓存分配器不把显存还给系统。** 跑完一批之后那些显存在 torch
     眼里是空闲的（reserved 里的空闲块），但 nvidia-smi / nvitop 照样算你头上。
     看着像"模型占了 20 G"，其实是"峰值时用过 20 G，现在留着备用"。
  2. **vLLM 按比例预占。** gpu-memory-utilization=0.5 的意思是一上来就把整张卡的
     一半划走，跟它实际用多少无关。32G 的卡就是 16 G，不管你问不问它问题。

这两件事，任何外部工具都看不出来，只有进程自己知道。所以这里在**每个模型加载的
前后各读一次**显存，差值就是这个模型的常驻占用；再把 torch 的"已分配 / 已保留"
和整卡的用量摆在一起，剩下的那部分明确标成"非 torch"（onnxruntime 的 arena、
CUDA context、cuDNN 内核都在里面）。

## 为什么不用 nvidia-smi 的每进程用量

容器里拿不到别的进程，而且它给的是整个进程的和，正是这里要拆开的那个数。
"""

from __future__ import annotations

import contextlib
import logging
import threading

_logger = logging.getLogger("vision.gpumem")

# 模型名 → {"mib": 加载后常驻多少, "device": 在哪}
_loaded: dict[str, dict] = {}
_lock = threading.Lock()

MIB = 1024 * 1024


def _torch():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 没装 torch / 没卡：这一整块就不报数，不炸
        return None


def _alloc(t) -> int | None:
    """读当前已分配量。读不到就返回 None——**诊断工具绝不能把被诊断的东西搞挂**。

    torch 的版本、被 mock 掉的假 torch、边缘的驱动问题，都可能让这几个调用抛异常。
    这里抛出去的话，模型加载会当成"搬上卡失败"而退回 CPU：为了看一眼显存，
    把整个服务拖成 CPU 跑，那是赔本买卖。
    """
    try:
        return int(t.cuda.memory_allocated())
    except Exception:  # noqa: BLE001
        return None


@contextlib.contextmanager
def track(name: str, device: str | None = None):
    """套在模型加载外面：记下这一个模型常驻多少显存。

    只量 torch 的已分配量（allocated），不量 reserved——reserved 里含着缓存的
    空闲块，跟"这个模型占了多少"不是一回事，混进来会让每个模型的数都偏大。
    """
    t = _torch()
    before = _alloc(t) if t is not None else None
    try:
        yield
    finally:
        after = _alloc(t) if t is not None else None
        with _lock:
            if before is None or after is None:
                _loaded.setdefault(name, {"mib": 0.0 if t is None else None,
                                          "device": device or ("cpu" if t is None else "cuda")})
            else:
                _loaded[name] = {"mib": round(max(0, after - before) / MIB, 1),
                                 "device": device or "cuda"}


def external(name: str, device: str, why: str) -> None:
    """不归 torch 管的模型（姿态走 onnxruntime）：量不到，但要在明细里出现。

    不列出来的话，人看着明细加起来对不上整卡用量，只会以为是统计错了。
    量不到就写「量不到」，别编一个数。
    """
    with _lock:
        _loaded[name] = {"mib": None, "device": device, "note": why}


def forget(name: str) -> None:
    """模型卸载了就把它从明细里去掉，别留个幽灵条目。"""
    with _lock:
        _loaded.pop(name, None)


def release() -> dict:
    """把 torch 缓存着、但已经不用的那部分还给系统（empty_cache）。

    **不动任何模型**，纯粹是把"用过一次之后留着备用"的那些块交回去。
    显存紧的时候（比如要起 vLLM）先按这个，往往就够了，不用重启服务。
    """
    t = _torch()
    if t is None:
        return {"ok": False, "why": "没有 CUDA"}
    try:
        before = t.cuda.memory_reserved()
        t.cuda.empty_cache()
        after = t.cuda.memory_reserved()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "why": f"{type(e).__name__}: {e}"}
    freed = round((before - after) / MIB, 1)
    _logger.info("empty_cache：还回去 %.1f MiB（保留 %.1f → %.1f MiB）",
                 freed, before / MIB, after / MIB)
    return {"ok": True, "freed_mib": freed,
            "reserved_mib": round(after / MIB, 1)}


def trim(keep_mib: float = 4096.0) -> float:
    """缓存着没用的超过 keep_mib 就还一部分回去。建完一路索引时调一次。

    为什么留一截不全还：下一路马上又要用，全还了每次都得重新向驱动要，
    而向驱动要显存是有代价的（几十毫秒，还会更碎）。留几个 G 当周转，
    超出的才还——这样稳态占用从"峰值"降到"峰值一次之后的常驻"。

    为什么不干脆每次都 empty_cache：那等于把缓存分配器关掉，建索引会更慢，
    而慢下来的是每一路、每一批，省下的显存却只有在别人要用卡时才有意义。

    返回还回去多少 MiB（0 = 没到线，什么都没做）。
    """
    t = _torch()
    if t is None:
        return 0.0
    try:
        idle = (t.cuda.memory_reserved() - t.cuda.memory_allocated()) / MIB
        if idle <= keep_mib:
            return 0.0
        before = t.cuda.memory_reserved()
        t.cuda.empty_cache()
        freed = (before - t.cuda.memory_reserved()) / MIB
    except Exception:  # noqa: BLE001 诊断/回收都不该把正事搞挂
        return 0.0
    if freed > 0:
        _logger.info("建完一路，还回 %.0f MiB 缓存（原本闲着 %.0f MiB）", freed, idle)
    return round(freed, 1)


def report() -> dict:
    """显存明细。数都是 MiB。

    - models：每个模型加载时常驻了多少（torch 已分配量的差值）
    - torch_allocated / torch_reserved：torch 眼里正在用的 / 向系统要来留着的
    - cached_idle = reserved - allocated：**留着备用、其实空着的那部分**。
      这一块在 nvitop 里照样算你头上，但它随时能还（见 release()）
    - device_used / device_total：整张卡的用量（所有进程加起来）
    - non_torch：整卡用量里 torch 解释不了的那部分——onnxruntime 的 arena
      （姿态模型走的是它，不归 torch 管）、CUDA context、cuDNN 内核，
      以及**别的进程**（vLLM 是单独进程，它预占的那一大块就落在这里）
    """
    t = _torch()
    with _lock:
        models = dict(sorted(_loaded.items(), key=lambda kv: -(kv[1]["mib"] or 0)))
    out: dict = {"models": models}
    if t is None:
        out["why"] = "没有 CUDA，显存明细不适用"
        return out
    try:
        alloc = t.cuda.memory_allocated() / MIB
        reserved = t.cuda.memory_reserved() / MIB
        free_b, total_b = t.cuda.mem_get_info()
        used = (total_b - free_b) / MIB
    except Exception as e:  # noqa: BLE001 读不到就如实说，别半截数据看着像真的
        out["why"] = f"读显存失败：{type(e).__name__}: {e}"
        return out
    out.update({
        "torch_allocated_mib": round(alloc, 1),
        "torch_reserved_mib": round(reserved, 1),
        "cached_idle_mib": round(reserved - alloc, 1),
        "device_used_mib": round(used, 1),
        "device_total_mib": round(total_b / MIB, 1),
        # 整卡用量减掉 torch 保留的：本进程的 onnxruntime / CUDA context + 别的进程
        "non_torch_mib": round(max(0.0, used - reserved), 1),
        "gpu": t.cuda.get_device_name(0),
    })
    out["note"] = (
        f"torch 保留 {out['torch_reserved_mib']:.0f} MiB，其中 {out['cached_idle_mib']:.0f} MiB "
        f"是缓存着没在用的（POST /api/v1/gpu/release 可以还回去，不影响已加载的模型）；"
        f"另外 {out['non_torch_mib']:.0f} MiB 不归 torch 管——姿态模型走 onnxruntime、"
        f"CUDA context 本身要一两百 MiB，vLLM 是**另一个进程**（按 gpu-memory-utilization "
        f"比例预占，跟它实际用多少无关），都落在这一项里。"
    )
    return out
