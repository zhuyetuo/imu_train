"""显存明细：容器里 nvitop 只看得到「这个进程 27 GiB」，拆不开是谁占的。"""

from __future__ import annotations

from vision_service import gpumem


def test_没有卡时不炸_只是不报数(monkeypatch):
    """开发机、CPU 部署都没有卡。这一块不该因此报错——它是个诊断工具，
    诊断工具自己先挂掉最难受。"""
    monkeypatch.setattr(gpumem, "_torch", lambda: None)
    with gpumem.track("某模型", "cpu"):
        pass
    r = gpumem.report()
    assert "没有 CUDA" in r["why"] and r["models"]["某模型"]["mib"] == 0.0
    assert gpumem.release()["ok"] is False


def test_逐个模型量到的是它自己那一份(monkeypatch):
    """量的是 allocated 的差值，不是 reserved——reserved 里含着缓存的空闲块，
    混进来每个模型的数都会偏大。"""
    seq = iter([0, 500 * gpumem.MIB, 500 * gpumem.MIB, 1700 * gpumem.MIB, 1700 * gpumem.MIB])

    class _Cuda:
        @staticmethod
        def memory_allocated():
            return next(seq)

        @staticmethod
        def memory_reserved():
            return 2600 * gpumem.MIB

        @staticmethod
        def mem_get_info():
            return (32 * 1024 - 27000) * gpumem.MIB, 32 * 1024 * gpumem.MIB

        @staticmethod
        def get_device_name(_i):
            return "RTX 5090"

        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(gpumem, "_torch", lambda: type("T", (), {"cuda": _Cuda})())
    gpumem._loaded.clear()
    with gpumem.track("检测 yolo", "cuda"):
        pass
    with gpumem.track("画面向量 siglip", "cuda"):
        pass
    gpumem.external("姿态 rtmpose（onnxruntime）", "cuda", "torch 量不到")

    r = gpumem.report()
    assert r["models"]["检测 yolo"]["mib"] == 500.0
    assert r["models"]["画面向量 siglip"]["mib"] == 1200.0
    # 量不到的写 None，不编一个数；而且照样排进明细，不然各项加起来对不上整卡用量
    assert r["models"]["姿态 rtmpose（onnxruntime）"]["mib"] is None
    assert list(r["models"])[0] == "画面向量 siglip"        # 大的排前面

    # 缓存着没在用的那一块：nvitop 照样算你头上，但它随时能还
    assert r["torch_reserved_mib"] == 2600.0 and r["cached_idle_mib"] == 900.0
    # 整卡用量里 torch 解释不了的：onnxruntime / CUDA context / **别的进程**（vLLM）
    assert r["device_used_mib"] == 27000.0 and r["non_torch_mib"] == 24400.0
    assert "gpu-memory-utilization" in r["note"] and "另一个进程" in r["note"]


def test_读不到就如实说_绝不把模型加载搞挂(monkeypatch):
    """**诊断工具绝不能把被诊断的东西搞挂。** torch 版本、假 torch、驱动边缘问题
    都可能让 memory_allocated 抛异常——抛出去的话，模型加载会当成"搬上卡失败"
    退回 CPU：为了看一眼显存把整个服务拖成 CPU 跑，那是赔本买卖。
    """
    class _Bad:
        @staticmethod
        def memory_allocated():
            raise RuntimeError("驱动不高兴")

        memory_reserved = memory_allocated

        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(gpumem, "_torch", lambda: type("T", (), {"cuda": _Bad})())
    gpumem._loaded.clear()
    ran = []
    with gpumem.track("检测 yolo", "cuda"):
        ran.append(1)                       # 被套住的那段照常跑完
    assert ran == [1]
    assert gpumem.report()["models"]["检测 yolo"]["mib"] is None      # 量不到就写量不到
    assert gpumem.release()["ok"] is False and "驱动不高兴" in gpumem.release()["why"]


def test_建完一路自动还缓存_但留一截周转(monkeypatch):
    """实测：25 GiB 里 23 GiB 是"用过还留着"的缓存，模型权重只有 1.1 GiB。
    不还的话，三路并行撑出来的峰值会一直挂在卡上，别人要用卡就被挡住。

    但也不能每次全还：下一路马上又要用，全还了每次都得重新向驱动要（几十毫秒，
    还更碎）。留几个 G 周转，超出的才还。
    """
    state = {"reserved": 25000 * gpumem.MIB, "alloc": 1700 * gpumem.MIB, "emptied": 0}

    class _Cuda:
        @staticmethod
        def memory_allocated():
            return state["alloc"]

        @staticmethod
        def memory_reserved():
            return state["reserved"]

        @staticmethod
        def empty_cache():
            state["emptied"] += 1
            state["reserved"] = state["alloc"]

        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(gpumem, "_torch", lambda: type("T", (), {"cuda": _Cuda})())
    freed = gpumem.trim(keep_mib=4096)
    assert freed == 23300.0 and state["emptied"] == 1

    # 闲着的没到线：一下都别动，省得白白去要一次显存
    state["reserved"] = state["alloc"] + 1000 * gpumem.MIB
    assert gpumem.trim(keep_mib=4096) == 0.0 and state["emptied"] == 1
