"""可以同时挂几个模型，推理时按标签选。

以前只有一个：`LABEL_MODEL` 指哪个就跑哪个，换模型要么重启、要么调
`/api/v1/label/model/switch`（那是**全局切**，切完所有人都换了）。
想在平台上把两个模型并排跑同一批样本做对比，这两种都不行。

现在：`LABEL_MODELS` 里登记几个，请求里带 `model=<标签>` 选。
**不传就还是原来那个**——默认模型、默认行为，一个字节都没变。

## 配置

    LABEL_MODELS='stable_v2=models/stable_v2_rf/ml_rf.pkl,acc3=results_acc3/*/16hz_*/rf/ml_rf.pkl'

逗号分隔的 `标签=路径`，路径支持通配符但**必须唯一匹配**（跟 LABEL_MODEL
一个规矩：匹配到多个不替你挑，挑错了不报错，只会让平台上的结果对应到
另一份模型）。相对路径相对于仓库根。

不配 `LABEL_MODELS` 时只有默认那一个，跟以前完全一样。

## 找不到文件时跳过，不是报错

实验模型（比如还没训出来的 acc3）在别的机器上不存在，而这个服务是
线上标注在用的——**不能因为一个实验模型没训就起不来**。所以找不到就跳过，
但**会在日志里吵一句**：安静跳过的话平台上少一个选项，而日志里一切正常，
人会以为是平台没刷新。

默认模型不适用这条：它找不到就是真的起不来，那是 config.resolve_model_path()
自己的事，这里不接管。
"""

from __future__ import annotations

import glob
import logging
import os
import threading

from label_service import config

log = logging.getLogger("label_service")

# 默认模型的标签。请求不带 model 时用它；平台上显示成「线上模型」那一行
DEFAULT_TAG = "default"


def _resolve(pattern: str) -> str | None:
    """通配符 → 唯一一个路径。找不到或匹配到多个都返回 None（调用方负责吵）。"""
    p = pattern if os.path.isabs(pattern) else os.path.join(config.REPO_ROOT, pattern)
    if any(c in p for c in "*?["):
        hits = sorted(glob.glob(p))
    else:
        hits = [p] if os.path.isfile(p) else []
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        log.warning("LABEL_MODELS 里 %s 匹配到 %d 个文件，**不挂这个模型**："
                    "挑错了不报错，只会让平台上的结果对应到另一份模型。"
                    "写具体一点：%s", pattern, len(hits), hits)
    return None


def parse_spec(spec: str) -> list[tuple[str, str]]:
    """`a=路径,b=路径` → [(标签, 路径模式)]。格式不对的条目跳过并吵一句。"""
    out: list[tuple[str, str]] = []
    for chunk in (spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            log.warning("LABEL_MODELS 里 %r 不是 标签=路径 的形式，跳过", chunk)
            continue
        tag, pattern = chunk.split("=", 1)
        tag, pattern = tag.strip(), pattern.strip()
        if not tag or not pattern:
            log.warning("LABEL_MODELS 里 %r 标签或路径是空的，跳过", chunk)
            continue
        if tag == DEFAULT_TAG:
            # 占了默认标签会把默认模型顶掉，而默认模型是线上标注在用的
            log.warning("LABEL_MODELS 里不能用 %r 当标签（那是默认模型的），跳过", tag)
            continue
        out.append((tag, pattern))
    return out


class Registry:
    """标签 → {path, bundle, pool}。pool **按需建**，没人用就不占内存。

    一个模型一个进程池：worker 启动时各自加载模型（见 pool._init_worker），
    共用一个池的话没法保证某个任务落到加载了对的模型的那个 worker 上。
    代价是每挂一个模型多一份进程池——所以是懒加载，只有真被请求过才建。
    """

    def __init__(self):
        self._paths: dict[str, str] = {}
        self._pools: dict[str, object] = {}
        self._bundles: dict[str, dict] = {}
        self._lock = threading.Lock()

    # -- 登记 ---------------------------------------------------------

    def set_default(self, path: str, bundle: dict, pool_obj) -> None:
        self._paths[DEFAULT_TAG] = path
        self._bundles[DEFAULT_TAG] = bundle
        self._pools[DEFAULT_TAG] = pool_obj

    def load_extra(self, spec: str) -> None:
        for tag, pattern in parse_spec(spec):
            path = _resolve(pattern)
            if path is None:
                log.warning("⚠ 额外模型 %s 找不到（%s），**不挂它**。"
                            "训好之后重启服务就有了。", tag, pattern)
                continue
            self._paths[tag] = path
            log.info("额外模型 %s = %s（第一次用到时才加载）", tag, path)

    # -- 取用 ---------------------------------------------------------

    def tags(self) -> list[str]:
        return list(self._paths)

    def path_of(self, tag: str) -> str:
        return self._paths[tag]

    def resolve_tag(self, tag: str | None) -> str:
        """None / 空 → 默认模型。**不认识的标签必须报错，不能退回默认**：
        悄悄降级的话，结果存进库里标着 acc3，内容却是默认模型跑的，
        而这件事没有任何迹象。"""
        if not tag:
            return DEFAULT_TAG
        if tag not in self._paths:
            raise KeyError(
                f"没有叫 {tag!r} 的模型。现在挂着的：{sorted(self._paths)}。"
                "（实验模型没训出来时不会挂上，看启动日志里有没有 ⚠）")
        return tag

    def get(self, tag: str):
        """→ (bundle, pool)。第一次用到某个模型时才真的加载它。"""
        with self._lock:
            if tag not in self._pools:
                from label_service import pool as pool_mod
                from label_service.model_loader import load_model_bundle

                path = self._paths[tag]
                log.info("首次使用模型 %s，加载中：%s", tag, path)
                b = load_model_bundle(path)
                b.pop("model", None)        # 主进程只留元数据，模型在 worker 里
                b["model_path"] = path
                self._bundles[tag] = b
                self._pools[tag] = pool_mod.create_pool(path)
            return self._bundles[tag], self._pools[tag]

    def describe(self) -> list[dict]:
        """给平台的下拉用。**已经加载的才报类别/几何**——为了列个下拉就把
        每个模型都加载一遍太贵（一个 pkl 50MB，还要起进程池）。"""
        out = []
        for tag, path in self._paths.items():
            b = self._bundles.get(tag) or {}
            out.append({
                "tag": tag,
                "is_default": tag == DEFAULT_TAG,
                "model_path": path,
                "name": os.path.basename(os.path.dirname(path)) or os.path.basename(path),
                "loaded": tag in self._pools,
                "classes": b.get("classes"),
                "hz": b.get("hz"),
                "window_s": b.get("window_s"),
                "stride_s": b.get("stride_s"),
            })
        return out

    def shutdown(self) -> None:
        for tag, p in list(self._pools.items()):
            if tag == DEFAULT_TAG:
                continue        # 默认那个由 app 的 lifespan 自己关
            try:
                p.shutdown(wait=False, cancel_futures=True)
            except Exception:   # noqa: BLE001
                log.exception("关闭模型 %s 的进程池失败", tag)


registry = Registry()
