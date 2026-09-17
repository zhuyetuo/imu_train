"""
画面向量索引：把每秒一张裁出来的狗算成向量存起来，之后"以图搜图"或"一句话搜"都是几毫秒。

跟「画面找片段」（seek.py，问大模型）是互补的两条路：

    seek    每个窗都要问一次模型，贵、慢，但能分清舔和啃、还能判部位
    embed   索引建一次，之后任何新问题（一帧样例、一句英文）都是向量比对，免费、瞬间；
            粗，"长得像"不等于"同一个动作"——但用来把 24 小时缩到几十段给人看，够了

**先框狗再算向量**：整帧算向量学到的是房间和地板，不是狗在干什么。这里复用
seek.sample_video 的采样（每秒一帧、YOLO 框、裁狗、缩到 512），只多做一步：过一遍
SigLIP 图像编码器。SigLIP 同时有文本编码器，所以同一份索引也能用英文句子查。

索引存成 npz（每个视频一个文件，按相对路径哈希命名）：t（秒，PTS）、emb（float16，
归一化过）、box。一小时视频约 3600 行 × 768 维 ≈ 5MB。改了模型要重建（meta 里记着）。

模型只在本地跑（这一步的价值就是索引一次、无限次免费查，走 API 没意义），
没装 transformers / 没下权重时 /status 如实说不可用，别的功能不受影响。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time

from . import config, dog, seek

_logger = logging.getLogger("vision_service.embed")

_model = None
_processor = None
_load_error: str | None = None
_lock = threading.RLock()
_device: str | None = None
_loading = False
# 下载进度（第一次要从 HF 拉约 400MB）：status 里报出来，deploy 脚本据此画进度条/估时间
_progress: dict = {"done": 0, "total": 0, "started": None, "file": None}

# 已加载的索引：rel_path -> (mtime, dict)。查一次要读好几十个 npz，缓存住
_cache: dict[str, tuple[float, dict]] = {}


# ── 编码器 ────────────────────────────────────────────────────────────

def _load(force: bool = False) -> None:
    global _model, _processor, _load_error, _device, _loading
    if _model is not None or (_load_error is not None and not force):
        return
    with _lock:
        if _model is not None:
            return
        _loading = True
        try:
            _do_load()
        finally:
            _loading = False
# 下载进度（第一次要从 HF 拉约 400MB）：status 里报出来，deploy 脚本据此画进度条/估时间
_progress: dict = {"done": 0, "total": 0, "started": None, "file": None}


def _predownload() -> None:
    """先把权重整个拉到 HF 缓存，边拉边记进度；之后 from_pretrained 直接命中缓存。

    不这么做的话进度只在 .run.log 里的 tqdm 条上，人在 deploy 那头看到的是几分钟的
    "available=False"，分不清是在下还是挂了。EMBED_MODEL 是本地目录就跳过。
    """
    if os.path.isdir(config.EMBED_MODEL):
        return
    try:
        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm as _base_tqdm
    except ImportError:
        return

    class _Tqdm(_base_tqdm):
        def __init__(self, *a, **kw):
            kw.setdefault("disable", False)
            super().__init__(*a, **kw)
            # 多个文件各一条：总量累加，进度按累加算
            _progress["total"] += int(self.total or 0)
            _progress["file"] = str(kw.get("desc") or "")
            if _progress["started"] is None:
                _progress["started"] = time.monotonic()

        def update(self, n=1):
            _progress["done"] += int(n or 0)
            return super().update(n)

    snapshot_download(config.EMBED_MODEL, tqdm_class=_Tqdm)


def _do_load() -> None:
    global _model, _processor, _load_error, _device
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
    except ImportError as e:
        _load_error = f"没装 transformers/torch：{e}（pip install transformers）"
        return
    try:
        _predownload()
        _processor = AutoProcessor.from_pretrained(config.EMBED_MODEL)
        m = AutoModel.from_pretrained(config.EMBED_MODEL)
        want = config.EMBED_DEVICE
        _device = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
        _model = m.to(_device).eval()
        _load_error = None
    except Exception as e:  # noqa: BLE001 权重下不动/版本不对，都要报出来
        _load_error = (f"加载 {config.EMBED_MODEL} 失败：{type(e).__name__}: {e}"
                       f"（这台机器下不动权重的话，先在能上网的机器上下好放到 HF 缓存目录，或设 EMBED_MODEL 指向本地路径）")


class Encoder:
    """默认编码器：SigLIP。测试里换成别的对象，只要有这两个方法。"""

    def encode_images(self, jpegs: list[bytes]):
        import io

        import numpy as np
        import torch
        from PIL import Image

        _load()
        if _model is None:
            raise RuntimeError(_load_error or "编码器没加载")
        out = []
        bs = config.EMBED_BATCH
        # 半精度：GPU 上快近一倍，向量差别在千分位以下。CPU 不用
        ac = torch.autocast("cuda", dtype=torch.float16) if _device == "cuda" else contextlib.nullcontext()
        with _lock, torch.no_grad(), ac:
            for i in range(0, len(jpegs), bs):
                imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs[i:i + bs]]
                inputs = _processor(images=imgs, return_tensors="pt").to(_device)
                feats = _model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype="float32")

    def encode_text(self, texts: list[str]):
        import torch

        _load()
        if _model is None:
            raise RuntimeError(_load_error or "编码器没加载")
        with _lock, torch.no_grad():
            inputs = _processor(text=texts, padding="max_length", return_tensors="pt").to(_device)
            feats = _model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu().numpy()


_default_encoder = Encoder()


def warmup() -> dict:
    """启动时后台加载（跟 SAM/狗检测一样），别让第一个建索引的人替所有人等。"""
    _load(force=True)
    return {"warm": _model is not None, "error": _load_error}


def status() -> dict:
    """**不触发加载**：加载 SigLIP 要十几秒，status 是给探活用的，得秒回。
    loading=True 表示后台还在加，过一会儿再看。"""
    n = 0
    try:
        n = sum(1 for f in os.listdir(config.EMBED_INDEX_DIR) if f.endswith(".npz"))
    except OSError:
        pass
    err = _load_error
    if _model is None and err is None:
        err = ("正在后台加载（第一次要下载权重约 400MB，按你的网速几分钟），过会儿再看" if _loading
               else "模型还没加载（启动预热关了或还没轮到），建索引时会加载")
    return {"available": _model is not None, "loading": _loading, "error": err if _model is None else None,
            "progress": download_progress(),
            "model": config.EMBED_MODEL, "device": _device, "indexed_videos": n, "index_dir": config.EMBED_INDEX_DIR}


def download_progress() -> dict | None:
    """{pct, done_mb, total_mb, speed_mbps, eta_s, file}；没在下载返回 None。"""
    total, done, started = _progress["total"], _progress["done"], _progress["started"]
    if not total or started is None:
        return None
    elapsed = max(1e-3, time.monotonic() - started)
    speed = done / elapsed                      # bytes/s
    remaining = max(0, total - done)
    eta = int(remaining / speed) if speed > 0 else None
    return {"pct": round(min(100.0, done / total * 100), 1), "done_mb": round(done / 1e6, 1),
            "total_mb": round(total / 1e6, 1), "speed_mbps": round(speed / 1e6, 2), "eta_s": eta,
            "file": _progress["file"], "finished": done >= total}


# ── 索引文件 ──────────────────────────────────────────────────────────

def index_path(rel_path: str) -> str:
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:24]
    return os.path.join(config.EMBED_INDEX_DIR, f"{h}.npz")


def has_index(rel_path: str) -> bool:
    return os.path.isfile(index_path(rel_path))


def load(rel_path: str) -> dict | None:
    """{t: (N,), emb: (N,D) float32, box: (N,4), meta: dict}；没有返回 None。"""
    import numpy as np

    p = index_path(rel_path)
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return None
    hit = _cache.get(rel_path)
    if hit and hit[0] == mtime:
        return hit[1]
    with np.load(p, allow_pickle=False) as z:
        d = {"t": z["t"].astype("float32"), "emb": z["emb"].astype("float32"), "box": z["box"],
             "meta": json.loads(str(z["meta"]))}
    _cache[rel_path] = (mtime, d)
    return d


def build(rel_path: str, full_path: str, every_sec: float = 1.0, force: bool = False,
          conf: float = 0.35, encoder: Encoder | None = None) -> dict:
    """给一路视频建索引。已有且模型一致就直接返回（force 重建）。"""
    import numpy as np

    enc = encoder or _default_encoder
    t0 = time.monotonic()
    if not force:
        old = load(rel_path)
        if old is not None and old["meta"].get("model") == config.EMBED_MODEL:
            return {"n": int(len(old["t"])), "cached": True, "seconds": 0.0, "model": config.EMBED_MODEL}
    samples = seek.sample_video(full_path, every_sec=every_sec, conf=conf)
    with_dog = [s for s in samples if s["jpeg"] is not None]
    if with_dog:
        emb = enc.encode_images([s["jpeg"] for s in with_dog])
    else:
        emb = np.zeros((0, 1), dtype="float32")
    t = np.array([s["t"] for s in with_dog], dtype="float32")
    box = np.array([seek.crop_rect(s["boxes"], 1, 1, margin=0.0, min_side=0) for s in with_dog], dtype="float32") \
        if with_dog else np.zeros((0, 4), dtype="float32")
    meta = {"model": config.EMBED_MODEL, "every_sec": every_sec, "sampled": len(samples),
            "with_dog": len(with_dog), "built_at": time.time(), "path": rel_path}
    os.makedirs(config.EMBED_INDEX_DIR, exist_ok=True)
    p = index_path(rel_path)
    tmp = p + ".tmp.npz"
    np.savez(tmp, t=t, emb=emb.astype("float16"), box=box, meta=np.array(json.dumps(meta, ensure_ascii=False)))
    os.replace(tmp, p)
    _cache.pop(rel_path, None)
    return {"n": int(len(t)), "cached": False, "seconds": round(time.monotonic() - t0, 1),
            "model": config.EMBED_MODEL, "sampled": len(samples), "with_dog": len(with_dog)}


# ── 查询向量 ──────────────────────────────────────────────────────────

def frame_query(full_path: str, t_s: float, conf: float = 0.35, encoder: Encoder | None = None) -> dict:
    """视频某一秒的那一帧 → 框狗、裁、算向量。返回 {vec, has_dog, t}。

    单帧允许 seek（CAP_PROP_POS_MSEC）：只取一帧，VFR 的近似误差是零点几秒，
    对"这一帧长什么样"没影响。
    """
    import cv2
    import numpy as np

    enc = encoder or _default_encoder
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        raise ValueError(f"打不开这个视频：{full_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"读不到 {t_s:.1f} 秒那一帧")
    boxes = dog.detect(frame, conf)
    h, w = frame.shape[:2]
    if boxes:
        x1, y1, x2, y2 = seek.crop_rect(boxes, w, h)
        crop = frame[y1:y2, x1:x2]
    else:
        crop = frame
    scale = 512 / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
    ok2, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    vec = enc.encode_images([bytes(np.asarray(buf).tobytes())])[0]
    return {"vec": vec, "has_dog": bool(boxes), "t": float(t_s)}


def text_query(text: str, encoder: Encoder | None = None):
    enc = encoder or _default_encoder
    return enc.encode_text([text])[0]


# ── 搜索 ──────────────────────────────────────────────────────────────

def search(vec, rel_paths: list[str], top_k: int = 50, min_score: float = 0.0, gap_s: float = 3.0,
           exclude: tuple[str, float, float] | None = None) -> dict:
    """在这些视频的索引里找最像的，按视频把相邻命中合成段。

    返回 {hits:[{path,t,score}], segments:[{path,start_s,end_s,score,n}], searched, missing}。
    exclude=(path, t0, t1)：把样例自己那一段排掉，不然第一名永远是它自己。
    """
    import numpy as np

    q = np.asarray(vec, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-9)
    hits: list[dict] = []
    missing: list[str] = []
    searched = 0
    for rp in rel_paths:
        d = load(rp)
        if d is None:
            missing.append(rp)
            continue
        searched += 1
        if not len(d["t"]):
            continue
        scores = d["emb"] @ q
        for i in np.argsort(-scores):
            s = float(scores[i])
            if s < min_score:
                break
            t = float(d["t"][i])
            if exclude and exclude[0] == rp and exclude[1] <= t <= exclude[2]:
                continue
            hits.append({"path": rp, "t": round(t, 2), "score": round(s, 4)})
    hits.sort(key=lambda h: -h["score"])
    hits = hits[:top_k]
    return {"hits": hits, "segments": group_hits(hits, gap_s), "searched": searched, "missing": missing}


def group_hits(hits: list[dict], gap_s: float = 3.0, pad_s: float = 1.0) -> list[dict]:
    """同一视频里时间相隔不超过 gap_s 的命中合成一段；每段前后各留 pad_s。"""
    by_path: dict[str, list[dict]] = {}
    for h in hits:
        by_path.setdefault(h["path"], []).append(h)
    segs: list[dict] = []
    for rp, hs in by_path.items():
        hs.sort(key=lambda h: h["t"])
        cur = None
        for h in hs:
            if cur and h["t"] - cur["_last"] <= gap_s:
                cur["_last"] = h["t"]
                cur["score"] = max(cur["score"], h["score"])
                cur["n"] += 1
            else:
                if cur:
                    segs.append(cur)
                cur = {"path": rp, "_first": h["t"], "_last": h["t"], "score": h["score"], "n": 1}
        if cur:
            segs.append(cur)
    out = []
    for s in segs:
        out.append({"path": s["path"], "start_s": round(max(0.0, s["_first"] - pad_s), 2),
                    "end_s": round(s["_last"] + pad_s, 2), "score": s["score"], "n": s["n"]})
    out.sort(key=lambda s: -s["score"])
    return out
