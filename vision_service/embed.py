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

from . import config, dog, gpumem, pose, posepart, seek, segmask

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
        import transformers  # noqa: F401  只验证装没装；具体类下面单独 import，失败要能看到真正原因
    except ImportError as e:
        _load_error = f"没装 transformers/torch：{e}（pip install transformers）"
        return
    try:
        _predownload()
        # 先用 SigLIP 自己的类（transformers 5.x 下 Auto* 的懒加载偶尔因为某个可选依赖缺失整个失败，
        # 只报一句「Could not import module 'AutoProcessor'」看不出真正原因）；不行再退回 Auto*
        errors = []
        m = None
        try:
            from transformers import SiglipModel, SiglipProcessor

            _processor = SiglipProcessor.from_pretrained(config.EMBED_MODEL)
            m = SiglipModel.from_pretrained(config.EMBED_MODEL)
        except Exception as e1:  # noqa: BLE001
            errors.append(f"SigLIP 类：{_cause(e1)}")
            try:
                from transformers import AutoModel, AutoProcessor

                _processor = AutoProcessor.from_pretrained(config.EMBED_MODEL)
                m = AutoModel.from_pretrained(config.EMBED_MODEL)
            except Exception as e2:  # noqa: BLE001
                errors.append(f"Auto 类：{_cause(e2)}")
        if m is None:
            raise RuntimeError("；".join(errors))
        want = config.EMBED_DEVICE
        _device = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
        with gpumem.track(f"画面向量 {config.EMBED_MODEL.split('/')[-1]}", _device):
            _model = m.to(_device).eval()
        _load_error = None
    except Exception as e:  # noqa: BLE001 权重下不动/版本不对，都要报出来
        _load_error = (f"加载 {config.EMBED_MODEL} 失败：{_cause(e)}"
                       f"（这台机器下不动权重的话，先在能上网的机器上下好放到 HF 缓存目录，或设 EMBED_MODEL 指向本地路径；"
                       f"报 import 错多半是 transformers / torchvision / pillow 版本对不上，pip install -U transformers）")


def _cause(e: BaseException) -> str:
    """异常连同它的 __cause__ 链一起写出来：transformers 的懒加载把真正原因藏在 cause 里。"""
    parts = []
    seen = 0
    cur: BaseException | None = e
    while cur is not None and seen < 4:
        parts.append(f"{type(cur).__name__}: {str(cur)[:300]}")
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return " ← ".join(parts)


def _as_tensor(out):
    """transformers 4.x 的 get_image_features/get_text_features 直接返回张量，
    5.x 返回 BaseModelOutputWithPooling（向量在 pooler_output）。两种都收。"""
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "last_hidden_state") and not hasattr(out, "norm"):
        return out.last_hidden_state[:, 0]
    return out


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
            from . import meter

            for i in range(0, len(jpegs), bs):
                imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs[i:i + bs]]
                with meter.timed("embed", frames=len(imgs)):
                    inputs = _processor(images=imgs, return_tensors="pt").to(_device)
                    feats = _as_tensor(_model.get_image_features(**inputs))
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype="float32")

    def encode_text(self, texts: list[str]):
        import torch

        _load()
        if _model is None:
            raise RuntimeError(_load_error or "编码器没加载")
        from . import meter

        with _lock, torch.no_grad(), meter.timed("embed", frames=len(texts)):
            inputs = _processor(text=texts, padding="max_length", return_tensors="pt").to(_device)
            feats = _as_tensor(_model.get_text_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.float().cpu().numpy()


_default_encoder = Encoder()

# ── 零样本"画面里有没有狗"：给狗检测漏检兜底 ─────────────────────────────
# 不出框，只回答有没有。提示词两组：像狗的 / 空房间的，取各组最高相似度比大小。
_DOG_PROMPTS = ["a photo of a dog", "a dog lying on the floor", "a dog sleeping curled up on tiles",
                "a black dog seen from above", "a dog in a kennel"]
_EMPTY_PROMPTS = ["an empty room with a tiled floor", "an empty dog kennel with nobody in it",
                  "a floor with nothing on it", "an empty cage"]
_prompt_vecs = None


def _prompts():
    global _prompt_vecs
    if _prompt_vecs is None:
        d = _default_encoder.encode_text(_DOG_PROMPTS)
        e = _default_encoder.encode_text(_EMPTY_PROMPTS)
        _prompt_vecs = (d, e)
    return _prompt_vecs


def looks_like_dog(frames_bgr: list, margin: float = 0.0, encoder=None) -> list[bool]:
    """每一帧：SigLIP 觉得更像"有狗"还是"空房间"。encoder 只在测试里换。"""
    import cv2
    import numpy as np

    if not frames_bgr:
        return []
    enc = encoder or _default_encoder
    if enc is _default_encoder:
        _load()
        if _model is None:
            raise RuntimeError(_load_error or "SigLIP 没加载")
        d, e = _prompts()
    else:
        d, e = enc.encode_text(_DOG_PROMPTS), enc.encode_text(_EMPTY_PROMPTS)
    jpegs = []
    for f in frames_bgr:
        ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        jpegs.append(buf.tobytes() if ok else b"")
    v = enc.encode_images(jpegs)
    dog_s = (v @ np.asarray(d).T).max(axis=1)
    empty_s = (v @ np.asarray(e).T).max(axis=1)
    return [bool(a - b > margin) for a, b in zip(dog_s, empty_s)]


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
            "model": config.EMBED_MODEL, "device": _device, "indexed_videos": n, "index_dir": config.EMBED_INDEX_DIR,
            "pose": pose.status(), "mask": segmask.status()}


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

def spent_summary(n: int = 30) -> dict:
    """最近建好的 n 份索引，各步各花了多少秒 → 汇总。

    建索引慢的时候，唯一有用的问题是"慢在哪一步"：解码+检测、姿态、抠狗、向量
    这四步的代价差着数量级，凭感觉调错了旋钮只会白慢一遍。每份索引的 meta 里
    本来就记着 spent，这里把最近几十份摊开加一加，直接说最重的是哪一步。

    只读 npz 的 meta（几 KB），不碰向量那几百万个浮点数。
    """
    import numpy as np

    d = config.EMBED_INDEX_DIR
    try:
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")]
    except OSError:
        return {"n": 0, "why": f"索引目录读不到：{d}"}
    files.sort(key=lambda p_: os.path.getmtime(p_), reverse=True)
    tot: dict[str, float] = {}
    frames = 0
    used = 0
    for f in files[:max(1, n)]:
        try:
            with np.load(f, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                nt = int(len(z["t"]))
        except Exception:  # noqa: BLE001 坏文件跳过，别让一份坏的挡住汇总
            continue
        sp = meta.get("spent") or {}
        if not sp:
            continue
        used += 1
        frames += nt
        for k, v in sp.items():
            tot[k] = tot.get(k, 0.0) + float(v or 0)
    total = sum(tot.values())
    name = {"scan": "解码+检测（老索引没拆）", "scan_wait": "等解码", "scan_detect": "检测",
            "scan_cpu": "裁图/帧差(CPU)", "pose": "姿态", "seg": "抠狗", "embed": "向量"}
    rows = [{"step": name.get(k, k), "sec": round(v, 1),
             "pct": round(v / total * 100, 1) if total else 0.0}
            for k, v in sorted(tot.items(), key=lambda kv: -kv[1])]
    out = {"n": used, "frames": frames, "total_sec": round(total, 1),
           "per_video_sec": round(total / used, 1) if used else 0.0, "steps": rows}
    if rows:
        top = rows[0]
        out["note"] = (f"最重的是「{top['step']}」，占 {top['pct']:.0f}%（{used} 份索引合计 "
                       f"{total:.0f} 秒，平均每路 {out['per_video_sec']:.0f} 秒）。"
                       "调旋钮之前先看这一行：占比低的那几步再怎么调都省不出时间。")
    return out


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
        # 姿态向量是后加的：老索引没有这一列，搜索时只用画面
        d["pose"] = z["pose"].astype("float32") if "pose" in z.files else None
        # 没抠背景那一列（一句话搜用）：也是后加的，老索引没有
        d["emb_raw"] = z["emb_raw"].astype("float32") if "emb_raw" in z.files else None
    _cache[rel_path] = (mtime, d)
    return d


def norm_box(boxes: list[dict]) -> tuple[float, float, float, float]:
    """几个狗框的并集，归一化的 (x1, y1, x2, y2)。

    **不要拿 seek.crop_rect 传 w=1,h=1 来算这个**：那个函数是给像素坐标用的，
    最后一步 `int()` 取整会把 0.35 砍成 0、0.75 也砍成 0——于是每一行都存成
    (0,0,0,0)。2026-09-20 才发现：444 路索引里的 box 列从写进去那天起全是零，
    而在此之前没有任何代码读过它，所以一直没人发现。

    读它的两条路（partask 按框裁帧、seek 走索引那条路）都表现成"裁出来是空图"，
    离真正的原因隔了好几层——为这件事猜了三轮。
    """
    xs1 = [b["bbox"][0] for b in boxes]
    ys1 = [b["bbox"][1] for b in boxes]
    xs2 = [b["bbox"][0] + b["bbox"][2] for b in boxes]
    ys2 = [b["bbox"][1] + b["bbox"][3] for b in boxes]
    return (round(min(xs1), 4), round(min(ys1), 4), round(max(xs2), 4), round(max(ys2), 4))


def box_ok(box) -> bool:
    """存下来的框能不能用。老索引里全是 (0,0,0,0)，读的那边要认得出来并自己兜底
    ——重建 444 路要一个多小时，不值得为这一列重来。"""
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def build(rel_path: str, full_path: str, every_sec: float = 1.0, force: bool = False,
          conf: float = 0.35, encoder: Encoder | None = None) -> dict:
    """给一路视频建索引。已有且模型一致就直接返回（force 重建）。"""
    import numpy as np

    enc = encoder or _default_encoder
    t0 = time.monotonic()
    # 抠不抠狗（背景涂灰）：抠了的和没抠的向量不在一个分布里，索引 meta 记着，不一致就重建
    use_mask = config.EMBED_MASK_BG and segmask.available()
    if not force:
        old = load(rel_path)
        want_raw = bool(use_mask and config.EMBED_RAW_TOO)
        if old is not None and old["meta"].get("model") == config.EMBED_MODEL \
                and bool(old["meta"].get("masked", False)) == bool(use_mask) \
                and bool(old["meta"].get("raw_wanted", False)) == want_raw:
            return {"n": int(len(old["t"])), "cached": True, "seconds": 0.0, "model": config.EMBED_MODEL,
                    "masked": bool(use_mask), "raw": want_raw}
    # 姿态可用就顺路算：每个有狗的帧一条姿态向量（整帧只在采样那一刻拿得到）
    use_pose = pose.available()

    # jpeg_raw：抠背景**之前**那张（按框裁的原图）。一句话搜要拿它算向量——
    # 文本塔是拿自然照片训的，跟涂灰背景的抠图对不上
    last = {"pose": None, "jpeg": None, "jpeg_raw": None}   # 上一个真算过的帧：静止的帧沿用它的姿态 / 抠图
    # 各步各花了多少秒。慢的时候不用猜是解码还是哪个模型：日志和索引 meta 里都记着
    spent = {"pose": 0.0, "seg": 0.0, "embed": 0.0}

    def on_frame(rec: dict, frame) -> None:
        if not use_pose:
            rec["pose"] = None
        elif rec.get("static") and last["pose"] is not None:
            rec["pose"] = last["pose"]
        else:
            t_ = time.monotonic()
            rec["pose"] = pose.frame_descriptor(frame, rec["boxes"])
            spent["pose"] += time.monotonic() - t_
            last["pose"] = rec["pose"]

    n_masked = 0
    n_static = 0

    def on_batch(items: list) -> None:
        # 抠狗：一批帧一起过分割模型（一张张送慢好几倍），抠到的替掉按框裁的那张 JPEG。
        # 静止的帧（画面跟上一帧没变）不抠，直接用上一帧抠好的图——狗睡着的几十分钟一张都不用算
        nonlocal n_masked, n_static
        import cv2

        t_seg = time.monotonic()
        # 静止的帧要沿用的是"前一帧"的图，前一帧可能就在这一批里——先算要算的，再按顺序补
        todo = [(rec, frame) for rec, frame in items if not rec.get("static")]
        if last["jpeg"] is None and items and items[0][0].get("static"):
            todo.insert(0, items[0])                # 一开头就是静止的、没有可沿用的：照常算
        imgs = segmask.masked_crop_batch([f for _r, f in todo], [r["boxes"] for r, _f in todo], seek.crop_rect) \
            if todo else []
        done = {}
        for (rec, _frame), img in zip(todo, imgs):
            rec["jpeg_raw"] = rec["jpeg"]           # 抠之前先留一份原图裁剪
            if img is not None:
                ok_, buf_ = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_:
                    rec["jpeg"] = bytes(np.asarray(buf_).tobytes())
                    n_masked += 1
            done[id(rec)] = True
        for rec, _frame in items:
            if id(rec) in done:
                last["jpeg"] = rec["jpeg"]
                last["jpeg_raw"] = rec.get("jpeg_raw")
            elif last["jpeg"] is not None:
                rec["jpeg"] = last["jpeg"]
                rec["jpeg_raw"] = last["jpeg_raw"]
                n_static += 1
        spent["seg"] += time.monotonic() - t_seg

    scan_stats: dict = {}
    t_scan = time.monotonic()
    samples = seek.sample_video(full_path, every_sec=every_sec, conf=conf, on_frame=on_frame,
                                on_batch=on_batch if use_mask else None, stats_out=scan_stats)
    # 过一遍视频的总时间里刨掉姿态和抠狗，剩下的是解码 + 狗检测 + 裁图
    # 「解码+检测」这一步实测占九成以上，所以它自己也要拆开报：等解码 / 检测 / CPU。
    # 三项之和就是原来的 scan，合计没变，只是能看出该调哪个旋钮了
    spent["scan"] = round(time.monotonic() - t_scan - spent["pose"] - spent["seg"], 1)
    for k_, name_ in (("wait_s", "scan_wait"), ("detect_s", "scan_detect"), ("cpu_s", "scan_cpu")):
        if k_ in scan_stats:
            spent[name_] = round(float(scan_stats[k_]), 1)
    if {"scan_wait", "scan_detect", "scan_cpu"} <= set(spent):
        del spent["scan"]      # 拆开之后别再留一个总数，不然汇总时重复计一遍
    with_dog = [s for s in samples if s["jpeg"] is not None]
    if with_dog:
        # 同一张图（静止沿用的）只算一次向量
        uniq: dict[bytes, int] = {}
        order = []
        for s_ in with_dog:
            if s_["jpeg"] not in uniq:
                uniq[s_["jpeg"]] = len(order)
                order.append(s_["jpeg"])
        t_ = time.monotonic()
        emb_u = enc.encode_images(order)
        spent["embed"] = round(time.monotonic() - t_, 1)
        emb = emb_u[[uniq[s_["jpeg"]] for s_ in with_dog]]
        # 一句话搜专用的那一条：**没抠背景**的裁剪。没开抠图时两条一样，就不重复存
        if use_mask and config.EMBED_RAW_TOO and all(s_.get("jpeg_raw") for s_ in with_dog):
            uniq_r: dict[bytes, int] = {}
            order_r = []
            for s_ in with_dog:
                if s_["jpeg_raw"] not in uniq_r:
                    uniq_r[s_["jpeg_raw"]] = len(order_r)
                    order_r.append(s_["jpeg_raw"])
            t_ = time.monotonic()
            emb_raw = enc.encode_images(order_r)[[uniq_r[s_["jpeg_raw"]] for s_ in with_dog]]
            spent["embed"] = round(spent["embed"] + time.monotonic() - t_, 1)
        else:
            emb_raw = None
    else:
        emb = np.zeros((0, 1), dtype="float32")
        emb_raw = None
    if use_mask and config.EMBED_RAW_TOO and with_dog and emb_raw is None:
        # 想要却没算出来：一句话搜会跳过这一路。不吭声的话，人只会看到"搜到的少"
        _logger.warning("%s：想存原图向量但没拿到（抠图那一步没跑？），一句话搜会跳过这一路", rel_path)
    t = np.array([s["t"] for s in with_dog], dtype="float32")
    box = np.array([norm_box(s["boxes"]) for s in with_dog], dtype="float32") \
        if with_dog else np.zeros((0, 4), dtype="float32")
    # 没测到点的帧记全 0（搜索时当"没姿态"，只用画面）
    pose_rows = np.array([s.get("pose") or [0.0] * pose.DIM for s in with_dog], dtype="float32") \
        if with_dog else np.zeros((0, pose.DIM), dtype="float32")
    n_pose = int(sum(1 for s in with_dog if s.get("pose")))
    meta = {"model": config.EMBED_MODEL, "every_sec": every_sec, "sampled": len(samples),
            "with_dog": len(with_dog), "built_at": time.time(), "path": rel_path,
            "pose": use_pose, "with_pose": n_pose, "masked": bool(use_mask), "with_mask": n_masked,
            "static_reused": n_static,
            # raw = 真存了那一列；raw_wanted = 这次想不想要。缓存按"想不想要"比：
            # 按"有没有"比的话，一旦哪一路没算出来（比如没抠成），每次建都会重建一遍
            "raw": emb_raw is not None, "raw_wanted": bool(use_mask and config.EMBED_RAW_TOO),
            "detected": scan_stats.get("detected", 0), "skipped": scan_stats.get("skipped", 0),
            "spent": {k: round(v, 1) for k, v in spent.items()}}
    os.makedirs(config.EMBED_INDEX_DIR, exist_ok=True)
    p = index_path(rel_path)
    tmp = p + ".tmp.npz"
    cols = {"t": t, "emb": emb.astype("float16"), "box": box, "pose": pose_rows.astype("float16"),
            "meta": np.array(json.dumps(meta, ensure_ascii=False))}
    if emb_raw is not None:
        cols["emb_raw"] = emb_raw.astype("float16")
    np.savez(tmp, **cols)
    os.replace(tmp, p)
    _cache.pop(rel_path, None)
    # 建完一路就把缓存着没用的还回去（留一截周转）。不还的话，三路并行撑出来的
    # 峰值会一直挂在卡上：实测 25 GiB 里 23 GiB 是这个，而模型权重只有 1.1 GiB。
    # 卡还是那张卡，别人（vLLM、另一个服务）要用时就被这堆"用过的空块"挡住了
    freed = gpumem.trim()
    return {"n": int(len(t)), "cached": False, "seconds": round(time.monotonic() - t0, 1),
            "model": config.EMBED_MODEL, "sampled": len(samples), "with_dog": len(with_dog),
            "with_pose": n_pose, "masked": bool(use_mask), "with_mask": n_masked, "static_reused": n_static,
            "raw": emb_raw is not None,
            "detected": scan_stats.get("detected", 0), "skipped": scan_stats.get("skipped", 0),
            "freed_mib": freed,
            "spent": {k: round(v, 1) for k, v in spent.items()}}


# ── 查询向量 ──────────────────────────────────────────────────────────

def _read_frame(full_path: str, t_s: float):
    """视频某一秒的那一帧（BGR）。单帧允许 seek（CAP_PROP_POS_MSEC）：只取一帧，
    VFR 的近似误差是零点几秒，对"这一帧长什么样"没影响。"""
    import cv2

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
    return frame


def frame_query(full_path: str, t_s: float, conf: float = 0.35, encoder: Encoder | None = None) -> dict:
    """视频某一秒的那一帧 → 框狗、裁、算向量。返回 {vec, has_dog, t}。"""
    import cv2
    import numpy as np

    enc = encoder or _default_encoder
    frame = _read_frame(full_path, t_s)
    boxes = dog.detect(frame, conf)
    h, w = frame.shape[:2]
    crop = None
    masked = False
    if boxes and config.EMBED_MASK_BG:
        # 跟建索引同一套：抠狗、涂灰。样例和索引必须同一种处理，不然比的不是一回事
        crop = segmask.masked_crop(frame, boxes, seek.crop_rect)
        masked = crop is not None
    if crop is None:
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
    # 姿态那一路：有狗且模型可用才有
    pvec = pose.frame_descriptor(frame, boxes) if boxes else None
    return {"vec": vec, "has_dog": bool(boxes), "t": float(t_s), "pose": pvec, "masked": masked}


# AP-10K 骨架：画姿态缩略图用（点的顺序见 pose.py 顶部）
_SKELETON = [(0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10),
             (4, 11), (11, 12), (12, 13), (4, 14), (14, 15), (15, 16)]


def draw_pose(img, kps, scores, offset=(0, 0), min_score: float | None = None):
    """在图上画关键点 + 骨架（原地改）。offset：图是从整帧裁出来的，点要减掉裁剪起点。"""
    import cv2
    import numpy as np

    thr = pose.MIN_KP_SCORE if min_score is None else min_score
    kps = np.asarray(kps, dtype="float32").reshape(-1, 2) - np.array(offset, dtype="float32")
    sc = np.asarray(scores, dtype="float32").reshape(-1)
    # 线粗 / 点大小按图的尺寸走，缩略图上不至于糊成一团
    lw = max(1, int(round(max(img.shape[:2]) / 200)))
    for a, b in _SKELETON:
        if a < len(sc) and b < len(sc) and sc[a] >= thr and sc[b] >= thr:
            cv2.line(img, (int(kps[a][0]), int(kps[a][1])), (int(kps[b][0]), int(kps[b][1])), (0, 200, 255), lw)
    for i, (x, y) in enumerate(kps):
        if i >= len(sc) or sc[i] < thr:
            continue
        color = (0, 0, 255) if i == pose.NOSE else (255, 80, 0) if i in pose.PAWS else (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), lw + 2, color, -1)
    return img


def frame_thumb(full_path: str, t_s: float, conf: float = 0.35, crop: bool = True, max_side: int = 320,
                view: str | None = None) -> bytes:
    """某一秒那一帧的缩略图 JPEG。view：
      mask  狗框那一块、抠掉背景（跟拿去比的那张一样；抠不到就退回 raw）
      raw   狗框那一块，原图
      pose  狗框那一块，原图上画关键点 + 骨架（姿态模型不可用就跟 raw 一样）
      box   整帧带检测框
    老参数 crop=True/False 对应 mask/box。没狗时 mask/raw/pose 都是整帧。"""
    import cv2
    import numpy as np

    view = view or ("mask" if crop else "box")
    frame = _read_frame(full_path, t_s)
    boxes = dog.detect(frame, conf)
    h, w = frame.shape[:2]
    img = None
    if view == "box" or not boxes:
        img = frame.copy()
        for b in boxes:
            bx, by, bw, bh = b["bbox"]
            cv2.rectangle(img, (int(bx * w), int(by * h)), (int((bx + bw) * w), int((by + bh) * h)), (26, 196, 82), 3)
    else:
        if view == "mask" and config.EMBED_MASK_BG:
            img = segmask.masked_crop(frame, boxes, seek.crop_rect, max_side=max_side)
        if img is None:
            x1, y1, x2, y2 = seek.crop_rect(boxes, w, h)
            img = frame[y1:y2, x1:x2].copy()
            if view == "pose" and pose.available():
                try:
                    r = pose.keypoints(frame, boxes)
                except Exception:  # noqa: BLE001 画不出来就给原图
                    r = None
                if r is not None:
                    draw_pose(img, r[0], r[1], offset=(x1, y1))
    scale = max_side / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    _ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return bytes(np.asarray(buf).tobytes())


def frame_preview(full_path: str, t_s: float, conf: float = 0.35, max_side: int = 640) -> dict:
    """给人看的：这一帧框到了哪几只狗、拿哪一块去搜。不算向量，不碰索引。

    返回 {t, has_dog, w, h, boxes: [{bbox:[x,y,w,h] 归一化, conf}], crop: [x1,y1,x2,y2] 归一化,
          jpeg: base64 的缩小整帧}。框和裁剪区都是归一化坐标，前端按显示尺寸画。
    """
    import base64

    import cv2
    import numpy as np

    frame = _read_frame(full_path, t_s)
    boxes = dog.detect(frame, conf)
    h, w = frame.shape[:2]
    if boxes:
        x1, y1, x2, y2 = seek.crop_rect(boxes, w, h)
        crop = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
    else:
        crop = [0.0, 0.0, 1.0, 1.0]
    scale = max_side / max(h, w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
    _ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return {"t": float(t_s), "has_dog": bool(boxes), "w": int(w), "h": int(h), "boxes": boxes, "crop": crop,
            "jpeg": base64.b64encode(np.asarray(buf).tobytes()).decode("ascii")}


def text_query(text: str, encoder: Encoder | None = None):
    enc = encoder or _default_encoder
    return enc.encode_text([text])[0]


# ── 搜索 ──────────────────────────────────────────────────────────────

def search(vec, rel_paths: list[str], top_k: int = 50, min_score: float = 0.0, gap_s: float = 3.0,
           exclude: tuple[str, float, float] | None = None, center: bool = True,
           pose_vec=None, pose_w: float | None = None, part: str | None = None,
           part_near_max: float | None = None, is_text: bool = False) -> dict:
    """在这些视频的索引里找最像的，按视频把相邻命中合成段。

    返回 {hits:[{path,t,score,vis_score,pose_score?}], segments:[{path,start_s,end_s,score,n}], searched, missing}。
    exclude=(path, t0, t1)：把样例自己那一段排掉，不然第一名永远是它自己。

    part：只要"鼻子够到了这个部位"的帧（几何硬条件，不是相似度）。SigLIP 分不清
    「头贴前左爪」和「头贴前右爪」——它看的是整体长相；这一条直接按关键点的距离卡，
    所以「舔后爪」能搜出来。认不出的部位名（腰、腹股沟……）**不筛**，如实在
    part_used 里说，免得人把空结果当成"索引里没这种数据"。

    center：减掉均值再比，去掉"共同背景"。三档：

      "video"（默认）每一帧减掉**它自己那一路**的平均向量，查询帧减它自己那一路的。
              比的是"这一帧相对这只狗平时的样子有什么不同"——狗的身份、毛色、房间、
              机位、红外色调在两边同时抵消，剩下的才是动作。**跨狗检索要用这一档。**
      "global" 所有视频一起算一个均值（老行为）。它只去掉全局共性，去不掉"这一路里
              这只狗长什么样"——同狗同房的帧余弦全在 0.95 以上，动作差别被淹没，
              结果永远是同一只狗同一个场景。2026-09-20 实测就是这个毛病。
      "none"   不减。

    传 True = "video"、False = "none"（老调用方传 True 本来就是想去共同背景，
    per-video 是更彻底的做法，不需要改调用方）。

    文本查询没有"自己那一路"可减，这时退回全局均值——文本向量本来就不带狗的身份。

    is_text：这是一句话搜。**改用没抠背景的那一列（emb_raw）比**——SigLIP 的文本塔
    是拿自然照片训的，而默认存的是"狗抠出来、背景涂灰"的图，不在它见过的分布里，
    文字跟它对不上，分永远在 0.2 上下。以图搜图两边都是抠图、同分布，所以那条路
    0.8 都有。没有那一列的老索引**这次不搜它**（scores 在两个空间里不可比，
    混着排出来的名次是假的），在 old_index 里如实报有几路被跳过。
    """
    import numpy as np

    # 先归一化再减均值：索引里存的都是单位向量，查询向量得先到同一尺度，减掉的均值才对得上
    q = np.asarray(vec, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-9)
    loaded: list[tuple[str, dict]] = []
    missing: list[str] = []
    for rp in rel_paths:
        d = load(rp)
        if d is None:
            missing.append(rp)
            continue
        loaded.append((rp, d))
    # 一句话搜：只认有「原图向量」那一列的索引；没有的这次跳过并如实上报。
    # 两个空间的分数不可比，混着排出来的名次是假的——宁可少搜几路，也别给一个假名次
    old_index: list[str] = []
    col = "emb"
    if is_text:
        col = "emb_raw"
        keep = []
        for rp, d in loaded:
            if d.get(col) is not None and len(d[col]) == len(d["t"]):
                keep.append((rp, d))
            else:
                old_index.append(rp)
        loaded = keep
    searched = len(loaded)
    mode = "video" if center is True else "none" if center is False else str(center or "none")
    # 每一路自己的均值 + 全局均值。索引存的是 float16，求和前转 float32，
    # 不然几百帧加起来精度全丢
    mus: dict[str, "np.ndarray"] = {}
    mu = None
    if mode in ("video", "global"):
        tot = sum(len(d["t"]) for _rp, d in loaded)
        if tot >= 20:
            sums = [(rp, len(d["t"]), d[col].astype("float32").sum(axis=0))
                    for rp, d in loaded if len(d["t"])]
            mu = sum(s_ for _rp, _n, s_ in sums) / tot
            if mode == "video":
                # 一路只有几帧时它自己的均值就约等于那几帧本身，减完剩不下东西——
                # 这种退回全局均值，别把一路好数据减成噪声
                mus = {rp: (s_ / n if n >= 20 else mu) for rp, n, s_ in sums}
    # 查询向量减哪一个：以图搜图时减它自己那一路的（exclude[0] 就是样例所在视频），
    # 文本查询没有"自己那一路"，退回全局
    q_mu = mus.get(exclude[0]) if (mus and exclude) else None
    if q_mu is None:
        q_mu = mu
    if q_mu is not None:
        q = q - q_mu
    q = q / (np.linalg.norm(q) + 1e-9)
    # 姿态那一路：样例有姿态、索引里也存了姿态的帧才混；两边缺一个就只看画面
    pw = config.POSE_W if pose_w is None else float(pose_w)
    pq = None
    if pose_vec is not None and pw > 0:
        pq = np.asarray(pose_vec, dtype="float32")
        pq = pq / (np.linalg.norm(pq) + 1e-9)
    used_pose = False
    # 部位是几何硬条件，跟相似度无关：认得出来才筛，认不出就整条不筛并如实上报
    part_key = posepart.part_of(part) if part else None
    n_part_kept = 0
    hits: list[dict] = []
    for rp, d in loaded:
        if not len(d["t"]):
            continue
        emb = d[col].astype("float32")
        sub = mus.get(rp, mu) if mode == "video" else mu
        if sub is not None:
            emb = emb - sub
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        scores = emb @ q
        vscores = scores                                # 纯画面分，混姿态之前的
        pscores = None
        if pq is not None and d.get("pose") is not None and len(d["pose"]) == len(d["t"]):
            P = d["pose"].astype("float32")
            has = np.linalg.norm(P, axis=1) > 0
            if has.any():
                pscores = P @ pq                         # 索引里的姿态向量本来就是单位向量
                # 没姿态的帧：姿态分记成画面分，等于这帧只看画面
                pscores = np.where(has, pscores, scores)
                scores = (1 - pw) * scores + pw * pscores
                used_pose = True
        keep = None
        if part_key and d.get("pose") is not None and len(d["pose"]) == len(d["t"]):
            keep = posepart.match(d["pose"], part_key, part_near_max)
            n_part_kept += int(keep.sum())
        for i in np.argsort(-scores):
            s = float(scores[i])
            if s < min_score:
                break
            if keep is not None and not keep[i]:
                continue                                 # 鼻子没够到这个部位，分再高也不是
            t = float(d["t"][i])
            if exclude and exclude[0] == rp and exclude[1] <= t <= exclude[2]:
                continue
            h = {"path": rp, "t": round(t, 2), "score": round(s, 4), "vis_score": round(float(vscores[i]), 4)}
            if pscores is not None:
                h["pose_score"] = round(float(pscores[i]), 4)
            hits.append(h)
    hits.sort(key=lambda h: -h["score"])
    hits = hits[:top_k]
    return {"hits": hits, "segments": group_hits(hits, gap_s), "searched": searched, "missing": missing,
            # 一句话搜用的是哪一列、有几路因为索引是旧版被跳过
            "text_space": ("原图" if col == "emb_raw" else "抠图") if is_text else None,
            "old_index": len(old_index),
            "centered": mu is not None, "center": mode if mu is not None else "none",
            "pose_used": used_pose, "pose_w": pw if used_pose else 0.0,
            # part_used=None 且 part 有值 = 这个部位判不了（腰、腹股沟……），**没筛**。
            # 前端要照实说，不然人会把"没筛出来的一堆"当成"筛过的结果"
            "part": part, "part_used": part_key, "part_frames": n_part_kept if part_key else None}


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
