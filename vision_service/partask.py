"""把姿态几何捞出来的部位候选送去问大模型：几何负责召回，大模型负责判断。

## 为什么是这个分工

2026-09-20 实测把这件事钉死了：

  姿态几何（posepart）  能把 100 万帧压到几百条，**但判断不了**——俯拍时
                        RTMPose(AP-10K) 是分布外的，狗躺着不动它会连着几十秒
                        犯同一个错，"鼻子够到后爪"大半是误检的副产品。
                        联系表十五格里真在舔的一格都没有。
  大模型                一眼就能分开"在舔"和"睡着了鼻子朝后"——那正是几何
                        回答不了的那一半。

所以几何**只当召回向的粗筛**，判断交给大模型。这不是妥协，是各干各擅长的：
几何做不到的是语义，大模型做不到的是扫一百万帧（那要几千美元）。

## 成本

448 条候选、每条一张 3–5 格拼图，几美元的量级。而直接扫全量是几千美元——
两者差三个数量级，差的就是前面那层免费的粗筛。所以 --dry-run 先报会花多少，
确认了再跑。

用法（在算法机上）：

    python -m vision_service.partask --part 后爪 --near-max 0.25 --min-run 5 --dry-run
    python -m vision_service.partask --part 后爪 --near-max 0.25 --min-run 5 --limit 50 \\
        --labels 舔,啃,抓挠 --out /tmp/houzhua.jsonl

结果看两个数：

    unclear 有多少     说的**不是模型好不好，是送进去的候选好不好**。居高不下
                       就回头修裁图 / 夜间亮度，不是调提示词
    命中有多少         448 条里真有多少能用。哪怕只有 10%，那也是几十条带部位的
                       样本，比现在的零强，而且每条都带依据，人复核一眼能过
"""

from __future__ import annotations

import argparse
import json
import os
import time

from . import config, embed, llm as llmmod, posepart, seek

# 问的时候给哪些类别。部位从 --part 推出来：问"是不是在舔后爪"比问"在干嘛"准得多，
# 但**必须留 none 这个出口**——候选里大半是睡觉的狗，不给出口模型会硬选一个，比没有还糟
DEFAULT_LABELS = ("舔", "啃", "抓挠")
LABEL_DESC = {
    "舔": "舌头反复舔舐身体某个部位，头保持在那个部位上",
    "啃": "用门牙啃咬身体某个部位，下颌有小幅快速开合",
    "抓挠": "用后爪快速挠身体某处，后腿高频往复",
}


def frames_around(full_path: str, rel_path: str, t: float, n: int = 6,
                  span_s: float = 3.0, max_side: int = 384,
                  step_s: float = 0.3) -> tuple[list[bytes], str]:
    """候选时刻前后几帧，按索引里存的框裁好 → (帧, 说不清楚时的原因)。

    ## 采样间隔为什么是 0.3 秒而不是 1 秒

    索引是每秒一帧，最早这里也照着每秒取一帧、跨 6 秒取 4 张。**那是个物理上
    不可能完成的任务**：舔和啃的定义里都有"反复"，而那是 2–4 Hz 的动作——
    每秒采一帧，两帧之间舌头来回好几个周期，采到的永远是随机相位，
    看不出任何"反复"。2026-09-20 实测：先是模型见什么都说"在舔"，
    加了"看不到反复就答 none"之后又变成见什么都说 none——两次都不是模型的错，
    是给它的四张图里本来就没有它要找的东西。

    这里**直接从视频解码，采样率不受索引限制**（索引只用来拿框和判时间范围）。
    0.3 秒 × 6 帧 = 1.5 秒窗口，2–4 Hz 的动作在相邻帧之间能看出位置差。

    **不重跑检测**：框就在索引里（建索引时那一次检测的结果），重跑一次既慢又可能
    跟当初判断用的框不一样。拿不到索引就返回空，调用方跳过这条——宁可少问一条，
    也别拿一张跟判断依据不一致的图去问。

    为什么要连原因一起返回：拿不到帧有四种完全不同的原因（索引找不到 / 索引是空的 /
    那一段时间不在索引里 / 解码解不出来），它们的解法各不相同。2026-09-20 实测
    20 条全军覆没，只说一句"取不到帧"，只能靠猜——这是这一轮里同一类错误的第四次。
    """
    import cv2
    import numpy as np

    d = embed.load(rel_path)
    if d is None:
        return [], f"索引里没有这一路：{embed.index_path(rel_path)}（rel={rel_path}）"
    if not len(d["t"]):
        return [], f"索引是空的（0 帧）：{rel_path}"
    ts = np.asarray(d["t"], dtype="float32")
    if "box" not in d or len(d.get("box", ())) != len(ts):
        return [], f"索引里没存框（老版本索引？）：{rel_path}"
    sel = np.flatnonzero((ts >= t - span_s) & (ts <= t + span_s))
    if not len(sel):
        return [], (f"{t:.0f}s 前后 {span_s:.0f} 秒不在索引里"
                    f"（索引覆盖 {float(ts[0]):.0f}~{float(ts[-1]):.0f}s）")
    # 框按索引里离得最近的那一帧取。狗在 1.5 秒里不会跑出框，所以一个框够用；
    # 而且用同一个框裁，几帧之间的差别就只剩狗自己的动作，正是要让模型看的东西
    near_i = int(sel[np.argmin(np.abs(ts[sel] - t))])
    box0 = np.asarray(d["box"][near_i], dtype="float32")
    half = step_s * (n - 1) / 2.0
    lo, hi = max(0.0, t - half), t + half
    out: list[bytes] = []
    n_redetect = 0
    for ft, frame in seek.iter_frames(full_path, step_s, start_s=lo, end_s=hi + step_s * 0.5):
        if len(out) >= n:
            break
        box = box0
        h, w = frame.shape[:2]
        if not embed.box_ok(box):
            # 2026-09-20 之前建的索引，box 列全是 (0,0,0,0)（见 embed.norm_box）。
            # 重建 444 路要一个多小时，不值得为这一列重来——当场对这一帧重跑一次检测。
            # 比用索引里的框慢，但结果一样对
            from . import dog
            boxes = dog.detect(frame)
            if not boxes:
                continue
            x1, y1, x2, y2 = seek.crop_rect(boxes, w, h, margin=0.0, min_side=0)
            n_redetect += 1
        else:
            x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        # 留一圈边：紧贴框裁出来常把爪子或尾巴切掉，而那正是要判的部位
        x1, y1 = max(0, int(x1 - bw * 0.25)), max(0, int(y1 - bh * 0.25))
        x2, y2 = min(w, int(x2 + bw * 0.25)), min(h, int(y2 + bh * 0.25))
        crop = frame[y1:y2, x1:x2]
        if not crop.size:
            continue
        sc = max_side / max(crop.shape[:2])
        if sc < 1:
            crop = cv2.resize(crop, (int(crop.shape[1] * sc), int(crop.shape[0] * sc)))
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            out.append(bytes(np.asarray(buf).tobytes()))
    if not out:
        return [], (f"解码没给出 {lo:.1f}~{hi:.1f}s 那几帧（ffmpeg / cv2 都没对上时间）："
                    f"{os.path.basename(full_path)}")
    return out, ""


def labels_for(part_key: str, names, wide: bool = True) -> list[seek.Label]:
    """问的时候带上部位选项。

    **默认给全部六个部位（四爪 + 尾根 + 颈部），不只给筛出来的那一两个。**

    原来只给"后左爪 / 后右爪"，理由是"候选本来就是鼻子够到后爪筛出来的，让模型从
    全表选等于把粗筛的信息扔了"。2026-09-20 的对照组把这个理由推翻了：只给两个选项时，
    一个乱猜的模型有一半概率蒙对部位，于是正式组和对照组的命中率都是 15%——那个数
    完全没有意义。给六个选项之后，蒙对的概率从 1/2 降到 1/6，而且**模型选的部位跟
    几何判的对不对得上，本身就成了一个可验证的信号**。

    wide=False 退回只给相关的那一两个（想对比两种问法时用）。
    """
    slots = posepart.PARTS.get(part_key) or ()
    narrow = [posepart.SLOT_NAMES[i] for i in slots] or [part_key]
    parts = list(posepart.SLOT_NAMES) if wide else narrow
    return [seek.Label(name=n, description=LABEL_DESC.get(n, ""), parts=list(parts)) for n in names]


def control_hits(index_dir: str, part: str, n: int, seed: int = 0) -> list[dict]:
    """对照组：几何明确判定「鼻子离任何爪子都很远」的帧。

    **为什么必须有这一组**：我们告诉模型"这几帧是舔/啃/抓挠的候选，部位在后左爪/
    后右爪之间选"，而几何筛出来的本来就全是"头靠近后爪"的画面。一个无脑总说
    「舔-后爪」的模型也能拿到很高的命中率——那个数就完全没有意义。

    对照组问的是**同一个问题、同样的选项**，只是画面里狗的鼻子离爪子很远。
    对照组命中率接近 0，正式那组的命中率才可信；两组差不多，说明模型在顺着
    提示词猜，得换问法。

    挑的是 nearest 距离 > 1.5 体长的帧（鼻子离最近的爪子还有一个半身长，
    不可能在舔），每路最多取几帧，跨视频均匀撒开。
    """
    import random

    import numpy as np

    rng = random.Random(seed)
    pool: list[dict] = []
    for name in sorted(os.listdir(index_dir)) if os.path.isdir(index_dir) else []:
        if not name.endswith(".npz"):
            continue
        try:
            with np.load(os.path.join(index_dir, name), allow_pickle=False) as z:
                if "meta" not in z.files or "pose" not in z.files:
                    continue
                meta = json.loads(str(z["meta"]))
                rows, ts = z["pose"], z["t"]
        except Exception:  # noqa: BLE001
            continue
        if rows.size == 0 or len(rows) != len(ts):
            continue
        dists, _ = posepart.decode(rows)
        slot, best = posepart.nearest(dists)
        far = np.flatnonzero((slot >= 0) & (best > 1.5) & (posepart.spread(rows) >= posepart.MIN_SPREAD))
        if not len(far):
            continue
        path = str(meta.get("path") or "")
        for i in rng.sample(list(far), min(2, len(far))):
            pool.append({"path": path, "t": round(float(ts[i]), 2),
                         "dist": round(float(best[i]), 3), "slot": "（对照：离爪子很远）",
                         "control": True})
    rng.shuffle(pool)
    return pool[:n]


def ask_one(hit: dict, part_key: str, labels: list[seek.Label], llm, *,
            n_frames: int, span_s: float, video_root: str, step_s: float = 0.3,
            client=None, http=None, debug: bool = False) -> dict:
    """一条候选 → 问一次。返回候选本身 + 模型的回答；拿不到帧就 skipped。"""
    full = os.path.join(video_root, hit["path"])
    if not os.path.isfile(full):
        return {**hit, "skipped": f"视频不在：{full}"}
    frames, why = frames_around(full, hit["path"], hit["t"], n=n_frames, span_s=span_s,
                                step_s=step_s)
    if not frames:
        return {**hit, "skipped": why or "取不到帧"}
    # 告诉模型这几帧一共跨多久：1.5 秒和 6 秒，"有没有反复"的判断标准完全不同
    a = seek.ask(frames, labels, step_s * (len(frames) - 1), llm, client=client, http=http,
                 debug=debug)
    return {**hit, "n_frames": len(frames), **{k: v for k, v in a.items() if k != "usage"},
            "usage": a.get("usage") or {}}


def agreement(rounds: list[list[dict]]) -> str:
    """同一批候选问几轮，看模型自己跟自己合不合得上。

    为什么这个比命中率更该先看：2026-09-20 同样 20 条问了两次，一次 7 条命中、
    一次 3 条，**只有 2 条重合**。判断不稳到这个程度时，命中率是多少都没意义，
    调提示词也没用——那说明模型压根没在看画面，在别的地方找答案。
    """
    n = min(len(r) for r in rounds)
    same_label = same_part = 0
    for i in range(n):
        labs = {r[i].get("label") for r in rounds}
        same_label += len(labs) == 1
        if len(labs) == 1 and next(iter(labs)):
            same_part += len({r[i].get("body_part") for r in rounds}) == 1
    hit_counts = [sum(1 for x in r if x.get("label")) for r in rounds]
    lines = ["", f"  自洽性（同一批问了 {len(rounds)} 轮）：",
             f"    每轮命中数：{hit_counts}",
             f"    {n} 条里 {same_label} 条（{same_label / max(n, 1) * 100:.0f}%）几轮答的类别一样"]
    if same_part:
        lines.append(f"    其中 {same_part} 条部位也一样")
    if all(c == 0 for c in hit_counts):
        # 全答 none 时"几轮答得一样"是白送的，不能当成稳定
        lines.append("    （每轮都是 0 命中，所以这个 100% 是白送的，说明不了稳不稳）")
    elif same_label / max(n, 1) < 0.8:
        lines.append("    ⚠ 自己跟自己都对不上 → **命中率是多少都没意义**。"
                     "不是提示词不够好，是模型没在看画面；换模型或换问法（比如只问"
                     "「几帧之间狗的头有没有反复动」这种单一可判的事）再说。")
    return "\n".join(lines)


def dump_debug(results: list[dict], out_dir: str, limit: int = 8) -> str:
    """把真正发出去的那张拼图、完整提示词、模型原始回答存下来。

    2026-09-20 为「一条都判不出来」改了五轮提示词和采样参数，**从来没看过一眼
    真正发出去的图**。狗在 720p 俯拍里只占 100x50 像素，裁进 384px 的格子再六格
    拼一张，舌头可能只剩几个像素——那样改什么提示词、换什么模型都没用。

    同一个毛病之前在部位检索上犯过一次（调了四轮参数才去看画面）。先看输入。
    """
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for i, r in enumerate(results[:limit]):
        tile = r.get("_tile")
        if not tile:
            continue
        base = os.path.join(out_dir, f"{i:02d}_{'ctl' if r.get('control') else 'hit'}")
        with open(base + ".jpg", "wb") as f:
            f.write(tile)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(f"{r.get('path')}  t={r.get('t')}s  几何判：{r.get('slot')} {r.get('dist')} 体长\n")
            f.write(f"发了 {r.get('n_frames')} 帧，拼图 {len(tile) / 1024:.0f}KB\n")
            f.write(f"\n===== system =====\n{r.get('_system')}\n")
            f.write(f"\n===== user =====\n{r.get('_user')}\n")
            f.write(f"\n===== 模型原始回答 =====\n{r.get('_raw')}\n")
        n += 1
    return (f"存了 {n} 组到 {out_dir}/（*.jpg 是真正发出去的图，*.txt 是提示词和原始回答）"
            "\n  **先看图**：狗在图上有多大？舌头看得见吗？看不见的话改提示词/换模型都没用")


def summarize(results: list[dict], llm=None) -> str:
    """跑完之后看什么：unclear 说的是候选好不好，命中说的是这条路值不值。

    有对照组时先看对照组：对照组也高的话，正式那组的命中率是假的（模型在顺着
    提示词猜），后面几个数都不用看了。
    """
    n = len(results)
    ok = [r for r in results if not r.get("skipped")]
    ctl = [r for r in ok if r.get("control")]
    ok = [r for r in ok if not r.get("control")]
    unclear = [r for r in ok if r.get("see") == "unclear"]
    hits = [r for r in ok if r.get("label")]
    tin = sum(int((r.get("usage") or {}).get("input") or 0) for r in ok)
    tout = sum(int((r.get("usage") or {}).get("output") or 0) for r in ok)
    lines = [
        "",
        f"问了 {len(ok) + len(ctl)}/{n} 条"
        f"（{n - len(ok) - len(ctl)} 条取不到帧）"
        + (f"，其中正式 {len(ok)} 条、对照 {len(ctl)} 条" if ctl else ""),
        f"  看不清 {len(unclear)}（{len(unclear) / max(len(ok), 1) * 100:.0f}%）"
        "  ← 这个数说的**不是模型好不好，是送进去的候选好不好**。"
        "高了就回头修裁图 / 夜间亮度，不是调提示词",
        f"  命中   {len(hits)}（{len(hits) / max(len(ok), 1) * 100:.0f}%）"
        "  ← 几何粗筛的真实精度。哪怕只有 10%，那也是几十条带部位的样本",
    ]
    if llm is not None:
        lines.append(f"  花费   约 ${llmmod.estimate_usd(llm, tin, tout):.2f}"
                     f"（in {tin:,} / out {tout:,} token）")
    if ctl:
        c_hit = sum(1 for r in ctl if r.get("label"))
        rate, c_rate = len(hits) / max(len(ok), 1), c_hit / len(ctl)
        lines.append("")
        lines.append(f"  对照组 {len(ctl)} 条（几何判定鼻子离爪子 >1.5 体长，不可能在舔）"
                     f"命中 {c_hit}（{c_rate * 100:.0f}%）")
        if not len(hits) and not c_hit:
            # 两边都是 0：这不是"顺着提示词猜"，是**一条都没判出来**，解法完全相反
            lines.append("  ⚠ 两组都是 0 → 模型对所有片段都答 none。这**不是**精度问题，"
                         "是提示词太保守、或者给的几帧里根本没有它要找的东西。")
            lines.append("    先查采样间隔：舔/啃是 2-4Hz 的反复动作，"
                         "帧间隔 1 秒只能采到随机相位，要求它'看到反复'是不可能完成的任务（--step 0.3）。")
        elif c_rate >= rate * 0.6:
            lines.append("  ⚠ 对照组跟正式组差不多 → **正式组那个命中率是假的**："
                         "模型在顺着提示词猜（我们已经告诉它这是舔/啃候选、部位在后爪里选）。")
            lines.append("    先换问法（比如把 none 的描述写得更具体、或者不告诉它候选来自哪个部位），"
                         "再谈命中率。")
        else:
            lines.append(f"  ✓ 对照组明显低于正式组（{c_rate * 100:.0f}% vs {rate * 100:.0f}%）"
                         "→ 模型是真在看画面，正式组那个数可信。")
        for r in sorted([x for x in ctl if x.get("label")],
                        key=lambda x: -(x.get("confidence") or 0))[:3]:
            lines.append(f"    对照组误报：{r.get('desc') or ''}；{r.get('note') or ''}")
    by = {}
    for r in hits:
        by[(r.get("label"), r.get("body_part"))] = by.get((r.get("label"), r.get("body_part")), 0) + 1
    if by:
        lines.append("")
        lines.append("  命中分布：" + "，".join(
            f"{lb}{'-' + (p or '不分部位')} {c}" for (lb, p), c in
            sorted(by.items(), key=lambda kv: -kv[1])))
        lines.append("")
        lines.append("  前几条（依据是给人一眼判断用的，不对的不用点开视频就能排掉）：")
        for r in sorted(hits, key=lambda x: -(x.get("confidence") or 0))[:8]:
            lines.append(f"    {(r.get('confidence') or 0) * 100:3.0f}%  {r.get('label')}"
                         f"-{r.get('body_part') or '?'}  {r['t']:.0f}s  "
                         f"{os.path.basename(r['path'])[9:28]}")
            lines.append(f"          {r.get('desc') or ''}；{r.get('note') or ''}")
    return "\n".join(lines)


def probe(hit: dict, video_root: str, span_s: float = 3.0, n: int = 4) -> str:
    """对一条候选，把每一层实际发生了什么原样打出来。

    为什么要这个：取不到帧时，索引层、ffmpeg 层、cv2 层各有自己的失败方式，而每一层
    的错都被上一层吞掉了——CLI 里连 logger 的 warning 都看不见（没配 logging）。
    2026-09-20 为这件事猜了两轮还没猜对，所以不猜了：把 ffmpeg 的真实命令、退出码、
    stderr，和 cv2 实际吐出来的时间戳，一条条打出来。
    """
    import subprocess

    import cv2
    import numpy as np

    out = [f"== {hit['path']}  t={hit['t']}s =="]
    full = os.path.join(video_root, hit["path"])
    out.append(f"视频：{full}  存在={os.path.isfile(full)}  "
               f"大小={os.path.getsize(full) / 1e6:.0f}MB" if os.path.isfile(full)
               else f"视频：{full}  **不存在**")
    if not os.path.isfile(full):
        return "\n".join(out)

    ip = embed.index_path(hit["path"])
    out.append(f"索引：{ip}  存在={os.path.isfile(ip)}")
    d = embed.load(hit["path"])
    if d is None:
        return "\n".join(out + ["索引读不出来，后面不用看了"])
    ts = np.asarray(d["t"], dtype="float32")
    out.append(f"索引里 {len(ts)} 帧，覆盖 {float(ts[0]):.0f}~{float(ts[-1]):.0f}s")
    sel = np.flatnonzero((ts >= hit["t"] - span_s) & (ts <= hit["t"] + span_s))
    if len(sel) > n:
        sel = sel[np.linspace(0, len(sel) - 1, n).astype(int)]
    want = [round(float(ts[i]), 2) for i in sel]
    out.append(f"要的时刻：{want}")
    if not want:
        return "\n".join(out)
    lo, hi = min(want), max(want)
    start_s, end_s = max(0.0, lo - 0.5), hi + 0.5

    out.append(f"\n-- 分辨率（cv2 读的，ffmpeg 靠它算每帧字节数）--")
    try:
        w, h = seek._video_size(full)
        out.append(f"   {w}x{h}")
    except Exception as e:  # noqa: BLE001
        out.append(f"   读不到：{type(e).__name__}: {e}")
        w = h = 0

    for hw in (True, False):
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
        if hw:
            cmd += ["-hwaccel", "cuda"]
        cmd += ["-ss", f"{start_s:.3f}", "-i", full, "-t", f"{end_s - start_s:.3f}",
                "-vf", "fps=1/1.0", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        out.append(f"\n-- ffmpeg hwaccel={hw} --")
        out.append("   " + " ".join(cmd))
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
            nbytes = len(r.stdout)
            out.append(f"   退出码={r.returncode}  stdout={nbytes:,} 字节"
                       f"（每帧 {w * h * 3:,}，够 {nbytes // max(w * h * 3, 1)} 帧）")
            err = r.stderr.decode("utf-8", "ignore").strip()
            out.append(f"   stderr：{err[:400] or '（空）'}")
        except Exception as e:  # noqa: BLE001
            out.append(f"   跑不起来：{type(e).__name__}: {e}")

    out.append(f"\n-- cv2（seek 到 {start_s - 2:.1f}s 再按 PTS 往前找）--")
    cap = cv2.VideoCapture(full)
    try:
        out.append(f"   打开={cap.isOpened()}  总帧数={cap.get(cv2.CAP_PROP_FRAME_COUNT):.0f}"
                   f"  fps={cap.get(cv2.CAP_PROP_FPS):.2f}")
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_s - 2.0) * 1000.0)
        out.append(f"   seek 后 POS_MSEC={cap.get(cv2.CAP_PROP_POS_MSEC):.0f}")
        got = []
        for _ in range(400):
            if not cap.grab():
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            got.append(ms / 1000.0)
            if ms / 1000.0 > end_s:
                break
        out.append(f"   grab 到 {len(got)} 帧，时间戳前几个：{[round(x, 2) for x in got[:8]]}")
        if got:
            out.append(f"   最后一个：{got[-1]:.2f}s")
    finally:
        cap.release()
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="把姿态捞出来的部位候选送去问大模型")
    ap.add_argument("--part", required=True, help="部位名，比如 后爪")
    ap.add_argument("--index-dir", default=config.EMBED_INDEX_DIR)
    ap.add_argument("--near-max", type=float, default=0.25)
    ap.add_argument("--min-run", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=60.0)
    ap.add_argument("--per-video", type=int, default=20)
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="问哪几个类别，逗号分隔")
    ap.add_argument("--frames", type=int, default=6, help="每条候选送几帧（拼成一张图）")
    ap.add_argument("--step", type=float, default=0.3,
                    help="相邻两帧隔多少秒。**别用 1 秒**：舔/啃是 2-4Hz 的反复动作，"
                         "每秒一帧只能采到随机相位，看不出任何反复")
    ap.add_argument("--span", type=float, default=3.0, help="候选时刻前后多少秒内算有效（对索引）")
    ap.add_argument("--limit", type=int, default=0, help="最多问几条（0 = 全部）。先小样本试一下再放开")
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--out", help="每条一行 JSON 写到这里，之后能反复分析不用重问")
    ap.add_argument("--sheet", metavar="PNG", help="把命中的拼成一张带骨架的图")
    ap.add_argument("--dry-run", action="store_true", help="只报会问多少条、大概多少钱，不调 API")
    ap.add_argument("--dump", metavar="DIR",
                    help="把**真正发出去的那张拼图**、完整提示词、模型原始回答存下来。"
                         "模型答得不对时第一件事是看这个，不是改提示词")
    ap.add_argument("--repeat", type=int, default=1, metavar="N",
                    help="每条问 N 次，报自洽率。同一批候选两次跑出 7 条和 3 条命中、"
                         "只有 2 条重合的话，再调提示词也没用——模型根本没在看画面")
    ap.add_argument("--narrow", action="store_true",
                    help="部位选项只给筛出来的那一两个（默认给全部六个）。"
                         "只给两个时乱猜有一半概率蒙对，命中率会虚高")
    ap.add_argument("--control", type=int, default=0, metavar="N",
                    help="混进 N 条对照（几何判定鼻子离爪子很远的帧），问同样的问题。"
                         "对照组也高就说明模型在顺着提示词猜，正式组的命中率是假的")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="只对前 N 条候选逐层打印实际发生了什么（索引/ffmpeg/cv2），不调 API。"
                         "取不到帧时用它，别猜")
    args = ap.parse_args()
    # CLI 里不配 logging 的话，各层的 warning（比如 ffmpeg 换一种解码）一句都看不见
    import logging
    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")

    key = posepart.part_of(args.part) or args.part
    r = posepart.find(args.index_dir, key, args.near_max, True, args.per_video,
                      args.min_gap, args.min_run)
    if not r["known"]:
        print(f"不认识的部位：{args.part} → {key}（认得的：{'、'.join(posepart.PARTS)}）")
        return
    hits = r["hits"][:args.limit] if args.limit else r["hits"]
    if args.probe:
        for h in hits[:args.probe]:
            print()
            print(probe(h, config.VIDEO_ROOT, args.span, args.frames))
        return
    print(f"{args.part} → {key}：几何粗筛出 {len(r['hits']):,} 条，这次问 {len(hits):,} 条"
          f"（近到 {args.near_max} 体长、连着 {args.min_run} 秒）")

    llm = llmmod.from_env()
    if args.dry_run or llm is None:
        # 一张 4 格拼图约 1.5k token（384px 的格子），输出约 120 token；数量级用的，不当账
        est_in, est_out = len(hits) * 1500, len(hits) * 120
        # 没 key 也要把钱估出来：「值不值得花这个钱」这个决定，缺了数字根本没法做
        print(f"  预估 in {est_in:,} / out {est_out:,} token")
        if llm is not None and (llm.price_in or llm.price_out):
            print(f"    {llm.label():<34} 约 ${llmmod.estimate_usd(llm, est_in, est_out):.2f}  ← 当前")
        elif llm is not None:
            print(f"    {llm.label():<34} 估不出钱——这家的价目表没配，"
                  "填 SEEK_PRICE_IN / SEEK_PRICE_OUT（$/百万 token）就能算")
        # Claude 那几档写死在代码里当参照：换一家值不值，得有个数能比
        for m, (pin, pout) in config.SEEK_PRICE_PER_M.items():
            cost = est_in / 1e6 * pin + est_out / 1e6 * pout
            cur = "  ← 当前" if (llm is not None and llm.provider == "anthropic"
                                and m == llm.model) else ""
            print(f"    {m:<34} 约 ${cost:.2f}{cur}")
        if llm is None:
            print("\n  没配 key，只能 dry-run。写进 vision_service/.env 再跑，两种写法：")
            print("    ANTHROPIC_API_KEY=sk-ant-...                   # 只用 Claude")
            print("  或者用平台上已经配好的那一家（豆包 / 智谱 / 本地服务都行）：")
            print("    SEEK_PROVIDER=doubao")
            print("    SEEK_API_KEY=...")
            print("    SEEK_MODEL=doubao-seed-1-6-vision-250815")
            print("    SEEK_PRICE_IN=... / SEEK_PRICE_OUT=...         # 可选，$/百万 token")
        else:
            print("\n  确认了去掉 --dry-run 再跑。建议先 --limit 20 看看准不准，再放开。")
        return

    if args.control:
        ctl = control_hits(args.index_dir, key, args.control)
        print(f"  另外混进 {len(ctl)} 条对照（鼻子离爪子 >1.5 体长，不可能在舔）"
              "——它们跟正式的问同一个问题，用来验命中率是不是模型顺着提示词猜出来的")
        hits = hits + ctl
    labels = labels_for(key, [x.strip() for x in args.labels.split(",") if x.strip()],
                        wide=not args.narrow)
    print(f"  问 {llm.label()}，类别 {'/'.join(l.name for l in labels)}，"
          f"部位选项 {'/'.join(labels[0].parts)}")
    print(f"  每条送 {args.frames} 帧、间隔 {args.step}s（窗口 {args.step * (args.frames - 1):.1f}s）"
          "——舔/啃是 2-4Hz 的反复动作，间隔太大只能采到随机相位")
    t0 = time.monotonic()
    from concurrent.futures import ThreadPoolExecutor

    def one(h):
        try:
            return ask_one(h, key, labels, llm, n_frames=args.frames, span_s=args.span,
                           step_s=args.step, video_root=config.VIDEO_ROOT, debug=bool(args.dump))
        except Exception as e:  # noqa: BLE001 一条问失败不该让整批白跑，但原因要留着
            return {**h, "skipped": f"{type(e).__name__}: {str(e)[:120]}"}

    with ThreadPoolExecutor(max_workers=args.concurrency or config.SEEK_CONCURRENCY) as ex:
        results = list(ex.map(one, hits))
        rounds = [results]
        for _ in range(max(0, args.repeat - 1)):
            rounds.append(list(ex.map(one, hits)))
    print(f"  用时 {time.monotonic() - t0:.0f}s")
    if len(rounds) > 1:
        print(agreement(rounds))

    if args.dump:
        print("  " + dump_debug(results, args.dump))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for x in results:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  明细写到 {args.out}")
    print(summarize(results, llm))

    skipped = [x for x in results if x.get("skipped")]
    if skipped:
        # 原因逐条不同（路径各不相同），所以按前缀归类再报，不然刷屏
        uniq = list(dict.fromkeys(x["skipped"] for x in skipped))
        print(f"\n  {len(skipped)} 条没问成，原因（最多列 3 种）：\n    " + "\n    ".join(uniq[:3]))
        if len(uniq) > 3:
            print(f"    …… 还有 {len(uniq) - 3} 种")
    if args.sheet:
        good = [x for x in results if x.get("label")]
        if good:
            print("  " + posepart.contact_sheet(good, args.sheet))
        else:
            print("  一条都没命中，不拼图了。")


if __name__ == "__main__":
    main()
