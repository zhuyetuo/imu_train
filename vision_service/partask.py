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


def frames_around(full_path: str, rel_path: str, t: float, n: int = 4,
                  span_s: float = 3.0, max_side: int = 384) -> list[bytes]:
    """候选时刻前后几秒的几帧，按索引里存的框裁好。

    **不重跑检测**：框就在索引里（建索引时那一次检测的结果），重跑一次既慢又可能
    跟当初判断用的框不一样。拿不到索引就返回空，调用方跳过这条——宁可少问一条，
    也别拿一张跟判断依据不一致的图去问。
    """
    import cv2
    import numpy as np

    d = embed.load(rel_path)
    if d is None or not len(d["t"]):
        return []
    ts = np.asarray(d["t"], dtype="float32")
    sel = np.flatnonzero((ts >= t - span_s) & (ts <= t + span_s))
    if not len(sel):
        return []
    if len(sel) > n:                       # 均匀取 n 个，保证跨过整个时间窗
        sel = sel[np.linspace(0, len(sel) - 1, n).astype(int)]
    want = {round(float(ts[i]), 2): np.asarray(d["box"][i], dtype="float32") for i in sel}
    lo, hi = min(want), max(want)
    out: list[bytes] = []
    for ft, frame in seek.iter_frames(full_path, 1.0, start_s=max(0.0, lo - 0.5), end_s=hi + 0.5):
        key = min(want, key=lambda x: abs(x - ft)) if want else None
        if key is None or abs(key - ft) > 0.6:
            continue
        box = want.pop(key)
        h, w = frame.shape[:2]
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
        if not want:
            break
    return out


def labels_for(part_key: str, names) -> list[seek.Label]:
    """问的时候带上部位选项：只给这个部位和它的左右两边，不给全表。

    为什么不给全表：候选本来就是"鼻子够到后爪"筛出来的，再让模型从 37 个部位里选
    等于把粗筛的信息扔了，而且选项越多越容易乱选。左右两个也给，是因为四爪全可见
    只有 58%，几何判的左右本来就不牢——让模型自己看。
    """
    slots = posepart.PARTS.get(part_key) or ()
    parts = [posepart.SLOT_NAMES[i] for i in slots] or [part_key]
    if len(parts) == 1 and parts[0] in posepart.SLOT_NAMES:
        parts = [parts[0]]
    return [seek.Label(name=n, description=LABEL_DESC.get(n, ""), parts=list(parts)) for n in names]


def ask_one(hit: dict, part_key: str, labels: list[seek.Label], llm, *,
            n_frames: int, span_s: float, video_root: str, client=None, http=None) -> dict:
    """一条候选 → 问一次。返回候选本身 + 模型的回答；拿不到帧就 skipped。"""
    full = os.path.join(video_root, hit["path"])
    if not os.path.isfile(full):
        return {**hit, "skipped": f"视频不在：{full}"}
    frames = frames_around(full, hit["path"], hit["t"], n=n_frames, span_s=span_s)
    if not frames:
        return {**hit, "skipped": "取不到帧（索引里没有这一段？）"}
    a = seek.ask(frames, labels, span_s * 2, llm, client=client, http=http)
    return {**hit, "n_frames": len(frames), **{k: v for k, v in a.items() if k != "usage"},
            "usage": a.get("usage") or {}}


def summarize(results: list[dict], llm=None) -> str:
    """跑完之后看什么：unclear 说的是候选好不好，命中说的是这条路值不值。"""
    n = len(results)
    ok = [r for r in results if not r.get("skipped")]
    unclear = [r for r in ok if r.get("see") == "unclear"]
    hits = [r for r in ok if r.get("label")]
    tin = sum(int((r.get("usage") or {}).get("input") or 0) for r in ok)
    tout = sum(int((r.get("usage") or {}).get("output") or 0) for r in ok)
    lines = [
        "",
        f"问了 {len(ok)}/{n} 条（{n - len(ok)} 条取不到帧）",
        f"  看不清 {len(unclear)}（{len(unclear) / max(len(ok), 1) * 100:.0f}%）"
        "  ← 这个数说的**不是模型好不好，是送进去的候选好不好**。"
        "高了就回头修裁图 / 夜间亮度，不是调提示词",
        f"  命中   {len(hits)}（{len(hits) / max(len(ok), 1) * 100:.0f}%）"
        "  ← 几何粗筛的真实精度。哪怕只有 10%，那也是几十条带部位的样本",
    ]
    if llm is not None:
        lines.append(f"  花费   约 ${llmmod.estimate_usd(llm, tin, tout):.2f}"
                     f"（in {tin:,} / out {tout:,} token）")
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


def main() -> None:
    ap = argparse.ArgumentParser(description="把姿态捞出来的部位候选送去问大模型")
    ap.add_argument("--part", required=True, help="部位名，比如 后爪")
    ap.add_argument("--index-dir", default=config.EMBED_INDEX_DIR)
    ap.add_argument("--near-max", type=float, default=0.25)
    ap.add_argument("--min-run", type=int, default=5)
    ap.add_argument("--min-gap", type=float, default=60.0)
    ap.add_argument("--per-video", type=int, default=20)
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="问哪几个类别，逗号分隔")
    ap.add_argument("--frames", type=int, default=4, help="每条候选送几帧（拼成一张图）")
    ap.add_argument("--span", type=float, default=3.0, help="取候选时刻前后多少秒")
    ap.add_argument("--limit", type=int, default=0, help="最多问几条（0 = 全部）。先小样本试一下再放开")
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--out", help="每条一行 JSON 写到这里，之后能反复分析不用重问")
    ap.add_argument("--sheet", metavar="PNG", help="把命中的拼成一张带骨架的图")
    ap.add_argument("--dry-run", action="store_true", help="只报会问多少条、大概多少钱，不调 API")
    args = ap.parse_args()

    key = posepart.part_of(args.part) or args.part
    r = posepart.find(args.index_dir, key, args.near_max, True, args.per_video,
                      args.min_gap, args.min_run)
    if not r["known"]:
        print(f"不认识的部位：{args.part} → {key}（认得的：{'、'.join(posepart.PARTS)}）")
        return
    hits = r["hits"][:args.limit] if args.limit else r["hits"]
    print(f"{args.part} → {key}：几何粗筛出 {len(r['hits']):,} 条，这次问 {len(hits):,} 条"
          f"（近到 {args.near_max} 体长、连着 {args.min_run} 秒）")

    llm = llmmod.from_env()
    if args.dry_run or llm is None:
        # 一张 4 格拼图约 1.5k token（384px 的格子），输出约 120 token；数量级用的，不当账
        est_in, est_out = len(hits) * 1500, len(hits) * 120
        # 没 key 也要把钱估出来：「值不值得花这个钱」这个决定，缺了数字根本没法做。
        # 按 config.SEEK_PRICE_PER_M 的价目表算，顺便把几个档位都列出来好挑
        print(f"  预估 in {est_in:,} / out {est_out:,} token")
        for m, (pin, pout) in config.SEEK_PRICE_PER_M.items():
            cost = est_in / 1e6 * pin + est_out / 1e6 * pout
            cur = "  ← 当前" if m == config.SEEK_MODEL else ""
            print(f"    {m:<20} 约 ${cost:.2f}{cur}")
        if llm is None:
            print("\n  没配 key，只能 dry-run。写进 vision_service/.env 再跑：")
            print("    echo 'ANTHROPIC_API_KEY=sk-ant-...' >> vision_service/.env")
            print("  换模型：再加一行 SEEK_MODEL=claude-haiku-4-5（便宜五倍，先拿它探路也行）")
        else:
            print("\n  确认了去掉 --dry-run 再跑。建议先 --limit 20 看看准不准，再放开。")
        return

    labels = labels_for(key, [x.strip() for x in args.labels.split(",") if x.strip()])
    print(f"  问 {llm.label()}，类别 {'/'.join(l.name for l in labels)}，"
          f"部位选项 {'/'.join(labels[0].parts)}")
    t0 = time.monotonic()
    from concurrent.futures import ThreadPoolExecutor

    def one(h):
        try:
            return ask_one(h, key, labels, llm, n_frames=args.frames, span_s=args.span,
                           video_root=config.VIDEO_ROOT)
        except Exception as e:  # noqa: BLE001 一条问失败不该让整批白跑，但原因要留着
            return {**h, "skipped": f"{type(e).__name__}: {str(e)[:120]}"}

    with ThreadPoolExecutor(max_workers=args.concurrency or config.SEEK_CONCURRENCY) as ex:
        results = list(ex.map(one, hits))
    print(f"  用时 {time.monotonic() - t0:.0f}s")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for x in results:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  明细写到 {args.out}")
    print(summarize(results, llm))

    skipped = [x for x in results if x.get("skipped")]
    if skipped:
        uniq = list(dict.fromkeys(x["skipped"] for x in skipped))
        print(f"\n  {len(skipped)} 条没问成，原因：\n    " + "\n    ".join(uniq[:3]))
    if args.sheet:
        good = [x for x in results if x.get("label")]
        if good:
            print("  " + posepart.contact_sheet(good, args.sheet))
        else:
            print("  一条都没命中，不拼图了。")


if __name__ == "__main__":
    main()
