"""从已建的索引里分层抽一批帧，给人标关键点——训自己的狗姿态模型用。

## 为什么要训自己的

2026-09-20 用真值量过（见 README「真值实测」）：现成的 RTMPose(AP-10K) 在我们的
画面上判部位 **1/6**，而且十一张里一张都没判对过后爪。原因是 AP-10K 那一万多张
基本是**野外侧面平视的清醒动物**，而我们的画面是俯拍 + 夜间红外 + 狗蜷成一团 +
目标只占画幅 0.5%——四条全在分布外，叠起来就垮了。

换别的开源权重大概率也一样：能选的那几家（ViTPose+ / HRNet / Animal Pose）
**训练数据跟 AP-10K 高度重合**。换架构只在"同分布上更准"这个维度提升，
而我们缺的是"分布外不要乱给高分"。

所以真要提升，得拿**我们自己场地的帧**微调。这个模块负责那一步的前半截：
把该标的帧挑出来。

## 只标 6 个点，不是 17 个

看过第一批 47 张之后砍的（2026-09-20）。肩 / 肘 / 髋 / 膝那 8 个点**在这批画面上
根本看不见**：狗场主力是卷毛，关节全埋在毛里；眼睛和脖子同理（俯拍 + 毛）。
标注员只能猜——而猜出来的标签只会教模型也猜。

而且**判部位本来就只用到 6 个**（posepart 算的是鼻子到四爪 / 尾根的距离）。
17 个点里有 8 个既看不见、又用不上。露在外面的爪尖和鼻子标得准，那几个才值得标。

## 但先把话说清楚：这件事的天花板

同一份真值里还有一个数：**45% 的帧人也判不了**（夜里低对比、裁得只剩半只狗、
多狗同框）。姿态模型再好也救不了那 45%——那是画面本身的上限。所以这件事最多
改善另外那 55%，**别指望它把"部位自动化"这条路救回来**。

## 怎么抽

随手抽一批的话，会抽到一大堆白天狗场 cam2 里同一只狗趴着的帧——那种模型本来就
会，标了也学不到东西。按四个维度分层，每一格均匀取：

    场地      狗场 / 影棚（视角和房间完全不同）
    机位      cam1..cam7（吊装高度和角度各不相同）
    昼夜      按小时分，夜里是红外灰度，白天是彩色
    姿势      用索引里的姿态向量粗分：摊开 / 蜷着（蜷着正是现在最容易崩的）

再加一条：**优先挑现在判得最不确定的帧**（可见关键点少的），那些才是模型要学的。
全挑最好认的等于白标。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from . import config, posepart
from .pose import DIM, K

_SITE_RE = re.compile(r"/(\d{4}_\d{1,2}_\d{1,2})_([a-z]+)/", re.IGNORECASE)
_CAM_RE = re.compile(r"_cam(\d+)", re.IGNORECASE)
# 文件名里的起始时刻：multicam_20260918_030016038_... → 03 点
_HOUR_RE = re.compile(r"_\d{8}_(\d{2})\d{7}")


def box_usable(box) -> bool:
    """这个框能不能拿来判姿势。2026-09-20 之前建的索引里 box 列全是 (0,0,0,0)。"""
    return bool(box) and len(box) == 4 and float(box[2]) > float(box[0]) \
        and float(box[3]) > float(box[1])


def curled_from_box(box, frame_ar: float = 16 / 9) -> bool:
    """蜷着还是摊开：看**检测框的长宽比**，不看姿态关键点。

    原来是拿 `spread < 本路中位数` 判的——而 spread 就是从 RTMPose 的关键点算出来的。
    骨架崩掉的时候 spread 也是噪声，**等于拿要评估的那个模型给数据分层**，
    分出来的"姿势"是假的。2026-09-20 实测印证：按那个分法，摊开和蜷着两组
    RTMPose 测到的点数完全没差（中位数 6.5 vs 9.0，≤5 个点的都是 38%）——
    因为那根本不是姿势，是"这一帧的骨架比本路平均更散还是更挤"。

    框来自狗检测，跟姿态模型无关：狗蜷成一团时框接近正方形，摊开躺着时是长条。
    框存的是相对整帧的归一化值，所以比长宽比之前要把画幅的宽高比乘回去。
    """
    if not box_usable(box):
        return False
    w = (float(box[2]) - float(box[0])) * frame_ar
    h = float(box[3]) - float(box[1])
    return max(w, h) / min(w, h) < 1.6          # 接近方的算蜷着；细长的算摊开


def _bucket(path: str, curled: bool, box_ok: bool = True) -> tuple:
    """一帧属于哪一格。四个维度：场地 / 机位 / 昼夜 / 姿势。"""
    m = _SITE_RE.search("/" + path)
    cam = _CAM_RE.search(os.path.basename(path))
    hh = _HOUR_RE.search(os.path.basename(path))
    hour = int(hh.group(1)) if hh else 12
    return (m.group(2).lower() if m else "?",
            f"cam{cam.group(1)}" if cam else "?",
            "夜" if (hour >= 19 or hour < 7) else "昼",
            ("蜷着" if curled else "摊开") if box_ok else "姿势未知")


def scan(index_dir: str, max_per_video: int = 40) -> list[dict]:
    """索引里每一帧一条候选，带上分层用的信息和"现在判得有多不确定"。"""
    import numpy as np

    out: list[dict] = []
    for name in sorted(os.listdir(index_dir)) if os.path.isdir(index_dir) else []:
        if not name.endswith(".npz"):
            continue
        try:
            with np.load(os.path.join(index_dir, name), allow_pickle=False) as z:
                if "meta" not in z.files or "pose" not in z.files:
                    continue
                meta = json.loads(str(z["meta"]))
                rows, ts = z["pose"], z["t"]
                box = z["box"] if "box" in z.files else None
        except Exception:  # noqa: BLE001 一个坏文件不该让整批抽不出来
            continue
        if rows.size == 0 or len(rows) != len(ts):
            continue
        path = str(meta.get("path") or "")
        vis = np.asarray(rows, dtype="float32")[:, DIM - K:] > 0
        n_vis = vis.sum(axis=1)
        spread = posepart.spread(rows)
        idx = np.argsort(n_vis)[:max_per_video]          # 可见点最少的优先：那才是要学的
        for i in idx:
            b = [round(float(v), 4) for v in box[i]] if box is not None else None
            out.append({"path": path, "t": round(float(ts[i]), 2),
                        "n_vis": int(n_vis[i]), "spread": round(float(spread[i]), 3),
                        "box": b,
                        # 姿势按检测框的长宽比分，不按姿态关键点——见 curled_from_box。
                        # 框不可用时记成"姿势未知"而不是默默算作摊开：默默算的话，
                        # 整个维度会悄悄塌成一格，而输出看起来完全正常
                        "box_ok": box_usable(b),
                        "bucket": _bucket(path, curled_from_box(b), box_usable(b))})
    return out


def pick(cands: list[dict], n: int, seed: int = 0) -> list[dict]:
    """分层轮取：一格一条地轮着拿，直到取够 n 条。

    不用"每格固定取几条"：格子数是数据决定的（有的机位只有几路），固定配额会在
    小格子上取空、在大格子上砍掉。轮取天然把量摊到有数据的格子上。
    """
    import random

    rng = random.Random(seed)
    by: dict[tuple, list[dict]] = defaultdict(list)
    for c in cands:
        by[c["bucket"]].append(c)
    for v in by.values():
        rng.shuffle(v)
        # 同一路视频里挨着的帧长得一模一样，标了也是重复：一路最多留几条
        seen: dict[str, int] = defaultdict(int)
        keep = []
        for c in v:
            if seen[c["path"]] < 3:
                seen[c["path"]] += 1
                keep.append(c)
        v[:] = keep
    order = sorted(by)
    out: list[dict] = []
    while len(out) < n and any(by[k] for k in order):
        for k in order:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def export(picks: list[dict], out_dir: str, video_root: str, *, max_side: int = 640) -> str:
    """把选中的帧裁出来存成图 + 一份 manifest。

    **裁的时候留一大圈边（0.6 倍框长）**：标关键点要看得见四肢伸出去的样子，
    紧贴框裁会把爪子和尾巴切掉——而那几个点正是最难标也最要紧的。
    小目标放大到 max_side，标的时候点得准一些。
    """
    import csv

    import cv2
    import numpy as np

    from . import embed, seek

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, c in enumerate(picks):
        full = os.path.join(video_root, c["path"])
        if not os.path.isfile(full):
            continue
        frame = None
        for _t, f in seek.iter_frames(full, 1.0, start_s=max(0.0, c["t"] - 0.5), end_s=c["t"] + 0.5):
            frame = f
            break
        if frame is None:
            continue
        h, w = frame.shape[:2]
        # **按单只狗裁，不用索引里那个并集框。** 索引存的是所有狗框的并集（norm_box），
        # 影棚一帧里常有两三只狗，并集裁出来是"一屋子狗"——标关键点时根本不知道该标哪只。
        # 这里当场检测一次，取面积最大的那只；顺便记下这一帧有几只狗，多的在 manifest
        # 里标出来让标注员跳过
        from . import dog
        boxes = dog.detect(frame)
        if not boxes:
            continue
        big = max(boxes, key=lambda b_: b_["bbox"][2] * b_["bbox"][3])
        bx, by, bw_, bh_ = big["bbox"]
        x1, y1, x2, y2 = bx * w, by * h, (bx + bw_) * w, (by + bh_) * h
        n_dogs = len(boxes)
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        x1, y1 = max(0, int(x1 - bw * 0.6)), max(0, int(y1 - bh * 0.6))
        x2, y2 = min(w, int(x2 + bw * 0.6)), min(h, int(y2 + bh * 0.6))
        crop = frame[y1:y2, x1:x2]
        if not crop.size:
            continue
        sc = max_side / max(crop.shape[:2])
        if sc > 1:                       # 只放大不缩小：标点要看得清
            crop = cv2.resize(crop, (int(crop.shape[1] * sc), int(crop.shape[0] * sc)),
                              interpolation=cv2.INTER_CUBIC)
        name = f"{len(rows):04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), crop)
        rows.append({"图": name, "场地": c["bucket"][0], "机位": c["bucket"][1],
                     "昼夜": c["bucket"][2], "姿势": c["bucket"][3],
                     "画面里几只狗": n_dogs, "现在测到几个点": c["n_vis"],
                     "视频": c["path"], "秒": c["t"]})
    with open(os.path.join(out_dir, "manifest.csv"), "w", encoding="utf-8-sig", newline="") as f:
        if rows:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    by = defaultdict(int)
    for r in rows:
        by[(r["场地"], r["昼夜"], r["姿势"])] += 1
    lines = [f"导了 {len(rows)} 张到 {out_dir}/", "", "  分层（场地/昼夜/姿势）："]
    for k, v in sorted(by.items()):
        lines.append(f"    {'/'.join(k):<20} {v}")
    multi = sum(1 for r in rows if int(r["画面里几只狗"]) > 1)
    lines += [
        "",
        "  **只标 6 个点，不是 17 个**（顺序跟 pose.py 的下标一致）：",
        "     2 鼻子   4 尾根",
        "     7 左前爪  10 右前爪  13 左后爪  16 右后爪",
        "",
        "  为什么砍掉另外 11 个：肩/肘/髋/膝那 8 个**在这批画面上根本看不见**——"
        "狗场那几只是卷毛，关节全埋在毛里；标注员只能猜，而猜出来的标签只会教模型也猜。"
        "眼睛和脖子同理（俯拍 + 毛）。而爪尖和鼻子是露在外面的，标得准。",
        "  更重要的是：**判部位本来就只用到这 6 个**（posepart 算的是鼻子到四爪/尾根的"
        "距离）。标 17 个点里有 8 个既看不见、又用不上。",
    ]
    if multi:
        lines.append(f"  ⚠ 其中 {multi} 张画面里不止一只狗（manifest 的「画面里几只狗」那一列）。"
                     "裁的时候取的是面积最大的那只，但标注时容易标串——**建议直接跳过**。")
    lines += [
        "",
        "  **先标这 50 张就停下来评一次**：拿它们当验证集，量一下现在的 RTMPose 到底"
        "错在哪（是前后爪反了，还是整副骨架乱）。错法不同，要不要继续标、标多少，答案不同。",
        "  另外记着：同一份真值里 45% 的帧人也判不了，姿态模型救不了那一半——"
        "这件事的天花板在那儿。",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="分层抽一批帧，给人标关键点（训自己的狗姿态模型）")
    ap.add_argument("--index-dir", default=config.EMBED_INDEX_DIR)
    ap.add_argument("--out", required=True, metavar="DIR")
    ap.add_argument("-n", type=int, default=300, help="抽多少张（建议先 50 张评一次）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    cands = scan(args.index_dir)
    n_ok = sum(1 for c in cands if c.get("box_ok"))
    print(f"索引里 {len(cands):,} 条可选（每路取可见点最少的几十条），"
          f"分 {len({c['bucket'] for c in cands})} 格")
    if cands and n_ok < len(cands) * 0.5:
        # 默默降级最要命：整个维度塌成一格，而输出看起来完全正常。
        # 2026-09-20 实测撞上——47 张全被分成"摊开"，一张"蜷着"都没有
        print(f"  ⚠ {len(cands) - n_ok:,}/{len(cands):,} 条的框是空的（2026-09-20 之前建的索引，"
              f"box 列全是 0，见 embed.norm_box）。**这些条判不出姿势**，"
              f"分层里记成「姿势未知」——不是「它们都摊开着」。")
        print(f"    想要姿势这一维，得把索引重建一遍（平台项目页「建画面索引」勾上强制重建）。"
              f"不重建也能标，只是抽样在姿势上没分层。")
    picks = pick(cands, args.n, args.seed)
    print(export(picks, args.out, config.VIDEO_ROOT))


if __name__ == "__main__":
    main()
