"""姿态到底好不好用：已建的画面索引里，关键点真正测出来的比例。

为什么要单独量这个：想靠姿态补「舔后爪」这类部位数据集，前提是索引里**真的有**
能用的关键点。俯拍红外、狗蜷成一团，RTMPose 很可能大面积测不准——之前找相似就
出现过「这次没用上姿态」。

**别拿「有没有姿态向量」当指标。** RTMPose 是 top-down 的：给它一个狗框，它必定
吐出 17 个点，从不返回"测不到"，只是分数低。所以"有姿态的帧 / 有狗的帧"恒等于
100%，量的是"有没有狗框"，没有任何信息量（2026-09-20 踩过这个坑）。

真正有信息的是**关键点的可见位**：descriptor 里按 MIN_KP_SCORE 给每个点算了
可见/不可见，这 17 位就存在 pose 向量的最后 17 维（L2 归一化过，所以判 >0 而不是 ==1）。
这里直接读它。

而对"判部位"来说，门槛不是"有几个点可见"，是**鼻子和至少一只爪同时可见**——
这两个点都在，「鼻子离哪只爪最近」才算得出来。所以主指标是「能判部位的帧」。

它读索引文件里的 meta 和 pose 那一列（不碰几十 MB 的画面向量），一分钟内出结果。

用法（在算法机上）：

    python -m vision_service.pose_coverage              # 总览 + 按场地 + 按机位
    python -m vision_service.pose_coverage --by-day     # 再按天拆
    python -m vision_service.pose_coverage --worst 20   # 最差的 20 路

几个数的意思：

    有狗     检测到狗的帧数（没狗的帧不进索引，也不可能有姿态）
    可判部位 鼻子 + 至少一只爪同时可见 —— **这一列才是能不能靠姿态筛部位**
    鼻子     鼻子可见的帧数
    四爪全   四只爪子全可见（判左右前后最稳的那部分）
    均可见   平均每帧有几个点可见（满分 17）
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from . import config
from .pose import K, NOSE, PAWS

# data_raw/2026_9_14_gouchang/multicam_..._cam4_imu15_raw.mp4 → 场地 gouchang、机位 cam4
_SITE_RE = re.compile(r"/(\d{4}_\d{1,2}_\d{1,2})_([a-z]+)/", re.IGNORECASE)
_CAM_RE = re.compile(r"_cam(\d+)", re.IGNORECASE)

# 汇总时要加起来的计数列
_COUNTS = ("with_dog", "part_ok", "nose_ok", "paws4_ok", "vis_sum")


def _read(npz_path: str) -> dict | None:
    """读一份索引的 meta + 关键点可见位统计。向量那几十 MB 不碰。"""
    import numpy as np

    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "meta" not in z.files:
                return None
            meta = json.loads(str(z["meta"]))
            rows = z["pose"] if "pose" in z.files else None
            if rows is None or rows.size == 0:
                vis = np.zeros((0, K), dtype=bool)
            else:
                # 最后 K 维是可见位。存的时候整条向量做了 L2 归一化，所以是 0 或 1/‖v‖
                vis = np.asarray(rows[:, -K:], dtype="float32") > 0
    except Exception:  # noqa: BLE001 单个文件坏了不该让整份统计跑不出来
        return None

    nose = vis[:, NOSE] if len(vis) else vis[:, :0]
    paw_any = vis[:, PAWS].any(axis=1) if len(vis) else nose
    meta["_vis"] = {
        "rows": int(len(vis)),
        "part_ok": int((nose & paw_any).sum()),
        "nose_ok": int(nose.sum()),
        "paws4_ok": int(vis[:, PAWS].all(axis=1).sum()) if len(vis) else 0,
        "vis_sum": int(vis.sum()),
    }
    return meta


def scan(index_dir: str) -> list[dict]:
    """索引目录里每一路视频一行。"""
    rows = []
    for name in sorted(os.listdir(index_dir)) if os.path.isdir(index_dir) else []:
        if not name.endswith(".npz"):
            continue
        m = _read(os.path.join(index_dir, name))
        if m is None:
            continue
        path = str(m.get("path") or "")
        site = _SITE_RE.search("/" + path)
        cam = _CAM_RE.search(os.path.basename(path))
        v = m["_vis"]
        rows.append({
            "path": path,
            "day": site.group(1) if site else "?",
            "site": site.group(2).lower() if site else "?",
            "cam": f"cam{cam.group(1)}" if cam else "?",
            "sampled": int(m.get("sampled") or 0),
            # 分母用 pose 那一列的行数：它跟索引里的帧一一对应，跟 meta 里的 with_dog
            # 可能差一点（老索引、建到一半重来），以文件里真有的为准
            "with_dog": v["rows"] or int(m.get("with_dog") or 0),
            "part_ok": v["part_ok"],
            "nose_ok": v["nose_ok"],
            "paws4_ok": v["paws4_ok"],
            "vis_sum": v["vis_sum"],
            # 建这一路索引时姿态模型在不在。False = 那时候根本没跑姿态，
            # 不是「跑了但测不准」——这两种混在一起看，会把没装模型误判成模型不行
            "pose_on": bool(m.get("pose")),
        })
    return rows


def _pct(a: int, b: int) -> str:
    return f"{a / b * 100:5.1f}%" if b else "    —"


def _line(label: str, rows: list[dict], width: int = 22) -> str:
    s = {k: sum(r[k] for r in rows) for k in _COUNTS}
    dog = s["with_dog"]
    off = sum(1 for r in rows if not r["pose_on"])
    tail = f"   （{off}/{len(rows)} 路建索引时没开姿态）" if off else ""
    return (f"{label:<{width}} {len(rows):>4} 路  有狗 {dog:>9,}  "
            f"可判部位 {s['part_ok']:>9,} {_pct(s['part_ok'], dog)}  "
            f"鼻子 {_pct(s['nose_ok'], dog)}  四爪全 {_pct(s['paws4_ok'], dog)}  "
            f"均可见 {s['vis_sum'] / dog:4.1f}/{K}{tail}" if dog else
            f"{label:<{width}} {len(rows):>4} 路  有狗         0{tail}")


def report(rows: list[dict], by_day: bool = False, worst: int = 0) -> str:
    if not rows:
        return (f"索引目录里一个 .npz 都没有：{config.EMBED_INDEX_DIR}\n"
                "先在平台的项目页点「建画面索引」。")
    out = ["关键点可见率（分母 = 有狗的帧）。**可判部位 = 鼻子和至少一只爪同时可见**，",
           "靠姿态筛「舔哪只爪」能不能走，只看这一列。",
           "注：RTMPose 是 top-down 的，给框必出 17 个点——"
           "「有没有姿态向量」恒等于 100%，不是指标。", ""]
    out.append(_line("全部", rows))
    out.append("")

    by_site: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_site[r["site"]].append(r)
    out.append("── 按场地 ──")
    for site, rs in sorted(by_site.items(), key=lambda kv: -sum(x["with_dog"] for x in kv[1])):
        out.append(_line(site, rs))

    out.append("")
    out.append("── 按机位（公共区 cam7 跟单间机位的视角完全不同，分开看）──")
    by_cam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cam[f'{r["site"]} {r["cam"]}'].append(r)
    for cam, rs in sorted(by_cam.items()):
        out.append(_line(cam, rs, width=22))

    if by_day:
        out.append("")
        out.append("── 按天 ──")
        by_d: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_d[f'{r["day"]} {r["site"]}'].append(r)
        for d, rs in sorted(by_d.items()):
            out.append(_line(d, rs, width=24))

    if worst:
        out.append("")
        out.append(f"── 有狗但最判不出部位的 {worst} 路（抽一帧画骨架看看是为什么）──")
        cand = [r for r in rows if r["with_dog"] >= 60 and r["pose_on"]]
        cand.sort(key=lambda r: r["part_ok"] / r["with_dog"])
        for r in cand[:worst]:
            out.append(f"  可判部位 {_pct(r['part_ok'], r['with_dog'])}  "
                       f"鼻子 {_pct(r['nose_ok'], r['with_dog'])}  "
                       f"有狗 {r['with_dog']:>6,}  {os.path.basename(r['path'])}")

    dog = sum(r["with_dog"] for r in rows)
    part = sum(r["part_ok"] for r in rows)
    cov = part / dog if dog else 0.0
    out.append("")
    out.append("── 这个数意味着什么 ──")
    if any(not r["pose_on"] for r in rows):
        out.append("  有索引是在没开姿态的时候建的，那几路的 0 不代表测不准。"
                   "要看真实可见率，先确认姿态模型已加载，再把那几路重建一次。")
    if cov >= 0.6:
        out.append(f"  {cov * 100:.0f}% 的有狗帧能判部位：够用。"
                   "按姿态条件筛部位（鼻子离哪只爪最近）这条路可以走。")
    elif cov >= 0.3:
        out.append(f"  {cov * 100:.0f}% 的有狗帧能判部位：能用但要挑。"
                   "姿态检索只在这部分有效，召回会偏低，当补充手段而不是主力；"
                   "剩下的那些还得靠人看画面。")
    else:
        out.append(f"  {cov * 100:.0f}% 的有狗帧能判部位：偏低。"
                   "先看上面「最判不出部位的」那几路，抽一帧把骨架画出来——"
                   "是俯拍看不见鼻子、还是狗蜷成一团爪子被挡住，"
                   "这两种的解法完全不同。别急着在它上面搭部位检索。")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="已建索引里关键点的可见率")
    ap.add_argument("--index-dir", default=config.EMBED_INDEX_DIR)
    ap.add_argument("--by-day", action="store_true", help="再按天拆一份")
    ap.add_argument("--worst", type=int, default=0, help="列出最判不出部位的几路")
    args = ap.parse_args()
    print(report(scan(args.index_dir), by_day=args.by_day, worst=args.worst))


if __name__ == "__main__":
    main()
