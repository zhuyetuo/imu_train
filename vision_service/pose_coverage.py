"""姿态关键点的覆盖率：已建的画面索引里，有多少帧真的测到了狗的关键点。

为什么要单独量这个：想靠姿态补「舔后爪」这类部位数据集，前提是索引里**真的有**
姿态。俯拍红外、狗蜷成一团，RTMPose 很可能大面积测不到——之前找相似就出现过
「这次没用上姿态」。覆盖率太低的话，那条路走不通，得先把夜间俯拍的检测/姿态
调好，否则后面全是空转。

它只读索引文件里的 meta（每路视频一行几百字节），不解码视频、不跑模型，几秒钟出结果。

用法（在算法机上）：

    python -m vision_service.pose_coverage              # 总览 + 按场地 + 按机位
    python -m vision_service.pose_coverage --by-day     # 再按天拆
    python -m vision_service.pose_coverage --worst 20   # 覆盖率最低的 20 路

三个数的意思：

    采样   索引每秒一帧，这一路一共采了多少帧
    有狗   其中检测到狗的帧数（没狗的帧不进索引，也不可能有姿态）
    有姿态 有狗的帧里，RTMPose 真的测出关键点的帧数

**覆盖率 = 有姿态 / 有狗**，不是除以采样数：没狗的帧本来就不该有姿态，
拿它当分母会把覆盖率压得虚低，看起来像姿态模型不行，其实是那段没狗。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from . import config

# data_raw/2026_9_14_gouchang/multicam_..._cam4_imu15_raw.mp4 → 场地 gouchang、机位 cam4
_SITE_RE = re.compile(r"/(\d{4}_\d{1,2}_\d{1,2})_([a-z]+)/", re.IGNORECASE)
_CAM_RE = re.compile(r"_cam(\d+)", re.IGNORECASE)


def _meta_of(npz_path: str) -> dict | None:
    """只读索引里的 meta，不碰向量那几十 MB。"""
    import numpy as np

    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "meta" not in z.files:
                return None
            return json.loads(str(z["meta"]))
    except Exception:  # noqa: BLE001 单个文件坏了不该让整份统计跑不出来
        return None


def scan(index_dir: str) -> list[dict]:
    """索引目录里每一路视频一行。"""
    rows = []
    for name in sorted(os.listdir(index_dir)) if os.path.isdir(index_dir) else []:
        if not name.endswith(".npz"):
            continue
        m = _meta_of(os.path.join(index_dir, name))
        if m is None:
            continue
        path = str(m.get("path") or "")
        site = _SITE_RE.search("/" + path)
        cam = _CAM_RE.search(os.path.basename(path))
        rows.append({
            "path": path,
            "day": site.group(1) if site else "?",
            "site": site.group(2).lower() if site else "?",
            "cam": f"cam{cam.group(1)}" if cam else "?",
            "sampled": int(m.get("sampled") or 0),
            "with_dog": int(m.get("with_dog") or 0),
            "with_pose": int(m.get("with_pose") or 0),
            # 建这一路索引时姿态模型在不在。False = 那时候根本没跑姿态，
            # 不是「跑了但测不到」——这两种混在一起看，会把没装模型误判成模型不行
            "pose_on": bool(m.get("pose")),
        })
    return rows


def _pct(a: int, b: int) -> str:
    return f"{a / b * 100:5.1f}%" if b else "    —"


def _line(label: str, rows: list[dict], width: int = 22) -> str:
    sampled = sum(r["sampled"] for r in rows)
    dog = sum(r["with_dog"] for r in rows)
    pose = sum(r["with_pose"] for r in rows)
    off = sum(1 for r in rows if not r["pose_on"])
    tail = f"   （{off}/{len(rows)} 路建索引时没开姿态）" if off else ""
    return (f"{label:<{width}} {len(rows):>4} 路  采样 {sampled:>9,}  "
            f"有狗 {dog:>9,} {_pct(dog, sampled)}  有姿态 {pose:>9,} {_pct(pose, dog)}{tail}")


def report(rows: list[dict], by_day: bool = False, worst: int = 0) -> str:
    if not rows:
        return (f"索引目录里一个 .npz 都没有：{config.EMBED_INDEX_DIR}\n"
                "先在平台的项目页点「建画面索引」。")
    out = ["姿态覆盖率 = 有姿态 / 有狗（没狗的帧本来就不该有姿态，不拿它当分母）", ""]
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
        out.append(f"── 有狗但最测不到姿态的 {worst} 路 ──")
        cand = [r for r in rows if r["with_dog"] >= 60 and r["pose_on"]]
        cand.sort(key=lambda r: r["with_pose"] / r["with_dog"])
        for r in cand[:worst]:
            out.append(f"  {_pct(r['with_pose'], r['with_dog'])}  "
                       f"有狗 {r['with_dog']:>6,}  {os.path.basename(r['path'])}")

    dog = sum(r["with_dog"] for r in rows)
    pose = sum(r["with_pose"] for r in rows)
    cov = pose / dog if dog else 0.0
    out.append("")
    out.append("── 这个数意味着什么 ──")
    if any(not r["pose_on"] for r in rows):
        out.append("  有索引是在没开姿态的时候建的，那几路的 0 不代表测不到。"
                   "要看真实覆盖率，先确认姿态模型已加载，再把那几路重建一次。")
    if cov >= 0.6:
        out.append(f"  {cov * 100:.0f}%：够用。按姿态条件筛部位（鼻子离哪只爪最近）这条路可以走。")
    elif cov >= 0.3:
        out.append(f"  {cov * 100:.0f}%：能用但要挑。姿态检索只在测得到的那部分有效，"
                   "召回会偏低，当补充手段而不是主力。")
    else:
        out.append(f"  {cov * 100:.0f}%：偏低。先查是模型没加载、还是俯拍红外上真的测不到"
                   "（看上面「最测不到姿态的」那几路，抽一帧画骨架看看），"
                   "别急着在它上面搭部位检索。")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="已建索引里姿态关键点的覆盖率")
    ap.add_argument("--index-dir", default=config.EMBED_INDEX_DIR)
    ap.add_argument("--by-day", action="store_true", help="再按天拆一份")
    ap.add_argument("--worst", type=int, default=0, help="列出覆盖率最低的几路")
    args = ap.parse_args()
    print(report(scan(args.index_dir), by_day=args.by_day, worst=args.worst))


if __name__ == "__main__":
    main()
