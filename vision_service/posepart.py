"""靠姿态几何筛部位：「鼻子离哪只爪最近」——不训练、不标注、不跑新模型。

为什么这条路成立：产品那张 22 类表里，「舔-后右爪」「啃-前爪」这类标签，在几何上
就是**鼻子够到了哪个部位**。而索引里每帧存的姿态向量**本来就有**鼻子到四爪 /
尾根 / 脖子的六个距离（见 pose.descriptor），已经按体长归一化过。也就是说：
新标签来了没数据集时，不用等模型、不用先标一批，直接从已建的索引里把候选捞出来。

它解决的是「有新标签，但一帧样本都没有」那个死结的**部位这一级**。时段这一级由
label_service/grooming.py 从 IMU 出候选（规则，不靠画面），两边各干各擅长的：
IMU 从 24 小时里挑出"这几分钟在理毛"，姿态回答"理的是哪个部位"。

2026-09-20 在 444 路真实索引上量过：84.3% 的有狗帧鼻子和至少一只爪同时可见
（狗场 87.3%，影棚 74.0%），四爪全可见 58.1%。所以判「前爪 / 后爪」够用，判
「左后 / 右后」会掉一截——见 pose_coverage。

## 存的时候归一化过，这里要还原

pose 向量是整条 L2 归一化存的，所以那六个距离**在行与行之间不可比**，直接拿去卡
阈值是错的。但可见位在归一化前恒等于 1，所以任何一个非零可见位就等于 1/‖v‖，
拿它就能精确还原尺度。argmin（谁最近）不受影响，绝对阈值必须还原。
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

from . import config
from .pose import DIM, K

# 距离块在向量里的位置：pose.descriptor 里是 [四爪, 尾根, 脖子]，紧跟在 34 维坐标后面
D0 = K * 2
N_DIST = 6
SLOT_NAMES = ("前左爪", "前右爪", "后左爪", "后右爪", "尾根", "颈部")

# 部位名 → 认哪几个距离槽。粗的（前爪）认两个，细的（前左爪）认一个
PARTS: dict[str, tuple[int, ...]] = {
    "前左爪": (0,), "前右爪": (1,), "后左爪": (2,), "后右爪": (3,),
    "前爪": (0, 1), "后爪": (2, 3), "爪": (0, 1, 2, 3),
    "尾根": (4,), "颈部": (5,),
}

# 鼻子离得多近才算"够到了"，单位是体长（脖子到尾根）。
# 0.6 是个起点不是结论：跑 `python -m vision_service.posepart --part 后爪 --calib`
# 看真实分布再定，每个场地/机位可能不一样
NEAR_MAX = float(os.environ.get("POSE_PART_NEAR_MAX", "0.6"))


def part_of(name: str) -> str | None:
    """标签里的部位名 → PARTS 的键。认不出来返回 None（**不过滤**，而不是滤成空）。

    平台那张表用的是「前左爪 / 后右爪 / 前爪 / 后爪 / 尾根/尾 / 颈侧/颈下」这类写法，
    左右和前后的顺序两种都有（「前左爪」「左前爪」），所以按关键字认不按全等认。
    认不出来时宁可不筛：滤成空会让人以为"索引里没有这种数据"，那是最坏的误导。
    """
    s = (name or "").strip()
    if not s:
        return None
    if "尾" in s:
        return "尾根"
    if "颈" in s or "喉" in s:
        return "颈部"
    if "爪" not in s and "肢" not in s and "趾" not in s:
        return None             # 腰、腹股沟、耳后……鼻子到爪的距离说明不了它们
    front = "前" in s
    hind = "后" in s
    left, right = "左" in s, "右" in s
    if front == hind:           # 两个都写了或都没写：不分前后
        return "爪"
    side = "左" if left and not right else "右" if right and not left else None
    if side is None:
        return "前爪" if front else "后爪"
    return f"{'前' if front else '后'}{side}爪"


def decode(rows):
    """存的姿态行 → (dists (N,6) 真实体长倍数, vis (N,17) bool)。

    整条向量 L2 归一化过，所以距离要乘回 ‖v‖ 才能跨行比。可见位归一化前恒等于 1，
    所以任何一个非零可见位就是 1/‖v‖——拿它精确还原，不用另存一个尺度。
    """
    import numpy as np

    r = np.asarray(rows, dtype="float32").reshape(-1, DIM)
    vis_raw = r[:, DIM - K:]
    mx = vis_raw.max(axis=1)                       # 非零可见位都等于 1/‖v‖，取最大的那个
    scale = np.where(mx > 0, 1.0 / np.maximum(mx, 1e-9), 0.0)
    dists = r[:, D0:D0 + N_DIST] * scale[:, None]
    return dists, vis_raw > 0


def nearest(dists):
    """每帧鼻子最近的那个槽 → (slot (N,) int，没有可用的记 -1; dist (N,) float)。

    距离 0 表示两头有一头没测到（descriptor 里就是这么记的），不能当"贴着"，
    要当"不知道"排除掉——否则测不到的帧会全部冒充成最近的那个。
    """
    import numpy as np

    d = np.asarray(dists, dtype="float32")
    ok = d > 0
    big = np.where(ok, d, np.inf)
    slot = np.argmin(big, axis=1)
    best = big[np.arange(len(d)), slot]
    return np.where(np.isfinite(best), slot, -1), np.where(np.isfinite(best), best, 0.0)


def match(rows, part: str, near_max: float | None = None, require_nearest: bool = True):
    """哪些帧的鼻子够到了这个部位 → bool mask (N,)。

    require_nearest：不光要够近，还得是**所有部位里最近的那个**。不加这条的话，
    狗蜷成一团时鼻子离四个爪都在阈值内，一帧会同时算进四个部位，等于没筛。
    """
    import numpy as np

    slots = PARTS.get(part)
    r = np.asarray(rows, dtype="float32").reshape(-1, DIM)
    if not slots or not len(r):
        return np.zeros(len(r), dtype=bool)
    thr = NEAR_MAX if near_max is None else float(near_max)
    dists, _vis = decode(r)
    want = np.zeros(len(r), dtype=bool)
    for s in slots:
        want |= (dists[:, s] > 0) & (dists[:, s] <= thr)
    if require_nearest:
        slot, _best = nearest(dists)
        want &= np.isin(slot, slots)
    return want


# ── 从整份索引里捞候选 ────────────────────────────────────────────────

def _scan_one(npz_path: str, part: str, near_max: float, require_nearest: bool):
    """一份索引 → (rel_path, 命中的时间点 list, 该路有狗帧数, 该路可判部位帧数)。"""
    import json

    import numpy as np

    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "meta" not in z.files or "pose" not in z.files:
                return None
            meta = json.loads(str(z["meta"]))
            rows, ts = z["pose"], z["t"]
    except Exception:  # noqa: BLE001 单个文件坏了不该让整份扫描跑不出来
        return None
    if rows.size == 0 or len(rows) != len(ts):
        return None
    dists, _vis = decode(rows)
    slot, best = nearest(dists)
    m = match(rows, part, near_max, require_nearest)
    hits = [{"t": round(float(ts[i]), 2), "dist": round(float(best[i]), 3),
             "slot": SLOT_NAMES[int(slot[i])] if slot[i] >= 0 else "?"}
            for i in np.flatnonzero(m)]
    return str(meta.get("path") or ""), hits, int(len(rows)), int((slot >= 0).sum())


def find(index_dir: str, part: str, near_max: float | None = None,
         require_nearest: bool = True, max_per_video: int = 20) -> dict:
    """整份索引里所有"鼻子够到这个部位"的帧。不跑任何模型，几十秒扫完。

    每路最多留 max_per_video 个（按最近的排）：一只狗舔一次后爪能连出几百帧，
    全塞进去的话一路就把清单占满了，人看到的还是同一个场景。
    """
    slots = PARTS.get(part)
    thr = NEAR_MAX if near_max is None else float(near_max)
    out: list[dict] = []
    n_dog = n_part = 0
    files = sorted(os.listdir(index_dir)) if os.path.isdir(index_dir) else []
    for name in files:
        if not name.endswith(".npz"):
            continue
        r = _scan_one(os.path.join(index_dir, name), part, thr, require_nearest)
        if r is None:
            continue
        path, hits, rows, part_ok = r
        n_dog += rows
        n_part += part_ok
        hits.sort(key=lambda h: h["dist"])
        for h in hits[:max_per_video]:
            out.append({"path": path, **h})
    out.sort(key=lambda h: h["dist"])
    return {"part": part, "known": slots is not None, "near_max": thr,
            "hits": out, "with_dog": n_dog, "part_ok": n_part, "videos": len(files)}


def calib(index_dir: str, part: str, require_nearest: bool = True) -> str:
    """这个部位的距离分布 + 几个阈值各能捞到多少帧。阈值该定多少看这个，不要拍脑袋。"""
    import json

    import numpy as np

    slots = PARTS.get(part)
    if not slots:
        return f"不认识的部位：{part}（认得的：{'、'.join(PARTS)}）"
    vals = []
    for name in sorted(os.listdir(index_dir)) if os.path.isdir(index_dir) else []:
        if not name.endswith(".npz"):
            continue
        try:
            with np.load(os.path.join(index_dir, name), allow_pickle=False) as z:
                if "pose" not in z.files:
                    continue
                rows = z["pose"]
        except Exception:  # noqa: BLE001
            continue
        if rows.size == 0:
            continue
        dists, _ = decode(rows)
        slot, best = nearest(dists)
        sel = np.isin(slot, slots) if require_nearest else np.ones(len(dists), bool)
        for s in slots:
            d = dists[sel, s]
            vals.append(d[d > 0])
    if not vals or not len(np.concatenate(vals)):
        return f"{part}：一帧都没有（这个部位在索引里从来没被判成最近的）"
    v = np.concatenate(vals)
    qs = [5, 25, 50, 75, 95]
    lines = [f"{part}：{len(v):,} 帧鼻子最近的是它（距离单位 = 体长，脖子到尾根）", ""]
    lines.append("  分位数  " + "  ".join(f"p{q}={np.percentile(v, q):.2f}" for q in qs))
    lines.append("")
    lines.append("  阈值   能捞到的帧数")
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        n = int((v <= thr).sum())
        lines.append(f"  ≤{thr:.1f}   {n:>9,}  {n / len(v) * 100:5.1f}%")
    lines.append("")
    lines.append("  挑阈值的办法：从能捞到几百帧的那一档开始，用 --part 把清单拉出来，"
                 "抽十几帧看准不准；宁可严一点——人排除误报比漏掉一个新场景贵。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="靠姿态几何从已建索引里捞部位候选（不跑模型）")
    ap.add_argument("--part", required=True, help="部位名，比如 后爪 / 后右爪 / 尾根")
    ap.add_argument("--index-dir", default=config.EMBED_INDEX_DIR)
    ap.add_argument("--near-max", type=float, default=None, help=f"多近算够到（体长倍数，默认 {NEAR_MAX}）")
    ap.add_argument("--any", action="store_true", help="只要够近就算，不要求是最近的那个")
    ap.add_argument("--per-video", type=int, default=20, help="每路最多留几个")
    ap.add_argument("--limit", type=int, default=50, help="最多打印几条")
    ap.add_argument("--calib", action="store_true", help="先看距离分布，定阈值用")
    args = ap.parse_args()

    key = part_of(args.part) or args.part
    if args.calib:
        print(calib(args.index_dir, key))
        return
    r = find(args.index_dir, key, args.near_max, not args.any, args.per_video)
    if not r["known"]:
        print(f"不认识的部位：{args.part} → {key}（认得的：{'、'.join(PARTS)}）")
        return
    print(f"{args.part} → {key}：{r['videos']} 路索引，有狗 {r['with_dog']:,} 帧，"
          f"其中 {r['part_ok']:,} 帧判得出部位，捞到 {len(r['hits']):,} 个候选"
          f"（近到 {r['near_max']} 体长以内，每路最多 {args.per_video} 个）")
    if not r["hits"]:
        print("一个都没有。先跑 --calib 看看这个部位的距离分布，阈值可能定得太严。")
        return
    print()
    for h in r["hits"][:args.limit]:
        print(f"  {h['dist']:.2f} 体长  {h['t']:>8.1f}s  {os.path.basename(h['path'])}")
    if len(r["hits"]) > args.limit:
        print(f"  …… 还有 {len(r['hits']) - args.limit:,} 个")


if __name__ == "__main__":
    main()
