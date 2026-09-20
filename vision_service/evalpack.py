"""打一份「拿去 web 上手试」的包：图 + 提示词 + 答题卡 + 算分。

## 要解决什么

"要不要充钱买更好的大模型 API" 这个决定，需要一个**能拿给别人看的数字**，
而不是"我试了几张感觉还行"。但真要为此先接一家 API、写代码、跑一遍，
成本太高——万一那家也不行呢。

所以这里反过来：把**跟线上一模一样的图和提示词**打成一个包，人直接在各家的
web 对话框里粘贴、拖图、记答案。试哪家都行，不用接口、不用充值、不用写代码。
填完跑 --score，出一张各家准确率的对比表。

## 三条今天用真值换来的规矩，都焊在这个包里

1. **要有对照组。** 混进几张"几何判定鼻子离爪子很远"的帧。一个见什么都说
   "在舔后爪"的模型，在正式组上照样能拿高分——对照组是唯一能拆穿它的东西。
   2026-09-20 实测：正式组 15%、对照组也 15%，那个 15% 完全没有意义。

2. **人先填，别看机器的答案。** 几何判的那一列单独放在密钥文件里，答题卡上
   没有——看着机器的答案填，填出来的"真值"里就掺了机器的错。

3. **「看不清」是个正经答案。** 同一份真值里 45% 的帧人也判不了（夜里低对比、
   狗蜷着、多狗同框）。逼人在看不清的帧上二选一，等于往真值里注噪声。
   算分时这些帧从分母里剔掉。

## 用法

    python -m vision_service.evalpack --out ./tmp/evalpack -n 20
    # 把 evalpack 整个目录发给要试的人
    # 他填完 答题卡.csv，再跑：
    python -m vision_service.evalpack --score ./tmp/evalpack
"""

from __future__ import annotations

import argparse
import csv
import os
import random

from . import config, partask, posepart, seek

# 答题卡上给几家留的列。多写几列没坏处，空着的列不算分
MODEL_COLS = ("模型A", "模型B", "模型C")
ANSWER_KEY = ".答案密钥.csv"       # 点开头：发包时不显眼，人不会顺手打开看


def build(out_dir: str, *, index_dir: str, part: str, n: int, n_control: int,
          near_max: float, min_motion: float, video_root: str,
          n_frames: int = 6, step_s: float = 0.3, seed: int = 0) -> str:
    """打包：图 + 提示词 + 答题卡 + 密钥。"""
    img_dir = os.path.join(out_dir, "图")
    os.makedirs(img_dir, exist_ok=True)
    key = posepart.part_of(part) or part
    found = posepart.find(index_dir, key, near_max, True, 20, 60.0, 1, min_motion)
    hits = found["hits"][:n]
    ctl = partask.control_hits(index_dir, key, n_control, seed)
    # **打乱之后再编号**：正式的排前面、对照的排后面的话，人翻到一半就看出规律了
    rows = hits + ctl
    random.Random(seed).shuffle(rows)

    sheet, answer = [], []
    for r in rows:
        full = os.path.join(video_root, r["path"])
        if not os.path.isfile(full):
            continue
        frames, _why = partask.frames_around(full, r["path"], r["t"],
                                             n=n_frames, span_s=3.0, step_s=step_s)
        if not frames:
            continue
        name = f"{len(sheet):02d}.jpg"
        with open(os.path.join(img_dir, name), "wb") as f:
            f.write(seek.tile_frames(frames) or frames[0])
        sheet.append({"图": name, "人工填这一列": "", **{c: "" for c in MODEL_COLS}})
        answer.append({"图": name, "是对照": "是" if r.get("control") else "",
                       "几何判的": r.get("slot"), "几何距离": r.get("dist"),
                       "视频": r["path"], "秒": r["t"]})

    def _w(path, rows_):
        with open(os.path.join(out_dir, path), "w", encoding="utf-8-sig", newline="") as f:
            if rows_:
                w = csv.DictWriter(f, fieldnames=list(rows_[0]))
                w.writeheader()
                w.writerows(rows_)

    _w("答题卡.csv", sheet)
    _w(ANSWER_KEY, answer)

    sys_, user = seek.build_contact_prompt(partask.CONTACT_PARTS, n_frames,
                                           step_s * (n_frames - 1), tiled=True)
    with open(os.path.join(out_dir, "提示词.txt"), "w", encoding="utf-8") as f:
        f.write("把下面两段一起粘进对话框，然后把「图/」里的一张图拖进去。\n"
                "一次一张，每张都重新开一个对话（不然上一张会影响这一张的答案）。\n"
                f"{'=' * 60}\n{sys_}\n\n{user}\n{'=' * 60}\n")
    # 26 个对话确实烦。给一份"一次拖几张"的版本——但**它跟线上不是同一个条件**，
    # 这句必须写在文件里，不然拿它的数去跟线上比就错了
    with open(os.path.join(out_dir, "提示词_一次多张（省事但打折）.txt"), "w",
              encoding="utf-8") as f:
        f.write(_BATCH_PROMPT.format(sys=sys_, user=user, sep="=" * 60))
    with open(os.path.join(out_dir, "怎么用.md"), "w", encoding="utf-8") as f:
        f.write(_HOWTO.format(n=len(sheet), n_ctl=sum(1 for a in answer if a["是对照"]),
                              cols="、".join(MODEL_COLS)))
    return (f"打好了：{out_dir}/（{len(sheet)} 张，其中对照 "
            f"{sum(1 for a in answer if a['是对照'])} 张）\n"
            f"  图/          跟线上发给 API 的**一模一样**的拼图\n"
            f"  提示词.txt   跟线上**一字不差**的提示词，粘进对话框就行\n"
            f"  答题卡.csv   人先填「人工填这一列」，再把各家的答案填进模型列\n"
            f"  怎么用.md    给要试的人看的\n"
            f"  填完：python -m vision_service.evalpack --score {out_dir}")


_HOWTO = """# 怎么试

一共 {n} 张图（其中 {n_ctl} 张是**对照**，混在里面，你不知道是哪几张——这是故意的）。

## 第一步：人先填，别看任何模型的答案

打开 `图/` 里的图，在 `答题卡.csv` 的**「人工填这一列」**写：这只狗的口鼻贴着
自己的哪个部位。

可填：`前左爪` `前右爪` `后左爪` `后右爪` `尾根` `没贴到` `看不清`

**「看不清」是正经答案，别硬选。** 夜里太暗、狗蜷着挡住了、画面里不止一只狗——
这些都填「看不清」。实测这种能占到四成多，逼自己二选一只会把真值搞脏。

> 这一列是**标尺**。后面所有模型的分都是拿它量出来的，所以它得干净。

## 第二步：拿各家模型试

把 `提示词.txt` 里两段粘进对话框，拖一张图进去，把模型答的 `part` 填进
`{cols}` 里对应那一列（没贴到就填「没贴到」）。

**一张图开一个新对话**——不然上一张的答案会影响这一张。{n} 张大概十几分钟。

> 嫌慢的话有一份 `提示词_一次多张（省事但打折）.txt`，可以一次拖几张进同一个
> 对话。但那跟线上不是同一个条件（线上是一张一次请求），**要给领导看的那个数
> 还是用一张一张的**。

> **别把整个目录压缩了丢给对话框**：网页端基本不会解压；就算解开了，26 张在
> 同一个对话里也会互相影响，测出来的不是单张的能力。

在表头把列名改成真实模型名（比如 `豆包1.6`、`GPT-5`、`Claude`），算分时会原样打出来。

## 第三步：算分

```bash
python -m vision_service.evalpack --score <这个目录>
```

出来一张表：每家在**正式组**的准确率、在**对照组**的表现、以及跟瞎猜（六选一
约 17%）的对比。

## 看结果时注意一件事

**对照组是用来拆穿"见什么都说在舔"的。** 那几张图里狗的口鼻离爪子很远，正确答案
基本都是「没贴到」。一个模型如果在对照组上也大量报「某只爪」，那它在正式组上的
分数是假的——它没在看画面，在顺着提示词猜。

2026-09-20 实测就撞上过：正式组 15%、对照组也 15%，那个 15% 一点意义都没有。
"""


_BATCH_PROMPT = """一次拖好几张图进同一个对话，让它按序号逐张回答。

**先说清楚代价，别拿这份的数去跟线上比：**

  线上是**一张图一次请求**，模型看不到别的图。一次给它好几张时：
    · 前面几张的答案会带着后面的走（它倾向于给出一致的答案）
    · 图多了之后每张分到的"注意力"变少，细节更容易漏
  所以这份测出来的数**通常比线上略好或略差，但不等于线上**。

**建议**：要给领导看的那个数，用「提示词.txt」一张一张来；这份只用来
  快速摸个底（比如先花五分钟看看某家值不值得认真试）。
  两种都做的话，在同一批图上对一下差多少，心里有个数。

{sep}
{sys}

{user}

**这次我会一次给你好几张图。它们是不同时刻、可能是不同的狗，彼此无关——
请分别独立判断，不要因为前一张是什么就倾向于后一张也是什么。**
按我拖进来的顺序，每张图输出一行 JSON，前面加上序号，像这样：

1. {{"see": "...", "desc": "...", "contact": ..., "part": ..., "moving": ..., "confidence": ...}}
2. {{...}}

{sep}
"""


def score(out_dir: str) -> str:
    """读答题卡 + 密钥，出各家的准确率对比。"""
    try:
        with open(os.path.join(out_dir, "答题卡.csv"), encoding="utf-8-sig") as f:
            sheet = list(csv.DictReader(f))
        with open(os.path.join(out_dir, ANSWER_KEY), encoding="utf-8-sig") as f:
            key = {r["图"]: r for r in csv.DictReader(f)}
    except OSError as e:
        return f"读不到答题卡或密钥：{e}"
    truth_col = next((c for c in sheet[0] if "人工" in c), None) if sheet else None
    if not truth_col:
        return "答题卡里没有「人工填这一列」，是不是改了表头？"
    filled = [r for r in sheet if (r.get(truth_col) or "").strip()]
    if not filled:
        return "「人工填这一列」还是空的——先让人填，再拿模型的答案跟它比"

    done = [r for r in filled if r[truth_col].strip() != "看不清"]
    blur = len(filled) - len(done)
    # 模型列 = 除了「图」和人工那一列之外、填了东西的列
    cols = [c for c in sheet[0] if c not in ("图", truth_col)
            and any((r.get(c) or "").strip() for r in done)]
    lines = [f"人工填了 {len(filled)} 张，其中「看不清」{blur} 张"
             f"（{blur / len(filled) * 100:.0f}%，从分母里剔掉），按 {len(done)} 张算。", ""]
    if not cols:
        lines.append("模型那几列还是空的——先去各家对话框里试，把答案填进去。")
        return "\n".join(lines)

    hdr = f"  {'':<12}" + "".join(f"{c:<14}" for c in cols)
    lines.append(hdr)
    ctl = [r for r in done if (key.get(r["图"], {}).get("是对照") or "").strip()]
    main = [r for r in done if r not in ctl]

    def acc(rows_, c):
        if not rows_:
            return "—"
        k = sum(1 for r in rows_ if (r.get(c) or "").strip() == r[truth_col].strip())
        return f"{k}/{len(rows_)} ({k / len(rows_) * 100:.0f}%)"

    lines.append(f"  {'正式组':<12}" + "".join(f"{acc(main, c):<14}" for c in cols))
    if ctl:
        lines.append(f"  {'对照组':<12}" + "".join(f"{acc(ctl, c):<14}" for c in cols))
        # 对照组上乱报部位 = 没在看画面。这一行比准确率更早暴露问题
        lines.append(f"  {'对照组乱报':<10}"
                     + "".join(f"{sum(1 for r in ctl if (r.get(c) or '').strip().endswith('爪')):<14}"
                               for c in cols))
    lines.append(f"  {'瞎猜':<12}" + f"{'约 17%':<14}")
    lines.append("")
    lines.append("  「对照组乱报」= 在「狗的口鼻离爪子很远」的图上还报出某只爪的张数。"
                 "这一行大的，正式组那个分就是假的——模型在顺着提示词猜，没在看画面。")
    best = max(cols, key=lambda c: sum(1 for r in main if (r.get(c) or "").strip() == r[truth_col].strip()))
    b = sum(1 for r in main if (r.get(best) or "").strip() == r[truth_col].strip())
    rate = b / len(main) if main else 0
    lines.append("")
    if rate < 0.3:
        lines.append(f"  最好的一家（{best}）也只有 {rate * 100:.0f}%，跟瞎猜（17%）差不多 → "
                     "**换更贵的模型解决不了这个问题**，别充。这一步交给人，"
                     "或者先把画面本身改好（机位、分辨率、夜间补光）。")
    elif rate < 0.6:
        lines.append(f"  最好的一家（{best}）{rate * 100:.0f}% → 比瞎猜强，但还不能自动落标签。"
                     "可以拿来排序、把最像的排前面给人看，省人的时间；别指望免掉人。")
    else:
        lines.append(f"  最好的一家（{best}）{rate * 100:.0f}% → 这个准确率配上人工复核是能用的。"
                     f"算一笔账再定：全量多少条 × 每条多少 token × 那家的价。")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="打一份拿去 web 上手试的包（图 + 提示词 + 答题卡）")
    ap.add_argument("--out", metavar="DIR", help="打包到哪儿")
    ap.add_argument("--score", metavar="DIR", help="读回填好的答题卡，出各家准确率对比")
    ap.add_argument("--index-dir", default=config.EMBED_INDEX_DIR)
    ap.add_argument("--part", default="后爪")
    ap.add_argument("-n", type=int, default=20, help="正式组几张")
    ap.add_argument("--control", type=int, default=6, help="混几张对照（别省，见 怎么用.md）")
    ap.add_argument("--near-max", type=float, default=0.4)
    ap.add_argument("--min-motion", type=float, default=0.06)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--step", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    if args.score:
        print(score(args.score))
        return
    if not args.out:
        ap.error("要么 --out 打包，要么 --score 算分")
    print(build(args.out, index_dir=args.index_dir, part=args.part, n=args.n,
                n_control=args.control, near_max=args.near_max, min_motion=args.min_motion,
                video_root=config.VIDEO_ROOT, n_frames=args.frames, step_s=args.step,
                seed=args.seed))


if __name__ == "__main__":
    main()
