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

2. **真值跟机器的答案分开放。** 人填 `人工答案.csv`，模型的答案进 `答题卡.csv`，
   几何判的在密钥文件里。看着机器的答案填，填出来的"真值"里就掺了机器的错——
   而这件事不该靠"记得先填"这条纪律，用文件结构保证就不会忘。

3. **「看不清」是个正经答案。** 同一份真值里 45% 的帧人也判不了（夜里低对比、
   狗蜷着、多狗同框）。逼人在看不清的帧上二选一，等于往真值里注噪声。
   算分时这些帧从分母里剔掉。

## 用法

    python -m vision_service.evalpack --out ./tmp/evalpack -n 20
    # 把 发给模型.zip 丢进各家 web 对话框，回复原样存成文件，然后：
    python -m vision_service.evalpack --fill ./tmp/evalpack --model 豆包1.6 --from 回复.txt
    # 人填完 人工答案.csv（跟上一步谁先谁后都行），再跑：
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
# 真值单独一个文件，不跟模型答案混在一张表里。
# 这样填真值的人**先跑模型再填也不会受影响**——他打开的那张表里根本没有模型的答案。
# 顺序不该成为一条要人记住的纪律，能用文件结构保证就别靠自觉
TRUTH_FILE = "人工答案.csv"
TRUTH_COL = "人工填这一列"
ZIP_NAME = "发给模型.zip"


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
        sheet.append({"图": name, **{c: "" for c in MODEL_COLS}})
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
    _w(TRUTH_FILE, [{"图": r["图"], TRUTH_COL: ""} for r in sheet])
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
    with open(os.path.join(out_dir, "提示词_压缩包.txt"), "w", encoding="utf-8") as f:
        f.write(_ZIP_PROMPT.format(sys=sys_, user=user, sep="=" * 60, n=len(sheet)))
    with open(os.path.join(out_dir, "怎么用.md"), "w", encoding="utf-8") as f:
        f.write(_HOWTO.format(n=len(sheet), n_ctl=sum(1 for a in answer if a["是对照"]),
                              cols="、".join(MODEL_COLS), truth_col=TRUTH_COL))
    # **压缩包里只放图和提示词。** 答题卡和密钥一旦进去，模型就能看到"正确答案"
    # 和"哪几张是对照"——那这一整套验证就白做了
    zip_path = os.path.join(out_dir, ZIP_NAME)
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(img_dir)):
            z.write(os.path.join(img_dir, name), f"图/{name}")
        z.write(os.path.join(out_dir, "提示词_压缩包.txt"), "提示词.txt")
    return (f"打好了：{out_dir}/（{len(sheet)} 张，其中对照 "
            f"{sum(1 for a in answer if a['是对照'])} 张）\n"
            f"  {ZIP_NAME}   **直接把这个丢给对话框**（只含图和提示词，"
            f"不含答案和对照名单）\n"
            f"  {TRUTH_FILE}     人填这个。**跟模型的答案分开放**，"
            f"所以先跑模型再填也不会受影响\n"
            f"  答题卡.csv     模型的答案填这里（用 --fill 灌，不用手抄）\n"
            f"  图/ 提示词.txt  想一张一张来的话用这两个\n"
            f"  怎么用.md      给要试的人看的\n\n"
            f"  下一步：把 {ZIP_NAME} 丢给对话框 → 回复存成文件 → \n"
            f"    python -m vision_service.evalpack --fill {out_dir} --model 豆包1.6 --from 回复.txt")


_HOWTO = """# 怎么试

一共 {n} 张图（其中 {n_ctl} 张是**对照**，混在里面，你不知道是哪几张——这是故意的）。

**人工答案和模型答案在两个文件里**（`人工答案.csv` / `答题卡.csv`），所以下面这
两步**谁先谁后都行**——你打开填真值的那张表里，根本没有模型的答案，想被带偏
也带不动。

## 一：拿模型试

把 `发给模型.zip` **整个丢进对话框**。包里只有图和提示词（答题卡和答案密钥
都不在里面——进去了模型就能看见正确答案和哪几张是对照，这套验证就白做了）。
模型会逐张输出一行 JSON。

把它那一整段回复**原样**存成文本文件，然后用第三步的 `--fill` 灌进去，不用手抄。

> **代价要说清楚**：线上是一张图一次请求，模型看不到别的图。一个压缩包里 {n} 张
> 在同一个对话里，前面的答案会带着后面走。**要给领导看的那个数，用 `图/` +
> `提示词.txt` 一张图开一个新对话**，十几分钟。压缩包这条路用来先花五分钟
> 摸个底：这家值不值得认真试。两种都做的话，对一下差多少，心里有个数。

## 二：人填真值

打开 `图/` 里的图，在 **`人工答案.csv`** 的「{truth_col}」写：这只狗的口鼻贴着
自己的哪个部位。

可填：`前左爪` `前右爪` `后左爪` `后右爪` `尾根` `没贴到` `看不清`

**「看不清」是正经答案，别硬选。** 夜里太暗、狗蜷着挡住了、画面里不止一只狗——
这些都填「看不清」。实测这种能占到四成多，逼自己二选一只会把真值搞脏。

> 这一列是**标尺**。后面所有模型的分都是拿它量出来的，所以它得干净。
> 别打开 `答题卡.csv` 照着填——那是模型的答案，照着填等于把机器的错抄进标尺。

## 三：把模型的答案填进去（不用手抄）

把模型那一整段回复**原样**存成一个文本文件（比如 `答案_豆包.txt`），然后：

```bash
python -m vision_service.evalpack --fill <这个目录> --model 豆包1.6 --from 答案_豆包.txt
```

它会把回复里的 JSON 按顺序填进 `答题卡.csv` 的「豆包1.6」那一列。再填别家就换个
`--model` 名字，上一家的列不会被覆盖（`{cols}` 只是占位列名，填过一家就没了）。

> 条数对不上时它**不会填**，会告诉你解析出几条、有几张图。别硬凑——从第一条错位
> 开始后面全错，比没有还糟。把缺的那几张单独问一次补上就行。

## 四：算分

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


_ZIP_PROMPT = """把这个压缩包解开，里面「图/」下有 {n} 张图（00.jpg、01.jpg …）。

**逐张看，按文件名从小到大的顺序，每张输出一行 JSON，前面带上文件名。**
像这样：

00.jpg {{"see": "...", "desc": "...", "contact": ..., "part": ..., "moving": ..., "confidence": ...}}
01.jpg {{...}}

这 {n} 张是不同时刻、可能是不同的狗，**彼此无关**——请分别独立判断，
不要因为前一张是什么就倾向于后一张也是什么。

{sep}
{sys}

{user}
{sep}
"""


def parse_reply(text: str, n: int) -> tuple[list[str], str]:
    """模型一次答几张时回的那一坨 → 按顺序的部位列表。

    手抄二十几个答案是整件事里最烦的一步，也最容易抄错。这里直接吃它的原文。

    认两种写法：带序号的 JSON（`1. {...}`）和光秃秃的一串 JSON。**按出现顺序对位**，
    不信序号——模型偶尔会把序号写错或跳号，而顺序几乎不会乱。数量对不上时如实说，
    不硬凑：凑出来的对位是错的，比没有还糟。
    """
    import json as _json
    import re as _re

    out: list[str] = []
    for m in _re.finditer(r"\{[^{}]*\}", text or "", _re.S):
        try:
            d = _json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(d, dict) or not ({"part", "contact", "see"} & set(d)):
            continue
        see = str(d.get("see") or "")
        part = d.get("part")
        if see == "unclear":
            out.append("看不清")
        elif not d.get("contact", bool(part)) or not part:
            out.append("没贴到")
        else:
            out.append(str(part))
    why = ""
    if len(out) != n:
        why = (f"解析出 {len(out)} 条，但有 {n} 张图——**没往里填**。"
               "常见原因：模型少答了几张、或者把几张合并成一条。"
               "对不上就别硬凑，凑出来的对位是错的，比没有还糟。"
               "把回复里缺的那几张补齐再来，或者那几张单独问一次。")
    return out, why


def fill(out_dir: str, model: str, reply_path: str) -> str:
    """把模型回复里的答案直接写进答题卡的某一列。"""
    try:
        with open(reply_path, encoding="utf-8") as f:
            text = f.read()
        with open(os.path.join(out_dir, "答题卡.csv"), encoding="utf-8-sig") as f:
            sheet = list(csv.DictReader(f))
    except OSError as e:
        return f"读不到：{e}"
    if not sheet:
        return "答题卡是空的"
    parts, why = parse_reply(text, len(sheet))
    if why:
        return why
    # 列名就用模型名：算分时原样打出来，一眼看得出是哪家
    # 保留已经填过的别家那几列，只把还没用过的占位列（模型A/B/C）去掉
    keep = [c for c in sheet[0] if c not in MODEL_COLS and c != model]
    fields = keep + [model]
    for r, p_ in zip(sheet, parts):
        for c in MODEL_COLS:
            r.pop(c, None)
        r[model] = p_
    with open(os.path.join(out_dir, "答题卡.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sheet)
    return (f"填好了 {len(parts)} 条到「{model}」这一列。\n"
            f"  再填别家：换个 --model 名字跑一次（老的那一列不会被覆盖）\n"
            f"  真值在 {TRUTH_FILE} 里，还得人自己填——那是标尺，机器填了就没意义了\n"
            f"  填完：python -m vision_service.evalpack --score {out_dir}")


def _truth(out_dir: str, sheet: list[dict]) -> tuple[dict[str, str], str]:
    """真值：优先读单独的 人工答案.csv；老包里它在答题卡上，也照样认。

    老包是在"真值和模型答案同一张表"的时候打的。换文件结构不该把已经填完的
    老包作废——那会逼人把标尺重填一遍，而标尺重填就意味着换了把尺子。
    """
    try:
        with open(os.path.join(out_dir, TRUTH_FILE), encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        col = next((c for c in rows[0] if "人工" in c), None) if rows else None
        if col:
            return {r["图"]: (r.get(col) or "").strip() for r in rows}, ""
    except OSError:
        pass
    col = next((c for c in sheet[0] if "人工" in c), None) if sheet else None
    if col:        # 老包：真值还在答题卡上
        return {r["图"]: (r.get(col) or "").strip() for r in sheet}, col
    return {}, ""


def score(out_dir: str) -> str:
    """读答题卡 + 人工答案 + 密钥，出各家的准确率对比。"""
    try:
        with open(os.path.join(out_dir, "答题卡.csv"), encoding="utf-8-sig") as f:
            sheet = list(csv.DictReader(f))
        with open(os.path.join(out_dir, ANSWER_KEY), encoding="utf-8-sig") as f:
            key = {r["图"]: r for r in csv.DictReader(f)}
    except OSError as e:
        return f"读不到答题卡或密钥：{e}"
    truth, truth_col = _truth(out_dir, sheet)
    if not truth:
        return f"找不到真值：{out_dir}/{TRUTH_FILE} 里要有「{TRUTH_COL}」这一列"
    filled = [r for r in sheet if truth.get(r["图"])]
    if not filled:
        return f"{TRUTH_FILE} 还是空的——先让人填，再拿模型的答案跟它比"

    done = [r for r in filled if truth[r["图"]] != "看不清"]
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
        k = sum(1 for r in rows_ if (r.get(c) or "").strip() == truth[r["图"]])
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
    best = max(cols, key=lambda c: sum(1 for r in main if (r.get(c) or "").strip() == truth[r["图"]]))
    b = sum(1 for r in main if (r.get(best) or "").strip() == truth[r["图"]])
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
    ap.add_argument("--fill", metavar="DIR",
                    help="把模型回复的原文直接填进答题卡，不用手抄。"
                         "配 --model（列名）和 --from（存着回复原文的文件）")
    ap.add_argument("--model", help="--fill 用：这一列叫什么，比如 豆包1.6")
    ap.add_argument("--from", dest="from_file", help="--fill 用：存着模型回复原文的文件")
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
    if args.fill:
        if not args.model or not args.from_file:
            ap.error("--fill 要配 --model（列名）和 --from（回复原文的文件）")
        print(fill(args.fill, args.model, args.from_file))
        return
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
