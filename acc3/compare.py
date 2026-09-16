"""并排打印两个 ml_*.json 的指标。

    python acc3/compare.py <没有陀螺仪的>.json <基线>.json

单独一个文件是因为 train_acc3.sh 和别的地方都要用；内嵌在 shell 里的话
改一次要改两处，而漏改的那处会安静地打印旧格式。

**中文是双宽字符**，`{:>12}` 那种按字符数对齐的写法在混排时会歪，
所以这里自己按显示宽度补空格。
"""

from __future__ import annotations

import json
import sys
import unicodedata


def _w(s: str) -> int:
    """显示宽度。CJK 全角算 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s, width, right=True):
    s = str(s)
    gap = " " * max(0, width - _w(s))
    return gap + s if right else s + gap


def _load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _row(name, a, b, w1, w):
    """a = 没有陀螺仪，b = 基线。差值 = a - b。"""
    if b is None:
        return _pad(name, w1, False) + _pad("—", w) + _pad(f"{a:.4f}", w) + _pad("—", w)
    return (_pad(name, w1, False) + _pad(f"{b:.4f}", w)
            + _pad(f"{a:.4f}", w) + _pad(f"{a - b:+.4f}", w))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    mine, base = _load(sys.argv[1]), _load(sys.argv[2])
    if not mine:
        print(f"✗ 读不到 {sys.argv[1]}，看看训练那步是不是失败了", file=sys.stderr)
        return 1
    if not base:
        # **说出来**。只印一列数的话，人会以为"没有变化"
        print(f"（找不到基线 {sys.argv[2]}，只印这次的结果。\n"
              f"  基线 = 同一份数据、带陀螺仪训出来的那个 ml_*.json，\n"
              f"  一般在 results/<同名不带 _acc3 的目录>/... 下面）\n")
        base = {}

    w1, w = 16, 16
    print(_pad("指标", w1, False) + _pad("基线(acc+gyro)", w)
          + _pad("只有加速计", w) + _pad("差值", w))
    print("-" * (w1 + w * 3))
    for name, key in (("总体准确率", "accuracy"), ("macro F1", "macro_f1")):
        if mine.get(key) is not None:
            print(_row(name, mine[key], base.get(key), w1, w))

    pa, pb = mine.get("per_class") or {}, base.get("per_class") or {}
    if pa:
        print()
        print(_pad("类别", w1, False) + _pad("基线 F1", w)
              + _pad("只有加速计 F1", w) + _pad("差值", w))
        print("-" * (w1 + w * 3))
        for k in pa:
            a = pa[k].get("f1-score")
            if a is None:
                continue
            print(_row(k, a, (pb.get(k) or {}).get("f1-score"), w1, w))

    nm, nb = mine.get("n_features"), base.get("n_features")
    if nm or nb:
        print(f"\n特征维度：基线 {nb or '193(8通道)'} → 只有加速计 {nm or '113(5通道)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
