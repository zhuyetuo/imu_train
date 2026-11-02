"""采样率可以是小数（QMI8658 的原生档位是 31.25 / 15.625 Hz，没有 16 / 25 / 50）。

以前到处是 int(hz)、gcd(src, dst)，31.25 进来要么被截成 31、要么直接报错。
统一走这里：
  parse_hz("31.25") -> 31.25，parse_hz("16") -> 16（整数还是 int，目录名 16hz 不变）
  rate_ratio(50, 31.25) -> (5, 8)：重采样的 up / down（resample_poly 要整数）
"""
from fractions import Fraction


def parse_hz(v) -> float | int:
    x = float(v)
    if x <= 0:
        raise ValueError(f"采样率要 > 0，收到 {v!r}")
    return int(x) if x.is_integer() else x


def rate_ratio(source_hz, target_hz) -> tuple[int, int]:
    """source → target 的 (up, down)，都是整数。50→31.25 = (5, 8)，50→16 = (8, 25)"""
    f = (Fraction(str(parse_hz(target_hz))) / Fraction(str(parse_hz(source_hz)))).limit_denominator(100000)
    return f.numerator, f.denominator


def n_points(seconds: float, hz: float) -> int:
    """秒 → 点数。推理侧按 meta 里存的 window_s（3 位小数）换回点数，四舍五入，
    不能用 int()：1.984 s × 31.25 = 61.99999… 截成 61，跟训练的 62 对不上"""
    return int(round(float(seconds) * float(hz)))
