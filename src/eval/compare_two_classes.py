"""
两个类别（默认 未佩戴 vs 静止/睡觉）的原始数据到底差在哪：逐个特征算可分性，回答
"模型分不开，是不是因为数据本来就几乎一样"，以及"有没有哪个量能把它们分开"。

思路：戴着的项圈哪怕狗睡得很沉，也贴着脖子，有呼吸（0.2~0.5Hz）、心跳（1~2.5Hz）
带来的微幅晃动；放在桌上 / 充电时这部分没有，只剩传感器本底噪声。所以除了常规的
标准差，重点看**微振动**那几个量（带通能量、模长的细小起伏、陀螺本底）。

用法（CSV 就是平台导出给训练用的那份：record_id,label,acc_x..gyr_z）：

    python src/eval/compare_two_classes.py --csv data/raw_custom/data.csv --hz 16 \
        --a 未佩戴 --b 静止,睡 --win 10 --out reports/notworn_vs_rest

--a / --b：类别名里带这些字就算（逗号分隔多个），比如 --b 静止,睡 把「静止/休息」「睡觉」「睡眠」都算进 b。
输出：终端一张表（按可分性排序）+ <out>.csv 每个窗口的特征 + <out>.png 前几个特征的分布（有 matplotlib 时）。

可分性用 AUC：0.5 = 完全分不开（两类这个量一样），1.0 / 0.0 = 一刀就能分开。
看 |AUC-0.5|：>0.4 很好分、0.25~0.4 有用、<0.15 基本没区别。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

ACC = ["acc_x", "acc_y", "acc_z"]
GYR = ["gyr_x", "gyr_y", "gyr_z"]


def _band(x: np.ndarray, hz: float, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, hz / 2 * 0.95)
    if lo >= hi or len(x) < 3 * int(hz):
        return np.zeros_like(x)
    sos = butter(2, [lo, hi], btype="band", fs=hz, output="sos")
    return sosfiltfilt(sos, x - x.mean())


def window_features(acc: np.ndarray, gyr: np.ndarray, hz: float) -> dict[str, float]:
    mag = np.linalg.norm(acc, axis=1)
    gmag = np.linalg.norm(gyr, axis=1)
    jerk = np.abs(np.diff(mag)) * hz if len(mag) > 1 else np.zeros(1)
    g = acc.mean(axis=0)
    gn = np.linalg.norm(g) or 1.0
    f = {
        "acc轴标准差(均)": float(acc.std(axis=0).mean()),
        "acc模长标准差": float(mag.std()),
        "acc模长均值": float(mag.mean()),
        "acc模长极差": float(np.ptp(mag)),
        "陀螺模长均值": float(gmag.mean()),
        "陀螺模长标准差": float(gmag.std()),
        "jerk中位数": float(np.median(jerk)),
        # 微振动：呼吸 / 心跳频段的能量（模长带通后的方差）
        "呼吸带能量(0.15-0.6Hz)": float(_band(mag, hz, 0.15, 0.6).var()),
        "心跳带能量(0.8-3Hz)": float(_band(mag, hz, 0.8, 3.0).var()),
        "陀螺呼吸带能量": float(_band(gmag, hz, 0.15, 0.6).var()),
        # 姿态：放桌上多半是平放（重力几乎全在一个轴上），戴着睡会歪
        "重力最大轴占比": float(np.abs(g).max() / gn),
        "俯仰角(度)": float(np.degrees(np.arctan2(g[0], np.hypot(g[1], g[2])))),
        "横滚角(度)": float(np.degrees(np.arctan2(g[1], g[2]))),
    }
    return f


def auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a 的值 > b 的值)，Mann-Whitney 口径"""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    ranks = pd.Series(allv).rank().to_numpy()
    ra = ranks[: len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def best_threshold(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """一刀切：哪个阈值两类分得最开（Youden），返回 (阈值, 准确率)"""
    allv = np.unique(np.concatenate([a, b]))
    if len(allv) > 400:
        allv = np.quantile(allv, np.linspace(0, 1, 400))
    best = (float("nan"), 0.0)
    for t in allv:
        acc1 = ((a > t).mean() + (b <= t).mean()) / 2
        acc2 = ((a <= t).mean() + (b > t).mean()) / 2
        acc_ = max(acc1, acc2)
        if acc_ > best[1]:
            best = (float(t), float(acc_))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--hz", type=float, required=True, help="CSV 的采样率（平台导出的看数据集 meta 里的 hz）")
    ap.add_argument("--a", default="未佩戴")
    ap.add_argument("--b", default="静止,睡")
    ap.add_argument("--win", type=float, default=10.0, help="每个窗口几秒（微振动要长一点，10~30 秒）")
    ap.add_argument("--max-per-class", type=int, default=20000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    keys_a = [k for k in args.a.split(",") if k]
    keys_b = [k for k in args.b.split(",") if k]
    cols = ["record_id", "label"] + ACC + GYR
    df = pd.read_csv(args.csv, usecols=lambda c: c in cols, dtype={"label": "category", "record_id": "category"})
    has_gyr = all(c in df.columns for c in GYR)

    def cls_of(lab: str) -> str | None:
        if any(k in lab for k in keys_a):
            return "A"
        if any(k in lab for k in keys_b):
            return "B"
        return None

    n = int(args.win * args.hz)
    rows = []
    for rid, g in df.groupby("record_id", observed=True):
        labs = g["label"].astype(str).to_numpy()
        acc = g[ACC].to_numpy(np.float64)
        gyr = g[GYR].to_numpy(np.float64) if has_gyr else np.zeros_like(acc)
        # 同一类的连续一段切成不重叠的窗口
        i = 0
        while i < len(labs):
            j = i
            while j < len(labs) and labs[j] == labs[i]:
                j += 1
            c = cls_of(labs[i])
            if c:
                for s in range(i, j - n + 1, n):
                    f = window_features(acc[s:s + n], gyr[s:s + n], args.hz)
                    f.update({"class": c, "label": labs[i], "record_id": rid})
                    rows.append(f)
            i = j
    feat = pd.DataFrame(rows)
    if feat.empty or feat["class"].nunique() < 2:
        raise SystemExit(f"两类窗口不够：{feat['class'].value_counts().to_dict() if not feat.empty else {}}")
    feat = pd.concat([d.sample(min(len(d), args.max_per_class), random_state=0)
                      for _, d in feat.groupby("class")], ignore_index=True)

    a_df, b_df = feat[feat["class"] == "A"], feat[feat["class"] == "B"]
    print(f"A = 名字含 {keys_a}：{len(a_df)} 个窗口，{a_df['record_id'].nunique()} 条记录，类别 {sorted(a_df['label'].unique())}")
    print(f"B = 名字含 {keys_b}：{len(b_df)} 个窗口，{b_df['record_id'].nunique()} 条记录，类别 {sorted(b_df['label'].unique())}")
    print(f"窗口 {args.win}s @ {args.hz}Hz\n")

    names = [c for c in feat.columns if c not in ("class", "label", "record_id")]
    res = []
    for k in names:
        va, vb = a_df[k].to_numpy(), b_df[k].to_numpy()
        u = auc(va, vb)
        t, accv = best_threshold(va, vb)
        res.append({"特征": k, "AUC": u, "可分性": abs(u - 0.5) * 2,
                    "A中位数": np.median(va), "A_p10": np.quantile(va, .1), "A_p90": np.quantile(va, .9),
                    "B中位数": np.median(vb), "B_p10": np.quantile(vb, .1), "B_p90": np.quantile(vb, .9),
                    "一刀阈值": t, "一刀准确率": accv})
    r = pd.DataFrame(res).sort_values("可分性", ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4g}".format):
        print(r.to_string(index=False))
    top = r.iloc[0]
    print(f"\n最能分开的是「{top['特征']}」：AUC {top['AUC']:.3f}，一刀切在 {top['一刀阈值']:.4g} 能对 {top['一刀准确率']:.1%}。")
    if top["可分性"] < 0.3:
        print("所有特征的可分性都很低：在这个采样率 / 窗口长度下两类原始数据确实几乎一样，"
              "模型分不开不是训练的问题。可以试 --win 30（微振动要长窗口才稳）、或者用更高采样率的原始数据再跑。")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        feat.to_csv(args.out + ".csv", index=False)
        r.to_csv(args.out + "_summary.csv", index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "WenQuanYi Zen Hei", "DejaVu Sans"]
            ks = list(r["特征"][:6])
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for ax, k in zip(axes.flat, ks):
                lo, hi = np.quantile(feat[k], [0.01, 0.99])
                bins = np.linspace(lo, hi, 50) if hi > lo else 30
                ax.hist(a_df[k], bins=bins, alpha=.55, density=True, label=f"A {args.a}")
                ax.hist(b_df[k], bins=bins, alpha=.55, density=True, label=f"B {args.b}")
                ax.set_title(f"{k}  AUC={r.set_index('特征').loc[k, 'AUC']:.2f}")
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.out + ".png", dpi=110)
            print(f"图：{args.out}.png")
        except Exception as e:  # noqa: BLE001
            print(f"（没出图：{e}）")
        print(f"明细：{args.out}.csv / {args.out}_summary.csv")


if __name__ == "__main__":
    main()
