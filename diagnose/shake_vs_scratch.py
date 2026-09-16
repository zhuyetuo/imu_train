"""甩身体为什么会被判成抓挠——用数据回答，不猜。

    python diagnose/shake_vs_scratch.py \
        --processed_dir data/processed_2026_8_11-2026_8_27_raw_missing_drop_window \
        --model results/processed_.../rf/ml_rf.pkl \
        --hz 16

分三段，各回答一个假设：

① **后处理**：抓挠会主动吞并前后的甩身体窗口（label_service 的
   STABLE_SHAKE_ABSORB_S，默认 3 秒），而且写时间轴时抓挠优先。
   这一段只是把规则和参数打出来——**平台上那个"甩身体变抓挠"很可能
   根本不是模型的问题**，验法见最后的建议。

② **标注**：模型高置信度判甩身体、而标注是抓挠的窗口（以及反向）。
   这些是可疑标注的候选，导出成 CSV 拿去复查。

③ **采样率**：抓挠和甩身体各自的主频分布。
   结论先说：**提到 25Hz 不会让这两类分开**——1 秒窗口下 16Hz 和 25Hz 的
   频率分辨率都是 1Hz，而这两类的主频只差 0.5Hz。要分开得**加长窗口**。
   实测见 diagnose/test_resolution.py。

不装 sklearn 也能跑 ③（它只要 npz）；① 只读配置；② 需要模型。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, os.path.join(_REPO, "src"), os.path.join(_REPO, "src", "data"),
           os.path.join(_REPO, "src", "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCRATCH, SHAKE = "抓挠", "甩身体"


# ── ① 后处理里那几条会把甩身体变成抓挠的规则 ──────────────────────────────


def report_postprocess_rules():
    print("① 后处理里有三条规则会把甩身体变成抓挠\n")
    try:
        from label_service import config as ls
        absorb = ls.STABLE_SHAKE_ABSORB_S
        gap = ls.STABLE_EVENT_GAP_S
        events = ls.STABLE_EVENT_LABELS
    except Exception as e:  # noqa: BLE001
        print(f"   （读不到 label_service/config.py：{e}，按默认值说）")
        absorb, gap, events = 3.0, 4.0, [SCRATCH, SHAKE]

    print(f"   1. 吞并：抓挠 bout 前后 {absorb} 秒内的甩身体窗口**并进抓挠**")
    print(f"      （STABLE_SHAKE_ABSORB_S={absorb}）")
    print("      往外扩的**累计**秒数不超过这个值。")
    print("      （早先的写法看的是相邻窗口的间隔，而那个间隔恒等于 0，")
    print("       于是链式吞到底——参数形同虚设，写 0 也关不掉。已修。）")
    print("   2. 被吞掉的窗口**从甩身体那边删掉**，不是两边都算")
    print("   3. 写时间轴时**抓挠优先**：抓挠能覆盖已经写上的甩身体，反过来不行")
    print()
    print("   postprocess.py 顶部写了这条规则的来由：「抓完常甩一下，模型也爱把")
    print("   抓挠的剧烈段判成甩身体」——**它是为了修反方向的错而加的**。")
    print("   你现在看到的是它矫枉过正。")
    print()
    print(f"   相关：同类事件间隔 ≤ {gap}s 会合成一个 bout（STABLE_EVENT_GAP_S），")
    print(f"        事件类是 {events}")
    print()
    print("   ▸ 怎么验（平台上现成就能做，不用改代码）：")
    print("     同一个样本分别用「调试版」和「稳定版 v2」跑一遍。")
    print("     调试版是模型逐窗口的原始输出，**不过后处理**。")
    print("       · 调试版里是甩身体、稳定版里变抓挠  → 100% 是后处理这几条规则")
    print("       · 两个版本都判成抓挠                → 才是模型/数据的问题，看下面 ②③")
    print()
    print("   ▸ 要关掉吞并：STABLE_SHAKE_ABSORB_S=0（0 = 关闭，跟 spectral_min 一个约定）")
    print("     （写进 label_service/.env，然后 bash label_service/up.sh -u）")
    print()


# ── ② 可疑标注 ────────────────────────────────────────────────────────────


def _load_split(processed_dir, hz, split):
    p = os.path.join(processed_dir, f"{hz}hz", f"{split}.npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        classes = z["classes"]
        classes = eval(str(classes)) if classes.ndim == 0 else list(classes)
        return {"X": z["X"], "y": z["y"], "classes": list(classes)}


def _extract(X, hz):
    """按通道数选特征提取，跟训练时一致。"""
    if X.shape[2] == 5:
        import features5  # noqa: F401  (acc3/ 那条路)
        sys.path.insert(0, os.path.join(_REPO, "acc3"))
        import features5 as f5
        return f5.extract_features(X, hz, show_progress=False)
    from features import extract_features
    return np.asarray(extract_features(X, hz, show_progress=False))


def report_confusion(processed_dir, hz, model_path, remap, out_csv, top):
    print("② 模型到底把这两类搞混到什么程度，以及哪些窗口的标注可疑\n")
    val = _load_split(processed_dir, hz, "val") or _load_split(processed_dir, hz, "train")
    if val is None:
        print("   ✗ 找不到 npz，跳过")
        return
    classes = val["classes"]
    y = val["y"].astype(int)

    if remap:
        from remap_utils import apply_remap, load_remap_yaml
        y, classes, keep = apply_remap(y, classes, load_remap_yaml(remap))
        val["X"] = val["X"][keep]

    try:
        import joblib
    except ImportError:
        print("   ✗ 没装 joblib，跳过（pip install joblib scikit-learn）")
        return
    model = joblib.load(model_path)

    F = _extract(val["X"], hz)
    want = getattr(model, "n_features_in_", None)
    if want is not None and F.shape[1] != want:
        print(f"   ✗ 模型要 {want} 维，这份数据算出来 {F.shape[1]} 维——"
              "模型和 --processed_dir 不是一套的")
        return
    proba = model.predict_proba(F)
    pred = proba.argmax(axis=1)

    # 混淆矩阵
    k = len(classes)
    cm = np.zeros((k, k), int)
    for t, pr in zip(y, pred):
        cm[t, pr] += 1
    w = max(len(c) for c in classes) + 2
    print("   混淆矩阵（行=标注，列=预测）")
    print("   " + " " * w + "".join(f"{c:>8}" for c in classes) + "     召回")
    for i, c in enumerate(classes):
        tot = cm[i].sum()
        rec = cm[i, i] / tot if tot else 0.0
        print(f"   {c:<{w}}" + "".join(f"{cm[i, j]:>8}" for j in range(k))
              + f"{rec:>9.2f}")
    print()

    if SCRATCH not in classes or SHAKE not in classes:
        print(f"   （这个模型没有 {SCRATCH}/{SHAKE} 两类，后面跳过）")
        return
    si, hi = classes.index(SCRATCH), classes.index(SHAKE)
    n_sh = cm[hi].sum()
    n_sc = cm[si].sum()
    print(f"   {SHAKE} 被判成 {SCRATCH}：{cm[hi, si]} / {n_sh}"
          f"（{cm[hi, si] / n_sh * 100:.1f}%）" if n_sh else "")
    print(f"   {SCRATCH} 被判成 {SHAKE}：{cm[si, hi]} / {n_sc}"
          f"（{cm[si, hi] / n_sc * 100:.1f}%）" if n_sc else "")
    print()
    print(f"   两类的样本量：{SCRATCH} {n_sc}，{SHAKE} {n_sh}"
          f"（相差 {n_sc / max(n_sh, 1):.0f} 倍）")
    if n_sh and n_sc / n_sh > 3:
        print(f"   ⚠ **严重不均衡**。树模型在这种比例下天然偏多数类——")
        print(f"     这本身就足以让 {SHAKE} 往 {SCRATCH} 上倒，不需要标注有错。")
    print()

    # 可疑标注：模型很确信、但跟标注相反
    rows = []
    for idx in range(len(y)):
        t, pr = int(y[idx]), int(pred[idx])
        if {t, pr} != {si, hi} or t == pr:
            continue
        rows.append({
            "窗口序号": idx,
            "标注": classes[t],
            "模型判": classes[pr],
            "模型置信度": round(float(proba[idx, pr]), 3),
            "标注那类的概率": round(float(proba[idx, t]), 3),
        })
    rows.sort(key=lambda r: -r["模型置信度"])
    print(f"   模型判反了的窗口共 {len(rows)} 个。**模型越确信、标注越可疑**，"
          f"下面按置信度排前 {top} 个：")
    print(f"   {'序号':>8}{'标注':>8}{'模型判':>8}{'置信度':>9}{'标注类概率':>11}")
    for r in rows[:top]:
        print(f"   {r['窗口序号']:>8}{r['标注']:>8}{r['模型判']:>8}"
              f"{r['模型置信度']:>9.3f}{r['标注那类的概率']:>11.3f}")
    if out_csv and rows:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\n   全部 {len(rows)} 条写到了 {out_csv}")
    print()
    print("   ⚠ 这些**不等于标注错了**。模型判反的原因可能是标注错，")
    print("     也可能是这两类本来就像（看 ③）。要确认只能回去看视频——")
    print("     但按置信度排序之后，前几十个里如果真有错标，很容易一眼看出来。")
    print()


# ── ③ 采样率够不够 ────────────────────────────────────────────────────────


def _dominant_freq(sig, hz):
    from scipy import signal as sp
    freqs, psd = sp.welch(sig, fs=hz, nperseg=min(len(sig), 32))
    return float(freqs[int(np.argmax(psd))])


def report_spectrum(processed_dir, hz, remap):
    print("③ 提到 25Hz 有没有用——看两类的主频分布\n")
    d = _load_split(processed_dir, hz, "val") or _load_split(processed_dir, hz, "train")
    if d is None:
        print("   ✗ 找不到 npz，跳过")
        return
    classes, y, X = d["classes"], d["y"].astype(int), d["X"]
    if remap:
        from remap_utils import apply_remap, load_remap_yaml
        y, classes, keep = apply_remap(y, classes, load_remap_yaml(remap))
        X = X[keep]
    if SCRATCH not in classes or SHAKE not in classes:
        print("   （这个模型没有这两类，跳过）")
        return

    nyq = hz / 2.0
    # 频率分辨率 = fs / nperseg。而 nperseg = 窗口点数 = fs × 窗口秒数，
    # 约掉 fs 之后 **分辨率 = 1 / 窗口秒数**——跟采样率没关系。
    n_t = X.shape[1]
    win_s = n_t / hz
    nperseg = min(n_t, 32)
    res = hz / nperseg
    print(f"   采样率 {hz}Hz → Nyquist {nyq:g}Hz（能表示的最高频率）")
    print(f"   窗口 {n_t} 点 = {win_s:g} 秒 → **频率分辨率 {res:g}Hz**，"
          f"整条谱只有 {nperseg // 2 + 1} 个点\n")
    print("   ⚠ 这一条比 Nyquist 更要紧，而且最容易搞反：")
    print("     **频率分辨率由窗口时长决定，不由采样率决定**")
    print("     （分辨率 = fs / nperseg，而 nperseg = fs × 窗口秒数，fs 约掉了）。")
    print()
    print("     实测对照（抓挠 5.0Hz、甩身体 4.5Hz，60 个噪声种子，")
    print("     见 diagnose/test_resolution.py）：")
    print("       16Hz/1s   分辨率 1.00Hz   甩身体在 4 和 5 之间跳，跟抓挠重合")
    print("       25Hz/1s   分辨率 1.00Hz   **跟 16Hz 一模一样**，照样重合")
    print("       50Hz/1s   分辨率 1.56Hz   反而更差（见下面 nperseg 那条）")
    print("       16Hz/2s   分辨率 0.50Hz   **稳定分开**")
    print()
    print("     所以「提到 25Hz 就能分开」这个直觉是错的：1 秒窗口下两者")
    print("     分辨率完全相同。而且 4.5 卡在格子边上，**落到哪一格由噪声决定**——")
    print("     同一个动作有时报 4Hz 有时报 5Hz，模型拿到的这一维本身就是抖的。")
    print()
    print(f"     ⚠ 还有个坑：nperseg = min(窗口点数, 32)（写死在")
    print(f"       src/ml/features.py:_freq_stats_1d）。点数超过 32 就被截断，")
    print(f"       于是 50Hz/1s 只有 1.56Hz、25Hz/2s 只有 0.78Hz。")
    print(f"       **想靠加长窗口拿分辨率，得连这个上限一起放开。**")
    print()
    print(f"     采样率也不是没用：它决定能看到多高的频率。9Hz 在 16Hz 下会混叠，")
    print(f"     25Hz 下看得到。所以下面还看一条「多少窗口贴着 Nyquist」。\n")
    print(f"   {'类别':<8}{'窗口数':>8}{'主频中位数':>12}{'p90':>8}"
          f"{'  >' + format(nyq * 0.75, 'g') + 'Hz 占比':>14}")
    stats = {}
    for lab in (SCRATCH, SHAKE, "活动"):
        if lab not in classes:
            continue
        sel = np.where(y == classes.index(lab))[0]
        if not len(sel):
            continue
        # acc 三轴模长的主频：抓挠/甩身体都是躯干的往复运动，
        # 模长比单轴稳（跟项圈戴歪多少无关）
        mags = np.sqrt((X[sel, :, 0:3] ** 2).sum(axis=2))
        f = np.array([_dominant_freq(m, hz) for m in mags])
        stats[lab] = f
        near = float((f > nyq * 0.75).mean())
        print(f"   {lab:<8}{len(sel):>8}{np.median(f):>12.2f}"
              f"{np.percentile(f, 90):>8.2f}{near * 100:>13.1f}%")
    print()

    if SCRATCH in stats and SHAKE in stats:
        a, b = stats[SCRATCH], stats[SHAKE]
        # 两个分布重叠多少：用四分位区间是否相交来看
        qa = np.percentile(a, [25, 75])
        qb = np.percentile(b, [25, 75])
        overlap = max(0.0, min(qa[1], qb[1]) - max(qa[0], qb[0]))
        span = max(qa[1], qb[1]) - min(qa[0], qb[0])
        print(f"   {SCRATCH} 的中间一半落在 {qa[0]:.2f}~{qa[1]:.2f} Hz")
        print(f"   {SHAKE} 的中间一半落在 {qb[0]:.2f}~{qb[1]:.2f} Hz")
        print(f"   重叠 {overlap:.2f} Hz / 总跨度 {span:.2f} Hz"
              f"（{overlap / span * 100 if span else 0:.0f}%）\n")
        # **跨度小于一格分辨率时，上面那个百分比没有意义**：两个分布落在
        # 同一格或相邻格里，谱根本分辨不出来，算出 0% 重叠只是量化的假象
        if span <= res:
            print(f"   → 两类的主频**落在同一格分辨率里**（跨度 {span:.2f}Hz ≤ "
                  f"{res:g}Hz）。")
            print("     这不是「分得开」，是**这套谱根本分辨不出来**——")
            print(f"     上面那个重叠百分比在这种情况下没有意义。")
            print(f"     要分开只有一条路：**加长窗口**（分辨率 = 1/窗口秒数）。")
            print(f"     提采样率不会改善这一点。")
        elif span and overlap / span > 0.5:
            print(f"   → **两类的主频基本重合。提采样率不会让它们分开**——")
            print(f"     那是物理上的相似（都是躯干 4~6Hz 的往复），不是采样不够。")
            print(f"     要分开得靠别的维度：持续时长、幅度、姿态（抓挠时身体是歪的）、")
            print(f"     或者更长的窗口看节律的规整程度。")
        else:
            print(f"   → 两类主频分得开，模型却分不开，说明问题不在频率这一维。")
        near_shake = float((b > nyq * 0.75).mean())
        if near_shake > 0.15:
            print(f"\n   ⚠ 但有 {near_shake * 100:.0f}% 的{SHAKE}窗口主频压在"
                  f" {nyq * 0.75:g}Hz 以上，**贴着 Nyquist**。")
            print(f"     这一部分 {hz}Hz 确实吃亏：谱在边界附近不可靠，"
                  f"而且二次谐波整个混叠掉了。")
            print(f"     提到 25Hz（Nyquist 12.5Hz）对**这部分**窗口会有帮助。")
        else:
            print(f"\n   {SHAKE} 只有 {near_shake * 100:.0f}% 的窗口贴近 Nyquist，"
                  f"{hz}Hz 对这一类不是瓶颈。")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", help="预处理目录（②③ 要）")
    ap.add_argument("--model", help="训好的 ml_*.pkl（② 要）")
    ap.add_argument("--hz", type=int, default=16)
    ap.add_argument("--remap", default="configs/remap_custom_3class.yaml")
    ap.add_argument("--out", default="tmp/suspect_labels.csv")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    report_postprocess_rules()
    if args.processed_dir and args.model:
        report_confusion(args.processed_dir, args.hz, args.model,
                         args.remap, args.out, args.top)
    else:
        print("② 跳过（要 --processed_dir 和 --model）\n")
    if args.processed_dir:
        report_spectrum(args.processed_dir, args.hz, args.remap)
    else:
        print("③ 跳过（要 --processed_dir）\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
