"""训练完的诊断：指标是什么意思、错都错成了什么、容易混的两类到底分不分得开。

要回答的是「效果不好的类别该怎么办」：
  · 补数据？——如果两类在现有特征下本来就分得开，多半是数据量 / 标注质量 / 多类模型被大类拖累，
    补干净的数据有用；
  · 改策略？——如果两类在现有特征下基本分不开（单独训一个只分这两类的模型都分不开），
    一直补数据也没用，要么合并类别 / 改类别定义，要么加特征 / 换模型 / 换采样率或传感器。

「分不分得开」的量法：只拿这两类，用**训练集**训一个二分类，在**验证集**上算 AUC
（验证集是按片段划分的，同一段的窗口不会两边都有，不会因为相邻窗口高度重叠而虚高）。
再按单个特征算 AUC，列出最能区分这两类的几个特征——看是哪方面的差异在起作用。
"""

from __future__ import annotations

import numpy as np

LEGEND = """  ── 指标怎么看 ──
  precision（查准率）：模型判成这一类的窗口里，真是这一类的比例。低 = 别的类常被错认成它（误报多）
  recall   （查全率）：真是这一类的窗口里，被模型认出来的比例。低 = 这一类常被认成别的（漏报多）
  f1-score          ：precision 和 recall 的调和平均，两个都高它才高
  support           ：验证集里这一类的窗口数。少于两三百时数字很不稳，看个大概
  accuracy          ：全部窗口里判对的比例。会被窗口多的大类主导
  macro avg         ：各类直接平均，不看窗口多少——小类跟大类一样重，反映「每一类都认得怎么样」
  weighted avg      ：按窗口数加权平均，接近 accuracy
  一类的 recall 低、另一类的 precision 低，多半是前者被大量认成了后者——看下面的混淆矩阵"""


def confusion(y_true, y_pred, labels: list[int], names: list[str]) -> tuple[str, dict]:
    """混淆矩阵（行 = 人标的，列 = 模型判的），每行再折成百分比；列出最大的几处混淆。"""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    w = max(6, max(len(n) for n in names) + 2)
    lines = ["  ── 混淆矩阵（行 = 人标的，列 = 模型判的；括号里是这一行的百分比）──",
             "  " + "人标 \\ 判成".ljust(w + 2) + "".join(n.rjust(w + 8) for n in names)]
    for i, n in enumerate(names):
        tot = cm[i].sum() or 1
        cells = "".join(f"{cm[i, j]:>{w}}({cm[i, j] / tot * 100:3.0f}%)" for j in range(len(names)))
        lines.append("  " + n.ljust(w + 2) + cells)
    off = [(int(cm[i, j]), names[i], names[j], cm[i, j] / (cm[i].sum() or 1))
           for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j] > 0]
    off.sort(reverse=True)
    if off:
        lines.append("  最大的几处混淆：")
        for n, a, b, frac in off[:5]:
            lines.append(f"    {a} → 判成 {b}：{n} 个窗口（{a} 的 {frac * 100:.0f}%）")
    return "\n".join(lines), {"labels": names, "matrix": cm.tolist(),
                              "top": [{"human": a, "pred": b, "n": n, "frac": round(float(f), 3)} for n, a, b, f in off[:10]]}


def confused_pairs(cm: np.ndarray, names: list[str], top: int = 3, min_n: int = 10) -> list[tuple[int, int]]:
    """两个方向加起来混得最多的几对（下标），太少的不看"""
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n = int(cm[i, j] + cm[j, i])
            if n >= min_n:
                pairs.append((n, i, j))
    pairs.sort(reverse=True)
    return [(i, j) for _n, i, j in pairs[:top]]


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def verdict(auc: float) -> str:
    if not np.isfinite(auc):
        return "算不了（验证集里这两类有一类没有）"
    if auc >= 0.9:
        return "分得开：现有特征够用。效果差多半是数据量 / 标注不一致 / 多类模型被大类拖累——先查标注、再补数据"
    if auc >= 0.75:
        return "能分但吃力：补干净的数据有帮助，再考虑加针对性的特征"
    return "基本分不开：现有特征下这两类长得差不多，一直补数据作用有限——考虑合并 / 改类别定义、加特征、换采样率或传感器"


def pair_separability(X_tr, y_tr, X_ev, y_ev, label_a: int, label_b: int, name_a: str, name_b: str,
                      feat_names: list[str] | None = None, n_jobs: int = 2, max_per_class: int = 20000,
                      seed: int = 0) -> dict:
    """只拿 a、b 两类：训练集上训二分类 RF，验证集上算 AUC；再按单特征 AUC 列出最能区分的几个。"""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)

    def take(X, y, lab, cap):
        idx = np.flatnonzero(y == lab)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        return X[idx]

    A_tr, B_tr = take(X_tr, y_tr, label_a, max_per_class), take(X_tr, y_tr, label_b, max_per_class)
    A_ev, B_ev = take(X_ev, y_ev, label_a, max_per_class), take(X_ev, y_ev, label_b, max_per_class)
    out = {"a": name_a, "b": name_b, "n_train": [len(A_tr), len(B_tr)], "n_eval": [len(A_ev), len(B_ev)]}
    if min(len(A_tr), len(B_tr)) < 5:
        out.update(auc=None, verdict="训练集里有一类太少，算不了", top_features=[])
        return out
    Xt = np.vstack([A_tr, B_tr])
    yt = np.r_[np.zeros(len(A_tr)), np.ones(len(B_tr))]
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced",
                                 n_jobs=n_jobs, random_state=seed).fit(Xt, yt)
    auc = float("nan")
    if min(len(A_ev), len(B_ev)) >= 1:
        Xe = np.vstack([A_ev, B_ev])
        ye = np.r_[np.zeros(len(A_ev)), np.ones(len(B_ev))]
        auc = _auc(ye, clf.predict_proba(Xe)[:, 1])
    # 单个特征的区分度（训练集上）：AUC 离 0.5 越远越有用；>0.5 表示 b 类的值更大
    feats = []
    for k in range(Xt.shape[1]):
        a = _auc(yt, Xt[:, k])
        if np.isfinite(a):
            feats.append((abs(a - 0.5), k, a))
    feats.sort(reverse=True)
    top = []
    for _d, k, a in feats[:6]:
        nm = feat_names[k] if feat_names and k < len(feat_names) else f"#{k}"
        top.append({"feature": nm, "auc": round(a, 3), "higher": name_b if a > 0.5 else name_a,
                    "mean_a": round(float(np.mean(A_tr[:, k])), 4), "mean_b": round(float(np.mean(B_tr[:, k])), 4)})
    out.update(auc=None if not np.isfinite(auc) else round(auc, 3), verdict=verdict(auc), top_features=top)
    return out


def separability_text(res: list[dict]) -> str:
    if not res:
        return ""
    lines = ["  ── 容易混的两类，单独看分不分得开（训练集训二分类、验证集算 AUC；0.5 = 瞎猜，1 = 完全分开）──"]
    for r in res:
        auc = "-" if r.get("auc") is None else f"{r['auc']:.3f}"
        lines.append(f"  {r['a']} vs {r['b']}：AUC {auc}（训练 {r['n_train'][0]}/{r['n_train'][1]} 窗，"
                     f"验证 {r['n_eval'][0]}/{r['n_eval'][1]} 窗）")
        lines.append(f"    → {r['verdict']}")
        if r.get("top_features"):
            lines.append("    最能区分它俩的特征（单特征 AUC，越接近 0 或 1 越有用）：")
            for f in r["top_features"][:5]:
                lines.append(f"      {f['feature']:<26} AUC {f['auc']:.2f}  {f['higher']} 更大"
                             f"（均值 {r['a']} {f['mean_a']} / {r['b']} {f['mean_b']}）")
    return "\n".join(lines)
