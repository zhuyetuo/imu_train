"""皮肤 ML 版的可解释性：用了哪些维度、每个维度多重要、某一天为什么判成这个档。

三样东西：
  · features()：全部特征的中文名 / 分组 / 怎么算 / 有什么用（说明取自
    skin_health/docs/model_ab_feature_glossary.md），加上全局重要性
    （置换重要性：把这一列在训练数据里打乱，模型准确率掉多少——掉得多就是模型真在用它）
  · explain()：某一天的预测，逐个特征算贡献——把这一个特征换成训练数据里的典型值（中位数），
    看预测的那一档概率变多少。正的是把它往这一档推，负的是往别的档拉。
    不是 SHAP，但不用装别的包、对任何模型都能算，含义也直白
  · 这一天每个特征的值在训练数据里排第几（分位），超出训练数据范围的标出来——
    模型没见过的情况，判出来的东西不可信（比如真实狗的品种合成数据里根本没有）
"""
from __future__ import annotations

import json
import os
import re
import threading

import numpy as np
import pandas as pd

from label_service import config
from label_service import skin_ml as M

_GLOSSARY = os.path.join(config.REPO_ROOT, "skin_health", "docs", "model_ab_feature_glossary.md")
_lock = threading.Lock()
_ref_cache: dict = {}

_NAMES = {
    "event_count": "抓挠次数", "event_rate_per_wear_hour": "每佩戴小时抓几次",
    "total_duration_min": "抓挠总时长(分)", "duration_rate_per_wear_hour": "每佩戴小时抓多久",
    "duration_mean": "单次平均时长", "duration_median": "单次时长中位数", "max_event_duration_sec": "最长一次(秒)",
    "duration_std": "单次时长波动", "night_ratio": "夜里抓的占比", "interval_mean": "两次之间平均间隔",
    "interval_std": "间隔波动（扎堆程度）", "sleep_disruption_count": "睡着被抓醒次数",
    "wear_completeness_ratio": "佩戴完整度", "history_days_available": "已有几天历史", "breed_or_size_class": "品种",
    "has_any_baseline": "有没有基线", "z_score_vs_self": "比自己平时偏离多少(z)",
    "consecutive_days_above_baseline": "连续几天高于基线", "flare_episode_count_90d": "90天内发作几次",
    "days_since_last_flare_end": "上次发作结束几天了",
    "skin_redness_level": "问卷·发红", "skin_pigment_abnormal": "问卷·色素沉着", "skin_lesion_severity": "问卷·皮损",
    "odor_level": "问卷·异味", "hair_loss_spot_count_level": "问卷·秃毛几处",
    "hair_loss_max_diameter_level": "问卷·最大秃毛", "coat_quality_level": "问卷·毛发状态",
}


def name_of(f: str) -> str:
    if f in _NAMES:
        return _NAMES[f]
    m = re.match(r"baseline_ratio_(count|duration)_(excl_)?recent(\d+)", f)
    if m:
        what = "次数" if m.group(1) == "count" else "时长"
        win = f"跳过近7天再往前{m.group(3)}天" if m.group(2) else f"近{m.group(3)}天"
        return f"{what}比基线（{win}）"
    m = re.match(r"rolling_(mean|std)_(\d+)d", f)
    if m:
        return f"近{m.group(2)}天抓挠率{'均值' if m.group(1) == 'mean' else '波动'}"
    m = re.match(r"model_a_proba_(C\d)", f)
    if m:
        return f"模型A判{m.group(1)}的概率"
    return f


def group_of(f: str) -> str:
    if f.startswith("model_a_proba_"):
        return "模型A的判断"
    if f in ("skin_redness_level", "skin_pigment_abnormal", "skin_lesion_severity", "odor_level",
             "hair_loss_spot_count_level", "hair_loss_max_diameter_level", "coat_quality_level"):
        return "问卷"
    if f.startswith("baseline_ratio_") or f in ("has_any_baseline", "z_score_vs_self",
                                                "consecutive_days_above_baseline"):
        return "跟自己基线比"
    if f.startswith("rolling_") or f in ("flare_episode_count_90d", "days_since_last_flare_end"):
        return "近期趋势 / 发作史"
    if f in ("night_ratio", "interval_mean", "interval_std", "sleep_disruption_count"):
        return "时间分布 / 睡眠"
    if f in ("wear_completeness_ratio", "history_days_available", "breed_or_size_class"):
        return "数据质量 / 背景"
    return "抓挠频率 / 时长"


def glossary() -> dict[str, dict]:
    """从文档里的表格解析：特征 → {calc, role}"""
    out: dict[str, dict] = {}
    try:
        text = open(_GLOSSARY, encoding="utf-8").read()
    except OSError:
        return out
    for line in text.splitlines():
        m = re.match(r"^\|\s*`([a-zA-Z0-9_]+)`\s*\|([^|]*)\|([^|]*)\|", line)
        if m:
            out[m.group(1)] = {"calc": m.group(2).strip(), "role": m.group(3).strip()}
    return out


# ── 训练数据的参照分布 ─────────────────────────────────────────────────

def _ref_path() -> str:
    return os.path.join(M.RUN_DIR, "feature_ref.json")


def reference() -> dict:
    """每个特征在合成训练数据里的分布（中位数 / 分位点），算一次存盘。重训合成版后会删掉重算"""
    with _lock:
        if _ref_cache:
            return _ref_cache
        try:
            _ref_cache.update(json.load(open(_ref_path(), encoding="utf-8")))
            return _ref_cache
        except (OSError, ValueError):
            pass
        ta, tb, feat_cols = M._synthetic_tables()
        extras = M.load_extras()
        if extras:
            ea, eb = M._extra_tables(extras)
            ta = pd.concat([ta, ea], ignore_index=True)
            tb = pd.concat([tb, eb], ignore_index=True)
        ref: dict = {"num": {}, "breeds": sorted({str(b) for b in ta["breed_or_size_class"].dropna()})}
        qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        for table, cols in ((ta, feat_cols), (tb, [c for c in tb.columns if c.startswith(("skin_", "odor", "hair_", "coat_"))])):
            for c in cols:
                if c == "breed_or_size_class" or c not in table.columns:
                    continue
                v = pd.to_numeric(table[c], errors="coerce").dropna()
                if v.empty:
                    continue
                ref["num"][c] = {"q": [float(v.quantile(x)) for x in qs], "min": float(v.min()), "max": float(v.max()),
                                 "sorted": [float(x) for x in np.quantile(v, np.linspace(0, 1, 101))]}
        os.makedirs(M.RUN_DIR, exist_ok=True)
        with open(_ref_path(), "w", encoding="utf-8") as f:
            json.dump(ref, f)
        _ref_cache.update(ref)
        return _ref_cache


def reset() -> None:
    with _lock:
        _ref_cache.clear()
    for p in (_ref_path(), os.path.join(M.RUN_DIR, "importance.json")):
        try:
            os.remove(p)
        except OSError:
            pass


def _percentile(c: str, v) -> float | None:
    r = reference()["num"].get(c)
    if r is None or v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(np.searchsorted(r["sorted"], float(v), side="right") - 1)


# ── 全局重要性 ──────────────────────────────────────────────────────────

def importance() -> dict:
    """置换重要性（合成训练数据上，打乱一列看准确率掉多少），算一次存盘"""
    p = os.path.join(M.RUN_DIR, "importance.json")
    try:
        return json.load(open(p, encoding="utf-8"))
    except (OSError, ValueError):
        pass
    from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS
    from sklearn.inspection import permutation_importance

    a, b = M.models()
    ta, tb, feat_cols = M._synthetic_tables()
    ta = ta.sample(min(1500, len(ta)), random_state=0)
    Xa = M._cat(ta.reindex(columns=a["feature_cols"]), a["breed_categories"])
    ra = permutation_importance(a["model"], Xa, ta["tier"].values, n_repeats=3, random_state=0, n_jobs=1)
    tbx = tb.copy()
    st = M._a_proba(a, tbx)
    for c in b["stack_cols"]:
        tbx[c] = st[c] if c in st.columns else 0.0
    Xb = M._cat(tbx.reindex(columns=b["feature_cols"]), b["breed_categories"])
    rb = permutation_importance(b["model"], Xb, tbx["s_tier"].values, n_repeats=3, random_state=0, n_jobs=1)
    out = {"a": {c: round(float(x), 4) for c, x in zip(a["feature_cols"], ra.importances_mean)},
           "b": {c: round(float(x), 4) for c, x in zip(b["feature_cols"], rb.importances_mean)},
           "model_b": b.get("source")}
    os.makedirs(M.RUN_DIR, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


def features() -> dict:
    a, b = M.models()
    gl = glossary()
    imp = importance()
    cols = list(dict.fromkeys(list(a["feature_cols"]) + list(b["feature_cols"])))
    return {
        "features": [{
            "feature": c, "name": name_of(c), "group": group_of(c),
            "calc": gl.get(c, {}).get("calc"), "role": gl.get(c, {}).get("role"),
            "in_a": c in a["feature_cols"], "in_b": c in b["feature_cols"],
            "imp_a": imp["a"].get(c), "imp_b": imp["b"].get(c),
        } for c in cols],
        "breeds_seen": reference()["breeds"],
        "model_b": b.get("source"),
    }


# ── 某一天为什么这么判 ──────────────────────────────────────────────────

def _contrib(bundle: dict, X: pd.DataFrame, k: int) -> list[dict]:
    """逐个特征换成训练典型值，看预测那一档（第 k 类）的概率变多少"""
    ref = reference()["num"]
    base = bundle["model"].predict_proba(X)[0][k]
    rows, names = [], []
    for c in bundle["feature_cols"]:
        Xc = X.copy()
        if c == "breed_or_size_class":
            Xc[c] = pd.Categorical([None], categories=bundle["breed_categories"])
        elif c in ref:
            Xc[c] = ref[c]["q"][3]
        elif c.startswith("model_a_proba_"):
            Xc[c] = 1 / 3
        else:
            continue
        rows.append(Xc)
        names.append(c)
    if not rows:
        return []
    P = bundle["model"].predict_proba(pd.concat(rows, ignore_index=True))[:, k]
    return [{"feature": c, "delta": round(float(base - p), 4)} for c, p in zip(names, P)]


def explain(rows: list[dict], pet_id: str, date: str, answers: list[dict] | None = None) -> dict:
    from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS

    a, b = M.models()
    ans = {(str(x["pet_id"]), x["date"]): x for x in answers or []}
    feats = M.features(rows)
    row = feats[(feats["pet_id"] == str(pet_id)) & (feats["date"] == date)]
    if row.empty:
        return {"ok": False, "message": "这一天没有数据"}
    r = row.iloc[[0]]
    if r.iloc[0]["data_quality_flag"] != "good":
        return {"ok": False, "message": f"这天数据不完整（{r.iloc[0]['data_quality_flag']}），不出预测"}
    gl = glossary()

    Xa = M._cat(r.reindex(columns=a["feature_cols"]), a["breed_categories"])
    pa = a["model"].predict_proba(Xa)[0]
    ka = int(np.argmax(pa))
    tb = M._b_table(r, a, b, ans)
    Xb = M._cat(tb.reindex(columns=b["feature_cols"]), b["breed_categories"])
    pb = b["model"].predict_proba(Xb)[0]
    kb = int(np.argmax(pb))

    ref = reference()

    def fval(c):
        v = tb.iloc[0][c] if c in tb.columns else None
        if isinstance(v, (np.floating, float)):
            return None if np.isnan(v) else round(float(v), 3)
        if isinstance(v, (np.integer, int)):
            return int(v)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

    def detail(c):
        v = fval(c)
        rr = ref["num"].get(c)
        out = {"feature": c, "name": name_of(c), "group": group_of(c), "value": v,
               "role": gl.get(c, {}).get("role")}
        if rr and isinstance(v, (int, float)):
            out.update({"p50": round(rr["q"][3], 3), "p10": round(rr["q"][1], 3), "p90": round(rr["q"][5], 3),
                        "pct": _percentile(c, v), "out_of_range": v < rr["min"] or v > rr["max"]})
        return out

    notes = []
    breed = fval("breed_or_size_class")
    if breed and breed not in ref["breeds"]:
        notes.append(f"品种「{breed}」合成训练数据里没有（只有 {'、'.join(ref['breeds'])}），模型当成「不知道品种」处理")
    cols = list(dict.fromkeys(list(a["feature_cols"]) + list(b["feature_cols"])))
    details = {c: detail(c) for c in cols}
    oor = [d["name"] for d in details.values() if d.get("out_of_range")]
    if oor:
        notes.append(f"这些维度超出了训练数据的范围，模型没见过这种情况，结果要打折扣：{'、'.join(oor[:8])}"
                     + ("…" if len(oor) > 8 else ""))
    if not any(tb.iloc[0].get(q) == tb.iloc[0].get(q) for q in QUESTIONNAIRE_FEATURE_COLUMNS if q in tb.columns):
        notes.append("这天没有问卷：模型 B 的合成训练数据每一行都有问卷（只有 C1 / C2 触发问卷的天），"
                     "没问卷对它来说是没见过的情况，基本只会给一个中间档 S1——没问卷的天看模型 A 的 C 档更有意义；"
                     "要让模型 B 在没问卷时也能判，得用兽医照片评级重训（真实数据里大部分天没问卷）")

    def top(contrib):
        cs = sorted(contrib, key=lambda x: -abs(x["delta"]))
        return [{**details.get(x["feature"], {"feature": x["feature"], "name": name_of(x["feature"])}), "delta": x["delta"]}
                for x in cs if abs(x["delta"]) >= 0.005][:12]

    return {
        "ok": True, "pet_id": pet_id, "date": date,
        "a": {"tier": str(a["classes"][ka]), "proba": M._proba_dict(a["classes"], pa), "top": top(_contrib(a, Xa, ka))},
        "b": {"tier": str(b["classes"][kb]), "proba": M._proba_dict(b["classes"], pb), "top": top(_contrib(b, Xb, kb)),
              "source": b.get("source")},
        "features": list(details.values()),
        "notes": notes,
    }
