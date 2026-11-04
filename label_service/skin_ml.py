"""皮肤 ML 版（模型 A 预测 C 档、模型 B 预测 S 档）—— 给 label_infra 整批调用的那一层。

跟 skin.py 里老的 ml_scan / ml_predict 的区别：
  · 输入不再自己去 NAS 扫 _infer 目录，而是 label_infra「项目联动」已经整理好存库的
    每天抓挠事件（两个项圈合成一行、只算戴着的那段、人工核对过的片段都在里面），
    一次把一只狗的全部历史送过来——特征里有基线 / 滚动均值，只给一天算不准
  · 一次算一批：每只狗特征只算一遍，所有天一起预测
  · 模型文件读不出来（训练时的 scikit-learn 版本跟现在装的不一样，pickle 不兼容）
    就用仓库里的合成数据当场重训一份，不再整个 ML 版用不了
  · 兽医照片评级攒够之后，可以用真实标签重训模型 B（train_b_real），交叉验证
    比 PM 规则准才启用（activate_b）

模型 A 的标签就是 PM 的抓挠规则算出来的 C 档，本质是 PM 规则的近似；真正有新信息的是
模型 B——它预测的 S0 / S1 / S2 跟兽医看照片评的档一一对应，所以真实标签只用来重训 B。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime

import numpy as np
import pandas as pd

from label_service import config
from label_service import skin_rules as R

_CODE = os.path.join(config.REPO_ROOT, "skin_health", "code")
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

_logger = logging.getLogger(__name__)

SYN_DIR = os.path.join(config.REPO_ROOT, "skin_health", "data", "rf_synthetic")
# 运行时产出（自愈重训的合成模型、真实标签训出来的模型 B、报告）。不进 git
RUN_DIR = os.environ.get("SKIN_ML_DIR") or os.path.join(config.REPO_ROOT, "skin_health", "data", "runtime")
_ACTIVE_FILE = "active.json"

_lock = threading.Lock()
_cache: dict[str, dict] = {}

_HGB = dict(max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=42)


def _ts(s: str) -> datetime:
    s = s.strip()
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f" if "." in s else "%Y-%m-%d %H:%M:%S")


# ── 模型加载 / 自愈 ──────────────────────────────────────────────────────

def _load(path: str) -> dict | None:
    import joblib
    import warnings

    if not os.path.exists(path):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = joblib.load(path)
        # 版本不兼容有时能读出来但一用就炸：拿一行全 NaN 试一下
        X = pd.DataFrame([{c: np.nan for c in b["feature_cols"]}])
        X["breed_or_size_class"] = pd.Categorical([None], categories=b["breed_categories"])
        b["model"].predict_proba(X)
        return b
    except Exception as e:  # noqa: BLE001
        _logger.warning("模型 %s 读不出来（多半是 scikit-learn 版本不一样）：%s", path, e)
        return None


def _fit_hgb(X: pd.DataFrame, y, sample_weight=None):
    from sklearn.ensemble import HistGradientBoostingClassifier

    m = HistGradientBoostingClassifier(categorical_features=[X.columns.get_loc("breed_or_size_class")], **_HGB)
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _usable_cols(table: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in table.columns and not table[c].isna().all()]


def _cat(X: pd.DataFrame, cats=None) -> pd.DataFrame:
    X = X.copy()
    X["breed_or_size_class"] = (pd.Categorical(X["breed_or_size_class"], categories=cats) if cats is not None
                                else X["breed_or_size_class"].astype("category"))
    return X


def _synthetic_tables():
    """合成数据的模型 A 训练表 + 模型 B 训练表（B 表里的 A 概率用 A 当场算）"""
    from rf_features import FEATURE_COLUMNS, compute_rf_features
    from scratch_burden import run_pipeline

    ev = pd.read_csv(os.path.join(SYN_DIR, "all_events.csv"), parse_dates=["start", "end"])
    wear = pd.read_csv(os.path.join(SYN_DIR, "all_wear.csv"), parse_dates=["date"])
    wear["date"] = wear["date"].dt.date
    breed_map = json.load(open(os.path.join(SYN_DIR, "breed_map.json"), encoding="utf-8"))
    feats = compute_rf_features(ev, wear, breed_map)
    labels = run_pipeline(ev, wear)[["pet_id", "date", "tier"]]
    ta = feats.merge(labels, on=["pet_id", "date"], how="inner")
    ta = ta[ta["tier"] != "insufficient_data"].reset_index(drop=True)
    tb = pd.read_csv(os.path.join(SYN_DIR, "model_b_training_table.csv"))
    return ta, tb, FEATURE_COLUMNS


def _train_synthetic(with_report: bool = False) -> tuple[dict, dict]:
    """用仓库里的合成数据 + 页面上加进来的模拟场景训一份 A / B（不算特征重要性，几十秒）。存到 RUN_DIR"""
    import joblib
    from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS

    ta, tb, FEATURE_COLUMNS = _synthetic_tables()
    extras = load_extras()
    if extras:
        ea, eb = _extra_tables(extras)
        ta = pd.concat([ta, ea], ignore_index=True)
        tb = pd.concat([tb, eb], ignore_index=True)
    report = _synthetic_cv(ta, tb, FEATURE_COLUMNS, QUESTIONNAIRE_FEATURE_COLUMNS) if with_report else None
    cols_a = _usable_cols(ta, FEATURE_COLUMNS)
    Xa = _cat(ta[cols_a])
    model_a = _fit_hgb(Xa, ta["tier"].values)
    a = {"model": model_a, "feature_cols": cols_a, "breed_categories": list(Xa["breed_or_size_class"].cat.categories),
         "classes": list(model_a.classes_), "source": "synthetic"}
    b = _fit_b(tb, a, QUESTIONNAIRE_FEATURE_COLUMNS, FEATURE_COLUMNS)
    os.makedirs(RUN_DIR, exist_ok=True)
    joblib.dump(a, os.path.join(RUN_DIR, "model_a.joblib"))
    joblib.dump(b, os.path.join(RUN_DIR, "model_b_synthetic.joblib"))
    rep = {"trained_at": datetime.now().isoformat(timespec="seconds"), "rows_a": int(len(ta)), "rows_b": int(len(tb)),
           "dogs_a": int(ta["pet_id"].nunique()), "extra_scenarios": len(extras),
           "c_counts": {k: int(v) for k, v in ta["tier"].value_counts().items()},
           "s_counts": {k: int(v) for k, v in tb["s_tier"].value_counts().items()}}
    if report:
        rep.update(report)
    with open(os.path.join(RUN_DIR, "synthetic_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    return a, b


def _synthetic_cv(ta, tb, feat_cols, q_cols) -> dict:
    """按场景狗分组 3 折：模型 A 跟规则真值、模型 B 跟合成 S 真值各对多少（只说明在合成数据上学得像不像）"""
    from sklearn.model_selection import GroupKFold

    def cv(table, label, fit_pred):
        y = table[label].values
        pred = np.empty(len(table), dtype=object)
        for tr, te in GroupKFold(n_splits=3).split(table, y, table["pet_id"]):
            pred[te] = fit_pred(table.iloc[tr], table.iloc[te])
        return round(float((pred == y).mean()), 3)

    cols_a = _usable_cols(ta, feat_cols)

    def fa(tr, te):
        m = _fit_hgb(_cat(tr[cols_a]), tr["tier"].values)
        return m.predict(_cat(te[cols_a], list(_cat(tr[cols_a])["breed_or_size_class"].cat.categories)))

    def fb(tr, te):
        a = {"model": _fit_hgb(_cat(tr.reindex(columns=cols_a)), tr["c_tier"].values), "feature_cols": cols_a,
             "breed_categories": list(_cat(tr[cols_a])["breed_or_size_class"].cat.categories)}
        a["classes"] = list(a["model"].classes_)
        b = _fit_b(tr, a, q_cols, feat_cols)
        tt = te.copy()
        st = _a_proba(a, tt)
        for c in b["stack_cols"]:
            tt[c] = st[c] if c in st.columns else 0.0
        return np.array(b["classes"])[b["model"].predict_proba(_cat(tt.reindex(columns=b["feature_cols"]), b["breed_categories"])).argmax(axis=1)]

    return {"cv_acc_a": cv(ta, "tier", fa), "cv_acc_b": cv(tb, "s_tier", fb)}


# ── 模拟：按参数造一只狗的抓挠历史，看模型怎么判 ─────────────────────────

_SIM_KEYS = ("scenario_id", "breed", "n_days", "pattern", "severity_multiplier", "onset_day", "peak_day",
             "recover_day", "spike_day", "cycle_period_days", "n_cycles", "baseline_events_per_day",
             "baseline_duration_sec_min", "baseline_duration_sec_max", "night_bias", "clustering_bias",
             "sleep_interrupt_injection", "long_scratch_injection", "wear_quality", "noise_dispersion",
             "questionnaire_behavior", "description")


def scenario_templates() -> list[dict]:
    with open(os.path.join(SYN_DIR, "scenarios.json"), encoding="utf-8") as f:
        base = json.load(f)["scenarios"]
    return [{k: x.get(k) for k in _SIM_KEYS if x.get(k) is not None} for x in base]


def _norm_scenario(s: dict) -> dict:
    s = {k: v for k, v in s.items() if k in _SIM_KEYS and v is not None and v != ""}
    s.setdefault("scenario_id", f"custom_{datetime.now():%Y%m%d%H%M%S}")
    s.setdefault("breed", "比熊")
    s["n_days"] = max(7, min(120, int(s.get("n_days") or 35)))
    s.setdefault("pattern", "stable")
    s["severity_multiplier"] = float(s.get("severity_multiplier") or 1.0)
    for k in ("onset_day", "peak_day", "recover_day", "spike_day", "cycle_period_days", "n_cycles"):
        if k in s:
            s[k] = int(s[k])
    if s["pattern"] in ("step_onset", "ramp_onset", "spike_recover", "gradual_decline_recovery", "cyclical"):
        s.setdefault("onset_day", s["n_days"] // 2)
    if s["pattern"] == "ramp_onset":
        s.setdefault("peak_day", min(s["n_days"] - 1, s["onset_day"] + 7))
    if s["pattern"] in ("spike_recover", "gradual_decline_recovery"):
        s.setdefault("recover_day", min(s["n_days"] - 1, s["onset_day"] + 7))
    if s["pattern"] == "single_day_spike":
        s.setdefault("spike_day", s["n_days"] - 3)
    if s["pattern"] == "cyclical":
        s.setdefault("cycle_period_days", 6)
        s.setdefault("n_cycles", 3)
    s.setdefault("questionnaire_behavior", "answered_consistent")
    return s


def _sim_tables(s: dict):
    """一个场景 → (事件, 佩戴, 特征+规则真值, 问卷/合成 S 真值)"""
    from gen_rf_synthetic_scenarios import simulate_scenario
    from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS, simulate_questionnaire_row, true_s_tier
    from rf_features import compute_rf_features
    from scratch_burden import run_pipeline

    ev, wear = simulate_scenario(s)
    feats = compute_rf_features(ev, wear, {s["scenario_id"]: s["breed"]})
    labels = run_pipeline(ev, wear)[["pet_id", "date", "tier", "red_flags"]]
    t = feats.merge(labels, on=["pet_id", "date"], how="left")
    import hashlib

    rng = np.random.default_rng(int(hashlib.sha256(s["scenario_id"].encode()).hexdigest(), 16) % (2 ** 32))
    answered = s.get("questionnaire_behavior") in ("answered_consistent", "answered_conflicting_worse",
                                                    "answered_conflicting_better")
    qrows = []
    for _, r in t.iterrows():
        q = {}
        if answered and r["tier"] in ("C1", "C2"):
            red = isinstance(r["red_flags"], list) and len(r["red_flags"]) > 0
            c_ord = {"C1": 1, "C2": 2}[r["tier"]]
            q = simulate_questionnaire_row(rng, c_ord, s["questionnaire_behavior"], red)
            q["s_tier"] = true_s_tier(c_ord, q.pop("_true_skin_latent"))
        qrows.append({c: q.get(c, np.nan) for c in QUESTIONNAIRE_FEATURE_COLUMNS} | {"s_tier": q.get("s_tier")})
    q = pd.DataFrame(qrows, index=t.index)
    return ev, wear, t, q


def simulate(s: dict) -> dict:
    """按参数造一只狗，每天：抓挠次数 / 时长、规则算的 C（真值）、模型 A 的 C、
    合成 S 真值（触发问卷的天才有）、模型 B 的 S（有问卷的天用上问卷）"""
    from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS

    s = _norm_scenario(s)
    a, b = models()
    ev, wear, t, q = _sim_tables(s)
    good = t["data_quality_flag"] == "good"
    out_days = []
    pa = a["model"].predict_proba(_cat(t.reindex(columns=a["feature_cols"]), a["breed_categories"]))
    tb = t.copy()
    st = _a_proba(a, tb)
    for c in b["stack_cols"]:
        tb[c] = st[c] if c in st.columns else 0.0
    for c in QUESTIONNAIRE_FEATURE_COLUMNS:
        tb[c] = q[c]
    pb = b["model"].predict_proba(_cat(tb.reindex(columns=b["feature_cols"]), b["breed_categories"]))
    for i, (_, r) in enumerate(t.iterrows()):
        ok = bool(good.iloc[i])
        out_days.append({
            "date": str(r["date"]), "day": i, "events": int(r.get("event_count") or 0),
            "duration_min": round(float(r.get("total_duration_min") or 0), 1),
            "wear_h": round(float(r.get("valid_wear_hours") or 0), 1), "quality": r["data_quality_flag"],
            "truth_c": r["tier"] if isinstance(r["tier"], str) else None,
            "ml_c": str(a["classes"][int(np.argmax(pa[i]))]) if ok else None,
            "truth_s": q["s_tier"].iloc[i] if isinstance(q["s_tier"].iloc[i], str) else None,
            "ml_s": str(b["classes"][int(np.argmax(pb[i]))]) if ok else None,
            "has_q": bool(q[QUESTIONNAIRE_FEATURE_COLUMNS].iloc[i].notna().any()),
        })

    def acc(k1, k2):
        pairs = [(d[k1], d[k2]) for d in out_days if d[k1] in ("C0", "C1", "C2", "S0", "S1", "S2") and d[k2]]
        return {"n": len(pairs), "acc": round(sum(x == y for x, y in pairs) / len(pairs), 3) if pairs else None}

    return {"scenario": s, "days": out_days, "acc_c": acc("truth_c", "ml_c"), "acc_s": acc("truth_s", "ml_s"),
            "model_b": b.get("source")}


# ── 页面上加进训练集的模拟场景（「补数据再训练」）────────────────────────

def _extras_path() -> str:
    return os.path.join(RUN_DIR, "extra_scenarios.json")


def load_extras() -> list[dict]:
    try:
        return json.load(open(_extras_path(), encoding="utf-8"))
    except (OSError, ValueError):
        return []


def add_extra(s: dict) -> list[dict]:
    s = _norm_scenario(s)
    xs = [x for x in load_extras() if x["scenario_id"] != s["scenario_id"]] + [s]
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(_extras_path(), "w", encoding="utf-8") as f:
        json.dump(xs, f, ensure_ascii=False, indent=1)
    return xs


def remove_extra(scenario_id: str) -> list[dict]:
    xs = [x for x in load_extras() if x["scenario_id"] != scenario_id]
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(_extras_path(), "w", encoding="utf-8") as f:
        json.dump(xs, f, ensure_ascii=False, indent=1)
    return xs


def _extra_tables(extras: list[dict]):
    """加进来的场景 → 模型 A 训练行（规则真值）+ 模型 B 训练行（触发问卷的天）"""
    from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS

    ta, tb = [], []
    for s in extras:
        s = _norm_scenario(s)
        _, _, t, q = _sim_tables(s)
        t = t[t["tier"].isin(["C0", "C1", "C2"])]
        ta.append(t.drop(columns=["red_flags"]))
        qb = pd.concat([t, q.loc[t.index]], axis=1)
        qb = qb[qb["s_tier"].notna()].copy()
        qb["c_tier"] = qb["tier"]
        tb.append(qb[[c for c in qb.columns if c not in ("tier", "red_flags")]])
    cols_q = QUESTIONNAIRE_FEATURE_COLUMNS
    eb = pd.concat(tb, ignore_index=True) if tb else pd.DataFrame(columns=list(cols_q) + ["s_tier"])
    return pd.concat(ta, ignore_index=True), eb


def retrain_synthetic() -> dict:
    """合成数据（含页面上加进来的场景）重训模型 A 和合成版模型 B，带 3 折交叉验证"""
    with _lock:
        _cache.clear()
        _train_synthetic(with_report=True)
    reset_cache()
    models()
    return status()


def _a_proba(a: dict, table: pd.DataFrame) -> pd.DataFrame:
    X = _cat(table.reindex(columns=a["feature_cols"]), a["breed_categories"])
    p = a["model"].predict_proba(X)
    return pd.DataFrame(p, columns=[f"model_a_proba_{c}" for c in a["classes"]], index=table.index)


def _fit_b(table: pd.DataFrame, a: dict, q_cols: list[str], feat_cols: list[str], sample_weight=None,
           source: str = "synthetic") -> dict:
    table = table.copy()
    stack = _a_proba(a, table)
    for c in stack.columns:
        table[c] = stack[c]
    cols = _usable_cols(table, list(feat_cols) + list(stack.columns) + list(q_cols))
    X = _cat(table[cols])
    m = _fit_hgb(X, table["s_tier"].values, sample_weight)
    return {"model": m, "feature_cols": cols, "stack_cols": list(stack.columns),
            "breed_categories": list(X["breed_or_size_class"].cat.categories), "classes": list(m.classes_),
            "source": source}


def _active() -> dict:
    try:
        return json.load(open(os.path.join(RUN_DIR, _ACTIVE_FILE), encoding="utf-8"))
    except (OSError, ValueError):
        return {"model_b": "synthetic"}


def models() -> tuple[dict, dict]:
    """(模型 A, 当前启用的模型 B)。读不出来就当场用合成数据重训"""
    with _lock:
        if "a" in _cache and "b" in _cache:
            return _cache["a"], _cache["b"]
        a = _load(os.path.join(RUN_DIR, "model_a.joblib")) or _load(os.path.join(SYN_DIR, "model_a.joblib"))
        b_syn = _load(os.path.join(RUN_DIR, "model_b_synthetic.joblib")) or _load(os.path.join(SYN_DIR, "model_b.joblib"))
        if a is None or b_syn is None:
            _logger.info("皮肤 ML 模型读不出来，用合成数据重训一份……")
            a, b_syn = _train_synthetic()
        b = b_syn
        if _active().get("model_b") == "real":
            b = _load(os.path.join(RUN_DIR, "model_b_real.joblib")) or b_syn
        _cache.update(a=a, b=b, b_syn=b_syn)
        return a, b


def reset_cache() -> None:
    with _lock:
        _cache.clear()


# ── 特征 ────────────────────────────────────────────────────────────────

def _frames(rows: list[dict]):
    """rows: [{pet_id, date, events:[[起,止],...], wear_seconds, breed}] → (events, wear, breed_map)"""
    ev, wear, breed = [], [], {}
    for r in rows:
        pid = str(r["pet_id"])
        breed.setdefault(pid, r.get("breed") or "未知")
        for s, e in r.get("events") or []:
            try:
                a, b = _ts(s), _ts(e)
            except (ValueError, AttributeError):
                continue
            if b > a:
                ev.append({"pet_id": pid, "start": a, "end": b, "duration_sec": (b - a).total_seconds()})
        wear.append({"pet_id": pid, "date": datetime.strptime(r["date"], "%Y-%m-%d").date(),
                     "valid_wear_hours": round(float(r.get("wear_seconds") or 0) / 3600, 2)})
    events = pd.DataFrame(ev, columns=["pet_id", "start", "end", "duration_sec"])
    events["start"] = pd.to_datetime(events["start"])
    events["end"] = pd.to_datetime(events["end"])
    wear_df = pd.DataFrame(wear, columns=["pet_id", "date", "valid_wear_hours"])
    wear_df = wear_df.groupby(["pet_id", "date"], as_index=False)["valid_wear_hours"].max()
    return events, wear_df, breed


def features(rows: list[dict]) -> pd.DataFrame:
    from rf_features import compute_rf_features

    events, wear, breed = _frames(rows)
    if wear.empty:
        return pd.DataFrame()
    f = compute_rf_features(events, wear, breed)
    f["date"] = f["date"].astype(str)
    return f


def _ordinals(answers: dict | None) -> dict:
    a = answers or {}
    return R._pm_answers_to_rf_ordinals(a.get("color"), a.get("odor"), a.get("lesion"), a.get("hair_spot"),
                                        a.get("hair_diameter"), a.get("coat"))


def _b_table(feats: pd.DataFrame, a: dict, b: dict, answers: dict) -> pd.DataFrame:
    from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS

    t = feats.copy()
    stack = _a_proba(a, t)
    for c in b["stack_cols"]:
        t[c] = stack[c] if c in stack.columns else 0.0
    for q in QUESTIONNAIRE_FEATURE_COLUMNS:
        t[q] = [(_ordinals(answers.get((p, d))) or {}).get(q, np.nan) for p, d in zip(t["pet_id"], t["date"])]
    return t


def _proba_dict(classes, p) -> dict:
    return {str(c): round(float(x), 3) for c, x in zip(classes, p)}


# ── 整批预测 ────────────────────────────────────────────────────────────

def predict_batch(rows: list[dict], answers: list[dict] | None = None) -> dict:
    """rows：一只或几只狗的全部历史 [{pet_id, date, events, wear_seconds, breed}]。
    answers：[{pet_id, date, color, odor, ...}]，有问卷的那天模型 B 会多用上问卷。
    返回 {rows: [{pet_id, date, c_tier, c_proba, s_tier, s_proba, used_questionnaire, reason}], model_b: 来源}"""
    a, b = models()
    ans = {(str(x["pet_id"]), x["date"]): x for x in answers or []}
    feats = features(rows)
    out = []
    if feats.empty:
        return {"rows": out, "model_b": b.get("source")}
    good = feats[feats["data_quality_flag"] == "good"]
    pa = a["model"].predict_proba(_cat(good.reindex(columns=a["feature_cols"]), a["breed_categories"])) if len(good) else []
    tb = _b_table(good, a, b, ans) if len(good) else good
    pb = b["model"].predict_proba(_cat(tb.reindex(columns=b["feature_cols"]), b["breed_categories"])) if len(good) else []
    by = {(p, d): i for i, (p, d) in enumerate(zip(good["pet_id"], good["date"]))}
    for p, d, flag in zip(feats["pet_id"], feats["date"], feats["data_quality_flag"]):
        i = by.get((p, d))
        if i is None:
            out.append({"pet_id": p, "date": d, "c_tier": None, "s_tier": None,
                        "reason": f"这天数据不完整（{flag}），跟 PM 版一样不出正式预测"})
            continue
        out.append({
            "pet_id": p, "date": d,
            "c_tier": str(a["classes"][int(np.argmax(pa[i]))]), "c_proba": _proba_dict(a["classes"], pa[i]),
            "s_tier": str(b["classes"][int(np.argmax(pb[i]))]), "s_proba": _proba_dict(b["classes"], pb[i]),
            "used_questionnaire": bool(_ordinals(ans.get((p, d)))),
        })
    return {"rows": out, "model_b": b.get("source")}


# ── 用兽医评级重训模型 B ────────────────────────────────────────────────

MIN_REAL = 30


def train_b_real(rows: list[dict], labels: list[dict], answers: list[dict] | None = None,
                 synthetic_weight: float = 0.2) -> dict:
    """rows：参与的狗的全部历史（同 predict_batch）；labels：[{pet_id, date, s_tier, pm_s_tier}]
    —— s_tier 是兽医看照片评的档（S0/S1/S2），pm_s_tier 是 PM 规则那天算的 S 档（拿来比）。

    合成数据打底（权重 synthetic_weight），真实标签为主（权重 1）。按狗分组交叉验证：
    每折拿几只狗的真实标签当考题，用合成 + 其余狗的真实标签训练，看它跟兽医对得上多少；
    同一批考题上也算 PM 规则、现在的合成版模型 B 各对多少，三个一起报。
    训练出来的存成 model_b_real.joblib，**不自动启用**：比 PM 准再在页面上点「启用」。"""
    import joblib
    from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS
    from rf_features import FEATURE_COLUMNS
    from sklearn.model_selection import GroupKFold

    a, _ = models()
    b_syn = _cache.get("b_syn")
    lab = pd.DataFrame(labels)
    if lab.empty:
        return {"ok": False, "message": "没有兽医评过的天"}
    lab = lab[lab["s_tier"].isin(["S0", "S1", "S2"])]
    lab["pet_id"] = lab["pet_id"].astype(str)
    n_dogs = lab["pet_id"].nunique()
    if len(lab) < MIN_REAL or n_dogs < 3 or lab["s_tier"].nunique() < 2:
        return {"ok": False, "message": f"真实标签还不够：现在 {len(lab)} 天 / {n_dogs} 只狗 / "
                                        f"{lab['s_tier'].nunique()} 种档位，至少要 {MIN_REAL} 天、3 只狗、2 种档位"}
    ans = {(str(x["pet_id"]), x["date"]): x for x in answers or []}
    feats = features(rows)
    real = feats.merge(lab, on=["pet_id", "date"], how="inner")
    real = real[real["data_quality_flag"] == "good"].reset_index(drop=True)
    if len(real) < MIN_REAL:
        return {"ok": False, "message": f"评过的天里佩戴数据完整的只有 {len(real)} 天，至少要 {MIN_REAL} 天"}
    for q in QUESTIONNAIRE_FEATURE_COLUMNS:
        real[q] = [(_ordinals(ans.get((p, d))) or {}).get(q, np.nan) for p, d in zip(real["pet_id"], real["date"])]

    syn = pd.read_csv(os.path.join(SYN_DIR, "model_b_training_table.csv"))
    syn["pet_id"] = "syn_" + syn["pet_id"].astype(str)

    def fit(real_part: pd.DataFrame) -> dict:
        t = pd.concat([syn, real_part], ignore_index=True)
        w = np.r_[np.full(len(syn), synthetic_weight), np.ones(len(real_part))]
        return _fit_b(t, a, QUESTIONNAIRE_FEATURE_COLUMNS, FEATURE_COLUMNS, sample_weight=w, source="real")

    def pred(b: dict, t: pd.DataFrame) -> np.ndarray:
        tt = t.copy()
        stack = _a_proba(a, tt)
        for c in b["stack_cols"]:
            tt[c] = stack[c] if c in stack.columns else 0.0
        X = _cat(tt.reindex(columns=b["feature_cols"]), b["breed_categories"])
        return np.array(b["classes"])[b["model"].predict_proba(X).argmax(axis=1)]

    y = real["s_tier"].values
    n_folds = min(5, n_dogs)
    pred_real = np.empty(len(real), dtype=object)
    for tr, te in GroupKFold(n_splits=n_folds).split(real, y, real["pet_id"]):
        pred_real[te] = pred(fit(real.iloc[tr]), real.iloc[te])
    pred_syn = pred(b_syn, real) if b_syn else np.array([None] * len(real))
    pm = real.get("pm_s_tier", pd.Series([None] * len(real))).values

    def score(p) -> dict:
        ok = [pi is not None and pi == yi for pi, yi in zip(p, y)]
        n = sum(1 for pi in p if pi is not None)
        return {"n": int(n), "acc": round(sum(ok) / n, 3) if n else None, "confusion": _confusion(y, p)}

    report = {
        "n_days": int(len(real)), "n_dogs": int(n_dogs),
        "label_counts": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
        "folds": n_folds, "synthetic_weight": synthetic_weight,
        "real": score(pred_real), "synthetic": score(pred_syn), "pm": score(pm),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    final = fit(real)
    final["report"] = report
    os.makedirs(RUN_DIR, exist_ok=True)
    joblib.dump(final, os.path.join(RUN_DIR, "model_b_real.joblib"))
    with open(os.path.join(RUN_DIR, "model_b_real_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return {"ok": True, "report": report, "active": _active().get("model_b")}


def _confusion(y, p) -> dict:
    out: dict = {}
    for yi, pi in zip(y, p):
        if pi is None:
            continue
        out.setdefault(str(yi), {}).setdefault(str(pi), 0)
        out[str(yi)][str(pi)] += 1
    return out


def status() -> dict:
    rep = None
    try:
        rep = json.load(open(os.path.join(RUN_DIR, "model_b_real_report.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    syn = None
    try:
        syn = json.load(open(os.path.join(RUN_DIR, "synthetic_report.json"), encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {"active": _active().get("model_b"), "real_report": rep, "synthetic_report": syn,
            "extras": [{k: x.get(k) for k in ("scenario_id", "pattern", "breed", "n_days", "severity_multiplier",
                                              "description")} for x in load_extras()],
            "has_real": os.path.exists(os.path.join(RUN_DIR, "model_b_real.joblib")), "min_real": MIN_REAL}


def activate_b(which: str) -> dict:
    if which not in ("real", "synthetic"):
        raise ValueError("which 只能是 real / synthetic")
    if which == "real" and not os.path.exists(os.path.join(RUN_DIR, "model_b_real.joblib")):
        raise ValueError("还没训练过真实数据版模型 B")
    os.makedirs(RUN_DIR, exist_ok=True)
    with open(os.path.join(RUN_DIR, _ACTIVE_FILE), "w", encoding="utf-8") as f:
        json.dump({"model_b": which, "at": datetime.now().isoformat(timespec="seconds")}, f)
    reset_cache()
    return status()
