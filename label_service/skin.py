"""
皮肤评估接口的业务层——PM 规则（skin_rules.py，从 questionnaire_app.py 逐字抽出）
+ IMU 日统计扫描 + ML 模型 A/B（skin_health/code/rf_infer.py）。所有函数返回
JSON 可序列化的 dict，供 app.py 里的 /api/v1/skin/* 路由直接返回；React 页面按
结构化字段渲染，同时保留 Gradio 版的 markdown 明细（breakdown_md）方便对照。

记录/周报表的持久化不在这里——那是 label_infra 的数据库（每只狗、每个填写人
都跟账号体系挂钩），这个服务只管"算"。
"""

import csv
import datetime as _dt
import glob
import os
import sys

from label_service import config
from label_service import skin_rules as R

_SKIN_HEALTH_CODE = os.path.join(config.REPO_ROOT, "skin_health", "code")
if _SKIN_HEALTH_CODE not in sys.path:
    sys.path.insert(0, _SKIN_HEALTH_CODE)
_SRC = os.path.join(config.REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

try:
    import rf_infer  # noqa: E402
    _RF_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001
    rf_infer = None
    _RF_IMPORT_ERROR = f"{type(e).__name__}: {e}"

ML_CAVEAT = ("这两个模型目前只在合成数据上训练过，没有真实兽医标签校准过，预测结果不代表真实准确率，"
             "只能看个大概方向、跟PM版的C值计算/S总分对照看两套方案在同一天差多少，不能当成真实诊断依据。")


# ── 选项/常量 ──────────────────────────────────────────────────────────

def options() -> dict:
    def opts(lst):
        return [{"text": t, "score": s, "letter": R._letter_of(t)} for t, s in lst]
    return {
        "dog_names": R.DOG_NAME_OPTIONS,
        "imu_dog_default_map": R.IMU_DOG_DEFAULT_MAP,
        "questions": {
            "has_hair_loss": {"label": "1. 您家宠物身上是否有毛发稀疏或出现没有毛的情况？", "options": ["是", "否"]},
            "color": {"label": "2. 您拨开宠物毛发看皮肤时，皮肤颜色是什么样的？", "group": "皮肤状态", "options": opts(R.SKIN_COLOR_OPTIONS)},
            "odor": {"label": "3. 宠物身上有没有明显异味？", "group": "皮肤状态", "options": opts(R.ODOR_OPTIONS)},
            "lesion": {"label": "4. 宠物的皮肤是否完整？", "group": "皮肤状态", "options": opts(R.LESION_OPTIONS)},
            "hair_spot": {"label": "5. 宠物身上没有毛或毛发稀疏的地方是如何分布的？", "group": "毛发状态", "only_if_hair_loss": True, "options": opts(R.HAIR_SPOT_OPTIONS)},
            "hair_diameter": {"label": "6. 最大的一块秃毛区域大概有多大？", "group": "毛发状态", "only_if_hair_loss": True, "options": opts(R.HAIR_DIAMETER_OPTIONS)},
            "coat": {"label": "7. 宠物整体毛发状态看起来怎么样？", "group": "毛发状态", "options": opts(R.COAT_QUALITY_OPTIONS)},
        },
        "weights": {"skin_group": R.SKIN_GROUP_WEIGHT, "hair_group": R.HAIR_GROUP_WEIGHT, "c": R.C_WEIGHT},
        "c_tiers": {"C0": "0≤C<30", "C1": "30≤C<50", "C2": "50≤C≤100（或触发任一红旗）"},
        "s_tiers": {"S0": "0≤S<12", "S1": "12≤S<20", "S2": "20≤S≤74.25（或问答单项满分/C2）"},
        "c_delta_tiers": [{"score": s, "min_count_increase": c, "min_duration_increase_min": d, "ratio_gt": r} for s, c, d, r in R.C_DELTA_TIERS],
        "c_baseline_denom_floor": R.C_BASELINE_DENOM_FLOOR,
        "default_stats_roots": R.DEFAULT_IMU_STATS_ROOTS,
        "weekly_report_columns": R.WEEKLY_REPORT_COLUMNS,
        "weekly_autofill_indices": sorted(R.WEEKLY_AUTOFILL_INDICES),
        "weekly_default_chain": [[t, s] for t, s in R._WEEKLY_DEFAULT_CHAIN],
        "record_columns": R.RECORD_COLUMNS,
        "ml_caveat": ML_CAVEAT,
    }


# ── 问答分数 ──────────────────────────────────────────────────────────

def questionnaire_score(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat) -> dict:
    g = R._question_group_scores(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat)
    total, md = R.compute_score(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat)
    titles = {"odor": "体味", "lesion": "皮损", "spot": "秃毛分布", "diameter": "秃毛面积", "coat": "整体毛质"}
    red = [titles[k] for k in titles if g[k] == 20]
    return {
        "total": total,
        "items": {k: g[k] for k in ("color", "odor", "lesion", "spot", "diameter", "coat")},
        "skin_group_raw": g["skin_group_raw"], "skin_group_score": g["skin_group_raw"] * R.SKIN_GROUP_WEIGHT,
        "hair_group_raw": g["hair_group_raw"], "hair_group_score": g["hair_group_raw"] * R.HAIR_GROUP_WEIGHT,
        "hair_questions_counted": has_hair_loss == "是",
        "red_flag_items": red,
        "missing": R._missing_questions(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat),
        "letters": {k: R._letter_of(v) for k, v in {"color": color, "odor": odor, "lesion": lesion, "hair_spot": hair_spot,
                                                    "hair_diameter": hair_diameter, "coat": coat}.items()},
        "breakdown_md": md,
    }


# ── C 值 ──────────────────────────────────────────────────────────────

def c_score(baseline_count, baseline_duration_min, today_count, today_duration_min,
            cluster_count, persistence_days, zn, zd, long_scratch, has_baseline=True) -> dict:
    total, tier, md = R.compute_c_score(baseline_count, baseline_duration_min, today_count, today_duration_min,
                                        cluster_count, persistence_days, zn, zd, long_scratch, has_baseline)
    # 分项跟 compute_c_score 内部用的是同一组函数，这里再算一遍只是为了给结构化字段
    bc, bd, tc, td = (baseline_count or 0), (baseline_duration_min or 0), (today_count or 0), (today_duration_min or 0)
    cl, pdays, zn_, zd_ = (cluster_count or 0), (persistence_days or 0), (zn or 0), (zd or 0)
    if has_baseline:
        delta_score, delta_by, delta_ratio = R._c_score_delta(bc, bd, tc, td)
        delta_red = delta_score >= 30
    else:
        delta_score, delta_by, delta_ratio, delta_red = 0, None, None, False
    cluster_score, cluster_red = R._c_score_cluster(cl)
    pers_score, pers_red = R._c_score_persistence(pdays)
    int_score, int_red = R._c_score_interruption(zn_, zd_, bool(long_scratch))
    reasons = []
    if delta_red: reasons.append(f"变化幅度({delta_by})相对基线>3倍")
    if cluster_red: reasons.append("聚集时段≥3个")
    if pers_red: reasons.append("持续≥3天")
    if int_red: reasons.append("睡眠中断≥3次或触发长时间抓挠")
    return {
        "total": total, "tier": tier,
        "components": {
            "delta": {"score": delta_score, "max": 30, "by": delta_by, "ratio": delta_ratio, "red_flag": delta_red, "counted": bool(has_baseline)},
            "cluster": {"score": cluster_score, "max": 20, "red_flag": cluster_red},
            "persistence": {"score": pers_score, "max": 20, "red_flag": pers_red},
            "interruption": {"score": int_score, "max": 30, "red_flag": int_red},
        },
        "red_flags": reasons,
        "has_baseline": bool(has_baseline),
        "max_possible": 100 if has_baseline else 70,
        "breakdown_md": md,
    }


# ── S 总分 ────────────────────────────────────────────────────────────

def s_total(c_value, c_tier_hint, has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat) -> dict:
    total, c_tier, s_tier, md = R.compute_s_total(c_value, c_tier_hint, has_hair_loss, color, odor, lesion,
                                                 hair_spot, hair_diameter, coat)
    g = R._question_group_scores(has_hair_loss, color, odor, lesion, hair_spot, hair_diameter, coat)
    c = c_value if c_value is not None else 0
    q_red = 20 in (g["odor"], g["lesion"], g["spot"], g["diameter"], g["coat"])
    reasons = []
    if q_red: reasons.append("问答部分有单项打了20分满分")
    if c_tier == "C2": reasons.append("C值判定为C2")
    return {
        "total": total, "c_tier": c_tier, "s_tier": s_tier,
        "c_score": c * R.C_WEIGHT, "c_value_used": c, "c_missing": c_value is None,
        "skin_group_raw": g["skin_group_raw"], "skin_group_score": g["skin_group_raw"] * R.SKIN_GROUP_WEIGHT,
        "hair_group_raw": g["hair_group_raw"], "hair_group_score": g["hair_group_raw"] * R.HAIR_GROUP_WEIGHT,
        "red_flags": reasons,
        "breakdown_md": md,
    }


# ── IMU 日统计（imu_daily_scratch_stats.csv）────────────────────────────

def _resolve_root(root: str) -> str:
    """根目录允许相对 imu_train 仓库（跟 Gradio 版在仓库根目录跑一样）或绝对路径"""
    return root if os.path.isabs(root) else os.path.join(config.REPO_ROOT, root)


def scan_stats(roots_str: str, target_label: str = "抓挠") -> dict:
    """跟 questionnaire_app.scan_imu_roots 一样的扫描逻辑（{root}/{day}/{label}/stats.csv），
    去掉 gr.update；行里的数值字段转成 float/bool，date 保留 2026_8_19 原样并附 iso。"""
    roots = [r.strip() for r in (roots_str or "").split(",") if r.strip()]
    if not roots:
        return {"ok": False, "rows": [], "message": "请先填至少一个根目录"}
    all_rows, skipped = [], []
    num_fields = ["valid_wear_hours", "event_count", "total_duration_min", "max_event_duration_sec", "cluster_count",
                  "night_event_count", "zn", "zd", "baseline_count", "baseline_duration_min", "n_baseline_days", "persistence_days"]
    for root in roots:
        full_root = _resolve_root(root)
        if not os.path.isdir(full_root):
            skipped.append(f"{root}（目录不存在）")
            continue
        stats_csvs = sorted(glob.glob(os.path.join(full_root, "*", target_label, R.STATS_CSV_NAME)))
        if not stats_csvs:
            skipped.append(f"{root}（没有找到任何「{target_label}」类别的{R.STATS_CSV_NAME}，可能还没跑IMU_STATS=1，或者TARGET_LABELS没包含「{target_label}」）")
            continue
        root_label = os.path.basename(full_root.rstrip("/\\")) or root
        for stats_csv in stats_csvs:
            try:
                with open(stats_csv, encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{stats_csv}（读取出错：{e}）")
                continue
            for row in rows:
                row = dict(row)
                for k in num_fields:
                    row[k] = R._to_float_or_none(row.get(k))
                row["long_scratch"] = R._parse_bool(row.get("long_scratch"))
                row["has_baseline"] = int(row.get("n_baseline_days") or 0) > 0
                row["date_iso"] = R._to_iso_date(row["date"])
                row["root"] = root
                row["root_label"] = root_label
                row["stats_label"] = target_label
                row["date_label"] = f"{row['date']} [{root_label}]"
                all_rows.append(row)
    if not all_rows:
        return {"ok": False, "rows": [], "message": "没有扫描到任何天/机位的数据" + ("；" + "、".join(skipped) if skipped else "")}
    dates = sorted({r["date_label"] for r in all_rows})
    msg = f"扫描到{len(all_rows)}条(天,机位)数据，共{len(dates)}个日期" + ("；跳过：" + "、".join(skipped) if skipped else "")
    return {"ok": True, "rows": all_rows, "date_labels": dates, "message": msg}


def stats_to_c_inputs(row: dict) -> dict:
    """一行日统计 → C 值计算的输入（跟 apply_stats_to_c_calc 的映射一致）+ 警示"""
    warnings = []
    if row.get("data_quality_flag") and row["data_quality_flag"] != "good":
        warnings.append("这天佩戴时长不足12小时（数据不完整），抓挠次数天然会偏低，不建议直接拿这天的数据定C档位")
    if not row.get("has_baseline"):
        warnings.append("这天没有可用的历史基线，「变化幅度」这一项不计分，C值上限只有70分")
    return {
        "baseline_count": row.get("baseline_count") or 0, "baseline_duration_min": row.get("baseline_duration_min") or 0,
        "today_count": row.get("event_count") or 0, "today_duration_min": row.get("total_duration_min") or 0,
        "cluster_count": row.get("cluster_count") or 0, "persistence_days": row.get("persistence_days") or 0,
        "zn": row.get("zn") or 0, "zd": row.get("zd") or 0, "long_scratch": bool(row.get("long_scratch")),
        "has_baseline": bool(row.get("has_baseline")),
        "fill_date": row.get("date_iso"), "dog_name": R.IMU_DOG_DEFAULT_MAP.get(row.get("imu")),
        "warnings": warnings,
    }


# ── ML 模型 A/B ───────────────────────────────────────────────────────

def ml_status() -> dict:
    if rf_infer is None:
        return {"available": False, "error": f"rf_infer 导入失败：{_RF_IMPORT_ERROR}", "model_a": False, "model_b": False}
    a, b = rf_infer.models_available()
    return {"available": True, "error": None, "model_a": a, "model_b": b, "model_dir": os.path.abspath(rf_infer.MODEL_DIR_DEFAULT)}


def ml_scan(roots_str: str) -> dict:
    """跟 questionnaire_app.ml_scan_roots 一样：只看目录名/文件名，不打开文件。
    兼容两种目录结构：{root}/{day}/_infer/*.json（rf_infer 原生）和现在多类别输出的
    {root}/{day}/{label}/_infer/*.json（只取「抓挠」那层，windows 各类别相同）。后者
    记在 rows 的 infer_sub 里，预测时由 _events_for 用临时软链拼成 rf_infer 认的结构。"""
    if rf_infer is None:
        return {"ok": False, "rows": [], "message": f"rf_infer模块导入失败：{_RF_IMPORT_ERROR}"}
    roots = [r.strip() for r in (roots_str or "").split(",") if r.strip()]
    if not roots:
        return {"ok": False, "rows": [], "message": "请先填至少一个根目录"}
    import re
    from extract_clips import extract_imu_label  # noqa: E402
    all_rows, skipped = [], []
    for root in roots:
        full_root = _resolve_root(root)
        if not os.path.isdir(full_root):
            skipped.append(f"{root}（目录不存在）")
            continue
        # 两种结构都找：{day}/_infer 和 {day}/{label}/_infer
        infer_dirs = sorted(glob.glob(os.path.join(full_root, "*", "_infer")) + glob.glob(os.path.join(full_root, "*", "*", "_infer")))
        if not infer_dirs:
            skipped.append(f"{root}（没有找到任何_infer目录）")
            continue
        root_label = os.path.basename(full_root.rstrip("/\\")) or root
        found, ignored = False, []
        for infer_dir in infer_dirs:
            parent = os.path.dirname(infer_dir)
            day_str = os.path.basename(parent)
            label_sub = None
            if not re.match(r"^\d{4}_\d{1,2}_\d{1,2}$", day_str):
                # {day}/{label}/_infer：上一级才是日期
                grand = os.path.basename(os.path.dirname(parent))
                if re.match(r"^\d{4}_\d{1,2}_\d{1,2}$", grand):
                    label_sub, day_str = day_str, grand
                else:
                    ignored.append(day_str)
                    continue
            # 多类别结构下只认"抓挠"这一层（各类别 windows 相同、segments 不同）
            if label_sub is not None and label_sub != "抓挠":
                continue
            imus = set()
            for path in glob.glob(os.path.join(infer_dir, "*_infer.json")):
                imus.add(extract_imu_label(os.path.basename(path)[:-len("_infer.json")]))
            for imu in sorted(imus):
                all_rows.append({"date": R._to_iso_date(day_str), "date_raw": day_str, "imu": imu, "root": root,
                                 "root_label": root_label, "infer_sub": label_sub,
                                 "date_label": f"{R._to_iso_date(day_str)} [{root_label}]"})
                found = True
        if not found:
            skipped.append(f"{root}（没有找到任何_infer.json）")
        if ignored:
            skipped.append(f"{root}下{len(ignored)}个非日期格式的目录已忽略：{'、'.join(sorted(set(ignored))[:5])}{'...' if len(ignored) > 5 else ''}")
    if not all_rows:
        return {"ok": False, "rows": [], "message": "没有扫描到任何天/机位的佩戴数据" + ("；" + "、".join(skipped) if skipped else "")}
    dates = sorted({r["date_label"] for r in all_rows})
    return {"ok": True, "rows": all_rows, "date_labels": dates,
            "message": f"扫描到{len(all_rows)}条(天,机位)真实数据，共{len(dates)}个日期" + ("；跳过：" + "、".join(skipped) if skipped else "")}


def _events_for(root: str, imu: str, infer_sub: str | None):
    """rf_infer.load_events_and_wear 只认 {root}/{day}/_infer；多类别结构（{day}/{label}/_infer）
    时把 glob 打到 {root}/{day}/{label} 这层——它内部是 glob(infer_root/*/_infer/*_infer.json)，
    所以传 {root} 时 * 匹配 day；传多类别结构就没法一个 glob 覆盖，改成临时目录软链。"""
    full_root = _resolve_root(root)
    if not infer_sub:
        return rf_infer.load_events_and_wear(full_root, imu)
    import tempfile
    tmp = tempfile.mkdtemp(prefix="skin_ml_")
    for day_dir in glob.glob(os.path.join(full_root, "*")):
        src = os.path.join(day_dir, infer_sub, "_infer")
        if os.path.isdir(src):
            os.makedirs(os.path.join(tmp, os.path.basename(day_dir)), exist_ok=True)
            os.symlink(src, os.path.join(tmp, os.path.basename(day_dir), "_infer"))
    try:
        return rf_infer.load_events_and_wear(tmp, imu)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def _row_of(rows: list, date_label: str, imu: str):
    return next((r for r in rows if r.get("date_label") == date_label and r.get("imu") == imu), None)


def _clean(v):
    import math
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    return v


def ml_preview(rows: list, date_label: str, imu: str, dog_name: str | None) -> dict:
    if rf_infer is None:
        return {"ok": False, "message": f"rf_infer模块导入失败：{_RF_IMPORT_ERROR}"}
    match = _row_of(rows, date_label, imu)
    if not match:
        return {"ok": False, "message": "请先扫描根目录，并选好日期和机位"}
    from rf_features import compute_rf_features  # noqa: E402
    breed = R._dog_breed(dog_name) or "未知"
    events, wear = _events_for(match["root"], imu, match.get("infer_sub"))
    feats = compute_rf_features(events, wear, {imu: breed})
    target = _dt.date.fromisoformat(match["date"])
    row = feats[(feats["pet_id"] == imu) & (feats["date"] == target)]
    if row.empty:
        return {"ok": True, "empty": True, "message": "这天没有可用的佩戴数据", "features": {}}
    r = row.iloc[0]
    keys = ["valid_wear_hours", "data_quality_flag", "event_count", "total_duration_min", "max_event_duration_sec",
            "cluster_count", "night_ratio", "sleep_disruption_count", "history_days_available", "has_any_baseline",
            "z_score_vs_self", "consecutive_days_above_baseline"]
    return {"ok": True, "empty": False, "features": {k: _clean(r[k]) for k in keys if k in r.index}}


def ml_predict(rows: list, date_label: str, imu: str, dog_name: str | None, which: str,
               answers: dict | None = None) -> dict:
    """which = "c"（模型A）或 "s"（模型B，answers 可选：has_hair_loss/color/...）"""
    if rf_infer is None:
        return {"ok": False, "message": f"rf_infer模块导入失败：{_RF_IMPORT_ERROR}"}
    match = _row_of(rows, date_label, imu)
    if not match:
        return {"ok": False, "message": "请先扫描根目录，并选好日期和机位"}
    breed = R._dog_breed(dog_name)
    if not breed:
        return {"ok": False, "message": "请先选好「对应狗狗」（用来对应品种）"}
    a_avail, b_avail = rf_infer.models_available()
    if not a_avail or (which == "s" and not b_avail):
        return {"ok": False, "message": "模型文件不存在，先在 skin_health/code/ 下跑 train_rf_model_a.py（和 gen_model_b_training_data.py + train_rf_model_b.py）"}
    target = _dt.date.fromisoformat(match["date"])
    events, wear = _events_for(match["root"], imu, match.get("infer_sub"))
    model_a = rf_infer.load_model_a()
    if which == "c":
        res = rf_infer.predict_c(model_a, events, wear, imu, breed, target)
        if not res["available"]:
            return {"ok": False, "message": f"模型A无法预测：{res['reason']}"}
        return {"ok": True, "model": "A", "tier": res["tier"], "proba": {k: float(v) for k, v in res["proba"].items()}, "caveat": ML_CAVEAT}
    model_b = rf_infer.load_model_b()
    a = answers or {}
    ordinals = R._pm_answers_to_rf_ordinals(a.get("color"), a.get("odor"), a.get("lesion"), a.get("hair_spot"),
                                            a.get("hair_diameter"), a.get("coat"))
    res = rf_infer.predict_s(model_a, model_b, events, wear, imu, breed, target, ordinals)
    if not res["available"]:
        return {"ok": False, "message": f"模型B无法预测：{res['reason']}", "caveat": ML_CAVEAT}
    return {"ok": True, "model": "B", "tier": res["tier"], "proba": {k: float(v) for k, v in res["proba"].items()},
            "used_questionnaire": bool(res.get("used_questionnaire")),
            "missing_features": list(res.get("missing_features") or []),
            "c_tier_from_a": (res.get("c_result") or {}).get("tier"), "caveat": ML_CAVEAT}
