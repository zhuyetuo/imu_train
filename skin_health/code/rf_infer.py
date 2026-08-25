"""
真实推理：加载train_rf_model_a.py/train_rf_model_b.py持久化的模型
(model_a.joblib/model_b.joblib)，对某个IMU的真实_infer.json历史数据算出
RF特征，输出C0/C1/C2、(可选，给了问答答案时)S0/S1/S2预测。

这是"算法组自己设计的RF方案"（跟pm_skin_scoring/的PM固定权重公式是两个
独立方案，见pm_skin_scoring/README或questionnaire_app.py开头的说明），
用同一份真实IMU数据同时跑两套方案，才能看出两边在同一天给出的结果差多少。

⚠️ 重要提醒：这两个模型目前只在skin_health/data/rf_synthetic/下的合成数据
上训练过，没有真实兽医标签校准过，预测结果不代表真实准确率——训练脚本
自己的文档注释也写得很清楚："这个结果不能代表真实准确率，等真实兽医标签
攒够后要重新训练评估，这里只验证代码/特征管道本身没问题"。这个模块只是
把已经存在的训练管道接到真实数据推理这一步，不改变这个已知限制。

跟imu_scratch_daily_stats.py不同：那边只算SBS引擎需要的聚合特征(次数/
时长/聚集/夜间占比)，这里要给rf_features.compute_rf_features()喂完整
的events+wear_hours两张表(多天历史)，才能算出全部44个特征喂给模型A/B——
模型A的基线特征跨3/7/14/21/30/60天窗口，不是单日结构化输入能替代的。
"""
import glob
import json
import os
import sys
from datetime import datetime

import joblib
import pandas as pd

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, os.path.join(_CODE_DIR, "..", "..", "src"))

from extract_clips import extract_imu_label  # noqa: E402
from questionnaire_features import QUESTIONNAIRE_FEATURE_COLUMNS  # noqa: E402
from rf_features import compute_rf_features  # noqa: E402

MODEL_DIR_DEFAULT = os.path.join(_CODE_DIR, "..", "data", "rf_synthetic")
MODEL_A_FILENAME = "model_a.joblib"
MODEL_B_FILENAME = "model_b.joblib"


def _parse_ts(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")


def models_available(data_dir=MODEL_DIR_DEFAULT):
    """两个模型文件是不是都在——训练脚本(train_rf_model_a.py/
    train_rf_model_b.py)没跑过的话这两个文件不存在，页面上要能给出清楚的
    提示，不是直接FileNotFoundError报错。"""
    a = os.path.exists(os.path.join(data_dir, MODEL_A_FILENAME))
    b = os.path.exists(os.path.join(data_dir, MODEL_B_FILENAME))
    return a, b


def load_model_a(data_dir=MODEL_DIR_DEFAULT):
    return joblib.load(os.path.join(data_dir, MODEL_A_FILENAME))


def load_model_b(data_dir=MODEL_DIR_DEFAULT):
    return joblib.load(os.path.join(data_dir, MODEL_B_FILENAME))


def load_events_and_wear(infer_root, imu_label, min_conf=0.8, conf_field="conf_mean"):
    """扫描infer_root下所有天的{day}/_infer/*_infer.json，按imu_label
    (比如"IMU2"，用extract_clips.extract_imu_label()从文件名的_imu{N}
    标识里取，不是camN机位号——见review_to_labelstudio.py/
    imu_scratch_daily_stats.py之前修过的"cam号≠imu号"那个bug，这里从
    一开始就用正确的取法)筛出这一个IMU的全部历史，构造
    rf_features.compute_rf_features()需要的events/wear_hours两张表。

    只保留conf_field>=min_conf的抓挠片段——跟ML_PRELABEL/
    imu_scratch_daily_stats.py是同一个"算不算抓挠"的标准，不会出现RF这边
    的输入跟别的地方统计口径不一致。

    pet_id统一用imu_label本身(比如"IMU2")当ID——这只是RF特征体系内部的
    身份标识，不需要接入真实的宠物库ID系统，跟pm_skin_scoring那边"选IMU
    对应哪只狗"是同一个思路，只是这里的"pet_id"字段直接借用IMU标签。"""
    infer_jsons = sorted(glob.glob(os.path.join(infer_root, "*", "_infer", "*_infer.json")))

    event_rows = []
    wear_spans = {}  # date -> [(start_dt, end_dt), ...]，同一天可能有多个session
    for path in infer_jsons:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        csv_basename = data.get("csv_basename", os.path.basename(path))
        stem = os.path.splitext(csv_basename)[0]
        if extract_imu_label(stem) != imu_label:
            continue

        for seg in data.get("scratch_segments", []):
            if not seg.get("start_ts") or not seg.get("end_ts"):
                continue
            if seg.get(conf_field, 0.0) < min_conf:
                continue
            try:
                start = _parse_ts(seg["start_ts"])
                end = _parse_ts(seg["end_ts"])
            except ValueError:
                continue
            event_rows.append({"pet_id": imu_label, "start": start, "end": end,
                               "duration_sec": (end - start).total_seconds()})

        window_ts = []
        for w in data.get("windows", []):
            if not w.get("ts"):
                continue
            try:
                window_ts.append(_parse_ts(w["ts"]))
            except ValueError:
                continue
        if window_ts:
            d = min(window_ts).date()
            wear_spans.setdefault(d, []).append((min(window_ts), max(window_ts)))

    events = pd.DataFrame(event_rows, columns=["pet_id", "start", "end", "duration_sec"])

    # 佩戴时长按区间并集算，不是直接累加——同一天同一个IMU的录制session
    # 会重叠，直接累加会把重叠部分算两次，跟imu_scratch_daily_stats.py的
    # _union_seconds()是同一个道理
    wear_rows = []
    for d, spans in wear_spans.items():
        merged = []
        for s, e in sorted(spans):
            if merged and s <= merged[-1][1]:
                if e > merged[-1][1]:
                    merged[-1][1] = e
            else:
                merged.append([s, e])
        hours = sum((e - s).total_seconds() for s, e in merged) / 3600
        wear_rows.append({"pet_id": imu_label, "date": d, "valid_wear_hours": round(hours, 2)})
    wear_hours = pd.DataFrame(wear_rows, columns=["pet_id", "date", "valid_wear_hours"])

    return events, wear_hours


def _prepare_categorical(X, breed_categories):
    """按训练时见过的品种类别集合构造pandas.Categorical——
    HistGradientBoostingClassifier的categorical_features是按fit时的
    category dtype做内部编码的，推理时如果类别集合对不上(比如喂了训练时
    没见过的品种字符串)会编码错位甚至报错。用pd.Categorical构造，训练时
    没见过的品种自动变成NaN(HistGradientBoostingClassifier原生支持NaN，
    当"未知类别"处理，不会报错，只是这个特征对这次预测没有信息量)。"""
    X = X.copy()
    X["breed_or_size_class"] = pd.Categorical(X["breed_or_size_class"], categories=breed_categories)
    return X


def predict_c(model_a_bundle, events, wear_hours, pet_id, breed, target_date):
    """返回dict：available(bool)、tier(str,不available时为None)、
    proba(dict{类别:概率})、reason(不available时的原因说明)、
    feature_row(pandas.Series，那一天的完整特征值，供展示用)。"""
    if wear_hours.empty or pet_id not in wear_hours["pet_id"].values:
        return {"available": False, "reason": "这个IMU完全没有可用的推理数据(_infer.json)"}

    breed_map = {pet_id: breed}
    feats = compute_rf_features(events, wear_hours, breed_map)
    row = feats[(feats["pet_id"] == pet_id) & (feats["date"] == target_date)]
    if row.empty:
        return {"available": False, "reason": f"{target_date} 这天没有佩戴数据，算不出特征"}
    if row.iloc[0]["data_quality_flag"] != "good":
        return {"available": False,
                "reason": f"{target_date} 佩戴时长不足(data_quality_flag="
                          f"{row.iloc[0]['data_quality_flag']})，RF模型跟PM版一样，"
                          f"数据不完整的天不出正式预测"}

    model = model_a_bundle["model"]
    feature_cols = model_a_bundle["feature_cols"]
    X = row[feature_cols].copy()
    X = _prepare_categorical(X, model_a_bundle["breed_categories"])

    proba = model.predict_proba(X)[0]
    classes = model_a_bundle["classes"]
    proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
    tier = classes[proba.argmax()]

    return {"available": True, "tier": tier, "proba": proba_dict,
            "feature_row": row.iloc[0], "reason": ""}


def predict_s(model_a_bundle, model_b_bundle, events, wear_hours, pet_id, breed, target_date,
             questionnaire_ordinals=None):
    """questionnaire_ordinals: dict，key跟questionnaire_features.
    QUESTIONNAIRE_FEATURE_COLUMNS一致(skin_redness_level/skin_pigment_
    abnormal/skin_lesion_severity/odor_level/hair_loss_spot_count_level/
    hair_loss_max_diameter_level/coat_quality_level)，值是对应的有序整数
    (含义见questionnaire_features.py顶部注释)。

    问答是可选的，不是必须的——传None/空dict时，问答那7个特征全部当NaN
    喂给模型B，HistGradientBoostingClassifier原生支持缺失值，照样能出一个
    预测，不是拒绝预测。跟PM版"S总分必须有C值+问答"的强制要求不一样：
    PM那套是固定公式，公式里没有问答这几项数字就没法算；这边是训练出来的
    模型，训练时NaN也是一种"合法"取值(模型自己学怎么处理确实值)，缺问答
    不等于没法预测，只是这几个特征这次没提供信息，预测出来的置信度可能
    会因为缺了这部分信号而更不确定——具体信不信这个预测，看返回的
    used_questionnaire字段和置信度自己判断，不是页面代码替用户下判断。

    模型B需要模型A的预测概率当"stacking"特征(见train_rf_model_b.py)，
    这里直接复用predict_c()算出来的概率，不重新训一个模型——两步预测
    共享同一个模型A的输出，逻辑上就该是这样，不是巧合。"""
    c_result = predict_c(model_a_bundle, events, wear_hours, pet_id, breed, target_date)
    if not c_result["available"]:
        return {"available": False, "reason": c_result["reason"], "c_result": c_result}

    used_questionnaire = bool(questionnaire_ordinals)
    questionnaire_ordinals = questionnaire_ordinals or {}

    model_b = model_b_bundle["model"]
    feature_cols = model_b_bundle["feature_cols"]
    stack_cols = model_b_bundle["stack_cols"]

    row = c_result["feature_row"]
    row_dict = {c: row[c] for c in row.index}
    for stack_col in stack_cols:
        # stack_col形如"model_a_proba_C1"，对应c_result["proba"]["C1"]
        cls_name = stack_col.replace("model_a_proba_", "")
        row_dict[stack_col] = c_result["proba"].get(cls_name, 0.0)
    for qc in QUESTIONNAIRE_FEATURE_COLUMNS:
        row_dict[qc] = questionnaire_ordinals.get(qc, float("nan"))

    missing = [c for c in feature_cols if c not in row_dict or row_dict[c] is None
              or (isinstance(row_dict[c], float) and pd.isna(row_dict[c]))]
    X = pd.DataFrame([{c: row_dict.get(c) for c in feature_cols}])
    X = _prepare_categorical(X, model_b_bundle["breed_categories"])

    proba = model_b.predict_proba(X)[0]
    classes = model_b_bundle["classes"]
    proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
    tier = classes[proba.argmax()]

    return {"available": True, "tier": tier, "proba": proba_dict,
            "c_result": c_result, "missing_features": missing,
            "used_questionnaire": used_questionnaire, "reason": ""}
