"""
ML 训练入口
用法: python src/ml/train.py --hz 50 --model xgb --processed_dir data/processed_a
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import yaml
import numpy as np
import joblib

from dataset import load_all_splits
from features import extract_features
from gravity_align import gravity_align_batch, append_raw_tilt_batch
from preprocess import split_windows_by_segment
from models.random_forest import build_rf
from models.xgboost_model import build_xgb
from models.lightgbm_model import build_lgbm
from models.catboost_model import build_catboost
from models.extra_trees import build_et
from models.histgb import build_histgb


MODELS = {
    "rf":         (build_rf,       "random_forest"),
    "xgb":        (build_xgb,      "xgboost"),
    "lgbm":       (build_lgbm,     "lightgbm"),
    "catboost":   (build_catboost, "catboost"),
    "extratrees": (build_et,       "extra_trees"),
    "histgb":     (build_histgb,   "histgb"),
}


def fit_with_progress(model, args, cfg, X_tr_f, y_tr, sample_weight=None):
    """训练并显示进度条（XGBoost / LightGBM 有原生支持，其余直接 fit）。"""
    from tqdm import tqdm

    if args.model == "xgb":
        from xgboost.callback import TrainingCallback

        class TqdmCallback(TrainingCallback):
            def __init__(self, total):
                self.pbar = tqdm(total=total, desc="XGBoost", unit="轮")
            def after_iteration(self, model, epoch, evals_log):
                self.pbar.update(1)
                return False
            def after_training(self, model):
                self.pbar.close()
                return model

        n_est = cfg["xgboost"]["n_estimators"]
        model.set_params(callbacks=[TqdmCallback(n_est)], verbosity=0)
        model.fit(X_tr_f, y_tr, sample_weight=sample_weight)
        model.set_params(callbacks=[], verbosity=0)

    elif args.model == "lgbm":
        from lightgbm import log_evaluation, record_evaluation
        n_est = cfg["lightgbm"]["n_estimators"]
        pbar = tqdm(total=n_est, desc="LightGBM", unit="轮")

        def _cb(env):
            pbar.update(1)
            if env.iteration + 1 == n_est:
                pbar.close()

        model.fit(X_tr_f, y_tr, sample_weight=sample_weight, callbacks=[_cb])

    elif args.model == "catboost":
        n_iter = cfg["catboost"]["iterations"]
        pbar = tqdm(total=n_iter, desc="CatBoost", unit="轮")

        class PbarCallback:
            def after_iteration(self, info):
                pbar.update(1)
                return True

        model.fit(X_tr_f, y_tr, sample_weight=sample_weight, callbacks=[PbarCallback()])
        pbar.close()

    else:
        with tqdm(total=1, desc=args.model, unit="step") as pbar:
            model.fit(X_tr_f, y_tr, sample_weight=sample_weight)
            pbar.update(1)

    return model


def apply_remap(y, classes, remap: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    remap: {"Lying chest": "睡觉", "Walking": "活动", ...}
    返回重映射后的 y、新 classes 列表、keep_mask。

    remap配置里没覆盖到的原始类别（比如自采数据里的"未佩戴"——这个不是
    行为，是设备没戴在狗身上，训练行为分类器不该把这类样本也塞进去，见
    src/data/wear_state.py专门的未佩戴/静置检测）会被过滤掉，不是直接
    KeyError崩溃：mapping字典本来就只收录remap覆盖到的类别(`if c in
    remap`)，但之前这里对y的每个样本都无条件查表，没有同步过滤，遇到
    没覆盖的类别就崩了。keep_mask标记哪些样本被保留，调用方要用同一个
    mask去过滤对应的X，保持X/y行数一致。
    """
    new_class_names = list(dict.fromkeys(remap.values()))  # 保序去重
    new_class2id = {c: i for i, c in enumerate(new_class_names)}
    mapping = {i: new_class2id[remap[c]] for i, c in enumerate(classes) if c in remap}
    keep_mask = np.array([int(label) in mapping for label in y], dtype=bool)
    new_y = np.array([mapping[int(label)] for label in y[keep_mask]], dtype=np.int64)
    return new_y, new_class_names, keep_mask


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[ml/train] hz={args.hz}, model={args.model}, processed_dir={args.processed_dir}")
    (X_tr, y_tr, _), (X_val, y_val, _), (X_te, y_te, _), meta = load_all_splits(args.hz, args.processed_dir)
    classes = eval(meta["classes"]) if isinstance(meta["classes"], str) else meta["classes"]

    # 打印映射前的类别分布
    counts_tr0  = np.bincount(y_tr.astype(int),  minlength=len(classes))
    counts_val0 = np.bincount(y_val.astype(int), minlength=len(classes))
    counts_te0  = np.bincount(y_te.astype(int),  minlength=len(classes))
    print(f"\n[ml/train] ── 原始类别分布（映射前）──")
    print(f"  {'类别':<12} {'训练':>8} {'验证':>8} {'测试':>8} {'合计':>8}")
    print(f"  {'-'*44}")
    for i, cls in enumerate(classes):
        total = int(counts_tr0[i] + counts_val0[i] + counts_te0[i])
        print(f"  {cls:<12} {int(counts_tr0[i]):>8} {int(counts_val0[i]):>8} {int(counts_te0[i]):>8} {total:>8}")
    print(f"  {'-'*44}")
    print(f"  {'合计':<12} {int(counts_tr0.sum()):>8} {int(counts_val0.sum()):>8} {int(counts_te0.sum()):>8} {int(counts_tr0.sum()+counts_val0.sum()+counts_te0.sum()):>8}")

    # 标签重映射（用于合并类别，如 6类→2类）
    remap_cfg = None
    if args.remap:
        with open(args.remap) as f:
            remap_cfg = yaml.safe_load(f)
        # 过滤掉注释行（以 # 开头的 key）
        remap_cfg = {k: v for k, v in remap_cfg.items() if not str(k).startswith("#")}
        print(f"\n[ml/train] 标签重映射: {args.remap}")
        for k, v in remap_cfg.items():
            print(f"  {k} → {v}")
        y_tr,  classes_new, keep_tr  = apply_remap(y_tr,  classes, remap_cfg)
        y_val, _,           keep_val = apply_remap(y_val, classes, remap_cfg)
        y_te,  _,           keep_te  = apply_remap(y_te,  classes, remap_cfg)
        # keep_mask过滤掉的是remap配置没覆盖到的类别(比如"未佩戴")，
        # X要跟着y一起过滤，不然行数对不上
        X_tr, X_val, X_te = X_tr[keep_tr], X_val[keep_val], X_te[keep_te]
        n_dropped = int((~keep_tr).sum() + (~keep_val).sum() + (~keep_te).sum())
        if n_dropped:
            print(f"[ml/train] 重映射: 丢弃了{n_dropped}个remap配置没覆盖到的样本"
                  f"（比如'未佩戴'这类不该当行为标签训练的类别，不是预期的话检查"
                  f"一下{args.remap}是不是漏配了某个类别）")
        classes = classes_new
        print(f"[ml/train] 重映射后类别: {classes}")

    # 合成数据注入（如抓挠伪数据）——支持一次注入多个类别的合成数据
    # （--synthetic_spec LABEL:PATH[:HZ]，可重复传），也兼容旧的单个
    # --synthetic/--synthetic_label 用法（只传一个类别时两种写法等价）
    if args.synthetic_spec:
        synthetic_specs = []
        for spec in args.synthetic_spec:
            parts = spec.split(":")
            if len(parts) == 2:
                label, path = parts
                hz = 0
            elif len(parts) == 3:
                label, path, hz = parts
                hz = int(hz)
            else:
                raise ValueError(f"--synthetic_spec 格式应为 LABEL:PATH 或 LABEL:PATH:HZ，收到: {spec}")
            synthetic_specs.append((label, path, hz))
    elif args.synthetic:
        synthetic_specs = [(args.synthetic_label, args.synthetic, args.synthetic_hz)]
    else:
        synthetic_specs = []

    is_synthetic_val = np.zeros(len(X_val), dtype=bool)
    is_synthetic_te  = np.zeros(len(X_te),  dtype=bool)
    syn_label_ids = []  # [(label, label_id), ...]，eval阶段按类别拆真实/合成用

    if synthetic_specs:
        gravity_aligned_meta = str(meta.get("gravity_aligned", "True")).lower() == "true"
        syn_train_r = float(meta.get("train_ratio", 0.8))
        syn_val_r   = float(meta.get("val_ratio",   0.1))

        X_syn_tr_all, y_syn_tr_all = [], []
        X_syn_val_all, y_syn_val_all = [], []
        X_syn_te_all, y_syn_te_all = [], []

        for syn_label, syn_path, syn_hz in synthetic_specs:
            syn = np.load(syn_path)
            X_syn = syn["X"]                          # (N, window_size, 6) 原始未对齐 acc+gyro
            # 合成数据历史上从未做过重力对齐，真实数据现在是 8 通道（对齐acc/gyro + 原始tilt），
            # 这里补齐同样的处理，否则和真实数据的特征空间不一致，且通道数拼接会直接报错
            tilt_syn = append_raw_tilt_batch(X_syn)[:, :, 6:8]
            if gravity_aligned_meta:
                X_syn = gravity_align_batch(X_syn)
            X_syn = np.concatenate([X_syn, tilt_syn], axis=2)

            if syn_label in classes:
                syn_label_id = classes.index(syn_label)   # 合并到已有类别
                print(f"[ml/train] 合成数据合并到已有类别 '{syn_label}'(id={syn_label_id})")
            else:
                syn_label_id = len(classes)               # 追加为新类别
                classes      = classes + [syn_label]
            syn_label_ids.append((syn_label, syn_label_id))

            # 按与真实数据相同的比例分配合成数据到 train/val/test。
            # 必须按"原始片段"分组划分，不能纯随机打乱窗口——合成数据是由少量真实
            # 片段增强放大出来的（同一片段可能产生几十个近乎重复的窗口），纯随机
            # 划分会让同一片段的窗口分散到train和val/test里，造成数据泄漏（验证集
            # 分数虚高但不代表真实泛化能力）。
            n = len(X_syn)
            if "seg_ids" in syn:
                syn_seg_ids = syn["seg_ids"]
            else:
                print("[ml/train] [警告] 合成数据文件缺少 seg_ids（旧版 synthesize_scratch.py 生成），"
                      "无法按片段分组，退化为按窗口随机划分，可能有数据泄漏。建议重新跑一遍"
                      "synthesize_scratch.py 生成带 seg_ids 的新文件。")
                syn_seg_ids = np.arange(n)  # 退化：每个窗口自成一组，等价于旧的纯随机划分
            dummy_y     = np.zeros(n, dtype=np.int64)
            dummy_y_seq = np.zeros((n, X_syn.shape[1]), dtype=np.int64)
            (X_syn_tr, _, _, X_syn_val, _, _, X_syn_te, _, _) = split_windows_by_segment(
                X_syn, dummy_y, dummy_y_seq, syn_seg_ids, syn_train_r, syn_val_r, seed=42)

            # 降采样对齐 window_size（合成数据 Hz 与训练 Hz 不同时才处理）
            src_hz = syn_hz if syn_hz > 0 else args.hz
            if src_hz != args.hz:
                from math import gcd
                g = gcd(src_hz, args.hz)
                up, down = args.hz // g, src_hz // g
                if up == 1:
                    step = down
                    X_syn_tr  = X_syn_tr[:,  ::step, :]
                    X_syn_val = X_syn_val[:, ::step, :]
                    X_syn_te  = X_syn_te[:,  ::step, :]
                else:
                    from scipy.signal import resample_poly
                    X_syn_tr  = resample_poly(X_syn_tr,  up, down, axis=1).astype(np.float32)
                    X_syn_val = resample_poly(X_syn_val, up, down, axis=1).astype(np.float32)
                    X_syn_te  = resample_poly(X_syn_te,  up, down, axis=1).astype(np.float32)

            X_syn_tr_all.append(X_syn_tr);   y_syn_tr_all.append(np.full(len(X_syn_tr),   syn_label_id, dtype=np.int64))
            X_syn_val_all.append(X_syn_val); y_syn_val_all.append(np.full(len(X_syn_val), syn_label_id, dtype=np.int64))
            X_syn_te_all.append(X_syn_te);   y_syn_te_all.append(np.full(len(X_syn_te),   syn_label_id, dtype=np.int64))
            print(f"[ml/train] 注入合成数据: {n} 窗口 → 类别 '{syn_label}'(id={syn_label_id})")

        # 记录哪些 val/test 样本是合成的（拼接前的长度就是真实样本数），
        # 后面单独拆开算一遍指标，避免"整体分数好看"掩盖"合成数据虚高、真实数据其实很差"
        n_val_real = len(X_val)
        n_te_real  = len(X_te)

        X_tr  = np.concatenate([X_tr]  + X_syn_tr_all,  axis=0)
        X_val = np.concatenate([X_val] + X_syn_val_all, axis=0)
        X_te  = np.concatenate([X_te]  + X_syn_te_all,  axis=0)
        y_tr  = np.concatenate([y_tr]  + y_syn_tr_all,  axis=0)
        y_val = np.concatenate([y_val] + y_syn_val_all, axis=0)
        y_te  = np.concatenate([y_te]  + y_syn_te_all,  axis=0)

        is_synthetic_val = np.zeros(len(X_val), dtype=bool)
        is_synthetic_val[n_val_real:] = True
        is_synthetic_te = np.zeros(len(X_te), dtype=bool)
        is_synthetic_te[n_te_real:] = True
        print(f"[ml/train] 更新后类别: {classes}")
        print(f"[ml/train] 训练集大小: {len(X_tr)}  val: {len(X_val)}  test: {len(X_te)}")

    # 打印注入后的完整类别分布
    counts_tr  = np.bincount(y_tr.astype(int),  minlength=len(classes))
    counts_val = np.bincount(y_val.astype(int), minlength=len(classes))
    counts_te  = np.bincount(y_te.astype(int),  minlength=len(classes))
    dist_title = "含合成数据" if synthetic_specs else "纯标注数据"
    print(f"\n[ml/train] ── 数据集类别分布（{dist_title}）──")
    print(f"  {'类别':<10} {'训练':>8} {'验证':>8} {'测试':>8} {'合计':>8}")
    print(f"  {'-'*42}")
    for i, cls in enumerate(classes):
        total = int(counts_tr[i] + counts_val[i] + counts_te[i])
        print(f"  {cls:<10} {int(counts_tr[i]):>8} {int(counts_val[i]):>8} {int(counts_te[i]):>8} {total:>8}")
    print(f"  {'-'*42}")
    print(f"  {'合计':<10} {int(counts_tr.sum()):>8} {int(counts_val.sum()):>8} {int(counts_te.sum()):>8} {int(counts_tr.sum()+counts_val.sum()+counts_te.sum()):>8}")
    min_ratio = counts_tr.min() / counts_tr.sum()
    if min_ratio < 0.1:
        print(f"\n  [警告] 最少类别占训练集比例 {min_ratio*100:.1f}%，类别严重不均衡，建议补充数据或调整合成量")

    if args.dry_run:
        print(f"\n[ml/train] --dry_run 模式，已退出（未训练）")
        return

    feat_dir = os.path.join(args.processed_dir, f"{args.hz}hz")
    feat_cache = os.path.join(feat_dir, "ml_features.npz")

    expected_dim = extract_features(X_tr[:1], args.hz, show_progress=False).shape[1] if len(X_tr) else None

    if os.path.exists(feat_cache):
        cache = np.load(feat_cache)
        X_tr_f, X_val_f, X_te_f = cache["X_tr"], cache["X_val"], cache["X_te"]
        # 缓存与当前 npz 样本数或特征维度不一致，说明数据/特征工程已变化，需重建缓存
        dim_mismatch = expected_dim is not None and X_tr_f.shape[1] != expected_dim
        if len(X_tr_f) != len(X_tr) or len(X_te_f) != len(X_te) or dim_mismatch:
            reason = "特征维度变化" if dim_mismatch else "样本数与数据不符"
            print(f"[ml/train] 缓存{reason}，重建缓存: {feat_cache}")
            os.remove(feat_cache)
            X_tr_f = extract_features(X_tr, args.hz, workers=args.feat_workers)
            X_val_f = extract_features(X_val, args.hz, workers=args.feat_workers)
            X_te_f = extract_features(X_te, args.hz, workers=args.feat_workers)
            np.savez_compressed(feat_cache, X_tr=X_tr_f, X_val=X_val_f, X_te=X_te_f)
        else:
            print(f"[ml/train] 加载缓存特征: {feat_cache}")
    else:
        print(f"[ml/train] 提取特征（首次，之后自动缓存）...")
        X_tr_f = extract_features(X_tr, args.hz, workers=args.feat_workers)
        X_val_f = extract_features(X_val, args.hz, workers=args.feat_workers)
        X_te_f = extract_features(X_te, args.hz, workers=args.feat_workers)
        np.savez_compressed(feat_cache, X_tr=X_tr_f, X_val=X_val_f, X_te=X_te_f)
        print(f"[ml/train] 特征已缓存至 {feat_cache}")

    # val 为空时（小数据集）用训练集末尾 10% 代替
    if len(X_val_f) == 0:
        n_fallback = max(1, len(X_tr_f) // 10)
        X_val_f, y_val = X_tr_f[-n_fallback:], y_tr[-n_fallback:]
        X_tr_f, y_tr   = X_tr_f[:-n_fallback], y_tr[:-n_fallback]
        print(f"[ml/train] val 集为空，从训练集末尾借用 {n_fallback} 个样本作为 val")

    print(f"[ml/train] 特征维度: {X_tr_f.shape[1]}")

    # 类别权重：按频率倒数自动平衡（解决类别不均衡问题）
    counts = np.bincount(y_tr.astype(int), minlength=len(classes))
    weights = len(y_tr) / (len(classes) * counts.clip(min=1))
    sample_weights = weights[y_tr.astype(int)]
    print(f"[ml/train] 类别分布: { {classes[i]: int(c) for i, c in enumerate(counts)} }")
    print(f"[ml/train] 类别权重: { {classes[i]: round(float(w), 3) for i, w in enumerate(weights)} }")

    build_fn, cfg_key = MODELS[args.model]
    model_cfg = dict(cfg[cfg_key])
    if args.n_jobs is not None:
        model_cfg["n_jobs"] = args.n_jobs
    # RF / ExtraTrees 走 class_weight="balanced"；XGB/LGBM/CatBoost/HistGB
    # 不支持class_weight构造参数（HistGradientBoostingClassifier是sklearn
    # 没实现，xgb/lgbm/catboost是各自库的设计），统一走下面的sample_weight
    if args.model in ("rf", "extratrees"):
        model_cfg["class_weight"] = "balanced"
    model = build_fn(model_cfg)

    print(f"[ml/train] 训练中...")
    if args.model in ("xgb", "lgbm", "catboost", "histgb"):
        model = fit_with_progress(model, args, cfg, X_tr_f, y_tr, sample_weight=sample_weights)
    else:
        model = fit_with_progress(model, args, cfg, X_tr_f, y_tr)

    from sklearn.metrics import accuracy_score, f1_score, classification_report

    has_test = len(X_te_f) > 0
    if has_test:
        eval_tag = "测试集"
        X_eval_f, y_eval = X_te_f, y_te
        is_synthetic_eval = is_synthetic_te
    else:
        # test_ratio=0 时无测试集，改用验证集汇报指标（仅供参考，验证集参与了早停/模型选择）
        eval_tag = "验证集（无测试集，仅供参考）"
        X_eval_f, y_eval = X_val_f, y_val
        is_synthetic_eval = is_synthetic_val

    y_pred = np.array(model.predict(X_eval_f)).flatten().astype(int)
    acc = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, average="macro")

    print(f"\n[ml/train] {eval_tag}结果:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    present_labels = sorted(set(y_eval) | set(y_pred))
    present_names = [classes[i] for i in present_labels]

    # 真实 vs 合成样本分开算指标：如果整体分数好看但只是因为合成样本占多数、
    # 合成样本"太好认"，这里能直接看出来（不能只看合成数据训练是否让整体分数
    # 变好看，得看它对真实样本有没有真实帮助）
    if syn_label_ids and is_synthetic_eval.sum() > 0 and (~is_synthetic_eval).sum() > 0:
        for syn_label, syn_label_id in syn_label_ids:
            syn_class_mask = (y_eval == syn_label_id)
            real_mask = syn_class_mask & (~is_synthetic_eval)
            fake_mask = syn_class_mask & is_synthetic_eval
            print(f"\n  ── '{syn_label}' 类别按来源拆分（不能只看整体分数）──")
            for tag, mask in (("真实样本", real_mask), ("合成样本", fake_mask)):
                n = int(mask.sum())
                if n == 0:
                    print(f"    {tag}: 0 个，跳过")
                    continue
                recall = float((y_pred[mask] == syn_label_id).mean())
                print(f"    {tag}: {n:>5} 个  recall={recall:.4f}"
                      f"{f'  ⚠️ 明显低于合成样本，说明模型对真实{syn_label}泛化不够' if tag == '真实样本' and recall < 0.7 else ''}")
    print(classification_report(y_eval, y_pred, labels=present_labels, target_names=present_names,
                                zero_division=0))

    window_size = int(meta.get("window_size", 0))
    stride      = int(meta.get("stride", 0))
    window_s    = round(window_size / args.hz, 3) if args.hz else 0
    stride_s    = round(stride      / args.hz, 3) if args.hz else 0

    def _fmt_duration(n_windows):
        # 窗口是按stride滑动切出来的，窗口之间有重叠（比如window=2s/
        # stride=1s时相邻窗口重叠1s），每个窗口新增的"没见过的"时长约等于
        # stride_s，用 窗口数×stride_s 估算这批数据实际覆盖的标注时长——
        # 不是精确值（忽略了每段片段第一个窗口多出来的window_s-stride_s
        # 首窗时长），但够用来判断"这个类别大概有多少时长、该补多少"。
        secs = n_windows * stride_s
        if secs >= 3600:
            return f"{secs/3600:.2f}h"
        if secs >= 60:
            return f"{secs/60:.1f}min"
        return f"{secs:.0f}s"

    # classification_report的support只算了eval集（验证集/测试集），训练用了
    # 多少窗口/多少时长看不出来——之前想确认"标注数据是不是都用上了、该给
    # 哪个类别补多少时长的数据"，只能翻前面"数据集类别分布"那张表，跟这里
    # 的precision/recall对不上号，不方便对照。这里把训练窗口数+估算时长
    # 也拼到同一张表里。
    train_counts = {lbl: int((y_tr == lbl).sum()) for lbl in present_labels}
    _pc = classification_report(y_eval, y_pred, labels=present_labels, target_names=present_names,
                                 zero_division=0, output_dict=True)
    print(f"  ── 各类别窗口数/时长（训练 vs {eval_tag}，时长是按stride估算的，仅供参考）──")
    print(f"  {'类别':<8}{'训练窗口数':>10}{'训练时长':>10}"
          f"{f'{eval_tag}窗口数':>14}{f'{eval_tag}时长':>10}"
          f"{'precision':>12}{'recall':>10}{'f1-score':>10}")
    for lbl, name in zip(present_labels, present_names):
        row = _pc[name]
        eval_n = int(row['support'])
        print(f"  {name:<8}{train_counts[lbl]:>10}{_fmt_duration(train_counts[lbl]):>10}"
              f"{eval_n:>14}{_fmt_duration(eval_n):>10}"
              f"{row['precision']:>12.2f}{row['recall']:>10.2f}{row['f1-score']:>10.2f}")

    dataset_tag = os.path.basename(args.processed_dir.rstrip("/"))
    remap_tag   = f"_{os.path.splitext(os.path.basename(args.remap))[0]}" if args.remap else ""
    syn_tag     = "_syn" if synthetic_specs else ""
    out_dir = os.path.join(args.results_dir, dataset_tag, f"{args.hz}hz{remap_tag}{syn_tag}")
    os.makedirs(out_dir, exist_ok=True)
    per_class = classification_report(y_eval, y_pred, labels=present_labels,
                                      target_names=present_names,
                                      zero_division=0, output_dict=True)
    gravity_aligned = meta.get("gravity_aligned", "True")
    if isinstance(gravity_aligned, str):
        gravity_aligned = gravity_aligned.lower() == "true"
    result = {
        "hz": args.hz, "model": args.model,
        "accuracy": acc, "macro_f1": f1,
        "classes": classes,  # 全部训练类别，而非仅测试集出现的类别
        "gravity_aligned": gravity_aligned,
        "window_size": window_size,
        "stride": stride,
        "window_s": window_s,
        "stride_s": stride_s,
        "label_mode": meta.get("label_mode", "majority"),  # 训练时窗口怎么打标签，推理侧重建事件要对齐这个
        "per_class": {k: {m: round(v, 4) for m, v in per_class[k].items()
                          if m in ("precision", "recall", "f1-score")}
                      for k in present_names},
    }
    with open(os.path.join(out_dir, f"ml_{args.model}.json"), "w") as f:
        json.dump(result, f, indent=2)

    joblib.dump(model, os.path.join(out_dir, f"ml_{args.model}.pkl"))
    print(f"[ml/train] 结果保存至 {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=int, required=True, choices=[5, 10, 15, 16, 20, 25, 50])
    parser.add_argument("--model", default="xgb", choices=list(MODELS))
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--processed_dir", default="data/processed_a")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="覆盖模型的 n_jobs（并行启动时限制每个任务的核数）")
    parser.add_argument("--feat_workers", type=int, default=1,
                        help="特征提取的并行进程数（默认1=不并行，原有行为不变）。"
                             "传-1用全部CPU核心，传正整数指定核数。特征提取是纯CPU计算，"
                             "核多、数据量大时开几个能明显提速")
    parser.add_argument("--remap", default="",
                        help="标签重映射 YAML 文件路径（用于合并类别，如 6类→2类）")
    parser.add_argument("--synthetic", default="",
                        help="合成数据 npz 路径（X 字段为窗口数组，追加为新类别）。"
                             "只能注入一个类别，多个类别用--synthetic_spec")
    parser.add_argument("--synthetic_label", default="抓挠",
                        help="合成数据的类别名称（默认：抓挠），配合--synthetic用")
    parser.add_argument("--synthetic_hz", type=int, default=0,
                        help="合成数据的采样率（默认0=与--hz相同，无需降采样），配合--synthetic用")
    parser.add_argument("--synthetic_spec", action="append", default=None,
                        help="LABEL:PATH[:HZ]，可重复传，一次注入多个类别的合成数据"
                             "（比如同时给抓挠和甩身体都补合成数据）。HZ留空=跟--hz相同。"
                             "传了这个就不再看--synthetic/--synthetic_label/--synthetic_hz。"
                             "例: --synthetic_spec 抓挠:data/synthetic/scratch.npz "
                             "--synthetic_spec 甩身体:data/synthetic/shake.npz")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印数据集分布，不训练模型")
    main(parser.parse_args())
