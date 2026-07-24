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
from models.random_forest import build_rf
from models.xgboost_model import build_xgb
from models.lightgbm_model import build_lgbm
from models.catboost_model import build_catboost


MODELS = {
    "rf":       (build_rf,       "random_forest"),
    "xgb":      (build_xgb,      "xgboost"),
    "lgbm":     (build_lgbm,     "lightgbm"),
    "catboost": (build_catboost, "catboost"),
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


def apply_remap(y, classes, remap: dict) -> tuple[np.ndarray, list[str]]:
    """
    remap: {"Lying chest": "睡觉", "Walking": "活动", ...}
    返回重映射后的 y 和新 classes 列表。
    """
    new_class_names = list(dict.fromkeys(remap.values()))  # 保序去重
    new_class2id = {c: i for i, c in enumerate(new_class_names)}
    mapping = {i: new_class2id[remap[c]] for i, c in enumerate(classes) if c in remap}
    new_y = np.array([mapping[int(label)] for label in y], dtype=np.int64)
    return new_y, new_class_names


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
        y_tr,  classes_new = apply_remap(y_tr,  classes, remap_cfg)
        y_val, _           = apply_remap(y_val, classes, remap_cfg)
        y_te,  _           = apply_remap(y_te,  classes, remap_cfg)
        classes = classes_new
        print(f"[ml/train] 重映射后类别: {classes}")

    # 合成数据注入（如抓挠伪数据）
    if args.synthetic:
        syn = np.load(args.synthetic)
        X_syn = syn["X"]                          # (N, window_size, 6)
        syn_label    = args.synthetic_label
        if syn_label in classes:
            syn_label_id = classes.index(syn_label)   # 合并到已有类别
            print(f"[ml/train] 合成数据合并到已有类别 '{syn_label}'(id={syn_label_id})")
        else:
            syn_label_id = len(classes)               # 追加为新类别
            classes      = classes + [syn_label]

        # 按 8:1:1 分配合成数据到 train/val/test
        n = len(X_syn)
        n_val = max(1, n // 10)
        n_te  = max(1, n // 10)
        n_tr  = n - n_val - n_te
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        X_syn_tr  = X_syn[idx[:n_tr]]
        X_syn_val = X_syn[idx[n_tr:n_tr+n_val]]
        X_syn_te  = X_syn[idx[n_tr+n_val:]]
        y_syn_tr  = np.full(n_tr,  syn_label_id, dtype=np.int64)
        y_syn_val = np.full(n_val, syn_label_id, dtype=np.int64)
        y_syn_te  = np.full(n_te,  syn_label_id, dtype=np.int64)

        # 降采样对齐 window_size（合成数据 Hz 与训练 Hz 不同时才处理）
        src_hz = args.synthetic_hz if args.synthetic_hz > 0 else args.hz
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

        X_tr  = np.concatenate([X_tr,  X_syn_tr],  axis=0)
        X_val = np.concatenate([X_val, X_syn_val], axis=0)
        X_te  = np.concatenate([X_te,  X_syn_te],  axis=0)
        y_tr  = np.concatenate([y_tr,  y_syn_tr],  axis=0)
        y_val = np.concatenate([y_val, y_syn_val], axis=0)
        y_te  = np.concatenate([y_te,  y_syn_te],  axis=0)
        print(f"[ml/train] 注入合成数据: {n} 窗口 → 类别 '{syn_label}'(id={syn_label_id})")
        print(f"[ml/train] 更新后类别: {classes}")
        print(f"[ml/train] 训练集大小: {len(X_tr)}  val: {len(X_val)}  test: {len(X_te)}")

    # 打印注入后的完整类别分布
    counts_tr  = np.bincount(y_tr.astype(int),  minlength=len(classes))
    counts_val = np.bincount(y_val.astype(int), minlength=len(classes))
    counts_te  = np.bincount(y_te.astype(int),  minlength=len(classes))
    dist_title = "含合成数据" if args.synthetic else "纯标注数据"
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

    if os.path.exists(feat_cache):
        cache = np.load(feat_cache)
        X_tr_f, X_val_f, X_te_f = cache["X_tr"], cache["X_val"], cache["X_te"]
        # 缓存与当前 npz 样本数不一致说明数据已重新预处理，需重建缓存
        if len(X_tr_f) != len(X_tr) or len(X_te_f) != len(X_te):
            print(f"[ml/train] 缓存样本数与数据不符，重建缓存: {feat_cache}")
            os.remove(feat_cache)
            X_tr_f = extract_features(X_tr, args.hz)
            X_val_f = extract_features(X_val, args.hz)
            X_te_f = extract_features(X_te, args.hz)
            np.savez_compressed(feat_cache, X_tr=X_tr_f, X_val=X_val_f, X_te=X_te_f)
        else:
            print(f"[ml/train] 加载缓存特征: {feat_cache}")
    else:
        print(f"[ml/train] 提取特征（首次，之后自动缓存）...")
        X_tr_f = extract_features(X_tr, args.hz)
        X_val_f = extract_features(X_val, args.hz)
        X_te_f = extract_features(X_te, args.hz)
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
    # RF / XGB / LightGBM / CatBoost 都支持 class_weight 或 sample_weight
    if args.model in ("rf",):
        model_cfg["class_weight"] = "balanced"
    model = build_fn(model_cfg)

    print(f"[ml/train] 训练中...")
    # XGB / LGBM / CatBoost 通过 sample_weight 传入权重
    if args.model in ("xgb", "lgbm", "catboost"):
        model = fit_with_progress(model, args, cfg, X_tr_f, y_tr, sample_weight=sample_weights)
    else:
        model = fit_with_progress(model, args, cfg, X_tr_f, y_tr)

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    y_pred = np.array(model.predict(X_te_f)).flatten().astype(int)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")

    print(f"\n[ml/train] 测试集结果:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    present_labels = sorted(set(y_te) | set(y_pred))
    present_names = [classes[i] for i in present_labels]
    print(classification_report(y_te, y_pred, labels=present_labels, target_names=present_names,
                                zero_division=0))

    dataset_tag = os.path.basename(args.processed_dir.rstrip("/"))
    remap_tag   = f"_{os.path.splitext(os.path.basename(args.remap))[0]}" if args.remap else ""
    syn_tag     = "_syn" if args.synthetic else ""
    out_dir = os.path.join(args.results_dir, dataset_tag, f"{args.hz}hz{remap_tag}{syn_tag}")
    os.makedirs(out_dir, exist_ok=True)
    per_class = classification_report(y_te, y_pred, labels=present_labels,
                                      target_names=present_names,
                                      zero_division=0, output_dict=True)
    gravity_aligned = meta.get("gravity_aligned", "True")
    if isinstance(gravity_aligned, str):
        gravity_aligned = gravity_aligned.lower() == "true"
    window_size = int(meta.get("window_size", 0))
    stride      = int(meta.get("stride", 0))
    window_s    = round(window_size / args.hz, 3) if args.hz else 0
    stride_s    = round(stride      / args.hz, 3) if args.hz else 0
    result = {
        "hz": args.hz, "model": args.model,
        "accuracy": acc, "macro_f1": f1,
        "classes": present_names,
        "gravity_aligned": gravity_aligned,
        "window_size": window_size,
        "stride": stride,
        "window_s": window_s,
        "stride_s": stride_s,
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
    parser.add_argument("--remap", default="",
                        help="标签重映射 YAML 文件路径（用于合并类别，如 6类→2类）")
    parser.add_argument("--synthetic", default="",
                        help="合成数据 npz 路径（X 字段为窗口数组，追加为新类别）")
    parser.add_argument("--synthetic_label", default="抓挠",
                        help="合成数据的类别名称（默认：抓挠）")
    parser.add_argument("--synthetic_hz", type=int, default=0,
                        help="合成数据的采样率（默认0=与--hz相同，无需降采样）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印数据集分布，不训练模型")
    main(parser.parse_args())
