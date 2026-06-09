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


def fit_with_progress(model, args, cfg, X_tr_f, y_tr):
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
        model.fit(X_tr_f, y_tr)
        model.set_params(callbacks=[], verbosity=0)

    elif args.model == "lgbm":
        from lightgbm import log_evaluation, record_evaluation
        n_est = cfg["lightgbm"]["n_estimators"]
        pbar = tqdm(total=n_est, desc="LightGBM", unit="轮")

        def _cb(env):
            pbar.update(1)
            if env.iteration + 1 == n_est:
                pbar.close()

        model.fit(X_tr_f, y_tr, callbacks=[_cb])

    elif args.model == "catboost":
        n_iter = cfg["catboost"]["iterations"]
        pbar = tqdm(total=n_iter, desc="CatBoost", unit="轮")

        class PbarCallback:
            def after_iteration(self, info):
                pbar.update(1)
                return True

        model.fit(X_tr_f, y_tr, callbacks=[PbarCallback()])
        pbar.close()

    else:
        with tqdm(total=1, desc=args.model, unit="step") as pbar:
            model.fit(X_tr_f, y_tr)
            pbar.update(1)

    return model


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[ml/train] hz={args.hz}, model={args.model}, processed_dir={args.processed_dir}")
    (X_tr, y_tr, _), (X_val, y_val, _), (X_te, y_te, _), meta = load_all_splits(args.hz, args.processed_dir)
    classes = eval(meta["classes"]) if isinstance(meta["classes"], str) else meta["classes"]

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

    build_fn, cfg_key = MODELS[args.model]
    model_cfg = dict(cfg[cfg_key])
    if args.n_jobs is not None:
        model_cfg["n_jobs"] = args.n_jobs
    model = build_fn(model_cfg)

    print(f"[ml/train] 训练中...")
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
    out_dir = os.path.join(args.results_dir, dataset_tag, f"{args.hz}hz")
    os.makedirs(out_dir, exist_ok=True)
    per_class = classification_report(y_te, y_pred, labels=present_labels,
                                      target_names=present_names,
                                      zero_division=0, output_dict=True)
    result = {
        "hz": args.hz, "model": args.model,
        "accuracy": acc, "macro_f1": f1,
        "classes": present_names,
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
    parser.add_argument("--hz", type=int, required=True, choices=[5, 10, 25, 50])
    parser.add_argument("--model", default="xgb", choices=list(MODELS))
    parser.add_argument("--config", default="configs/ml.yaml")
    parser.add_argument("--processed_dir", default="data/processed_a")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="覆盖模型的 n_jobs（并行启动时限制每个任务的核数）")
    main(parser.parse_args())
