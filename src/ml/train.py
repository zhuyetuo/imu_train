"""
ML 训练入口
用法: python src/ml/train.py --hz 50
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from dataset import load_all_splits
from features import extract_features
from models.random_forest import build_rf
from models.svm import build_svm
from models.xgboost_model import build_xgb


MODELS = {
    "rf": build_rf,
    "svm": build_svm,
    "xgb": build_xgb,
}


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\n[ml/train] hz={args.hz}, model={args.model}")
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), meta = load_all_splits(args.hz)
    classes = eval(meta["classes"]) if isinstance(meta["classes"], str) else meta["classes"]

    print(f"[ml/train] 提取特征...")
    X_tr_f = extract_features(X_tr, args.hz)
    X_val_f = extract_features(X_val, args.hz)
    X_te_f = extract_features(X_te, args.hz)
    print(f"[ml/train] 特征维度: {X_tr_f.shape[1]}")

    model_fn = MODELS[args.model]
    model = model_fn(cfg[{"rf": "random_forest", "svm": "svm", "xgb": "xgboost"}[args.model]])

    print(f"[ml/train] 训练中...")
    if args.model == "xgb":
        from tqdm import tqdm
        from xgboost.callback import TrainingCallback

        class TqdmCallback(TrainingCallback):
            def __init__(self, total):
                self.pbar = tqdm(total=total, desc="XGBoost训练", unit="轮")
            def after_iteration(self, model, epoch, evals_log):
                self.pbar.update(1)
                return False
            def after_training(self, model):
                self.pbar.close()
                return model

        n_est = cfg["xgboost"]["n_estimators"]
        model.set_params(callbacks=[TqdmCallback(n_est)], verbosity=0)

    elif args.model == "rf":
        model.set_params(verbose=1)

    model.fit(X_tr_f, y_tr)

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    y_pred = model.predict(X_te_f)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")

    print(f"\n[ml/train] 测试集结果:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    present_labels = sorted(set(y_te) | set(y_pred))
    present_names = [classes[i] for i in present_labels]
    print(classification_report(y_te, y_pred, labels=present_labels, target_names=present_names))

    # 保存结果
    out_dir = os.path.join("results", f"{args.hz}hz")
    os.makedirs(out_dir, exist_ok=True)
    result = {"hz": args.hz, "model": args.model, "accuracy": acc, "macro_f1": f1}
    with open(os.path.join(out_dir, f"ml_{args.model}.json"), "w") as f:
        json.dump(result, f, indent=2)

    # 保存模型
    joblib.dump(model, os.path.join(out_dir, f"ml_{args.model}.pkl"))
    print(f"[ml/train] 结果保存至 {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=int, required=True, choices=[5, 10, 25, 50])
    parser.add_argument("--model", default="rf", choices=["rf", "svm", "xgb"])
    parser.add_argument("--config", default="configs/ml.yaml")
    main(parser.parse_args())
