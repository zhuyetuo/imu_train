def build_xgb(cfg: dict):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.1),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=cfg.get("random_state", 42),
            eval_metric="mlogloss",
        )
    except ImportError:
        raise ImportError("请安装 xgboost: pip install xgboost")
