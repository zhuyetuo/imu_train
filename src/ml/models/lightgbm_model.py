def build_lgbm(cfg: dict):
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", -1),
            num_leaves=cfg.get("num_leaves", 63),
            learning_rate=cfg.get("learning_rate", 0.05),
            n_jobs=cfg.get("n_jobs", -1),
            random_state=cfg.get("random_state", 42),
            verbose=-1,
        )
    except ImportError:
        raise ImportError("请安装 lightgbm: pip install lightgbm")
