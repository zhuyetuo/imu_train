def build_catboost(cfg: dict):
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=cfg.get("iterations", 300),
            depth=cfg.get("depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            random_seed=cfg.get("random_state", 42),
            verbose=0,
        )
    except ImportError:
        raise ImportError("请安装 catboost: pip install catboost")
