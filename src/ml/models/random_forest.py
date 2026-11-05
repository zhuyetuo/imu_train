from sklearn.ensemble import RandomForestClassifier


def build_rf(cfg: dict) -> RandomForestClassifier:
    # class_weight 以前没传进来：train.py 给 rf 设了 "balanced"，到这里被丢掉，
    # 随机森林一直是不加权训的——抓挠这种窗口少的类在投票里天然吃亏
    return RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 200),
        max_depth=cfg.get("max_depth", None),
        min_samples_leaf=cfg.get("min_samples_leaf", 1),
        max_features=cfg.get("max_features", "sqrt"),
        n_jobs=cfg.get("n_jobs", -1),
        random_state=cfg.get("random_state", 42),
        class_weight=cfg.get("class_weight", None),
    )
