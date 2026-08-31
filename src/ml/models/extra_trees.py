from sklearn.ensemble import ExtraTreesClassifier


def build_et(cfg: dict) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", None),
        n_jobs=cfg.get("n_jobs", -1),
        random_state=cfg.get("random_state", 42),
        class_weight=cfg.get("class_weight", None),
    )
