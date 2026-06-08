from sklearn.ensemble import RandomForestClassifier


def build_rf(cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 200),
        max_depth=cfg.get("max_depth", None),
        n_jobs=cfg.get("n_jobs", -1),
        random_state=cfg.get("random_state", 42),
    )
