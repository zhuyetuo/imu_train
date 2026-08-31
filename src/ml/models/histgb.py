from sklearn.ensemble import HistGradientBoostingClassifier


def build_histgb(cfg: dict) -> HistGradientBoostingClassifier:
    # HistGradientBoostingClassifier不支持class_weight参数（sklearn没实现），
    # 类别不均衡靠train.py统一算好的sample_weight在.fit()时传入，
    # 不在这里的构造参数里处理——跟xgb/lgbm/catboost走的是同一条路径
    return HistGradientBoostingClassifier(
        max_iter=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", None),
        learning_rate=cfg.get("learning_rate", 0.1),
        random_state=cfg.get("random_state", 42),
    )
