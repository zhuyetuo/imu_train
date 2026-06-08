from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_svm(cfg: dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=cfg.get("C", 1.0),
            kernel=cfg.get("kernel", "rbf"),
            probability=cfg.get("probability", True),
        )),
    ])
