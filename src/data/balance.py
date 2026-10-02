"""类别均衡：按类别下采样训练集窗口。src/ml/train.py 和 src/dl/train.py 共用。"""

import numpy as np


def balance_train(X, y, classes, spec: str | None, seed: int = 0):
    """按类别下采样训练集。spec：none / min / cap:N。返回 (X, y, info 或 None)。

    砍的是**窗口**不是段，随机取，同一个 seed 可复现。只砍训练集：验证集砍了的话
    指标是在"均衡过的"数据上算的，跟别的版本不可比，也不代表线上分布。
    砍完类别权重照样按砍完的分布重算（下面那段），不会双重补偿。
    """
    spec = (spec or "none").strip()
    if spec == "none" or len(y) == 0:
        return X, y, None
    y_int = np.asarray(y).astype(int)
    counts = np.bincount(y_int, minlength=len(classes))
    present = [c for c in counts if c > 0]
    if spec == "min":
        cap = int(min(present)) if present else 0
    elif spec.startswith("cap:"):
        cap = int(spec.split(":", 1)[1])
    else:
        raise ValueError(f"--balance 不认识: {spec}（none / min / cap:N）")
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(y_int), dtype=bool)
    for c in range(len(classes)):
        idx = np.flatnonzero(y_int == c)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep[idx] = True
    after = np.bincount(y_int[keep], minlength=len(classes))
    info = {"spec": spec, "cap": cap,
            "before": {classes[i]: int(c) for i, c in enumerate(counts)},
            "after": {classes[i]: int(c) for i, c in enumerate(after)}}
    return X[keep], y[keep], info
