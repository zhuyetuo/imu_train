"""
跨数据集标签统一映射。
两个数据集合并时，只保留共有的4类行为。
如需单数据集训练，直接用各自的 keep_labels 配置即可。
"""

# 数据集A 原始标签 → 统一标签
LABEL_MAP_A = {
    "Standing":    "Standing",
    "Sitting":     "Sitting",
    "Lying chest": "Lying",
    "Walking":     "Walking",
    # 以下类别在合并模式下丢弃（数据集B无对应）
    # "Trotting":  丢弃
    # "Sniffing":  丢弃
}

# 数据集B 原始标签 → 统一标签
LABEL_MAP_B = {
    "Standing":    "Standing",
    "Sitting":     "Sitting",
    "Lying down":  "Lying",
    "Walking":     "Walking",
    # "Body shake": 丢弃
}

# 合并后的统一类别（训练时使用）
UNIFIED_LABELS = ["Lying", "Sitting", "Standing", "Walking"]


def apply_map(labels: "np.ndarray", label_map: dict) -> "np.ndarray":
    """将原始标签数组按 label_map 映射，未在 map 中的标签置为 None。"""
    import numpy as np
    return np.array([label_map.get(l, None) for l in labels])
