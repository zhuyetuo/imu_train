"""
跨数据集标签统一映射。
合并模式只保留两个数据集共有的4类行为。
"""

# 数据集A 原始标签 → 统一标签
LABEL_MAP_A = {
    "Standing":    "Standing",
    "Sitting":     "Sitting",
    "Lying chest": "Lying",
    "Walking":     "Walking",
    # Trotting / Sniffing 数据集B无对应，丢弃
}

# 数据集B 原始标签 → 统一标签（数据集B全小写）
LABEL_MAP_B = {
    "standing":    "Standing",
    "sitting":     "Sitting",
    "lying down":  "Lying",
    "walking":     "Walking",
    # body shake 数据集A无对应，丢弃
}

# 合并后统一类别
UNIFIED_LABELS = ["Lying", "Sitting", "Standing", "Walking"]


def apply_map(labels, label_map: dict):
    """将标签数组按 label_map 映射，未在 map 中的标签置为 None。"""
    import numpy as np
    return np.array([label_map.get(l, None) for l in labels])
