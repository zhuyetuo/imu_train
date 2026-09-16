"""验四件事。不需要训练数据，不需要显卡，几秒钟。

    python acc3/selfcheck5.py

① 砍掉陀螺仪那三列 ≡ 从来就没采过陀螺仪（逐位相同）
   —— 这是"直接转 npz、不重跑预处理"成立的前提。
② features5 的 113 维 = 仓库那 193 维里去掉所有 gyr_* 之后的那些，
   **顺序一致、数值逐位相同**（对照 src/ml/features.py 的标量参照实现）。
③ 特征名一一对上。
④ 5 通道走仓库那份 extract_features 只有 95 维——也就是**不新写这一份的话
   会少 34 维**（acc 模长 19、jerk 11、SMA+三轴相关 4），而那 34 维
   只靠加速计就能算。这一条是这个目录存在的理由，钉住它。

②③ 是这条路线的命门：顺序或维度一旦跟 193 维那份分家，
"推理时算 193 维再按下标取 113 列"这个办法就不成立了，
而取错列**不报错**——每一维都对到别的特征上，模型照样给得出结果。
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "src"), os.path.join(_REPO, "src", "data"),
           os.path.join(_REPO, "src", "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import features as F8  # noqa: E402
import features5 as f5  # noqa: E402
from gravity_align import append_raw_tilt_batch, gravity_align_batch  # noqa: E402

HZ, WIN = 16, 16


def _data(n=120, seed=5):
    rng = np.random.default_rng(seed)
    acc = (rng.normal(0, 2.0, (n, WIN, 3)).astype(np.float32)
           + np.array([0, 0, 9.8], np.float32))
    gyr = rng.normal(0, 40.0, (n, WIN, 3)).astype(np.float32)
    return acc, gyr


def check_dropping_gyro_equals_never_having_it():
    acc, gyr = _data()
    # 有陀螺仪：走完整预处理，最后把那三列砍掉
    X8r = np.concatenate([acc, gyr], axis=2)
    tilt8 = append_raw_tilt_batch(X8r)[:, :, 6:8]
    full = np.concatenate([gravity_align_batch(X8r), tilt8], axis=2)
    dropped = np.ascontiguousarray(full[:, :, [0, 1, 2, 6, 7]])
    # 从来没有陀螺仪：acc 三列直接走
    tilt3 = append_raw_tilt_batch(acc)[:, :, -2:]
    never = np.concatenate([gravity_align_batch(acc), tilt3], axis=2)
    if not np.array_equal(dropped, never):
        return False, f"最大差 {float(np.abs(dropped - never).max()):.3g}，不是逐位相同"
    print("  重力对齐的旋转矩阵只从 acc 算，所以砍列和没采过是同一个数")
    return True, ""


def _pair():
    """同一批数据的 8 通道版和 5 通道版。"""
    acc, gyr = _data()
    X8r = np.concatenate([acc, gyr], axis=2)
    tilt = append_raw_tilt_batch(X8r)[:, :, 6:8]
    X8 = np.concatenate([gravity_align_batch(X8r), tilt], axis=2)
    X5 = np.ascontiguousarray(X8[:, :, [0, 1, 2, 6, 7]])
    return X8, X5


def check_113_is_the_gyro_free_subset_of_193():
    X8, X5 = _pair()
    idx = f5.gyro_free_indices()
    # 对照**标量参照实现** _extract_one，不是向量化那条。
    # 仓库自己这两份就差一个 float32 的 eps（1.2e-7）——那是既有的差异，
    # 不是这里引入的。参照定义是标量那份，所以跟它对
    ref = np.stack([F8._extract_one(X8[i], HZ) for i in range(len(X8))])
    got = f5.extract_features(X5, HZ, show_progress=False)
    if got.shape[1] != len(idx):
        return False, f"features5 给 {got.shape[1]} 维，去掉 gyr 应该是 {len(idx)} 维"
    if not np.array_equal(got, ref[:, idx]):
        d = np.abs(got - ref[:, idx])
        bad = np.unique(np.where(d > 0)[1])
        names = f5.feature_names()
        return False, (f"{len(bad)} 维对不上，最大差 {float(d.max()):.3g}："
                       + ", ".join(names[i] for i in bad[:8]))
    print(f"  {got.shape[1]} 维，跟 193 维里的非陀螺仪部分逐位相同")
    return True, ""


def check_names_line_up():
    mine = f5.feature_names()
    theirs = [n for n in F8.feature_names(8) if "gyr" not in n]
    if mine != theirs:
        diff = [(a, b) for a, b in zip(mine, theirs) if a != b]
        return False, f"特征名对不上，前几个：{diff[:5]}（长度 {len(mine)} vs {len(theirs)}）"
    print(f"  {len(mine)} 个特征名一一对上")
    return True, ""


def check_repo_extractor_would_lose_34_dims():
    """这个目录存在的理由，钉住它。

    仓库那份 `_extract_one` 里，全局/模长/jerk 三块挂在 window.shape[1] >= 6
    底下。5 通道时整块跳过，只剩 95 维——少掉的 34 维只靠加速计就能算。
    """
    _, X5 = _pair()
    naive = F8._extract_one(X5[0], HZ)
    if len(naive) != 95:
        return False, (f"仓库那份对 5 通道给 {len(naive)} 维，预期 95。"
                       "src/ml/features.py 改过了？这个目录要跟着复查一遍")
    gap = f5.N_FEATURES - len(naive)
    # 95 里含 pitch/roll 的频域 16 维，而 113 那份故意不算（姿态角不是振荡信号，
    # 跟 8 通道那份保持一致）。所以差值不是简单的"少 34 维"，两边各有各的
    acc_derived = 11 + 8 + 11 + 1 + 3   # acc 模长(时域+频域) + jerk + SMA + 三轴相关
    print(f"  仓库那份对 5 通道只给 {len(naive)} 维（113 - 95 = {gap}）")
    print(f"  少的是 acc 模长/jerk/SMA/三轴相关 共 {acc_derived} 维，"
          "这些只靠加速计就能算")
    return True, ""


CHECKS = [
    ("① 砍掉陀螺仪列 ≡ 从来没采过", check_dropping_gyro_equals_never_having_it),
    ("② 113 维 = 193 维去掉 gyr，逐位相同", check_113_is_the_gyro_free_subset_of_193),
    ("③ 特征名一一对上", check_names_line_up),
    ("④ 不新写这一份会少掉哪些维", check_repo_extractor_would_lose_34_dims),
]


def main() -> int:
    ok = True
    for title, fn in CHECKS:
        print(title)
        good, why = fn()
        ok &= good
        print("   " + ("✓ 通过" if good else f"✗ {why}"))
        print()
    print("全部通过。" if ok else "有检查没过，别用这条路训出来的模型。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
