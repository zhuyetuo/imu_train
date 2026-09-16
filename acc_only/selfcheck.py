"""验三件事，不需要训练数据，也不需要显卡：

  1. 陀螺仪置零之后，193 维特征里**正好 80 维变成常数**，剩下 113 维
     全部只依赖加速计（以及由加速计派生的 pitch/roll、acc 模长、jerk）。
  2. 在这样的数据上训出来的树模型，**永远不会用到陀螺仪那些维度**——
     把陀螺仪换成任意随机值，预测结果逐条不变。
  3. 这一条是给训好的真模型用的（--model 指向 ml_rf.pkl）：拿真实窗口
     跑两遍，一遍陀螺仪原样、一遍陀螺仪随机，预测必须完全一致。

用法：
    python acc_only/selfcheck.py                     # 1 和 2
    python acc_only/selfcheck.py --model results_acc_only/.../ml_rf.pkl   # 再加 3

第 2、3 条是这条路线**唯一的风险点**：如果模型其实偷偷用到了陀螺仪，
那它在真机（只有加速计）上就会失效，而在平台上（CSV 里有陀螺仪）看着完全正常。
这种错只会在换到真设备的时候暴露，那时候已经很难查了。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "src", "data"),
           os.path.join(_REPO, "src", "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features import extract_features, feature_names          # noqa: E402
from gravity_align import append_raw_tilt_batch, gravity_align_batch  # noqa: E402

GYRO_SLICE = slice(3, 6)
HZ = 16
WIN = 16


def make_windows(n, seed, gyro="zero"):
    """造 n 个 [T, 8] 窗口，走跟训练一模一样的那条预处理。

    顺序很要紧：pitch/roll 必须在**重力对齐之前**算（对齐会把每个窗口的
    平均倾角归零，绝对姿态就没了）。这里照抄 preprocess.py 的顺序。
    """
    rng = np.random.default_rng(seed)
    acc = (rng.normal(0, 2.0, (n, WIN, 3)).astype(np.float32)
           + np.array([0, 0, 9.8], np.float32))
    if gyro == "zero":
        g = np.zeros((n, WIN, 3), np.float32)
    else:
        g = rng.normal(0, 40.0, (n, WIN, 3)).astype(np.float32)
    X = np.concatenate([acc, g], axis=2)
    tilt = append_raw_tilt_batch(X)[:, :, 6:8]
    return np.concatenate([gravity_align_batch(X), tilt], axis=2), acc


def check_constant_dims():
    X, _ = make_windows(400, seed=1, gyro="zero")
    F = np.asarray(extract_features(X, HZ))
    if not np.isfinite(F).all():
        # 全零通道过 FFT 求谱熵是 0/0，很容易出 NaN——出了的话 RF 会直接拒收，
        # 而报错发生在训练中途，跟"陀螺仪置零"看不出关系
        return False, f"特征里有 {int((~np.isfinite(F)).sum())} 个 NaN/inf"

    names = feature_names(8)
    const = [i for i in range(F.shape[1]) if np.ptp(F[:, i]) == 0]
    gyro_ish = [i for i in const if "gyr" in names[i]]
    if len(const) != len(gyro_ish):
        others = [names[i] for i in const if "gyr" not in names[i]]
        return False, f"有非陀螺仪的维度也变成常数了：{others[:10]}"
    print(f"  193 维里 {len(const)} 维变成常数，全部是陀螺仪派生的")
    print(f"  剩下 {F.shape[1] - len(const)} 维只依赖加速计")

    # 反过来也要成立：把陀螺仪换成随机值，**只有那 80 维会动**。
    # 少了这一步的话，"某个 acc 特征其实偷偷混了陀螺仪"这种事查不出来
    Xr, _ = make_windows(400, seed=1, gyro="random")
    Fr = np.asarray(extract_features(Xr, HZ))
    moved = [i for i in range(F.shape[1]) if not np.allclose(F[:, i], Fr[:, i])]
    leaked = [names[i] for i in moved if i not in const]
    if leaked:
        return False, f"这些非陀螺仪维度受陀螺仪影响：{leaked[:10]}"
    print(f"  陀螺仪换成随机值时，只有那 {len(const)} 维在动，其余逐位不变")
    return True, ""


def check_model_ignores_gyro(model=None):
    """训好（或现训一个）之后，陀螺仪必须对预测毫无影响。"""
    if model is None:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            # 没装 sklearn 时**说清楚是跳过，不是通过**。返回 None 让上层
            # 区分这三种情况：通过 / 没过 / 没验——混成"通过"最要命
            return None, "没装 sklearn，这一条没验（pip install scikit-learn）"
        X, acc = make_windows(300, seed=2, gyro="zero")
        F = np.asarray(extract_features(X, HZ))
        # 标签故意跟加速计强相关，好让树真的学到东西——
        # 随机标签训出来的树几乎不分裂，那样"没用陀螺仪"是因为什么都没用
        y = (acc[:, :, 0].std(axis=1) > acc[:, :, 0].std(axis=1).mean()).astype(int)
        model = RandomForestClassifier(n_estimators=20, max_depth=6,
                                       random_state=0).fit(F, y)

    Xz, _ = make_windows(200, seed=7, gyro="zero")
    Xr = Xz.copy()
    rng = np.random.default_rng(11)
    Xr[:, :, GYRO_SLICE] = rng.normal(0, 40.0, Xr[:, :, GYRO_SLICE].shape)
    Fz = np.asarray(extract_features(Xz, HZ))
    Fr = np.asarray(extract_features(Xr, HZ))
    if Fz.shape[1] != getattr(model, "n_features_in_", Fz.shape[1]):
        return False, (f"模型要 {model.n_features_in_} 维特征，这里算出来 "
                       f"{Fz.shape[1]} 维——这个模型不是这套特征训的")
    pz = model.predict(Fz)
    pr = model.predict(Fr)
    diff = int((pz != pr).sum())
    if diff:
        return False, (f"{diff}/{len(pz)} 条预测因为陀螺仪而改变——"
                       "这个模型用到了陀螺仪，在只有加速计的设备上会失效")
    print(f"  {len(pz)} 条窗口，陀螺仪换成随机值后预测逐条不变")
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="", help="训好的 ml_*.pkl，验第 3 条")
    args = ap.parse_args()

    ok, skipped = True, 0

    def report(good, why):
        """good 是 None = 没验成。**跟"通过"分开报**，不然缺依赖的机器上
        跑一遍全绿，而其实一条都没验。"""
        nonlocal ok, skipped
        if good is None:
            skipped += 1
            print(f"   ⚠ 跳过：{why}")
        elif good:
            print("   ✓ 通过")
        else:
            ok = False
            print(f"   ✗ {why}")

    print("① 陀螺仪置零 → 哪些特征维度失去信息")
    report(*check_constant_dims())

    print("\n② 树模型会不会用到陀螺仪（现场训一个小模型验原理）")
    report(*check_model_ignores_gyro())

    if args.model:
        print(f"\n③ 真模型 {args.model}")
        if not os.path.exists(args.model):
            print("   ✗ 文件不存在")
            ok = False
        else:
            import joblib
            report(*check_model_ignores_gyro(joblib.load(args.model)))
    else:
        print("\n③ 跳过（没给 --model）。训完之后建议补一次：")
        print("   python acc_only/selfcheck.py --model results_acc_only/.../ml_rf.pkl")

    if not ok:
        print("\n有检查没过，别用这个模型。")
        return 1
    print(f"\n通过。{f'（有 {skipped} 条没验成，见上面的 ⚠）' if skipped else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
