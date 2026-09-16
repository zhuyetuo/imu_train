"""把已经预处理好的窗口复制一份出来，**陀螺仪三个通道置零**。

    python acc_only/make_acc_only_npz.py \
        --src data/processed_2026_8_11-2026_8_27_raw_missing_drop_window \
        --dst data/processed_2026_8_11-2026_8_27_raw_missing_drop_window_acc_only \
        --hz 16

## 为什么是"置零"而不是"只留 3 通道"

想回答的问题是「**只有加速计，效果掉多少**」。有两种做法：

  1. 真的只留 acc 三列（configs 里把 sensor_cols 改成 3 个）
  2. 保留 8 通道的形状，把 gyr_x/y/z 置零

选 2，有三个理由：

**a) 做法 1 会连带砍掉一堆本来算得出来的特征。**
src/ml/features.py 的 `_extract_one` 里，全局特征、模长特征、jerk 特征全都
挂在 `window.shape[1] >= 6` 这个条件下。给 3 通道的话，这一整块直接跳过——
acc 模长、acc jerk、acc 三轴相关系数、SMA 全没了，**而这些只靠加速计就能算**。
pitch/roll 也没了（preprocess.py 里取的是 `[:, :, 6:8]`，3 通道时那是空切片，
**不报错，静默变成没有姿态角**）。最后只剩 57 维，其中一大半信息是被
实现细节砍掉的，不是"加速计本来就没有"。拿它跟 193 维的基线比，
比出来的是两套特征工程的差距，不是"有没有陀螺仪"的差距。

**b) 做法 2 的窗口跟基线逐位相同，只有陀螺仪那 6 列不一样。**
同一份 npz 复制出来的，窗口切分、训练/验证划分、标签、重力对齐全都一样。
所以准确率的差值**只能**归因于陀螺仪。做法 1 要重跑一遍预处理，
划分的随机性、窗口边界都可能不一样，差值里就混进了别的东西。

**c) 做法 2 训出来的模型能直接在现有服务上跑。**
特征还是 193 维，几何还是 16 点 @16Hz，label_service / algo_service /
端侧服务一行代码都不用改。而做法 1 的 57 维模型喂进去会当场报
"X has 193 features, but RandomForestClassifier is expecting 57"。

## 置零真的等于"没有陀螺仪"吗

等于。置零之后那 6 列在所有窗口上都是同一个常数，由它们算出来的特征
也全是常数——**常数特征的信息增益恒为 0，树模型永远不会选它**。
实测 193 维里正好 80 维变成常数（gyr_x/y/z 各 19 维 + gyro_mag 19 维 +
sma_gyro + 三个 corr_gyro），剩下 113 维全部只依赖加速计和它派生的
pitch/roll/acc 模长/jerk。`acc_only/selfcheck.py` 会把这件事验一遍。

要提醒的一点：这**不是**端侧成本的验证。板子上真做 3 轴的话只用算 57 维
（core/tm_features.c 的 n_ch<6 分支），而这里仍然是 193 维的计算量。
这一版回答的是"效果"，不是"省多少 flash / 多少 ms"。

## 不复制特征缓存

`<hz>hz/ml_features.npz` 是 193 维特征的缓存。**维度一样**，所以
src/ml/train.py 那边"维度对不上就重建"的保护**不会触发**——复制过去的话
训练读到的是原始（带陀螺仪）的特征，训出来的模型跟基线一模一样，
而日志里没有任何异常，指标也完全正常。这是这个脚本最容易出的错，
所以这里不但不复制，还会主动删掉目标目录里已有的那份。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 陀螺仪在通道里的位置。跟 src/ml/features.py 顶部那份约定一致：
#   0:3 acc / 3:6 gyr / 6:8 pitch,roll
GYRO_SLICE = slice(3, 6)
SPLITS = ("train", "val", "test")
FEAT_CACHE = "ml_features.npz"


def zero_gyro(path_in: str, path_out: str) -> dict:
    """复制一个 npz，X 的陀螺仪通道置零，其余键**原样带过去**。

    meta 是以零维数组的形式跟 X/y 一起存在 npz 里的（见 preprocess.py），
    所以这里不能只存 X/y——漏掉的话 train.py 读不到 window_size/classes，
    而报错信息会是 KeyError，跟"少复制了几个键"完全看不出关系。
    """
    with np.load(path_in, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    X = data["X"]
    if X.ndim != 3:
        raise SystemExit(f"{path_in} 里的 X 是 {X.shape}，要的是 [N, T, C]")
    n_ch = X.shape[2]
    if n_ch < 6:
        # 已经是只有加速计的数据了，再置零没有意义——而且会让人以为
        # 自己成功做了一次"去掉陀螺仪"，其实什么都没做
        raise SystemExit(
            f"{path_in} 只有 {n_ch} 个通道，本来就没有陀螺仪，不用跑这个脚本。")

    before = X.copy()
    X = X.copy()
    X[:, :, GYRO_SLICE] = 0.0
    data["X"] = X

    os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
    np.savez_compressed(path_out, **data)

    # 自检：加速计和姿态角必须**逐位不变**。这里用 array_equal 而不是 allclose——
    # 复制过程中如果不小心过了一次 float32→float64→float32，值会几乎一样
    # 但不是同一个数，而那点差别足以让"跟基线唯一的差别是陀螺仪"这句话不成立
    acc_same = np.array_equal(before[:, :, :3], X[:, :, :3])
    tail_same = np.array_equal(before[:, :, 6:], X[:, :, 6:])
    if not (acc_same and tail_same):
        raise SystemExit("置零过程动到了加速计或姿态角通道——这是 bug，别用这份数据")
    changed = int(np.count_nonzero(before[:, :, GYRO_SLICE]))
    return {"n": int(X.shape[0]), "n_ch": n_ch, "changed": changed}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="已有的预处理目录（8 通道那份）")
    ap.add_argument("--dst", required=True, help="输出目录，**必须是新的**")
    ap.add_argument("--hz", type=int, default=16)
    args = ap.parse_args()

    src = os.path.join(args.src, f"{args.hz}hz")
    dst = os.path.join(args.dst, f"{args.hz}hz")
    if not os.path.isdir(src):
        return _die(f"找不到 {src}。--src 要给预处理的输出目录"
                    f"（里面有 {args.hz}hz/train.npz），--hz 要跟它对上。")
    if os.path.abspath(src) == os.path.abspath(dst):
        return _die("--src 和 --dst 是同一个目录。就地置零会把基线数据毁掉，"
                    "而且不可逆——换个 --dst。")

    os.makedirs(dst, exist_ok=True)
    total = 0
    for split in SPLITS:
        fin = os.path.join(src, f"{split}.npz")
        if not os.path.exists(fin):
            # test 集可以是空的（train_custom.sh 默认 test_ratio=0），
            # 但 train 不见了就是路径给错了
            if split == "train":
                return _die(f"{fin} 不存在，--src 给错了？")
            print(f"  {split:<6} 没有这个划分，跳过")
            continue
        info = zero_gyro(fin, os.path.join(dst, f"{split}.npz"))
        total += info["n"]
        print(f"  {split:<6} {info['n']:>7} 个窗口，{info['n_ch']} 通道，"
              f"置零了 {info['changed']} 个非零的陀螺仪采样点")

    # 见模块顶部："不复制特征缓存"。这里是**删**，不是跳过不复制——
    # 目标目录可能是上一轮跑剩下的，里面那份缓存是上一轮的特征
    stale = os.path.join(dst, FEAT_CACHE)
    if os.path.exists(stale):
        os.remove(stale)
        print(f"  已删掉旧的特征缓存 {stale}（它是按原始数据算的，会悄悄污染这次训练）")

    print(f"\n✅ 共 {total} 个窗口 → {dst}/")
    print("   加速计和 pitch/roll 逐位不变，只有陀螺仪三轴被置零。")
    print("\n下一步：")
    print(f"  python src/ml/train.py --hz {args.hz} --model rf \\")
    print(f"    --processed_dir {args.dst} \\")
    print("    --remap configs/remap_custom_3class.yaml \\")
    print("    --results_dir results_acc_only")
    return 0


def _die(msg: str) -> int:
    print(f"✗ {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
