"""把 8 通道的预处理结果转成 **5 通道（acc 三轴 + pitch/roll）**，也就是
一个只有加速度计的设备本来就该有的样子。

    python acc3/make_acc3_npz.py \
        --src data/processed_2026_8_11-2026_8_27_raw_missing_drop_window \
        --dst data/processed_2026_8_11-2026_8_27_raw_missing_drop_window_acc3 \
        --hz 16

## 为什么可以直接砍列，不用重跑一遍预处理

因为**重力对齐的旋转矩阵只从加速度计算**：

    src/data/gravity_align.py:28   g_est = acc.mean(axis=0)
    src/data/gravity_align.py:57   out[:, :3] = (R @ acc.T).T
    src/data/gravity_align.py:59   out[:, 3:6] = (R @ gyr.T).T   ← 只是把同一个 R 也用在陀螺仪上

pitch/roll 同理，`raw_tilt()` 只读 `acc[:, :3]`。

所以 `[acc 对齐后, pitch, roll]` 这五列，**有没有采陀螺仪都是同一个数**——
不是"约等于"，是逐位相同（`acc3/selfcheck5.py` 会验）。
既然如此就没必要重跑一遍预处理：重跑的话窗口划分的随机性会变，
基线和 3 轴版的差值里就混进了别的东西，不能全归因于陀螺仪。

## 一个**不**等价的地方，要知道

`--missing_strategy drop_window` 会丢掉含 NaN 的窗口。有些窗口是**因为陀螺仪
那几列断联**才被丢掉的——真的只有加速度计的设备上，那些窗口会留下来。

所以这份 5 通道数据的窗口集合 = 基线的窗口集合，**比真 3 轴设备偏少一点**。
对"砍掉陀螺仪影响多大"这个问题来说这是好事：两边窗口完全一样，
差值只能来自通道。要量那部分影响的话得从 CSV 重跑，那是另一件事。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 8 通道里要留下的列：acc 三轴 + pitch/roll。跳过 3:6 的陀螺仪
KEEP = [0, 1, 2, 6, 7]
SPLITS = ("train", "val", "test")
FEAT_CACHE = "ml_features.npz"


def to_acc3(path_in: str, path_out: str) -> dict:
    with np.load(path_in, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    X = data["X"]
    if X.ndim != 3:
        raise SystemExit(f"{path_in} 里的 X 是 {X.shape}，要的是 [N, T, C]")
    if X.shape[2] == len(KEEP):
        raise SystemExit(f"{path_in} 已经是 {X.shape[2]} 通道了，不用再转。")
    if X.shape[2] != 8:
        raise SystemExit(
            f"{path_in} 是 {X.shape[2]} 通道，这个脚本只认 8 通道"
            "（acc3 + gyro3 + pitch/roll）。")

    data["X"] = np.ascontiguousarray(X[:, :, KEEP])
    # n_channels 存在 npz 里，**必须跟着改**——不改的话 train.py 那边
    # 读出来还是 8，而实际是 5，后面每一处按它算的东西都错，且不报错
    data["n_channels"] = np.asarray("5")
    os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
    np.savez_compressed(path_out, **data)
    return {"n": int(X.shape[0])}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="基线那份预处理目录（8 通道）")
    ap.add_argument("--dst", required=True, help="输出目录，**必须是新的**")
    ap.add_argument("--hz", type=int, default=16)
    args = ap.parse_args()

    src = os.path.join(args.src, f"{args.hz}hz")
    dst = os.path.join(args.dst, f"{args.hz}hz")
    if not os.path.isdir(src):
        print(f"✗ 找不到 {src}。--src 要给预处理的输出目录"
              f"（里面有 {args.hz}hz/train.npz），--hz 要跟它对上。", file=sys.stderr)
        return 2
    if os.path.abspath(src) == os.path.abspath(dst):
        print("✗ --src 和 --dst 是同一个目录，就地砍列会把基线数据毁掉。", file=sys.stderr)
        return 2

    os.makedirs(dst, exist_ok=True)
    total = 0
    for split in SPLITS:
        fin = os.path.join(src, f"{split}.npz")
        if not os.path.exists(fin):
            if split == "train":
                print(f"✗ {fin} 不存在，--src 给错了？", file=sys.stderr)
                return 2
            print(f"  {split:<6} 没有这个划分，跳过")
            continue
        info = to_acc3(fin, os.path.join(dst, f"{split}.npz"))
        total += info["n"]
        print(f"  {split:<6} {info['n']:>7} 个窗口 → 5 通道")

    # 旧特征缓存**必须删**。它是 193 维的，而 5 通道这条是 113 维——
    # 维度不一样，train.py 那边会自己重建；但目标目录如果是上一轮跑剩的，
    # 里面可能已经是 113 维了，那就不会重建，于是用的是上一轮的数据
    stale = os.path.join(dst, FEAT_CACHE)
    if os.path.exists(stale):
        os.remove(stale)
        print(f"  已删掉旧的特征缓存 {stale}")

    print(f"\n✅ 共 {total} 个窗口 → {dst}/")
    print("   通道：acc_x, acc_y, acc_z（重力对齐后）, pitch, roll（对齐前算的）")
    print("   陀螺仪三轴整个不要了。")
    print("\n下一步：")
    print(f"  python acc3/train_acc3.py --hz {args.hz} --model rf \\")
    print(f"    --processed_dir {args.dst} \\")
    print("    --remap configs/remap_custom_3class.yaml \\")
    print("    --results_dir results_acc3 --feat_workers -1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
