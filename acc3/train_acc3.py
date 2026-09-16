"""在 5 通道（acc 三轴 + pitch/roll）数据上训练，特征用 `acc3/features5.py`（113 维）。

    python acc3/train_acc3.py --hz 16 --model rf \
        --processed_dir data/processed_..._acc3 \
        --remap configs/remap_custom_3class.yaml \
        --results_dir results_acc3 --feat_workers -1

参数跟 `src/ml/train.py` 一样（少了合成数据那几个，见下）。

## 这个脚本干的唯一一件事：换掉特征提取函数

训练流程本身**整段用 `src/ml/train.py:main()`**——标签重映射、类别分布打印、
特征缓存、验证集兜底、评估、保存 pkl+json，一行都没重写。
抄一份的话，重映射那套（哪些类别合并、哪些丢掉）迟早跟原版分家，
而分家的表现是"两个模型指标能比，但比的不是同一件事"。

所以这里只做一件事：把 `train.extract_features` 换成 5 通道那份。

**为什么是替换模块属性而不是改 train.py**：仓库原有文件不动是这个项目的规矩。
train.py 里是 `from features import extract_features`（模块级绑定），
所以要替的是 `train.extract_features` 这个名字，替 `features.extract_features`
没用——那时候 train 模块里的绑定已经指向老函数了。

替换完会**当场核一遍**（见 `_patch`）：核不过就退出，不往下训。
悄悄没替上的后果是：5 通道的 X 喂给 8 通道那份 extract_features，
得到 95 维——**不报错**，训出来的模型少 34 维（acc 模长、jerk、SMA、三轴相关），
而那 34 维只靠加速计就能算，本来不该少。指标会偏低，而偏低的原因看不出来。

## 不支持合成数据

`--synthetic*` 那条路在 train.py 里会把 6 通道的合成窗口补成 8 通道
（`append_raw_tilt_batch(X_syn)[:, :, 6:8]` 那一段），跟 5 通道对不上。
你现在的训练命令带的是 `--skip_syn`，所以用不到。要用的话得先把那段
也做一份 5 通道的，别指望它自己能对。
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "src"), os.path.join(_REPO, "src", "data"),
           os.path.join(_REPO, "src", "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import features5  # noqa: E402


def _patch():
    """换掉 train.py 里的特征提取，并**当场核一遍**。"""
    import train

    train.extract_features = features5.extract_features
    if train.extract_features is not features5.extract_features:
        sys.exit("特征函数没替上（train.extract_features 还是老的）")
    # 真跑一遍，别只看指针：签名对不上的话要在这里炸，不是训到一半才炸
    probe = np.zeros((2, 16, features5.N_CHANNELS), np.float32)
    got = train.extract_features(probe, 16, show_progress=False)
    if got.shape[1] != features5.N_FEATURES:
        sys.exit(f"替上了但维度不对：{got.shape[1]}，应该是 {features5.N_FEATURES}")
    return train


def _check_channels(processed_dir, hz):
    """npz 必须是 5 通道。

    拿 8 通道的目录跑这个脚本**不会报错**：features5 会拒收，但那要等到
    特征提取那一步，前面的类别分布已经打印了一大屏，人容易以为在正常跑。
    所以在最前面就查。
    """
    p = os.path.join(processed_dir, f"{hz}hz", "train.npz")
    if not os.path.exists(p):
        sys.exit(f"找不到 {p}。--processed_dir 要给 5 通道那份"
                 f"（用 acc3/make_acc3_npz.py 生成），--hz 要跟它对上。")
    with np.load(p, allow_pickle=False) as z:
        c = int(z["X"].shape[2])
    if c != features5.N_CHANNELS:
        sys.exit(
            f"{p} 是 {c} 通道，这个脚本要 {features5.N_CHANNELS} 通道"
            "（acc 三轴 + pitch/roll）。\n"
            "  先转一下：python acc3/make_acc3_npz.py --src <8通道目录> "
            "--dst <新目录> --hz %d\n"
            "  8 通道的数据请照常用 src/ml/train.py 训（那是基线）。" % hz)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hz", type=int, required=True,
                    choices=[5, 10, 15, 16, 20, 25, 50])
    ap.add_argument("--model", default="rf")
    ap.add_argument("--config", default="configs/ml.yaml")
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--results_dir", default="results_acc3")
    ap.add_argument("--remap", default="")
    ap.add_argument("--feat_workers", type=int, default=1)
    ap.add_argument("--n_jobs", type=int, default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    _check_channels(args.processed_dir, args.hz)
    train = _patch()
    if args.model not in train.MODELS:
        sys.exit(f"--model 只支持 {sorted(train.MODELS)}")

    print(f"[acc3/train] 5 通道（acc3 + pitch/roll），"
          f"{features5.N_FEATURES} 维特征")

    # train.main() 还会读这几个合成数据相关的参数。给成"没有"，
    # 而不是不给——不给的话是 AttributeError，看不出是这里的问题
    args.synthetic_spec = None
    args.synthetic = ""
    args.synthetic_label = ""
    args.synthetic_hz = 0
    train.main(args)
    _stamp(args)
    return 0


def _stamp(args):
    """往 train.py 写出来的 ml_*.json 里补几个字段。

    train.py 的 result 里**没有** n_channels / n_features / 特征名——8 通道
    那条路上只有一种可能，所以不用记。现在有两种了，不记的话推理侧
    只能靠"维度对得上"来判断，而 113 跟 193 都是合法的维度，
    对不上时 sklearn 的报错（"expecting 113 features"）会让人去查特征提取，
    而真正的原因是"这个模型不是那套特征训的"。

    顺便把**该取哪 113 列**也存进去。端侧服务照常算 193 维，然后按这个
    下标取列，就能直接跑 5 通道的模型——不需要一条新的预处理链。
    存下标而不是让服务自己按名字筛：服务那边依赖的是它自己那份 imu_train，
    版本一旦不同，筛出来的下标会**整体错位**，而错位不报错。
    存 from_dim 是为了让错位当场暴露：来的不是 193 维就报错。
    """
    import json

    from features import feature_names as names8

    remap_tag = (f"_{os.path.splitext(os.path.basename(args.remap))[0]}"
                 if args.remap else "")
    out = os.path.join(args.results_dir,
                       os.path.basename(args.processed_dir.rstrip("/")),
                       f"{args.hz}hz{remap_tag}", args.model,
                       f"ml_{args.model}.json")
    if not os.path.exists(out):
        print(f"[acc3/train] 没找到 {out}，跳过补字段（训练是不是失败了？）")
        return
    with open(out, encoding="utf-8") as f:
        d = json.load(f)
    idx = features5.gyro_free_indices()
    d["n_channels"] = features5.N_CHANNELS
    d["n_features"] = int(features5.N_FEATURES)
    d["channel_names"] = list(features5.CHANNEL_NAMES)
    d["feature_set"] = "acc3_gyro_free_113"
    d["feature_select"] = {"from_dim": len(names8(8)),
                           "indices": [int(i) for i in idx]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"[acc3/train] 已在 {out} 里记下通道数/特征维度/取列下标")


if __name__ == "__main__":
    sys.exit(main())
