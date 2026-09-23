"""3 轴 / 6 轴：喂几个通道必须跟模型训练时一致。

拿 6 轴数据喂 3 轴模型（或反过来）特征维数就对不上——轻则报错，重则悄悄
算出一串没意义的数。所以通道数要从模型元数据一路带到推理。

另一件同样要命的事：3 轴和 6 轴的预处理产物（窗口张量）通道数不同，**不能
共用一个目录**。先跑 6 轴再跑 3 轴，没加 --clean 的话会直接拿旧的 6 轴 npz
接着训，训出来的模型标着 3 轴、实际吃的是 6 轴数据，一点报错都没有。
"""

import re
import subprocess

import numpy as np
import pytest


def _sh(*args):
    """DRY_RUN 跑一遍 train_custom.sh，只看它拼出来的命令。"""
    out = subprocess.run(["bash", "-n", "train_custom.sh"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_脚本语法没坏():
    _sh()


def test_预处理目录按轴数分开():
    """同一份数据、同一个标签，3 轴和 6 轴要落到不同目录。"""
    src = open("train_custom.sh", encoding="utf-8").read()
    assert '_axes_suffix="_acc3"' in src, "3 轴没有单独的目录后缀，会跟 6 轴互相覆盖"
    line = next(l for l in src.splitlines() if l.startswith("PROCESSED_DIR="))
    assert "${_axes_suffix}" in line


def test_赋值里不能用会失败的命令替换():
    """脚本开着 set -e。VAR="$( [[ 条件 ]] && echo x )" 在条件为假时，命令替换
    返回 1，赋值语句就继承这个 1——整个脚本在这一行悄无声息地退出（此时输出
    还没 tee 进日志，一个字都不打）。

    第一版 3 轴就是这么写的，结果默认 6 轴的训练全都起不来。当命令参数用
    （cmd $( [[ ]] && echo --x )）不受影响，只查赋值。"""
    import re

    src = open("train_custom.sh", encoding="utf-8").read()
    bad = [l for l in src.splitlines()
           if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*=.*\$\(\s*\[\[", l) and not l.lstrip().startswith("#")]
    assert not bad, f"赋值里用了会失败的命令替换，set -e 下会让脚本静默退出：{bad}"


def test_整个仓库都不许写死六到八():
    """append_raw_tilt_batch 是把姿态角追加在**最后两列**的：6 轴时是 6:8，3 轴时是 3:5。

    写死 6:8 最阴的地方是**numpy 不报错**：5 通道的数组取 [:, :, 6:8] 返回的是
    一个宽度为 0 的空数组。于是 3 轴时姿态角被悄悄丢掉，训练照样跑完、模型照样
    保存——2026-09-23 第一次 3 轴训练，「纯标注」那一版就是这么训出来的，
    只有 3 路、没有 pitch/roll。

    这个坑前后踩了三回（推理、训练里的合成数据、预处理），所以扫整个仓库。
    统一写 -2:，6 轴时跟 6:8 取的是同一列，一个数都不变。"""
    import pathlib

    bad = []
    for root in ("src", "label_service"):
        for f in pathlib.Path(root).rglob("*.py"):
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if "[:, :, 6:8]" in line and not line.lstrip().startswith("#"):
                    bad.append(f"{f}:{i}")
    assert not bad, f"又写死了 6:8，3 轴时会悄悄变成空数组：{bad}"


def test_写死六到八在三轴时确实是悄悄变空():
    """上面那条规矩的依据——numpy 越界切片不报错。它哪天变了要能知道。"""
    X5 = np.zeros((2, 4, 5), dtype=np.float32)
    assert X5[:, :, 6:8].shape[2] == 0
    assert X5[:, :, -2:].shape[2] == 2


def test_预处理后三轴是五通道六轴是八通道():
    """姿态角两列必须在。3 轴时不在的话，模型是在只有 3 路的数据上训的。"""
    from src.data.gravity_align import append_raw_tilt_batch, gravity_align_batch

    for c, want in ((3, 5), (6, 8)):
        X = np.zeros((2, 16, c), dtype=np.float32)
        X[:, :, 2] = 1.0
        tilt = append_raw_tilt_batch(X)[:, :, -2:]
        out = np.concatenate([gravity_align_batch(X), tilt], axis=2)
        assert out.shape[2] == want, f"{c} 轴预处理后该有 {want} 通道，实际 {out.shape[2]}"


def test_姿态角函数本来就支持三通道():
    """这是"管线本来就能跑 3 轴"的依据，不是我们新改的——它变了要能发现。"""
    from src.data.gravity_align import append_raw_tilt_batch

    X3 = np.zeros((2, 8, 3), dtype=np.float32)
    X3[:, :, 2] = 1.0                       # z 轴 1g，平放
    out = append_raw_tilt_batch(X3)
    assert out.shape == (2, 8, 5), "3 通道进去该出 5 通道（追加 pitch/roll）"
    X6 = np.zeros((2, 8, 6), dtype=np.float32)
    X6[:, :, 2] = 1.0
    assert append_raw_tilt_batch(X6).shape == (2, 8, 8)


def test_特征维数按通道数自适应():
    """features.py 在 <6 通道时会跳过 gyro 相关那几组。"""
    from src.ml.features import feature_names

    n8 = len(feature_names(8))
    n5 = len(feature_names(5))
    assert n5 < n8, "3 轴的特征数该比 6 轴少"
    # **5 通道时第 4、5 个通道其实是 pitch/roll，不是 gyro。**
    # 按下标直接查 CHANNEL_NAMES 会把它们叫成 gyr_x/gyr_y——数值没错，
    # 但看特征重要性时完全是误导
    assert not [n for n in feature_names(5) if "gyr" in n]
    assert [n for n in feature_names(8) if "gyr" in n]
    assert [n for n in feature_names(5) if n.startswith("pitch_")]
    assert [n for n in feature_names(5) if n.startswith("roll_")]


def test_频域只算传感器通道_不算姿态角():
    """姿态角是慢变量，不是振荡信号，对它做 FFT 没意义。
    原来写死 n_ch=6：6 轴时总通道 8、传感器 6 正好对上，看不出问题；
    3 轴时总通道 5，写死 6 就把 pitch/roll 也当振荡信号算进去了。"""
    import sys

    sys.path.insert(0, "src/ml")
    from features import n_sensor_channels

    assert n_sensor_channels(8) == 6      # 6 轴：acc3+gyro3
    assert n_sensor_channels(5) == 3      # 3 轴：只有 acc3
    # 频域特征名只出现在传感器通道上
    from features import feature_names

    freq = [n for n in feature_names(5) if "_spec_mean" in n]
    assert len(freq) == 3, f"3 轴该只有 3 路频域，实际 {freq}"


def test_两份特征实现逐位一致():
    """features.py 是逐窗口循环，features_vec.py 是向量化，真正跑的是后者。
    两边都要按通道数自适应，差一点模型就吃到对不上的特征。"""
    import sys

    sys.path.insert(0, "src/ml")
    from features import _extract_one, feature_names
    from features_vec import extract_features_vec

    rng = np.random.default_rng(0)
    for c in (8, 5):
        X = rng.normal(size=(4, 32, c)).astype("float32")
        slow = np.stack([_extract_one(X[i], 16) for i in range(len(X))])
        fast = extract_features_vec(X, 16)
        assert slow.shape == fast.shape
        assert np.allclose(slow, fast, atol=1e-5, rtol=1e-4), f"{c} 通道时两份实现对不上"
        assert len(feature_names(c)) == slow.shape[1], f"{c} 通道的特征名数量对不上特征维数"


def test_合成数据按轴数生成(monkeypatch):
    """合成数据的通道数必须跟真实数据一致。3 轴时还吐 6 路的话，train.py 把两边
    拼起来直接崩——2026-09-23 第一次 3 轴训练，「带合成」那一版就是这么挂的。"""
    src = open("src/data/synthesize_scratch.py", encoding="utf-8").read()
    assert "if axes == 3:" in src
    sh = open("train_custom.sh", encoding="utf-8").read()
    # 调合成脚本那一段里要把 --axes 传过去
    seg = sh[sh.index("python src/data/synthesize_scratch.py"):]
    seg = seg[: seg.index("SYNTHETIC_SPEC_ARGS")]
    assert '--axes "$AXES"' in seg, "train_custom.sh 调合成脚本时没传 --axes"


def test_训练时合成数据路数多了就切_少了说人话():
    src = open("src/ml/train.py", encoding="utf-8").read()
    assert "X_syn = X_syn[:, :, :want_sensor]" in src
    assert "合成数据是按 3 轴生成的" in src
