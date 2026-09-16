"""盯住 `train_acc3.py` 那个特征函数替换真的替上了。

    python -m pytest acc3/test_train_acc3.py -q

**为什么这件事值得单独测**：没替上不会报错。5 通道的 X 喂给 8 通道那份
`extract_features`，它会走 `window.shape[1] >= 6` 为假的分支，给 95 维——
形状合法、训练照跑、指标照出，只是少了 34 维（acc 模长、jerk、SMA、三轴相关），
而那 34 维只靠加速计就能算，本来不该少。指标偏低，原因看不出来。

这台机器上没装 sklearn / joblib（离线装不上），而 `src/ml/train.py` 在模块
顶层 import 它们。所以这里打桩——**验的仍然是 train_acc3.py 的真代码**，
桩只负责让 import 过得去。
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REPO, "src"), os.path.join(_REPO, "src", "data"),
           os.path.join(_REPO, "src", "ml")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import features5  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _stubs():
    """给 train.py 顶层那几个 import 打桩（只在真的没装时）。"""
    added = []

    def stub(name, **attrs):
        if name in sys.modules:
            return
        try:
            __import__(name)
            return
        except ImportError:
            pass
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if sub not in sys.modules:
                m = types.ModuleType(sub)
                sys.modules[sub] = m
                added.append(sub)
        for k, v in attrs.items():
            setattr(sys.modules[name], k, v)

    class _Any:
        def __init__(self, *a, **k):
            pass

    stub("joblib", dump=lambda *a, **k: None, load=lambda *a, **k: None,
         Parallel=_Any, delayed=lambda f: f)
    stub("sklearn.ensemble", RandomForestClassifier=_Any, ExtraTreesClassifier=_Any,
         HistGradientBoostingClassifier=_Any)
    stub("sklearn.svm", SVC=_Any)
    stub("sklearn.pipeline", Pipeline=_Any)
    stub("sklearn.preprocessing", StandardScaler=_Any, LabelEncoder=_Any)
    stub("sklearn.model_selection", train_test_split=lambda *a, **k: None)
    stub("sklearn.metrics", classification_report=lambda *a, **k: {},
         accuracy_score=lambda *a, **k: 0.0, f1_score=lambda *a, **k: 0.0,
         confusion_matrix=lambda *a, **k: np.zeros((1, 1)))
    stub("xgboost", XGBClassifier=_Any)
    stub("lightgbm", LGBMClassifier=_Any)
    stub("catboost", CatBoostClassifier=_Any)
    yield
    for name in added:
        sys.modules.pop(name, None)


def test_patch_replaces_the_extractor_in_train_module():
    """要替的是 `train.extract_features`，不是 `features.extract_features`。

    train.py 写的是 `from features import extract_features`——那是**模块级绑定**，
    改 features 模块里的名字对已经 import 过的 train 没有任何影响。
    """
    import train_acc3

    train = train_acc3._patch()
    assert train.extract_features is features5.extract_features


def test_patched_extractor_gives_113_not_95():
    """替上之后 5 通道要给 113 维。95 维就是没替上（走了仓库那份）。"""
    import train_acc3

    train = train_acc3._patch()
    X = np.zeros((3, 16, 5), np.float32)
    got = train.extract_features(X, 16, show_progress=False)
    assert got.shape[1] == 113, f"给了 {got.shape[1]} 维"
    assert got.shape[1] != 95, "没替上：这是仓库那份对 5 通道的结果"


def test_patch_signature_matches_the_original():
    """train.py 是按 `extract_features(X, hz, workers=...)` 调的。

    参数名/顺序不一样的话，替换本身会成功，而爆炸发生在训练跑起来之后——
    那时候前面的类别分布已经打印了一大屏。
    """
    import inspect

    import features as F8

    a = list(inspect.signature(F8.extract_features).parameters)
    b = list(inspect.signature(features5.extract_features).parameters)
    assert a == b, f"签名对不上：仓库 {a}，这边 {b}"


def test_wrong_channel_count_is_rejected_early(tmp_path):
    """拿 8 通道的目录跑要**在最前面**就停。

    不停的话 features5 也会拒收，但那要等到特征提取那一步，
    人容易以为前面都在正常跑。
    """
    import train_acc3

    d = tmp_path / "16hz"
    d.mkdir(parents=True)
    np.savez_compressed(d / "train.npz", X=np.zeros((2, 16, 8), np.float32),
                        y=np.zeros(2))
    with pytest.raises(SystemExit) as e:
        train_acc3._check_channels(str(tmp_path), 16)
    assert "8 通道" in str(e.value)


def test_five_channel_dir_passes_the_check(tmp_path):
    import train_acc3

    d = tmp_path / "16hz"
    d.mkdir(parents=True)
    np.savez_compressed(d / "train.npz", X=np.zeros((2, 16, 5), np.float32),
                        y=np.zeros(2))
    train_acc3._check_channels(str(tmp_path), 16)      # 不该抛


# ── npz 转换 ──────────────────────────────────────────────────────────────


def test_npz_conversion_keeps_acc_and_tilt_bit_exact(tmp_path):
    import make_acc3_npz

    rng = np.random.default_rng(0)
    X = rng.normal(0, 3, (20, 16, 8)).astype(np.float32)
    src = tmp_path / "a.npz"
    np.savez_compressed(src, X=X, y=rng.integers(0, 5, 20), n_channels="8")
    dst = tmp_path / "b.npz"
    make_acc3_npz.to_acc3(str(src), str(dst))
    with np.load(dst) as z:
        out, n_ch = z["X"], z["n_channels"]
    assert out.shape == (20, 16, 5)
    assert np.array_equal(out[:, :, 0:3], X[:, :, 0:3]), "acc 被动过了"
    assert np.array_equal(out[:, :, 3:5], X[:, :, 6:8]), "pitch/roll 取错列了"
    # meta 里的 n_channels **必须跟着改**：不改的话下游按 8 算，且不报错
    assert str(n_ch) == "5"


def test_npz_conversion_refuses_a_five_channel_input(tmp_path):
    """已经转过的再转一次要拒绝——不拒的话会把 pitch/roll 当成陀螺仪砍掉。"""
    import make_acc3_npz

    src = tmp_path / "a.npz"
    np.savez_compressed(src, X=np.zeros((2, 16, 5), np.float32))
    with pytest.raises(SystemExit):
        make_acc3_npz.to_acc3(str(src), str(tmp_path / "b.npz"))
