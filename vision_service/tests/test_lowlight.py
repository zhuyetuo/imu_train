"""夜视增强：拉伸和多帧堆栈。

要守住的：放大倍数必须封顶（全黑画面的高低分位几乎重合，不封顶就是把纯噪声
放成雪花）；堆栈确实在压噪声（而不是只把画面弄糊）；没配模型权重时要说清楚
缺什么、放哪儿，而不是抛一句看不懂的错。
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_service import lowlight


def test_拉伸把用到的那一小段铺满_倍数封顶():
    # 只用到 16~38 这 22 级：跟实测那一路夜间画面一样
    img = np.random.randint(16, 39, (64, 64, 3), dtype=np.uint8)
    out, info = lowlight.stretch(img)
    assert 15 <= info["lo"] <= 20 and 35 <= info["hi"] <= 40
    assert out.mean() > img.mean() * 3, "22 级铺满之后应该明显变亮"

    # 全黑（高低分位重合）：倍数必须被封住，不然纯噪声被放成雪花，
    # 看着像有东西——那比黑着更糟
    flat = np.full((64, 64, 3), 3, dtype=np.uint8)
    _o, info2 = lowlight.stretch(flat, max_gain=12.0)
    assert info2["gain"] <= 12.0


def test_堆栈是在压噪声_不是把画面弄糊(monkeypatch):
    """固定机位 + 随机噪声：平均之后标准差该掉下来，而画面内容不变。

    这一条是整件事的地基——堆栈要是压不住噪声，后面上任何模型都只是在
    给噪声画画。
    """
    rng = np.random.default_rng(0)
    base = np.zeros((48, 64, 3), dtype="float32")
    base[10:30, 20:40] = 40.0          # "狗"：比背景亮一点
    frames = []
    for _ in range(24):
        noisy = np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype("uint8")
        frames.append((0.0, noisy))
    monkeypatch.setattr(lowlight.seek, "iter_frames", lambda *a, **k: iter(frames))

    avg, info = lowlight.stack("x.mp4", t_s=1.0, window_s=2.0)
    assert info["frames"] == 24 and info["snr_gain"] == pytest.approx(4.9, abs=0.1)
    # 背景那块：单帧噪声 std≈12，平均 24 帧之后应该掉到 1/4 以下
    assert avg[0:8, 0:8].std() < frames[0][1][0:8, 0:8].std() / 3
    # "狗"还在：压噪声不能把内容一起压掉
    assert float(avg[10:30, 20:40].mean()) - float(avg[0:8, 0:8].mean()) > 25


def test_没配权重时说清楚缺什么放哪儿_并且指明该用哪个(monkeypatch):
    """「自动下载失败」那种报错最难查——人只看到一句超时。

    这里不光说缺什么、放哪儿，还要指明**用哪一个**：这批是带真实噪声的监控
    视频，该用 SMID / SDSD_indoor（拿低光视频训的）；LOL_v1/v2 是照片、几乎
    没噪声，拿来只会输出一张好看但不对的图。
    """
    monkeypatch.setattr(lowlight.config, "LOWLIGHT_WEIGHTS", "")
    with pytest.raises(RuntimeError) as e:
        lowlight.run_model(np.zeros((8, 8, 3), dtype=np.uint8))
    msg = str(e.value)
    assert "LOWLIGHT_WEIGHTS" in msg and "SMID" in msg and "LOL_v1" in msg


def test_只加载那一个网络结构文件_不被训练依赖挡住(tmp_path):
    """实测 2026-09-22：仓库和权重都齐了，却卡在 `No module named 'lmdb'`。

    因为走的是 `import basicsr.models.archs...`，那会先执行 basicsr 的 __init__，
    把整个训练框架的依赖一起拉起来（lmdb、tb_logger、数据加载器……）——而 lmdb
    是训练读数据用的，推理一个字节都用不上。

    改成按文件路径单独加载：basicsr/__init__.py 里写什么都不影响。
    """
    repo = tmp_path / "Retinexformer"
    arch = repo / "basicsr" / "models" / "archs"
    arch.mkdir(parents=True)
    # 这个 __init__ 一旦被执行就会炸——正是要证明它**不会**被执行
    (repo / "basicsr" / "__init__.py").write_text("import lmdb  # 训练才用得上\n", encoding="utf-8")
    (arch / "RetinexFormer_arch.py").write_text("class RetinexFormer:\n    pass\n", encoding="utf-8")

    cls = lowlight._import_arch(str(repo))
    assert cls.__name__ == "RetinexFormer"


def test_结构文件自己缺依赖时点名是哪个(tmp_path):
    """basicsr 的训练依赖可以不管，但这个文件自己要的（比如 einops）得说清楚。"""
    repo = tmp_path / "R2"
    arch = repo / "basicsr" / "models" / "archs"
    arch.mkdir(parents=True)
    (arch / "RetinexFormer_arch.py").write_text("import einops_not_installed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="einops_not_installed"):
        lowlight._import_arch(str(repo))


def test_去色噪只动色度不动亮度():
    """压色噪不能改画面内容——亮度通道必须一个字节都不变。

    这是它能摆在「不编造」那一列的前提：去掉的是已知为噪声的色度，
    人看到的轮廓、明暗全是原片里本来就有的。
    """
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    img = np.full((120, 160, 3), 90, np.uint8)
    img[40:90, 50:110] = 45                                  # 一块更暗的，当作狗
    f = img.astype("float32")
    f[:, :, 0] += 25                                         # 洋红偏色：蓝、红一起抬
    f[:, :, 2] += 20
    f += rng.normal(0, 18, f.shape) * np.array([1, 0.2, 1])  # 噪声集中在色度上
    noisy = np.clip(f, 0, 255).astype("uint8")

    out = lowlight.kill_chroma_noise(noisy)

    def 彩度(x):
        ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb).astype("float32")
        return float(np.mean(np.abs(ycc[:, :, 1] - 128) + np.abs(ycc[:, :, 2] - 128)))

    def 亮度(x):
        return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype("float32")

    assert 彩度(out) < 彩度(noisy) / 4                        # 色噪和偏色都压下去了
    assert np.mean(np.abs(亮度(out) - 亮度(noisy))) < 0.5     # 亮度没动
    # 那块"狗"跟背景的反差原样保留——去色不该让轮廓变淡
    反差 = lambda x: 亮度(x)[40:90, 50:110].mean() - 亮度(x)[0:20, 0:20].mean()
    assert abs(反差(out) - 反差(noisy)) < 0.5


def test_整段增强的亮度不能逐帧跳(tmp_path, monkeypatch):
    """全段共用一套拉伸映射——每帧各算各的就会闪得没法看。

    造一段"背景不动、一个小方块在走"的暗视频：如果按帧量分位数，方块走过
    亮区时统计量变、整帧亮度就会跳。这个测试把"不许跳"钉死。
    """
    import numpy as np

    from vision_service import lowlight as ll

    n = 12
    seq = []
    for i in range(n):
        f = np.full((60, 120, 3), 18, np.uint8)      # 暗背景
        f[:, 60:] = 34                                # 右半边亮一点
        f[20:40, 5 * i:5 * i + 20] = 10               # 一个更暗的方块，从左走到右
        seq.append(f)
    monkeypatch.setattr(ll.seek, "iter_frames",
                        lambda *a, **k: [(i / 10, f) for i, f in enumerate(seq)])

    r = ll.enhance_seq("x.mp4", 0.0, 1.2, fps=10, smooth=3, width=0)
    assert r["info"]["n"] == n

    import cv2
    亮度 = []
    for b in r["frames"]:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        亮度.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()))
    # 方块只占 2.8% 面积，全帧平均亮度本就该几乎不变；真跳了说明映射是按帧算的
    assert max(亮度) - min(亮度) < 8, f"逐帧亮度在跳：{[round(x) for x in 亮度]}"
    # 拉伸确实起作用了：原片只用到 10~34，增强后该铺开
    assert max(亮度) > 100
