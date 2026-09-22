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
