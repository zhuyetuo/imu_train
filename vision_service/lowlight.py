"""夜里那几秒黑得看不见狗在干嘛——把那一小段捞出来看清楚。

## 为什么只做「一小段」

人要的不是把 224 路视频都变亮，而是：**IMU/候选说这几秒是抓挠，可画面是黑的，
到底是不是**。所以这里按时间段处理，几秒钟的量，贵一点的做法也用得起。

## 两条路，先后顺序是有理由的

1. **多帧堆栈**（stack）：固定机位、狗动得慢，噪声每帧随机而画面基本不动。
   对齐后平均 N 帧，信噪比按 √N 涨——**一个像素都不编造**，出来的东西就是
   传感器真收到的。这条永远先跑：它同时回答了"原始信号里到底有没有东西"。

2. **模型增强**（Retinexformer 这类）：能补出看起来合理的细节。对**标注**来说
   这是有风险的——人会照着编出来的像素去确认「它在舔后腿」，那条标注就进了
   训练集。所以模型这条永远跟堆栈的结果并排给，让人自己看两边一不一致。

## 怎么判断模型有没有在编

狗场夜里公共区那一路（cam7）是亮的。同一时刻、同一只狗，模型把黑屋子"增强"
出来的样子跟 cam7 看到的对不上，那就是编的。这个对照不需要任何模型，是
这套东西里唯一的硬参照。
"""

from __future__ import annotations

import io
import logging

from . import config, seek

_logger = logging.getLogger("vision_service.lowlight")


def _to_jpeg(bgr, quality: int = 90) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return bytes(buf.tobytes())


def stretch(bgr, lo_q: float = 0.01, hi_q: float = 0.995, fill: float = 1.0, max_gain: float = 12.0):
    """把这一帧真正用到的那一小段亮度铺满 0~255。

    跟前端那个是同一套：量分位数、线性拉伸、放大倍数封顶。封顶是必须的——
    全黑画面的高低分位几乎重合，不封顶就是把纯噪声放成雪花，看着像有东西。

    返回 (拉伸后的图, 说明这一帧原本用到哪一段的字典)。**那个字典要一路带到
    界面上**：放大 10 倍还是一片噪点，说明这一路夜间根本没拍到东西，该去补
    红外补光，不是接着调算法。
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lo = float(np.quantile(gray, lo_q))
    hi = float(max(lo + 1.0, np.quantile(gray, hi_q)))
    gain = min(max_gain, 255.0 * fill / (hi - lo))
    out = np.clip((bgr.astype("float32") - lo) * gain, 0, 255).astype("uint8")
    return out, {"lo": round(lo, 1), "hi": round(hi, 1), "gain": round(gain, 2)}


def stack(path: str, t_s: float, window_s: float = 2.0, max_frames: int = 30,
          align: bool = True) -> tuple:
    """把 t_s 前后 window_s 秒的帧对齐后平均。返回 (平均后的图, 统计)。

    为什么要对齐：摄像头固定，但压缩和轻微晃动会让画面错开一两个像素，直接平均
    会糊。用相位相关估平移（整幅一个位移就够，狗只占一小块，不影响全局估计）。

    为什么不用中值：中值对"狗动了"更稳，但对**噪声**的压制不如均值（同样 N 帧，
    中值的标准差大约是均值的 1.25 倍）。这里要压的就是噪声，所以用均值；狗动
    造成的拖影反而是有用的信息——它说明这几秒里有动作。
    """
    import cv2
    import numpy as np

    start = max(0.0, t_s - window_s / 2)
    end = start + window_s
    frames = [f for _t, f in seek.iter_frames(path, every_sec=1.0 / 30, start_s=start, end_s=end)]
    if not frames:
        raise RuntimeError(f"这一段没解出帧：{path} {start:.1f}~{end:.1f}s")
    frames = frames[:max_frames]
    ref = frames[len(frames) // 2]
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype("float32")
    acc = np.zeros(ref.shape, dtype="float64")
    n = 0
    shifts = []
    for f in frames:
        if align and f.shape == ref.shape:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype("float32")
            (dx, dy), _resp = cv2.phaseCorrelate(ref_g, g)
            if abs(dx) < 8 and abs(dy) < 8:     # 位移太大多半是狗动了，不是抖动
                shifts.append((dx, dy))
                M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        acc += f
        n += 1
    avg = (acc / n).astype("uint8")
    return avg, {
        "frames": n,
        # 信噪比大约按 √N 涨。这是理论上限（噪声完全独立时），实际会低一些
        "snr_gain": round(n ** 0.5, 1),
        "aligned": len(shifts),
    }


def enhance_clip(path: str, t_s: float, window_s: float = 2.0, fill: float = 1.0,
                 model: str | None = None) -> dict:
    """一小段的夜视增强。返回 {原图, 只拉伸, 堆栈后再拉伸, 各自的统计}。

    **三张一起给**，不是只给最好看的那张：
      raw    原样。用来说明"原片就是这样"，不然人会以为是平台把画面弄黑了
      stretch  只拉伸。不编造，但噪声照样放大
      stacked  先堆栈再拉伸。这是软件能做到的上限

    model 给了就再加一张模型增强的，但**永远排在这三张后面**，而且界面上要
    写明它是模型生成的——它好看，却不能当证据。
    """
    import cv2

    frames = [f for _t, f in seek.iter_frames(path, every_sec=1.0, start_s=t_s, end_s=t_s + 0.5)]
    if not frames:
        raise RuntimeError(f"这一刻没解出帧：{path} @{t_s:.1f}s")
    raw = frames[0]
    st, st_info = stretch(raw, fill=fill)
    avg, stack_info = stack(path, t_s, window_s=window_s)
    stacked, stacked_info = stretch(avg, fill=fill)
    out = {
        "raw": _to_jpeg(raw),
        "stretch": _to_jpeg(st),
        "stacked": _to_jpeg(stacked),
        "stretch_info": st_info,
        "stack_info": stack_info | stacked_info,
    }
    if model:
        try:
            out["model"] = _to_jpeg(run_model(avg, model))
            out["model_name"] = model
        except Exception as e:  # noqa: BLE001 模型这条是加分项，挂了不该把前三张一起拖没
            out["model_error"] = f"{type(e).__name__}: {e}"
    return out


def run_model(bgr, name: str):
    """模型增强。权重要自己放到 LOWLIGHT_WEIGHTS 指的地方。

    为什么不自动下载：Retinexformer 这类的官方权重放在网盘上，脚本拉不下来；
    而"自动下载失败"的报错最难查——人只看到一句超时，不知道该去哪儿放文件。
    所以这里直接说清楚缺什么、放哪儿。
    """
    path = config.LOWLIGHT_WEIGHTS
    if not path:
        raise RuntimeError(
            "没配模型权重。先把 Retinexformer 的 .pth 放到算法机上，再在 "
            "vision_service/.env 里写 LOWLIGHT_WEIGHTS=/绝对路径/xxx.pth。"
            "在那之前，「只拉伸」和「堆栈」两张照常出——堆栈那张是不编造的上限"
        )
    raise RuntimeError(
        f"还没接 {name} 的推理代码（权重在 {path}）。"
        "接之前请先看「堆栈」那张：它要是也一片噪点，说明原始信号里就没有东西，"
        "模型只会把噪声画成看起来合理的画面——那种画面不能拿来确认标注"
    )
