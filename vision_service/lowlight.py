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


def kill_chroma_noise(bgr, blur: int = 9):
    """把色度通道上的噪声和整体偏色去掉，只留亮度。

    为什么必须做：实测这一路夜间只用到 14~40 这 26 级，拉伸要放大 9.8 倍。
    在这个亮度上，**色度通道里已经没有任何真实信息了**——传感器收到的红绿蓝
    差异全是读出噪声。放大之后就是满屏紫色麻点，把本来能看出来的轮廓盖住。

    做两件事，都只减不加：
      1. 色度通道重度模糊——真实的颜色在这种光线下不可能有像素级的细节，
         所以糊掉的一定是噪声，不是信息。
      2. 把色度中位数拉回中性——整块的偏色（这里是洋红）是直流偏置，模糊
         去不掉，得平移。

    结果近似灰度图。**这不是在编造像素**：亮度通道一个字节没动，只是把已知
    为噪声的那两个通道压平了。
    """
    import cv2
    import numpy as np

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype("float32")
    for i in (1, 2):
        c = cv2.GaussianBlur(ycc[:, :, i], (blur, blur), 0)
        ycc[:, :, i] = np.clip(c + (128.0 - float(np.median(c))), 0, 255)
    return cv2.cvtColor(np.clip(ycc, 0, 255).astype("uint8"), cv2.COLOR_YCrCb2BGR)


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
    # 拉伸把色噪一起放大了 9 倍多，满屏紫麻点会把轮廓盖住。色度通道在这个
    # 亮度上没有信息，压平它只会让人看得更清楚，不会让人看见不存在的东西
    out = {
        "raw": _to_jpeg(raw),
        "stretch": _to_jpeg(kill_chroma_noise(st)),
        "stacked": _to_jpeg(kill_chroma_noise(stacked)),
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


def enhance_seq(path: str, start_s: float, end_s: float, fps: float = 10.0,
                max_frames: int = 150, smooth: int = 3, width: int = 640,
                fill: float = 1.0) -> dict:
    """把一整段增强成一串帧，让人循环着看。

    ## 为什么非得是动的

    抓挠是动作。单帧最多能回答"狗在不在、什么姿势"，回答不了"它在不在抓"——
    而人要判断的恰恰是后者。所以看清了单帧还不够，得让这一段动起来。

    ## 两件必须做对的事

    1. **拉伸映射全段只算一次**，用整段的时间平均图去量。每帧各算各的话，
       狗一动画面统计量就变，亮度会一帧一跳，闪得没法看——这个坑跟当初
       前端那版夜视是同一个。
    2. **滑动平均，不是整段平均。** 整段平均会把动作糊成拖影（那正是 stack
       做的事，它要的就是静态画面）。这里取前后各一两帧的均值：噪声压下去
       一半多，动作基本还在。

    一个像素都不编造：只有平均、线性拉伸、去色噪三步。
    """
    import cv2
    import numpy as np

    if end_s <= start_s:
        raise RuntimeError(f"这一段时间不对：{start_s:.1f}~{end_s:.1f}s")
    frames = [f for _t, f in seek.iter_frames(path, every_sec=1.0 / fps,
                                              start_s=start_s, end_s=end_s)]
    if not frames:
        raise RuntimeError(f"这一段没解出帧：{path} {start_s:.1f}~{end_s:.1f}s")
    frames = frames[:max_frames]
    if width and frames[0].shape[1] > width:
        h = int(round(frames[0].shape[0] * width / frames[0].shape[1]))
        frames = [cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA) for f in frames]

    # 映射只量一次：拿整段的时间平均图（它最干净）定 lo/hi/gain
    mean_img = np.mean([f.astype("float32") for f in frames], axis=0).astype("uint8")
    _ref, info = stretch(mean_img, fill=fill)

    half = max(0, int(smooth) // 2)
    out = []
    for i in range(len(frames)):
        lo_i, hi_i = max(0, i - half), min(len(frames), i + half + 1)
        f = np.mean([frames[j].astype("float32") for j in range(lo_i, hi_i)], axis=0)
        f = np.clip((f - info["lo"]) * info["gain"], 0, 255).astype("uint8")
        out.append(_to_jpeg(kill_chroma_noise(f), quality=80))
    return {
        "frames": out,
        "fps": fps,
        # 这些要摆到界面上：放大顶到上限还是一片噪点 = 这一路夜间没拍到东西
        "info": info | {"n": len(out), "smooth": max(1, 2 * half + 1), "width": frames[0].shape[1]},
    }


_model_cache: dict = {}


def _import_arch(repo: str):
    """把 RetinexFormer 那一个网络结构文件单独加载进来。

    **不走 `import basicsr.models.archs...`**：那条路会先执行 basicsr 的
    __init__，把整个训练框架的依赖一起拉起来（lmdb、tb_logger、数据加载器……）。
    实测 2026-09-22：仓库和权重都齐了，卡在 `No module named 'lmdb'` ——
    而 lmdb 是训练读数据用的，推理一个字节都用不上。

    按文件路径直接加载，只需要 torch + einops。装不上的训练依赖不该挡住推理。
    """
    import importlib.util
    import os

    if not repo:
        raise RuntimeError(
            "没配 LOWLIGHT_REPO。先 git clone https://github.com/caiyuanhao1998/Retinexformer，"
            "再跑 ./vision_service/get_lowlight_weights.sh 把路径写进 .env"
        )
    f = os.path.join(repo, "basicsr", "models", "archs", "RetinexFormer_arch.py")
    if not os.path.isfile(f):
        raise RuntimeError(
            f"{repo} 里没找到 basicsr/models/archs/RetinexFormer_arch.py。"
            "LOWLIGHT_REPO 指的应该是 git clone 下来的 Retinexformer 目录本身"
        )
    spec = importlib.util.spec_from_file_location("_retinexformer_arch", f)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"加载不了 {f}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as e:
        # 到这一步还缺的，才是这个文件自己真正要的（多半是 einops）
        raise RuntimeError(
            f"{f} 还缺一个依赖：{e.name}。装上就行：pip install {e.name}。"
            "（basicsr 那一堆训练依赖不用装，这里是按文件单独加载的）"
        ) from e
    if not hasattr(mod, "RetinexFormer"):
        raise RuntimeError(f"{f} 里没有 RetinexFormer 这个类——仓库版本对不上？")
    return mod.RetinexFormer


def _load_model(weights: str, repo: str):
    """从官方仓库里拿网络结构，把权重灌进去。

    为什么不自己照着论文重写一份：跟 checkpoint 差一层、差个通道数，推理不会
    报错，只会输出一堆**看着像画面的垃圾**——而这正是最查不出来的一类错。
    仓库 clone 下来就有现成的 arch，对得上。
    """
    import sys

    import torch

    key = (weights, repo)
    if key in _model_cache:
        return _model_cache[key]
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    RetinexFormer = _import_arch(repo)

    ck = torch.load(weights, map_location="cpu")
    sd = ck.get("params", ck) if isinstance(ck, dict) else ck
    # 权重里能读出通道数和每个 stage 的层数，不用人去 yml 里对——对错了就是
    # 上面说的那种"不报错但输出垃圾"
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(
            f"权重跟网络结构对不上（缺 {len(missing)} 个参数，多 {len(unexpected)} 个）。"
            "多半是这份 checkpoint 用的是另一组 n_feat/num_blocks——"
            "去仓库 Options/ 下找对应那个 yml 看参数"
        )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev).eval()
    _model_cache[key] = (net, dev)
    return net, dev


def run_model(bgr, name: str = "retinexformer"):
    """模型增强。权重和仓库都要自己放到算法机上，不自动下载。

    为什么不自动下载：这类权重放在 Google Drive / 百度网盘上，脚本拉不下来；
    而"自动下载失败"的报错最难查——人只看到一句超时，不知道该去哪儿放文件。

    **输出不能当证据。** 它是模型根据训练数据补出来的"合理画面"，在纯噪声上
    照样能画出毛发和轮廓。拿它确认「这是不是抓挠」之前，先看堆栈那张，
    再拿同一时刻公共区那一路对一眼。
    """
    # 先查配置再 import torch：没配权重时该给的是"去哪儿放文件"，
    # 不是一句 ModuleNotFoundError
    if not config.LOWLIGHT_WEIGHTS:
        raise RuntimeError(
            "没配模型权重。把 Retinexformer 的 .pth（这批监控用 SMID.pth 或 "
            "SDSD_indoor.pth，它们是拿真实低光视频训的；LOL_v1/v2 是照片、"
            "几乎没噪声，不对口）放到算法机上，再在 vision_service/.env 里写 "
            "LOWLIGHT_WEIGHTS=/绝对路径/xxx.pth"
        )
    import numpy as np
    import torch

    net, dev = _load_model(config.LOWLIGHT_WEIGHTS, config.LOWLIGHT_REPO)
    # BGR uint8 → RGB float [0,1]
    x = np.ascontiguousarray(bgr[:, :, ::-1]).astype("float32") / 255.0
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(dev)
    # 边长补到 4 的倍数：网络内部要下采样，尺寸不整除会对不齐
    _b, _c, h, w = t.shape
    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        y = net(t)
    y = y[:, :, :h, :w].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (y[:, :, ::-1] * 255).astype("uint8")
