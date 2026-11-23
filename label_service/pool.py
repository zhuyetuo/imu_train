"""
推理进程池：跟 run_review_bins_all_days.sh 的 WORKERS=-1 一样按文件并行——
每个 worker 进程启动时各自加载一次模型（RF 模型跨进程 pickle 一次几十 MB，
不能每个请求都传），之后一个请求一个文件丢给空闲进程跑。RF 本身预测很快，
慢的是特征提取/重采样这些纯 CPU 的 Python 循环，单进程吃不满多核，进程池
才能把 14900K 这种 24 核跑起来。

worker 里跑的 _infer_one 跟单进程版一模一样：让 infer_file 在临时目录按它
自己的格式落 *_infer.json 再读回来，返回的就是命令行产出的同一份东西。
"""

import datetime as dt
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from label_service import config

_bundle: dict = {}



log = logging.getLogger(__name__)


def _missing_holes(ts_vals, bad, min_s: float = 0.2, merge_gap_s: float = 0.2) -> list[tuple]:
    """六轴全 0 / 空值（bad）的连续时间段，**按时间轴算，不按行数**。

    为什么不能按行数：掉数据时采集端只按 25 Hz 写占位行，正常是 47~50 Hz。
    拿固定的 window_s×device_hz 行去切，一个"8 秒窗口"实际会覆盖到 11 秒，
    零段被正常数据稀释，比例掉到 0.5 以下——整段掉线就漏判了（线上 333
    项目 704–708s、750–755s 都漏过）。直接从原始样本找连续零段就没有这个问题，
    也不受"掉线必须长于半个窗口"的限制。
    """
    if ts_vals is None or bad is None or len(bad) == 0:
        return []
    idx = np.flatnonzero(bad)
    if len(idx) == 0:
        return []
    n = len(bad)
    splits = np.where(np.diff(idx) > 1)[0] + 1
    diffs = np.diff(ts_vals).astype("timedelta64[us]").astype(np.float64) / 1e6
    step = float(np.median(diffs)) if len(diffs) else 0.02
    step = step if step > 0 else 0.02
    spans: list[list] = []
    for g in np.split(idx, splits):
        a = ts_vals[g[0]]
        last = int(g[-1])
        # 零段结束时间取"最后一个零样本之后的下一个时间戳"，这样这一段覆盖到
        # 恢复前的整个空档；文件末尾就往后延一个采样间隔
        b = ts_vals[last + 1] if last + 1 < n else ts_vals[last] + np.timedelta64(int(step * 1e6), "us")
        spans.append([a, b])
    merged: list[list] = []
    gap = np.timedelta64(int(merge_gap_s * 1e6), "us")
    for a, b in spans:
        if merged and (a - merged[-1][1]) <= gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        sec = float((b - a).astype("timedelta64[us]").astype(np.float64) / 1e6)
        if sec >= min_s:
            out.append((a.astype("datetime64[us]").astype(dt.datetime),
                        b.astype("datetime64[us]").astype(dt.datetime)))
    return out

def _memoize_load_csv() -> None:
    """
    同一个文件在一次推理里要读两遍 CSV：infer_file 自己读一遍算特征，
    _spectral_ratio_per_window 再读一遍算频谱。一小时 50Hz 的数据是 18 万行，
    pandas 解析一遍就大几百毫秒。这里给 load_csv 套一个只存最近一个文件的缓存，
    第二次直接命中；换文件就自动淘汰，不会把内存堆起来。
    """
    import infer_csv_scratch as m

    if getattr(m, "_lru_wrapped", False):
        return
    orig = m.load_csv
    cache: dict = {}

    def cached(path):
        if cache.get("path") != path:
            cache.clear()
            cache["path"], cache["val"] = path, orig(path)
        return cache["val"]

    m.load_csv = cached
    m._lru_wrapped = True


def _limit_threads_in_worker() -> None:
    """
    每个 worker 只跑单线程的 numpy/BLAS/OpenMP。

    并行度已经由进程池提供（默认 CPU 核数 - 2 个进程），如果每个进程里的 numpy /
    scikit-learn 再各自开满线程，30 个进程 × 几十个线程会有上千个线程去抢几十个核，
    绝大部分时间花在上下文切换上，实测比单线程还慢。这是 multiprocessing + numpy
    最常见的一个坑。

    注意只限制推理 worker，不设成全局环境变量——训练（train_custom.sh 里的
    XGBoost/LightGBM）是靠 OpenMP 多线程加速的，那边要用满核心。
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    try:
        # numpy 已经在父进程 import 过了，fork 出来的子进程里改环境变量对已经建好的
        # 线程池不生效；threadpoolctl 是运行时改，才真正管用（sklearn 自带这个依赖）
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 限不了就算了，只是慢点，不该影响推理
        pass


def _init_worker(model_path: str) -> None:
    # 子进程是 fork 出来的，父进程初始化过 CUDA 时子进程不能再初始化（会整池崩），
    # CNN 推理改走 CPU：模型小、延迟低，又不用跟训练抢显存
    os.environ["INFER_FORCE_CPU"] = "1"
    _limit_threads_in_worker()
    sys.path.insert(0, os.path.join(config.REPO_ROOT, "src"))
    from label_service.logging_setup import worker_setup_logging
    from label_service.model_loader import load_model_bundle
    worker_setup_logging()
    _bundle.update(load_model_bundle(model_path))
    _bundle["model_path"] = model_path
    # 模型自己也别再开多线程/多进程：随机森林是用 n_jobs=-1 训出来的，predict 时
    # 会照着这个值再 fork 一堆 joblib worker，跟进程池打架
    model = _bundle.get("model")
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    _memoize_load_csv()


def _spectral_ratio_per_window(full_path: str, windows: list[dict], window_s: float,
                               device_hz: float | None = None) -> tuple[dict | None, list[tuple]]:
    """
    给每个窗口算陀螺仪 4–8 Hz 能量占 1–15 Hz 的比例，写进 window["spec"]。抓挠是
    后腿高频往复，这个频段有明显峰；模型之外的独立证据，稳定版可按它过滤误报
    （config.STABLE_SPECTRAL_MIN）。算不出来（没陀螺仪/没时间戳）就全 None。
    同时返回 10 Hz 的陀螺仪能量包络 {t0, hz, energy}，给边界微调用（postprocess.refine_boundaries）。

    第二个返回值是**掉数据（六轴全 0 / 空值）的连续时间区间**，直接按原始样本
    的时间轴算，见 _missing_holes。app.py 拿它挖掉预测结果里落在断连段的部分。
    """
    envelope = None
    holes: list[tuple] = []
    try:
        from infer_csv_scratch import load_csv
        acc, gyro, ts, valid_mask, _null = load_csv(full_path)
        if gyro is None or ts is None or len(gyro) == 0:
            return None, []
        # 采样率按这个文件实际的来：8-11 之前的数据是采集端就降到 16Hz 存的，
        # 8-11 起才是 50Hz 原始流。用全局默认值去算 16Hz 的文件，频段和窗口长度全错
        hz = float(device_hz or config.DEVICE_HZ)
        n_samp = int(window_s * hz)
        ts_vals = ts.values.astype("datetime64[ns]")
        mag = np.linalg.norm(gyro, axis=1).astype(np.float32)
        # 能量包络：去均值后的幅值平方按 0.1s 分箱求均值
        step = max(1, int(hz) // 10)
        dm = mag - float(np.nanmean(mag))
        sq = np.nan_to_num(dm * dm)
        n_bins = len(sq) // step
        if n_bins > 0:
            env = sq[: n_bins * step].reshape(n_bins, step).mean(axis=1)
            envelope = {"t0": ts.iloc[0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "hz": hz / step,
                        "energy": [round(float(v), 4) for v in env]}
            # 精准版起止用：只留 3–9 Hz（抓挠时后腿往复的节律）再算能量。转身、走动这些低频大动作
            # 在总能量里很大、在这里很小，所以不会把起点带到抓之前的动作上（postprocess.precise_boundaries）
            try:
                from scipy.signal import butter, sosfiltfilt
                if len(mag) > 30 and hz > 12:
                    sos = butter(4, [3.0, min(9.0, hz / 2 * 0.95)], btype="band", fs=hz, output="sos")
                    bm = sosfiltfilt(sos, np.nan_to_num(dm))
                    bsq = bm * bm
                    envelope["band"] = [round(float(v), 5) for v in bsq[: n_bins * step].reshape(n_bins, step).mean(axis=1)]
            except Exception:  # noqa: BLE001 算不出来就只平移，不影响别的版本
                pass
        freqs = np.fft.rfftfreq(n_samp, d=1.0 / hz)
        band = (freqs >= 4) & (freqs <= 8)
        wide = (freqs >= 1) & (freqs <= 15)
        hann = np.hanning(n_samp).astype(np.float32)

        # 所有窗口一次性堆成矩阵做批量 FFT——逐窗口在 Python 里循环几千次
        # rfft，光解释器开销就比 FFT 本身还贵
        # 掉数据的样本点。采集端（witmotion_imu）在那一帧对不上 IMU 时六个轴都写
        # 空串，load_csv 记在 valid_mask 里（False = 这行没数据），并且已经
        # ffill/bfill 过——所以不能拿返回的数值去判断，填完之后看着完全正常，
        # 模型看到的是一段"冻住不动"的值，很容易判成睡觉。必须用 valid_mask。
        # 再兜一层"六轴全 0"：真实佩戴时六轴不可能同时精确为 0（重力总落在某个
        # 轴上），别的产出源真写 0 占位也能认出来。
        try:
            bad = ~np.asarray(valid_mask, dtype=bool)
            both = np.concatenate([np.asarray(acc, dtype=np.float32), np.asarray(gyro, dtype=np.float32)], axis=1)
            bad = bad | (np.abs(both) < 1e-9).all(axis=1)
        except Exception:  # noqa: BLE001 判不出来就不判，别让整次推理失败
            bad = None

        # 掉数据的**真实时间段**：按每行占的时长累加，不按行数。掉数据时采集端
        # 只写 25 Hz 的占位行，按行数算会把零段的比例低估掉一半（见 _missing_holes）
        if bad is not None and len(bad):
            holes = _missing_holes(ts_vals, bad)
        # 每个样本点实际覆盖的时长，用来算窗口的缺数据比例
        cum_bad = None
        if bad is not None:
            dd = np.diff(ts_vals).astype("timedelta64[us]").astype(np.float64) / 1e6
            durs = np.append(dd, dd[-1] if len(dd) else 0.02)
            durs = np.clip(durs, 0.0, 0.5)
            cum_bad = np.concatenate(([0.0], np.cumsum(durs * bad)))

        starts, idx_of = [], []
        for k, w in enumerate(windows):
            w["spec"] = None
            w["missing"] = None
            w["amag"] = None
            w["gstd"] = None
            w["arhythm"] = None
            t = w.get("ts")
            if not t:
                continue
            try:
                t0 = np.datetime64(t.replace(" ", "T"))
            except ValueError:
                continue
            i0 = int(np.searchsorted(ts_vals, t0))
            # 缺数据比例先算——末尾不足半个窗口的会在下面被跳过，但它照样要算。
            # 窗口按**时间**取 [t0, t0+window_s) 而不是固定 n_samp 行：采样率不是
            # 恒定的时候（掉数据段 25 Hz、正常 47~50 Hz）按行数切出来的窗口根本不是
            # window_s 秒，比例就失真了
            if cum_bad is not None and i0 < len(ts_vals):
                t1 = t0 + np.timedelta64(int(round(window_s * 1e6)), "us")
                span_end = min(ts_vals[-1], t1)
                elapsed = float((span_end - t0).astype("timedelta64[us]").astype(np.float64) / 1e6)
                if elapsed > 0.5:
                    i1 = int(np.searchsorted(ts_vals, t1))
                    miss = float(cum_bad[min(i1, len(ts_vals))] - cum_bad[i0])
                    w["missing"] = round(min(1.0, max(0.0, miss / elapsed)), 3)
            if len(mag) - i0 < n_samp // 2:
                continue
            starts.append(i0)
            idx_of.append(k)
        if starts:
            padded = np.zeros(len(mag) + n_samp, dtype=np.float32)
            padded[: len(mag)] = mag
            offsets = np.asarray(starts)[:, None] + np.arange(n_samp)[None, :]
            segs = padded[offsets]
            segs = segs - segs.mean(axis=1, keepdims=True)
            pw = np.abs(np.fft.rfft(segs * hann, axis=1)) ** 2
            tot = pw[:, wide].sum(axis=1)
            ratio = np.divide(pw[:, band].sum(axis=1), tot, out=np.zeros_like(tot), where=tot > 0)
            # 加速度模长 / 陀螺仪幅值 std：给「是不是真在挠」的质量门限用。抓挠是
            # 身体在动（陀螺仪几十到几百 °/s、模长 1.1–2.5g）；设备没戴、躺着被碰
            # 时陀螺仪几乎不动、模长还可能低于 1g。都在同一份窗口矩阵上算，省一次读文件
            am = np.linalg.norm(np.asarray(acc, dtype=np.float32), axis=1)
            gm = np.linalg.norm(np.asarray(gyro, dtype=np.float32), axis=1)
            pa = np.zeros(len(am) + n_samp, dtype=np.float32)
            pa[: len(am)] = am
            pg = np.zeros(len(gm) + n_samp, dtype=np.float32)
            pg[: len(gm)] = gm
            w_amag = pa[offsets].mean(axis=1)
            w_gstd = pg[offsets].std(axis=1)
            # 加速度自身的「节律」：真抓挠是后腿高频往复，模长在 2–10 Hz 有明显能量；
            # 信号不好/设备被碰的波动是低频漂移（实测误报段 84% 能量在 1 Hz 以下）。
            # 这样即使陀螺仪那一路不可用，也能拦住"加速度乱抖"冒充抓挠
            asegs = pa[offsets]
            asegs = asegs - asegs.mean(axis=1, keepdims=True)
            apw = np.abs(np.fft.rfft(asegs * hann, axis=1)) ** 2
            aband = (freqs >= 2) & (freqs <= 10)
            awide = (freqs >= 0.5) & (freqs <= 15)
            atot = apw[:, awide].sum(axis=1)
            aratio = np.divide(apw[:, aband].sum(axis=1), atot, out=np.zeros_like(atot), where=atot > 0)
            for k, r, tt, am_v, gs_v, ar, at in zip(idx_of, ratio, tot, w_amag, w_gstd, aratio, atot):
                windows[k]["spec"] = round(float(r), 3) if tt > 0 else None
                windows[k]["amag"] = round(float(am_v), 3)
                windows[k]["gstd"] = round(float(gs_v), 1)
                windows[k]["arhythm"] = round(float(ar), 3) if at > 0 else None
    except Exception:  # noqa: BLE001 频谱只是附加信息，算不出来不影响推理
        for w in windows:
            w.setdefault("spec", None)
            w.setdefault("missing", None)
            w.setdefault("amag", None)
            w.setdefault("gstd", None)
            w.setdefault("arhythm", None)
    return envelope, holes


def _infer_one(full_path: str, device_hz: float | None = None) -> dict:
    from infer_csv_scratch import infer_file  # worker 进程里 src/ 已在 sys.path
    b = _bundle
    # 要输出哪些类别的片段。**默认跟着模型自己的类别走**——配置里写死一串的话，
    # 模型输出「抓挠-头颈耳」这种拆了二级的新类别时一个片段都不会出来，
    # 而平台那边看到的就是"模型什么都没检出"，完全看不出是配置挡的
    target_labels = config.target_labels_for(b["classes"])
    model_hz = b["hz"]
    # 四舍五入：31.25 Hz 时 1.984 s × 31.25 = 61.999… 用 int() 会截成 61，跟训练的 62 对不上
    window_size = int(round(b["window_s"] * model_hz))
    stride = int(round(b["stride_s"] * model_hz))
    with tempfile.TemporaryDirectory(prefix="label_infer_") as tmp:
        infer_file(
            full_path, b["model"], b["classes"], window_size, stride,
            float(device_hz or config.DEVICE_HZ), model_hz, b["gravity_aligned"],
            quiet=True, scratch_only=True, label_mode=b["label_mode"], output_dir=tmp,
            resample_method=config.RESAMPLE_METHOD,
            target_labels=target_labels, is_dl=b["is_dl"],
            # 3 轴模型只能喂加速度——喂 6 轴进去特征维数对不上
            n_channels=b.get("n_channels", 8),
            channel_mode=b.get("channel_mode", "tilt"),
        )
        stem = os.path.splitext(os.path.basename(full_path))[0]
        segments, windows, n_windows = {}, [], 0
        for label in target_labels:
            p = os.path.join(tmp, label, "_infer", f"{stem}_infer.json")
            if not os.path.exists(p):
                segments[label] = []
                continue
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            segments[label] = data["scratch_segments"]
            windows, n_windows = data["windows"], data["n_windows"]  # 各类别文件里这两项相同
    envelope, missing_holes = (_spectral_ratio_per_window(full_path, windows, b["window_s"], device_hz)
                               if windows else (None, []))
    # 疑似舔/啃候选：只看加速度姿态，跟模型无关，但要读同一份 CSV（load_csv 在 worker
    # 里 memoize 过，不会再读一次盘）。挖不出来就空，绝不能让它把推理整个拖垮
    # 同一次读盘顺带出：姿态段按节律分成 舔 / 进食 / 饮水，再加疑似吠叫（见 behaviors.py）
    grooming: list[dict] = []
    if (config.GROOM_ENABLED or config.BARK_ENABLED or config.SHAKE_ENABLED or config.TOILET_ENABLED) and windows:
        try:
            from label_service import behaviors as _behaviors
            grooming = _behaviors.mine_file(
                full_path, float(device_hz or config.DEVICE_HZ), _groom_params(), _behavior_params(),
                split=config.GROOM_SPLIT_ENABLED, barks=config.BARK_ENABLED,
                shakes=config.SHAKE_ENABLED, toilet=config.TOILET_ENABLED)
            if not config.GROOM_ENABLED:
                grooming = [c for c in grooming if c.get("reason") in ("barking", "shaking", "toileting")]
        except Exception:  # noqa: BLE001 候选是锦上添花，出错只记日志
            log.exception("疑似舔/啃/进食/饮水/吠叫/甩头/如厕候选挖掘失败 %s", full_path)
    return {"segments": segments, "windows": windows, "n_windows": n_windows, "envelope": envelope,
            "missing_holes": missing_holes, "grooming": grooming}


def _groom_params():
    from label_service import grooming as _grooming
    return _grooming.GroomParams(
        tilt_min_deg=config.GROOM_TILT_MIN_DEG, motion_min=config.GROOM_MOTION_MIN,
        motion_max=config.GROOM_MOTION_MAX, min_s=config.GROOM_MIN_S, gap_s=config.GROOM_GAP_S,
        label=config.GROOM_LABEL,
    )


def _behavior_params():
    from label_service import behaviors as _behaviors
    return _behaviors.BehaviorParams(
        label_eat=config.EAT_LABEL, label_drink=config.DRINK_LABEL, label_bark=config.BARK_LABEL,
        bark_peak_min_g=config.BARK_PEAK_MIN_G, bark_peak_k=config.BARK_PEAK_K, bark_min_n=config.BARK_MIN_N,
        label_shake=config.SHAKE_LABEL, shake_rms_min_g=config.SHAKE_RMS_MIN_G,
        label_urinate=config.TOILET_URINE_LABEL, label_defecate=config.TOILET_FECES_LABEL,
    )


def create_pool(model_path: str) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=config.INFER_WORKERS,
        initializer=_init_worker,
        initargs=(model_path,),
    )
