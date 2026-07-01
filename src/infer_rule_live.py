"""
实时 BLE 行为识别 — 支持 HICC_PetCollar 和 WitMotion WT901SDCL-BT50
支持多种算法并排比较

用法:
  # 规则算法（默认）
  python src/infer_rule_live.py --device hicc
  python src/infer_rule_live.py --device wit

  # ML 模型
  python src/infer_rule_live.py --device hicc --algo ml --model results/processed_a/25hz/ml_rf.pkl

  # 多算法同时对比
  python src/infer_rule_live.py --device hicc --algo rule ml --model results/processed_a/25hz/ml_rf.pkl

  # 扫描设备
  python src/infer_rule_live.py --scan
"""

import argparse
import asyncio
import collections
import sys
import os
import numpy as np
from datetime import datetime

# ── 导入 witmotion_imu 解析模块 ───────────────────────────────────────────────
# witmotion_imu 是本项目的 git submodule（位于项目根目录下）
REPO = os.path.join(os.path.dirname(__file__), "..", "witmotion_imu")
sys.path.insert(0, os.path.abspath(REPO))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))

from gravity_align import gravity_align

try:
    from bleak import BleakClient, BleakScanner
    HAS_BLEAK = True
except ImportError:
    HAS_BLEAK = False

try:
    from hicc_parse import (FrameBuffer as HiccFrameBuffer, parse_frame as hicc_parse_frame,
                             build_timesync_frame, find_tx_uuid, find_rx_uuid)
    HAS_HICC = True
except ImportError:
    HAS_HICC = False

try:
    from wit_parse import StreamingByteBuffer, parse_one_packet
    HAS_WIT = True
except ImportError:
    HAS_WIT = False

G = 9.81

# ── 规则阈值 ──────────────────────────────────────────────────────────────────
CFG = {
    "sleep_std_thresh":   0.08,
    "scratch_freq_low":   2.5,
    "scratch_freq_high":  7.0,
    "scratch_freq_power": 0.25,
    "scratch_std_low":    0.08,
    "scratch_std_high":   3.0,
}

LABEL_COLOR = {
    "睡觉": "\033[94m", "抓挠": "\033[93m", "活动": "\033[92m",
    "Lying chest": "\033[94m", "Walking": "\033[92m", "Trotting": "\033[92m",
}
RESET = "\033[0m"


# ── 算法1：规则 ───────────────────────────────────────────────────────────────
def rule_classify(acc_win: np.ndarray, hz: int) -> str:
    mag = np.linalg.norm(acc_win, axis=1)
    std = float(mag.std())
    fft_vals = np.abs(np.fft.rfft(mag - mag.mean()))
    freqs    = np.fft.rfftfreq(len(mag), d=1.0 / hz)
    fft_vals[0] = 0
    sp  = float(fft_vals[(freqs >= CFG["scratch_freq_low"]) & (freqs <= CFG["scratch_freq_high"])].sum()
                / (fft_vals.sum() + 1e-9))
    dom = float(freqs[fft_vals.argmax()]) if len(freqs) > 1 else 0.0

    if std < CFG["sleep_std_thresh"]:
        return "睡觉"
    if (CFG["scratch_freq_low"] <= dom <= CFG["scratch_freq_high"]
            and sp >= CFG["scratch_freq_power"]
            and CFG["scratch_std_low"] <= std <= CFG["scratch_std_high"]):
        return "抓挠"
    return "活动"


# 中文标签映射（英文原始类别 → 中文）
BEHAVIOR_ZH = {
    "Lying chest": "趴卧", "Sitting": "坐", "Sniffing": "嗅闻",
    "Standing": "站立", "Trotting": "小跑", "Walking": "行走",
    "活动": "活动", "睡觉": "睡觉", "抓挠": "抓挠",
}


# ── 算法2：ML 模型 ────────────────────────────────────────────────────────────
class MLClassifier:
    def __init__(self, model_path: str, use_gravity_align: bool = True,
                 infer_hz: int = 0, infer_window_s: float = 0, infer_stride_s: float = 0):
        import joblib, json
        from features import extract_features as _extract
        self.model    = joblib.load(model_path)
        self._extract = _extract
        self.trained_gravity_aligned = None

        json_path = os.path.splitext(model_path)[0] + ".json"
        meta = {}
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            self.class_names = meta.get("classes", [])
            ga = meta.get("gravity_aligned")
            if ga is not None:
                self.trained_gravity_aligned = bool(ga)
        else:
            self.class_names = []

        print(f"[ml] 加载模型: {model_path}")
        print(f"[ml] 类别: {self.class_names}")

        # 打印训练参数，并与当前推理参数对比
        t_hz       = meta.get("hz")
        t_window_s = meta.get("window_s")
        t_stride_s = meta.get("stride_s")

        if t_hz is not None:
            hz_ok  = (infer_hz == 0 or infer_hz == t_hz)
            win_ok = (infer_window_s == 0 or abs(infer_window_s - t_window_s) < 0.01)
            str_ok = (infer_stride_s == 0 or abs(infer_stride_s - t_stride_s) < 0.01)
            print(f"[ml] 训练参数: 采样率={t_hz}Hz  窗口={t_window_s}s  步长={t_stride_s}s")
            if infer_hz:
                warns = []
                if not hz_ok:  warns.append(f"采样率 训练={t_hz}Hz 推理={infer_hz}Hz")
                if not win_ok: warns.append(f"窗口 训练={t_window_s}s 推理={infer_window_s}s")
                if not str_ok: warns.append(f"步长 训练={t_stride_s}s 推理={infer_stride_s}s")
                if warns:
                    print(f"[ml] ⚠️  参数不一致: {' | '.join(warns)}")
                    if not hz_ok:
                        print(f"[ml]    建议改为: --hz {t_hz} --window_s {t_window_s} --stride_s {t_stride_s}")

        if self.trained_gravity_aligned is not None:
            trained_str = "开" if self.trained_gravity_aligned else "关"
            current_str = "开" if use_gravity_align else "关"
            print(f"[ml] 重力对齐: 训练={trained_str}  推理={current_str}", end="")
            if self.trained_gravity_aligned != use_gravity_align:
                hint = "--no_gravity_align" if not self.trained_gravity_aligned else "去掉 --no_gravity_align"
                print(f"  ⚠️  不一致！建议: {hint}")
            else:
                print()

    def predict(self, acc_win: np.ndarray, gyro_win: np.ndarray, hz: int) -> tuple[str, float]:
        win6 = np.concatenate([acc_win, gyro_win], axis=1)[np.newaxis]  # (1, window_n, 6)
        feat = self._extract(win6, hz)
        pred_id = int(self.model.predict(feat)[0])
        # 置信度：取最高类别概率
        if hasattr(self.model, "predict_proba"):
            conf = float(self.model.predict_proba(feat)[0].max())
        else:
            conf = 1.0
        if pred_id < len(self.class_names):
            en = self.class_names[pred_id]
            label = BEHAVIOR_ZH.get(en, en)
        else:
            label = str(pred_id)
        return label, conf


# ── 输出 ──────────────────────────────────────────────────────────────────────
def fmt_label(label: str) -> str:
    color = LABEL_COLOR.get(label, "")
    return f"{color}{label}{RESET}"


def print_row(chip_ts: str, results: dict):
    """results: {algo_name: (label, conf_or_None)}"""
    pc_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if chip_ts:
        ts_str = f"PC {pc_ts} | 片上 {chip_ts}"
    else:
        ts_str = f"PC {pc_ts}"
    parts = [f"[{ts_str}]"]
    for name, (label, conf) in results.items():
        if conf is not None:
            parts.append(f"{name}={fmt_label(label)}({conf:.0%})")
        else:
            parts.append(f"{name}={fmt_label(label)}")
    print("  ".join(parts))


# ── 滑动窗口推理器 ────────────────────────────────────────────────────────────
class LiveInfer:
    def __init__(self, hz: int, window_s: float, stride_s: float, algos: list,
                 conf_threshold: float = 0.0, use_gravity_align: bool = True):
        self.hz                 = hz
        self.window_n           = int(window_s * hz)
        self.stride_n           = int(stride_s * hz)
        self.algos              = algos   # list of (name, callable, has_conf)
        self.conf_threshold     = conf_threshold
        self.use_gravity_align  = use_gravity_align
        self.acc_ring  = collections.deque(maxlen=self.window_n)
        self.gyro_ring = collections.deque(maxlen=self.window_n)
        self.counter   = 0

    def push(self, acc3: list, gyro3: list, ts: str):
        self.acc_ring.append(acc3)
        self.gyro_ring.append(gyro3)
        self.counter += 1
        if self.counter >= self.stride_n and len(self.acc_ring) == self.window_n:
            self.counter = 0
            acc_win  = np.array(self.acc_ring,  dtype=np.float32)
            gyro_win = np.array(self.gyro_ring, dtype=np.float32)
            if self.use_gravity_align:
                win6 = np.concatenate([acc_win, gyro_win], axis=1)
                win6 = gravity_align(win6)
                acc_win, gyro_win = win6[:, :3], win6[:, 3:]
            results  = {}
            for name, fn, has_conf in self.algos:
                try:
                    if has_conf:
                        label, conf = fn(acc_win, gyro_win)
                        if self.conf_threshold > 0 and conf < self.conf_threshold:
                            label = "Unknown"
                        results[name] = (label, conf)
                    else:
                        label = fn(acc_win, gyro_win)
                        results[name] = (label, None)
                except Exception as e:
                    results[name] = (f"ERR({e})", None)
            print_row(ts, results)


# ── HICC_PetCollar ────────────────────────────────────────────────────────────
def run_hicc(args, infer: LiveInfer):
    if not HAS_HICC:
        print("[错误] 找不到 hicc_parse，请确认 ~/witmotion_imu/ 路径正确")
        sys.exit(1)

    buf     = HiccFrameBuffer()
    ts_sent = [False]
    cli_ref = [None]
    rx_ref  = [None]

    def handler(sender, data: bytearray):
        for frame in buf.feed(bytes(data)):
            cmd = frame[3]
            if cmd == 0x01 and not ts_sent[0] and cli_ref[0] and rx_ref[0]:
                asyncio.get_event_loop().create_task(
                    cli_ref[0].write_gatt_char(rx_ref[0], build_timesync_frame(), response=False)
                )
                ts_sent[0] = True
                continue
            d = hicc_parse_frame(frame)
            if d is None or d.get("frame_type") != "6axis":
                continue
            acc  = [d["acc_x"],  d["acc_y"],  d["acc_z"]]
            gyro = [d.get("gyro_x", 0), d.get("gyro_y", 0), d.get("gyro_z", 0)]
            infer.push(acc, gyro, d.get("timestamp_str", ""))

    async def main():
        addr = args.address or "EA:CB:3E:CF:00:1B"
        print(f"[hicc] 连接 {addr} ...")
        async with BleakClient(addr) as client:
            cli_ref[0] = client
            tx_uuid = await find_tx_uuid(client)
            rx_ref[0] = await find_rx_uuid(client)
            print(f"[hicc] 已连接  Ctrl+C 停止")
            await client.start_notify(tx_uuid, handler)
            try:
                while True:
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await client.stop_notify(tx_uuid)
                print("\n[hicc] 已断开")

    asyncio.run(main())


# ── WitMotion WT901SDCL-BT50 ─────────────────────────────────────────────────
def run_wit(args, infer: LiveInfer):
    if not HAS_WIT:
        print("[错误] 找不到 wit_parse，请确认 ~/witmotion_imu/ 路径正确")
        sys.exit(1)

    buf = StreamingByteBuffer()
    NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

    def handler(sender, data: bytearray):
        for pkt in buf.feed(bytes(data)):
            p = parse_one_packet(pkt)
            if p is None or "acc" not in p:
                continue
            acc  = [v * G for v in p["acc"]]
            gyro = list(p.get("gyro", [0, 0, 0]))
            ct   = p.get("chip_time")
            ts   = ct.strftime("%Y-%m-%d %H:%M:%S") if ct else ""
            infer.push(acc, gyro, ts)

    async def find_device(name_kw: str):
        print(f"[wit] 扫描设备（关键字: {name_kw}）...")
        for d in await BleakScanner.discover(timeout=5.0):
            if d.name and name_kw.lower() in d.name.lower():
                print(f"[wit] 找到: {d.name}  {d.address}")
                return d.address
        return None

    async def main():
        addr = args.address or await find_device(args.name or "WTSDCL")
        if not addr:
            print("[wit] 未找到设备")
            return
        print(f"[wit] 连接 {addr} ...")
        async with BleakClient(addr) as client:
            print(f"[wit] 已连接  Ctrl+C 停止")
            await client.start_notify(NOTIFY_UUID, handler)
            try:
                while True:
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await client.stop_notify(NOTIFY_UUID)
                print("\n[wit] 已断开")

    asyncio.run(main())


# ── 扫描 ──────────────────────────────────────────────────────────────────────
async def scan():
    print("扫描附近 BLE 设备（5秒）...")
    for d in sorted(await BleakScanner.discover(timeout=5.0), key=lambda x: x.name or ""):
        print(f"  {d.address}  {d.name or '(无名称)'}")


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    if not HAS_BLEAK:
        print("[错误] 请先安装: pip install bleak")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="实时 IMU 行为识别")
    parser.add_argument("--device",   choices=["hicc", "wit"], default="hicc")
    parser.add_argument("--address",  default="", help="BLE MAC 地址")
    parser.add_argument("--name",     default="", help="WitMotion 设备名关键字（默认 WTSDCL）")
    parser.add_argument("--hz",       type=int, default=0,
                        help="采样率（HICC默认25，WitMotion默认50，可按设备实际调整）")
    parser.add_argument("--window_s", type=float, default=2.0, help="判断窗口长度（秒）")
    parser.add_argument("--stride_s", type=float, default=1.0, help="判断间隔（秒）")
    parser.add_argument("--algo",     nargs="+", default=["rule"],
                        choices=["rule", "ml"],
                        help="推理算法，可同时选多个: --algo rule ml")
    parser.add_argument("--model",    default="", help="ML 模型路径（.pkl），--algo ml 时必填")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="ML 置信度阈值（0-1），低于此值显示 Unknown，0=不过滤（默认）")
    parser.add_argument("--no_gravity_align", action="store_true",
                        help="禁用重力轴对齐（默认启用，与训练保持一致）")
    parser.add_argument("--scan",     action="store_true", help="扫描附近设备后退出")
    args = parser.parse_args()

    if args.scan:
        asyncio.run(scan())
        return

    hz = args.hz or (25 if args.device == "hicc" else 50)

    # 构建算法列表：(name, fn, has_conf)
    algos = []
    for algo in args.algo:
        if algo == "rule":
            fn = lambda acc, gyro, _hz=hz: rule_classify(acc, _hz)
            algos.append(("规则", fn, False))
        elif algo == "ml":
            if not args.model:
                print("[错误] --algo ml 需要指定 --model <路径>")
                sys.exit(1)
            clf = MLClassifier(args.model,
                               use_gravity_align=not args.no_gravity_align,
                               infer_hz=hz,
                               infer_window_s=args.window_s,
                               infer_stride_s=args.stride_s)
            fn  = lambda acc, gyro, _clf=clf, _hz=hz: _clf.predict(acc, gyro, _hz)
            algos.append(("ML", fn, True))

    use_ga = not args.no_gravity_align
    infer = LiveInfer(hz, args.window_s, args.stride_s, algos,
                      conf_threshold=args.confidence_threshold,
                      use_gravity_align=use_ga)

    ga_str = "开" if use_ga else "关"
    print(f"推理参数: 算法={[name for name, _, _ in algos]}  采样率={hz}Hz  窗口={args.window_s}s  步长={args.stride_s}s  重力对齐={ga_str}")

    if args.device == "hicc":
        run_hicc(args, infer)
    else:
        run_wit(args, infer)


if __name__ == "__main__":
    main()
