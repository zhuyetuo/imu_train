"""
实时 BLE 行为识别 — 支持 HICC_PetCollar 和 WitMotion WT901SDCL-BT50

用法:
  # HICC 设备（默认）
  python src/infer_rule_live.py --device hicc
  python src/infer_rule_live.py --device hicc --address EA:CB:3E:CF:00:1B

  # WitMotion 设备
  python src/infer_rule_live.py --device wit
  python src/infer_rule_live.py --device wit --name WTSDCL

  # 扫描附近设备
  python src/infer_rule_live.py --scan

  # 调整窗口
  python src/infer_rule_live.py --device hicc --window_s 2 --stride_s 1
"""

import argparse
import asyncio
import collections
import sys
import os
import numpy as np

# ── 导入 witmotion_imu 解析模块 ───────────────────────────────────────────────
REPO = os.path.expanduser("~/witmotion_imu")
sys.path.insert(0, REPO)

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

# ── 规则阈值 ──────────────────────────────────────────────────────────────────
CFG = {
    "sleep_std_thresh":   0.08,
    "scratch_freq_low":   2.5,
    "scratch_freq_high":  7.0,
    "scratch_freq_power": 0.25,
    "scratch_std_low":    0.08,
    "scratch_std_high":   3.0,
}

LABEL_COLOR = {"睡觉": "\033[94m", "抓挠": "\033[93m", "活动": "\033[92m"}
RESET = "\033[0m"

# WitMotion acc 单位是 g，需转换为 m/s²
G = 9.81


# ── 特征提取 + 规则判断 ───────────────────────────────────────────────────────
def classify(acc_win: np.ndarray, hz: int) -> tuple[str, dict]:
    mag = np.linalg.norm(acc_win, axis=1)
    std = float(mag.std())
    fft_vals = np.abs(np.fft.rfft(mag - mag.mean()))
    freqs    = np.fft.rfftfreq(len(mag), d=1.0 / hz)
    fft_vals[0] = 0
    total_power  = fft_vals.sum() + 1e-9
    scratch_mask = (freqs >= CFG["scratch_freq_low"]) & (freqs <= CFG["scratch_freq_high"])
    sp  = float(fft_vals[scratch_mask].sum() / total_power)
    dom = float(freqs[fft_vals.argmax()]) if len(freqs) > 1 else 0.0
    feat = {"std": std, "dom": dom, "sp": sp}

    if std < CFG["sleep_std_thresh"]:
        return "睡觉", feat
    if (CFG["scratch_freq_low"] <= dom <= CFG["scratch_freq_high"]
            and sp >= CFG["scratch_freq_power"]
            and CFG["scratch_std_low"] <= std <= CFG["scratch_std_high"]):
        return "抓挠", feat
    return "活动", feat


def print_result(ts: str, label: str, feat: dict):
    color = LABEL_COLOR.get(label, "")
    print(f"[{ts}]  {color}{label}{RESET}"
          f"  std={feat['std']:.3f}  dom={feat['dom']:.1f}Hz  sp={feat['sp']:.2f}")


# ── HICC_PetCollar ────────────────────────────────────────────────────────────
def run_hicc(args):
    if not HAS_HICC:
        print("[错误] 找不到 hicc_parse，请确认 ~/witmotion_imu/ 路径正确")
        sys.exit(1)

    hz       = args.hz or 25
    window_n = int(args.window_s * hz)
    stride_n = int(args.stride_s * hz)
    buf      = HiccFrameBuffer()
    ring     = collections.deque(maxlen=window_n)
    counter  = [0]
    ts_sent  = [False]
    cli_ref  = [None]
    rx_ref   = [None]

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
            ring.append([d["acc_x"], d["acc_y"], d["acc_z"]])  # 单位 m/s²
            counter[0] += 1
            if counter[0] >= stride_n and len(ring) == window_n:
                counter[0] = 0
                label, feat = classify(np.array(ring, dtype=np.float32), hz)
                print_result(d.get("timestamp_str", ""), label, feat)

    async def main():
        addr = args.address or "EA:CB:3E:CF:00:1B"
        print(f"[hicc] 连接 {addr} ...")
        async with BleakClient(addr) as client:
            cli_ref[0] = client
            tx_uuid = await find_tx_uuid(client)
            rx_ref[0] = await find_rx_uuid(client)
            print(f"[hicc] 已连接  窗口={args.window_s}s  间隔={args.stride_s}s  Ctrl+C 停止")
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
def run_wit(args):
    if not HAS_WIT:
        print("[错误] 找不到 wit_parse，请确认 ~/witmotion_imu/ 路径正确")
        sys.exit(1)

    hz       = args.hz or 50
    window_n = int(args.window_s * hz)
    stride_n = int(args.stride_s * hz)
    buf      = StreamingByteBuffer()
    ring     = collections.deque(maxlen=window_n)
    counter  = [0]

    NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"

    def handler(sender, data: bytearray):
        for pkt in buf.feed(bytes(data)):
            p = parse_one_packet(pkt)
            if p is None or "acc" not in p:
                continue
            # WitMotion acc 单位是 g，转 m/s²
            ax, ay, az = [v * G for v in p["acc"]]
            ring.append([ax, ay, az])
            counter[0] += 1
            if counter[0] >= stride_n and len(ring) == window_n:
                counter[0] = 0
                label, feat = classify(np.array(ring, dtype=np.float32), hz)
                ct = p.get("chip_time")
                ts = ct.strftime("%Y-%m-%d %H:%M:%S") if ct else ""
                print_result(ts, label, feat)

    async def find_device(name_kw: str):
        print(f"[wit] 扫描设备（关键字: {name_kw}）...")
        devices = await BleakScanner.discover(timeout=5.0)
        for d in devices:
            if d.name and name_kw.lower() in d.name.lower():
                print(f"[wit] 找到: {d.name}  {d.address}")
                return d.address
        return None

    async def main():
        if args.address:
            addr = args.address
        else:
            name_kw = args.name or "WTSDCL"
            addr = await find_device(name_kw)
            if not addr:
                print(f"[wit] 未找到设备，请用 --address 指定 MAC 或检查蓝牙")
                return

        print(f"[wit] 连接 {addr} ...")
        async with BleakClient(addr) as client:
            print(f"[wit] 已连接  窗口={args.window_s}s  间隔={args.stride_s}s  Ctrl+C 停止")
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
    devices = await BleakScanner.discover(timeout=5.0)
    for d in sorted(devices, key=lambda x: x.name or ""):
        print(f"  {d.address}  {d.name or '(无名称)'}")


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    if not HAS_BLEAK:
        print("[错误] 请先安装: pip install bleak")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="实时 IMU 行为识别")
    parser.add_argument("--device",   choices=["hicc", "wit"], default="hicc",
                        help="设备类型: hicc（HICC_PetCollar）或 wit（WitMotion WT901）")
    parser.add_argument("--address",  default="", help="BLE MAC 地址（可选，不填自动搜索）")
    parser.add_argument("--name",     default="", help="WitMotion 设备名关键字（默认 WTSDCL）")
    parser.add_argument("--hz",       type=int, default=0,
                        help="采样率（HICC默认25，WitMotion默认50）")
    parser.add_argument("--window_s", type=float, default=2.0, help="判断窗口长度（秒）")
    parser.add_argument("--stride_s", type=float, default=1.0, help="判断间隔（秒）")
    parser.add_argument("--scan",     action="store_true", help="扫描附近设备后退出")
    args = parser.parse_args()

    if args.scan:
        asyncio.run(scan())
        return

    if args.device == "hicc":
        run_hicc(args)
    else:
        run_wit(args)


if __name__ == "__main__":
    main()
