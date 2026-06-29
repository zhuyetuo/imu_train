"""
实时 BLE 行为识别 — HICC_PetCollar
基于 infer_rule.py 的规则，无需训练模型。

用法:
  python src/infer_rule_live.py
  python src/infer_rule_live.py --address EA:CB:3E:CF:00:1B
  python src/infer_rule_live.py --window_s 2 --stride_s 1
"""

import argparse
import asyncio
import collections
import sys
import os
import numpy as np

# ── 复用 witmotion_imu 仓库的解析模块（需要在 PYTHONPATH 或同目录下）──────────
# 如果没有 hicc_parse，脚本会提示安装路径
try:
    sys.path.insert(0, os.path.expanduser("~/witmotion_imu"))
    from hicc_parse import FrameBuffer, parse_frame, build_timesync_frame, find_tx_uuid, find_rx_uuid
    from bleak import BleakClient, BleakScanner
    HAS_BLE = True
except ImportError as e:
    HAS_BLE = False
    _import_err = str(e)

# ── 默认参数 ──────────────────────────────────────────────────────────────────
DEFAULT_ADDRESS  = "EA:CB:3E:CF:00:1B"
DEFAULT_HZ       = 25      # 设备采样率
DEFAULT_WINDOW_S = 2.0     # 窗口长度（秒）
DEFAULT_STRIDE_S = 1.0     # 判断间隔（秒）

# ── 规则阈值（与 infer_rule.py 保持一致）─────────────────────────────────────
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


# ── 特征提取 + 规则判断（与 infer_rule.py 相同逻辑）─────────────────────────
def extract_and_classify(acc_win: np.ndarray, hz: int) -> tuple[str, dict]:
    mag = np.linalg.norm(acc_win, axis=1)
    std = float(mag.std())
    n = len(mag)
    fft_vals = np.abs(np.fft.rfft(mag - mag.mean()))
    freqs    = np.fft.rfftfreq(n, d=1.0 / hz)
    fft_vals[0] = 0
    total_power = fft_vals.sum() + 1e-9

    scratch_mask  = (freqs >= CFG["scratch_freq_low"]) & (freqs <= CFG["scratch_freq_high"])
    scratch_power = float(fft_vals[scratch_mask].sum() / total_power)
    dominant_freq = float(freqs[fft_vals.argmax()]) if len(freqs) > 1 else 0.0

    feat = {"std": std, "dominant_freq": dominant_freq, "scratch_power": scratch_power}

    if std < CFG["sleep_std_thresh"]:
        return "睡觉", feat
    if (CFG["scratch_freq_low"] <= dominant_freq <= CFG["scratch_freq_high"]
            and scratch_power >= CFG["scratch_freq_power"]
            and CFG["scratch_std_low"] <= std <= CFG["scratch_std_high"]):
        return "抓挠", feat
    return "活动", feat


# ── 主程序 ────────────────────────────────────────────────────────────────────
def run_live(args):
    hz        = args.hz
    window_n  = int(args.window_s * hz)
    stride_n  = int(args.stride_s * hz)
    buf_ble   = FrameBuffer()
    ring      = collections.deque(maxlen=window_n)   # 滑动窗口
    since_last = [0]                                  # 距上次判断累计帧数
    timesync_sent = [False]
    client_ref    = [None]
    rx_uuid_ref   = [None]

    def notification_handler(sender, data: bytearray):
        frames = buf_ble.feed(bytes(data))
        for frame in frames:
            cmd = frame[3]

            # 时间同步
            if cmd == 0x01 and not timesync_sent[0]:
                if client_ref[0] and rx_uuid_ref[0]:
                    asyncio.get_event_loop().create_task(
                        client_ref[0].write_gatt_char(
                            rx_uuid_ref[0], build_timesync_frame(), response=False
                        )
                    )
                timesync_sent[0] = True
                continue

            d = parse_frame(frame)
            if d is None or d.get("frame_type") != "6axis":
                continue

            acc = [d["acc_x"], d["acc_y"], d["acc_z"]]
            ring.append(acc)
            since_last[0] += 1

            # 每 stride_n 帧判断一次，且窗口已满
            if since_last[0] >= stride_n and len(ring) == window_n:
                since_last[0] = 0
                win = np.array(ring, dtype=np.float32)
                label, feat = extract_and_classify(win, hz)
                color = LABEL_COLOR.get(label, "")
                ts = d.get("timestamp_str", "")
                print(f"[{ts}]  {color}{label}{RESET}"
                      f"  std={feat['std']:.3f}"
                      f"  dom={feat['dominant_freq']:.1f}Hz"
                      f"  sp={feat['scratch_power']:.2f}")

    async def main():
        print(f"[live] 连接设备: {args.address} ...")
        async with BleakClient(args.address) as client:
            client_ref[0] = client
            tx_uuid = await find_tx_uuid(client)
            rx_uuid = await find_rx_uuid(client)
            rx_uuid_ref[0] = rx_uuid
            print(f"[live] 已连接，窗口={args.window_s}s  判断间隔={args.stride_s}s  按 Ctrl+C 停止")
            await client.start_notify(tx_uuid, notification_handler)
            try:
                while True:
                    await asyncio.sleep(1)
            except (KeyboardInterrupt, asyncio.CancelledError):
                pass
            finally:
                await client.stop_notify(tx_uuid)
                print("\n[live] 已断开")

    asyncio.run(main())


def main():
    if not HAS_BLE:
        print(f"[错误] 缺少依赖: {_import_err}")
        print("请先安装: pip install bleak")
        print("并确保 witmotion_imu 仓库在 ~/witmotion_imu/")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--address",  default=DEFAULT_ADDRESS, help="设备 BLE MAC 地址")
    parser.add_argument("--hz",       type=int,   default=DEFAULT_HZ)
    parser.add_argument("--window_s", type=float, default=DEFAULT_WINDOW_S, help="判断窗口长度（秒）")
    parser.add_argument("--stride_s", type=float, default=DEFAULT_STRIDE_S, help="判断间隔（秒）")
    args = parser.parse_args()
    run_live(args)


if __name__ == "__main__":
    main()
