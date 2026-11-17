"""原始 IMU 压缩上传：拿数据池里自己的项圈原始 CSV（随机 60 个），按设备的原始量化（加速度 ±8 g 16 位 = 4096 LSB/g，
陀螺仪 ±2000 °/s 16 位 ≈ 16.4 LSB/(°/s)）在 20 Hz 和 25 Hz 下量各种编码的字节数，换算成每天多少 MB。
  raw16      int16 原样
  delta+vlq  每通道差分 + zigzag + 变长字节（最简单，单片机好写）
  delta+rice 每通道差分 + 分块（32 点）自适应 Golomb-Rice（无损，单片机也好写，跟 FLAC 一个思路）
  zstd/lzma  通用压缩（参考上限；单片机上跑不动 lzma）
  有损       加速度降到 1/512 g、陀螺仪 0.5 °/s 再 delta+rice（误差远小于传感器噪声）
用法：python compress_eval.py"""
import glob
import json
import lzma
import os
import random
import sys

import numpy as np

sys.path[:0] = ["/home/toky/imu_train/src/ml", "/home/toky/imu_train/src/data"]
import train_v2 as T  # noqa: E402


def zigzag(d):
    return np.where(d >= 0, 2 * d, -2 * d - 1).astype(np.uint64)


def vlq_bytes(u):
    n = np.ones(len(u), np.int64)
    v = u >> 7
    while v.any():
        n += (v > 0)
        v >>= 7
    return int(n.sum())


def rice_bits(u, block=32):
    bits = 0
    for i in range(0, len(u), block):
        b = u[i:i + block]
        best = None
        for k in range(0, 16):
            c = int(((b >> k) + 1 + k).sum())
            if best is None or c < best:
                best = c
        bits += best + 4      # 每块记一个 k
    return bits


def encode_sizes(q):
    """q: [n, C] int 采样。返回各方案字节数"""
    out = {"raw16": q.size * 2}
    d = np.diff(q, axis=0, prepend=q[:1] * 0)
    zz = [zigzag(d[:, c].astype(np.int64)) for c in range(q.shape[1])]
    out["delta+vlq"] = sum(vlq_bytes(z) for z in zz)
    out["delta+rice"] = sum(rice_bits(z) for z in zz) // 8
    raw = q.astype("<i2").tobytes()
    out["lzma(raw)"] = len(lzma.compress(raw, preset=6))
    out["lzma(delta)"] = len(lzma.compress(d.astype("<i2").tobytes(), preset=6))
    try:
        import zstandard as zs
        out["zstd(delta)"] = len(zs.ZstdCompressor(level=19).compress(d.astype("<i2").tobytes()))
    except ImportError:
        pass
    return out


def main():
    exp = json.load(open("/home/toky/ai_data/data_train/pool_20261116_all/merged_tmp.json", encoding="utf-8"))
    paths = sorted({os.path.join("/home/toky/ai_data", (t.get("data") or {}).get("csv", "")) for t in exp if (t.get("data") or {}).get("csv")})
    paths = [p for p in paths if "2026_8_2" in p or "2026_9" in p or "2026_10" in p or "2026_11" in p]
    random.seed(0)
    paths = random.sample(paths, min(60, len(paths)))
    res = {}
    for hz in (20.0, 25.0):
        tot = {}
        secs = 0.0
        for p in paths:
            tl = T.load_timeline(p, hz, None, 6)
            if tl is None or len(tl["x"]) < 200:
                continue
            x = tl["x"].astype(np.float64)
            secs += len(x) / hz
            acc, gyr = np.round(x[:, :3] * 4096), np.round(x[:, 3:6] * 16.4)
            for name, q in (("6 轴", np.c_[acc, gyr]), ("3 轴", acc)):
                for k, v in encode_sizes(q.astype(np.int64)).items():
                    tot[(name, k)] = tot.get((name, k), 0) + v
            accL, gyrL = np.round(x[:, :3] * 512), np.round(x[:, 3:6] * 2)
            for name, q in (("6 轴", np.c_[accL, gyrL]), ("3 轴", accL)):
                v = encode_sizes(q.astype(np.int64))["delta+rice"]
                tot[(name, "有损 delta+rice")] = tot.get((name, "有损 delta+rice"), 0) + v
        print(f"\n=== {hz:g} Hz，{len(paths)} 个文件，{secs / 3600:.1f} 小时")
        res[str(hz)] = {}
        for (name, k), v in sorted(tot.items()):
            bps = v / secs
            print(f"  {name} {k:<16} {bps:7.1f} 字节/秒  {bps * 86400 / 1e6:6.2f} MB/天  压缩比 {tot[(name, 'raw16')] / v:4.1f}")
            res[str(hz)][f"{name} {k}"] = {"bytes_per_s": round(bps, 1), "mb_per_day": round(bps * 86400 / 1e6, 2)}
    json.dump(res, open("compress_eval.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
