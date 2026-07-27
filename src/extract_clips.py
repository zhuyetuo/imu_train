"""
读取推理 JSON，把每段抓挠裁剪成与 witmotion_imu 完全一致的三件套：
  {stem}_clip{N:02d}_{HHMMSS}-{HHMMSS}.mp4   ← cam1 视频片段
  {cam2_stem}_clip{N:02d}_{HHMMSS}-{HHMMSS}.mp4  ← cam2 视频片段
  {stem}_clip{N:02d}_{HHMMSS}-{HHMMSS}.csv   ← cam1 CSV 裁剪

文件按置信度分桶放入 clips_{bin}/ 子目录，同时写 scratch_log_review_{bin}.txt。

用法:
  python src/extract_clips.py \
    --infer_dir  infer_result/2026_7_17/_infer \
    --video_dir  data/raw_custom/data/2026_7_17 \
    --output_dir infer_result/2026_7_17 \
    --context_s  3
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def get_bin(conf: float) -> str:
    for i in range(len(BINS) - 1):
        if BINS[i] <= conf < BINS[i + 1]:
            lo = BINS[i]
            hi = BINS[i + 1] if BINS[i + 1] <= 1.0 else 1.0
            return f"{lo:.1f}-{hi:.1f}"
    return "0.9-1.0"


def ts_to_sec(ts_str: str) -> float:
    if not ts_str:
        return 0.0
    dt = datetime.strptime(ts_str[:23], "%Y-%m-%d %H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


def ts_to_hhmmss(ts_str: str) -> str:
    """'2026-07-17 18:59:07.681' → '185907'"""
    if not ts_str:
        return "000000"
    return ts_str[11:13] + ts_str[14:16] + ts_str[17:19]


HAS_CUDA = subprocess.run(
    ["ffmpeg", "-hide_banner", "-hwaccels"],
    capture_output=True, text=True
).stdout.find("cuda") >= 0


def find_video(stem: str, video_dir: str) -> str:
    for ext in (".mp4", ".MP4", ".avi", ".mov"):
        p = os.path.join(video_dir, stem + ext)
        if os.path.exists(p):
            return p
    return ""


def csv_start_sec(csv_path: str) -> float:
    """读取 CSV 第一行数据的时间戳，作为视频 t=0 的绝对时间（秒）。"""
    try:
        with open(csv_path, encoding="utf-8") as f:
            header = f.readline()
            first  = f.readline()
        cols = [c.strip().strip('"') for c in header.split(",")]
        try:
            ts_idx = next(i for i, c in enumerate(cols) if "timestamp" in c.lower())
        except StopIteration:
            ts_idx = 0
        ts_str = first.split(",")[ts_idx].strip().strip('"')
        return ts_to_sec(ts_str)
    except Exception:
        return 0.0


def sibling_stem(stem: str, target_cam: int) -> str:
    """cam1_imu1 → cam2_imu2（或反向），target_cam=1 或 2"""
    return re.sub(r"cam\d_imu\d", f"cam{target_cam}_imu{target_cam}", stem)


def cut_clip(video_path: str, start_abs: float, end_abs: float,
             out_path: str, context_s: float, video_t0: float = 0.0) -> bool:
    """
    start_abs / end_abs: 午夜起始的绝对秒数
    video_t0: 视频第一帧对应的绝对秒数（从 CSV 第一行读取）
    """
    rel_start = max(0.0, (start_abs - video_t0) - context_s)
    rel_end   = (end_abs - video_t0) + context_s
    duration  = rel_end - rel_start
    if duration <= 0:
        print(f"    [跳过] 时间偏移为负（video_t0={video_t0:.1f}, start={start_abs:.1f}）")
        return False

    # 浏览器兼容优先：libx264 baseline，faststart，yuv420p，无音频
    # nvenc 输出某些浏览器不支持，统一用软编码保证兼容性
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{rel_start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-crf", "23",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-an",                       # 原始视频无音频，明确丢弃避免 aac 编码报错
        "-movflags", "+faststart",   # moov atom 放文件头，浏览器流式播放必需
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # 确保宽高为偶数
        out_path,
    ]
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0:
        err = ret.stderr.decode("utf-8", errors="replace")[-400:]
        print(f"    [ffmpeg错误] {err}")
    return ret.returncode == 0


def slice_csv(src_csv: str, dst_csv: str,
              pad_start: float, pad_end: float) -> bool:
    """裁剪 CSV 到 [pad_start, pad_end] 秒范围内的行。"""
    try:
        with open(src_csv, encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return False
        header = lines[0]
        cols = [c.strip().strip('"') for c in header.split(",")]
        try:
            ts_idx = next(i for i, c in enumerate(cols) if "timestamp" in c.lower())
        except StopIteration:
            ts_idx = 0
        kept = [header]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= ts_idx:
                continue
            try:
                sec = ts_to_sec(parts[ts_idx].strip().strip('"'))
            except Exception:
                continue
            if pad_start <= sec <= pad_end:
                kept.append(line)
        if len(kept) <= 1:
            return False
        with open(dst_csv, "w", encoding="utf-8") as f:
            f.writelines(kept)
        return True
    except Exception as e:
        print(f"    [CSV裁剪错误] {e}")
        return False


def _process_one(task: dict) -> dict:
    """处理单个 clip 任务（在线程池中运行）。"""
    args       = task["args"]
    has_ffmpeg = task["has_ffmpeg"]
    stem1      = task["stem1"]
    stem2      = task["stem2"]
    src_csv1   = task["src_csv1"]
    cam1_mp4   = task["cam1_mp4"]
    cam2_mp4   = task["cam2_mp4"]
    video_t0   = task["video_t0"]
    clip_dir   = task["clip_dir"]
    suffix     = task["suffix"]
    seg        = task["seg"]

    start_sec = ts_to_sec(seg["start_ts"])
    end_sec   = ts_to_sec(seg["end_ts"]) if seg["end_ts"] else start_sec + 2.0
    pad_start = max(0.0, start_sec - args.context_s)
    pad_end   = end_sec + args.context_s

    detected_stem = task["detected_stem"]
    cam1_clip_mp4 = os.path.join(clip_dir, stem1 + suffix + ".mp4")
    cam2_clip_mp4 = os.path.join(clip_dir, stem2 + suffix + ".mp4")
    cam1_clip_csv = os.path.join(clip_dir, detected_stem + suffix + ".csv")  # 检测狗的 CSV

    ok_cam1 = cut_clip(cam1_mp4, start_sec, end_sec, cam1_clip_mp4, args.context_s, video_t0) if (cam1_mp4 and has_ffmpeg) else False
    ok_cam2 = cut_clip(cam2_mp4, start_sec, end_sec, cam2_clip_mp4, args.context_s, video_t0) if (cam2_mp4 and has_ffmpeg) else False
    ok_csv  = slice_csv(src_csv1, cam1_clip_csv, pad_start, pad_end) if os.path.exists(src_csv1) else False

    return {**task, "ok_cam1": ok_cam1, "ok_cam2": ok_cam2, "ok_csv": ok_csv}


def main():
    parser = argparse.ArgumentParser(description="按置信度区间裁剪抓挠视频片段（witmotion_imu 格式）")
    parser.add_argument("--infer_dir",  required=True)
    parser.add_argument("--video_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_s",  type=float, default=3.0)
    parser.add_argument("--workers",    type=int,   default=4,
                        help="并行裁剪线程数（默认 4，有 CUDA 建议 2-4）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer_jsons = sorted(glob.glob(os.path.join(args.infer_dir, "*_infer.json")))
    if not infer_jsons:
        print(f"[错误] {args.infer_dir} 下没有 *_infer.json")
        sys.exit(1)

    has_ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    if not has_ffmpeg:
        print("[警告] 找不到 ffmpeg，只裁剪 CSV")

    print(f"编码器: {'CUDA (h264_nvenc)' if HAS_CUDA else 'CPU (libx264)'}  并行线程: {args.workers}")

    # ── 收集所有 clip 任务 ────────────────────────────────────────────────────
    bin_dirs  = {}
    lock      = threading.Lock()
    all_tasks = []

    for infer_path in infer_jsons:
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)
        csv_basename = data["csv_basename"]
        segs = data.get("scratch_segments", [])
        if not segs:
            continue

        # 判断是哪只狗的 IMU：cam1_imu1=狗1，cam2_imu2=狗2
        detected_stem = os.path.splitext(csv_basename)[0]
        is_cam2 = bool(re.search(r"cam2_imu2", csv_basename))
        if is_cam2:
            # 狗2检测到抓挠：cam1=视角1(兄弟), cam2=检测狗
            stem2 = detected_stem                  # cam2_imu2（检测狗）
            stem1 = sibling_stem(detected_stem, 1) # cam1_imu1（视角1兄弟）
        else:
            # 狗1检测到抓挠：cam1=检测狗, cam2=视角2(兄弟)
            stem1 = detected_stem                  # cam1_imu1（检测狗）
            stem2 = sibling_stem(detected_stem, 2) # cam2_imu2（视角2兄弟）
        src_csv1 = os.path.join(args.video_dir, csv_basename)
        cam1_mp4 = find_video(stem1, args.video_dir)
        cam2_mp4 = find_video(stem2, args.video_dir)
        video_t0 = csv_start_sec(src_csv1) if os.path.exists(src_csv1) else 0.0

        for idx, seg in enumerate(segs, 1):
            t0_str = seg.get("start_ts", "") or ""
            t1_str = seg.get("end_ts",   "") or ""
            if not t0_str:
                continue

            conf      = seg.get("conf_mean", 0.0)
            conf_max  = seg.get("conf_max",  0.0)
            bin_label = get_bin(conf)
            suffix    = f"_clip{idx:02d}_{ts_to_hhmmss(t0_str)}-{ts_to_hhmmss(t1_str or t0_str)}"

            with lock:
                if bin_label not in bin_dirs:
                    clip_dir = os.path.join(args.output_dir, f"clips_{bin_label}")
                    os.makedirs(clip_dir, exist_ok=True)
                    bin_dirs[bin_label] = clip_dir
                clip_dir = bin_dirs[bin_label]

            all_tasks.append({
                "args": args, "has_ffmpeg": has_ffmpeg,
                "stem1": stem1, "stem2": stem2,
                "detected_stem": detected_stem,  # CSV clip 用检测狗的 stem
                "src_csv1": src_csv1, "cam1_mp4": cam1_mp4, "cam2_mp4": cam2_mp4,
                "video_t0": video_t0, "clip_dir": clip_dir, "suffix": suffix,
                "seg": {"start_ts": t0_str, "end_ts": t1_str,
                        "conf_mean": conf, "conf_max": conf_max},
                "csv_basename": csv_basename, "bin_label": bin_label,
            })

    print(f"共 {len(all_tasks)} 个 clip 任务，开始并行处理...")

    # ── 并行执行 ─────────────────────────────────────────────────────────────
    bin_logs  = {bl: [] for bl in bin_dirs}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one, t): t for t in all_tasks}
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            seg       = res["seg"]
            t0_str    = seg["start_ts"]
            t1_str    = seg["end_ts"]
            conf      = seg["conf_mean"]
            conf_max  = seg["conf_max"]
            bin_label = res["bin_label"]
            stem1     = res["stem1"]
            suffix    = res["suffix"]

            parts = [
                "cam1✅" if res["ok_cam1"] else ("cam1⚠️" if not res["cam1_mp4"] else "cam1❌"),
                "cam2✅" if res["ok_cam2"] else ("cam2⚠️" if not res["cam2_mp4"] else "cam2❌"),
                "csv✅"  if res["ok_csv"]  else "csv❌",
            ]
            status = " ".join(parts)
            print(f"  [{done}/{len(all_tasks)}] [{bin_label}] {res['csv_basename']}  "
                  f"{t0_str[11:19]}→{t1_str[11:19] if t1_str else '?'}  "
                  f"conf={conf:.2f}  {status}")

            line = (f"{res['csv_basename']}\t{t0_str[11:19]}\t{t1_str[11:19] if t1_str else '?'}"
                    f"\tconf_mean={conf:.3f}\tconf_max={conf_max:.3f}"
                    f"\t{stem1 + suffix + '.mp4'}\t{status}")
            with lock:
                bin_logs[bin_label].append(line)

    for bin_label, lines in sorted(bin_logs.items()):
        log_path = os.path.join(args.output_dir, f"scratch_log_review_{bin_label}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("csv_file\tstart\tend\tconf_mean\tconf_max\tclip_file\tstatus\n")
            f.write("\n".join(lines) + "\n")
        print(f"  → {log_path}  ({len(lines)} 条)")

    total = sum(len(v) for v in bin_logs.values())
    print(f"\n共 {total} 段抓挠，分布在 {len(bin_logs)} 个置信度区间")
    for bl in sorted(bin_logs):
        print(f"  clips_{bl}/: {len(bin_logs[bl])} 段")


if __name__ == "__main__":
    main()
