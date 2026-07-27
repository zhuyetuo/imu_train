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


def find_video(stem: str, video_dir: str) -> str:
    for ext in (".mp4", ".MP4", ".avi", ".mov"):
        p = os.path.join(video_dir, stem + ext)
        if os.path.exists(p):
            return p
    return ""


def sibling_stem(stem: str, target_cam: int) -> str:
    """cam1_imu1 ↔ cam2_imu2"""
    return re.sub(r"cam(\d)_imu(\d)", f"cam{target_cam}_imu{target_cam}", stem)


def cut_clip(video_path: str, start_sec: float, end_sec: float,
             out_path: str, context_s: float) -> bool:
    t_start  = max(0.0, start_sec - context_s)
    duration = (end_sec + context_s) - t_start
    # -i 先于 -ss：精确 seek，避免快速 seek 产生损坏帧
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", f"{t_start:.3f}",
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-avoid_negative_ts", "make_zero",
        "-map_metadata", "-1",
        out_path,
    ]
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0:
        print(f"    [ffmpeg错误] {ret.stderr.decode('utf-8', errors='replace')[-300:]}")
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


def main():
    parser = argparse.ArgumentParser(description="按置信度区间裁剪抓挠视频片段（witmotion_imu 格式）")
    parser.add_argument("--infer_dir",  required=True)
    parser.add_argument("--video_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_s",  type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer_jsons = sorted(glob.glob(os.path.join(args.infer_dir, "*_infer.json")))
    if not infer_jsons:
        print(f"[错误] {args.infer_dir} 下没有 *_infer.json")
        sys.exit(1)

    has_ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    if not has_ffmpeg:
        print("[警告] 找不到 ffmpeg，只裁剪 CSV")

    bin_logs = {}
    bin_dirs = {}

    for infer_path in infer_jsons:
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)

        csv_basename = data["csv_basename"]
        segs = data.get("scratch_segments", [])
        if not segs:
            continue

        stem1 = os.path.splitext(csv_basename)[0]   # cam1 stem
        stem2 = sibling_stem(stem1, 2)               # cam2 stem

        src_csv1  = os.path.join(args.video_dir, csv_basename)
        cam1_mp4  = find_video(stem1, args.video_dir)
        cam2_mp4  = find_video(stem2, args.video_dir)

        for idx, seg in enumerate(segs, 1):
            conf     = seg.get("conf_mean", 0.0)
            conf_max = seg.get("conf_max",  0.0)
            t0_str   = seg.get("start_ts", "") or ""
            t1_str   = seg.get("end_ts",   "") or ""
            if not t0_str:
                continue

            start_sec = ts_to_sec(t0_str)
            end_sec   = ts_to_sec(t1_str) if t1_str else start_sec + 2.0
            pad_start = max(0.0, start_sec - args.context_s)
            pad_end   = end_sec + args.context_s

            bin_label = get_bin(conf)
            if bin_label not in bin_dirs:
                bin_logs[bin_label] = []
                clip_dir = os.path.join(args.output_dir, f"clips_{bin_label}")
                os.makedirs(clip_dir, exist_ok=True)
                bin_dirs[bin_label] = clip_dir
            clip_dir = bin_dirs[bin_label]

            # 命名: {stem}_clip{N:02d}_{HHMMSS}-{HHMMSS}
            suffix = f"_clip{idx:02d}_{ts_to_hhmmss(t0_str)}-{ts_to_hhmmss(t1_str or t0_str)}"

            cam1_clip_mp4 = os.path.join(clip_dir, stem1 + suffix + ".mp4")
            cam2_clip_mp4 = os.path.join(clip_dir, stem2 + suffix + ".mp4")
            cam1_clip_csv = os.path.join(clip_dir, stem1 + suffix + ".csv")

            ok_cam1 = cut_clip(cam1_mp4, start_sec, end_sec, cam1_clip_mp4, args.context_s) if (cam1_mp4 and has_ffmpeg) else False
            ok_cam2 = cut_clip(cam2_mp4, start_sec, end_sec, cam2_clip_mp4, args.context_s) if (cam2_mp4 and has_ffmpeg) else False
            ok_csv  = slice_csv(src_csv1, cam1_clip_csv, pad_start, pad_end) if os.path.exists(src_csv1) else False

            parts = [
                "cam1✅" if ok_cam1 else ("cam1⚠️" if not cam1_mp4 else "cam1❌"),
                "cam2✅" if ok_cam2 else ("cam2⚠️" if not cam2_mp4 else "cam2❌"),
                "csv✅"  if ok_csv  else "csv❌",
            ]
            status = " ".join(parts)
            print(f"  [{bin_label}] {csv_basename} #{idx}  "
                  f"{t0_str[11:19]}→{t1_str[11:19] if t1_str else '?'}  "
                  f"conf={conf:.2f}  {status}")

            line = (f"{csv_basename}\t{t0_str[11:19]}\t{t1_str[11:19] if t1_str else '?'}"
                    f"\tconf_mean={conf:.3f}\tconf_max={conf_max:.3f}"
                    f"\t{stem1 + suffix + '.mp4'}\t{status}")
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
