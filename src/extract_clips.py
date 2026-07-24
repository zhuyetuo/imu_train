"""
读取推理 JSON，按置信度区间把抓挠片段分桶：
  - scratch_log_review_0.5-0.6.txt  ← 该区间的文字记录
  - clips_0.5-0.6/                  ← 视频片段 + 对应 CSV

用法:
  python src/extract_clips.py \
    --infer_dir  infer_result/2026_7_23 \
    --video_dir  data/raw_custom/data/2026_7_23 \
    --output_dir infer_result/2026_7_23 \
    --context_s  3
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

# 置信度区间边界
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def get_bin(conf: float):
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


def find_video(csv_basename: str, video_dir: str) -> str:
    stem = os.path.splitext(csv_basename)[0]
    for ext in (".mp4", ".MP4", ".avi", ".mov"):
        p = os.path.join(video_dir, stem + ext)
        if os.path.exists(p):
            return p
    return ""


def cut_clip(video_path: str, start_sec: float, end_sec: float, out_path: str, context_s: float) -> bool:
    t_start  = max(0.0, start_sec - context_s)
    duration = (end_sec + context_s) - t_start
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-an",
        out_path,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def main():
    parser = argparse.ArgumentParser(description="按置信度区间裁剪抓挠视频片段")
    parser.add_argument("--infer_dir",  required=True, help="推理 JSON 目录（*_infer.json）")
    parser.add_argument("--video_dir",  required=True, help="原始 MP4 + CSV 目录")
    parser.add_argument("--output_dir", required=True, help="输出根目录（clips_*/ 和 log 放在此处）")
    parser.add_argument("--context_s",  type=float, default=3.0, help="前后上下文秒数（默认 3s）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer_jsons = sorted(glob.glob(os.path.join(args.infer_dir, "*_infer.json")))
    if not infer_jsons:
        print(f"[错误] {args.infer_dir} 下没有 *_infer.json 文件")
        sys.exit(1)

    has_ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    if not has_ffmpeg:
        print("[警告] 找不到 ffmpeg，将只复制 CSV，跳过视频裁剪")

    # bin_label → list of log lines
    bin_logs  = {}
    # bin_label → clip 目录
    bin_dirs  = {}

    for infer_path in infer_jsons:
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)

        csv_basename = data["csv_basename"]
        segs         = data.get("scratch_segments", [])
        if not segs:
            continue

        csv_src    = os.path.join(args.video_dir, csv_basename)
        video_path = find_video(csv_basename, args.video_dir)

        for idx, seg in enumerate(segs, 1):
            conf     = seg.get("conf_mean", 0.0)
            conf_max = seg.get("conf_max",  0.0)
            t0_str   = seg.get("start_ts", "") or ""
            t1_str   = seg.get("end_ts",   "") or ""
            if not t0_str:
                continue

            bin_label = get_bin(conf)

            # 初始化 bin
            if bin_label not in bin_logs:
                bin_logs[bin_label] = []
                clip_dir = os.path.join(args.output_dir, f"clips_{bin_label}")
                os.makedirs(clip_dir, exist_ok=True)
                bin_dirs[bin_label] = clip_dir

            clip_dir = bin_dirs[bin_label]
            stem     = os.path.splitext(csv_basename)[0]
            t0_label = t0_str[11:19].replace(":", "h", 1).replace(":", "m") + "s"

            # 复制对应 CSV 到 clip_dir（同名只复制一次）
            if os.path.exists(csv_src):
                dst_csv = os.path.join(clip_dir, csv_basename)
                if not os.path.exists(dst_csv):
                    shutil.copy2(csv_src, dst_csv)

            # 裁剪视频
            clip_name = f"{stem}_scratch_{idx:02d}_{t0_label}.mp4"
            clip_path = os.path.join(clip_dir, clip_name)
            ok = False
            if video_path and has_ffmpeg:
                start_sec = ts_to_sec(t0_str)
                end_sec   = ts_to_sec(t1_str) if t1_str else start_sec + 2.0
                ok = cut_clip(video_path, start_sec, end_sec, clip_path, args.context_s)

            status = "✅" if ok else ("⚠️ 无视频" if not video_path else "❌ffmpeg失败")
            line = (f"{csv_basename}\t{t0_str[11:19]}\t{t1_str[11:19] if t1_str else '?'}"
                    f"\tconf_mean={conf:.3f}\tconf_max={conf_max:.3f}\t{clip_name}\t{status}")
            bin_logs[bin_label].append(line)
            print(f"  [{bin_label}] {csv_basename} #{idx}  {t0_str[11:19]}→{t1_str[11:19] if t1_str else '?'}  conf={conf:.2f}  {status}")

    # 写 scratch_log_review_*.txt
    for bin_label, lines in sorted(bin_logs.items()):
        log_path = os.path.join(args.output_dir, f"scratch_log_review_{bin_label}.txt")
        header = "csv_file\tstart\tend\tconf_mean\tconf_max\tclip_file\tstatus"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
            f.write("\n".join(lines) + "\n")
        print(f"  → {log_path}  ({len(lines)} 条)")

    total = sum(len(v) for v in bin_logs.values())
    print(f"\n共 {total} 段抓挠，分布在 {len(bin_logs)} 个置信度区间")
    for bl in sorted(bin_logs):
        print(f"  clips_{bl}/: {len(bin_logs[bl])} 段")


if __name__ == "__main__":
    main()
