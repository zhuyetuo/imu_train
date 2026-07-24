"""
读取推理 JSON，从对应 MP4 裁剪出抓挠片段（及前后上下文），保存为短视频。

用法:
  python src/extract_clips.py \
    --infer_dir infer_result/2026_7_23 \
    --video_dir data/raw_custom/data/2026_7_23 \
    --clip_dir  infer_result/2026_7_23/clips \
    --context_s 3

输出（clip_dir 下）:
  multicam_20260723_184859877_cam1_imu1_resampled16hz_scratch_01_08h52m37s.mp4
  multicam_20260723_184859877_cam1_imu1_resampled16hz_scratch_02_08h55m13s.mp4
  ...
  index.txt   ← 所有片段汇总（文件名、时间、置信度）
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta


def ts_to_sec(ts_str: str) -> float:
    """'2026-07-23 18:52:37.123' → 距当天 00:00 的秒数。"""
    if not ts_str:
        return 0.0
    dt = datetime.strptime(ts_str[:23], "%Y-%m-%d %H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


def find_video(csv_basename: str, video_dir: str) -> str:
    """根据 CSV 文件名找同名 MP4。"""
    stem = os.path.splitext(csv_basename)[0]
    for ext in (".mp4", ".MP4", ".avi", ".mov"):
        p = os.path.join(video_dir, stem + ext)
        if os.path.exists(p):
            return p
    return ""


def cut_clip(video_path: str, start_sec: float, end_sec: float, out_path: str, context_s: float):
    """用 ffmpeg 裁剪视频片段（含前后 context_s 秒）。"""
    t_start = max(0.0, start_sec - context_s)
    duration = (end_sec + context_s) - t_start
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t_start:.3f}",
        "-i", video_path,
        "-t",  f"{duration:.3f}",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="从推理结果裁剪抓挠视频片段")
    parser.add_argument("--infer_dir",  required=True, help="推理 JSON 目录")
    parser.add_argument("--video_dir",  required=True, help="原始 MP4 所在目录")
    parser.add_argument("--clip_dir",   required=True, help="输出片段目录")
    parser.add_argument("--context_s",  type=float, default=3.0,
                        help="抓挠片段前后额外保留的秒数（默认 3s）")
    parser.add_argument("--min_conf",   type=float, default=0.0,
                        help="只裁剪平均置信度 >= 此值的片段（默认 0=全部）")
    args = parser.parse_args()

    os.makedirs(args.clip_dir, exist_ok=True)

    infer_jsons = sorted(glob.glob(os.path.join(args.infer_dir, "*_infer.json")))
    if not infer_jsons:
        print(f"[错误] {args.infer_dir} 下没有 *_infer.json 文件")
        sys.exit(1)

    # 检查 ffmpeg
    if subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0:
        print("[错误] 找不到 ffmpeg，请先安装: sudo apt install ffmpeg")
        sys.exit(1)

    index_lines = ["文件\t开始\t结束\t置信度(mean)\t置信度(max)\t片段文件"]
    total_clips = 0

    for infer_path in infer_jsons:
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)

        csv_basename = data["csv_basename"]
        segs = data.get("scratch_segments", [])
        if not segs:
            continue

        video_path = find_video(csv_basename, args.video_dir)
        if not video_path:
            print(f"  [跳过] 找不到对应 MP4: {csv_basename}")
            continue

        stem = os.path.splitext(csv_basename)[0]
        print(f"\n── {csv_basename} → {os.path.basename(video_path)}")

        for idx, seg in enumerate(segs, 1):
            conf_mean = seg.get("conf_mean", 0.0)
            conf_max  = seg.get("conf_max",  0.0)
            if conf_mean < args.min_conf:
                print(f"  [跳过] 片段 {idx}: 置信度 {conf_mean:.2f} < {args.min_conf}")
                continue

            t0_str = seg.get("start_ts", "")
            t1_str = seg.get("end_ts",   "")
            if not t0_str or not t1_str:
                continue

            start_sec = ts_to_sec(t0_str)
            end_sec   = ts_to_sec(t1_str)
            if end_sec <= start_sec:
                end_sec = start_sec + 2.0

            # 时间标签（用于文件名）
            t0_label = t0_str[11:19].replace(":", "h", 1).replace(":", "m") + "s"
            clip_name = f"{stem}_scratch_{idx:02d}_{t0_label}.mp4"
            clip_path = os.path.join(args.clip_dir, clip_name)

            ok = cut_clip(video_path, start_sec, end_sec, clip_path, args.context_s)
            if ok:
                print(f"  ✅ 片段 {idx}: {t0_str[11:19]} → {t1_str[11:19]}  conf={conf_mean:.2f}  → {clip_name}")
                index_lines.append(f"{csv_basename}\t{t0_str[11:19]}\t{t1_str[11:19]}\t{conf_mean:.3f}\t{conf_max:.3f}\t{clip_name}")
                total_clips += 1
            else:
                print(f"  ❌ 片段 {idx}: ffmpeg 失败 {t0_str[11:19]} → {t1_str[11:19]}")

    # 写 index.txt
    index_path = os.path.join(args.clip_dir, "index.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))

    print(f"\n共裁剪 {total_clips} 个片段 → {args.clip_dir}/")
    print(f"片段索引: {index_path}")


if __name__ == "__main__":
    main()
