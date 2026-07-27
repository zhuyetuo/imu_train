"""
把推理结果或裁剪片段转换为 Label Studio 任务。

两种模式:
  --use_clips   扫描 clips_*/ 目录，每个 clip 三件套(cam1.mp4 + cam2.mp4 + cam1.csv)
                生成一个 Label Studio 任务（推荐，与 witmotion_imu 格式一致）
  默认          从 *_infer.json 生成任务（引用完整录制文件）

Label Studio 项目配置:
  video1 = cam1 视频, video2 = cam2 视频, csv1 = cam1 IMU CSV
  from_name="label", to_name="ts"

用法:
  # 裁剪片段模式（推荐）
  python src/review_to_labelstudio.py \
    --infer_dir infer_result/2026_7_17 \
    --output infer_result/2026_7_17/labelstudio_review.json \
    --use_clips \
    --csv_url_prefix http://192.168.2.140:8182

  # 全录制模式
  python src/review_to_labelstudio.py \
    --infer_dir infer_result/2026_7_17/_infer \
    --output infer_result/2026_7_17/labelstudio_review.json \
    --csv_url_prefix http://192.168.2.140:8182
"""

import argparse
import glob
import json
import os
import re


# ── 文件名解析 ────────────────────────────────────────────────────────────────

def parse_cam(basename: str) -> str:
    """返回 'cam1' / 'cam2' / ''"""
    m = re.search(r"(cam\d)", basename, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def session_key(basename: str) -> str:
    """提取会话前缀（cam_tag 之前的部分）"""
    stem = os.path.splitext(basename)[0]
    m = re.search(r"(cam\d)", stem, re.IGNORECASE)
    if not m:
        return stem
    return stem[: m.start()].rstrip("_")


def clip_key(basename: str) -> str:
    """提取 clip 唯一键（去掉 cam1/cam2 差异）
    multicam_20260717_185620_cam1_imu1_resampled16hz_clip01_185907-185910
    → multicam_20260717_185620_clip01_185907-185910
    """
    stem = os.path.splitext(basename)[0]
    stem = re.sub(r"_cam\d_imu\d", "", stem)
    return stem


# ── URL 构建 ──────────────────────────────────────────────────────────────────

def csv_url(basename: str, csv_prefix: str) -> str:
    return f"{csv_prefix.rstrip('/')}/{basename}"


def video_url(basename: str, video_prefix: str) -> str:
    stem = os.path.splitext(basename)[0]
    return f"{video_prefix.rstrip('/')}/{stem}.mp4"


# ── 标注生成 ──────────────────────────────────────────────────────────────────

def make_annotation(start_ts, end_ts, label):
    return {
        "type": "timeserieslabels",
        "from_name": "label",
        "to_name": "ts",
        "value": {
            "start": start_ts or "",
            "end":   end_ts   or "",
            "timeserieslabels": [label],
        }
    }


# ── 模式 1：clips_*/ 目录扫描 ─────────────────────────────────────────────────

def build_tasks_from_clips(infer_dir, csv_url_prefix, video_url_prefix, label_name):
    """
    扫描 infer_dir/clips_*/ 目录，按 clip 键分组：
      cam1 mp4 → video1
      cam2 mp4 → video2
      cam1 csv → csv1 + 标注（覆盖整个 clip）
    """
    tasks = []
    task_id = 1

    clip_dirs = sorted(glob.glob(os.path.join(infer_dir, "clips_*")))
    if not clip_dirs:
        print(f"[警告] {infer_dir} 下没有 clips_*/ 目录，请先运行 extract_clips.py")
        return tasks

    for clip_dir in clip_dirs:
        # 按 clip_key 分组
        groups = {}  # clip_key → {cam1_mp4, cam2_mp4, cam1_csv}

        for fname in sorted(os.listdir(clip_dir)):
            fpath = os.path.join(clip_dir, fname)
            if not os.path.isfile(fpath):
                continue
            cam = parse_cam(fname)
            key = clip_key(fname)
            if key not in groups:
                groups[key] = {}
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".mp4", ".avi", ".mov"):
                if cam in ("cam1", ""):
                    groups[key]["cam1_mp4"] = fname
                elif cam == "cam2":
                    groups[key]["cam2_mp4"] = fname
            elif ext == ".csv":
                if cam in ("cam1", ""):
                    groups[key]["cam1_csv"] = fname

        for key in sorted(groups):
            g = groups[key]
            cam1_mp4_name = g.get("cam1_mp4", "")
            cam2_mp4_name = g.get("cam2_mp4", "")
            cam1_csv_name = g.get("cam1_csv", "")

            if not cam1_mp4_name and not cam1_csv_name:
                continue

            v1_url = video_url(cam1_mp4_name, video_url_prefix) if cam1_mp4_name else ""
            v2_url = video_url(cam2_mp4_name, video_url_prefix) if cam2_mp4_name else v1_url
            c1_url = csv_url(cam1_csv_name, csv_url_prefix) if cam1_csv_name else ""

            task_data = {
                "video1": v1_url,
                "video2": v2_url,
                "csv1":   c1_url,
            }

            # 从 key 解析日期和时间（如 multicam_20260717_185620_clip03_190408-190410）
            # 日期从会话前缀中提取：multicam_YYYYMMDD_...
            date_m = re.search(r"(\d{4})(\d{2})(\d{2})_\d{6}", key)
            time_m = re.search(r"(\d{6})-(\d{6})$", key)
            if date_m and time_m:
                date_str = f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}"
                def hms_to_ts(hms: str) -> str:
                    return f"{date_str} {hms[:2]}:{hms[2:4]}:{hms[4:6]}.000"
                start_ts = hms_to_ts(time_m.group(1))
                end_ts   = hms_to_ts(time_m.group(2))
                results = [make_annotation(start_ts, end_ts, label_name)]
            else:
                results = []

            tasks.append({
                "id":   task_id,
                "data": task_data,
                "annotations": [{"result": results}] if results else [],
                "meta": {
                    "clip_key": key,
                    "bin":      os.path.basename(clip_dir),
                    "note":     "模型检测到抓挠片段，请核实",
                }
            })
            task_id += 1

    return tasks


# ── 模式 2：_infer.json 扫描（全录制文件）────────────────────────────────────

def build_tasks_from_infer(infer_jsons, csv_url_prefix, video_url_prefix,
                           mode, low_threshold, high_threshold, label_name):
    sessions = {}
    for infer_path in sorted(infer_jsons):
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)
        csv_basename = data["csv_basename"]
        sess = session_key(csv_basename)
        cam  = parse_cam(csv_basename)
        if sess not in sessions:
            sessions[sess] = {}
        sessions[sess][cam or "cam1"] = data

    tasks = []
    task_id = 1

    for sess in sorted(sessions):
        cams = sessions[sess]
        main_data = cams.get("cam1") or next(iter(cams.values()))
        main_csv  = main_data["csv_basename"]
        stem1     = os.path.splitext(main_csv)[0]
        stem2     = re.sub(r"cam1_imu1", "cam2_imu2", stem1)

        c1_url = csv_url(main_csv, csv_url_prefix)
        v1_url = f"{video_url_prefix.rstrip('/')}/{stem1}.mp4"
        v2_url = f"{video_url_prefix.rstrip('/')}/{stem2}.mp4" if "cam2" in cams else v1_url

        task_data = {"csv1": c1_url, "video1": v1_url, "video2": v2_url}

        scratch_segs = main_data.get("scratch_segments", [])
        windows      = main_data.get("windows", [])
        results      = []

        if mode in ("scratch_only", "all") and scratch_segs:
            for seg in scratch_segs:
                if seg.get("start_ts") and seg.get("end_ts"):
                    results.append(make_annotation(seg["start_ts"], seg["end_ts"], label_name))

        if mode in ("uncertain", "all"):
            for w in windows:
                prob = w.get("probs", {}).get(label_name, 0.0)
                if w.get("label") != label_name and low_threshold < prob <= high_threshold:
                    if w.get("ts"):
                        results.append(make_annotation(w["ts"], w["ts"], f"{label_name}?"))

        if not results:
            continue

        tasks.append({
            "id":   task_id,
            "data": task_data,
            "annotations": [{"result": results}],
            "meta": {
                "session":  sess,
                "csv_file": main_csv,
                "note":     f"模型检测到 {len(scratch_segs)} 段抓挠，请核实",
            }
        })
        task_id += 1

    return tasks


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="推理结果 → Label Studio 复查任务")
    parser.add_argument("--infer_dir",  required=True,
                        help="包含 *_infer.json 或 clips_*/ 的目录")
    parser.add_argument("--output",     required=True,
                        help="输出 Label Studio JSON 路径")
    parser.add_argument("--csv_url_prefix", default="http://192.168.2.140:8182",
                        help="CSV 文件 URL 前缀")
    parser.add_argument("--video_url_prefix", default="",
                        help="MP4 文件 URL 前缀（默认 csv_url_prefix/transcoded）")
    parser.add_argument("--use_clips",  action="store_true",
                        help="扫描 clips_*/ 目录（推荐），而非 _infer.json")
    parser.add_argument("--mode", default="scratch_only",
                        choices=["scratch_only", "uncertain", "all"])
    parser.add_argument("--low_threshold",  type=float, default=0.3)
    parser.add_argument("--high_threshold", type=float, default=0.65)
    parser.add_argument("--label", default="抓挠")
    args = parser.parse_args()

    video_prefix = args.video_url_prefix or f"{args.csv_url_prefix.rstrip('/')}/transcoded"

    if args.use_clips:
        print("模式: clips 裁剪片段")
        tasks = build_tasks_from_clips(
            args.infer_dir, args.csv_url_prefix, video_prefix, args.label)
    else:
        infer_jsons = glob.glob(os.path.join(args.infer_dir, "**", "*_infer.json"), recursive=True)
        infer_jsons += glob.glob(os.path.join(args.infer_dir, "*_infer.json"))
        infer_jsons = sorted(set(infer_jsons))
        if not infer_jsons:
            print(f"[错误] {args.infer_dir} 下没有找到 *_infer.json")
            return
        print(f"模式: 全录制文件，找到 {len(infer_jsons)} 个推理结果")
        tasks = build_tasks_from_infer(
            infer_jsons, args.csv_url_prefix, video_prefix,
            args.mode, args.low_threshold, args.high_threshold, args.label)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"共生成 {len(tasks)} 个 Label Studio 任务")
    print(f"已保存: {args.output}")
    print(f"\n导入方式: Label Studio → Import → 选择上面的 JSON 文件")


if __name__ == "__main__":
    main()
