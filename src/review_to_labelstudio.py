"""
把推理结果 JSON 转换为 Label Studio 任务，用于人工复查和重新标注。

Label Studio 项目配置格式（单摄像头 + 双视频）:
  数据键: csv1 (IMU TimeSeries), video1, video2 (可选)
  标注:   from_name="label", to_name="ts"

文件命名约定:
  multicam_20260717_185620_cam1_imu1_resampled16hz.csv  → csv1
  multicam_20260717_185620_cam1_imu1_resampled16hz.mp4  → video1
  multicam_20260717_185620_cam2_imu2_resampled16hz.mp4  → video2

用法:
  python src/review_to_labelstudio.py \
    --infer_dir infer_result/2026_7_15/_infer \
    --output infer_result/2026_7_15/labelstudio_review.json \
    --csv_url_prefix http://localhost:8080/data/local-files/?d=raw_wit \
    --mode scratch_only
"""

import argparse
import glob
import json
import os
import re


# ── 文件名解析 ────────────────────────────────────────────────────────────────

def parse_filename(basename: str):
    """
    解析多摄像头文件名，提取会话 key 和角色 (cam1/cam2)。

    示例:
      multicam_20260717_185620_cam1_imu1_resampled16hz.csv
        → session="multicam_20260717_185620", role="cam1"
      multicam_20260717_185620_cam2_imu2_resampled16hz.csv
        → session="multicam_20260717_185620", role="cam2"
    """
    stem = os.path.splitext(basename)[0]
    # 匹配 cam1 或 cam2
    m = re.search(r"(cam\d)", stem, re.IGNORECASE)
    if not m:
        return stem, "cam1"  # 单摄像头文件，默认 cam1
    cam_tag = m.group(1).lower()  # "cam1" / "cam2"
    # 会话前缀 = cam_tag 之前的部分
    session = stem[: m.start()].rstrip("_")
    return session, cam_tag


def build_urls(csv_basename: str, csv_url_prefix: str, video_url_prefix: str):
    """根据 CSV 文件名构建 CSV 和 MP4 的 URL。"""
    stem = os.path.splitext(csv_basename)[0]
    csv_url = f"{csv_url_prefix.rstrip('/')}/{csv_basename}"
    mp4_url = f"{video_url_prefix.rstrip('/')}/{stem}.mp4"
    return csv_url, mp4_url


# ── 标注生成 ──────────────────────────────────────────────────────────────────

def make_annotation(start_ts, end_ts, label):
    """生成一条 Label Studio timeserieslabels 标注。"""
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


# ── 任务构建 ──────────────────────────────────────────────────────────────────

def build_tasks(infer_jsons, csv_url_prefix, video_url_prefix, mode, low_threshold,
                high_threshold, label_name):
    """
    将推理 JSON 列表转换为 Label Studio 任务。

    分组规则: 同一会话前缀（如 multicam_20260717_185620）的 cam1/cam2 合并为一个任务。

    mode:
      scratch_only  - 只上传检测到抓挠的片段
      uncertain     - 只上传低置信度疑似抓挠窗口
      all           - 两者都上传
    """
    # 按会话分组: session → {cam1: data, cam2: data}
    sessions = {}
    for infer_path in sorted(infer_jsons):
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)
        csv_basename = data["csv_basename"]
        session, cam_tag = parse_filename(csv_basename)
        if session not in sessions:
            sessions[session] = {}
        sessions[session][cam_tag] = data

    tasks = []
    task_id = 1

    for session in sorted(sessions):
        cams = sessions[session]
        # 优先使用 cam1 作为主摄像头（提供 csv1/video1）
        main_data = cams.get("cam1") or next(iter(cams.values()))
        main_csv  = main_data["csv_basename"]
        csv1_url, video1_url = build_urls(main_csv, csv_url_prefix, video_url_prefix)

        # 构建 task data（video2 必须存在，Label Studio 项目配置要求）
        if "cam2" in cams:
            cam2_csv = cams["cam2"]["csv_basename"]
            _, video2_url = build_urls(cam2_csv, csv_url_prefix, video_url_prefix)
        else:
            video2_url = video1_url  # 没有 cam2 时用 cam1 视频占位

        task_data = {
            "csv1":   csv1_url,
            "video1": video1_url,
            "video2": video2_url,
        }

        scratch_segs = main_data.get("scratch_segments", [])
        windows      = main_data.get("windows", [])
        results      = []

        # ── 模式 1：检测到的抓挠片段 ──────────────────────────────────────
        if mode in ("scratch_only", "all") and scratch_segs:
            for seg in scratch_segs:
                if seg.get("start_ts") and seg.get("end_ts"):
                    results.append(make_annotation(seg["start_ts"], seg["end_ts"], label_name))

        # ── 模式 2：低置信度疑似抓挠 ──────────────────────────────────────
        if mode in ("uncertain", "all"):
            for w in windows:
                scratch_prob = w.get("probs", {}).get(label_name, 0.0)
                if w.get("label") != label_name and low_threshold < scratch_prob <= high_threshold:
                    if w.get("ts"):
                        results.append(make_annotation(w["ts"], w["ts"], f"{label_name}?"))

        if not results:
            continue

        n_scratch = len(scratch_segs)
        tasks.append({
            "id":   task_id,
            "data": task_data,
            "annotations": [{"result": results}],
            "meta": {
                "session":  session,
                "csv_file": main_csv,
                "note":     f"模型检测到 {n_scratch} 段抓挠，请核实",
            }
        })
        task_id += 1

    return tasks


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="推理结果 → Label Studio 复查任务")
    parser.add_argument("--infer_dir",  required=True,
                        help="包含 *_infer.json 的目录")
    parser.add_argument("--output",     required=True,
                        help="输出 Label Studio JSON 路径")
    parser.add_argument("--csv_url_prefix", default="http://192.168.2.140:8182",
                        help="CSV 文件的 URL 前缀（默认 http://192.168.2.140:8182）")
    parser.add_argument("--video_url_prefix", default="",
                        help="MP4 文件的 URL 前缀（默认为 csv_url_prefix/transcoded）")
    parser.add_argument("--mode", default="scratch_only",
                        choices=["scratch_only", "uncertain", "all"],
                        help="生成任务类型（默认 scratch_only）")
    parser.add_argument("--low_threshold",  type=float, default=0.3,
                        help="疑似抓挠下限概率（默认 0.3）")
    parser.add_argument("--high_threshold", type=float, default=0.65,
                        help="疑似抓挠上限概率（默认 0.65）")
    parser.add_argument("--label", default="抓挠",
                        help="要复查的目标类别（默认 抓挠）")
    args = parser.parse_args()

    infer_jsons = glob.glob(os.path.join(args.infer_dir, "**", "*_infer.json"), recursive=True)
    infer_jsons += glob.glob(os.path.join(args.infer_dir, "*_infer.json"))
    infer_jsons = sorted(set(infer_jsons))

    if not infer_jsons:
        print(f"[错误] {args.infer_dir} 下没有找到 *_infer.json 文件")
        return

    print(f"找到 {len(infer_jsons)} 个推理结果文件")

    video_url_prefix = args.video_url_prefix or f"{args.csv_url_prefix.rstrip('/')}/transcoded"

    tasks = build_tasks(
        infer_jsons,
        csv_url_prefix=args.csv_url_prefix,
        video_url_prefix=video_url_prefix,
        mode=args.mode,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        label_name=args.label,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    n_scratch   = sum(1 for t in tasks if "scratch" in t["meta"].get("note", ""))
    print(f"共生成 {len(tasks)} 个 Label Studio 任务（{n_scratch} 个含抓挠标注）")
    print(f"已保存: {args.output}")
    print(f"\n导入方式: Label Studio → Import → 选择上面的 JSON 文件")


if __name__ == "__main__":
    main()
