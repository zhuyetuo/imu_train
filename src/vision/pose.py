"""
用 YOLOv8-pose 从视频或图片提取狗的关键点，保存为 JSON。

用法:
  python src/video/pose.py --source data/raw_vision/videos/dog01.mp4 --output data/processed_vision/poses/dog01_pose.json
  python src/video/pose.py --source data/raw_vision/videos/ --output data/processed_vision/poses/  # 批量处理目录
"""

import argparse
import json
import os
import sys


def extract_pose(source: str, output: str, model_path: str = "yolov8n-pose.pt",
                 fps: int = 25, conf: float = 0.5, device: str = ""):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[pose] 请先安装: pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_path)
    if device:
        model.to(device)

    import cv2

    def process_video(video_path: str, out_path: str):
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, round(src_fps / fps))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        records = []
        frame_idx = 0
        read_idx = 0

        print(f"[pose] {os.path.basename(video_path)}  src_fps={src_fps:.1f}  step={step}  total={total}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if read_idx % step == 0:
                results = model(frame, conf=conf, verbose=False)
                r = results[0]
                dogs = []
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    for kp, box in zip(r.keypoints.data, r.boxes.data):
                        dogs.append({
                            "bbox": box[:4].cpu().tolist(),
                            "confidence": float(box[4]),
                            "keypoints": kp.cpu().tolist(),   # (N, 3): x, y, conf
                        })
                records.append({
                    "frame": frame_idx,
                    "time_s": round(frame_idx / fps, 4),
                    "dogs": dogs,
                })
                frame_idx += 1
            read_idx += 1

        cap.release()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"source": video_path, "fps": fps, "frames": records}, f)
        print(f"[pose] 保存 {len(records)} 帧 → {out_path}")

    # 单文件 or 目录批量
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    if os.path.isdir(source):
        files = [f for f in os.listdir(source) if f.lower().endswith(video_exts)]
        os.makedirs(output, exist_ok=True)
        for fname in sorted(files):
            stem = os.path.splitext(fname)[0]
            process_video(os.path.join(source, fname),
                          os.path.join(output, f"{stem}_pose.json"))
    else:
        stem = os.path.splitext(os.path.basename(source))[0]
        out_path = output if output.endswith(".json") else os.path.join(output, f"{stem}_pose.json")
        process_video(source, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="视频文件或目录")
    parser.add_argument("--output", required=True, help="输出 JSON 文件或目录")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLOv8-pose 模型权重")
    parser.add_argument("--fps", type=int, default=25, help="提帧采样率（默认 25）")
    parser.add_argument("--conf", type=float, default=0.5, help="关键点置信度阈值")
    parser.add_argument("--device", default="", help="cuda / cpu / 空=自动")
    args = parser.parse_args()
    extract_pose(args.source, args.output, args.model, args.fps, args.conf, args.device)
