"""
快速验证：用预训练 YOLOv8-pose 跑你的视频，输出关键点可视化结果。

步骤:
  1. 自动下载 yolov8n-pose.pt（首次运行）
  2. 逐帧提取狗的关键点
  3. 输出标注后的视频 + 关键点 JSON

用法:
  pip install ultralytics opencv-python

  # 基础用法
  python src/vision/quick_validate.py --video data/raw_vision/videos/dog01.mp4

  # 使用 Dog-Pose 专用模型（24关键点，需先训练或指定）
  python src/vision/quick_validate.py --video dog01.mp4 --model runs/pose/train/weights/best.pt

  # 批量处理目录
  python src/vision/quick_validate.py --video data/raw_vision/videos/
"""

import argparse
import json
import os
import sys


def run(video_path: str, output_dir: str, model_path: str,
        fps_out: int, conf: float, show_skeleton: bool, imgsz: int = 1280):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        print("请先安装: pip install ultralytics opencv-python")
        sys.exit(1)

    import queue
    import threading

    model = YOLO(model_path)
    model.fuse()  # 融合 BN 层，推理更快
    print(f"[validate] 模型: {model_path}")
    print(f"[validate] 视频: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[validate] 无法打开视频: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps_out))

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(output_dir, f"{stem}_keypoints.mp4")
    out_json_path  = os.path.join(output_dir, f"{stem}_pose.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_out, (w, h))

    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    print(f"[validate] 分辨率={w}x{h}  原始fps={src_fps:.1f}  输出fps={fps_out}  总帧={total}")

    # ── 后台线程：预读帧到队列，与 GPU 推理并行 ──────────────────────────────
    frame_queue = queue.Queue(maxsize=8)

    def frame_reader():
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                frame_queue.put(None)
                break
            if idx % step == 0:
                frame_queue.put((idx, frame))
            idx += 1

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    # ── 后台线程：异步写视频，与 GPU 推理并行 ────────────────────────────────
    write_queue = queue.Queue(maxsize=8)

    def frame_writer():
        while True:
            item = write_queue.get()
            if item is None:
                break
            writer.write(item)

    writer_thread = threading.Thread(target=frame_writer, daemon=True)
    writer_thread.start()

    frames_data  = []
    frame_out_idx = 0
    dogs_detected = 0
    no_dog_frames = 0

    while True:
        item = frame_queue.get()
        if item is None:
            break
        read_idx, frame = item

        results = model(frame, conf=conf, verbose=False, imgsz=imgsz, half=True)
        r = results[0]

        dogs = []
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kps_all   = r.keypoints.data.cpu().numpy()
            boxes_all = r.boxes.data.cpu().numpy()
            for kps, box in zip(kps_all, boxes_all):
                cls_id = int(box[5]) if len(box) > 5 else 0
                dogs.append({
                    "bbox":       box[:4].tolist(),
                    "confidence": float(box[4]),
                    "behavior":   r.names.get(cls_id, str(cls_id)),
                    "keypoints":  kps.tolist(),
                })
            dogs_detected += 1

            if show_skeleton:
                for kps, box in zip(kps_all, boxes_all):
                    x1, y1, x2, y2 = map(int, box[:4])
                    cls_id   = int(box[5]) if len(box) > 5 else 0
                    cls_name = r.names.get(cls_id, str(cls_id))
                    label    = f"{cls_name} {float(box[4]):.0%}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), (0, 200, 0), -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    for a, b in SKELETON:
                        if a < len(kps) and b < len(kps):
                            xa, ya, va = kps[a]; xb, yb, vb = kps[b]
                            if va > 0.3 and vb > 0.3:
                                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                                         (255, 128, 0), 2)
                    for kp in kps:
                        x, y, v = kp
                        if v > 0.3:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 100, 255), -1)
        else:
            no_dog_frames += 1

        behaviors    = [d.get("behavior", "") for d in dogs if d.get("behavior")]
        behavior_str = ", ".join(behaviors) if behaviors else ""
        info = f"frame {frame_out_idx}  {behavior_str or f'dogs={len(dogs)}'}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frames_data.append({
            "frame":  frame_out_idx,
            "time_s": round(frame_out_idx / fps_out, 4),
            "dogs":   dogs,
        })
        write_queue.put(frame)
        frame_out_idx += 1

        if frame_out_idx % 100 == 0:
            det_rate = dogs_detected / max(frame_out_idx, 1) * 100
            print(f"  {frame_out_idx} 帧  检测率={det_rate:.0f}%  未检出={no_dog_frames}")

    write_queue.put(None)
    writer_thread.join()
    cap.release()
    writer.release()

    # 保存 JSON
    with open(out_json_path, "w") as f:
        json.dump({"source": video_path, "fps": fps_out, "frames": frames_data}, f)

    det_rate = dogs_detected / max(frame_out_idx, 1) * 100
    print(f"\n[validate] 完成！")
    print(f"  总帧数:   {frame_out_idx}")
    print(f"  检测率:   {det_rate:.1f}%（检出狗的帧）")
    print(f"  未检出:   {no_dog_frames} 帧")
    print(f"  视频输出: {out_video_path}")
    print(f"  JSON输出: {out_json_path}")

    if det_rate < 50:
        print(f"\n  ⚠️  检测率偏低，建议:")
        print(f"     - 降低置信度阈值: --conf 0.3")
        print(f"     - 检查视频角度，狗是否清晰可见")
    else:
        print(f"\n  ✅ 关键点质量良好，可以进行行为标注和训练")
        print(f"     下一步: 对视频片段手动标注行为标签 → data/annotations_vision/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True,
                        help="视频文件或目录")
    parser.add_argument("--output_dir", default="results/vision/validate",
                        help="输出目录（默认 results/vision/validate）")
    parser.add_argument("--model", default="yolov8n-pose.pt",
                        help="模型权重，默认 yolov8n-pose.pt（自动下载）")
    parser.add_argument("--fps", type=int, default=10,
                        help="输出视频帧率，同时控制关键点提取密度（默认 10）")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="检测置信度阈值（默认 0.5，降低可提高检测率）")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="推理图像尺寸（默认 1280，4K 视频建议用 1280 或 1920）")
    parser.add_argument("--no_skeleton", action="store_true",
                        help="不绘制骨骼线，只画关键点")
    args = parser.parse_args()

    import glob
    video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    if os.path.isdir(args.video):
        files = []
        for ext in video_exts:
            files += glob.glob(os.path.join(args.video, ext))
        if not files:
            print(f"[validate] 目录下没有视频文件: {args.video}")
            sys.exit(1)
        for f in sorted(files):
            run(f, args.output_dir, args.model, args.fps, args.conf,
                not args.no_skeleton, args.imgsz)
    else:
        run(args.video, args.output_dir, args.model, args.fps, args.conf,
            not args.no_skeleton, args.imgsz)


if __name__ == "__main__":
    main()
