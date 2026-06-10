"""
用 CLIP 零样本分类对视频/图片做狗行为识别，无需训练。

用法:
  python src/vision/clip_infer.py --video 你的视频.mp4
  python src/vision/clip_infer.py --video videos/ --fps 5   # 批量处理目录
  python src/vision/clip_infer.py --image 你的图片.jpg       # 单张图片
  python src/vision/clip_infer.py --image images/           # 批量图片目录
"""

import argparse
import csv
import os
import sys
from collections import Counter


DEFAULT_LABELS = [
    "a dog lying down resting",
    "a dog sitting",
    "a dog standing still",
    "a dog walking",
    "a dog trotting or running",
    "a dog sniffing the ground",
]

# 标签映射到简短名称（用于输出）
LABEL_SHORT = {
    "a dog lying down resting":   "Lying",
    "a dog sitting":              "Sitting",
    "a dog standing still":       "Standing",
    "a dog walking":              "Walking",
    "a dog trotting or running":  "Trotting",
    "a dog sniffing the ground":  "Sniffing",
}


def load_model(model_name: str):
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("[clip] 请先安装: pip install transformers pillow")
        sys.exit(1)
    print(f"[clip] 加载模型 {model_name} ...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def infer_video(video_path: str, output_dir: str, model, processor,
                labels: list[str], fps: int, device, draw: bool):
    try:
        import cv2
        import torch
        from PIL import Image
    except ImportError:
        print("[clip] 请先安装: pip install opencv-python torch pillow")
        sys.exit(1)

    short = [LABEL_SHORT.get(l, l) for l in labels]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[clip] 无法打开: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(src_fps / fps))

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{stem}_clip.csv")
    vid_path = os.path.join(output_dir, f"{stem}_clip.mp4") if draw else None

    writer = None
    if draw:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    # 预编码文字特征（只算一次）
    import torch
    with torch.no_grad():
        text_inputs = processor(text=labels, return_tensors="pt", padding=True).to(device)
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    records = []
    read_idx = 0
    frame_out_idx = 0

    print(f"[clip] {os.path.basename(video_path)}  src_fps={src_fps:.1f}  采样fps={fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if read_idx % step == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                img_inputs = processor(images=img, return_tensors="pt").to(device)
                img_feat = model.get_image_features(**img_inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sims = (img_feat @ text_feats.T)[0]
                probs = sims.softmax(dim=0).cpu().tolist()

            pred_idx = probs.index(max(probs))
            pred_label = short[pred_idx]
            confidence = probs[pred_idx]
            time_s = round(frame_out_idx / fps, 3)

            records.append({
                "frame": frame_out_idx,
                "time_s": time_s,
                "prediction": pred_label,
                "confidence": round(confidence, 4),
            })

            if draw:
                # 叠加标签和置信度条
                bar_w = int(w * 0.35)
                cv2.rectangle(frame, (0, 0), (bar_w, 36), (0, 0, 0), -1)
                cv2.putText(frame, f"{pred_label}  {confidence:.0%}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 128), 2)
                # 小概率条形图（右下角）
                bar_h = 18
                bar_margin = 6
                total_h = len(labels) * (bar_h + bar_margin)
                y0 = h - total_h - 10
                for i, (lb, p) in enumerate(zip(short, probs)):
                    y = y0 + i * (bar_h + bar_margin)
                    bar_len = int(p * 180)
                    color = (0, 200, 100) if i == pred_idx else (80, 80, 80)
                    cv2.rectangle(frame, (w - 200, y), (w - 200 + bar_len, y + bar_h),
                                  color, -1)
                    cv2.putText(frame, f"{lb} {p:.0%}",
                                (w - 198, y + bar_h - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                writer.write(frame)

            frame_out_idx += 1
            if frame_out_idx % 50 == 0:
                print(f"  {frame_out_idx} 帧  当前: {pred_label} ({confidence:.0%})")

        read_idx += 1

    cap.release()
    if writer:
        writer.release()

    # 保存 CSV
    with open(csv_path, "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=["frame", "time_s", "prediction", "confidence"])
        dw.writeheader()
        dw.writerows(records)

    # 统计
    counts = Counter(r["prediction"] for r in records)
    total = len(records)
    summary = "  ".join(f"{k}:{v/total:.0%}" for k, v in counts.most_common())
    print(f"\n[clip] {os.path.basename(video_path)}: {total} 帧  [{summary}]")
    print(f"  CSV  → {csv_path}")
    if vid_path:
        print(f"  视频 → {vid_path}")


def infer_image(image_path: str, output_dir: str, model, processor,
                labels: list[str], device):
    try:
        import torch
        from PIL import Image
    except ImportError:
        print("[clip] 请先安装: pip install torch pillow")
        sys.exit(1)

    short = [LABEL_SHORT.get(l, l) for l in labels]

    img = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        text_inputs = processor(text=labels, return_tensors="pt", padding=True).to(device)
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        img_inputs = processor(images=img, return_tensors="pt").to(device)
        img_feat = model.get_image_features(**img_inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sims = (img_feat @ text_feats.T)[0]
        probs = sims.softmax(dim=0).cpu().tolist()

    pred_idx = probs.index(max(probs))
    print(f"\n[clip] {os.path.basename(image_path)}")
    for lb, p in sorted(zip(short, probs), key=lambda x: -x[1]):
        marker = " ◀" if lb == short[pred_idx] else ""
        print(f"  {lb:<20} {p:.1%}{marker}")

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(output_dir, f"{stem}_clip.csv")
    with open(csv_path, "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=["label", "probability"])
        dw.writeheader()
        for lb, p in zip(short, probs):
            dw.writerow({"label": lb, "probability": round(p, 4)})
    print(f"  → {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="视频文件或目录")
    parser.add_argument("--image", default=None, help="图片文件或目录")
    parser.add_argument("--output_dir", default="results/vision/clip")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32",
                        help="CLIP 模型（默认 clip-vit-base-patch32，更准: clip-vit-large-patch14）")
    parser.add_argument("--fps", type=int, default=5,
                        help="采样帧率（默认 5，越高越慢）")
    parser.add_argument("--no_video", action="store_true",
                        help="不输出标注视频，只输出 CSV")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="自定义标签（英文描述），如: 'a dog sleeping' 'a dog playing'")
    args = parser.parse_args()

    if not args.video and not args.image:
        parser.error("请指定 --video 或 --image")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] 设备: {device}")

    labels = args.labels if args.labels else DEFAULT_LABELS
    if args.labels:
        global LABEL_SHORT
        LABEL_SHORT = {l: l for l in labels}
    print(f"[clip] 识别类别: {[LABEL_SHORT.get(l, l) for l in labels]}")

    model, processor = load_model(args.model)
    model = model.to(device)

    import glob

    if args.image:
        img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        if os.path.isdir(args.image):
            files = []
            for ext in img_exts:
                files += glob.glob(os.path.join(args.image, ext))
            if not files:
                print(f"[clip] 目录下没有图片: {args.image}")
                sys.exit(1)
            for f in sorted(files):
                infer_image(f, args.output_dir, model, processor, labels, device)
        else:
            infer_image(args.image, args.output_dir, model, processor, labels, device)

    if args.video:
        video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        if os.path.isdir(args.video):
            files = []
            for ext in video_exts:
                files += glob.glob(os.path.join(args.video, ext))
            if not files:
                print(f"[clip] 目录下没有视频: {args.video}")
                sys.exit(1)
            for f in sorted(files):
                infer_video(f, args.output_dir, model, processor,
                            labels, args.fps, device, not args.no_video)
        else:
            infer_video(args.video, args.output_dir, model, processor,
                        labels, args.fps, device, not args.no_video)


if __name__ == "__main__":
    main()
