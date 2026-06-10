"""
用多模态大模型 API 做狗行为识别（火山引擎豆包）。

优点：理解能力强，无需训练，支持自然语言描述行为细节。
缺点：需要 API Key，有费用，速度比 CLIP 慢。

用法:
  export ARK_API_KEY="your_key"
  python src/vision/llm_infer.py --image 你的图片.jpg
  python src/vision/llm_infer.py --video 你的视频.mp4
  python src/vision/llm_infer.py --image datasets/dog-pose/images/val/

获取火山引擎 API Key:
  https://console.volcengine.com/ark → 开通豆包大模型 → 创建 API Key
"""

import argparse
import base64
import csv
import io
import json
import os
import sys
from collections import Counter


PROMPT = """You are an expert dog behavior analyst.
Classify the dog's behavior in the image into exactly one of these categories:
- Lying: dog is lying down, resting on its side or chest
- Sitting: dog is sitting with hindquarters on ground
- Standing: dog is standing still on all four legs
- Walking: dog is walking at a relaxed pace
- Trotting: dog is trotting or running at faster speed
- Sniffing: dog is sniffing the ground or an object

Reply with JSON only, no explanation:
{"behavior": "<category>", "confidence": <0.0-1.0>, "description": "<one sentence>"}"""


def build_client(api_key: str, base_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("[llm] 请先安装: pip install openai")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=base_url)


def image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    ext  = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "bmp": "bmp", "webp": "webp"}.get(ext, "jpeg")
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


def frame_to_b64(frame_bgr) -> str:
    from PIL import Image
    img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR→RGB
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def call_api(client, model: str, image_b64: str) -> dict:
    response = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text",  "text": PROMPT},
                {"type": "input_image", "image_url": image_b64},
            ],
        }],
        max_output_tokens=200,
    )
    # 提取文本
    raw = ""
    for item in response.output:
        if item.type == "message":
            for c in item.content:
                if c.type == "output_text":
                    raw = c.text.strip()
                    break
    # 解析 JSON
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            return json.loads(raw[s:e])
        except json.JSONDecodeError:
            pass
    return {"behavior": raw or "Unknown", "confidence": 0.0, "description": raw}


def infer_image(image_path: str, output_dir: str, client, model: str):
    b64    = image_to_b64(image_path)
    result = call_api(client, model, b64)

    behavior    = result.get("behavior", "Unknown")
    confidence  = result.get("confidence", 0.0)
    description = result.get("description", "")

    print(f"\n[llm] {os.path.basename(image_path)}")
    print(f"  行为: {behavior}  (置信度 {confidence:.0%})")
    print(f"  描述: {description}")

    os.makedirs(output_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(output_dir, f"{stem}_llm.csv")
    with open(csv_path, "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=["behavior", "confidence", "description"])
        dw.writeheader()
        dw.writerow({"behavior": behavior, "confidence": round(confidence, 4),
                     "description": description})
    print(f"  → {csv_path}")
    return result


def infer_video(video_path: str, output_dir: str, client, model: str, fps: int):
    try:
        import cv2
    except ImportError:
        print("[llm] 请先安装: pip install opencv-python pillow")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[llm] 无法打开: {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step    = max(1, round(src_fps / fps))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(output_dir, f"{stem}_llm.csv")

    records  = []
    read_idx = 0
    out_idx  = 0

    print(f"[llm] {os.path.basename(video_path)}  采样fps={fps}  预计帧数≈{total//step}")

    with open(csv_path, "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=["frame", "time_s", "behavior", "confidence", "description"])
        dw.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if read_idx % step == 0:
                b64 = frame_to_b64(frame)
                try:
                    result = call_api(client, model, b64)
                except Exception as e:
                    print(f"  [warn] 帧 {out_idx} 失败: {e}")
                    result = {"behavior": "Unknown", "confidence": 0.0, "description": ""}

                row = {
                    "frame":       out_idx,
                    "time_s":      round(out_idx / fps, 3),
                    "behavior":    result.get("behavior", "Unknown"),
                    "confidence":  round(result.get("confidence", 0.0), 4),
                    "description": result.get("description", ""),
                }
                dw.writerow(row)
                records.append(row)
                print(f"  帧{out_idx:4d}  {row['behavior']:<10} ({row['confidence']:.0%})  {row['description'][:50]}")
                out_idx += 1

            read_idx += 1

    cap.release()

    counts  = Counter(r["behavior"] for r in records)
    total_r = len(records)
    summary = "  ".join(f"{k}:{v/total_r:.0%}" for k, v in counts.most_common())
    print(f"\n[llm] 完成  {total_r} 帧  [{summary}]")
    print(f"  CSV → {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      default=None, help="图片文件或目录")
    parser.add_argument("--video",      default=None, help="视频文件或目录")
    parser.add_argument("--output_dir", default="results/vision/llm")
    parser.add_argument("--api_key",    default=None,
                        help="API Key（也可设环境变量 ARK_API_KEY）")
    parser.add_argument("--base_url",   default="https://ark.cn-beijing.volces.com/api/v3",
                        help="API 地址（默认火山引擎豆包）")
    parser.add_argument("--model",      default="doubao-seed-2-0-lite-260215",
                        help="模型名称（默认 doubao-seed-2-0-lite-260215）")
    parser.add_argument("--fps",        type=int, default=1,
                        help="视频采样帧率（默认 1fps，省钱）")
    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("请指定 --image 或 --video")

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        print("[llm] 请提供 API Key: --api_key YOUR_KEY 或 export ARK_API_KEY=YOUR_KEY")
        sys.exit(1)

    client = build_client(api_key, args.base_url)
    print(f"[llm] 模型: {args.model}")

    import glob

    if args.image:
        img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        if os.path.isdir(args.image):
            files = []
            for ext in img_exts:
                files += glob.glob(os.path.join(args.image, ext))
            if not files:
                print(f"[llm] 目录下没有图片: {args.image}")
                sys.exit(1)
            for f in sorted(files):
                infer_image(f, args.output_dir, client, args.model)
        else:
            infer_image(args.image, args.output_dir, client, args.model)

    if args.video:
        video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        if os.path.isdir(args.video):
            files = []
            for ext in video_exts:
                files += glob.glob(os.path.join(args.video, ext))
            if not files:
                print(f"[llm] 目录下没有视频: {args.video}")
                sys.exit(1)
            for f in sorted(files):
                infer_video(f, args.output_dir, client, args.model, args.fps)
        else:
            infer_video(args.video, args.output_dir, client, args.model, args.fps)


if __name__ == "__main__":
    main()
