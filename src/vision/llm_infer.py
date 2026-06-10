"""
用多模态大模型 API 做狗行为识别（火山引擎豆包 / OpenAI 兼容接口）。

优点：理解能力强，无需训练，支持自然语言描述行为细节。
缺点：需要 API Key，有费用，速度比 CLIP 慢。

用法:
  # 单张图片
  python src/vision/llm_infer.py --image 你的图片.jpg --api_key YOUR_KEY

  # 视频（每秒采样）
  python src/vision/llm_infer.py --video 你的视频.mp4 --api_key YOUR_KEY

  # 把 key 写到环境变量，不用每次传
  export ARK_API_KEY="your_key"
  python src/vision/llm_infer.py --image 你的图片.jpg

获取火山引擎 API Key:
  https://console.volcengine.com/ark  → 开通豆包大模型服务 → 创建 API Key
"""

import argparse
import base64
import csv
import json
import os
import sys
from collections import Counter


DEFAULT_LABELS = ["Lying", "Sitting", "Standing", "Walking", "Trotting", "Sniffing"]

SYSTEM_PROMPT = """You are an expert dog behavior analyst.
Classify the dog's behavior in the image into exactly one of these categories:
- Lying: dog is lying down, resting on its side or chest
- Sitting: dog is sitting with hindquarters on ground
- Standing: dog is standing still on all four legs
- Walking: dog is walking at a relaxed pace
- Trotting: dog is trotting or running at faster speed
- Sniffing: dog is sniffing the ground or an object

Reply with JSON only, no explanation:
{"behavior": "<category>", "confidence": <0.0-1.0>, "description": "<one sentence>"}"""


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "bmp": "bmp", "webp": "webp"}.get(ext, "jpeg")
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


def call_api(client, model: str, image_b64: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_b64}},
                {"type": "text", "text": "What behavior is this dog doing?"},
            ]},
        ],
        max_tokens=200,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    # 提取 JSON（有时模型会在 JSON 前后加文字）
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(raw[start:end])
    return {"behavior": raw, "confidence": 1.0, "description": raw}


def infer_image(image_path: str, output_dir: str, client, model: str, verbose: bool = True):
    image_b64 = image_to_base64(image_path)
    result = call_api(client, model, image_b64)

    behavior    = result.get("behavior", "Unknown")
    confidence  = result.get("confidence", 0.0)
    description = result.get("description", "")

    if verbose:
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
    if verbose:
        print(f"  → {csv_path}")

    return result


def infer_video(video_path: str, output_dir: str, client, model: str, fps: int):
    try:
        import cv2
        from PIL import Image
        import io
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

    records    = []
    read_idx   = 0
    out_idx    = 0

    print(f"[llm] {os.path.basename(video_path)}  采样fps={fps}  预计帧数={total//step}")

    with open(csv_path, "w", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=["frame", "time_s", "behavior", "confidence", "description"])
        dw.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if read_idx % step == 0:
                # 转 base64
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

                try:
                    result = call_api(client, model, b64)
                except Exception as e:
                    print(f"  [warn] 帧 {out_idx} API 失败: {e}")
                    result = {"behavior": "Unknown", "confidence": 0.0, "description": ""}

                time_s = round(out_idx / fps, 3)
                row = {
                    "frame": out_idx,
                    "time_s": time_s,
                    "behavior": result.get("behavior", "Unknown"),
                    "confidence": round(result.get("confidence", 0.0), 4),
                    "description": result.get("description", ""),
                }
                dw.writerow(row)
                records.append(row)

                if out_idx % 10 == 0:
                    print(f"  帧 {out_idx}  {row['behavior']} ({row['confidence']:.0%})  {row['description'][:40]}")

                out_idx += 1

            read_idx += 1

    cap.release()

    counts  = Counter(r["behavior"] for r in records)
    total_r = len(records)
    summary = "  ".join(f"{k}:{v/total_r:.0%}" for k, v in counts.most_common())
    print(f"\n[llm] 完成  {total_r} 帧  [{summary}]")
    print(f"  CSV → {csv_path}")


def build_client(api_key: str, base_url: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("[llm] 请先安装: pip install openai")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=base_url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="图片文件或目录")
    parser.add_argument("--video", default=None, help="视频文件或目录")
    parser.add_argument("--output_dir", default="results/vision/llm")
    parser.add_argument("--api_key", default=None,
                        help="API Key（也可设环境变量 ARK_API_KEY）")
    parser.add_argument("--base_url", default="https://ark.cn-beijing.volces.com/api/v3",
                        help="API 地址（默认火山引擎豆包）")
    parser.add_argument("--model", default="doubao-vision-pro-32k",
                        help="模型名称（默认 doubao-vision-pro-32k）")
    parser.add_argument("--fps", type=int, default=1,
                        help="视频采样帧率（默认 1，API 有费用不宜过高）")
    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("请指定 --image 或 --video")

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        print("[llm] 请提供 API Key: --api_key YOUR_KEY 或 export ARK_API_KEY=YOUR_KEY")
        print("[llm] 获取地址: https://console.volcengine.com/ark")
        sys.exit(1)

    client = build_client(api_key, args.base_url)
    print(f"[llm] 模型: {args.model}")
    print(f"[llm] API:  {args.base_url}")

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
