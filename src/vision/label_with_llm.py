"""
用 LLM 批量给 dog-pose 数据集图片打行为标签。

用法:
  export ARK_API_KEY="your_key"
  python src/vision/label_with_llm.py --split train
  python src/vision/label_with_llm.py --split val
  python src/vision/label_with_llm.py --split train --workers 8  # 并发加速

输出:
  data/processed/llm_labels/train_labels.csv
  data/processed/llm_labels/val_labels.csv
  格式: stem,behavior,confidence,description
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROMPT = """You are an expert dog behavior analyst.
Classify the dog's behavior in the image into exactly one of these categories:
- Lying: dog is lying down, resting on its side or chest
- Sitting: dog is sitting with hindquarters on ground
- Standing: dog is standing still on all four legs
- Walking: dog is walking at a relaxed pace
- Trotting: dog is trotting or running at faster speed
- Sniffing: dog is sniffing the ground or an object

Reply with JSON only:
{"behavior": "<category>", "confidence": <0.0-1.0>}"""

VALID_BEHAVIORS = {"Lying", "Sitting", "Standing", "Walking", "Trotting", "Sniffing"}


def build_client(api_key: str, base_url: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def label_image(client, model: str, image_path: str, retries: int = 3) -> dict:
    with open(image_path, "rb") as f:
        b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": [
                    {"type": "input_text",  "text": PROMPT},
                    {"type": "input_image", "image_url": b64},
                ]}],
                max_output_tokens=200,
                extra_body={"thinking": {"type": "disabled"}},
            )
            # 收集所有文本输出（兼容 reasoning/message 两种类型）
            raw = ""
            for item in response.output:
                if item.type == "message":
                    for c in item.content:
                        if c.type == "output_text":
                            raw = c.text.strip()
                            break
                elif item.type == "reasoning":
                    # reasoning 模型：从 summary 里找 JSON
                    for s_item in getattr(item, "summary", []):
                        text = getattr(s_item, "text", "")
                        if "{" in text and "behavior" in text:
                            raw = text.strip()
                            break
                if raw:
                    break

            s, e = raw.find("{"), raw.rfind("}") + 1
            if s >= 0 and e > s:
                result = json.loads(raw[s:e])
                behavior = result.get("behavior", "").strip()
                # 修正大小写
                for v in VALID_BEHAVIORS:
                    if behavior.lower() == v.lower():
                        behavior = v
                        break
                if behavior in VALID_BEHAVIORS:
                    return {
                        "behavior":   behavior,
                        "confidence": round(float(result.get("confidence", 0.0)), 4),
                    }
        except Exception as e:
            if attempt == 0:
                print(f"\n[API错误] {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"behavior": "Unknown", "confidence": 0.0, "error": str(e)}

    return {"behavior": "Unknown", "confidence": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="datasets/dog-pose")
    parser.add_argument("--split",       default="train", choices=["train", "val"])
    parser.add_argument("--output_dir",  default="data/processed/llm_labels")
    parser.add_argument("--model",       default="doubao-seed-1-6-flash-250828")
    parser.add_argument("--api_key",     default=None)
    parser.add_argument("--base_url",    default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--workers",     type=int, default=4, help="并发线程数（默认 4）")
    parser.add_argument("--resume",      action="store_true", help="跳过已标注的图片，断点续跑")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ARK_API_KEY")
    if not api_key:
        print("请设置 ARK_API_KEY 环境变量或传入 --api_key")
        sys.exit(1)

    images_dir = Path(args.dataset_dir) / "images" / args.split
    if not images_dir.exists():
        print(f"找不到图片目录: {images_dir}")
        sys.exit(1)

    img_files = sorted(images_dir.glob("*.jpg")) + \
                sorted(images_dir.glob("*.jpeg")) + \
                sorted(images_dir.glob("*.png"))
    print(f"[label] {args.split}: 共 {len(img_files)} 张图片")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{args.split}_labels.csv")

    # 读取已有标注（断点续跑）
    done = {}
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                done[row["stem"]] = row
        print(f"[label] 已有标注 {len(done)} 条，跳过继续")

    todo = [f for f in img_files if f.stem not in done]
    print(f"[label] 待标注 {len(todo)} 张  workers={args.workers}")
    if not todo:
        print("[label] 全部已完成！")
        return

    client = build_client(api_key, args.base_url)

    results = dict(done)
    completed = len(done)
    failed = 0

    # 写文件（追加模式，支持断点续跑）
    write_header = not (args.resume and os.path.exists(csv_path))
    f_out = open(csv_path, "a" if args.resume else "w", newline="")
    dw = csv.DictWriter(f_out, fieldnames=["stem", "behavior", "confidence"])
    if write_header:
        dw.writeheader()

    def process(img_path):
        result = label_image(client, args.model, str(img_path))
        return img_path.stem, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, f): f for f in todo}
        try:
            from tqdm import tqdm
            bar = tqdm(as_completed(futures), total=len(todo), desc="标注进度",
                       unit="张", dynamic_ncols=True)
        except ImportError:
            bar = as_completed(futures)

        for future in bar:
            stem, result = future.result()
            behavior   = result.get("behavior", "Unknown")
            confidence = result.get("confidence", 0.0)

            dw.writerow({"stem": stem, "behavior": behavior, "confidence": confidence})
            f_out.flush()
            results[stem] = {"stem": stem, "behavior": behavior, "confidence": confidence}

            completed += 1
            if behavior == "Unknown":
                failed += 1
            if hasattr(bar, "set_postfix"):
                bar.set_postfix({"失败": failed, "最新": f"{behavior}"})

    f_out.close()

    # 统计
    from collections import Counter
    counts = Counter(v["behavior"] for v in results.values())
    print(f"\n[label] 完成！保存至 {csv_path}")
    print(f"[label] 标注分布:")
    for behavior, count in counts.most_common():
        print(f"  {behavior:<12} {count:4d}  ({count/len(results):.1%})")


if __name__ == "__main__":
    main()
