"""
对视频文件进行行为推理，输出每帧/每窗口的预测标签。

用法:
  # MLP 单帧推理
  python src/infer.py --video data/raw_vision/videos/dog01.mp4 --model results/best_mlp.pt --mode mlp

  # LSTM 序列推理
  python src/infer.py --video data/raw_vision/videos/dog01.mp4 --model results/best_lstm.pt --mode lstm --window 25
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.dataset import kp_to_vector
from vision.models.mlp import PoseMLP
from vision.models.lstm_classifier import PoseLSTM
from vision.pose import extract_pose


def infer(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    classes = cfg["labels"]
    n_kp = cfg["pose"].get("n_keypoints", 17)
    input_dim = n_kp * 2
    window_size = args.window if args.mode == "lstm" else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    if args.mode == "mlp":
        model = PoseMLP(input_dim, len(classes)).to(device)
    else:
        model = PoseLSTM(input_dim, len(classes)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"[infer] 模型: {args.model}  模式: {args.mode}  设备: {device}")

    # 提取姿态（存临时文件）
    tmp_pose = args.video.replace(".mp4", "_pose_tmp.json").replace(".avi", "_pose_tmp.json")
    if not os.path.exists(tmp_pose):
        print("[infer] 提取关键点...")
        extract_pose(args.video, tmp_pose, args.pose_model, args.fps)

    with open(tmp_pose) as f:
        pose_data = json.load(f)
    frames = pose_data["frames"]
    fps = pose_data["fps"]

    # 提取特征向量
    vecs = []
    times = []
    for frm in frames:
        dogs = frm.get("dogs", [])
        kp = max(dogs, key=lambda d: d.get("confidence", 0))["keypoints"] if dogs else []
        vecs.append(kp_to_vector(kp, n_kp))
        times.append(frm["time_s"])
    vecs = np.array(vecs, dtype=np.float32)

    results = []

    if args.mode == "mlp":
        X = torch.from_numpy(vecs).to(device)
        with torch.no_grad():
            logits = model(X)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for i, (t, p, prob) in enumerate(zip(times, preds, probs)):
            results.append({
                "frame": i, "time_s": round(t, 3),
                "prediction": classes[p],
                "confidence": round(float(prob[p]), 4),
            })

    else:  # lstm
        stride = max(1, window_size // 5)
        for start in range(0, len(vecs) - window_size + 1, stride):
            window = vecs[start:start + window_size]
            x = torch.from_numpy(window).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(x)
                prob = F.softmax(logit, dim=1).cpu().numpy()[0]
            p = prob.argmax()
            mid_t = times[start + window_size // 2]
            results.append({
                "window_start_s": round(times[start], 3),
                "window_end_s": round(times[start + window_size - 1], 3),
                "time_s": round(mid_t, 3),
                "prediction": classes[p],
                "confidence": round(float(prob[p]), 4),
            })

    # 输出
    import pandas as pd
    from collections import Counter
    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_path = os.path.join(args.output_dir, f"{stem}_{args.mode}_predictions.csv")
    df.to_csv(out_path, index=False)

    counts = Counter(df["prediction"])
    total = len(df)
    summary = "  ".join(f"{k}:{v/total:.0%}" for k, v in counts.most_common())
    print(f"[infer] {os.path.basename(args.video)}: {total} 条  [{summary}]")
    print(f"[infer] 结果保存至 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--model", required=True, help="模型权重 .pt 路径")
    parser.add_argument("--mode", default="mlp", choices=["mlp", "lstm"])
    parser.add_argument("--window", type=int, default=25,
                        help="LSTM 序列窗口帧数（默认 25）")
    parser.add_argument("--fps", type=int, default=25, help="提帧采样率")
    parser.add_argument("--pose_model", default="yolov8n-pose.pt")
    parser.add_argument("--output_dir", default="results/vision/infer")
    parser.add_argument("--config", default="configs/vision.yaml")
    infer(parser.parse_args())
