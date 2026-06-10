"""
训练关键点行为分类器（MLP 或 LSTM）。

用法:
  # 从 Roboflow 转换后的 npz 训练（推荐入门）
  python src/train.py --mode mlp --npz_dir data/processed/roboflow

  # 从自采视频 + 标注训练
  python src/train.py --mode mlp --pose_dir data/processed_vision/poses --annot_dir data/annotations_vision

  # 序列 LSTM
  python src/train.py --mode lstm --npz_dir data/processed/roboflow --window 25 --stride 5
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vision.dataset import load_all_sessions, DogBehaviorDataset
from vision.models.mlp import PoseMLP
from vision.models.lstm_classifier import PoseLSTM


def load_npz_split(npz_dir: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    path = os.path.join(npz_dir, f"{split}.npz")
    if not os.path.exists(path):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64), []
    d = np.load(path, allow_pickle=True)
    classes = d["classes"].tolist()
    return d["X"], d["y"], classes


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    window_size = args.window if args.mode == "lstm" else 1
    stride = args.stride if args.mode == "lstm" else 1

    # ── 数据来源：npz（Roboflow转换）或 视频+标注 ─────────────────────────────
    if args.npz_dir:
        print(f"[train] 从 npz 加载: {args.npz_dir}")
        X_tr, y_tr, classes = load_npz_split(args.npz_dir, "train")
        X_val, y_val, _     = load_npz_split(args.npz_dir, "val")
        X_te,  y_te,  _     = load_npz_split(args.npz_dir, "test")
        n_kp = int(np.load(os.path.join(args.npz_dir, "train.npz"))["n_kp"])
        # LSTM 模式：图片数据没有时序，跳过序列窗口
        if args.mode == "lstm" and window_size > 1:
            print("[train] ⚠️  npz 是单帧图片数据，LSTM 窗口设为 1（等同 MLP）")
            window_size = 1
    else:
        classes = cfg["labels"]
        n_kp = cfg["pose"].get("n_keypoints", 17)

        def load_split(split):
            return load_all_sessions(
                args.pose_dir, args.annot_dir, classes,
                window_size=window_size, stride=stride,
                split=split,
                train_ratio=cfg.get("train_ratio", 0.7),
                val_ratio=cfg.get("val_ratio", 0.15),
                seed=cfg.get("seed", 42),
                n_kp=n_kp,
            )

        X_tr, y_tr = load_split("train")
        X_val, y_val = load_split("val")
        X_te, y_te = load_split("test")

    input_dim = n_kp * 2
    print(f"[train] 模式: {args.mode}  窗口: {window_size}帧  步长: {stride}帧")
    print(f"[train] 类别 ({len(classes)}): {classes}")
    print(f"[train] train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

    if len(X_tr) == 0:
        print("[train] 没有训练数据，请先运行 src/video/pose.py 并准备标注文件。")
        return

    train_ds = DogBehaviorDataset(X_tr, y_tr, classes)
    val_ds = DogBehaviorDataset(X_val, y_val, classes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 使用设备: {device}")

    if args.mode == "mlp":
        model = PoseMLP(input_dim, len(classes)).to(device)
    else:
        model = PoseLSTM(input_dim, len(classes)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 类别权重：解决不平衡问题（少数类权重更高）
    counts = np.bincount(y_tr.astype(int), minlength=len(classes))
    weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
    weights = weights / weights.sum() * len(classes)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_f1 = 0.0
    patience_count = 0
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"best_{args.mode}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())
        val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        scheduler.step(1 - val_f1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  loss={total_loss/len(train_loader):.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[train] Early stopping at epoch {epoch}")
                break

    # 测试集评估
    if len(X_te) > 0:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        te_ds = DogBehaviorDataset(X_te, y_te, classes)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size)
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in te_loader:
                preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())
        print(f"\n[train] 测试集结果:")
        print(f"  Accuracy: {accuracy_score(all_true, all_preds):.4f}")
        print(f"  Macro F1: {f1_score(all_true, all_preds, average='macro', zero_division=0):.4f}")
        print(classification_report(all_true, all_preds,
                                    target_names=classes, zero_division=0))

    print(f"[train] 模型保存至 {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mlp", choices=["mlp", "lstm"],
                        help="mlp=单帧, lstm=序列（默认 mlp）")
    parser.add_argument("--npz_dir", default="",
                        help="Roboflow 转换后的 npz 目录（优先使用）")
    parser.add_argument("--pose_dir", default="data/processed_vision/poses",
                        help="姿态 JSON 目录（自采视频用）")
    parser.add_argument("--annot_dir", default="data/annotations_vision",
                        help="标注 CSV 目录（自采视频用）")
    parser.add_argument("--output_dir", default="results/vision")
    parser.add_argument("--config", default="configs/vision.yaml")
    parser.add_argument("--window", type=int, default=25,
                        help="序列窗口帧数（LSTM 用，默认 25 = 1秒@25fps）")
    parser.add_argument("--stride", type=int, default=5, help="滑窗步长（默认 5）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience（默认 15）")
    train(parser.parse_args())
