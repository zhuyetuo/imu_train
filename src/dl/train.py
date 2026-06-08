"""
DL 训练入口
用法: python src/dl/train.py --hz 50 --model cnn_lstm
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))

import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_all_splits


def load_model(model_name: str, n_channels: int, window_size: int, n_classes: int, cfg: dict):
    if model_name == "cnn":
        from models.cnn import CNN
        return CNN(n_channels, window_size, n_classes, cfg["cnn"])
    elif model_name == "cnn_lstm":
        from models.cnn_lstm import CNNLSTM
        return CNNLSTM(n_channels, window_size, n_classes, cfg["cnn_lstm"])
    elif model_name == "transformer":
        from models.transformer import TransformerClassifier
        return TransformerClassifier(n_channels, window_size, n_classes, cfg["transformer"])
    else:
        raise ValueError(f"未知模型: {model_name}")


def make_loader(X, y, batch_size, shuffle):
    X_t = torch.from_numpy(X).float()          # (N, T, C)
    X_t = X_t.permute(0, 2, 1)                 # → (N, C, T) 适配 Conv1d
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[dl/train] hz={args.hz}, model={args.model}, device={device}")

    (X_tr, y_tr), (X_val, y_val), (X_te, y_te), meta = load_all_splits(args.hz)
    classes = eval(meta["classes"]) if isinstance(meta["classes"], str) else meta["classes"]
    n_channels = X_tr.shape[2]
    window_size = X_tr.shape[1]
    n_classes = len(classes)
    print(f"[dl/train] 数据形状: {X_tr.shape}, 类别数: {n_classes}")

    train_loader = make_loader(X_tr, y_tr, cfg["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, cfg["batch_size"], shuffle=False)
    test_loader = make_loader(X_te, y_te, cfg["batch_size"], shuffle=False)

    model = load_model(args.model, n_channels, window_size, n_classes, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    from tqdm import tqdm

    best_val_acc = 0.0
    patience = cfg["early_stopping_patience"]
    no_improve = 0

    out_dir = os.path.join("results", f"{args.hz}hz")
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, f"dl_{args.model}_best.pt")

    pbar = tqdm(range(1, cfg["epochs"] + 1), desc="训练", unit="epoch")
    for epoch in pbar:
        # 训练
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            no_improve = 0
        else:
            no_improve += 1

        avg_loss = total_loss / len(train_loader)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")

        if no_improve >= patience:
            tqdm.write(f"  Early stopping at epoch {epoch}, best val_acc={best_val_acc:.4f}")
            break

    # 测试
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            all_preds.extend(model(xb).argmax(dim=1).cpu().numpy())
            all_true.extend(yb.numpy())

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average="macro")
    print(f"\n[dl/train] 测试集结果 (best val model):")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print(classification_report(all_true, all_preds, target_names=classes))

    result = {"hz": args.hz, "model": args.model, "accuracy": acc, "macro_f1": f1}
    with open(os.path.join(out_dir, f"dl_{args.model}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[dl/train] 结果保存至 {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=int, required=True, choices=[5, 10, 25, 50])
    parser.add_argument("--model", default="cnn_lstm", choices=["cnn", "cnn_lstm", "transformer"])
    parser.add_argument("--config", default="configs/dl.yaml")
    main(parser.parse_args())
