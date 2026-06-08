"""
DL 训练入口
用法:
  python src/dl/train.py --hz 50 --model cnn_lstm --processed_dir data/processed_a
  python src/dl/train.py --hz 50 --model filternet_m2m --processed_dir data/processed_a
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../data"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_all_splits

M2M_MODELS = {"filternet_m2m"}


def load_model(model_name, n_channels, window_size, n_classes, cfg):
    if model_name == "cnn":
        from models.cnn import CNN
        return CNN(n_channels, window_size, n_classes, cfg["cnn"])
    elif model_name == "collar_cnn":
        from models.collar_cnn import CollarCNN
        return CollarCNN(n_channels, window_size, n_classes, cfg["collar_cnn"])
    elif model_name == "cnn_lstm":
        from models.cnn_lstm import CNNLSTM
        return CNNLSTM(n_channels, window_size, n_classes, cfg["cnn_lstm"])
    elif model_name == "transformer":
        from models.transformer import TransformerClassifier
        return TransformerClassifier(n_channels, window_size, n_classes, cfg["transformer"])
    elif model_name == "filternet":
        from models.filternet import FilterNet
        return FilterNet(n_channels, window_size, n_classes, cfg["filternet"])
    elif model_name == "filternet_m2m":
        from models.filternet_m2m import FilterNetM2M
        return FilterNetM2M(n_channels, window_size, n_classes, cfg["filternet_m2m"])
    else:
        raise ValueError(f"未知模型: {model_name}")


def make_loader(X, y, y_seq, batch_size, shuffle, m2m=False):
    X_t = torch.from_numpy(X).float().permute(0, 2, 1)   # (N, C, T)
    y_t = torch.from_numpy(y).long()
    if m2m and y_seq is not None:
        # y_seq: (N, T) — 逐帧标签，-1 表示未映射帧（训练时忽略）
        ys_t = torch.from_numpy(y_seq).long()
        ds = TensorDataset(X_t, y_t, ys_t)
    else:
        ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def m2m_loss(logits, y_seq, criterion):
    """
    logits : (B, n_classes, T)
    y_seq  : (B, T)  — 逐帧标签，-1 的帧忽略
    """
    B, C, T = logits.shape
    logits_flat = logits.permute(0, 2, 1).reshape(-1, C)   # (B*T, C)
    labels_flat = y_seq.reshape(-1)                         # (B*T,)
    mask = labels_flat >= 0
    return criterion(logits_flat[mask], labels_flat[mask])


def m2m_predict(logits):
    """(B, n_classes, T) → (B,) 多数投票"""
    per_frame = logits.argmax(dim=1)        # (B, T)
    preds = []
    for row in per_frame:
        vals, counts = row.unique(return_counts=True)
        preds.append(vals[counts.argmax()].item())
    return torch.tensor(preds, device=logits.device)


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m2m = args.model in M2M_MODELS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "many-to-many" if m2m else "many-to-one"
    print(f"\n[dl/train] hz={args.hz}, model={args.model} ({mode_str}), device={device}")
    print(f"[dl/train] processed_dir={args.processed_dir}")

    (X_tr, y_tr, y_seq_tr), (X_val, y_val, y_seq_val), (X_te, y_te, y_seq_te), meta = \
        load_all_splits(args.hz, args.processed_dir)

    classes = eval(meta["classes"]) if isinstance(meta["classes"], str) else meta["classes"]
    n_channels  = X_tr.shape[2]
    window_size = X_tr.shape[1]
    n_classes   = len(classes)
    print(f"[dl/train] 数据形状: {X_tr.shape}, 类别数: {n_classes}")

    train_loader = make_loader(X_tr, y_tr, y_seq_tr, cfg["batch_size"], shuffle=True,  m2m=m2m)
    val_loader   = make_loader(X_val, y_val, y_seq_val, cfg["batch_size"], shuffle=False, m2m=m2m)
    test_loader  = make_loader(X_te,  y_te,  y_seq_te,  cfg["batch_size"], shuffle=False, m2m=m2m)

    model     = load_model(args.model, n_channels, window_size, n_classes, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    from tqdm import tqdm

    best_val_acc = 0.0
    patience     = cfg["early_stopping_patience"]
    no_improve   = 0

    dataset_tag     = os.path.basename(args.processed_dir.rstrip("/"))
    out_dir         = os.path.join(args.results_dir, dataset_tag, f"{args.hz}hz")
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, f"dl_{args.model}_best.pt")

    pbar = tqdm(range(1, cfg["epochs"] + 1), desc="训练", unit="epoch")
    for epoch in pbar:
        model.train()
        total_loss = 0
        for batch in train_loader:
            xb = batch[0].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if m2m:
                ys = batch[2].to(device)
                loss = m2m_loss(logits, ys, criterion)
            else:
                yb = batch[1].to(device)
                loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证：无论 m2m 还是 m2o，都用窗口级别 accuracy 评估
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                xb, yb = batch[0].to(device), batch[1].to(device)
                logits = model(xb)
                preds = m2m_predict(logits) if m2m else logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total   += len(yb)
        val_acc = correct / total
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            no_improve = 0
        else:
            no_improve += 1

        avg_loss = total_loss / len(train_loader)
        lr_now   = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{avg_loss:.4f}", val_acc=f"{val_acc:.4f}",
                         best=f"{best_val_acc:.4f}", lr=f"{lr_now:.0e}")

        if no_improve >= patience:
            tqdm.write(f"  Early stopping at epoch {epoch}, best val_acc={best_val_acc:.4f}")
            break

    # 测试
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            xb, yb = batch[0].to(device), batch[1]
            logits = model(xb)
            preds  = m2m_predict(logits).cpu() if m2m else logits.argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_true.extend(yb.numpy())

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
    print(f"\n[dl/train] 测试集结果 ({mode_str}, best val model):")
    print(f"  Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")
    present_labels = sorted(set(all_true) | set(all_preds))
    present_names  = [classes[i] for i in present_labels]
    print(classification_report(all_true, all_preds, labels=present_labels,
                                target_names=present_names, zero_division=0))

    result = {"hz": args.hz, "model": args.model, "accuracy": acc, "macro_f1": f1}
    with open(os.path.join(out_dir, f"dl_{args.model}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[dl/train] 结果保存至 {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=int, required=True, choices=[5, 10, 25, 50])
    parser.add_argument("--model", default="cnn_lstm",
                        choices=["cnn", "collar_cnn", "cnn_lstm", "transformer",
                                 "filternet", "filternet_m2m"])
    parser.add_argument("--config", default="configs/dl.yaml")
    parser.add_argument("--processed_dir", default="data/processed_a")
    parser.add_argument("--results_dir", default="results")
    main(parser.parse_args())
