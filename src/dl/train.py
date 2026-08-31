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
import subprocess
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import load_all_splits
from remap_utils import load_remap_yaml, apply_remap, apply_remap_seq

M2M_MODELS = {"filternet_m2m"}


def _resolve_processed_dir(args):
    """--processed_dir没显式传、但传了--date时，跟train_custom.sh用完全
    相同的命名规则算出processed_dir（data/processed_<DATE>_<TAG>，TAG
    默认missing_<MISSING_STRATEGY>，见train_custom.sh里TAG的自动派生
    逻辑），两边不用再手动对齐目录名。目录下对应hz的train.npz已经存在
    就直接用；不存在就自动调train_custom.sh --skip_ml去生成（只做
    预处理，不训练ML模型/不生成合成数据，DL不需要那些），不用每次先
    手动跑一遍train_custom.sh再跑dl/train.py。
    """
    if args.processed_dir:
        return args.processed_dir
    if not args.date:
        raise SystemExit("[dl/train] 必须传 --processed_dir，或者传 --date"
                          "（配合--missing_strategy/--tag等，自动定位/生成预处理数据）")

    tag = args.tag or f"missing_{args.missing_strategy}"
    processed_dir = f"data/processed_{args.date}_{tag}"
    marker = os.path.join(processed_dir, f"{args.hz}hz", "train.npz")
    if os.path.exists(marker):
        print(f"[dl/train] 找到已有预处理数据: {marker}")
        return processed_dir

    print(f"[dl/train] {marker} 不存在，先调用 train_custom.sh --skip_ml 生成预处理数据...")
    cmd = ["bash", "train_custom.sh", "--date", args.date, "--hz", str(args.hz),
           "--missing_strategy", args.missing_strategy, "--skip_ml"]
    if args.tag:
        cmd += ["--tag", args.tag]
    if args.source_hz:
        cmd += ["--source_hz", str(args.source_hz)]
    for ed in args.extra_date:
        cmd += ["--extra_date", ed]
    if args.window_s:
        cmd += ["--window_s", str(args.window_s)]
    if args.stride_s:
        cmd += ["--stride_s", str(args.stride_s)]
    if args.label_mode:
        cmd += ["--label_mode", args.label_mode]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    print(f"[dl/train] 执行: {' '.join(cmd)}  (cwd={repo_root})")
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        raise SystemExit(f"[dl/train] train_custom.sh 预处理失败 (exit={result.returncode})")
    if not os.path.exists(marker):
        raise SystemExit(f"[dl/train] train_custom.sh 跑完了但 {marker} 还是不存在，"
                          f"检查上面train_custom.sh输出的报错")
    return processed_dir


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

    args.processed_dir = _resolve_processed_dir(args)

    m2m = args.model in M2M_MODELS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "many-to-many" if m2m else "many-to-one"
    print(f"\n[dl/train] hz={args.hz}, model={args.model} ({mode_str}), device={device}")
    print(f"[dl/train] processed_dir={args.processed_dir}")

    (X_tr, y_tr, y_seq_tr), (X_val, y_val, y_seq_val), (X_te, y_te, y_seq_te), meta = \
        load_all_splits(args.hz, args.processed_dir)

    classes = eval(meta["classes"]) if isinstance(meta["classes"], str) else meta["classes"]

    # 标签重映射（用于合并类别，如16类原始细分动作→4类行为）——之前这里
    # 完全没有remap支持，只能拿16个原始细分类别直接训练，跟ml/train.py
    # 那边训出来的4类模型不是同一个任务，没法对比。跟ml/train.py共用
    # src/data/remap_utils.py的同一份逻辑，避免两边各写一份、改一边漏改
    # 另一边。
    if args.remap:
        remap_cfg = load_remap_yaml(args.remap)
        print(f"\n[dl/train] 标签重映射: {args.remap}")
        for k, v in remap_cfg.items():
            print(f"  {k} → {v}")
        y_tr,  classes_new, keep_tr  = apply_remap(y_tr,  classes, remap_cfg)
        y_val, _,           keep_val = apply_remap(y_val, classes, remap_cfg)
        y_te,  _,           keep_te  = apply_remap(y_te,  classes, remap_cfg)
        X_tr, X_val, X_te = X_tr[keep_tr], X_val[keep_val], X_te[keep_te]
        # y_seq是逐帧标签，只给many-to-many模型用，但不管是不是m2m这里都
        # 统一处理，保持train/val/te三个y_seq数组形状跟对应X/y一致，避免
        # 后面代码要分m2m/非m2m两套长度检查逻辑
        if y_seq_tr is not None:
            y_seq_tr  = apply_remap_seq(y_seq_tr[keep_tr],   classes, remap_cfg, classes_new)
            y_seq_val = apply_remap_seq(y_seq_val[keep_val], classes, remap_cfg, classes_new)
            y_seq_te  = apply_remap_seq(y_seq_te[keep_te],   classes, remap_cfg, classes_new)
        n_dropped = int((~keep_tr).sum() + (~keep_val).sum() + (~keep_te).sum())
        if n_dropped:
            print(f"[dl/train] 重映射: 丢弃了{n_dropped}个remap配置没覆盖到的样本"
                  f"（比如'未佩戴'这类不该当行为标签训练的类别，不是预期的话检查"
                  f"一下{args.remap}是不是漏配了某个类别）")
        classes = classes_new
        print(f"[dl/train] 重映射后类别: {classes}")

    n_channels  = X_tr.shape[2]
    window_size = X_tr.shape[1]
    n_classes   = len(classes)
    print(f"[dl/train] 数据形状: {X_tr.shape}, 类别数: {n_classes}")

    # 原始窗口数据里如果混了NaN/Inf（这个项目的自采数据管线上游出过好几次
    # 数据质量问题：空文件、缺列、重采样异常……），树模型(ml/train.py)大多
    # 数实现能容忍/绕过，但下面标准化一算均值/方差，NaN会直接传染到整个
    # 数组，"标准化"等于白做，还会把问题伪装成"看起来跑起来了、实际全是
    # 垃圾"——不能让它悄悄过去，训练前必须先挡住并报出来，不然debug起来
    # 很难定位到底是哪一步出的问题。
    for _name, _arr in (("X_tr", X_tr), ("X_val", X_val), ("X_te", X_te)):
        if len(_arr) and not np.isfinite(_arr).all():
            n_bad = int((~np.isfinite(_arr)).sum())
            raise ValueError(
                f"[dl/train] {_name} 里有 {n_bad} 个非法值(NaN/Inf)，不是标准化能解决的问题——"
                f"说明预处理/合并阶段的原始数据本身就有问题，需要往前查是哪批数据、"
                f"哪几个窗口坏的（可以用np.isnan(X).any(axis=(1,2))对着record_id/窗口"
                f"索引定位），不能带着非法值继续训练")

    # 逐通道标准化（z-score，用训练集自己的均值/方差）——树模型(ml/train.py
    # 那边)对特征量纲不敏感，但CNN/LSTM这类梯度下降训练的模型很敏感：
    # 加速度/陀螺仪/姿态角这8个通道数值范围差异很大，不做标准化容易训练
    # 不稳定，实测直接表现为loss变nan、模型坍缩成只会预测样本最多的那个
    # 类别（比如这次val_acc卡在0.4710，正好等于"活动"类占比）。统计量
    # 只用训练集算，不能用val/test算（会造成信息泄漏），val/test套用
    # 训练集的均值方差做同样的变换。
    ch_mean = X_tr.reshape(-1, n_channels).mean(axis=0)
    ch_std  = X_tr.reshape(-1, n_channels).std(axis=0)
    ch_std[ch_std < 1e-6] = 1.0  # 避免除以接近0的标准差（比如某个通道基本不变化）
    X_tr  = (X_tr  - ch_mean) / ch_std
    X_val = (X_val - ch_mean) / ch_std
    if len(X_te) > 0:
        X_te = (X_te - ch_mean) / ch_std
    X_tr, X_val, X_te = X_tr.astype(np.float32), X_val.astype(np.float32), X_te.astype(np.float32)

    # val 为空时（小数据集）用训练集末尾 10% 代替
    if len(X_val) == 0:
        n_fallback = max(1, len(X_tr) // 10)
        X_val, y_val, y_seq_val = X_tr[-n_fallback:], y_tr[-n_fallback:], y_seq_tr[-n_fallback:] if y_seq_tr is not None else None
        X_tr,  y_tr,  y_seq_tr  = X_tr[:-n_fallback],  y_tr[:-n_fallback],  y_seq_tr[:-n_fallback] if y_seq_tr is not None else None
        print(f"[dl/train] val 集为空，从训练集末尾借用 {n_fallback} 个样本作为 val")

    train_loader = make_loader(X_tr, y_tr, y_seq_tr, cfg["batch_size"], shuffle=True,  m2m=m2m)
    val_loader   = make_loader(X_val, y_val, y_seq_val, cfg["batch_size"], shuffle=False, m2m=m2m)
    # test_ratio=0（train_custom.sh纠错循环阶段的默认值，不专门切测试集）
    # 时X_te是空的，最后"测试"这步直接拿去评估会得到一份没有意义的空结果
    # ——改成用验证集代替测试集评估，跟ml/train.py的eval_tag处理方式一致
    has_test = len(X_te) > 0
    eval_tag = "测试集" if has_test else "验证集（无测试集，仅供参考）"
    test_loader = make_loader(X_te, y_te, y_seq_te, cfg["batch_size"], shuffle=False, m2m=m2m) \
        if has_test else make_loader(X_val, y_val, y_seq_val, cfg["batch_size"], shuffle=False, m2m=m2m)

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
    # remap_tag跟ml/train.py保持一致的拼法——用没用remap、用的哪份remap
    # 配置，产出目录要分开，不然16类原始模型和4类重映射模型会被放到
    # 同一个{hz}hz/目录下，dl_{model}.pt文件名又一样，后训练的直接覆盖
    # 先训练的
    remap_tag = f"_{os.path.splitext(os.path.basename(args.remap))[0]}" if args.remap else ""
    out_dir         = os.path.join(args.results_dir, dataset_tag, f"{args.hz}hz{remap_tag}")
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(out_dir, f"dl_{args.model}_best.pt")

    # 推理端(src/infer_csv_scratch.py)要能直接load这个.pt跑实时推理，
    # 但torch.save(model.state_dict(),...)只存了权重，没有：(1)怎么
    # 重建模型结构（model_name+超参cfg）(2)输出的n个logit对应哪个类别、
    # 顺序是什么(3)推理输入要用哪套window_size/hz/gravity_align(4)最关键的——
    # 训练时对输入做了逐通道z-score标准化(ch_mean/ch_std)，推理端如果不用
    # 完全相同的均值方差重新标准化，输入分布跟训练时对不上，模型效果会
    # 明显下降但不会报错，是那种很难定位的隐性bug。所以单独存一份
    # dl_{model}_best.json，路径命名跟ml/train.py的.pkl+.json配对方式
    # 保持一致，推理端认出.pt后缀就去找同名.json。
    infer_meta = {
        "model": args.model,
        "hz": args.hz,
        "window_size": window_size,
        "stride": int(meta["stride"]),
        "n_channels": n_channels,
        "classes": classes,
        "gravity_aligned": str(meta.get("gravity_aligned", "True")) == "True",
        "label_mode": meta.get("label_mode", "majority"),
        "m2m": m2m,
        "ch_mean": ch_mean.tolist(),
        "ch_std": ch_std.tolist(),
        "model_cfg": cfg[args.model],
        "remap": args.remap,
    }
    with open(os.path.join(out_dir, f"dl_{args.model}_best.json"), "w", encoding="utf-8") as f:
        json.dump(infer_meta, f, ensure_ascii=False, indent=2)

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
            # 梯度裁剪：不管nan/inf是输入脏数据引起的还是单纯学习率/架构
            # 导致的梯度爆炸，都先用这个兜底防止某一步梯度突然爆炸把权重
            # 冲成nan（一旦权重变nan，后面每一步loss都是nan，训练等于
            # 全部作废，只能靠这一步提前防住）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
    print(f"\n[dl/train] {eval_tag}结果 ({mode_str}, best val model):")
    print(f"  Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")
    present_labels = sorted(set(all_true) | set(all_preds))
    present_names  = [classes[i] for i in present_labels]
    print(classification_report(all_true, all_preds, labels=present_labels,
                                target_names=present_names, zero_division=0))

    per_class = classification_report(all_true, all_preds, labels=present_labels,
                                      target_names=present_names,
                                      zero_division=0, output_dict=True)
    result = {
        "hz": args.hz, "model": args.model,
        "accuracy": acc, "macro_f1": f1,
        "classes": present_names,
        "per_class": {k: {m: round(v, 4) for m, v in per_class[k].items()
                          if m in ("precision", "recall", "f1-score")}
                      for k in present_names},
    }
    with open(os.path.join(out_dir, f"dl_{args.model}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[dl/train] 结果保存至 {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hz", type=int, required=True, choices=[5, 10, 15, 16, 20, 25, 50])
    parser.add_argument("--model", default="cnn_lstm",
                        choices=["cnn", "collar_cnn", "cnn_lstm", "transformer",
                                 "filternet", "filternet_m2m"])
    parser.add_argument("--config", default="configs/dl.yaml")
    parser.add_argument("--processed_dir", default="",
                        help="预处理数据目录，直接指定的话跳过下面--date/--missing_strategy"
                             "自动定位/生成那一套逻辑")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--remap", default="",
                        help="标签重映射 YAML 文件路径（用于合并类别，如16类原始细分"
                             "动作→4类行为，见configs/remap_custom_3class.yaml），"
                             "跟ml/train.py的--remap是同一份格式")
    # 以下几个参数不直接使用，只在--processed_dir留空时，用来跟
    # train_custom.sh用同一套规则算出processed_dir路径、以及在数据
    # 还没预处理过时自动调train_custom.sh --skip_ml生成（见
    # _resolve_processed_dir()），参数含义和train_custom.sh完全一致
    parser.add_argument("--date", default="",
                        help="主批次日期目录名，配合下面几个参数自动定位/生成预处理数据"
                             "（跟train_custom.sh的--date是同一个概念）")
    parser.add_argument("--missing_strategy", default="none",
                        choices=["none", "drop", "ffill", "drop_window"],
                        help="acc/gyro缺失值处理方式（默认none=历史行为，见"
                             "train_custom.sh的--missing_strategy说明），只在"
                             "--processed_dir留空、需要自动定位/生成数据时用得上，"
                             "同时也决定自动派生的--tag后缀（missing_<S>）")
    parser.add_argument("--tag", default="",
                        help="跟train_custom.sh的--tag一样，不传时自动等于"
                             "missing_<missing_strategy>")
    parser.add_argument("--source_hz", type=int, default=0)
    parser.add_argument("--extra_date", action="append", default=[],
                        help="DATE:HZ，可重复传，跟train_custom.sh的--extra_date一样")
    parser.add_argument("--window_s", type=float, default=0.0)
    parser.add_argument("--stride_s", type=float, default=0.0)
    parser.add_argument("--label_mode", default="")
    main(parser.parse_args())
