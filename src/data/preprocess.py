"""
预处理流程：原始 100Hz CSV → 降采样 → 窗口切分 → 保存 .npz

支持多个独立数据集，每个数据集单独预处理，互不影响。
用法示例：
  python src/data/preprocess.py --dataset a --output_dir data/processed_a
  python src/data/preprocess.py --dataset b --output_dir data/processed_b
"""

import os
import argparse
import numpy as np
import yaml
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from gravity_align import gravity_align_batch


def downsample(data, labels, source_hz, target_hz):
    if source_hz == target_hz:
        return data, labels
    from math import gcd
    g = gcd(source_hz, target_hz)
    up, down = target_hz // g, source_hz // g
    if up == 1:
        # 整除降采样，直接取点（快速路径）
        step = down
        data_ds = data[::step]
        # 标签用多数投票对齐到降采样后长度
        n_out = len(data_ds)
        labels_ds = np.array([
            Counter(labels[i * step: (i + 1) * step]).most_common(1)[0][0]
            for i in range(n_out)
        ])
    else:
        # 非整除比例，用多项式重采样（scipy）
        from scipy.signal import resample_poly
        data_ds = resample_poly(data, up, down, axis=0).astype(np.float32)
        n_out = len(data_ds)
        # 标签：按比例映射每个输出点回原始索引
        orig_indices = (np.arange(n_out) * (source_hz / target_hz)).astype(int)
        orig_indices = np.clip(orig_indices, 0, len(labels) - 1)
        labels_ds = labels[orig_indices]
    return data_ds, labels_ds


def sliding_window(data, labels, window_size, stride, keep_label_set=None):
    """返回 (X, y_majority, y_seq): y_seq 是每窗口内的逐帧标签，供 many-to-many 模型使用。"""
    X, y, y_seq = [], [], []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        frame_labels = labels[start:end]
        majority = Counter(frame_labels).most_common(1)[0][0]
        if keep_label_set is not None and majority not in keep_label_set:
            continue
        X.append(data[start:end])
        y.append(majority)
        y_seq.append(frame_labels)
    if not X:
        empty_X = np.empty((0, window_size, data.shape[1]), dtype=np.float32)
        return empty_X, np.empty((0,)), np.empty((0, window_size), dtype=np.int64)
    return (np.array(X, dtype=np.float32),
            np.array(y),
            np.array(y_seq, dtype=np.int64))


def split_by_dog(records, train_r, val_r, seed):
    rng = np.random.default_rng(seed)
    dog_ids = [r["dog_id"] for r in records]
    rng.shuffle(dog_ids)
    n = len(dog_ids)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    # 数据集较小时保证 val 和 test 各至少 1 个 ID
    if n_val == 0 and n >= 3:
        n_val = 1
    if n - n_train - n_val == 0 and n >= 3:
        n_train = max(1, n_train - 1)
    return (set(dog_ids[:n_train]),
            set(dog_ids[n_train:n_train + n_val]),
            set(dog_ids[n_train + n_val:]))


def split_windows_random(X_all, y_all, y_seq_all, train_r, val_r, seed):
    """按窗口随机划分（适合 subject 数太少的小数据集）。"""
    rng = np.random.default_rng(seed)
    n = len(X_all)
    idx = rng.permutation(n)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    # 保证 val/test 各至少 1 个窗口
    n_val  = max(1, n_val)  if n >= 3 else n_val
    n_test = max(1, n - n_train - n_val) if n >= 3 else n - n_train - n_val
    n_train = n - n_val - n_test
    i_train = idx[:n_train]
    i_val   = idx[n_train:n_train + n_val]
    i_test  = idx[n_train + n_val:]
    return (X_all[i_train], y_all[i_train], y_seq_all[i_train],
            X_all[i_val],   y_all[i_val],   y_seq_all[i_val],
            X_all[i_test],  y_all[i_test],  y_seq_all[i_test])


def process_split(records, dog_ids_set, window_size, stride, le, keep_label_set=None, use_gravity_align=True):
    X_all, y_all, y_seq_all = [], [], []
    valid_encoded = set(le.transform(list(keep_label_set))) if keep_label_set else None
    for r in records:
        if r["dog_id"] not in dog_ids_set:
            continue
        data, labels = r["data"], r["labels"]
        mask = np.isin(labels, list(keep_label_set)) if keep_label_set else np.ones(len(labels), bool)
        labels_enc = np.full(len(labels), -1, dtype=np.int64)
        labels_enc[mask] = le.transform(labels[mask])
        X, y, y_seq = sliding_window(data, labels_enc, window_size, stride, valid_encoded)
        if len(X) == 0:
            continue
        if use_gravity_align:
            X = gravity_align_batch(X)
        X_all.append(X)
        y_all.append(y)
        y_seq_all.append(y_seq)
    if not X_all:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))
    return np.concatenate(X_all), np.concatenate(y_all), np.concatenate(y_seq_all)


def load_records(args, cfg):
    """根据 --dataset 选择对应的 loader，返回 (records, keep_label_set)。"""
    if args.dataset == "a":
        from loader import load_dataset_files
        print("[preprocess] 数据集 A")
        records, _, _ = load_dataset_files(args.raw_csv_dir, args.dog_info)
        keep_labels_list = cfg.get("keep_labels", None)

    elif args.dataset == "b":
        from loader_b import load_dataset_b
        print("[preprocess] 数据集 B")
        records, _, _ = load_dataset_b(args.raw_csv_b)
        keep_labels_list = cfg.get("keep_labels_b", cfg.get("keep_labels", None))

    elif args.dataset == "custom":
        from loader_custom import load_dataset_custom
        print("[preprocess] 自采数据集")
        custom_cfg = cfg.get("custom", {})
        csv_path = args.raw_csv_custom or custom_cfg.get("csv_path", "data/raw_custom/data.csv")
        if "source_hz" in custom_cfg:
            cfg["source_hz"] = custom_cfg["source_hz"]
        records, _, _ = load_dataset_custom(csv_path, custom_cfg)
        keep_labels_list = custom_cfg.get("keep_labels") or None

    elif args.dataset in ("cat_smit2023", "cat_dunford2024", "cat_smit2024"):
        from loader_cat import load_dataset_cat
        print(f"[preprocess] 猫咪数据集: {args.dataset}")
        cat_cfg = cfg.get(args.dataset, {})
        csv_path = cat_cfg.get("csv_path", f"data/raw_{args.dataset}/data.csv")
        if "source_hz" in cat_cfg:
            cfg["source_hz"] = cat_cfg["source_hz"]
        records, _, _ = load_dataset_cat(csv_path, cat_cfg)
        keep_labels_list = cat_cfg.get("keep_labels") or None

    else:
        raise ValueError(
            f"未知数据集: {args.dataset}\n"
            f"支持: a / b / custom / cat_smit2023 / cat_dunford2024 / cat_smit2024"
        )

    keep_label_set = set(keep_labels_list) if keep_labels_list else None
    if keep_label_set:
        print(f"[preprocess] 过滤标签，只保留: {keep_labels_list}")
    return records, keep_label_set


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    source_hz = cfg["source_hz"]
    target_hz_list = cfg["target_hz_list"]
    window_sec = cfg["window_seconds"]
    stride_sec = cfg["stride_seconds"]
    seed = cfg["seed"]
    train_r = cfg["train_ratio"]
    val_r = cfg["val_ratio"]

    records, keep_label_set = load_records(args, cfg)

    # 拟合标签编码器
    if keep_label_set:
        all_labels = np.concatenate([
            r["labels"][np.isin(r["labels"], list(keep_label_set))] for r in records
        ])
    else:
        all_labels = np.concatenate([r["labels"] for r in records])

    all_labels = all_labels[all_labels != np.array(None)]

    le = LabelEncoder()
    le.fit(all_labels)
    classes = list(le.classes_)
    print(f"[preprocess] 行为类别 ({len(classes)}): {classes}")

    # 决定划分策略
    n_subjects = len(records)
    strategy = args.split_strategy
    if strategy == "auto":
        strategy = "subject" if n_subjects >= 10 else "random"
        if strategy == "random":
            print(f"[preprocess] subject 数={n_subjects} < 10，自动使用窗口随机划分（可用 --split_strategy subject 强制按subject划分）")

    if strategy == "subject":
        train_ids, val_ids, test_ids = split_by_dog(records, train_r, val_r, seed)
        print(f"[preprocess] 按 subject 划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    else:
        train_ids = val_ids = test_ids = None
        print(f"[preprocess] 按窗口随机划分: train={train_r:.0%} / val={val_r:.0%} / test={1-train_r-val_r:.0%}")

    for target_hz in target_hz_list:
        if target_hz > source_hz:
            print(f"\n[preprocess] 跳过 {target_hz}Hz（高于源采样率 {source_hz}Hz，无法上采样）")
            continue

        window_size = int(window_sec * target_hz)
        stride = int(stride_sec * target_hz)

        print(f"\n[preprocess] 处理 {target_hz}Hz ...")

        ds_records = []
        for r in records:
            data_ds, labels_ds = downsample(r["data"], r["labels"], source_hz, target_hz)
            ds_records.append({**r, "data": data_ds, "labels": labels_ds})

        ga = not args.no_gravity_align

        if strategy == "subject":
            X_train, y_train, y_seq_train = process_split(ds_records, train_ids, window_size, stride, le, keep_label_set, ga)
            X_val,   y_val,   y_seq_val   = process_split(ds_records, val_ids,   window_size, stride, le, keep_label_set, ga)
            X_test,  y_test,  y_seq_test  = process_split(ds_records, test_ids,  window_size, stride, le, keep_label_set, ga)
        else:
            # 先把所有窗口提取出来，再随机划分
            all_ids = set(r["dog_id"] for r in ds_records)
            X_all, y_all, y_seq_all = process_split(ds_records, all_ids, window_size, stride, le, keep_label_set, ga)
            (X_train, y_train, y_seq_train,
             X_val,   y_val,   y_seq_val,
             X_test,  y_test,  y_seq_test) = split_windows_random(X_all, y_all, y_seq_all, train_r, val_r, seed)

        print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

        out_dir = os.path.join(args.output_dir, f"{target_hz}hz")
        os.makedirs(out_dir, exist_ok=True)

        meta = {
            "hz": target_hz, "window_size": window_size, "stride": stride,
            "n_channels": str(X_train.shape[2] if X_train.ndim == 3 else 6),
            "classes": str(classes),
            "split_strategy": strategy,
            "train_dog_ids": str(list(train_ids)) if train_ids is not None else "[]",
            "val_dog_ids":   str(list(val_ids))   if val_ids   is not None else "[]",
            "test_dog_ids":  str(list(test_ids))  if test_ids  is not None else "[]",
            "dataset": args.dataset,
            "gravity_aligned": str(not args.no_gravity_align),
        }
        if not args.no_gravity_align:
            print(f"  [重力对齐] 已启用")

        np.savez_compressed(os.path.join(out_dir, "train.npz"), X=X_train, y=y_train, y_seq=y_seq_train, **meta)
        np.savez_compressed(os.path.join(out_dir, "val.npz"),   X=X_val,   y=y_val,   y_seq=y_seq_val,   **meta)
        np.savez_compressed(os.path.join(out_dir, "test.npz"),  X=X_test,  y=y_test,  y_seq=y_seq_test,  **meta)
        feat_cache = os.path.join(out_dir, "ml_features.npz")
        if os.path.exists(feat_cache):
            os.remove(feat_cache)
            print(f"  🗑  已删除旧特征缓存 {feat_cache}")
        print(f"  ✅ 保存至 {out_dir}/")

    print("\n[preprocess] 全部完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["a", "b", "custom",
                                 "cat_smit2023", "cat_dunford2024", "cat_smit2024"],
                        help="数据集标识")
    parser.add_argument("--raw_csv_dir", default="data/raw/csv",
                        help="数据集A的 CSV 目录")
    parser.add_argument("--dog_info", default="data/raw/DogInfo.csv",
                        help="数据集A的 DogInfo.csv")
    parser.add_argument("--raw_csv_b", default="data/raw_b/df_raw.csv",
                        help="数据集B的 df_raw.csv")
    parser.add_argument("--raw_csv_custom", default="",
                        help="自采数据集CSV路径，留空则读 configs/data.yaml custom.csv_path")
    parser.add_argument("--output_dir", required=True,
                        help="输出目录，如 data/processed_a")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--no_gravity_align", action="store_true",
                        help="不做重力轴对齐（默认启用）")
    parser.add_argument("--split_strategy", default="auto",
                        choices=["auto", "subject", "random"],
                        help="划分策略: auto=subject数>=10用subject否则用random, subject=按动物ID划分, random=窗口随机划分")
    main(parser.parse_args())
