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
from gravity_align import gravity_align_batch, append_raw_tilt_batch


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


def compute_segment_ids(labels, next_id=0):
    """给连续同标签片段分配全局唯一id，相邻标签不同就换一个新id。
    返回 (seg_ids数组, 下一个可用id)——next_id 在多条记录之间传递，
    保证不同狗/不同文件产生的片段id不会撞车。"""
    n = len(labels)
    if n == 0:
        return np.empty(0, dtype=np.int64), next_id
    change = np.empty(n, dtype=bool)
    change[0] = False
    change[1:] = labels[1:] != labels[:-1]
    seg_ids = next_id + np.cumsum(change)
    return seg_ids.astype(np.int64), int(seg_ids[-1]) + 1


def sliding_window(data, labels, window_size, stride, keep_label_set=None,
                   label_mode="majority", seg_id_labels=None):
    """返回 (X, y, y_seq, y_seg)：
      y_seq 是每窗口内的逐帧标签，供 many-to-many 模型使用；
      y_seg 是每个窗口所属的"连续标注片段"编号，供按片段分组划分数据集、
      避免同一片段产生的高度重叠窗口被拆到 train/val/test 不同集合（数据泄漏）。

    label_mode:
      "majority"（默认，原有行为）：窗口标签取窗口内出现次数最多的标签，
        跨越标注边界的窗口会被强行归为占比更多的那一类（窗口越靠近边界，
        标签跟窗口内实际内容的吻合度越低，但保持原样不动，供对比用）。
      "center"：窗口标签取窗口正中心那一帧的标签，代表"这个窗口的特征描述
        的是中心点这一瞬间"，不做多数投票压制。要真正发挥效果需要配合更密的
        步长（否则中心点之间会有大段没有预测覆盖的空隙）。
    """
    X, y, y_seq, y_seg = [], [], [], []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        frame_labels = labels[start:end]
        if label_mode == "center":
            label = frame_labels[window_size // 2]
        else:
            label = Counter(frame_labels).most_common(1)[0][0]
        if keep_label_set is not None and label not in keep_label_set:
            continue
        X.append(data[start:end])
        y.append(label)
        y_seq.append(frame_labels)
        if seg_id_labels is not None:
            seg_frame = seg_id_labels[start:end]
            if label_mode == "center":
                y_seg.append(int(seg_frame[window_size // 2]))
            else:
                y_seg.append(int(Counter(seg_frame).most_common(1)[0][0]))
        else:
            y_seg.append(-1)
    if not X:
        empty_X = np.empty((0, window_size, data.shape[1]), dtype=np.float32)
        return (empty_X, np.empty((0,)), np.empty((0, window_size), dtype=np.int64),
                np.empty((0,), dtype=np.int64))
    return (np.array(X, dtype=np.float32),
            np.array(y),
            np.array(y_seq, dtype=np.int64),
            np.array(y_seg, dtype=np.int64))


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


def split_windows_by_segment(X_all, y_all, y_seq_all, seg_ids_all, train_r, val_r, seed):
    """按"连续标注片段"分组划分，而不是按窗口随机划分。

    同一个连续片段（比如一次完整的抓挠事件）产生的所有窗口——不管重叠率多高、
    步长多密——保证被整体分进同一个集合，不会有一部分进train、一部分进val/test。
    这是为了避免"高度重叠的近似窗口分别出现在训练集和验证/测试集里"造成的数据
    泄漏（验证集分数虚高，但不代表真实泛化能力）。步长越密、label_mode=center
    时这个问题越严重，这个函数从根上解决，不管当前用什么步长/label_mode都适用。

    做法：**按类别分别**把该类别的片段打乱，各自按窗口数量累计分配到
    train/val/test，尽量逼近目标比例。必须按类别分别分配，而不是把所有类别的
    片段混在一起只看总窗口数——否则哪个类别的片段先被随机打乱到前面，val/test
    的目标窗口数就会被哪个类别占满，导致片段数少、单个片段窗口数多的类别
    （比如一大段连续"活动"）很容易被整体分进train、val里几乎没有，验证集上
    该类别的样本量跟训练集比例严重失调，指标失真。
    """
    rng = np.random.default_rng(seed)
    n = len(X_all)
    if n == 0:
        return (X_all, y_all, y_seq_all, X_all, y_all, y_seq_all, X_all, y_all, y_seq_all)

    test_r = max(0.0, 1.0 - train_r - val_r)

    # 按片段id分组，收集每个片段对应的窗口下标 + 该片段的类别（片段内窗口标签一致）
    seg_to_indices = {}
    for i, sid in enumerate(seg_ids_all):
        seg_to_indices.setdefault(int(sid), []).append(i)

    # 按类别分桶：每个类别内部独立打乱、独立按比例分配，避免类别间互相挤占目标名额
    label_to_segs = {}
    for sid, idxs in seg_to_indices.items():
        lbl = y_all[idxs[0]]
        label_to_segs.setdefault(lbl, []).append(sid)

    train_idx, val_idx, test_idx = [], [], []
    for lbl, sids in label_to_segs.items():
        rng.shuffle(sids)
        n_lbl = sum(len(seg_to_indices[sid]) for sid in sids)
        target_val  = val_r * n_lbl
        target_test = test_r * n_lbl
        val_count = test_count = 0
        for sid in sids:
            idxs = seg_to_indices[sid]
            if val_count < target_val:
                val_idx.extend(idxs)
                val_count += len(idxs)
            elif test_count < target_test:
                test_idx.extend(idxs)
                test_count += len(idxs)
            else:
                train_idx.extend(idxs)

    i_train = np.array(sorted(train_idx), dtype=np.int64)
    i_val   = np.array(sorted(val_idx),   dtype=np.int64)
    i_test  = np.array(sorted(test_idx),  dtype=np.int64)
    return (X_all[i_train], y_all[i_train], y_seq_all[i_train],
            X_all[i_val],   y_all[i_val],   y_seq_all[i_val],
            X_all[i_test],  y_all[i_test],  y_seq_all[i_test])


def process_label_concat(records, window_size, stride, le, keep_label_set=None, use_gravity_align=True):
    """按类别汇总所有片段的窗口：每个连续片段内部各自滑窗，汇总后合并。
    不跨片段边界取窗口，避免不同时间/动物的数据拼接产生无意义的假窗口。

    注：这里的窗口本来就只在单一标签的连续片段内部滑动，天然不会跨边界，
    所以 --label_mode 对这个策略没有意义（majority 和 center 结果完全一样），
    不需要传参数进来。
    """
    from collections import defaultdict as _dd
    # label_id → list of (seg_id, continuous segment array)
    label_segs = _dd(list)
    next_seg_id = 0

    for r in records:
        data, labels = r["data"], r["labels"]
        # 找出每段连续的同类别行
        i = 0
        n = len(labels)
        while i < n:
            lbl = labels[i]
            if keep_label_set and lbl not in keep_label_set:
                i += 1
                continue
            # 找连续同标签的结束位置
            j = i + 1
            while j < n and labels[j] == lbl:
                j += 1
            seg = data[i:j]
            if len(seg) >= window_size:
                lbl_id = le.transform([lbl])[0]
                label_segs[lbl_id].append((next_seg_id, seg.astype(np.float32)))
                next_seg_id += 1
            i = j

    X_all, y_all, y_seq_all, y_seg_all = [], [], [], []
    for lbl_id in sorted(label_segs.keys()):
        wins, seg_ids = [], []
        for seg_id, seg in label_segs[lbl_id]:
            for start in range(0, len(seg) - window_size + 1, stride):
                wins.append(seg[start:start + window_size])
                seg_ids.append(seg_id)
        if not wins:
            continue
        arr = np.array(wins, dtype=np.float32)
        tilt = append_raw_tilt_batch(arr)[:, :, 6:8]  # 原始（未对齐）姿态角，须在重力对齐前算
        if use_gravity_align:
            arr = gravity_align_batch(arr)
        arr = np.concatenate([arr, tilt], axis=2)
        X_all.append(arr)
        y_all.append(np.full(len(arr), lbl_id, dtype=np.int64))
        y_seq_all.append(np.tile(
            np.full(window_size, lbl_id, dtype=np.int64), (len(arr), 1)
        ))
        y_seg_all.append(np.array(seg_ids, dtype=np.int64))

    if not X_all:
        return (np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=np.int64))
    return (np.concatenate(X_all), np.concatenate(y_all),
            np.concatenate(y_seq_all), np.concatenate(y_seg_all))


def process_split(records, dog_ids_set, window_size, stride, le, keep_label_set=None,
                  use_gravity_align=True, label_mode="majority"):
    X_all, y_all, y_seq_all, y_seg_all = [], [], [], []
    valid_encoded = set(le.transform(list(keep_label_set))) if keep_label_set else None
    next_seg_id = 0
    for r in records:
        if r["dog_id"] not in dog_ids_set:
            continue
        data, labels = r["data"], r["labels"]
        mask = np.isin(labels, list(keep_label_set)) if keep_label_set else np.ones(len(labels), bool)
        labels_enc = np.full(len(labels), -1, dtype=np.int64)
        labels_enc[mask] = le.transform(labels[mask])
        seg_id_labels, next_seg_id = compute_segment_ids(labels, next_seg_id)
        X, y, y_seq, y_seg = sliding_window(data, labels_enc, window_size, stride, valid_encoded,
                                            label_mode, seg_id_labels)
        if len(X) == 0:
            continue
        tilt = append_raw_tilt_batch(X)[:, :, 6:8]  # 原始（未对齐）姿态角，须在重力对齐前算
        if use_gravity_align:
            X = gravity_align_batch(X)
        X = np.concatenate([X, tilt], axis=2)
        X_all.append(X)
        y_all.append(y)
        y_seq_all.append(y_seq)
        y_seg_all.append(y_seg)
    if not X_all:
        return (np.empty((0,)), np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=np.int64))
    return (np.concatenate(X_all), np.concatenate(y_all),
            np.concatenate(y_seq_all), np.concatenate(y_seg_all))


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

    target_hz_list = [args.hz] if args.hz else cfg["target_hz_list"]
    window_sec = cfg["window_seconds"]
    stride_sec = args.stride_s if args.stride_s > 0 else cfg["stride_seconds"]
    seed = cfg["seed"]
    train_r = args.train_ratio if args.train_ratio > 0 else cfg["train_ratio"]
    val_r   = args.val_ratio   if args.val_ratio   > 0 else cfg["val_ratio"]
    if args.test_ratio >= 0:
        test_r = args.test_ratio
        # 用户显式指定了测试集比例，重新归一化 train/val
        total = train_r + val_r + test_r
        if abs(total - 1.0) > 1e-6:
            print(f"[preprocess] 警告: train+val+test={total:.3f}，自动归一化")
            train_r, val_r = train_r / total, val_r / total

    records, keep_label_set = load_records(args, cfg)
    source_hz = cfg["source_hz"]  # load_records 可能修改 cfg["source_hz"]（如 custom 数据集），需在其后读取

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

    test_r_display = max(0.0, round(1 - train_r - val_r, 6))  # 避免浮点误差显示成 -0%
    if strategy == "subject":
        train_ids, val_ids, test_ids = split_by_dog(records, train_r, val_r, seed)
        print(f"[preprocess] 按 subject 划分: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    elif strategy == "label_concat":
        train_ids = val_ids = test_ids = None
        print(f"[preprocess] 按类别拼接滑窗+按片段分组划分: train={train_r:.0%} / val={val_r:.0%} / test={test_r_display:.0%}")
    else:
        train_ids = val_ids = test_ids = None
        print(f"[preprocess] 按连续片段分组划分: train={train_r:.0%} / val={val_r:.0%} / test={test_r_display:.0%}")

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
            # 按狗分组划分，狗与狗之间本来就不会共享同一个连续片段，天然无泄漏，seg_id 不需要
            X_train, y_train, y_seq_train, _ = process_split(ds_records, train_ids, window_size, stride, le, keep_label_set, ga, args.label_mode)
            X_val,   y_val,   y_seq_val,   _ = process_split(ds_records, val_ids,   window_size, stride, le, keep_label_set, ga, args.label_mode)
            X_test,  y_test,  y_seq_test,  _ = process_split(ds_records, test_ids,  window_size, stride, le, keep_label_set, ga, args.label_mode)
        elif strategy == "label_concat":
            # 按类别拼接所有片段后滑窗，窗口纯粹属于一个类别（label_mode 在这里没有意义，不传）
            X_all, y_all, y_seq_all, y_seg_all = process_label_concat(ds_records, window_size, stride, le, keep_label_set, ga)
            (X_train, y_train, y_seq_train,
             X_val,   y_val,   y_seq_val,
             X_test,  y_test,  y_seq_test) = split_windows_by_segment(X_all, y_all, y_seq_all, y_seg_all, train_r, val_r, seed)
        else:
            # 先把所有窗口提取出来，再按连续片段分组划分（避免高度重叠窗口跨集合泄漏）
            all_ids = set(r["dog_id"] for r in ds_records)
            X_all, y_all, y_seq_all, y_seg_all = process_split(ds_records, all_ids, window_size, stride, le, keep_label_set, ga, args.label_mode)
            (X_train, y_train, y_seq_train,
             X_val,   y_val,   y_seq_val,
             X_test,  y_test,  y_seq_test) = split_windows_by_segment(X_all, y_all, y_seq_all, y_seg_all, train_r, val_r, seed)


        out_dir = os.path.join(args.output_dir, f"{target_hz}hz")
        os.makedirs(out_dir, exist_ok=True)

        meta = {
            "hz": target_hz, "window_size": window_size, "stride": stride,
            "n_channels": str(X_train.shape[2] if X_train.ndim == 3 else 6),
            "classes": str(classes),
            "split_strategy": strategy,
            "label_mode": args.label_mode,
            "train_ratio": str(train_r),
            "val_ratio":   str(val_r),
            "test_ratio":  str(round(1.0 - train_r - val_r, 6)),
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
                        choices=["auto", "subject", "random", "label_concat"],
                        help="划分策略: auto=subject数>=10用subject否则用random, subject=按动物ID划分, random=窗口随机划分, label_concat=按类别拼接片段后滑窗（充分利用短片段）")
    parser.add_argument("--train_ratio", type=float, default=0.0,
                        help="训练集比例（0=从 configs/data.yaml 读取）")
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="验证集比例（0=从 configs/data.yaml 读取）")
    parser.add_argument("--test_ratio", type=float, default=-1.0,
                        help="测试集比例（-1=由 1-train-val 推导，0=无测试集）")
    parser.add_argument("--label_mode", default="majority", choices=["majority", "center"],
                        help="窗口标签怎么定：majority=多数投票（默认，原有行为不变），"
                             "center=取窗口正中心那一帧的标签（对 label_concat 策略无影响，"
                             "该策略下窗口本来就纯净）")
    parser.add_argument("--hz", type=int, default=0,
                        help="只处理指定采样率（0=处理所有，默认0）")
    parser.add_argument("--stride_s", type=float, default=0.0,
                        help="训练窗口步长（秒），覆盖 configs/data.yaml 的 stride_seconds"
                             "（0=用配置文件默认值）。步长越密，训练窗口越多、"
                             "越能覆盖标注边界过渡区，但预处理和训练耗时也越高")
    main(parser.parse_args())
