"""
从 Label Studio JSON 标注中提取指定类别的真实片段，数据增强后生成合成数据。

不再硬编码时间段，直接读取当天导出的 JSON，自动提取任意类别的所有标注片段。

用法:
  # 合成抓挠数据（默认）
  python src/data/synthesize_scratch.py \
    --json data/raw_custom/2026_7_23/merged_tmp.json \
    --csv_dir data/raw_wit/ \
    --output data/synthetic/scratch_2026_7_23.npz \
    --label 抓挠 --hz 16 --n_aug 30

  # 合成睡觉数据（类别数据少时同理）
  python src/data/synthesize_scratch.py \
    --json data/raw_custom/2026_7_23/merged_tmp.json \
    --csv_dir data/raw_wit/ \
    --output data/synthetic/sleep_2026_7_23.npz \
    --label 睡觉 --hz 16 --n_aug 10

  # 验证生成数量
  python -c "import numpy as np; d=np.load('data/synthetic/scratch_2026_7_23.npz'); print(d['X'].shape)"
"""

import argparse
import json
import os
import urllib.request
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ACC_CANDIDATES  = [
    ["acc_x", "acc_y", "acc_z"],
    ["AccX",  "AccY",  "AccZ"],
    ["AX",    "AY",    "AZ"],
    ["ax",    "ay",    "az"],
    ["Ax",    "Ay",    "Az"],
]
GYRO_CANDIDATES = [
    ["gyro_x", "gyro_y", "gyro_z"],
    ["GyroX",  "GyroY",  "GyroZ"],
    ["GX",     "GY",     "GZ"],
    ["gx",     "gy",     "gz"],
    ["Gx",     "Gy",     "Gz"],
    ["wx",     "wy",     "wz"],
]
TS_KEYWORDS = ["time", "timestamp", "datetime", "date", "chip_time"]


def _find_cols(cols, candidates):
    for grp in candidates:
        if all(c in cols for c in grp):
            return grp
    return None


def _find_ts_col(cols):
    low = [c.lower() for c in cols]
    for kw in TS_KEYWORDS:
        for i, cl in enumerate(low):
            if kw in cl:
                return cols[i]
    return None


def _load_sensor(url, csv_dir, name):
    if not url:
        return None
    try:
        fname = os.path.basename(urlparse(url).path)
        if csv_dir:
            local = os.path.join(csv_dir, fname)
            if os.path.exists(local):
                df = pd.read_csv(local)
            else:
                tmp = f"/tmp/_syn_{fname}"
                urllib.request.urlretrieve(url, tmp)
                df = pd.read_csv(tmp)
        else:
            tmp = f"/tmp/_syn_{fname}"
            urllib.request.urlretrieve(url, tmp)
            df = pd.read_csv(tmp)
    except Exception as e:
        print(f"  [错误] {name} 加载失败: {e}")
        return None
    df.columns = [c.strip() for c in df.columns]
    acc_cols  = _find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = _find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    ts_col    = _find_ts_col(df.columns.tolist())
    if acc_cols is None or ts_col is None:
        print(f"  [错误] {name}: 找不到加速度列或时间戳列")
        return None
    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    return df, acc_cols, gyro_cols


def extract_segments_from_json(tasks, csv_dir, target_label, min_rows=16, verbose=True):
    """从 Label Studio JSON 中提取所有 target_label 的原始片段，返回 list of (N,6) ndarray。
    verbose=True 时打印每一类跳过原因的计数，方便核对"标注里明明有N段，怎么只提取出M个"。"""
    segments = []
    # 缓存已加载的 CSV，避免同一文件重复读取
    _csv_cache = {}

    # 跳过原因计数：方便定位"标注段数"跟"实际提取出的片段数"之间的差距来自哪一步
    skip_no_annotation = 0     # task 本身没有 annotations
    skip_no_url = 0            # data 里没有可用的 csv/csv1/csv2 链接
    skip_load_fail = 0         # csv 下载/解析失败（_load_sensor 内部已打印具体错误）
    skip_no_label_match = 0    # 该段标签不是 target_label（正常情况，不算异常）
    skip_sensor_mismatch = 0   # from_name 在 sensor_map 里找不到对应的CSV（双传感器from_name对不上）
    skip_too_short = 0         # 时间段内匹配到的行数 < min_rows
    n_matched_label = 0        # 标签匹配上 target_label 的段总数（跟 analyze_label_segments.py 的片段数对应）

    for task in tasks:
        task_id = task["id"]
        data    = task.get("data", {})
        anns    = task.get("annotations", [])
        if not anns:
            skip_no_annotation += 1
            continue

        is_multi = "csv1" in data or "csv2" in data

        task_has_url = False   # 这个task至少有一个csv/csv1/csv2链接
        load_failed  = False   # 有链接，但至少一个下载/解析失败了

        if is_multi:
            sensor_map = {}
            for idx in ("1", "2"):
                url = data.get(f"csv{idx}", "")
                if not url:
                    continue
                task_has_url = True
                if url not in _csv_cache:
                    _csv_cache[url] = _load_sensor(url, csv_dir, f"task{task_id}_imu{idx}")
                if _csv_cache[url]:
                    sensor_map[f"label{idx}"] = _csv_cache[url]
                else:
                    load_failed = True
        else:
            url = data.get("csv", "")
            task_has_url = bool(url)
            sensor_map = {}
            if url:
                if url not in _csv_cache:
                    _csv_cache[url] = _load_sensor(url, csv_dir, f"task{task_id}_imu")
                if _csv_cache[url]:
                    sensor_map = {"label": _csv_cache[url]}
                else:
                    load_failed = True

        for ann in anns:
            for seg in ann.get("result", []):
                val    = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                t0_str = val.get("start", "")
                t1_str = val.get("end",   "")
                fn     = seg.get("from_name", "")
                if not labels or labels[0] != target_label or not t0_str or not t1_str:
                    skip_no_label_match += 1
                    continue
                n_matched_label += 1
                if not task_has_url:
                    skip_no_url += 1
                    continue
                if fn not in sensor_map:
                    # 该task的CSV要么下载/解析失败了，要么加载成功但from_name跟label1/label2对不上
                    if load_failed:
                        skip_load_fail += 1
                    else:
                        skip_sensor_mismatch += 1
                    continue
                df, acc_cols, gyro_cols = sensor_map[fn]
                t0   = pd.to_datetime(t0_str)
                t1   = pd.to_datetime(t1_str)
                mask = (df["_ts"] >= t0) & (df["_ts"] <= t1)
                sub  = df[mask]
                if len(sub) < min_rows:
                    skip_too_short += 1
                    print(f"  [跳过] task{task_id} {t0_str} 只有 {len(sub)} 行")
                    continue
                acc  = sub[acc_cols].values.astype(np.float32)
                gyro = sub[gyro_cols].values.astype(np.float32) if gyro_cols \
                       else np.zeros((len(sub), 3), dtype=np.float32)
                segments.append(np.concatenate([acc, gyro], axis=1))

    if verbose:
        print(f"\n  [提取明细] 标签='{target_label}' 匹配到的标注段: {n_matched_label}")
        print(f"    ✅ 成功提取:                 {len(segments)}")
        print(f"    ⚠️  task无标注被跳过:          {skip_no_annotation}（不影响，跟本标签无关）")
        print(f"    ❌ 段所在task没有csv链接:      {skip_no_url}")
        print(f"    ❌ csv下载/解析失败:          {skip_load_fail}（具体错误见上面 [错误] 行）")
        print(f"    ❌ from_name跟sensor_map对不上: {skip_sensor_mismatch}"
              f"（双传感器标注用的from_name不是label1/label2）")
        print(f"    ❌ 时间段内行数<{min_rows}(约{min_rows/16:.2f}s@16Hz): {skip_too_short}")
        accounted = len(segments) + skip_no_url + skip_load_fail + skip_sensor_mismatch + skip_too_short
        if accounted != n_matched_label:
            print(f"    [提示] 明细加总({accounted})跟匹配段数({n_matched_label})对不上，"
                  f"可能有未覆盖到的分支，欢迎反馈")

    return segments


# ── 增强函数 ──────────────────────────────────────────────────────────────────

def aug_noise(seg, scale=0.02):
    return seg + np.random.randn(*seg.shape).astype(np.float32) * scale


def aug_scale(seg, low=0.85, high=1.15):
    """只缩放"动态成分"（每个通道减去自身均值后剩下的部分），均值（重力/静态偏置）
    原样保留。如果连均值一起按各通道独立随机缩放，加速度计的重力模长会被破坏——
    真实数据静止/缓动时模长应接近9.8，这个物理约束被我们的特征（acc_mag、pitch/
    roll、SMA等）显式捕捉，破坏它会让合成样本"一眼假"，模型学到的会是"这是不是
    合成数据"而不是"这是不是抓挠"，验证分数好看但没有真实意义。"""
    mean = seg.mean(axis=0, keepdims=True)
    dynamic = seg - mean
    s = np.random.uniform(low, high, size=(1, seg.shape[1])).astype(np.float32)
    return mean + dynamic * s


def aug_time_shift(seg, max_frac=0.1):
    shift = np.random.randint(1, max(2, int(len(seg) * max_frac)))
    return np.roll(seg, shift, axis=0)


def aug_time_stretch(seg, low=0.8, high=1.2):
    from scipy.signal import resample
    factor    = np.random.uniform(low, high)
    new_len   = max(4, int(len(seg) * factor))
    stretched = resample(seg, new_len, axis=0).astype(np.float32)
    return resample(stretched, len(seg), axis=0).astype(np.float32)


def augment_segment(seg, n_aug, rng):
    # aug_flip_axis 已移除：翻转轴虽不改变模长，但可能把重力方向翻到"上下颠倒"，
    # 这种朝向在真实录制里几乎不会持续2秒以上，会给 pitch/roll 这类姿态特征
    # 制造不真实的异常值，风险不确定，直接去掉比留着更稳妥
    aug_fns  = [aug_noise, aug_scale, aug_time_shift]
    variants = []
    for _ in range(n_aug):
        out    = seg.copy()
        chosen = rng.choice(len(aug_fns), size=rng.integers(1, 4), replace=False)
        for i in chosen:
            try:
                out = aug_fns[i](out)
            except Exception:
                pass
        if rng.random() < 0.2:
            out = aug_time_stretch(out)
        variants.append(out)
    return variants


def sliding_windows(data, window_size, stride):
    wins = []
    for start in range(0, len(data) - window_size + 1, stride):
        wins.append(data[start:start + window_size])
    return wins


def _auto_target_from_processed(processed_dir, hz, label, remap_cfg=None):
    """读取已预处理的训练集，返回（去掉 label 类后）最大类别的训练窗口数。"""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__)))
    try:
        from dataset import load_all_splits
    except ImportError:
        sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "../ml"))
        from dataset import load_all_splits  # type: ignore

    try:
        (_, y_tr, _), _, _, meta = load_all_splits(hz, processed_dir)
    except Exception as e:
        print(f"  [auto_target] 无法读取 processed_dir: {e}")
        return 0

    import ast
    classes = meta.get("classes", [])
    if isinstance(classes, str):
        classes = ast.literal_eval(classes)
    if remap_cfg:
        # 应用 remap 后统计
        remap = {}
        for src, dst in remap_cfg.items():
            if src in classes and dst in classes:
                remap[classes.index(src)] = classes.index(dst)
        y_remapped = np.array([remap.get(int(v), int(v)) for v in y_tr])
        counts = np.bincount(y_remapped, minlength=len(classes))
    else:
        counts = np.bincount(y_tr.astype(int), minlength=len(classes))

    # 排除要合成的 label 本身（它的真实数据可能很少）
    label_idx = classes.index(label) if label in classes else -1
    other_counts = [c for i, c in enumerate(counts) if i != label_idx]
    target = int(max(other_counts)) if other_counts else 0
    print(f"  [auto_target] 各类训练窗口数: { {classes[i]: int(counts[i]) for i in range(len(classes))} }")
    print(f"  [auto_target] 自动设置 target_windows = {target}（最大的其他类别窗口数）")
    return target


def main():
    parser = argparse.ArgumentParser(description="从 Label Studio JSON 生成指定类别的合成数据")
    parser.add_argument("--json",     required=True, help="Label Studio 导出的 JSON（支持 merged_tmp.json）")
    parser.add_argument("--csv_dir",  default="",    help="本地 CSV 目录（不填则从 URL 下载）")
    parser.add_argument("--output",   required=True, help="输出 .npz 路径")
    parser.add_argument("--label",    default="抓挠", help="要合成的类别名称（默认：抓挠）")
    parser.add_argument("--hz",       type=int,   default=16,  help="采样率（默认 16）")
    parser.add_argument("--window_s", type=float, default=2.0, help="窗口秒数（默认 2.0）")
    parser.add_argument("--stride_s", type=float, default=1.0, help="步长秒数（默认 1.0）")
    parser.add_argument("--n_aug",    type=int,   default=30,  help="每个原始片段生成的增强数量（默认 30）")
    parser.add_argument("--target_windows", type=int, default=-1,
                        help="目标窗口总数（默认-1=自动从 processed_dir 推算；0=不限制；正整数=手动指定）")
    parser.add_argument("--processed_dir", default="",
                        help="已预处理的数据目录，用于自动推算 target_windows（默认：自动推算时必填）")
    parser.add_argument("--remap",    default="",   help="remap YAML 路径（用于自动推算时的类别映射）")
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    rng         = np.random.default_rng(args.seed)
    window_size = int(args.window_s * args.hz)
    stride      = int(args.stride_s * args.hz)
    print(f"目标类别='{args.label}'  窗口={window_size}点  步长={stride}点  采样率={args.hz}Hz")

    # 自动推算 target_windows
    target_windows = args.target_windows
    if target_windows == -1:
        if args.processed_dir:
            remap_cfg = None
            if args.remap and os.path.exists(args.remap):
                import yaml
                with open(args.remap, encoding="utf-8") as f:
                    remap_cfg = yaml.safe_load(f)
            target_windows = _auto_target_from_processed(
                args.processed_dir, args.hz, args.label, remap_cfg)
            if target_windows == 0:
                print("  [auto_target] 推算失败，将不限制窗口数")
        else:
            print("  [提示] 未指定 --processed_dir，无法自动推算 target_windows，将不限制窗口数")
            print("         建议加上 --processed_dir data/processed_<DATE> 让脚本自动计算")
            target_windows = 0

    with open(args.json, encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"\n加载 JSON: {len(tasks)} 个 task")

    print(f"\n── 提取 '{args.label}' 片段 ──")
    segments = extract_segments_from_json(tasks, args.csv_dir, args.label)
    print(f"\n共提取 {len(segments)} 个原始片段")
    if not segments:
        print("[错误] 未找到任何片段，请检查 --label 名称和 JSON 内容")
        return

    # seg_id 记录每个窗口来自 segments 里的第几个原始片段（增强出来的变体沿用同一个id）：
    # 同一个原始事件产生的原始窗口+所有增强变体窗口，是"近乎重复"的内容，必须在
    # train/val/test 划分时被当作同一组，否则训练集和验证集里会混进彼此的近似
    # 副本，造成数据泄漏（验证集分数虚高但不代表真实泛化能力）。
    raw_windows, raw_seg_ids = [], []
    for seg_id, seg in enumerate(segments):
        wins = sliding_windows(seg, window_size, stride)
        raw_windows.extend(wins)
        raw_seg_ids.extend([seg_id] * len(wins))
    print(f"原始片段滑窗: {len(raw_windows)} 个窗口")

    aug_windows, aug_seg_ids = [], []
    for seg_id, seg in enumerate(segments):
        for v in augment_segment(seg, args.n_aug, rng):
            wins = sliding_windows(v, window_size, stride)
            aug_windows.extend(wins)
            aug_seg_ids.extend([seg_id] * len(wins))

    all_windows = raw_windows + aug_windows
    all_seg_ids = raw_seg_ids + aug_seg_ids
    print(f"增强后总窗口: {len(all_windows)} 个（来自 {len(segments)} 个原始片段）")

    if not all_windows:
        print("[错误] 没有生成任何窗口")
        return

    # 按目标数量随机采样，避免合成数据压过真实数据（seg_id 跟着同步采样，保持对应关系）
    if target_windows > 0 and len(all_windows) > target_windows:
        idx = rng.choice(len(all_windows), size=target_windows, replace=False)
        all_windows = [all_windows[i] for i in idx]
        all_seg_ids = [all_seg_ids[i] for i in idx]
        print(f"采样到目标窗口数: {len(all_windows)} 个（原 {len(raw_windows) + len(aug_windows)} 个）")
    elif target_windows > 0:
        print(f"[提示] 生成窗口数 {len(all_windows)} 少于目标 {target_windows}，可调大 --n_aug")

    X = np.stack(all_windows, axis=0)
    seg_ids = np.array(all_seg_ids, dtype=np.int64)
    print(f"\n输出形状: {X.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, X=X, seg_ids=seg_ids)
    print(f"已保存: {args.output}")

    print(f"\n下一步（注入合成数据训练）:")
    print(f"  python src/ml/train.py --hz {args.hz} --model rf \\")
    print(f"    --processed_dir <your_processed_dir> \\")
    print(f"    --remap configs/remap_custom_3class.yaml \\")
    print(f"    --synthetic {args.output} --synthetic_label '{args.label}'")


if __name__ == "__main__":
    main()
