"""
推理脚本：对无标签的 TXT/CSV 文件逐帧预测行为，输出 CSV 结果。

数据格式（与数据集A项圈传感器一致）:
  HH:MM:SS.MS,AX,AY,AZ,GX,GY,GZ
  16:00:00.000,0.137,1.149,8.603,...

用法示例:
  # 用 ML 模型推理
  python src/infer.py --model_type ml --model_path results/processed_a/50hz/ml_xgb.pkl \
      --processed_dir data/processed_a --hz 50 --input_dir data/infer

  # 用 DL 模型推理
  python src/infer.py --model_type dl --model_name cnn_lstm \
      --model_path results/processed_a/50hz/dl_cnn_lstm_best.pt \
      --processed_dir data/processed_a --hz 50 --input_dir data/infer

  # 指定单个文件
  python src/infer.py --model_type ml --model_path results/processed_a/50hz/ml_xgb.pkl \
      --processed_dir data/processed_a --hz 50 --input_file data/infer/26060316.TXT
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import glob
import json

import numpy as np
import pandas as pd
import yaml
from gravity_align import gravity_align_batch

SOURCE_HZ = 50   # TXT 文件采样率
SENSOR_COLS = ["AX", "AY", "AZ", "GX", "GY", "GZ"]
# 兼容 timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z 格式
ALT_COLS    = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

BEHAVIOR_ZH = {
    "Lying chest": "趴卧",
    "Sitting":     "坐",
    "Sniffing":    "嗅闻",
    "Standing":    "站立",
    "Trotting":    "小跑",
    "Walking":     "行走",
    "Unknown":     "未知",
}


# ── 数据读取 ──────────────────────────────────────────────────────────────────

def load_txt(path: str) -> tuple[np.ndarray, int, pd.Timestamp | None]:
    """读取单个 TXT/CSV 文件，返回 (N, 6) float32 传感器数据、采样率、起始时间戳。"""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if all(c in df.columns for c in SENSOR_COLS):
        data = df[SENSOR_COLS].values.astype(np.float32)
    elif all(c in df.columns for c in ALT_COLS):
        data = df[ALT_COLS].values.astype(np.float32)
    else:
        raise ValueError(f"{path}: 缺少传感器列，现有列: {list(df.columns)}")
    # 自动检测采样率和起始时间戳
    detected_hz = SOURCE_HZ
    start_ts = None
    ts_col = next((c for c in df.columns if "time" in c.lower()), None)
    if ts_col and len(df) > 10:
        try:
            ts = pd.to_datetime(df[ts_col])
            start_ts = ts.iloc[0]
            median_interval = (ts.diff().dropna().dt.total_seconds()).median()
            if median_interval > 0:
                detected_hz = max(1, round(1.0 / median_interval))
        except Exception:
            pass
    return data, detected_hz, start_ts


def collect_files(input_dir: str = None, input_file: str = None, input_url: str = None) -> list[tuple[str, str]]:
    """返回 [(实际路径, 显示文件名), ...]"""
    if input_url:
        import tempfile, urllib.request
        from urllib.parse import urlparse
        print(f"[infer] 下载: {input_url}")
        url_fname = os.path.basename(urlparse(input_url).path) or "download.csv"
        suffix = ".csv" if input_url.lower().endswith(".csv") else ".txt"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        urllib.request.urlretrieve(input_url, tmp.name)
        print(f"[infer] 已保存到临时文件: {tmp.name}")
        return [(tmp.name, url_fname)]
    if input_file:
        return [(input_file, os.path.basename(input_file))]
    exts = ("*.TXT", "*.txt", "*.CSV", "*.csv")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(input_dir, ext))
    files = sorted(set(files))
    files = [f for f in files if os.path.basename(f).upper() not in ("README.TXT", "README.CSV")]
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 下未找到 TXT/CSV 文件")
    return [(f, os.path.basename(f)) for f in files]


# ── 预处理 ────────────────────────────────────────────────────────────────────

def downsample(data: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    step = source_hz // target_hz
    return data[::step]


def sliding_window_infer(data: np.ndarray, window_size: int, stride: int):
    """返回 (windows, start_indices)，windows shape = (N, window_size, 6)。"""
    windows, starts = [], []
    n = len(data)
    for start in range(0, n - window_size + 1, stride):
        windows.append(data[start:start + window_size])
        starts.append(start)
    if not windows:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32), []
    return np.array(windows, dtype=np.float32), starts


# ── 模型推理 ──────────────────────────────────────────────────────────────────

def predict_ml(model, X_windows: np.ndarray, hz: int) -> tuple[np.ndarray, np.ndarray]:
    from features import extract_features
    X_feat = extract_features(X_windows, hz)
    preds = np.array(model.predict(X_feat)).flatten().astype(int)
    proba = model.predict_proba(X_feat)          # (N, n_classes)
    confidence = proba.max(axis=1)
    return preds, confidence


def predict_dl(model, X_windows: np.ndarray, device, m2m: bool, cfg_dl) -> tuple[np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F
    from dl.train import m2m_predict

    model.eval()
    batch_size = cfg_dl.get("batch_size", 64)
    all_preds, all_conf = [], []
    X_t = torch.from_numpy(X_windows).float().permute(0, 2, 1)  # (N, C, T)
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(device)
            logits = model(xb)
            if m2m:
                # logits: (B, C, T) → pool over T → softmax
                pooled = logits.mean(dim=2)          # (B, C)
                probs = F.softmax(pooled, dim=1)
                preds = m2m_predict(logits).cpu().numpy()
            else:
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1).cpu().numpy()
            conf = probs.max(dim=1).values.cpu().numpy()
            all_preds.extend(preds)
            all_conf.extend(conf)
    return np.array(all_preds), np.array(all_conf)


# ── 加载元数据（类别名、窗口参数）────────────────────────────────────────────

def load_meta(processed_dir: str, hz: int) -> dict:
    import numpy as np
    npz_path = os.path.join(processed_dir, f"{hz}hz", "train.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到 {npz_path}，请先运行 bash setup.sh")
    data = np.load(npz_path, allow_pickle=True)
    meta = {k: data[k].item() if data[k].ndim == 0 else data[k].tolist()
            for k in data.files if k not in ("X", "y", "y_seq")}
    return meta


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main(args):
    meta = load_meta(args.processed_dir, args.hz)
    classes = eval(meta["classes"]) if isinstance(meta["classes"], str) else meta["classes"]
    window_size = int(meta["window_size"])
    stride = int(meta["stride"])

    # 重力对齐：优先用命令行参数，否则跟随训练时的设置
    trained_with_ga = str(meta.get("gravity_aligned", "True")).lower() == "true"
    if args.gravity_align is None:
        use_gravity_align = trained_with_ga
    else:
        use_gravity_align = args.gravity_align
        if use_gravity_align != trained_with_ga:
            print(f"[infer] ⚠️  警告：训练时重力对齐={'是' if trained_with_ga else '否'}，"
                  f"当前设置={'是' if use_gravity_align else '否'}，可能影响精度")

    print(f"[infer] hz={args.hz}, window={window_size}帧, stride={stride}帧")
    print(f"[infer] 类别: {classes}")
    print(f"[infer] 重力对齐: {'启用' if use_gravity_align else '禁用'}")

    # 加载模型
    if args.model_type == "ml":
        import joblib
        model = joblib.load(args.model_path)
        print(f"[infer] 已加载 ML 模型: {args.model_path}")
        predict_fn = lambda X: predict_ml(model, X, args.hz)
        print(f"[infer] 置信度阈值: {args.confidence_threshold}  (低于此值标记为 Unknown)")

    else:  # dl
        import torch
        from dl.train import load_model, M2M_MODELS
        with open(args.dl_config) as f:
            cfg_dl = yaml.safe_load(f)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_channels = int(meta.get("n_channels", 6))
        n_classes = len(classes)
        m2m = args.model_name in M2M_MODELS
        model = load_model(args.model_name, n_channels, window_size, n_classes, cfg_dl).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"[infer] 已加载 DL 模型: {args.model_path}  device={device}")
        print(f"[infer] 置信度阈值: {args.confidence_threshold}  (低于此值标记为 Unknown)")
        predict_fn = lambda X: predict_dl(model, X, device, m2m, cfg_dl)

    # 收集文件
    files = collect_files(args.input_dir, args.input_file, args.input_url)
    print(f"[infer] 共 {len(files)} 个文件待推理")

    all_results = []

    for fpath, fname in files:
        try:
            raw, file_hz, start_ts = load_txt(fpath)
        except Exception as e:
            print(f"  [跳过] {fname}: {e}")
            continue

        if file_hz != SOURCE_HZ:
            print(f"  [info] {fname}: 检测到采样率={file_hz}Hz")
        src_hz = file_hz
        if src_hz < args.hz:
            print(f"  [warn] {fname}: 文件采样率({src_hz}Hz) < 目标({args.hz}Hz)，跳过降采样直接使用")
            data_ds = raw
        else:
            data_ds = downsample(raw, src_hz, args.hz)
        windows, starts = sliding_window_infer(data_ds, window_size, stride)

        if len(windows) == 0:
            print(f"  [跳过] {fname}: 数据太短，无法生成窗口（需至少 {window_size} 帧）")
            continue

        if use_gravity_align:
            windows = gravity_align_batch(windows)

        preds, confidences = predict_fn(windows)
        thresh = args.confidence_threshold
        pred_labels = [
            classes[p] if conf >= thresh else "Unknown"
            for p, conf in zip(preds, confidences)
        ]

        time_starts = [s / args.hz for s in starts]

        import datetime
        for i, (t, label, conf) in enumerate(zip(time_starts, pred_labels, confidences)):
            t_end = t + window_size / args.hz
            row = {
                "file": fname,
                "window_idx": i,
                "time_start_s": round(t, 3),
                "time_end_s": round(t_end, 3),
                "prediction": label,
                "prediction_zh": BEHAVIOR_ZH.get(label, label),
                "confidence": round(float(conf), 4),
            }
            if start_ts is not None:
                row["abs_start"] = (start_ts + pd.Timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                row["abs_end"]   = (start_ts + pd.Timedelta(seconds=t_end)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            all_results.append(row)

        # 统计（Unknown 单独计）
        from collections import Counter
        counts = Counter(pred_labels)
        total = len(pred_labels)
        n_unknown = counts.pop("Unknown", 0)
        summary = "  ".join(f"{k}:{v/total:.0%}" for k, v in counts.most_common())
        unknown_str = f"  Unknown:{n_unknown/total:.0%}" if n_unknown else ""
        print(f"  {fname}: {total} 窗口  [{summary}{unknown_str}]")

    if not all_results:
        print("[infer] 无有效结果")
        return

    out_df = pd.DataFrame(all_results)
    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = os.path.splitext(os.path.basename(args.model_path))[0]
    if len(files) == 1:
        src_stem = os.path.splitext(files[0][1])[0]
        out_path        = os.path.join(args.output_dir, f"{src_stem}_{model_tag}.csv")
        out_path_merged = os.path.join(args.output_dir, f"{src_stem}_{model_tag}_segments.csv")
    else:
        out_path        = os.path.join(args.output_dir, f"predictions_{model_tag}.csv")
        out_path_merged = os.path.join(args.output_dir, f"predictions_{model_tag}_segments.csv")

    out_df.to_csv(out_path, index=False)

    # 合并连续相同行为为时间段
    merged = []
    for _, grp in out_df.groupby("file", sort=False):
        prev = None
        for row in grp.itertuples():
            label = row.prediction
            if prev is None or label != prev["prediction"]:
                if prev:
                    merged.append(prev)
                prev = {
                    "file": row.file,
                    "prediction": label,
                    "prediction_zh": row.prediction_zh,
                    "time_start_s": row.time_start_s,
                    "time_end_s": row.time_end_s,
                    "abs_start": getattr(row, "abs_start", ""),
                    "abs_end":   getattr(row, "abs_end",   ""),
                }
            else:
                prev["time_end_s"] = row.time_end_s
                if hasattr(row, "abs_end"):
                    prev["abs_end"] = row.abs_end
        if prev:
            merged.append(prev)

    merged_df = pd.DataFrame(merged)
    merged_df["duration_s"] = (merged_df["time_end_s"] - merged_df["time_start_s"]).round(1)
    merged_df.to_csv(out_path_merged, index=False)

    print(f"\n[infer] 明细已保存至 {out_path}  ({len(out_df)} 行)")
    print(f"[infer] 时间段已保存至 {out_path_merged}  ({len(merged_df)} 段)")
    print()
    for row in merged_df.itertuples():
        t = f"{row.abs_start} → {row.abs_end}" if row.abs_start else f"{row.time_start_s}s → {row.time_end_s}s"
        print(f"  {t}  {row.prediction_zh}({row.prediction})  {row.duration_s}秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["ml", "dl"])
    parser.add_argument("--model_path", required=True,
                        help="ML: .pkl 路径；DL: .pt 路径")
    parser.add_argument("--model_name", default="cnn_lstm",
                        help="DL 模型名称（仅 --model_type dl 时需要）",
                        choices=["cnn", "collar_cnn", "cnn_lstm", "transformer",
                                 "filternet", "filternet_m2m"])
    parser.add_argument("--processed_dir", required=True,
                        help="对应数据集的预处理目录，用于读取类别和窗口参数")
    parser.add_argument("--hz", type=int, required=True, choices=[5, 10, 25, 50])
    parser.add_argument("--input_dir", default="data/infer",
                        help="存放待推理 TXT/CSV 文件的目录")
    parser.add_argument("--input_file", default="",
                        help="单个文件路径（与 --input_dir 二选一）")
    parser.add_argument("--input_url", default="",
                        help="CSV/TXT 文件的 URL，自动下载后推理")
    parser.add_argument("--output_dir", default="results/infer")
    parser.add_argument("--dl_config", default="configs/dl.yaml")
    parser.add_argument("--confidence_threshold", type=float, default=0.6,
                        help="置信度阈值，低于此值标记为 Unknown（默认 0.6，设为 0 禁用）")
    ga_group = parser.add_mutually_exclusive_group()
    ga_group.add_argument("--gravity_align", dest="gravity_align", action="store_true", default=None,
                          help="强制启用重力轴对齐（默认跟随训练时设置）")
    ga_group.add_argument("--no_gravity_align", dest="gravity_align", action="store_false",
                          help="强制禁用重力轴对齐")
    args = parser.parse_args()
    if args.input_url:
        args.input_dir = None
        args.input_file = ""
    elif args.input_file:
        args.input_dir = None
    main(args)
