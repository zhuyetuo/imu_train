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


# ── 数据读取 ──────────────────────────────────────────────────────────────────

def load_txt(path: str) -> np.ndarray:
    """读取单个 TXT 文件，返回 (N, 6) float32 传感器数据。"""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 缺少列 {missing}，现有列: {list(df.columns)}")
    return df[SENSOR_COLS].values.astype(np.float32)


def collect_files(input_dir: str = None, input_file: str = None) -> list[str]:
    if input_file:
        return [input_file]
    exts = ("*.TXT", "*.txt", "*.CSV", "*.csv")
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(input_dir, ext))
    files = sorted(set(files))
    # 跳过非数据文件（README、说明文件等）
    files = [f for f in files if os.path.basename(f).upper() not in ("README.TXT", "README.CSV")]
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 下未找到 TXT/CSV 文件")
    return files


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
    print(f"[infer] hz={args.hz}, window={window_size}帧, stride={stride}帧")
    print(f"[infer] 类别: {classes}")

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
    files = collect_files(args.input_dir, args.input_file)
    print(f"[infer] 共 {len(files)} 个文件待推理")

    all_results = []

    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            raw = load_txt(fpath)
        except Exception as e:
            print(f"  [跳过] {fname}: {e}")
            continue

        data_ds = downsample(raw, SOURCE_HZ, args.hz)
        windows, starts = sliding_window_infer(data_ds, window_size, stride)

        if len(windows) == 0:
            print(f"  [跳过] {fname}: 数据太短，无法生成窗口（需至少 {window_size} 帧）")
            continue

        windows = gravity_align_batch(windows)

        preds, confidences = predict_fn(windows)
        thresh = args.confidence_threshold
        pred_labels = [
            classes[p] if conf >= thresh else "Unknown"
            for p, conf in zip(preds, confidences)
        ]

        # 每个窗口的时间起点（秒）
        time_starts = [s / args.hz for s in starts]

        for i, (t, label, conf) in enumerate(zip(time_starts, pred_labels, confidences)):
            all_results.append({
                "file": fname,
                "window_idx": i,
                "time_start_s": round(t, 3),
                "time_end_s": round(t + window_size / args.hz, 3),
                "prediction": label,
                "confidence": round(float(conf), 4),
            })

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
    out_path = os.path.join(args.output_dir, f"predictions_{model_tag}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[infer] 结果已保存至 {out_path}  ({len(out_df)} 行)")


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
    parser.add_argument("--output_dir", default="results/infer")
    parser.add_argument("--dl_config", default="configs/dl.yaml")
    parser.add_argument("--confidence_threshold", type=float, default=0.6,
                        help="置信度阈值，低于此值标记为 Unknown（默认 0.6）")
    args = parser.parse_args()
    if args.input_file:
        args.input_dir = None
    main(args)
