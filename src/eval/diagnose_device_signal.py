"""
诊断"新设备CSV喂给已训练模型，识别效果异常"这类问题的第一步：检查基本的信号量级、
采样率、类别预测分布，把最常见的几个坑（单位不对、采样率传错、列名错配、类别一致性
偏差）一次性排查掉，而不是猜。

用法：
    # 只看信号本身的量级/采样率，不需要模型
    python src/eval/diagnose_device_signal.py --csv data/raw_tf_csv/26080807.csv

    # 同时对比一份已知正常的数据（比如现有 witmotion 采集的CSV），量级差异一眼看出来
    python src/eval/diagnose_device_signal.py \
        --csv data/raw_tf_csv/26080807.csv \
        --compare_csv data/raw_wit/some_known_good.csv

    # 加上模型，看预测类别分布是不是明显异常（比如"抓挠"全程0%但其他类别也集中在
    # 少数几类，往往说明特征量级跟训练分布对不上，而不是这只狗真的没有对应行为）
    python src/eval/diagnose_device_signal.py \
        --csv data/raw_tf_csv/26080807.csv \
        --model results/xxx/ml_rf.pkl --device_hz 50
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # src/，找 infer_csv_scratch
from infer_csv_scratch import load_csv, downsample, sliding_windows  # noqa: E402


def estimate_hz(ts):
    if ts is None or ts.isna().all():
        return None
    dt = ts.diff().dropna().dt.total_seconds()
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    return round(1.0 / dt.median(), 2)


def report_signal(label, acc, gyro, ts):
    hz = estimate_hz(ts)
    mag = np.linalg.norm(acc, axis=1)
    print(f"\n=== {label} ===")
    print(f"  行数: {len(acc)}")
    print(f"  从时间戳估算采样率: {hz}Hz" if hz else "  时间戳缺失/无法估算采样率")
    print(f"  加速度模长(|acc|): mean={mag.mean():.3f}  std={mag.std():.3f}  "
          f"p5={np.percentile(mag,5):.3f}  p50={np.percentile(mag,50):.3f}  p95={np.percentile(mag,95):.3f}")
    print(f"  ── 关键判断：静止/低活动时 |acc| 应该接近重力常数。"
          f"训练数据(docs/datasets.md样本)用的是 m/s²，重力≈9.8；"
          f"如果这里的 p50 接近 1.0 而不是 9.8，大概率是单位是 g，需要 ×9.8 转换 ──")
    print(f"  acc_x: mean={acc[:,0].mean():.3f} std={acc[:,0].std():.3f}  "
          f"min={acc[:,0].min():.3f} max={acc[:,0].max():.3f}")
    print(f"  acc_y: mean={acc[:,1].mean():.3f} std={acc[:,1].std():.3f}  "
          f"min={acc[:,1].min():.3f} max={acc[:,1].max():.3f}")
    print(f"  acc_z: mean={acc[:,2].mean():.3f} std={acc[:,2].std():.3f}  "
          f"min={acc[:,2].min():.3f} max={acc[:,2].max():.3f}")
    if gyro is not None and gyro.any():
        gmag = np.linalg.norm(gyro, axis=1)
        print(f"  角速度模长(|gyro|): mean={gmag.mean():.3f}  std={gmag.std():.3f}  max={gmag.max():.3f}")
    return hz


def report_predictions(csv_path, model_path, device_hz, model_hz):
    import joblib
    import json as _json
    from gravity_align import gravity_align_batch, append_raw_tilt_batch
    from features import extract_features

    model = joblib.load(model_path)
    meta_path = model_path.replace(".pkl", ".json")
    classes, t_hz, window_s, stride_s = [], 16, 2.0, 1.0
    if os.path.exists(meta_path):
        meta = _json.load(open(meta_path))
        classes = meta.get("classes", [])
        t_hz = int(meta.get("hz", 16))
        window_s = float(meta.get("window_s", 2.0))
        stride_s = float(meta.get("stride_s", 1.0))
    else:
        classes = list(model.classes_) if hasattr(model, "classes_") else []

    model_hz = model_hz or t_hz
    device_hz = device_hz or model_hz
    window_size = int(window_s * model_hz)
    stride = int(stride_s * model_hz)

    acc, gyro, ts, valid_mask, null_ratio = load_csv(csv_path)
    acc_ds = downsample(acc, device_hz, model_hz)
    gyro_ds = downsample(gyro, device_hz, model_hz)
    data6 = np.concatenate([acc_ds, gyro_ds], axis=1)
    X, _ = sliding_windows(data6, window_size, stride)
    if len(X) == 0:
        print("  [警告] 数据太短，切不出一个完整窗口")
        return

    tilt = append_raw_tilt_batch(X)[:, :, 6:8]
    X_aligned = gravity_align_batch(X)
    X_aligned = np.concatenate([X_aligned, tilt], axis=2)
    feats = extract_features(X_aligned, model_hz, show_progress=False)
    probs = model.predict_proba(feats)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    print(f"\n  预测类别分布（共{len(preds)}个窗口）：")
    for i, c in enumerate(classes):
        mask = preds == i
        n = mask.sum()
        avg_conf = confs[mask].mean() if n else 0
        print(f"    {c}: {n} ({n/len(preds)*100:.1f}%)  平均置信度={avg_conf:.2f}")
    print(f"  整体平均置信度: {confs.mean():.2f}（如果所有类别置信度都异常低，"
          f"说明特征普遍落在训练分布之外，不是模型正常判断的结果）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--compare_csv", default=None, help="已知正常的CSV，量级对比用")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device_hz", type=int, default=0)
    ap.add_argument("--model_hz", type=int, default=0)
    args = ap.parse_args()

    acc, gyro, ts, valid_mask, null_ratio = load_csv(args.csv)
    hz = report_signal(f"目标文件: {args.csv}", acc, gyro, ts)
    if null_ratio > 0.05:
        print(f"  [警告] 缺失率={null_ratio*100:.1f}%")

    if args.compare_csv:
        acc2, gyro2, ts2, valid_mask2, null_ratio2 = load_csv(args.compare_csv)
        report_signal(f"对比文件(已知正常): {args.compare_csv}", acc2, gyro2, ts2)
        mag1 = np.linalg.norm(acc, axis=1).mean()
        mag2 = np.linalg.norm(acc2, axis=1).mean()
        ratio = mag1 / mag2 if mag2 else float("nan")
        print(f"\n  两份数据 |acc| 均值比例: {ratio:.3f}"
              f"（接近1没问题；接近0.1或10，大概率是单位差了一个数量级）")

    if args.model:
        print(f"\n=== 用模型 {args.model} 跑一遍预测分布 ===")
        report_predictions(args.csv, args.model, args.device_hz, args.model_hz)


if __name__ == "__main__":
    main()
