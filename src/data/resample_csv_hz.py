"""
把 labelstudio_to_custom.py 生成的训练CSV（record_id,label,timestamp,acc_x..gyr_z）从
source_hz 重采样到 target_hz——用于把不同采集批次统一到同一个采样率再合并
训练（比如老数据本来就是16Hz采集的，新数据是50Hz原始的，混合训练前把50Hz
的降采样到16Hz对齐）。

按record_id分组各自重采样，不跨record_id窗口（不同record_id之间时间上本来
就不连续，混在一起重采样会产生跨片段的伪造过渡数据）。

用法:
  python src/data/resample_csv_hz.py \\
    --input data/raw_custom/2026_8_11-2026_8_27_raw/merged_2026_8_11-2026_8_27_raw.csv \\
    --output data/raw_custom/2026_8_11-2026_8_27_raw/merged_2026_8_11-2026_8_27_raw_16hz.csv \\
    --source_hz 50 --target_hz 16
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def resample_df(df, source_hz: int, target_hz: int, method: str = "training_match"):
    """按 record_id 分组把训练 CSV 从 source_hz 降到 target_hz。返回新的 DataFrame。

    **method 必须跟推理用的一致。** label_service 推理默认 RESAMPLE_METHOD=
    training_match（滑动平均低通 + 线性插值），这里就得用同一个；用 poly 的话
    两边输出差 6~8%（见 resample_training_match.py 顶部）——模型在训练时见到的
    信号跟推理时喂给它的不是一个样子。

    labels：每个新采样点取**时间上最近**的原始点的标签。
    """
    import pandas as pd

    if source_hz == target_hz:
        return df.drop(columns=["timestamp"], errors="ignore")
    out_rows = []
    for rid, g in df.groupby("record_id", sort=False):
        data = g[SENSOR_COLS].to_numpy(dtype=np.float64)
        labels = g["label"].to_numpy()
        if method == "training_match":
            from resample_training_match import resample_training_match

            data_ds = resample_training_match(data, source_hz, target_hz)
            # 跟 resample_training_match 同一套时间轴：原始第 i 点在 i/source_hz 秒，
            # 新第 k 点在 k/target_hz 秒
            idx = np.rint(np.arange(len(data_ds)) * (source_hz / target_hz)).astype(int)
            labels_ds = labels[np.clip(idx, 0, len(labels) - 1)]
        else:
            from preprocess import downsample  # 要 sklearn，只有 poly 用得到
            data_ds, labels_ds = downsample(data, labels, source_hz, target_hz)
        out = pd.DataFrame(data_ds, columns=SENSOR_COLS)
        out.insert(0, "label", labels_ds)
        out.insert(0, "record_id", rid)
        out_rows.append(out)
    if not out_rows:
        return df.iloc[0:0].drop(columns=["timestamp"], errors="ignore")
    return pd.concat(out_rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source_hz", type=int, required=True)
    ap.add_argument("--target_hz", type=int, required=True)
    ap.add_argument("--method", default="training_match", choices=["training_match", "poly"],
                    help="降采样算法，必须跟推理一致（label_service 默认 training_match）")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in SENSOR_COLS + ["record_id", "label"] if c not in df.columns]
    if missing:
        print(f"[resample_csv_hz] 输入CSV缺少列: {missing}")
        sys.exit(1)

    if args.source_hz == args.target_hz:
        print(f"[resample_csv_hz] source_hz == target_hz == {args.source_hz}，原样复制")
        df.to_csv(args.output, index=False)
        return

    result = resample_df(df, args.source_hz, args.target_hz, args.method)
    result.to_csv(args.output, index=False)
    print(f"[resample_csv_hz] {args.source_hz}Hz → {args.target_hz}Hz 完成: "
          f"{len(df)}行 → {len(result)}行，写入 {args.output}")


if __name__ == "__main__":
    main()
