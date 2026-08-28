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
from preprocess import downsample  # noqa: E402

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source_hz", type=int, required=True)
    ap.add_argument("--target_hz", type=int, required=True)
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

    out_rows = []
    n_groups = df["record_id"].nunique()
    for i, (rid, g) in enumerate(df.groupby("record_id", sort=False)):
        data = g[SENSOR_COLS].to_numpy(dtype=np.float64)
        labels = g["label"].to_numpy()
        data_ds, labels_ds = downsample(data, labels, args.source_hz, args.target_hz)
        out = pd.DataFrame(data_ds, columns=SENSOR_COLS)
        out.insert(0, "label", labels_ds)
        out.insert(0, "record_id", rid)
        out_rows.append(out)
        if (i + 1) % 50 == 0 or (i + 1) == n_groups:
            print(f"[resample_csv_hz] {i + 1}/{n_groups} 个record_id重采样完成")

    result = pd.concat(out_rows, ignore_index=True)
    result.to_csv(args.output, index=False)
    print(f"[resample_csv_hz] {args.source_hz}Hz → {args.target_hz}Hz 完成: "
          f"{len(df)}行 → {len(result)}行，写入 {args.output}")


if __name__ == "__main__":
    main()
