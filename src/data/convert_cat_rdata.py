"""
将猫咪数据集的 R dataframe (.RDATA) 转换为本项目可用的 CSV 格式。

支持:
  Smit 2023  — figshare.com/articles/dataset/23605842
  Smit 2024  — figshare.com/articles/dataset/24848292

用法:
  python src/data/convert_cat_rdata.py \\
      --rdata   data/raw_cat_smit2023/accel_data.RDATA \\
      --annot   data/raw_cat_smit2023/anno_data.RDATA \\
      --out     data/raw_cat_smit2023/cat_smit2023.csv \\
      --dataset smit2023

依赖: pip install rpy2
如果无法安装 rpy2（Windows/无 R 环境），可在 R 中手动导出：
  load("accel_data.RDATA")
  load("anno_data.RDATA")
  write.csv(accel_data, "accel_data.csv", row.names=FALSE)
  write.csv(anno_data,  "anno_data.csv",  row.names=FALSE)
然后用 --csv_accel / --csv_annot 参数代替 --rdata / --annot。
"""

import argparse
import pandas as pd
import numpy as np
import os


# ── Smit 2023 列名映射（基于论文描述推断，下载后请核对）──────────────────────
# 论文描述: ActiGraph GT9X, 1秒 epoch, 项圈 + 胸背带
# accel_data 列: 动物ID、时间戳、三轴加速度统计量
# anno_data 列: 动物ID、时间戳、行为标签
SMIT2023_CONFIG = {
    "cat_id_col":  "animal_id",    # 请下载后核对实际列名
    "time_col":    "timestamp",
    "label_col":   "behaviour",
    "sensor_cols": ["acc_x", "acc_y", "acc_z"],  # 请下载后核对
}

# ── Smit 2024 家庭环境列名映射 ────────────────────────────────────────────────
SMIT2024_CONFIG = {
    "cat_id_col":  "cat_id",
    "time_col":    "timestamp",
    "label_col":   "behavior",
    "sensor_cols": ["acc_x", "acc_y", "acc_z"],
}


def load_rdata(path: str, varname: str = None) -> pd.DataFrame:
    """用 rpy2 读取 .RDATA 文件，返回指定变量名的 dataframe。"""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()

        ro.r(f'load("{path}")')
        if varname is None:
            varname = str(ro.r('ls()')[0])
            print(f"  自动选择变量: {varname}")
        return ro.r[varname]
    except ImportError:
        raise ImportError(
            "需要安装 rpy2: pip install rpy2\n"
            "或在 R 中手动 write.csv() 后用 --csv_accel / --csv_annot 参数"
        )


def merge_smit(accel_df: pd.DataFrame, annot_df: pd.DataFrame,
               cfg: dict) -> pd.DataFrame:
    """按动物ID + 时间戳合并加速度和标签，输出标准格式 CSV。"""
    cat_id  = cfg["cat_id_col"]
    time_c  = cfg["time_col"]
    label_c = cfg["label_col"]

    print(f"  加速度数据列: {list(accel_df.columns)}")
    print(f"  标注数据列:   {list(annot_df.columns)}")

    merged = pd.merge(accel_df, annot_df[[cat_id, time_c, label_c]],
                      on=[cat_id, time_c], how="inner")
    print(f"  合并后行数: {len(merged)}")

    # 重命名为标准列名
    rename = {cat_id: "cat_id", label_c: "behavior"}
    merged = merged.rename(columns=rename)
    return merged


def main(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cfg = SMIT2023_CONFIG if args.dataset == "smit2023" else SMIT2024_CONFIG

    if args.csv_accel and args.csv_annot:
        print("[convert] 从 CSV 读取...")
        accel_df = pd.read_csv(args.csv_accel)
        annot_df = pd.read_csv(args.csv_annot)
    else:
        print(f"[convert] 从 RDATA 读取: {args.rdata}")
        accel_df = load_rdata(args.rdata)
        annot_df = load_rdata(args.annot)

    merged = merge_smit(accel_df, annot_df, cfg)
    merged.to_csv(args.out, index=False)
    print(f"[convert] ✅ 已保存至 {args.out}")
    print(f"  列: {list(merged.columns)}")
    print(f"  行为分布:\n{merged['behavior'].value_counts().to_string()}")
    print(f"\n下一步: 更新 configs/data.yaml 中 cat_smit2023.sensor_cols 为实际列名，然后运行:")
    print(f"  bash setup.sh --dataset cat_smit2023")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["smit2023", "smit2024"], required=True)
    parser.add_argument("--rdata",    default="", help=".RDATA 加速度文件")
    parser.add_argument("--annot",    default="", help=".RDATA 标注文件")
    parser.add_argument("--csv_accel", default="", help="已手动导出的加速度 CSV")
    parser.add_argument("--csv_annot", default="", help="已手动导出的标注 CSV")
    parser.add_argument("--out", required=True, help="输出 CSV 路径")
    main(parser.parse_args())
