#!/usr/bin/env Rscript
# 将 Smit 2023 猫咪数据集转换为本项目标准 CSV
# 用法: Rscript src/data/convert_smit2023.R
#   可选参数: --accel <path> --annot <path> --out <path>

args <- commandArgs(trailingOnly=TRUE)

# 默认路径
accel_path <- "data/raw_cat_smit2023/accel_data.RDATA"
annot_path  <- "data/raw_cat_smit2023/anno_data.RDATA"
out_path    <- "data/raw_cat_smit2023/cat_smit2023.csv"

for (i in seq_along(args)) {
  if (args[i] == "--accel") accel_path <- args[i+1]
  if (args[i] == "--annot") annot_path  <- args[i+1]
  if (args[i] == "--out")   out_path    <- args[i+1]
}

cat("[convert] 加载数据...\n")
load(accel_path)   # -> longaccl
load(annot_path)   # -> longanno

# ── 1. 筛选项圈位置 ────────────────────────────────────────────────────────────
cat("[convert] 筛选 Collar 位置...\n")
collar <- longaccl[longaccl$Position == "Collar", ]
cat("  Collar 行数:", nrow(collar), "\n")

# 只保留三轴均值（作为传感器通道）
collar <- collar[, c("Cat_id", "Timestamp", "X_Mean", "Y_Mean", "Z_Mean")]
colnames(collar) <- c("cat_id", "timestamp", "acc_x", "acc_y", "acc_z")

# ── 2. one-hot 标注 → 单列标签 ──────────────────────────────────────────────────
cat("[convert] 转换标签...\n")

# 排除非行为列
non_behavior <- c("Timestamp", "Pen", "Cat_id",
                  "Other_ActigraphOff", "Other_Outofsight",
                  "Other_Other", "Other_Start", "Other_Social.Human",
                  "Other_Social.Allogrooming")
behavior_cols <- setdiff(colnames(longanno), non_behavior)

anno <- longanno[, c("Cat_id", "Timestamp", behavior_cols)]

# 找每行值为1的列名作为标签（多标签时取第一个）
get_label <- function(row) {
  hits <- behavior_cols[row[behavior_cols] == 1]
  if (length(hits) == 0) return(NA)
  hits[1]
}
cat("  计算标签（可能需要几分钟）...\n")
anno$behavior <- apply(anno, 1, get_label)
anno <- anno[!is.na(anno$behavior), c("Cat_id", "Timestamp", "behavior")]
colnames(anno) <- c("cat_id", "timestamp", "behavior")
cat("  有效标注行数:", nrow(anno), "\n")
cat("  行为类别:", length(unique(anno$behavior)), "\n")

# ── 3. 合并 ────────────────────────────────────────────────────────────────────
cat("[convert] 合并...\n")
merged <- merge(collar, anno, by=c("cat_id", "timestamp"), all=FALSE)
merged <- merged[!is.na(merged$behavior), ]
cat("  合并后行数:", nrow(merged), "\n")

# ── 4. 输出 ────────────────────────────────────────────────────────────────────
dir.create(dirname(out_path), showWarnings=FALSE, recursive=TRUE)
write.csv(merged[, c("cat_id", "acc_x", "acc_y", "acc_z", "behavior")],
          out_path, row.names=FALSE)

cat("[convert] ✅ 已保存至", out_path, "\n")
cat("行为分布:\n")
print(sort(table(merged$behavior), decreasing=TRUE))
cat("\n下一步:\n")
cat("  bash setup.sh --dataset cat_smit2023\n")
