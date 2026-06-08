#!/usr/bin/env bash
# 一键数据整理脚本
# 用法:
#   bash setup.sh           # 只用数据集A
#   bash setup.sh --merge   # 合并数据集A + B（需先下载 data/raw_b/df_raw.csv）
set -e

RAW_DIR="data/raw"
RAW_B="data/raw_b/df_raw.csv"
PROCESSED_DIR="data/processed"
MERGE=0

for arg in "$@"; do
    [ "$arg" = "--merge" ] && MERGE=1
done

echo "================================================"
echo "  IMU 犬只行为识别 - 数据整理脚本"
[ $MERGE -eq 1 ] && echo "  模式: 合并数据集A + B" || echo "  模式: 单数据集A"
echo "================================================"

# 1. 检查数据集A文件
echo ""
echo "[1/4] 检查数据集A文件..."
MISSING=0
for f in "DogMoveData_csv_format.zip" "DogInfo.csv"; do
    if [ ! -f "$RAW_DIR/$f" ]; then
        echo "  ❌ 缺少: $RAW_DIR/$f"
        MISSING=1
    else
        echo "  ✅ $f"
    fi
done
if [ $MISSING -eq 1 ]; then
    echo "  下载地址: https://data.mendeley.com/datasets/vxhx934tbn/2"
    exit 1
fi

# 检查数据集B（合并模式）
if [ $MERGE -eq 1 ]; then
    if [ ! -f "$RAW_B" ]; then
        echo "  ❌ 合并模式缺少: $RAW_B"
        echo "  下载地址: https://data.mendeley.com/datasets/mpph6bmn7g/1"
        exit 1
    else
        echo "  ✅ df_raw.csv (数据集B)"
    fi
fi

# 2. 解压数据集A
echo ""
echo "[2/4] 解压 DogMoveData_csv_format.zip ..."
CSV_DIR="$RAW_DIR/csv"
if [ -d "$CSV_DIR" ] && [ "$(ls -A $CSV_DIR)" ]; then
    echo "  已存在，跳过解压"
else
    mkdir -p "$CSV_DIR"
    unzip -q "$RAW_DIR/DogMoveData_csv_format.zip" -d "$CSV_DIR"
    echo "  ✅ 解压完成 → $CSV_DIR"
fi

# 3. 探查数据结构
echo ""
echo "[3/4] 探查数据文件结构..."
FIRST_CSV=$(find "$CSV_DIR" -name "*.csv" | head -1)
echo "  数据集A 列名预览:"
head -1 "$FIRST_CSV" | tr ',' '\n' | nl
echo "  行数: $(wc -l < "$FIRST_CSV") (含表头)"
if [ $MERGE -eq 1 ]; then
    echo ""
    echo "  数据集B 列名预览:"
    head -1 "$RAW_B" | tr ',' '\n' | nl
    echo "  行数: $(wc -l < "$RAW_B") (含表头)"
fi

# 4. 运行预处理
echo ""
echo "[4/4] 运行预处理..."

if [ $MERGE -eq 1 ]; then
    python src/data/preprocess.py \
        --raw_csv_dir "$CSV_DIR" \
        --dog_info "$RAW_DIR/DogInfo.csv" \
        --raw_csv_b "$RAW_B" \
        --output_dir "$PROCESSED_DIR" \
        --config configs/data.yaml
    OUT_DIR="${PROCESSED_DIR}_merged"
else
    python src/data/preprocess.py \
        --raw_csv_dir "$CSV_DIR" \
        --dog_info "$RAW_DIR/DogInfo.csv" \
        --output_dir "$PROCESSED_DIR" \
        --config configs/data.yaml
    OUT_DIR="$PROCESSED_DIR"
fi

echo ""
echo "================================================"
echo "  ✅ 数据整理完成！"
echo ""
echo "  处理结果:"
find "$OUT_DIR" -name "*.npz" 2>/dev/null | sort | sed 's/^/  /'
echo ""
echo "  下一步："
if [ $MERGE -eq 1 ]; then
    echo "    训练 ML: python src/ml/train.py --hz 50 --processed_dir data/processed_merged"
    echo "    训练 DL: python src/dl/train.py --hz 50 --model cnn_lstm --processed_dir data/processed_merged"
    echo "    全量对比: python src/eval/compare.py --results_dir results/processed_merged"
else
    echo "    训练 ML: python src/ml/train.py --hz 50"
    echo "    训练 DL: python src/dl/train.py --hz 50 --model cnn_lstm"
    echo "    全量对比: python src/eval/compare.py"
fi
echo "    两套结果同时对比: python src/eval/compare.py --all"
echo "================================================"
