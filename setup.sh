#!/usr/bin/env bash
# 一键数据整理脚本
# 使用方法：bash setup.sh
set -e

RAW_DIR="data/raw"
PROCESSED_DIR="data/processed"

echo "================================================"
echo "  IMU 犬只行为识别 - 数据整理脚本"
echo "================================================"

# 1. 检查文件是否存在
echo ""
echo "[1/4] 检查原始数据文件..."

MISSING=0
for f in "DogMoveData_csv_format.zip" "DogInfo.csv"; do
    if [ ! -f "$RAW_DIR/$f" ]; then
        echo "  ❌ 缺少文件: $RAW_DIR/$f"
        MISSING=1
    else
        echo "  ✅ $f"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "请将以下文件下载后放到 $RAW_DIR/ 目录："
    echo "  - DogMoveData_csv_format.zip  (421 MB)"
    echo "  - DogInfo.csv                 (1.38 KB)"
    echo ""
    echo "下载地址: https://data.mendeley.com/datasets/vxhx934tbn/2"
    exit 1
fi

# 2. 解压 CSV 数据
echo ""
echo "[2/4] 解压 DogMoveData_csv_format.zip ..."

CSV_DIR="$RAW_DIR/csv"
if [ -d "$CSV_DIR" ] && [ "$(ls -A $CSV_DIR)" ]; then
    echo "  已存在解压目录，跳过解压"
else
    mkdir -p "$CSV_DIR"
    unzip -q "$RAW_DIR/DogMoveData_csv_format.zip" -d "$CSV_DIR"
    echo "  ✅ 解压完成 → $CSV_DIR"
fi

# 3. 探查数据结构
echo ""
echo "[3/4] 探查数据文件结构..."
FIRST_CSV=$(find "$CSV_DIR" -name "*.csv" | head -1)
if [ -z "$FIRST_CSV" ]; then
    echo "  ❌ 未找到 CSV 文件，请检查压缩包内容"
    exit 1
fi
echo "  首个 CSV 文件: $FIRST_CSV"
echo "  列名预览:"
head -1 "$FIRST_CSV" | tr ',' '\n' | nl
echo ""
echo "  行数: $(wc -l < "$FIRST_CSV") (含表头)"
echo ""
echo "  全部 CSV 文件数: $(find "$CSV_DIR" -name "*.csv" | wc -l)"

# 4. 运行预处理
echo ""
echo "[4/4] 运行预处理（降采样 + 窗口切分）..."

python src/data/preprocess.py \
    --raw_csv_dir "$CSV_DIR" \
    --dog_info "$RAW_DIR/DogInfo.csv" \
    --output_dir "$PROCESSED_DIR" \
    --config configs/data.yaml

echo ""
echo "================================================"
echo "  ✅ 数据整理完成！"
echo ""
echo "  处理结果目录结构:"
find "$PROCESSED_DIR" -name "*.npz" | sort | sed 's/^/  /'
echo ""
echo "  下一步："
echo "    训练 ML: python src/ml/train.py --hz 50"
echo "    训练 DL: python src/dl/train.py --hz 50 --model cnn_lstm"
echo "    全量对比: python src/eval/compare.py"
echo "================================================"
