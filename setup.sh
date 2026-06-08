#!/usr/bin/env bash
# 数据预处理脚本
# 用法:
#   bash setup.sh --dataset a   # 预处理数据集A → data/processed_a/
#   bash setup.sh --dataset b   # 预处理数据集B → data/processed_b/
set -e

DATASET=""
for arg in "$@"; do
    case "$arg" in
        --dataset) shift; DATASET="$1" ;;
        a|b) DATASET="$arg" ;;
    esac
done

# 解析 --dataset 值（支持 --dataset a 或直接传 a）
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--dataset" ]; then
        DATASET="${ARGS[$((i+1))]}"
    fi
done

if [ -z "$DATASET" ]; then
    echo "用法: bash setup.sh --dataset <a|b>"
    echo "  --dataset a  预处理数据集A (Mendeley vxhx934tbn)"
    echo "  --dataset b  预处理数据集B (Mendeley mpph6bmn7g)"
    exit 1
fi

echo "================================================"
echo "  IMU 犬只行为识别 - 数据整理脚本"
echo "  数据集: $DATASET"
echo "================================================"

if [ "$DATASET" = "a" ]; then
    RAW_DIR="data/raw"
    OUT_DIR="data/processed_a"

    # 1. 检查文件
    echo ""
    echo "[1/3] 检查数据集A文件..."
    MISSING=0
    for f in "DogMoveData_csv_format.zip" "DogInfo.csv"; do
        if [ ! -f "$RAW_DIR/$f" ]; then
            echo "  ❌ 缺少: $RAW_DIR/$f"
            echo "  下载地址: https://data.mendeley.com/datasets/vxhx934tbn/2"
            MISSING=1
        else
            echo "  ✅ $f"
        fi
    done
    [ $MISSING -eq 1 ] && exit 1

    # 2. 解压
    echo ""
    echo "[2/3] 解压 DogMoveData_csv_format.zip ..."
    CSV_DIR="$RAW_DIR/csv"
    if [ -d "$CSV_DIR" ] && [ "$(ls -A $CSV_DIR)" ]; then
        echo "  已存在，跳过解压"
    else
        mkdir -p "$CSV_DIR"
        unzip -q "$RAW_DIR/DogMoveData_csv_format.zip" -d "$CSV_DIR"
        echo "  ✅ 解压完成 → $CSV_DIR"
    fi

    # 3. 预处理
    echo ""
    echo "[3/3] 运行预处理..."
    python src/data/preprocess.py \
        --dataset a \
        --raw_csv_dir "$CSV_DIR" \
        --dog_info "$RAW_DIR/DogInfo.csv" \
        --output_dir "$OUT_DIR" \
        --config configs/data.yaml

elif [ "$DATASET" = "b" ]; then
    RAW_B="data/raw_b/df_raw.csv"
    OUT_DIR="data/processed_b"

    # 1. 检查文件
    echo ""
    echo "[1/2] 检查数据集B文件..."
    if [ ! -f "$RAW_B" ]; then
        echo "  ❌ 缺少: $RAW_B"
        echo "  下载地址: https://data.mendeley.com/datasets/mpph6bmn7g/1"
        exit 1
    else
        echo "  ✅ df_raw.csv"
    fi

    # 2. 预处理
    echo ""
    echo "[2/2] 运行预处理..."
    python src/data/preprocess.py \
        --dataset b \
        --raw_csv_b "$RAW_B" \
        --output_dir "$OUT_DIR" \
        --config configs/data.yaml

else
    echo "未知数据集: $DATASET（只支持 a 或 b）"
    exit 1
fi

echo ""
echo "================================================"
echo "  ✅ 数据集${DATASET}整理完成！"
echo ""
echo "  处理结果:"
find "$OUT_DIR" -name "*.npz" 2>/dev/null | sort | sed 's/^/  /'
echo ""
echo "  下一步训练（以50Hz为例）："
echo "    ML: python src/ml/train.py --hz 50 --model xgb --processed_dir $OUT_DIR"
echo "    DL: python src/dl/train.py --hz 50 --model cnn_lstm --processed_dir $OUT_DIR"
echo ""
echo "  多数据集对比（处理完所有数据集后）："
echo "    python src/eval/compare.py --all"
echo "================================================"
