#!/usr/bin/env bash
# 数据预处理脚本
# 用法:
#   bash setup.sh --dataset a   # 预处理数据集A → data/processed_a/
#   bash setup.sh --dataset b   # 预处理数据集B → data/processed_b/
set -e

DATASET=""
NO_GRAVITY_ALIGN=""
for arg in "$@"; do
    case "$arg" in
        --dataset) shift; DATASET="$1" ;;
        a|b) DATASET="$arg" ;;
        --no_gravity_align) NO_GRAVITY_ALIGN="--no_gravity_align" ;;
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
    echo "用法: bash setup.sh --dataset <数据集>"
    echo "  --dataset a              预处理数据集A (Mendeley vxhx934tbn，犬)"
    echo "  --dataset b              预处理数据集B (Mendeley mpph6bmn7g，犬)"
    echo "  --dataset custom         预处理自采数据集"
    echo "  --dataset cat_smit2023   预处理猫咪数据集 Smit 2023 (FigShare)"
    echo "  --dataset cat_dunford2024  预处理猫咪数据集 Dunford 2024 (Dryad)"
    echo "  --dataset cat_smit2024   预处理猫咪数据集 Smit 2024 (FigShare)"
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
        --config configs/data.yaml $NO_GRAVITY_ALIGN

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
        --config configs/data.yaml $NO_GRAVITY_ALIGN

elif [ "$DATASET" = "custom" ]; then
    # 从 configs/data.yaml 读取 csv_path（简单 grep）
    CSV_CUSTOM=$(python -c "import yaml; c=yaml.safe_load(open('configs/data.yaml')); print(c.get('custom',{}).get('csv_path','data/raw_custom/data.csv'))")
    OUT_DIR="data/processed_custom"

    echo ""
    echo "[1/2] 检查自采数据集文件..."
    if [ ! -f "$CSV_CUSTOM" ]; then
        echo "  ❌ 缺少: $CSV_CUSTOM"
        echo "  请将你的 CSV 文件放到该路径，或修改 configs/data.yaml 中 custom.csv_path"
        echo ""
        echo "  CSV 格式要求（列名可在 configs/data.yaml custom 节自定义）:"
        echo "    dog_id, label, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z"
        exit 1
    else
        echo "  ✅ $CSV_CUSTOM"
    fi

    echo ""
    echo "[2/2] 运行预处理..."
    python src/data/preprocess.py \
        --dataset custom \
        --raw_csv_custom "$CSV_CUSTOM" \
        --output_dir "$OUT_DIR" \
        --config configs/data.yaml $NO_GRAVITY_ALIGN

elif [[ "$DATASET" == cat_* ]]; then
    # 根据 dataset 名确定 csv_path（从 configs/data.yaml 读取）
    CSV_CAT=$(python -c "
import yaml
cfg = yaml.safe_load(open('configs/data.yaml'))
print(cfg.get('$DATASET', {}).get('csv_path', 'data/raw_$DATASET/data.csv'))
")
    OUT_DIR="data/processed_$DATASET"

    echo ""
    echo "[1/2] 检查猫咪数据集文件..."
    if [ ! -f "$CSV_CAT" ]; then
        echo "  ❌ 缺少: $CSV_CAT"
        echo ""
        echo "  请先准备数据："
        if [[ "$DATASET" == "cat_smit2023" ]]; then
            echo "  1. 从 FigShare 下载: https://figshare.com/articles/dataset/23605842"
            echo "  2. 放到 data/raw_cat_smit2023/"
            echo "  3. 转换格式: python src/data/convert_cat_rdata.py --dataset smit2023 \\"
            echo "       --rdata data/raw_cat_smit2023/accel_data.RDATA \\"
            echo "       --annot data/raw_cat_smit2023/anno_data.RDATA \\"
            echo "       --out data/raw_cat_smit2023/cat_smit2023.csv"
        elif [[ "$DATASET" == "cat_dunford2024" ]]; then
            echo "  1. 从 Dryad 下载: https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83sx"
            echo "  2. 放到 data/raw_cat_dunford2024/"
            echo "  3. 确认列名并更新 configs/data.yaml cat_dunford2024 节"
        elif [[ "$DATASET" == "cat_smit2024" ]]; then
            echo "  1. 从 FigShare 下载: https://figshare.com/articles/dataset/24848292"
            echo "  2. 放到 data/raw_cat_smit2024/"
            echo "  3. 转换格式: python src/data/convert_cat_rdata.py --dataset smit2024 \\"
            echo "       --rdata data/raw_cat_smit2024/accel_data.RDATA \\"
            echo "       --annot data/raw_cat_smit2024/anno_data.RDATA \\"
            echo "       --out data/raw_cat_smit2024/cat_smit2024.csv"
        fi
        exit 1
    else
        echo "  ✅ $CSV_CAT"
    fi

    echo ""
    echo "[2/2] 运行预处理..."
    python src/data/preprocess.py \
        --dataset "$DATASET" \
        --output_dir "$OUT_DIR" \
        --config configs/data.yaml $NO_GRAVITY_ALIGN

else
    echo "未知数据集: $DATASET"
    echo "运行 bash setup.sh 查看支持的数据集列表"
    exit 1
fi

echo ""
echo "================================================"
DATASET_LABEL="$DATASET"
[ "$DATASET" = "custom" ] && DATASET_LABEL="自采"
echo "  ✅ 数据集${DATASET_LABEL}整理完成！"
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
