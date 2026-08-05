#!/usr/bin/env bash
# 一键预处理 + 训练：同时生成纯标注和带合成数据两个模型
#
# 用法:
#   bash train_custom.sh --date 2026_7_23
#   bash train_custom.sh --date 2026_7_23 --n_aug 30
#
# 输出:
#   results/processed_<DATE>/16hz_remap_custom_3class/ml_rf.pkl      ← 纯标注
#   results/processed_<DATE>/16hz_remap_custom_3class_syn/ml_rf.pkl  ← 带合成

set -e

# ── 默认参数 ──────────────────────────────────────────────
DATE=""
HZ=16
N_AUG=50
LABEL="抓挠"
REMAP="configs/remap_custom_3class.yaml"
CSV_DIR="data/raw_wit/"
RESULTS_DIR="results"
SPLIT_STRATEGY="random"   # random=混合所有狗后按片段分组划分（默认），subject=按狗ID划分，label_concat=按类别拼接
TRAIN_RATIO="0.9"         # 训练集比例（默认0.9）
VAL_RATIO="0.1"           # 验证集比例（默认0.1）
TEST_RATIO="0.0"          # 测试集比例（默认0=无测试集，纠错循环阶段不需要）
LABEL_MODE="majority"     # majority=多数投票（默认，原有行为），center=窗口标签取中心帧
STRIDE_S=""               # 训练窗口步长秒数（留空=用 configs/data.yaml 默认值1秒）
FEAT_WORKERS="1"          # 特征提取并行进程数（默认1=不并行，传-1用全部核心）

# ── 解析参数 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --date)           DATE="$2";           shift 2 ;;
    --hz)             HZ="$2";             shift 2 ;;
    --n_aug)          N_AUG="$2";          shift 2 ;;
    --label)          LABEL="$2";          shift 2 ;;
    --split_strategy) SPLIT_STRATEGY="$2"; shift 2 ;;
    --train_ratio)    TRAIN_RATIO="$2";    shift 2 ;;
    --val_ratio)      VAL_RATIO="$2";      shift 2 ;;
    --test_ratio)     TEST_RATIO="$2";     shift 2 ;;
    --label_mode)     LABEL_MODE="$2";     shift 2 ;;
    --stride_s)       STRIDE_S="$2";       shift 2 ;;
    --feat_workers)   FEAT_WORKERS="$2";   shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATE" ]]; then
  echo "用法: bash train_custom.sh --date <DATE>  (例: --date 2026_7_23)"
  exit 1
fi

PROCESSED_DIR="data/processed_${DATE}"
CSV="data/raw_custom/${DATE}/merged_${DATE}.csv"
JSON="data/raw_custom/${DATE}/merged_tmp.json"
SYNTHETIC="data/synthetic/scratch_${DATE}.npz"

echo "=================================================="
echo "  日期: $DATE   采样率: ${HZ}Hz   增强倍数: $N_AUG"
echo "=================================================="

# ── 检查输入文件 ──────────────────────────────────────────
if [[ ! -f "$CSV" ]]; then
  echo "[错误] 找不到训练 CSV: $CSV"
  echo "请先运行步骤 1（labelstudio_to_custom.py）生成该文件"
  exit 1
fi

# ── 预处理（自动清除旧缓存）──────────────────────────────
echo ""
echo "▶ 预处理数据..."
python src/data/preprocess.py \
  --dataset custom \
  --raw_csv_custom "$CSV" \
  --output_dir "$PROCESSED_DIR" \
  --config configs/data.yaml \
  --split_strategy "$SPLIT_STRATEGY" \
  --train_ratio "$TRAIN_RATIO" \
  --val_ratio "$VAL_RATIO" \
  --test_ratio "$TEST_RATIO" \
  --label_mode "$LABEL_MODE" \
  $( [[ -n "$STRIDE_S" ]] && echo "--stride_s $STRIDE_S" ) \
  --hz "$HZ"

# ── 方案 A：纯标注模型（后台运行）────────────────────────
echo ""
echo "▶ 方案 A：纯标注模型（后台运行）..."
python src/ml/train.py --hz "$HZ" --model rf \
  --processed_dir "$PROCESSED_DIR" \
  --remap "$REMAP" \
  --results_dir "$RESULTS_DIR" \
  --feat_workers "$FEAT_WORKERS" \
  > /tmp/train_no_syn.log 2>&1 &
PID_A=$!

# ── 生成合成数据 ──────────────────────────────────────────
echo "▶ 生成合成数据（${LABEL}，n_aug=${N_AUG}）..."
python src/data/synthesize_scratch.py \
  --json "$JSON" \
  --csv_dir "$CSV_DIR" \
  --output "$SYNTHETIC" \
  --processed_dir "$PROCESSED_DIR" \
  --remap "$REMAP" \
  --label "$LABEL" \
  --hz "$HZ" \
  --n_aug "$N_AUG"

# ── 方案 B：带合成数据模型（后台运行）───────────────────
echo ""
echo "▶ 方案 B：带合成数据模型（后台运行）..."
python src/ml/train.py --hz "$HZ" --model rf \
  --processed_dir "$PROCESSED_DIR" \
  --remap "$REMAP" \
  --synthetic "$SYNTHETIC" \
  --synthetic_label "$LABEL" \
  --results_dir "$RESULTS_DIR" \
  --feat_workers "$FEAT_WORKERS" \
  > /tmp/train_with_syn.log 2>&1 &
PID_B=$!

# ── 等待两个训练完成 ──────────────────────────────────────
echo ""
echo "⏳ 等待两个模型训练完成..."
wait $PID_A && echo "  ✅ 方案 A 完成" || echo "  ❌ 方案 A 失败，见 /tmp/train_no_syn.log"
wait $PID_B && echo "  ✅ 方案 B 完成" || echo "  ❌ 方案 B 失败，见 /tmp/train_with_syn.log"

# ── 打印结果对比（过滤进度条噪音）────────────────────────
_show_log() {
  grep -v "██\|提取特征\|step/s\|窗口/s" "$1" 2>/dev/null \
    | grep -A 999 "数据集类别分布" \
    || cat "$1"
}

echo ""
echo "=================================================="
echo "  训练结果对比"
echo "=================================================="
echo ""
echo "── 方案 A（纯标注）──"
_show_log /tmp/train_no_syn.log
echo ""
echo "── 方案 B（带合成）──"
_show_log /tmp/train_with_syn.log

echo ""
echo "模型路径:"
echo "  纯标注: ${RESULTS_DIR}/processed_${DATE}/${HZ}hz_remap_custom_3class/ml_rf.pkl"
echo "  带合成: ${RESULTS_DIR}/processed_${DATE}/${HZ}hz_remap_custom_3class_syn/ml_rf.pkl"
