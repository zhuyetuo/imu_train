#!/usr/bin/env bash
# 一键训练：合成数据 + 不带合成数据，两个模型并行跑
#
# 用法:
#   bash train_custom.sh --date 2026_7_23
#   bash train_custom.sh --date 2026_7_23 --n_aug 30
#
# 输出:
#   results/processed_<DATE>/16hz_remap_custom_3class/ml_rf.pkl         ← 纯标注
#   results_synthetic/processed_<DATE>/16hz_remap_custom_3class/ml_rf.pkl ← 带合成

set -e

# ── 默认参数 ──────────────────────────────────────────────
DATE=""
HZ=16
N_AUG=50
LABEL="抓挠"
REMAP="configs/remap_custom_3class.yaml"
CSV_DIR="data/raw_wit/"
RESULTS_DIR="results"
RESULTS_SYN_DIR="results_synthetic"

# ── 解析参数 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --date)      DATE="$2";       shift 2 ;;
    --hz)        HZ="$2";         shift 2 ;;
    --n_aug)     N_AUG="$2";      shift 2 ;;
    --label)     LABEL="$2";      shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATE" ]]; then
  echo "用法: bash train_custom.sh --date <DATE>  (例: --date 2026_7_23)"
  exit 1
fi

PROCESSED_DIR="data/processed_${DATE}"
JSON="data/raw_custom/${DATE}/merged_tmp.json"
SYNTHETIC="data/synthetic/scratch_${DATE}.npz"

echo "=================================================="
echo "  日期: $DATE   采样率: ${HZ}Hz   增强倍数: $N_AUG"
echo "  预处理目录: $PROCESSED_DIR"
echo "=================================================="

# ── 检查预处理数据 ────────────────────────────────────────
if [[ ! -f "${PROCESSED_DIR}/${HZ}hz/train.npz" ]]; then
  echo "[错误] 找不到预处理数据: ${PROCESSED_DIR}/${HZ}hz/train.npz"
  echo "请先运行预处理:"
  echo "  python src/data/preprocess.py --dataset custom \\"
  echo "    --raw_csv_custom data/raw_custom/${DATE}/merged_${DATE}.csv \\"
  echo "    --output_dir ${PROCESSED_DIR} --config configs/data.yaml \\"
  echo "    --split_strategy label_concat --hz ${HZ}"
  exit 1
fi

# ── 方案 A：纯标注模型（后台运行）────────────────────────
echo ""
echo "▶ 方案 A：纯标注模型（后台运行）..."
python src/ml/train.py --hz "$HZ" --model rf \
  --processed_dir "$PROCESSED_DIR" \
  --remap "$REMAP" \
  --results_dir "$RESULTS_DIR" \
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
  --results_dir "$RESULTS_SYN_DIR" \
  > /tmp/train_with_syn.log 2>&1 &
PID_B=$!

# ── 等待两个训练完成 ──────────────────────────────────────
echo ""
echo "⏳ 等待两个模型训练完成..."
wait $PID_A && echo "  ✅ 方案 A 完成" || echo "  ❌ 方案 A 失败，见 /tmp/train_no_syn.log"
wait $PID_B && echo "  ✅ 方案 B 完成" || echo "  ❌ 方案 B 失败，见 /tmp/train_with_syn.log"

# ── 打印结果对比（过滤进度条噪音）────────────────────────
_show_log() {
  # 显示数据分布表 + 测试集结果，去掉进度条行
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
echo "  带合成: ${RESULTS_SYN_DIR}/processed_${DATE}/${HZ}hz_remap_custom_3class/ml_rf.pkl"
