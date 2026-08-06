#!/usr/bin/env bash
# 一键预处理 + 训练：同时生成纯标注和带合成数据两个模型
#
# 需要先有 data/raw_custom/<DATE>/merged_tmp.json（Label Studio 导出的合并JSON，
# 见README"标注分析"一节），训练CSV（merged_<DATE>.csv）会自动生成，不用手动
# 先跑 labelstudio_to_custom.py 这一步。
#
# 用法:
#   bash train_custom.sh --date 2026_7_23
#   bash train_custom.sh --date 2026_7_23 --n_aug 30
#   bash train_custom.sh --date 2026_7_23 --clean   # 先删掉旧缓存再全新生成，
#                                                     # 改过预处理/特征相关代码后建议加上
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
TAG=""                    # 输出目录后缀（留空=不加，跟原来路径一致）。
                           # 同一个DATE想同时保留majority/center两个版本的模型时，
                           # 各自传一个不同的--tag，避免第二次跑把第一次的processed_dir/
                           # results覆盖掉（两次跑的PROCESSED_DIR/结果目录名都会带上这个后缀）
CLEAN=0                    # 1=跑之前先删掉这个DATE+TAG对应的旧缓存(processed_dir/
                           # results/合成数据npz)再重新生成，默认0=不删（复用已有的）。
                           # 数据处理逻辑改了但没删缓存，新旧代码生成的中间产物混用，
                           # 是这几天踩过好几次的坑，改动过预处理/特征相关代码后
                           # 强烈建议加这个参数，保证是从头全新生成、不会跟旧缓存混着用

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
    --tag)            TAG="$2";            shift 2 ;;
    --clean)          CLEAN=1;             shift 1 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

if [[ -z "$DATE" ]]; then
  echo "用法: bash train_custom.sh --date <DATE>  (例: --date 2026_7_23)"
  exit 1
fi

PROCESSED_DIR="data/processed_${DATE}${TAG:+_$TAG}"
CSV="data/raw_custom/${DATE}/merged_${DATE}.csv"
JSON="data/raw_custom/${DATE}/merged_tmp.json"
SYNTHETIC="data/synthetic/scratch_${DATE}${TAG:+_$TAG}.npz"
DATASET_TAG=$(basename "$PROCESSED_DIR")

echo "=================================================="
echo "  日期: $DATE   采样率: ${HZ}Hz   增强倍数: $N_AUG${TAG:+   tag: $TAG}"
echo "=================================================="

# ── --clean：先删掉这个DATE+TAG对应的旧缓存，保证从头全新生成 ──────
if [[ "$CLEAN" == "1" ]]; then
  echo ""
  echo "▶ --clean：删除旧缓存..."
  echo "  rm -rf $PROCESSED_DIR"
  rm -rf "$PROCESSED_DIR"
  echo "  rm -rf $RESULTS_DIR/$DATASET_TAG"
  rm -rf "${RESULTS_DIR:?}/$DATASET_TAG"
  echo "  rm -f  $SYNTHETIC"
  rm -f "$SYNTHETIC"
fi

# ── 步骤1：生成训练CSV（CSV不存在，或者--clean要求从头全新生成时自动跑）──
# 注意：这里不传 --keep_labels 的值，等于保留全部标签——remap_custom_3class.yaml
# 需要看到甩身体/舔身体/啃身体等原始细分类别才能把它们折算进"活动"类当负样本，
# 之前这里默认只留3类，把这些数据在到达remap之前就丢掉了，是踩过的坑，见
# configs/data.yaml 里 custom.keep_labels 的注释。
if [[ ! -f "$CSV" || "$CLEAN" == "1" ]]; then
  if [[ ! -f "$JSON" ]]; then
    echo "[错误] 找不到标注JSON: $JSON"
    echo "需要先有 Label Studio 导出的合并JSON才能生成训练CSV"
    exit 1
  fi
  echo ""
  echo "▶ 步骤1：生成训练CSV（$CSV 不存在或 --clean 要求重新生成）..."
  python src/data/labelstudio_to_custom.py \
    --json "$JSON" \
    --output "$CSV" \
    --csv_dir "$CSV_DIR" \
    --keep_labels
fi

if [[ ! -f "$CSV" ]]; then
  echo "[错误] 生成训练CSV失败: $CSV"
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

# ── 等待两个训练完成，实时把两边的日志打到当前终端（加[A]/[B]前缀区分）──
# 之前这里是纯等待，进度条/特征提取日志全写进/tmp的文件里，只能另开
# 终端手动tail -f才看得到；现在直接在这个终端里实时滚动显示，不用切窗口。
echo ""
echo "⏳ 等待两个模型训练完成（下面实时滚动的是训练日志，[A]=纯标注 [B]=带合成）..."
tail -f -n +1 /tmp/train_no_syn.log 2>/dev/null | sed -u 's/^/[A] /' &
TAIL_A=$!
tail -f -n +1 /tmp/train_with_syn.log 2>/dev/null | sed -u 's/^/[B] /' &
TAIL_B=$!

wait $PID_A && echo "  ✅ 方案 A 完成" || echo "  ❌ 方案 A 失败，见 /tmp/train_no_syn.log"
wait $PID_B && echo "  ✅ 方案 B 完成" || echo "  ❌ 方案 B 失败，见 /tmp/train_with_syn.log"

kill "$TAIL_A" "$TAIL_B" 2>/dev/null
wait "$TAIL_A" "$TAIL_B" 2>/dev/null

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
echo "  纯标注: ${RESULTS_DIR}/${DATASET_TAG}/${HZ}hz_remap_custom_3class/ml_rf.pkl"
echo "  带合成: ${RESULTS_DIR}/${DATASET_TAG}/${HZ}hz_remap_custom_3class_syn/ml_rf.pkl"
