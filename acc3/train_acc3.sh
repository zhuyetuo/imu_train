#!/usr/bin/env bash
# 训一个**没有陀螺仪**的模型：acc 三轴 + 由它派生的 pitch/roll，共 5 通道。
#
# 用法跟 train_custom.sh **完全一样**，把脚本名换掉就行：
#
#   bash acc3/train_acc3.sh --date "2026_8_11-2026_8_27_raw" --source_hz 50 --hz 16 \
#     --extra_date "2026_8_28_imu1_bb:50" \
#     --extra_date "2026_8_28_imu2_bali:50" \
#     --extra_date "2026_8_28_imu3_lulu:50" \
#     --extra_date "2026_8_28_imu4_xiaoman_unwear:50" \
#     --label_mode majority --window_s 1 --stride_s 0.5 \
#     --missing_strategy drop_window \
#     --skip_syn --model rf --clean --feat_workers -1
#
# 做四步：
#   1. 调 train_custom.sh --skip_ml    ← CSV 生成、多批次重采样合并、预处理，
#                                        **一行都没重写**，跟基线走的是同一条路
#   2. 砍掉陀螺仪三列 → 5 通道          （acc3/make_acc3_npz.py）
#   3. 用 113 维特征训练                （acc3/train_acc3.py + features5.py）
#   4. 跟基线对比并打印
#
# 为什么第 1 步复用 train_custom.sh：--extra_date 那套多批次对齐重采样的逻辑
# 有几百行，抄一份迟早分家；而分家之后两个模型的训练数据其实不一样，
# 指标还能比——比出来的东西没有意义。
#
# 第 2 步为什么可以直接砍列而不重跑预处理：重力对齐的旋转矩阵只从 acc 算，
# pitch/roll 也只从 acc 算，所以那五列有没有采过陀螺仪都是**逐位相同**的数。
# 见 acc3/make_acc3_npz.py 顶部，acc3/selfcheck5.py 第 ① 条会验。
#
# 额外参数（放在最后，其余原样转给 train_custom.sh）：
#   --acc3_results_dir DIR   默认 results_acc3
#   --baseline_results_dir DIR  对比用，默认 results

set -euo pipefail
cd "$(dirname "$0")/.."

PASS=()
HZ=16
DATE=""
TAG=""
MISSING_STRATEGY="none"
MODEL=rf
REMAP="configs/remap_custom_3class.yaml"
FEAT_WORKERS=1
ACC3_RESULTS="results_acc3"
BASE_RESULTS="results"
CLEAN=0

# 这些是**边转发边偷看**：目录名要按 train_custom.sh 同一套规则拼出来，
# 拼错的话后面找不到 npz。故意不自己定义默认值之外的行为——
# 规则变了这里会对不上，所以下面转完之后会核对目录是否真的存在
while [[ $# -gt 0 ]]; do
  case "$1" in
    --acc3_results_dir)     ACC3_RESULTS="$2"; shift 2 ;;
    --baseline_results_dir) BASE_RESULTS="$2"; shift 2 ;;
    --date)               DATE="$2";              PASS+=("$1" "$2"); shift 2 ;;
    --hz)                 HZ="$2";                PASS+=("$1" "$2"); shift 2 ;;
    --tag)                TAG="$2";               PASS+=("$1" "$2"); shift 2 ;;
    --missing_strategy)   MISSING_STRATEGY="$2";  PASS+=("$1" "$2"); shift 2 ;;
    --model)              MODEL="$2";             PASS+=("$1" "$2"); shift 2 ;;
    --feat_workers)       FEAT_WORKERS="$2";      PASS+=("$1" "$2"); shift 2 ;;
    --clean)              CLEAN=1;                PASS+=("$1");      shift 1 ;;
    -h|--help)            sed -n '2,40p' "$0"; exit 0 ;;
    *)                    PASS+=("$1");           shift 1 ;;
  esac
done

if [[ -z "$DATE" ]]; then
  echo "✗ 要给 --date（跟 train_custom.sh 一样）。-h 看用法。"
  exit 1
fi

# 跟 train_custom.sh 同一套规则（它第 213-216、237 行）
[[ -z "$TAG" ]] && TAG="missing_${MISSING_STRATEGY}"
PROCESSED_DIR="data/processed_${DATE}${TAG:+_$TAG}"
ACC3_DIR="${PROCESSED_DIR}_acc3"

echo "═══ 1/4 预处理（走 train_custom.sh --skip_ml，跟基线同一条路）═══"
echo "    预期输出：$PROCESSED_DIR/${HZ}hz/"
bash train_custom.sh "${PASS[@]}" --skip_ml

if [[ ! -f "${PROCESSED_DIR}/${HZ}hz/train.npz" ]]; then
  # 目录名规则跟 train_custom.sh 对不上了。**当场停**——继续下去会在
  # 一个空目录上训出一个没有意义的模型，而日志里只是"0 个窗口"
  echo ""
  echo "✗ 没找到 ${PROCESSED_DIR}/${HZ}hz/train.npz。"
  echo "  说明这里拼目录名的规则跟 train_custom.sh 不一致了。实际生成的是："
  ls -d data/processed_* 2>/dev/null | sed 's/^/    /'
  echo "  可以直接手动跑后面两步："
  echo "    python acc3/make_acc3_npz.py --src <上面那个目录> --dst <它>_acc3 --hz $HZ"
  exit 1
fi

echo ""
echo "═══ 2/4 砍掉陀螺仪三列 → 5 通道 ═══"
[[ "$CLEAN" == "1" ]] && rm -rf "$ACC3_DIR"
python acc3/make_acc3_npz.py --src "$PROCESSED_DIR" --dst "$ACC3_DIR" --hz "$HZ"

echo ""
echo "═══ 3/4 训练（113 维特征）═══"
python acc3/train_acc3.py \
  --hz "$HZ" --model "$MODEL" \
  --processed_dir "$ACC3_DIR" \
  --remap "$REMAP" \
  --results_dir "$ACC3_RESULTS" \
  --feat_workers "$FEAT_WORKERS"

REMAP_TAG="_$(basename "${REMAP%.*}")"
OUT_DIR="${ACC3_RESULTS}/$(basename "$ACC3_DIR")/${HZ}hz${REMAP_TAG}/${MODEL}"
BASE_DIR="${BASE_RESULTS}/$(basename "$PROCESSED_DIR")/${HZ}hz${REMAP_TAG}/${MODEL}"

echo ""
echo "═══ 4/4 对比 ═══"
python acc3/compare.py "$OUT_DIR/ml_${MODEL}.json" "$BASE_DIR/ml_${MODEL}.json"

echo ""
echo "模型在：$OUT_DIR/"
