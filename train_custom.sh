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
#   # 合并多个采集批次一起训练（不同批次采样率可以不一样，非--hz的批次会先
#   # 被重采样对齐到--hz，见src/data/resample_csv_hz.py）：
#   bash train_custom.sh --date 2026_8_11-2026_8_27_raw --hz 16 \
#     --extra_date 2026_7_17-2026_7_29:16 \
#     --extra_date 2026_7_30-2026_8_11:16 \
#     --tag merged --clean
#   # 上面例子里主数据是50Hz原始采集，--hz 16表示训练目标采样率是16Hz，
#   # 三个批次（50Hz的主数据+两个本来就是16Hz的旧数据）都会被统一到16Hz后
#   # 合并训练。--extra_date原有数据本来是16Hz采的就不需要重采样，冒号后面
#   # 的数字写它自己真实的采样率（16），跟主数据的--hz一样就会跳过重采样。
#   # 单独--date不加--extra_date时完全是原来的行为，不受影响。
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
EXTRA_DATES=()             # --extra_date DATE:HZ，可重复传，跟主--date合并一起训练。
                           # HZ跟主--hz不一样的批次会先重采样对齐（见上面用法示例）

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
    --extra_date)     EXTRA_DATES+=("$2"); shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

for _ed in "${EXTRA_DATES[@]:-}"; do
  if [[ -n "$_ed" && "$_ed" != *:* ]]; then
    echo "[错误] --extra_date 格式应为 DATE:HZ（例: 2026_7_17-2026_7_29:16），收到: $_ed"
    exit 1
  fi
done

if [[ -z "$DATE" ]]; then
  echo "用法: bash train_custom.sh --date <DATE>  (例: --date 2026_7_23)"
  exit 1
fi

PROCESSED_DIR="data/processed_${DATE}${TAG:+_$TAG}"
DATA_DIR="data/raw_custom/${DATE}"
CSV="${DATA_DIR}/merged_${DATE}.csv"
JSON="${DATA_DIR}/merged_tmp.json"
SYNTHETIC="data/synthetic/scratch_${DATE}${TAG:+_$TAG}.npz"
DATASET_TAG=$(basename "$PROCESSED_DIR")

# 训练日志放项目自己的tmp/目录下，不是系统/tmp——系统/tmp是全机器共用的，
# 日志文件名又没带DATASET_TAG，不同用户/不同次跑（尤其带不同--tag同时跑）
# 容易互相覆盖，日志内容对不上这次跑的是哪个数据集；放项目tmp/目录下、
# 文件名带上DATASET_TAG，各自独立，也方便跑完之后回头翻日志（不用记得
# 是哪次登录、哪个系统临时目录）
TMP_DIR="tmp"
mkdir -p "$TMP_DIR"
LOG_NO_SYN="${TMP_DIR}/train_no_syn_${DATASET_TAG}.log"
LOG_WITH_SYN="${TMP_DIR}/train_with_syn_${DATASET_TAG}.log"
LOG_FULL="${TMP_DIR}/train_full_${DATASET_TAG}.log"

# 把这个脚本自己打印的全部内容（步骤0/1/1.5的合并/转CSV/重采样日志、
# 预处理输出、每一行echo的进度提示……）也整份记下来，不只是后台跑的
# 两个python src/ml/train.py各自的LOG_NO_SYN/LOG_WITH_SYN——那两个只
# 覆盖训练那一步，前面预处理/数据合并阶段出的问题之前是看不到历史记录
# 的。用exec重定向脚本自身的stdout/stderr，同时通过tee照常打印到终端，
# 不影响交互时的实时可见性。
exec > >(tee -a "$LOG_FULL") 2>&1
echo ""
echo "完整日志: $LOG_FULL"

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

# ── 步骤0：合并 project-*.json → merged_tmp.json（唯一需要手动维护的源文件
# 是 Label Studio 导出的 project-*.json，其余都是可以随时删掉、重新生成的
# 派生文件：merged_tmp.json/merged_<DATE>.csv/processed_dir/results/合成npz。
# --clean 或 merged_tmp.json 不存在时自动重新合并，不用再手动跑那段
# python -c 合并脚本）──
if [[ ! -f "$JSON" || "$CLEAN" == "1" ]]; then
  n_project_json=$(find "$DATA_DIR" -maxdepth 1 -name "project-*.json" 2>/dev/null | wc -l)
  if [[ "$n_project_json" -eq 0 ]]; then
    echo "[错误] $DATA_DIR 下没有找到任何 project-*.json"
    echo "需要先把 Label Studio 导出的 project-*.json 放到这个目录下"
    exit 1
  fi
  echo ""
  echo "▶ 步骤0：合并 $DATA_DIR 下 $n_project_json 个 project-*.json → $JSON ..."
  python -c "
import json, glob, sys
files = sorted(glob.glob(sys.argv[1]))
merged = []
for f in files:
    merged += json.load(open(f, encoding='utf-8'))
    print(f'  加载: {f}')
json.dump(merged, open(sys.argv[2], 'w'), ensure_ascii=False)
print(f'合并完成，共 {len(merged)} 条任务')
" "${DATA_DIR}/project-*.json" "$JSON"
fi

# ── 步骤1：生成训练CSV（CSV不存在，或者--clean要求从头全新生成时自动跑）──
# 注意：这里不传 --keep_labels 的值，等于保留全部标签——remap_custom_3class.yaml
# 需要看到甩身体/舔身体/啃身体等原始细分类别才能把它们折算进"活动"类当负样本，
# 之前这里默认只留3类，把这些数据在到达remap之前就丢掉了，是踩过的坑，见
# configs/data.yaml 里 custom.keep_labels 的注释。
if [[ ! -f "$CSV" || "$CLEAN" == "1" ]]; then
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

# ── 步骤1.5：合并 --extra_date 指定的其它批次（可选）────────────────────
# 每个额外批次走跟主--date一样的 步骤0(合并project json)+步骤1(生成CSV)，
# 采样率跟主--hz不一样的话，先用resample_csv_hz.py重采样对齐到--hz，再
# 合并进一份CSV里给后面的预处理/训练用（预处理只认一个全局source_hz，
# 混不同采样率的原始数据在一起会用错误的hz去插值/降采样，见
# src/data/resample_csv_hz.py 顶部说明）。record_id前面统一加上各自的
# 日期前缀，避免不同批次导出的task编号刚好撞车导致数据集划分时误判成
# 同一段录制。
if [[ ${#EXTRA_DATES[@]} -gt 0 ]]; then
  MERGED_CSV="${DATA_DIR}/merged_${DATE}${TAG:+_$TAG}_combined.csv"
  if [[ ! -f "$MERGED_CSV" || "$CLEAN" == "1" ]]; then
    echo ""
    echo "▶ 步骤1.5：合并 ${#EXTRA_DATES[@]} 个额外批次 → $MERGED_CSV ..."
    CSV_PARTS=("${DATE}:${CSV}:${HZ}")
    for _ed in "${EXTRA_DATES[@]}"; do
      _edate="${_ed%%:*}"
      _ehz="${_ed##*:}"
      _edata_dir="data/raw_custom/${_edate}"
      _ejson="${_edata_dir}/merged_tmp.json"
      _ecsv="${_edata_dir}/merged_${_edate}.csv"

      if [[ ! -f "$_ejson" || "$CLEAN" == "1" ]]; then
        _en=$(find "$_edata_dir" -maxdepth 1 -name "project-*.json" 2>/dev/null | wc -l)
        if [[ "$_en" -eq 0 ]]; then
          echo "[错误] $_edata_dir 下没有找到任何 project-*.json"
          exit 1
        fi
        echo "  ▶ 合并 $_edata_dir 下 $_en 个 project-*.json → $_ejson ..."
        python -c "
import json, glob, sys
files = sorted(glob.glob(sys.argv[1]))
merged = []
for f in files:
    merged += json.load(open(f, encoding='utf-8'))
    print(f'    加载: {f}')
json.dump(merged, open(sys.argv[2], 'w'), ensure_ascii=False)
print(f'  合并完成，共 {len(merged)} 条任务')
" "${_edata_dir}/project-*.json" "$_ejson"
      fi

      if [[ ! -f "$_ecsv" || "$CLEAN" == "1" ]]; then
        echo "  ▶ 生成训练CSV: $_ecsv ..."
        python src/data/labelstudio_to_custom.py \
          --json "$_ejson" \
          --output "$_ecsv" \
          --csv_dir "$CSV_DIR" \
          --keep_labels
      fi

      if [[ ! -f "$_ecsv" ]]; then
        echo "[错误] 生成训练CSV失败: $_ecsv"
        exit 1
      fi

      CSV_PARTS+=("${_edate}:${_ecsv}:${_ehz}")
    done

    python -c "
import sys
sys.path.insert(0, 'src/data')
import pandas as pd
from resample_csv_hz import SENSOR_COLS
from preprocess import downsample
import numpy as np

target_hz = int(sys.argv[1])
parts = sys.argv[2:]
frames = []
for part in parts:
    date_tag, path, src_hz = part.split(':')
    src_hz = int(src_hz)
    df = pd.read_csv(path)
    if src_hz != target_hz:
        print(f'  重采样 {path}: {src_hz}Hz -> {target_hz}Hz')
        out_rows = []
        for rid, g in df.groupby('record_id', sort=False):
            data = g[SENSOR_COLS].to_numpy(dtype=np.float64)
            labels = g['label'].to_numpy()
            data_ds, labels_ds = downsample(data, labels, src_hz, target_hz)
            out = pd.DataFrame(data_ds, columns=SENSOR_COLS)
            out.insert(0, 'label', labels_ds)
            out.insert(0, 'record_id', rid)
            out_rows.append(out)
        df = pd.concat(out_rows, ignore_index=True)
    else:
        print(f'  {path}: 已经是{target_hz}Hz，跳过重采样')
    df['record_id'] = date_tag + '_' + df['record_id'].astype(str)
    frames.append(df)

merged = pd.concat(frames, ignore_index=True)
merged.to_csv('${MERGED_CSV}', index=False)
print(f'合并完成: {len(frames)}个批次, 共{len(merged)}行 -> ${MERGED_CSV}')
" "$HZ" "${CSV_PARTS[@]}"
  fi
  CSV="$MERGED_CSV"
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
  > "$LOG_NO_SYN" 2>&1 &
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
  > "$LOG_WITH_SYN" 2>&1 &
PID_B=$!

# ── 等待两个训练完成，实时把两边的日志打到当前终端（加[A]/[B]前缀区分）──
# 之前这里是纯等待，进度条/特征提取日志全写进/tmp的文件里，只能另开
# 终端手动tail -f才看得到；现在直接在这个终端里实时滚动显示，不用切窗口。
echo ""
echo "⏳ 等待两个模型训练完成（下面实时滚动的是训练日志，[A]=纯标注 [B]=带合成）..."
# tqdm进度条是靠\r（回车不换行）原地刷新的，不是每次都换行；sed要等到\n
# 才会把攒的内容当一整行输出，\r的更新会一直卡在缓冲区里，直到最后进度条
# 跑完打印真正的\n才一次性冒出来，等于白等——这也是之前"卡住不动，跑完
# 才突然出现"的原因。用 tr 把\r也当成换行处理，每次进度更新都单独成行、
# 立刻输出，代价是不再是原地刷新的动画效果，而是一行行往下滚动，但是真
# 实时的，不用干等。
# 光加tr还不够：tr写到管道（不是终端）时默认是全缓冲的，会把转换后的内容
# 攒在自己的缓冲区里不立刻往下传，一样会卡住——用 stdbuf -oL 强制tr按行
# 缓冲，才能真正做到每次更新都立刻显示。
tail -f -n +1 "$LOG_NO_SYN" 2>/dev/null | stdbuf -oL tr '\r' '\n' | sed -u 's/^/[A] /' &
TAIL_A=$!
tail -f -n +1 "$LOG_WITH_SYN" 2>/dev/null | stdbuf -oL tr '\r' '\n' | sed -u 's/^/[B] /' &
TAIL_B=$!

wait $PID_A && echo "  ✅ 方案 A 完成" || echo "  ❌ 方案 A 失败，见 $LOG_NO_SYN"
wait $PID_B && echo "  ✅ 方案 B 完成" || echo "  ❌ 方案 B 失败，见 $LOG_WITH_SYN"

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
_show_log "$LOG_NO_SYN"
echo ""
echo "── 方案 B（带合成）──"
_show_log "$LOG_WITH_SYN"

echo ""
echo "模型路径:"
echo "  纯标注: ${RESULTS_DIR}/${DATASET_TAG}/${HZ}hz_remap_custom_3class/ml_rf.pkl"
echo "  带合成: ${RESULTS_DIR}/${DATASET_TAG}/${HZ}hz_remap_custom_3class_syn/ml_rf.pkl"
