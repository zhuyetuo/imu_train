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
#   # 合并多个采集批次一起训练（不同批次采样率可以不一样，非目标--hz的批次
#   # 会先被重采样对齐，见src/data/resample_csv_hz.py）：
#   bash train_custom.sh --date 2026_8_11-2026_8_27_raw --source_hz 50 --hz 16 \
#     --extra_date 2026_7_17-2026_7_29:16 \
#     --extra_date 2026_7_30-2026_8_11:16 \
#     --tag merged --clean
#   # 上面例子里主数据（--date）是50Hz原始采集的，--source_hz 50必须显式
#   # 写它自己真实的采样率，不写默认等于--hz（当成"已经是目标采样率"，
#   # 不会重采样，主数据是原始采样率跟--hz不一样时一定要传，否则会静默
#   # 当成已经是目标Hz、直接跳过重采样，训练进去的时间轴是错的）。
#   # --hz 16是训练目标采样率，三个批次（50Hz的主数据+两个本来就是16Hz的
#   # 旧数据）都会被统一到16Hz后合并训练。--extra_date后面冒号跟的是那个
#   # 批次自己真实的采样率，跟--hz一样就会跳过重采样。
#   # 单独--date不加--extra_date/--source_hz时完全是原来的行为，不受影响。
#
# 输出:
#   results/processed_<DATE>/16hz_remap_custom_3class/<model>/ml_<model>.pkl      ← 纯标注
#   results/processed_<DATE>/16hz_remap_custom_3class_syn/<model>/ml_<model>.pkl  ← 带合成
#   <model>是rf(默认)/xgb/lgbm/catboost/extratrees/histgb，见--model参数。
#   每个模型类型各自一层子目录，同一份数据+不同--model训练互不覆盖，
#   --clean只清同一个模型类型自己的目录，不影响其它模型训练好的结果

set -e

# Ctrl+C 杀不掉后台训练进程的原因：这个脚本是用`bash train_custom.sh`
# 跑的非交互式脚本，没有开job control，`&`起的后台进程（PID_A/PID_B/
# TAIL_A/TAIL_B）默认会忽略SIGINT——Ctrl+C只会打断前台的`wait`，两个
# python src/ml/train.py（以及--feat_workers>1时它们fork出来的特征
# 提取子进程）会变成孤儿继续在后台跑，看着像"中断不了"。这里用trap
# 显式在收到INT/TERM时把它们都杀掉。
cleanup() {
  echo ""
  echo "⚠ 收到中断信号，清理后台进程..."
  for _pid in "${PID_A:-}" "${PID_B:-}" "${TAIL_A:-}" "${TAIL_B:-}"; do
    if [[ -n "$_pid" ]]; then
      # pkill/kill 找不到匹配进程时返回非0——没有子进程（feat_workers=1）
      # 或者进程已经退出都是正常情况，加||true避免被set -e在清理一半时
      # 直接终止掉，导致后面本该杀的kill "$_pid"根本没执行到
      pkill -P "$_pid" 2>/dev/null || true   # 先杀子进程（feat_workers多进程）
      kill "$_pid" 2>/dev/null || true
    fi
  done
  exit 130
}
trap cleanup INT TERM

# ── 默认参数 ──────────────────────────────────────────────
DATE=""
HZ=16
MISSING_STRATEGY="none"    # none(默认)/drop/ffill/drop_window——acc/gyro缺失值
                           # (蓝牙断联)怎么处理，见src/data/labelstudio_to_custom.py的
                           # --missing_strategy说明。默认none是为了跟这个选项加之前
                           # 所有历史模型的实际行为一致（那时候labelstudio_to_custom.py
                           # 根本没处理过NaN，等价于现在的none，是个隐藏bug）——不传
                           # 这个参数训出来的模型，效果基准跟以前能直接比。想用修复后
                           # 的处理方式，显式传drop/ffill/drop_window。四种方式：
                           #   none        原样保留NaN，不处理（历史行为，树模型大多能
                           #               容忍，DL不能，会loss变nan）
                           #   drop        丢弃含NaN的那一整行，不做前后值填充
                           #   ffill       前后值填充（编造数值，跟实时推理infer_csv_
                           #               scratch.py的处理方式一致）
                           #   drop_window 行级不处理(等价none)，改成切窗口那一步整窗
                           #               丢弃(src/data/preprocess.py的--drop_nan_windows)，
                           #               窗口内哪怕只有一帧NaN也整窗不要，比drop(只丢
                           #               NaN那一行、窗口其它帧留着)更严格
                           # 配合--tag分别跑几个版本，能直接对比这几种处理方式训出来的
                           # 模型效果
MODEL_TYPE="rf"            # rf(默认)/xgb/lgbm/catboost/extratrees/histgb，
                           # 对应src/ml/train.py的--model，见该文件MODELS字典
N_AUG=50
LABELS=()                 # 要合成的类别，--label可重复传（比如--label 抓挠 --label 甩身体
                           # 同时给两个类别都补合成数据）。不传时默认只合成"抓挠"（见下面
                           # 解析完参数后的默认值兜底）
REMAP="configs/remap_custom_3class.yaml"
CSV_DIR="data/raw_wit/"
RESULTS_DIR="results"
SPLIT_STRATEGY="random"   # random=混合所有狗后按片段分组划分（默认），subject=按狗ID划分，label_concat=按类别拼接
TRAIN_RATIO="0.9"         # 训练集比例（默认0.9）
VAL_RATIO="0.1"           # 验证集比例（默认0.1）
TEST_RATIO="0.0"          # 测试集比例（默认0=无测试集，纠错循环阶段不需要）
LABEL_MODE="majority"     # majority=多数投票（默认，原有行为），center=窗口标签取中心帧
STRIDE_S=""               # 训练窗口步长秒数（留空=用 configs/data.yaml 默认值1秒）
WINDOW_S=""               # 训练窗口长度秒数（留空=用 configs/data.yaml 默认值2秒）。
                           # 甩身体这类短促动作（1~1.5秒）比默认2秒窗口还短，建不成
                           # 窗口、直接被跳过，试着调小（比如--window_s 1 --stride_s 0.5）
                           # 能不能把这类短片段也纳入训练
FEAT_WORKERS="1"          # 特征提取并行进程数（默认1=不并行，传-1用全部核心）
TAG=""                    # 输出目录后缀。留空时不是真的不加后缀，而是自动用
                           # missing_${MISSING_STRATEGY}（见下面解析完参数后的
                           # 兜底逻辑）——不同missing_strategy的数据不该共用
                           # 同一个processed_dir。同一个DATE+missing_strategy下
                           # 还想同时保留majority/center等其它版本时，显式传
                           # --tag会覆盖这个自动值（两次跑的PROCESSED_DIR/结果
                           # 目录名都会带上最终生效的这个后缀）
SKIP_SYN=0                 # 1=只训练方案A(纯标注)，跳过生成合成数据和方案B。
                           # 已经确认合成数据在短窗口下会让活动/甩身体的误判
                           # 变多（见pm_skin_scoring/docs或对话记录），不想用
                           # 合成数据时加这个参数，省掉合成数据生成+方案B训练
                           # 的时间
CLEAN=0                    # 1=跑之前先删掉这个DATE+TAG对应的旧缓存(processed_dir/
                           # results/合成数据npz)再重新生成，默认0=不删（复用已有的）。
                           # 数据处理逻辑改了但没删缓存，新旧代码生成的中间产物混用，
                           # 是这几天踩过好几次的坑，改动过预处理/特征相关代码后
                           # 强烈建议加这个参数，保证是从头全新生成、不会跟旧缓存混着用
EXTRA_DATES=()             # --extra_date DATE:HZ，可重复传，跟主--date合并一起训练。
                           # HZ跟目标--hz不一样的批次会先重采样对齐（见上面用法示例）
SOURCE_HZ=""               # 主--date数据自己真实的采样率，留空默认等于--hz（当成"已经
                           # 是目标采样率不用重采样"）。只有加了--extra_date合并多批次、
                           # 且主数据的原始采样率跟--hz不一样时才需要显式传（比如主数据
                           # 是50Hz原始采集，--hz 16是训练目标，就要传--source_hz 50，
                           # 不传会被静默当成已经是16Hz，不做重采样，时间轴是错的）

_print_help() {
  sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'HELPEOF'

全部参数:
  --date DATE            必填，主批次日期目录名（data/raw_custom/<DATE>/）
  --extra_date DATE:HZ   可重复传，额外合并的批次，HZ填该批次自己真实采样率
  --source_hz HZ         主批次自己真实的采样率（跟--extra_date配合用，不传
                          默认等于--hz，见上面用法示例的说明）
  --hz HZ                训练目标采样率（默认16）
  --model TYPE           rf(默认)/xgb/lgbm/catboost/extratrees/histgb
  --missing_strategy S   acc/gyro缺失值(蓝牙断联)处理方式，四选一:
                            none(默认)  原样保留NaN不处理（历史行为，树模型
                                        大多能容忍，DL不能，会loss变nan）
                            drop        丢弃含NaN的那一整行，不编造数值
                            ffill       前后值填充（编造数值，跟实时推理
                                        infer_csv_scratch.py处理方式一致）
                            drop_window 行级不处理，改成切窗口那一步整窗
                                        丢弃（哪怕只有一帧NaN也整窗不要，
                                        比drop更严格，见src/data/preprocess.py
                                        的--drop_nan_windows）
  --label_mode MODE      majority(默认，多数投票)/center(窗口中心帧)
  --window_s SEC         训练窗口长度秒数（默认用configs/data.yaml的2秒）
  --stride_s SEC         训练窗口步长秒数（默认用configs/data.yaml的1秒）
  --split_strategy S     random(默认)/subject/label_concat
  --train_ratio R        训练集比例（默认0.9）
  --val_ratio R          验证集比例（默认0.1）
  --test_ratio R         测试集比例（默认0）
  --n_aug N              每个原始片段生成的合成增强数量（默认50，配合--label用）
  --label LABEL          可重复传，要合成数据的类别（不传默认只合成"抓挠"）
  --skip_syn             跳过生成合成数据和方案B，只训练方案A(纯标注)
  --feat_workers N        特征提取并行进程数（默认1，传-1用全部核心）
  --tag TAG               输出目录后缀（不传=自动用missing_${MISSING_STRATEGY}，
                          显式传会覆盖这个自动值），同一个DATE想同时保留多个
                          版本时用
  --clean                 先删掉这个DATE+TAG对应的旧缓存再全新生成
  -h, --help              打印这份帮助
HELPEOF
}

# ── 解析参数 ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)        _print_help;          exit 0 ;;
    --date)           DATE="$2";           shift 2 ;;
    --hz)             HZ="$2";             shift 2 ;;
    --model)          MODEL_TYPE="$2";      shift 2 ;;
    --missing_strategy) MISSING_STRATEGY="$2"; shift 2 ;;
    --n_aug)          N_AUG="$2";          shift 2 ;;
    --label)          LABELS+=("$2");      shift 2 ;;
    --split_strategy) SPLIT_STRATEGY="$2"; shift 2 ;;
    --train_ratio)    TRAIN_RATIO="$2";    shift 2 ;;
    --val_ratio)      VAL_RATIO="$2";      shift 2 ;;
    --test_ratio)     TEST_RATIO="$2";     shift 2 ;;
    --label_mode)     LABEL_MODE="$2";     shift 2 ;;
    --stride_s)       STRIDE_S="$2";       shift 2 ;;
    --window_s)       WINDOW_S="$2";       shift 2 ;;
    --feat_workers)   FEAT_WORKERS="$2";   shift 2 ;;
    --tag)            TAG="$2";            shift 2 ;;
    --clean)          CLEAN=1;             shift 1 ;;
    --skip_syn)       SKIP_SYN=1;          shift 1 ;;
    --extra_date)     EXTRA_DATES+=("$2"); shift 2 ;;
    --source_hz)      SOURCE_HZ="$2";       shift 2 ;;
    *) echo "未知参数: $1（--help 查看全部参数）"; exit 1 ;;
  esac
done

for _ed in "${EXTRA_DATES[@]:-}"; do
  if [[ -n "$_ed" && "$_ed" != *:* ]]; then
    echo "[错误] --extra_date 格式应为 DATE:HZ（例: 2026_7_17-2026_7_29:16），收到: $_ed"
    exit 1
  fi
done

# 没显式传--tag时，自动用missing_strategy当后缀——不同missing_strategy
# 跑出来的train.npz本来就是不同的数据（none保留NaN/drop丢行/ffill填充/
# drop_window丢窗口，见--missing_strategy说明），不该共用同一个
# processed_dir互相覆盖。之前必须手动传--tag missing_xxx才能避免覆盖，
# 容易忘，干脆自动化：想同时对比几种策略，直接换--missing_strategy跑
# 就行，目录自动分开；真需要在同一个missing_strategy下再细分几个版本
# （比如同策略但--window_s不同），显式传--tag会覆盖这个自动值
if [[ -z "$TAG" ]]; then
  TAG="missing_${MISSING_STRATEGY}"
fi

if [[ ${#LABELS[@]} -eq 0 ]]; then
  LABELS=("抓挠")
fi

if [[ -z "$DATE" ]]; then
  echo "用法: bash train_custom.sh --date <DATE>  (例: --date 2026_7_23，--help 查看全部参数)"
  exit 1
fi

# drop_window是preprocess.py那边的概念（整窗丢弃），labelstudio_to_custom.py
# 本身不认这个值——它只负责行级处理，要传"none"让它原样保留NaN，不然NaN在
# 行级就被drop/ffill处理掉了，preprocess.py那边根本看不到、没法按窗口过滤
LS_MISSING_STRATEGY="$MISSING_STRATEGY"
DROP_NAN_WINDOWS_FLAG=""
if [[ "$MISSING_STRATEGY" == "drop_window" ]]; then
  LS_MISSING_STRATEGY="none"
  DROP_NAN_WINDOWS_FLAG="--drop_nan_windows"
fi

PROCESSED_DIR="data/processed_${DATE}${TAG:+_$TAG}"
DATA_DIR="data/raw_custom/${DATE}"
CSV="${DATA_DIR}/merged_${DATE}.csv"
JSON="${DATA_DIR}/merged_tmp.json"
# 每个要合成的类别各自一份npz文件，文件名带上类别名区分（多个类别时避免
# 互相覆盖）；SYNTHETIC_PATHS按LABELS顺序一一对应，后面--synthetic_spec
# 会拿LABELS[i]:SYNTHETIC_PATHS[i]拼起来传给train.py
SYNTHETIC_PATHS=()
for _lbl in "${LABELS[@]}"; do
  SYNTHETIC_PATHS+=("data/synthetic/scratch_${DATE}${TAG:+_$TAG}_${_lbl}.npz")
done
DATASET_TAG=$(basename "$PROCESSED_DIR")

# 训练日志放项目自己的tmp/目录下，不是系统/tmp——系统/tmp是全机器共用的，
# 日志文件名又没带DATASET_TAG，不同用户/不同次跑（尤其带不同--tag同时跑）
# 容易互相覆盖，日志内容对不上这次跑的是哪个数据集；放项目tmp/目录下、
# 文件名带上DATASET_TAG，各自独立，也方便跑完之后回头翻日志（不用记得
# 是哪次登录、哪个系统临时目录）
TMP_DIR="tmp"
mkdir -p "$TMP_DIR"
# 文件名带上$MODEL_TYPE——之前只带DATASET_TAG，同一份数据先跑rf再跑
# xgb，第二次会把第一次的训练日志覆盖掉，跟--clean误删结果目录是同一类
# 问题（模型类型没体现在产出文件名里）
LOG_NO_SYN="${TMP_DIR}/train_no_syn_${DATASET_TAG}_${MODEL_TYPE}.log"
LOG_WITH_SYN="${TMP_DIR}/train_with_syn_${DATASET_TAG}_${MODEL_TYPE}.log"
LOG_FULL="${TMP_DIR}/train_full_${DATASET_TAG}_${MODEL_TYPE}.log"

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
  # 只删这次要训练的模型类型($MODEL_TYPE)自己的输出子目录，不删整个
  # DATASET_TAG——train.py现在把模型类型作为out_dir最后一层子目录
  # （.../{hz}hz_remap.../{model}/ml_{model}.pkl），删整个DATASET_TAG会
  # 把之前用别的--model训练好、保留着的结果也一起删掉，之前踩过这个坑
  # （换个--model重新--clean跑一次，之前rf训练出来的模型文件全没了）
  # train.py的remap_tag是"_"+remap文件名(不含扩展名)，比如
  # configs/remap_custom_3class.yaml → "_remap_custom_3class"，拼出来是
  # "16hz_remap_custom_3class"——这里要跟train.py的拼法完全一致，不能
  # 多加"remap_"前缀
  remap_stem=$(basename "$REMAP" .yaml)
  for _variant in "${HZ}hz_${remap_stem}" "${HZ}hz_${remap_stem}_syn"; do
    _model_out_dir="$RESULTS_DIR/$DATASET_TAG/$_variant/$MODEL_TYPE"
    echo "  rm -rf $_model_out_dir"
    rm -rf "$_model_out_dir"
  done
  for _sp in "${SYNTHETIC_PATHS[@]}"; do
    echo "  rm -f  $_sp"
    rm -f "$_sp"
  done
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
    --missing_strategy "$LS_MISSING_STRATEGY" \
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
  MAIN_SOURCE_HZ="${SOURCE_HZ:-$HZ}"
  if [[ -z "$SOURCE_HZ" ]]; then
    echo ""
    echo "⚠ 未传 --source_hz，主数据(--date $DATE)会被当成已经是 ${HZ}Hz（不重采样）。" \
         "如果它实际采样率不是${HZ}Hz，加 --source_hz <实际采样率> 重跑，否则合并进去的" \
         "时间轴是错的。"
  fi
  # 每个额外批次的project json→merged_tmp.json、merged_tmp.json→CSV，不管
  # MERGED_CSV/MERGED_JSON缓存在不在，都要跑一遍（后面两个缓存判断各自
  # 独立，都依赖这里算出来的_ejson/_ecsv路径；如果这段只在MERGED_CSV
  # 缺失时才跑，MERGED_JSON单独失效时_ejson会是空的，合成数据那步就
  # 又会读不到额外批次）
  JSON_PARTS=("$JSON")
  CSV_PARTS=("${DATE}:${CSV}:${MAIN_SOURCE_HZ}")
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
        --missing_strategy "$LS_MISSING_STRATEGY" \
        --keep_labels
    fi

    if [[ ! -f "$_ecsv" ]]; then
      echo "[错误] 生成训练CSV失败: $_ecsv"
      exit 1
    fi

    JSON_PARTS+=("$_ejson")
    CSV_PARTS+=("${_edate}:${_ecsv}:${_ehz}")
  done

  # 合成数据(synthesize_scratch.py)之前只读主批次的JSON，额外批次里的
  # 抓挠真实标注根本没被用来生成合成数据——这里把所有批次的JSON也合并
  # 一份，后面生成合成数据时改用这份，让它能看到全部真实抓挠数据。
  MERGED_JSON="${DATA_DIR}/merged_tmp_${DATE}${TAG:+_$TAG}_combined.json"
  if [[ ! -f "$MERGED_JSON" || "$CLEAN" == "1" ]]; then
    echo ""
    echo "▶ 合并全部批次的JSON → $MERGED_JSON（给合成数据用，能看到全部真实抓挠数据）..."
    python -c "
import json, sys
merged = []
for f in sys.argv[1:]:
    merged += json.load(open(f, encoding='utf-8'))
json.dump(merged, open('${MERGED_JSON}', 'w'), ensure_ascii=False)
print(f'  合并完成，共 {len(merged)} 条任务 -> ${MERGED_JSON}')
" "${JSON_PARTS[@]}"
  fi
  JSON="$MERGED_JSON"

  MERGED_CSV="${DATA_DIR}/merged_${DATE}${TAG:+_$TAG}_combined.csv"
  if [[ ! -f "$MERGED_CSV" || "$CLEAN" == "1" ]]; then
    echo ""
    echo "▶ 步骤1.5：合并 ${#EXTRA_DATES[@]} 个额外批次 → $MERGED_CSV ..."
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
    # 重采样批次不会保留timestamp列（重采样后行数变了，原时间戳对不上），
    # 直通批次的CSV还带着原始timestamp列——两种批次concat到一起会让这一列
    # 一部分是NaN一部分是时间戳字符串，读混合CSV时pandas会报DtypeWarning。
    # loader_custom.py只认sensor_cols/label_col/record_id_col，不用
    # timestamp，这里统一丢掉，两边保持列一致
    df = df.drop(columns=['timestamp'], errors='ignore')
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

# ── 预处理（复用已有缓存，--clean才强制重新生成）─────────────────
# preprocess.py本身不认--clean/不会自己判断"要不要重新处理"，每次调用
# 都会无条件重新切分train/val/test.npz、顺带删掉ml_features.npz特征缓存
# （逼着train.py重新提特征）。之前这里没有跳过判断，每次跑（哪怕只是
# 换个--model重新训练、预处理这步的输出根本没变）都要重新预处理+重新
# 提特征，白白浪费好几分钟。预处理的产出（train/val/test.npz、特征）
# 只跟数据本身+hz/窗口/划分这些参数有关，跟--model完全无关，没必要
# 因为换模型就重新跑。跟前面步骤0/1一样，用文件是否已存在做跳过判断，
# --clean时才强制重新生成。
PREPROCESSED_MARKER="${PROCESSED_DIR}/${HZ}hz/train.npz"
if [[ -f "$PREPROCESSED_MARKER" && "$CLEAN" != "1" ]]; then
  echo ""
  echo "▶ 预处理数据：$PREPROCESSED_MARKER 已存在，跳过（换--model不需要重新预处理；"
  echo "  改过预处理/特征相关代码、或者数据有更新，加--clean强制重新生成）"
else
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
    $( [[ -n "$WINDOW_S" ]] && echo "--window_s $WINDOW_S" ) \
    $DROP_NAN_WINDOWS_FLAG \
    --hz "$HZ"
fi

# ── 方案 A：纯标注模型（后台运行）────────────────────────
echo ""
echo "▶ 方案 A：纯标注模型（后台运行）..."
python src/ml/train.py --hz "$HZ" --model "$MODEL_TYPE" \
  --processed_dir "$PROCESSED_DIR" \
  --remap "$REMAP" \
  --results_dir "$RESULTS_DIR" \
  --feat_workers "$FEAT_WORKERS" \
  > "$LOG_NO_SYN" 2>&1 &
PID_A=$!

if [[ "$SKIP_SYN" == "1" ]]; then
  echo ""
  echo "▶ --skip_syn：跳过生成合成数据和方案B，只训练方案A"
else
  # ── 生成合成数据（每个--label各自生成一份）──────────────────────────
  # --window_s/--stride_s要跟上面预处理真实数据用的保持一致，不然合成
  # 窗口跟真实窗口点数对不上，train.py合并训练集时特征维度会对不齐
  SYNTHETIC_SPEC_ARGS=()
  for _i in "${!LABELS[@]}"; do
    _lbl="${LABELS[$_i]}"
    _sp="${SYNTHETIC_PATHS[$_i]}"
    echo "▶ 生成合成数据（${_lbl}，n_aug=${N_AUG}）..."
    python src/data/synthesize_scratch.py \
      --json "$JSON" \
      --csv_dir "$CSV_DIR" \
      --output "$_sp" \
      --processed_dir "$PROCESSED_DIR" \
      --remap "$REMAP" \
      --label "$_lbl" \
      --hz "$HZ" \
      --n_aug "$N_AUG" \
      $( [[ -n "$STRIDE_S" ]] && echo "--stride_s $STRIDE_S" ) \
      $( [[ -n "$WINDOW_S" ]] && echo "--window_s $WINDOW_S" )
    SYNTHETIC_SPEC_ARGS+=(--synthetic_spec "${_lbl}:${_sp}")
  done

  # ── 方案 B：带合成数据模型（后台运行）───────────────────
  echo ""
  echo "▶ 方案 B：带合成数据模型（后台运行，合成类别: ${LABELS[*]}）..."
  python src/ml/train.py --hz "$HZ" --model "$MODEL_TYPE" \
    --processed_dir "$PROCESSED_DIR" \
    --remap "$REMAP" \
    "${SYNTHETIC_SPEC_ARGS[@]}" \
    --results_dir "$RESULTS_DIR" \
    --feat_workers "$FEAT_WORKERS" \
    > "$LOG_WITH_SYN" 2>&1 &
  PID_B=$!
fi

# ── 等待两个训练完成，实时把两边的日志打到当前终端（加[A]/[B]前缀区分）──
# 之前这里是纯等待，进度条/特征提取日志全写进/tmp的文件里，只能另开
# 终端手动tail -f才看得到；现在直接在这个终端里实时滚动显示，不用切窗口。
echo ""
if [[ "$SKIP_SYN" == "1" ]]; then
  echo "⏳ 等待方案A训练完成（下面实时滚动的是训练日志）..."
else
  echo "⏳ 等待两个模型训练完成（下面实时滚动的是训练日志，[A]=纯标注 [B]=带合成）..."
fi
# tqdm进度条是靠\r（回车不换行）原地刷新的，不是每次都换行；sed要等到\n
# 才会把攒的内容当一整行输出，\r的更新会一直卡在缓冲区里，直到最后进度条
# 跑完打印真正的\n才一次性冒出来，等于白等——这也是之前"卡住不动，跑完
# 才突然出现"的原因。用 tr 把\r也当成换行处理，每次进度更新都单独成行、
# 立刻输出，代价是不再是原地刷新的动画效果，而是一行行往下滚动，但是真
# 实时的，不用干等。
# 光加tr还不够：tr写到管道（不是终端）时默认是全缓冲的，会把转换后的内容
# 攒在自己的缓冲区里不立刻往下传，一样会卡住——用 stdbuf -oL 强制tr按行
# 缓冲，才能真正做到每次更新都立刻显示。
# "提取特征"这条进度条每秒刷新上百次，改行输出后一秒钟能刷几十行、刷屏
# 刷得根本看不清——限速成同一路（A/B各自算）的"提取特征"行至少间隔2秒
# 才打印一次，其它行（阶段提示、最终结果等）不受影响照常立刻打印。
# 这里特意没用awk实现限速：这台机器/usr/bin/awk是mawk，mawk从管道读
# 输入时是整段攒起来的，不是来一行处理一行，用它会导致进度条从"实时
# 但刷屏"变成"完全不显示、等到最后才一次性吐出来"——比刷屏还倒退回
# 了当初tr那个坑。改用纯bash的while read循环，实测是真的边读边处理。
_throttle() {
  local last=0 now
  while IFS= read -r line; do
    if [[ "$line" == *"提取特征"* ]]; then
      now=$(date +%s)
      if (( now - last < 2 )); then
        continue
      fi
      last=$now
    fi
    printf '%s\n' "$line"
  done
}
tail -f -n +1 "$LOG_NO_SYN" 2>/dev/null | stdbuf -oL tr '\r' '\n' | _throttle | sed -u 's/^/[A] /' &
TAIL_A=$!
if [[ "$SKIP_SYN" != "1" ]]; then
  tail -f -n +1 "$LOG_WITH_SYN" 2>/dev/null | stdbuf -oL tr '\r' '\n' | _throttle | sed -u 's/^/[B] /' &
  TAIL_B=$!
fi

wait $PID_A && echo "  ✅ 方案 A 完成" || echo "  ❌ 方案 A 失败，见 $LOG_NO_SYN"
if [[ "$SKIP_SYN" != "1" ]]; then
  wait $PID_B && echo "  ✅ 方案 B 完成" || echo "  ❌ 方案 B 失败，见 $LOG_WITH_SYN"
fi

kill "$TAIL_A" "${TAIL_B:-}" 2>/dev/null || true
wait "$TAIL_A" "${TAIL_B:-}" 2>/dev/null || true

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
if [[ "$SKIP_SYN" != "1" ]]; then
  echo ""
  echo "── 方案 B（带合成）──"
  _show_log "$LOG_WITH_SYN"
fi

echo ""
echo "模型路径:"
echo "  纯标注: ${RESULTS_DIR}/${DATASET_TAG}/${HZ}hz_remap_custom_3class/${MODEL_TYPE}/ml_${MODEL_TYPE}.pkl"
if [[ "$SKIP_SYN" != "1" ]]; then
  echo "  带合成: ${RESULTS_DIR}/${DATASET_TAG}/${HZ}hz_remap_custom_3class_syn/${MODEL_TYPE}/ml_${MODEL_TYPE}.pkl"
fi
