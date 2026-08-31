#!/usr/bin/env bash
# 批量推理所有日期的 CSV，输出抓挠识别结果，并生成 Label Studio 复查任务。
#
# 环境变量:
#   DATA_ROOT     CSV 数据根目录，按日期子目录组织（必填）
#   MODEL         ML 模型路径（必填）
#   RESULT_ROOT   推理结果输出目录（默认 infer_result）
#   TARGET_LABELS 要检测的目标类别，逗号分隔（默认"抓挠"，保持跟改造前完全
#                 一致的行为——不设这个变量的旧用法不受影响）。传多个类别
#                 比如"抓挠,甩身体"时，每个类别在 RESULT_ROOT/{day}/{label}/
#                 下单独产出一整套结果（by_conf_max/、_infer/、
#                 imu_daily_scratch_stats.csv、labelstudio_review_full_ml_IMU*.json
#                 等），互相独立、内容只统计各自那个类别的片段。CSV只读一次、
#                 特征只提一次（在infer_csv_scratch.py一次调用内部按类别拆分
#                 输出），不是每个类别重新跑一遍推理——那部分才是真正耗时的，
#                 类别数量不该线性拖慢它
#   EXCLUDE_DAYS  空格分隔的跳过日期列表（默认空）
#   WORKERS       并行进程数（默认 8）
#   PATTERN       CSV 文件名通配符（默认 *.csv）
#   DEVICE_HZ     设备采样率（默认 0=从模型元数据读取，等同于跟MODEL_HZ一样，也就是"不降采样"，
#                 喂raw等非16Hz原生数据时必须显式传对，不传的话不会报错但结果完全是错的）
#   MODEL_HZ      模型采样率（默认 0=与DEVICE_HZ相同）
#   RESAMPLE_METHOD  DEVICE_HZ != MODEL_HZ 时用哪种降采样算法（默认poly=scipy resample_poly，
#                 可选training_match=复刻witmotion_imu采集端当年生成训练数据用的算法）
#   LS_URL_PREFIX Label Studio CSV 的 URL 前缀（默认 http://localhost:8080/data/local-files/?d=raw_wit）
#   LS_MODE       Label Studio 任务模式: scratch_only/uncertain/all（默认 scratch_only）
#   CAM_MODE      Label Studio 任务固定生成几个videoN字段: auto/2/3（默认 auto，
#                 按每天实际探测到的机位数量走）。传2或3强制固定字段数量，要跟
#                 Label Studio项目里配置的Video组件个数一致——比如项目按3机位配置
#                 好了，但某天原始视频只拍了2个视角，传CAM_MODE=3能保证那天的任务
#                 也带上video3字段（空字符串占位，不拿别的机位画面顶替），不会因为
#                 缺字段导致那天的任务导入Label Studio报错
#   SPLIT_BY_IMU  除了生成混合所有IMU的主labelstudio_review.json，额外按触发
#                 检测的IMU拆分出labelstudio_review_IMU1.json/IMU2.json/...
#                 各一份，方便只导入某一条狗的复查任务（默认0=不拆，只生成
#                 主JSON，跟以前行为一致；传1开启）
#   MIN_CONF      除了生成完整JSON，额外按置信度下限再筛一份，逗号分隔可传多个
#                 阈值（比如"0.7"或"0.7,0.9"）——每个阈值都会另外生成
#                 labelstudio_review_{阈值}.json（SPLIT_BY_IMU=1时还会有
#                 labelstudio_review_IMU1_{阈值}.json等），只保留置信度分桶
#                 下界>=阈值的任务，完整版JSON照常生成，筛选版是额外多出来的，
#                 不是替代（默认空=不筛）
#   ML_PRELABEL   传1时，额外生成一份"全录制视频+ML自动预标注"的Label Studio
#                 任务，每个机位(IMU)一个文件(labelstudio_review_full_ml_IMU1.json,
#                 _IMU2.json, _IMU3.json)，各自用自己的CSV和检测结果，不裁剪clip，标注人标成
#                 ML，跟clips那一套完全独立、互不影响（默认0=不生成）。每个录制
#                 场次不管有没有检测到达标片段都会生成task(没检测到的标注结果
#                 为空)，让复查的人能看到全部录制数据，方便顺便核查模型有没有
#                 漏检，不是只导出命中的部分
#   ML_MIN_CONF   ML_PRELABEL=1或IMU_STATS=1时，只有置信度>=这个值的抓挠片段才算数
#                 （默认0.8，跟MIN_CONF是两回事——MIN_CONF是给clips模式筛任务文件
#                 用的，这个是给ML预标注/IMU统计筛"算不算抓挠"用的，两处共用同一个
#                 阈值和字段，保证网页上看到的C值统计跟Label Studio里实际标注出来
#                 的片段是同一批，不会对不上）
#   ML_CONF_FIELD ML_PRELABEL=1或IMU_STATS=1时，用conf_max还是conf_mean判断置信度
#                 达没达标（默认conf_mean——比conf_max更稳，误报更少；conf_max更
#                 容易找出漏检但也更容易把噪声算进去，想换回conf_max就显式传这个）
#   IMU_STATS     传1时，全部天推理完之后额外跑一遍src/imu_scratch_daily_stats.py，
#                 用ML_MIN_CONF/ML_CONF_FIELD同一套置信度标准筛"算不算抓挠"，统计
#                 每天每个机位(IMU)的抓挠次数/时长/聚集/持续/中断等，每天各产出一份
#                 CSV(RESULT_ROOT/{day}/imu_daily_scratch_stats.csv)，供pm_skin_scoring
#                 网页的「C值计算」标签按日期+IMU读取自动填充（默认0=不生成）。
#                 只会写本次INCLUDE_DAYS那几天的统计文件，不碰别的日期目录；
#                 但基线是拿RESULT_ROOT下全部已跑过推理的天算的，所以哪怕只跑
#                 一天，基线也是有历史可比的（前提是之前跑过别的天）
#   ENCODER       裁剪片段用的编码器: cpu（默认，libx264软编码，兼容性最好）或
#                 cuda（h264_nvenc GPU硬编码，明显更快，但历史上部分浏览器/Label Studio
#                 播放有兼容性问题，不确定现在还有没有——想试就传 ENCODER=cuda，裁完先在
#                 Label Studio里实际播放确认没问题，有问题下次不传这个变量就自动改回cpu）
#   PRESET        --encoder cpu 时libx264的preset（默认veryfast，跟以前行为一致）。
#                 实测同款其他参数下 veryfast→superfast 提速约31.6%，profile/level/pix_fmt
#                 三档preset输出一致，不影响Label Studio兼容性，建议优先试 PRESET=superfast；
#                 ultrafast能再提速但快速动作画面出现宏块伪影概率略高，建议先抽查几个clip画面
#   FFMPEG_THREADS --encoder cpu 时单个ffmpeg进程内部线程数（默认2，跟以前行为一致）。
#                 要配合CLIP_WORKERS一起调（两者乘积是总CPU线程需求，别明显超过nproc），
#                 P核/E核混合架构上建议实测，比如 FFMPEG_THREADS=1 配合更高的CLIP_WORKERS
#
# 用法:
#   DATA_ROOT=data/multicam_multiimu EXCLUDE_DAYS="test" \
#     MODEL=results/processed_2026_7_23/16hz_remap_custom_3class/ml_rf.pkl \
#     WORKERS=16 RESULT_ROOT=infer_result \
#     ./run_review_bins_all_days.sh

set -e

# 全靠环境变量驱动、没有位置参数，之前没接-h/--help，直接跑到下面
# DATA_ROOT/MODEL的必填检查就退出了，看不到任何说明——退出码非0，
# 用${VAR:?...}这种语法在没设-h专门判断时没法优雅打印帮助。这里在
# 必填检查前拦一下，打印文件头这段环境变量说明(sed提取第2~82行的
# 注释块，跟train_custom.sh的_print_help()是同一个套路)
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,82p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

# sklearn新版本把force_all_finite改名成ensure_all_finite了，旧版本
# 训出来的模型/依赖库内部还在用旧参数名，每个worker进程都会打印一遍
# FutureWarning，WORKERS开多的时候刷屏刷得很厉害——纯粹是版本过渡期的
# 噪音，不影响推理结果，默认压掉。只在没有显式设置PYTHONWARNINGS时才
# 给默认值，不覆盖用户自己的设置（比如想看到别的警告）
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::FutureWarning}"

# ── 参数与默认值 ──────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:?请设置 DATA_ROOT 环境变量，例: DATA_ROOT=data/multicam_multiimu}"
MODEL="${MODEL:?请设置 MODEL 环境变量，例: MODEL=results/processed_2026_7_23/16hz_remap_custom_3class/ml_rf.pkl}"
# 支持传通配符（比如某个目录下只有一个.pkl，懒得写全名的话直接
# MODEL="results/.../lgbm/*.pkl"）——shopt nullglob避免匹配不到时
# $MODEL 原样当成字面量传给python报"文件不存在"这种更难懂的报错，
# 直接在这里就报清楚到底匹配到几个文件
if [[ "$MODEL" == *'*'* ]]; then
  shopt -s nullglob
  _model_matches=($MODEL)
  shopt -u nullglob
  if [[ ${#_model_matches[@]} -eq 0 ]]; then
    echo "[错误] MODEL 通配符 \"$MODEL\" 没有匹配到任何文件"
    exit 1
  elif [[ ${#_model_matches[@]} -gt 1 ]]; then
    echo "[错误] MODEL 通配符 \"$MODEL\" 匹配到多个文件，写具体文件名指定用哪个:"
    printf '  %s\n' "${_model_matches[@]}"
    exit 1
  fi
  MODEL="${_model_matches[0]}"
fi
RESULT_ROOT="${RESULT_ROOT:-infer_result}"
TARGET_LABELS="${TARGET_LABELS:-抓挠}"   # 逗号分隔，默认只有"抓挠"，向后兼容旧用法
EXCLUDE_DAYS="${EXCLUDE_DAYS:-}"
WORKERS="${WORKERS:-8}"
PATTERN="${PATTERN:-*.csv}"
DEVICE_HZ="${DEVICE_HZ:-0}"
MODEL_HZ="${MODEL_HZ:-0}"
RESAMPLE_METHOD="${RESAMPLE_METHOD:-poly}"  # poly（默认，scipy resample_poly）或 training_match
                                             # （复刻witmotion_imu采集端当年生成训练数据用的滑动平均+
                                             # 线性插值算法）。DEVICE_HZ==MODEL_HZ时这个参数不生效
                                             # （不会做任何重采样），只有喂raw等非16Hz原生数据时才有意义。
                                             # 见 src/eval/compare_resample_methods.py
LS_URL_PREFIX="${LS_URL_PREFIX:-http://192.168.2.140:8182}"
LS_VIDEO_URL_PREFIX="${LS_VIDEO_URL_PREFIX:-}"   # 默认为 LS_URL_PREFIX/transcoded
LS_MODE="${LS_MODE:-scratch_only}"
CAM_MODE="${CAM_MODE:-auto}"
SPLIT_BY_IMU="${SPLIT_BY_IMU:-0}"
MIN_CONF="${MIN_CONF:-}"
ML_PRELABEL="${ML_PRELABEL:-0}"
ML_MIN_CONF="${ML_MIN_CONF:-0.8}"
ML_CONF_FIELD="${ML_CONF_FIELD:-conf_mean}"
IMU_STATS="${IMU_STATS:-0}"
CONTEXT_S="${CONTEXT_S:-3}"        # 片段前后保留秒数
MERGE_GAP="${MERGE_GAP:-1}"            # 合并相邻抓挠片段的最大间隔秒数（默认1s，event_eval.py 验证过
                                        # 3s会导致约一半真实事件被错误合并，1s已能消除碎片化且合并更少）
MIN_WINDOWS="${MIN_WINDOWS:-1}"        # 片段最少窗口数，不足则丢弃（默认1=不过滤）
KEEP_ISOLATED="${KEEP_ISOLATED:-1}"    # 是否保留孤立单窗口片段（默认1=保留）
BIN_BY="${BIN_BY:-conf_max}"           # 置信度分桶依据：conf_max（默认）或 conf_mean
EXTRACT_CLIPS="${EXTRACT_CLIPS:-1}" # 是否裁剪视频（0=跳过）
MEDIA_DIR="${MEDIA_DIR:-$HOME/label_infra/data/media}"  # Nginx 媒体目录（软链接目标）
SYMLINK_CSV="${SYMLINK_CSV:-1}"     # 是否自动为 CSV 创建软链接（0=跳过）
INCLUDE_DAYS="${INCLUDE_DAYS:-}"    # 空格分隔的白名单，非空时只处理列出的日期

# ── 构建排除/包含集合 ─────────────────────────────────────
declare -A EXCLUDE_SET
for day in $EXCLUDE_DAYS; do
    EXCLUDE_SET["$day"]=1
done
declare -A INCLUDE_SET
for day in $INCLUDE_DAYS; do
    INCLUDE_SET["$day"]=1
done

echo "=============================================="
echo "  批量推理"
echo "  数据根目录: $DATA_ROOT"
echo "  模型: $MODEL"
echo "  结果目录: $RESULT_ROOT"
echo "  仅处理: ${INCLUDE_DAYS:-（全部）}"
echo "  排除: ${EXCLUDE_DAYS:-（无）}"
echo "=============================================="

# ── 收集所有日期目录 ──────────────────────────────────────
days=()
for d in "$DATA_ROOT"/*/; do
    day=$(basename "$d")
    # 白名单过滤（INCLUDE_DAYS 非空时生效）
    if [[ ${#INCLUDE_SET[@]} -gt 0 && -z "${INCLUDE_SET[$day]}" ]]; then
        continue
    fi
    if [[ -n "${EXCLUDE_SET[$day]}" ]]; then
        echo "  跳过: $day"
        continue
    fi
    # 检查目录下是否有匹配的 CSV
    n=$(find "$d" -maxdepth 1 -name "$PATTERN" 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then
        echo "  跳过: $day（无 $PATTERN 文件）"
        continue
    fi
    days+=("$day")
done

if [[ ${#days[@]} -eq 0 ]]; then
    echo "[错误] 没有找到有效的日期目录"
    exit 1
fi

echo ""
echo "共 ${#days[@]} 个日期: ${days[*]}"
echo ""

# ── 目标类别列表 ──────────────────────────────────────────
# 逗号分隔转数组，下面所有按类别产出的步骤都复用这一份，任何步骤要新增
# 类别时只用改 TARGET_LABELS 这一个环境变量
IFS=',' read -ra labels <<< "$TARGET_LABELS"
echo "  目标类别: ${labels[*]}"
echo ""

# ── 逐日期推理 ────────────────────────────────────────────
# CSV只读一次、特征只提一次、模型只预测一次——多个类别的片段提取/合并/JSON
# 输出都在 infer_csv_scratch.py 一次调用内部按类别拆分完成（--target_labels），
# 不是每个类别重跑一次完整推理，避免类别数量线性拖慢这一步（这是整个流水线
# 里最耗时的部分）
hz_args=""
[[ "$DEVICE_HZ" -gt 0 ]] && hz_args="$hz_args --device_hz $DEVICE_HZ"
[[ "$MODEL_HZ"  -gt 0 ]] && hz_args="$hz_args --model_hz $MODEL_HZ"

for day in "${days[@]}"; do
    csv_dir="$DATA_ROOT/$day"
    out_dir="$RESULT_ROOT/$day"
    mkdir -p "$out_dir"

    echo "▶ $day ..."
    # --output_dir 传的是"这一天"这一级目录，infer_csv_scratch.py 自己按
    # --target_labels 把每个类别的结果分别落在 out_dir/{label}/_infer/ 下
    combined_log="$out_dir/.infer_combined.log"
    python src/infer_csv_scratch.py \
        --csv_dir "$csv_dir" \
        --pattern "$PATTERN" \
        --model "$MODEL" \
        --workers "$WORKERS" \
        --output_dir "$out_dir" \
        --target_labels "$TARGET_LABELS" \
        --quiet \
        --scratch_only \
        --merge_gap "$MERGE_GAP" \
        --min_windows "$MIN_WINDOWS" \
        --resample_method "$RESAMPLE_METHOD" \
        $( [[ "$KEEP_ISOLATED" == "0" ]] && echo "--no_keep_isolated" ) \
        $hz_args \
        2>&1 | tee "$combined_log"

    # infer.log 按类别各存一份——单次python调用的日志本身已经用[label]标记
    # 区分了不同类别的输出，复制同一份日志到每个类别目录下，保持"每个类别
    # 子目录下都能看到当天完整推理日志"这个结构，不用为拆日志重新跑多次
    for label in "${labels[@]}"; do
        mkdir -p "$out_dir/$label"
        cp "$combined_log" "$out_dir/$label/infer.log"
    done
    rm -f "$combined_log"

    echo "  结果已保存至 $out_dir/"
done

# ── 裁剪视频片段 ─────────────────────────────────────────
# 每个目标类别各裁一份，clip落在 out_dir/{label}/ 下——每个类别的_infer.json
# 只含它自己那个类别的片段（infer_csv_scratch.py已经按类别拆好），extract_clips.py
# 本身不认label，只是读传给它的--infer_dir/--output_dir，所以按类别各调一次
# 就能天然得到按类别分开的clips_*/输出，不需要改extract_clips.py本身
if [[ "$EXTRACT_CLIPS" == "1" ]]; then
    echo ""
    echo "▶ 按置信度区间裁剪视频片段..."
    for day in "${days[@]}"; do
        video_dir="$DATA_ROOT/$day"
        for label in "${labels[@]}"; do
            label_dir="$RESULT_ROOT/$day/$label"
            echo "  $day [$label] ..."
            # --run_tag 自动用 RESULT_ROOT 的名字，保证不同次运行（比如对比
            # 带合成/不带合成两个模型）裁出来的clip文件名不会撞在一起——如果
            # 后面还会把clip复制到共享的Nginx媒体目录，撞名会导致后一次运行
            # 静默覆盖前一次的文件，看似还在但内容已经被换掉了
            python src/extract_clips.py \
                --infer_dir  "$label_dir/_infer" \
                --video_dir  "$video_dir" \
                --output_dir "$label_dir" \
                --context_s  "$CONTEXT_S" \
                --workers    "${CLIP_WORKERS:-4}" \
                --bin_by     "$BIN_BY" \
                --encoder    "${ENCODER:-cpu}" \
                --preset     "${PRESET:-veryfast}" \
                --ffmpeg_threads "${FFMPEG_THREADS:-2}" \
                --cam_mode   "$CAM_MODE" \
                --run_tag    "$(basename "$RESULT_ROOT")"
        done
    done
fi

# ── 生成 Label Studio 复查任务 ────────────────────────────
# BIN_BY=both 时，clips_*/ 分别在 out_dir/by_conf_max/ 和 out_dir/by_conf_mean/
# 两个子目录下（extract_clips.py --bin_by both 的输出结构），不再直接在 out_dir
# 下面，所以要分别对这两个子目录各生成一份 labelstudio_review.json；
# 单一分桶模式(conf_max/conf_mean)行为不变，还是 out_dir 下一份。
echo ""
echo "▶ 生成 Label Studio 复查任务..."

video_prefix_arg=""
[[ -n "$LS_VIDEO_URL_PREFIX" ]] && video_prefix_arg="--video_url_prefix $LS_VIDEO_URL_PREFIX"

for day in "${days[@]}"; do
    for label in "${labels[@]}"; do
        label_dir="$RESULT_ROOT/$day/$label"

        if [[ "$BIN_BY" == "both" ]]; then
            clip_subdirs=("by_conf_max" "by_conf_mean")
        else
            clip_subdirs=("")
        fi

        split_arg=""
        [[ "$SPLIT_BY_IMU" == "1" ]] && split_arg="--split_by_imu"
        min_conf_arg=""
        [[ -n "$MIN_CONF" ]] && min_conf_arg="--min_conf $MIN_CONF"

        for sub in "${clip_subdirs[@]}"; do
            scan_dir="$label_dir${sub:+/$sub}"
            ls_json="$scan_dir/labelstudio_review.json"
            python src/review_to_labelstudio.py \
                --infer_dir "$scan_dir" \
                --output "$ls_json" \
                --csv_url_prefix "$LS_URL_PREFIX" \
                $video_prefix_arg \
                --use_clips \
                --cam_mode "$CAM_MODE" \
                $split_arg \
                $min_conf_arg \
                --mode "$LS_MODE" \
                --label "$label"
            echo "  $day [$label]${sub:+ [$sub]} → $ls_json"
        done
    done
done

# ── ML自动预标注（全录制视频，不裁剪clip）────────────────
# 跟上面clips那一套完全独立：读label_dir/_infer下的原始推理结果，不依赖
# extract_clips.py的输出，只标注高置信度片段，标注人标成ML；每个目标类别
# 各自一份，落在各自的 RESULT_ROOT/{day}/{label}/ 下
if [[ "$ML_PRELABEL" == "1" ]]; then
    echo ""
    echo "▶ 生成 ML 自动预标注任务（全录制视频，$ML_CONF_FIELD>=$ML_MIN_CONF）..."
    for day in "${days[@]}"; do
        for label in "${labels[@]}"; do
            label_dir="$RESULT_ROOT/$day/$label"
            ls_json="$label_dir/labelstudio_review.json"
            python src/review_to_labelstudio.py \
                --infer_dir "$label_dir/_infer" \
                --output "$ls_json" \
                --csv_url_prefix "$LS_URL_PREFIX" \
                $video_prefix_arg \
                --ml_full_video \
                --ml_min_conf "$ML_MIN_CONF" \
                --ml_conf_field "$ML_CONF_FIELD" \
                --cam_mode "$CAM_MODE" \
                --label "$label"
            echo "  $day [$label] → ${ls_json%.json}_full_ml_IMU{1,2,3}.json（按机位各自一份，视CAM_MODE而定）"
        done
    done
fi

# ── 每天每个IMU的抓挠统计（给pm_skin_scoring网页的C值计算用）──────────
# 每个目标类别各跑一次imu_scratch_daily_stats.py（--label区分），输出CSV
# 落在各自的 RESULT_ROOT/{day}/{label}/imu_daily_scratch_stats.csv 下，
# 不同类别的事件次数/时长互相独立统计，不会混在一起
if [[ "$IMU_STATS" == "1" ]]; then
    echo ""
    echo "▶ 统计每天每个IMU的情况（$ML_CONF_FIELD>=$ML_MIN_CONF才算数，供C值计算网页读取）..."
    # 只写本次实际跑的那几天的统计文件——基线那边会自己去读root下全部
    # 已有的天(见compute_stats的注释)，但"写哪几天的文件"必须严格限制在
    # 本次INCLUDE_DAYS，不然会在所有历史日期目录下都生成一份统计文件，
    # 网页上就分不清"哪些天是我真的处理过的"了
    days_arg=$(IFS=,; echo "${days[*]}")
    for label in "${labels[@]}"; do
      echo "  类别: $label"
      python src/imu_scratch_daily_stats.py \
        --infer_root "$RESULT_ROOT" \
        --days "$days_arg" \
        --label "$label" \
        --min_conf "$ML_MIN_CONF" \
        --conf_field "$ML_CONF_FIELD"
    done
fi

# ── 复制 CSV/MP4 到 Nginx 媒体目录 ──────────────────────
# 注意：Nginx 在 Docker 容器内运行，软链接目标不可见，需复制实体文件
if [[ "$SYMLINK_CSV" == "1" ]]; then
    echo ""
    echo "▶ 复制 CSV/MP4 到 Nginx 媒体目录 ($MEDIA_DIR)..."
    mkdir -p "$MEDIA_DIR"
    mkdir -p "$MEDIA_DIR/transcoded"
    n_copied=0
    n_skip=0

    _copy_file() {
        local src="$1" dst="$2"
        # 始终覆盖，避免旧的损坏文件留在 Nginx 目录
        [[ -L "$dst" ]] && rm "$dst"
        cp -f "$src" "$dst"
        n_copied=$((n_copied + 1))
    }

    for day in "${days[@]}"; do
        # clips现在按类别分目录（RESULT_ROOT/{day}/{label}/），每个类别各扫
        # 一遍自己的clips_*/——不同类别裁出来的clip文件名各不相同（片段时间
        # 段不一样），不会重复拷贝，扫多个label目录也不会把n_copied算重复
        for label in "${labels[@]}"; do
            label_dir="$RESULT_ROOT/$day/$label"
            # BIN_BY=both 时只扫 by_conf_max/（canonical，实际文件所在地），
            # by_conf_mean/ 下都是指向同一批文件的软链接，同名文件复制一次就够，
            # 扫两遍只会重复拷贝相同内容、把 n_copied 算重复。
            if [[ "$BIN_BY" == "both" ]]; then
                clips_root="$label_dir/by_conf_max"
            else
                clips_root="$label_dir"
            fi
            # clips_*/ 里的 CSV → MEDIA_DIR/，MP4 → MEDIA_DIR/transcoded/
            while IFS= read -r -d '' f; do
                ext="${f##*.}"
                if [[ "${ext,,}" == "csv" ]]; then
                    _copy_file "$f" "$MEDIA_DIR/$(basename "$f")"
                else
                    _copy_file "$f" "$MEDIA_DIR/transcoded/$(basename "$f")"
                fi
            done < <(find "$clips_root"/clips_* -maxdepth 1 \( -name "*.csv" -o -name "*.mp4" -o -name "*.MP4" \) -print0 2>/dev/null)
        done

        # ML_PRELABEL引用的是原始完整录制视频(不是裁剪出来的clip)，上面
        # 那段只同步了clips_*/里的文件，这里额外把DATA_ROOT/$day下的原始
        # CSV/MP4也同步一份过去，不然ML预标注JSON里的URL会404（同样的
        # 文件名，同步一次两边都能用，不冲突）。这一步跟label无关（原始
        # 录制文件对所有类别都是同一批），只需要在day层级做一次，不要放进
        # 上面的label循环里重复同步同一批文件。
        if [[ "$ML_PRELABEL" == "1" ]]; then
            while IFS= read -r -d '' f; do
                ext="${f##*.}"
                if [[ "${ext,,}" == "csv" ]]; then
                    _copy_file "$f" "$MEDIA_DIR/$(basename "$f")"
                else
                    _copy_file "$f" "$MEDIA_DIR/transcoded/$(basename "$f")"
                fi
            done < <(find "$DATA_ROOT/$day" -maxdepth 1 \( -name "*.csv" -o -name "*.mp4" -o -name "*.MP4" \) -print0 2>/dev/null)
        fi
    done
    echo "  新复制: $n_copied 个，已存在跳过: $n_skip 个"
fi

# ── 汇总所有日期 ──────────────────────────────────────────
echo ""
echo "=============================================="
echo "  汇总"
echo "=============================================="
# 按类别分开汇总——不同类别的片段数不该加在一起变成一个模糊的总数，
# 每个类别各自一行才能看出"这个类别检出了多少"
for label in "${labels[@]}"; do
    echo "  【$label】"
    total_scratch=0
    total_files=0
    for day in "${days[@]}"; do
        label_dir="$RESULT_ROOT/$day/$label"
        n_files=$(find "$label_dir" -name "*_infer.json" 2>/dev/null | wc -l)
        n_scratch=$(python -c "
import glob, json, sys
total = 0
for f in glob.glob('$label_dir/**/*_infer.json', recursive=True):
    d = json.load(open(f))
    total += len(d.get('scratch_segments', []))
print(total)
" 2>/dev/null || echo 0)
        echo "    $day: $n_files 个文件，检测到 $n_scratch 段"
        total_scratch=$((total_scratch + n_scratch))
        total_files=$((total_files + n_files))
    done
    echo "    ────────────────"
    echo "    合计: $total_files 个文件，$total_scratch 段"
done
echo ""
echo "Label Studio 导入方式:"
if [[ "$BIN_BY" == "both" ]]; then
    echo "  BIN_BY=both：每个日期/类别目录下有两份，分别导入："
    echo "    <日期>/<类别>/by_conf_max/labelstudio_review.json"
    echo "    <日期>/<类别>/by_conf_mean/labelstudio_review.json"
else
    echo "  每个日期/类别目录下的 labelstudio_review.json 可直接导入 Label Studio"
fi
echo "  Label Studio → Import → 选择 JSON 文件"
echo ""
echo "完成！结果目录: $RESULT_ROOT"
