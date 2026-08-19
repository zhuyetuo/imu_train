#!/usr/bin/env bash
# 批量推理所有日期的 CSV，输出抓挠识别结果，并生成 Label Studio 复查任务。
#
# 环境变量:
#   DATA_ROOT     CSV 数据根目录，按日期子目录组织（必填）
#   MODEL         ML 模型路径（必填）
#   RESULT_ROOT   推理结果输出目录（默认 infer_result）
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

# ── 参数与默认值 ──────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:?请设置 DATA_ROOT 环境变量，例: DATA_ROOT=data/multicam_multiimu}"
MODEL="${MODEL:?请设置 MODEL 环境变量，例: MODEL=results/processed_2026_7_23/16hz_remap_custom_3class/ml_rf.pkl}"
RESULT_ROOT="${RESULT_ROOT:-infer_result}"
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

# ── 逐日期推理 ────────────────────────────────────────────
hz_args=""
[[ "$DEVICE_HZ" -gt 0 ]] && hz_args="$hz_args --device_hz $DEVICE_HZ"
[[ "$MODEL_HZ"  -gt 0 ]] && hz_args="$hz_args --model_hz $MODEL_HZ"

for day in "${days[@]}"; do
    csv_dir="$DATA_ROOT/$day"
    out_dir="$RESULT_ROOT/$day"
    mkdir -p "$out_dir"

    infer_json_dir="$out_dir/_infer"
    mkdir -p "$infer_json_dir"
    echo "▶ $day ..."
    python src/infer_csv_scratch.py \
        --csv_dir "$csv_dir" \
        --pattern "$PATTERN" \
        --model "$MODEL" \
        --workers "$WORKERS" \
        --output_dir "$infer_json_dir" \
        --quiet \
        --scratch_only \
        --merge_gap "$MERGE_GAP" \
        --min_windows "$MIN_WINDOWS" \
        --resample_method "$RESAMPLE_METHOD" \
        $( [[ "$KEEP_ISOLATED" == "0" ]] && echo "--no_keep_isolated" ) \
        $hz_args \
        2>&1 | tee "$out_dir/infer.log"

    echo "  结果已保存至 $out_dir/"
done

# ── 裁剪视频片段 ─────────────────────────────────────────
if [[ "$EXTRACT_CLIPS" == "1" ]]; then
    echo ""
    echo "▶ 按置信度区间裁剪视频片段..."
    for day in "${days[@]}"; do
        out_dir="$RESULT_ROOT/$day"
        video_dir="$DATA_ROOT/$day"
        echo "  $day ..."
        # --run_tag 自动用 RESULT_ROOT 的名字，保证不同次运行（比如对比
        # 带合成/不带合成两个模型）裁出来的clip文件名不会撞在一起——如果
        # 后面还会把clip复制到共享的Nginx媒体目录，撞名会导致后一次运行
        # 静默覆盖前一次的文件，看似还在但内容已经被换掉了
        python src/extract_clips.py \
            --infer_dir  "$out_dir/_infer" \
            --video_dir  "$video_dir" \
            --output_dir "$out_dir" \
            --context_s  "$CONTEXT_S" \
            --workers    "${CLIP_WORKERS:-4}" \
            --bin_by     "$BIN_BY" \
            --encoder    "${ENCODER:-cpu}" \
            --preset     "${PRESET:-veryfast}" \
            --ffmpeg_threads "${FFMPEG_THREADS:-2}" \
            --cam_mode   "$CAM_MODE" \
            --run_tag    "$(basename "$RESULT_ROOT")"
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
    out_dir="$RESULT_ROOT/$day"

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
        scan_dir="$out_dir${sub:+/$sub}"
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
            --mode "$LS_MODE"
        echo "  $day${sub:+ [$sub]} → $ls_json"
    done
done

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
        out_dir="$RESULT_ROOT/$day"
        # BIN_BY=both 时只扫 by_conf_max/（canonical，实际文件所在地），
        # by_conf_mean/ 下都是指向同一批文件的软链接，同名文件复制一次就够，
        # 扫两遍只会重复拷贝相同内容、把 n_copied 算重复。
        if [[ "$BIN_BY" == "both" ]]; then
            clips_root="$out_dir/by_conf_max"
        else
            clips_root="$out_dir"
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
    echo "  新复制: $n_copied 个，已存在跳过: $n_skip 个"
fi

# ── 汇总所有日期 ──────────────────────────────────────────
echo ""
echo "=============================================="
echo "  汇总"
echo "=============================================="
total_scratch=0
total_files=0
for day in "${days[@]}"; do
    out_dir="$RESULT_ROOT/$day"
    n_files=$(find "$out_dir" -name "*_infer.json" | wc -l)
    n_scratch=$(python -c "
import glob, json, sys
total = 0
for f in glob.glob('$out_dir/**/*_infer.json', recursive=True):
    d = json.load(open(f))
    total += len(d.get('scratch_segments', []))
print(total)
" 2>/dev/null || echo 0)
    echo "  $day: $n_files 个文件，检测到 $n_scratch 段抓挠"
    total_scratch=$((total_scratch + n_scratch))
    total_files=$((total_files + n_files))
done
echo "  ────────────────"
echo "  合计: $total_files 个文件，$total_scratch 段抓挠"
echo ""
echo "Label Studio 导入方式:"
if [[ "$BIN_BY" == "both" ]]; then
    echo "  BIN_BY=both：每个日期目录下有两份，分别导入："
    echo "    <日期>/by_conf_max/labelstudio_review.json"
    echo "    <日期>/by_conf_mean/labelstudio_review.json"
else
    echo "  每个日期目录下的 labelstudio_review.json 可直接导入 Label Studio"
fi
echo "  Label Studio → Import → 选择 JSON 文件"
echo ""
echo "完成！结果目录: $RESULT_ROOT"
