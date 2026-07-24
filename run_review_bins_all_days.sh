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
#   DEVICE_HZ     设备采样率（默认 0=从模型元数据读取）
#   MODEL_HZ      模型采样率（默认 0=与DEVICE_HZ相同）
#   LS_URL_PREFIX Label Studio CSV 的 URL 前缀（默认 http://localhost:8080/data/local-files/?d=raw_wit）
#   LS_MODE       Label Studio 任务模式: scratch_only/uncertain/all（默认 scratch_only）
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
LS_URL_PREFIX="${LS_URL_PREFIX:-http://192.168.2.140:8182}"
LS_VIDEO_URL_PREFIX="${LS_VIDEO_URL_PREFIX:-}"   # 默认为 LS_URL_PREFIX/transcoded
LS_MODE="${LS_MODE:-scratch_only}"
CONTEXT_S="${CONTEXT_S:-3}"        # 片段前后保留秒数
EXTRACT_CLIPS="${EXTRACT_CLIPS:-1}" # 是否裁剪视频（0=跳过）
MEDIA_DIR="${MEDIA_DIR:-$HOME/label_infra/data/media}"  # Nginx 媒体目录（软链接目标）
SYMLINK_CSV="${SYMLINK_CSV:-1}"     # 是否自动为 CSV 创建软链接（0=跳过）

# ── 构建排除集合 ──────────────────────────────────────────
declare -A EXCLUDE_SET
for day in $EXCLUDE_DAYS; do
    EXCLUDE_SET["$day"]=1
done

echo "=============================================="
echo "  批量推理"
echo "  数据根目录: $DATA_ROOT"
echo "  模型: $MODEL"
echo "  结果目录: $RESULT_ROOT"
echo "  排除: ${EXCLUDE_DAYS:-（无）}"
echo "=============================================="

# ── 收集所有日期目录 ──────────────────────────────────────
days=()
for d in "$DATA_ROOT"/*/; do
    day=$(basename "$d")
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
        python src/extract_clips.py \
            --infer_dir  "$out_dir/_infer" \
            --video_dir  "$video_dir" \
            --output_dir "$out_dir" \
            --context_s  "$CONTEXT_S"
    done
fi

# ── 生成 Label Studio 复查任务 ────────────────────────────
echo ""
echo "▶ 生成 Label Studio 复查任务..."

for day in "${days[@]}"; do
    out_dir="$RESULT_ROOT/$day"
    ls_json="$out_dir/labelstudio_review.json"

    video_prefix_arg=""
    [[ -n "$LS_VIDEO_URL_PREFIX" ]] && video_prefix_arg="--video_url_prefix $LS_VIDEO_URL_PREFIX"
    python src/review_to_labelstudio.py \
        --infer_dir "$out_dir/_infer" \
        --output "$ls_json" \
        --csv_url_prefix "$LS_URL_PREFIX" \
        $video_prefix_arg \
        --mode "$LS_MODE"

    echo "  $day → $ls_json"
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
        if [[ -f "$dst" ]]; then
            n_skip=$((n_skip + 1))
        else
            [[ -L "$dst" ]] && rm "$dst"
            cp "$src" "$dst"
            n_copied=$((n_copied + 1))
        fi
    }

    for day in "${days[@]}"; do
        csv_dir="$DATA_ROOT/$day"
        # CSV → MEDIA_DIR/
        while IFS= read -r -d '' f; do
            _copy_file "$f" "$MEDIA_DIR/$(basename "$f")"
        done < <(find "$csv_dir" -maxdepth 1 -name "$PATTERN" -print0 2>/dev/null)
        # MP4 → MEDIA_DIR/transcoded/
        while IFS= read -r -d '' f; do
            _copy_file "$f" "$MEDIA_DIR/transcoded/$(basename "$f")"
        done < <(find "$csv_dir" -maxdepth 1 \( -name "*.mp4" -o -name "*.MP4" \) -print0 2>/dev/null)
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
for f in glob.glob('$out_dir/*_infer.json'):
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
echo "  每个日期目录下的 labelstudio_review.json 可直接导入 Label Studio"
echo "  Label Studio → Import → 选择 JSON 文件"
echo ""
echo "完成！结果目录: $RESULT_ROOT"
