#!/usr/bin/env bash
# 批量推理 TF 存储卡版本（无蓝牙）转换出来的 CSV，只做推理+存结果，不裁视频。
#
# 跟 run_review_bins_all_days.sh 的区别：
#   1. TF 数据是按 YYMMDDHH 命名的扁平文件（一小时一个CSV），不是 run_review_bins_all_days.sh
#      要求的"按日期分子目录、CSV和视频放一起"的结构，所以这里直接对整个目录跑一次，不按天循环。
#   2. 小米摄像头监控录像跟这批CSV没有放在一起、也还没建立时间对应关系，所以不裁视频、
#      不生成 Label Studio 任务——检测到抓挠的时间点，你自己拿 _infer.json 里的 start_ts/end_ts
#      去小米监控里对时间点找。等以后CSV和视频的对应关系理清楚了，可以在这个脚本基础上
#      加一步类似 extract_clips.py 的视频裁剪（需要先解决"怎么根据csv文件名/时间戳找到
#      对应的小米监控文件"这个问题，跟 extract_clips.py 现在"视频和CSV在同一个目录"的
#      假设不一样，得单独写匹配逻辑）。加上以后会有裁剪好的视频，输出目录从一开始就把
#      "推理/CSV相关的东西"和"以后的视频"分开放（$RESULT_ROOT/csv/ vs 以后的
#      $RESULT_ROOT/clips/），重跑推理时 rm -rf $RESULT_ROOT/csv 不会连带删掉已经裁好的视频。
#
# 环境变量:
#   CSV_DIR       TF CSV 所在目录（必填），例: data/raw_tf_csv
#   MODEL         ML 模型路径（必填）
#   RESULT_ROOT   推理结果输出目录（默认 infer_result_tf），实际内容在 $RESULT_ROOT/csv/ 下
#   WORKERS       并行进程数（默认 8）
#   DEVICE_HZ     设备采样率（默认 50，TF存储卡版本实测时间戳步长20ms=50Hz；如果你这批数据
#                 不是50Hz，一定要传对，传错了下采样比例会算错）
#   MODEL_HZ      模型采样率（默认 0=从模型元数据读取，一般是16）
#   RESAMPLE_METHOD  device_hz != model_hz 时用哪种降采样算法（默认 poly=scipy
#                 resample_poly）。训练数据是用 witmotion_imu 里的滑动平均低通+线性插值
#                 生成的，跟 poly 不是同一套算法，实测有约6~8%输出差异（见
#                 src/eval/compare_resample_methods.py）。不确定哪个更准时，两种都跑一遍
#                 对比"抓挠"检出情况，改成 RESAMPLE_METHOD=training_match 复刻训练时的算法
#   CONFIDENCE_THRESHOLD  置信度过滤（默认 0.65，减少人工核实量）
#   MERGE_GAP     合并相邻抓挠片段的最大间隔秒数（默认 1s）
#   MIN_WINDOWS   片段最少窗口数（默认 1=不过滤）
#   KEEP_ISOLATED 是否保留孤立单窗口片段（默认 1=保留）
#
# 用法:
#   # 重跑前先清理（只删 csv/ 这层，不会动以后加的 clips/）：
#   rm -rf infer_result_tf/csv
#
#   CSV_DIR=data/raw_tf_csv \
#     MODEL=results/processed_xxx/16hz_remap_custom_3class/ml_rf.pkl \
#     WORKERS=16 RESULT_ROOT=infer_result_tf \
#     ./run_infer_tf.sh

set -e

CSV_DIR="${CSV_DIR:?请设置 CSV_DIR 环境变量，例: CSV_DIR=data/raw_tf_csv}"
MODEL="${MODEL:?请设置 MODEL 环境变量，例: MODEL=results/xxx/ml_rf.pkl}"
RESULT_ROOT="${RESULT_ROOT:-infer_result_tf}"
WORKERS="${WORKERS:-8}"
DEVICE_HZ="${DEVICE_HZ:-50}"
MODEL_HZ="${MODEL_HZ:-0}"
RESAMPLE_METHOD="${RESAMPLE_METHOD:-poly}"  # poly（默认）或 training_match，
                                             # 见 src/eval/compare_resample_methods.py
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.65}"
MERGE_GAP="${MERGE_GAP:-1}"
MIN_WINDOWS="${MIN_WINDOWS:-1}"
KEEP_ISOLATED="${KEEP_ISOLATED:-1}"

# 推理/CSV相关的东西都收在这层，跟以后的 $RESULT_ROOT/clips/（视频）分开，
# 方便重跑推理时只清这一层：rm -rf $RESULT_ROOT/csv
CSV_OUT_DIR="$RESULT_ROOT/csv"

echo "=============================================="
echo "  TF 数据批量推理（仅推理，不裁视频）"
echo "  CSV 目录: $CSV_DIR"
echo "  模型: $MODEL"
echo "  设备采样率: ${DEVICE_HZ}Hz"
echo "  结果目录: $CSV_OUT_DIR"
echo "=============================================="

infer_json_dir="$CSV_OUT_DIR/_infer"
mkdir -p "$infer_json_dir"

hz_args="--device_hz $DEVICE_HZ"
[[ "$MODEL_HZ" -gt 0 ]] && hz_args="$hz_args --model_hz $MODEL_HZ"

python src/infer_csv_scratch.py \
    --csv_dir "$CSV_DIR" \
    --pattern "*.csv" \
    --model "$MODEL" \
    --workers "$WORKERS" \
    --output_dir "$infer_json_dir" \
    --confidence_threshold "$CONFIDENCE_THRESHOLD" \
    --quiet \
    --scratch_only \
    --merge_gap "$MERGE_GAP" \
    --min_windows "$MIN_WINDOWS" \
    --resample_method "$RESAMPLE_METHOD" \
    $( [[ "$KEEP_ISOLATED" == "0" ]] && echo "--no_keep_isolated" ) \
    $hz_args \
    2>&1 | tee "$CSV_OUT_DIR/infer.log"

echo ""
echo "=============================================="
echo "  汇总"
echo "=============================================="
python - "$infer_json_dir" <<'PYEOF'
import glob, json, sys

infer_dir = sys.argv[1]
total_scratch = 0
total_files = 0
for f in sorted(glob.glob(f"{infer_dir}/*_infer.json")):
    d = json.load(open(f))
    segs = d.get("scratch_segments", [])
    total_files += 1
    if not segs:
        continue
    total_scratch += len(segs)
    print(f"  {d.get('csv_basename', f)}: {len(segs)} 段抓挠")
    for s in segs:
        print(f"      {s['start_ts']} → {s['end_ts']}  (conf_max={s['conf_max']:.2f}, "
              f"conf_mean={s['conf_mean']:.2f}, n_windows={s['n_windows']})")

print(f"\n  合计: {total_files} 个文件，{total_scratch} 段抓挠")
PYEOF

echo ""
echo "完成！逐文件 JSON 结果在: $infer_json_dir"
echo "去小米监控里核实时用上面打印的 start_ts/end_ts（或直接看JSON里的 scratch_segments）对时间点。"
