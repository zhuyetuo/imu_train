#!/usr/bin/env bash
# 训一个「只用加速计」的模型，并跟同一份数据的基线对比。
#
#   bash acc_only/train_acc_only.sh \
#     --processed_dir data/processed_2026_8_11-2026_8_27_raw_missing_drop_window
#
# 做的事（两步，都不碰仓库原有的任何文件）：
#   1. 把 --processed_dir 里的窗口复制一份，陀螺仪三轴置零 → *_acc_only/
#   2. 在这份数据上训练 → results_acc_only/
#
# 为什么是"置零"不是"只留 3 通道"：见 acc_only/make_acc_only_npz.py 顶部。
# 一句话——置零之后 193 维里正好 80 维变成常数（树模型永远不会选常数特征），
# 剩下 113 维全部只依赖加速计；而真只留 3 通道会连 acc 模长、jerk、
# pitch/roll 一起丢掉（那些只靠加速计就能算），比出来的就不是"有没有陀螺仪"了。
#
# 参数：
#   --processed_dir DIR   必填，基线那份预处理输出（里面有 <hz>hz/train.npz）
#   --hz HZ               默认 16
#   --model TYPE          默认 rf，跟 src/ml/train.py 的 --model 一样
#   --remap FILE          默认 configs/remap_custom_3class.yaml
#   --config FILE         传给 src/ml/train.py 的 --config（默认不传，用它自己的默认）
#   --results_dir DIR     默认 results_acc_only
#   --feat_workers N      特征提取并行进程数，默认 -1（全部核心）
#   --keep_npz            不删中间那份置零数据（默认保留，这个开关留给以后）
#
# 训完会打印「基线 vs 只用加速计」的对比表。

set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

HZ=16
MODEL=rf
REMAP="configs/remap_custom_3class.yaml"
CONFIG=""
RESULTS_DIR="results_acc_only"
FEAT_WORKERS=-1
PROCESSED_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --processed_dir) PROCESSED_DIR="$2"; shift 2 ;;
    --hz)            HZ="$2";            shift 2 ;;
    --model)         MODEL="$2";         shift 2 ;;
    --remap)         REMAP="$2";         shift 2 ;;
    --config)        CONFIG="$2";        shift 2 ;;
    --results_dir)   RESULTS_DIR="$2";   shift 2 ;;
    --feat_workers)  FEAT_WORKERS="$2";  shift 2 ;;
    --keep_npz)      shift 1 ;;
    -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数: $1（-h 看帮助）"; exit 1 ;;
  esac
done

if [[ -z "$PROCESSED_DIR" ]]; then
  echo "✗ 要给 --processed_dir，也就是基线那份预处理目录。"
  echo "  没跑过预处理的话，先照常跑一次 train_custom.sh（这个脚本不重复那一步——"
  echo "  重跑一遍预处理的话，窗口划分的随机性会混进对比里，"
  echo "  差值就不能全归因于陀螺仪了）。"
  echo ""
  echo "  现有的预处理目录："
  ls -d data/processed_* 2>/dev/null | sed 's/^/    /' || echo "    （一个都没有）"
  exit 1
fi

PROCESSED_DIR="${PROCESSED_DIR%/}"
ACC_DIR="${PROCESSED_DIR}_acc_only"
DATASET_TAG=$(basename "$ACC_DIR")
REMAP_TAG=""
[[ -n "$REMAP" ]] && REMAP_TAG="_$(basename "${REMAP%.*}")"
OUT_DIR="${RESULTS_DIR}/${DATASET_TAG}/${HZ}hz${REMAP_TAG}/${MODEL}"

echo "▶ 1/2 陀螺仪置零：$PROCESSED_DIR → $ACC_DIR"
python acc_only/make_acc_only_npz.py --src "$PROCESSED_DIR" --dst "$ACC_DIR" --hz "$HZ"

echo ""
echo "▶ 2/2 训练（$MODEL，$HZ Hz）"
python src/ml/train.py \
  --hz "$HZ" --model "$MODEL" \
  --processed_dir "$ACC_DIR" \
  --remap "$REMAP" \
  --results_dir "$RESULTS_DIR" \
  --feat_workers "$FEAT_WORKERS" \
  ${CONFIG:+--config "$CONFIG"}

echo ""
echo "▶ 自检：模型有没有偷偷用到陀螺仪"
# 这一条是整条路线唯一的风险点：如果模型其实用了陀螺仪，它在只有加速计的
# 真设备上会失效，而在平台上（CSV 里有陀螺仪）看着完全正常
python acc_only/selfcheck.py --model "${OUT_DIR}/ml_${MODEL}.pkl" || true

echo ""
echo "▶ 对比"
BASE_DIR="results/$(basename "$PROCESSED_DIR")/${HZ}hz${REMAP_TAG}/${MODEL}"
python - "$OUT_DIR/ml_${MODEL}.json" "$BASE_DIR/ml_${MODEL}.json" <<'PY'
import json, os, sys

def load(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

acc_only, base = load(sys.argv[1]), load(sys.argv[2])
if not acc_only:
    sys.exit("读不到刚训出来的结果，看看上面训练那步是不是失败了")
if not base:
    # 基线不在默认位置很正常（--results_dir 可以是别的）。**说出来**，
    # 不然只印一列数，人会以为"没有变化"
    print(f"（找不到基线 {sys.argv[2]}，只印这次的结果。手动对比的话，"
          f"基线是同一份数据不置零训出来的那个 ml_*.json）")
    base = {}

rows = [("总体准确率", "accuracy"), ("macro F1", "macro_f1")]
w = 14
print(f"\n{'指标':<12}{'基线(acc+gyro)':>{w}}{'只用加速计':>{w}}{'差值':>{w}}")
for name, key in rows:
    a, b = acc_only.get(key), base.get(key)
    if a is None:
        continue
    if b is None:
        print(f"{name:<12}{'—':>{w}}{a:>{w}.4f}{'—':>{w}}")
    else:
        print(f"{name:<12}{b:>{w}.4f}{a:>{w}.4f}{a - b:>+{w}.4f}")

pa, pb = acc_only.get("per_class") or {}, base.get("per_class") or {}
if pa:
    print(f"\n{'类别':<10}{'基线 F1':>{w}}{'只用加速计 F1':>{w}}{'差值':>{w}}")
    for k in pa:
        a = pa[k].get("f1-score")
        b = (pb.get(k) or {}).get("f1-score")
        if b is None:
            print(f"{k:<10}{'—':>{w}}{a:>{w}.4f}{'—':>{w}}")
        else:
            print(f"{k:<10}{b:>{w}.4f}{a:>{w}.4f}{a - b:>+{w}.4f}")
PY

echo ""
echo "模型在：$OUT_DIR/"
echo ""
echo "要在平台上验的话，把它挂到端侧服务（algo_tinyml）："
echo "  1. 复制过去：  mkdir -p ~/algo_tinyml/models/acc_only_rf && \\"
echo "                 cp $OUT_DIR/ml_${MODEL}.pkl $OUT_DIR/ml_${MODEL}.json \\"
echo "                    ~/algo_tinyml/models/acc_only_rf/"
echo "  2. 重启服务：  cd ~/algo_tinyml && ./serve.sh -d"
echo "  3. 平台上「版本」下拉里会多出一项  端侧 · acc_only_rf · 稳定版 v2"
echo "     （后处理跟线上「稳定版 v2」是同一份代码，所以差的只有模型本身）"
