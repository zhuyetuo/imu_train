#!/usr/bin/env bash
# 把姿态关键点模型（RTMPose-m AP-10K，ONNX 一个文件）下到 models/vision/pose/，写进 .env。
#
#   ./vision_service/get_pose_weights.sh          下（已经有了就跳过）
#   ./vision_service/get_pose_weights.sh --force  重下
#
# ./up.sh deploy 会自动调它。下不到不算部署失败：姿态这一路自动关，以图搜图只用画面。
#
# 几个源挨个试。OpenMMLab 的下载站在国内一般能通；都不通就按最后打印的办法自己导一个 ONNX 拷过来。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEST_DIR="models/vision/pose"
DEST="$DEST_DIR/rtmpose_ap10k.onnx"
ENV_FILE="vision_service/.env"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

write_env() {
    local abs
    abs="$(cd "$(dirname "$DEST")" && pwd)/$(basename "$DEST")"
    touch "$ENV_FILE"
    if grep -q '^POSE_ONNX=' "$ENV_FILE"; then
        sed -i "s|^POSE_ONNX=.*|POSE_ONNX=$abs|" "$ENV_FILE"
    else
        echo "POSE_ONNX=$abs" >> "$ENV_FILE"
    fi
    echo "已写入 $ENV_FILE：POSE_ONNX=$abs"
}

if [ "$FORCE" = "0" ] && [ -s "$DEST" ]; then
    echo "姿态模型已在本地：$DEST（要重下加 --force）"
    grep -q "^POSE_ONNX=" "$ENV_FILE" 2>/dev/null || write_env
    exit 0
fi
[ "${DRY_RUN:-0}" = "1" ] && { echo "（DRY_RUN）会把 RTMPose AP-10K 下到 $DEST"; exit 0; }

mkdir -p "$DEST_DIR"
TMP="$DEST_DIR/.download.tmp"
# 候选地址：OpenMMLab 的 rtmpose 发布件（SDK 打包的 ONNX）。哪个通用哪个
URLS=(
  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip"
  "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.onnx"
)
for url in "${URLS[@]}"; do
    echo "▶ 试 $url"
    rm -f "$TMP"
    if curl -L --fail --connect-timeout 15 --max-time 900 --progress-bar -o "$TMP" "$url"; then
        case "$url" in
          *.zip)
            rm -rf "$DEST_DIR/.unzip" && mkdir -p "$DEST_DIR/.unzip"
            if python - "$TMP" "$DEST_DIR/.unzip" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PY
            then
                f="$(find "$DEST_DIR/.unzip" -name '*.onnx' | head -1)"
                if [ -n "$f" ]; then mv "$f" "$DEST"; fi
            fi
            rm -rf "$DEST_DIR/.unzip" "$TMP"
            ;;
          *) mv "$TMP" "$DEST" ;;
        esac
        if [ -s "$DEST" ]; then
            echo "✓ 姿态模型下好了：$DEST"
            write_env
            exit 0
        fi
    fi
    echo "  ✗ 这个源不通或包里没有 .onnx"
done

echo
echo "✗ 姿态模型没下到。姿态那一路会自动关（以图搜图只用画面），别的都不受影响。"
echo "  要用的话在能上网的电脑上拿到 RTMPose AP-10K 的 ONNX（mmpose 仓库 projects/rtmpose 里的 ap10k 模型，"
echo "  用 mmdeploy 导成 onnx，输入 256x256），拷到 $(pwd)/$DEST，再跑一次这个脚本。"
exit 0
