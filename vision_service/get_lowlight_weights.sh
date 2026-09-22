#!/usr/bin/env bash
# 夜视增强（Retinexformer）的仓库和权重。**可选**——不配也能用：
# 「夜视」里的「只拉伸」和「多帧堆栈」照常出，而且那两张不编造像素。
#
# 为什么要有这个脚本：上一次是让人「把这几行写进 .env」，结果那几行被当成命令
# 粘进了 shell——变成临时的 shell 变量，一关终端就没了，而服务照样起得来、
# 只是永远读不到配置。这种错最难发现，所以改成脚本来写。
set -euo pipefail
cd "$(dirname "$0")/.."

REPO_DIR="${1:-$HOME/Retinexformer}"
WEIGHTS="${2:-}"
ENV_FILE="vision_service/.env"

if [ ! -d "$REPO_DIR" ]; then
  echo "clone 官方仓库到 $REPO_DIR（要它的网络结构，自己重写一份跟权重对不上会输出垃圾）"
  git clone --depth 1 https://github.com/caiyuanhao1998/Retinexformer "$REPO_DIR"
fi

if [ -z "$WEIGHTS" ]; then
  # 自己找：这批是带真实噪声的监控视频，SMID / SDSD_indoor 最对口
  for n in SMID.pth SDSD_indoor.pth SID.pth; do
    p="models/vision/lowlight/$n"
    [ -f "$p" ] && WEIGHTS="$(pwd)/$p" && break
  done
fi

if [ -z "$WEIGHTS" ]; then
  cat <<'EOF'
还缺权重。去 Retinexformer 的 README 里那个 Google Drive / 百度网盘下一个放到
  models/vision/lowlight/
这批是**带真实噪声的监控视频**，用 SMID.pth 或 SDSD_indoor.pth（拿低光视频训的）。
LOL_v1/v2 是照片、几乎没噪声，拿来只会输出一张好看但不对的图。
放好之后再跑一次这个脚本。
EOF
  exit 1
fi

touch "$ENV_FILE"
# 已有的同名行先删掉，避免越写越多、而生效的是最后一条
sed -i '/^LOWLIGHT_REPO=/d;/^LOWLIGHT_WEIGHTS=/d' "$ENV_FILE"
{
  echo "LOWLIGHT_REPO=$REPO_DIR"
  echo "LOWLIGHT_WEIGHTS=$WEIGHTS"
} >> "$ENV_FILE"

echo "写好了 $ENV_FILE："
grep '^LOWLIGHT' "$ENV_FILE"
echo
echo "重启生效： ./vision_service/run.sh down && ./vision_service/run.sh -d"
echo "然后在平台「模型服务」页应该能看到「夜视增强（Retinexformer）」这一行。"
