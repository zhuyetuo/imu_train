#!/usr/bin/env bash
# 起 vision_service（SAM 辅助标注）。装不上 SAM 也能起来：/status 会如实说不可用，
# 平台那边把按钮置灰，其它功能不受影响。
set -euo pipefail
cd "$(dirname "$0")/.."
export VISION_SERVICE_PORT="${VISION_SERVICE_PORT:-8385}"
export MATERIAL_ROOT="${MATERIAL_ROOT:-/home/toky/alg_material}"
echo "vision_service 监听 :$VISION_SERVICE_PORT    素材库 $MATERIAL_ROOT"
exec python -m vision_service.app
