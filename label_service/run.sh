#!/usr/bin/env bash
# 启动数据标注平台用的 AI 服务（推理 + 训练），跟命令行共用同一份代码。
#
# 用法（在 imu_train 仓库根目录）：
#   bash label_service/run.sh
#
# 可选环境变量（默认值都在 config.py 里，跟 run_review_bins_all_days.sh 那条常用命令一致）：
#   LABEL_MODEL="results/.../rf/*.pkl"   换模型时才需要传
#   DEVICE_HZ=50  RESAMPLE_METHOD=training_match
#   TARGET_LABELS="活动,睡觉,抓挠,未佩戴,甩身体"
#   NAS_ROOT=/home/toky/ai_data   LABEL_SERVICE_PORT=8383
#
# 起来之后：curl http://localhost:8383/health   接口文档 http://localhost:8383/docs
#
# 注意：现在推荐用容器跑（bash label_service/up.sh），这个脚本只在本地调试、
# 想看实时输出时才用。两者不能同时跑——会抢同一个端口，日志文件也会打架。
set -euo pipefail
cd "$(dirname "$0")/.."

# 容器已经在跑就别再前台起一个：端口会冲突，而且容器是以 root 建的日志文件，
# 宿主机用户往里写会 Permission denied（踩过）。改代码只要重启容器就生效：
#   bash label_service/up.sh -d        （加 -g 用 GPU 版）
if command -v docker >/dev/null 2>&1 &&
   docker ps --format '{{.Image}}' 2>/dev/null | grep -q '^imu-train-label-service'; then
    echo "⚠ label_service 的容器已经在跑了，不用再前台启动。"
    echo ""
    echo "  代码改了要生效：  bash label_service/up.sh -d      （GPU 版加 -g）"
    echo "  看日志：          bash label_service/up.sh logs -f"
    echo "  真要停掉容器再前台跑：bash label_service/up.sh down"
    exit 1
fi

exec python -m uvicorn label_service.app:app --host 0.0.0.0 --port "${LABEL_SERVICE_PORT:-8383}"
