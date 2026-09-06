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
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m uvicorn label_service.app:app --host 0.0.0.0 --port "${LABEL_SERVICE_PORT:-8383}"
