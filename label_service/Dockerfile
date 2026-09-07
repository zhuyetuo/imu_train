# label_service 的运行环境。构建上下文是仓库根目录（见 label_service/docker-compose.yml）。
#
# 只装依赖，不 COPY 代码——compose 把整个仓库挂进 /app：
#   - 改 label_service/*.py 重启容器就生效，不用重建镜像
#   - 训练产出（results/、data/、tmp/）直接落在宿主机仓库里，跟命令行跑出来的一样
#   - 模型文件（results/.../*.pkl）本来就在仓库里，不用另外拷进镜像
#
# 依赖装的是仓库根目录那份完整 requirements.txt：训练走 train_custom.sh，需要
# torch/xgboost/lightgbm/catboost 一整套；牙齿检测要 ultralytics + opencv。
# 镜像因此比较大（几 GB），但只在依赖变了的时候才重建。
FROM python:3.11-slim

# ffmpeg：src/extract_clips.py 在 import 时就会探测它
# libgl1 / libglib2.0-0：opencv 的运行时依赖
# tzdata：容器时区跟宿主机一致（compose 里设 TZ）
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖单独一层：仓库代码变了不会让 pip 重装一遍
COPY requirements.txt /tmp/req/base.txt
COPY label_service/requirements.txt /tmp/req/service.txt
RUN pip install --no-cache-dir -r /tmp/req/base.txt -r /tmp/req/service.txt

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LABEL_SERVICE_PORT=8383

EXPOSE 8383

CMD ["python", "-m", "uvicorn", "label_service.app:app", "--host", "0.0.0.0", "--port", "8383"]
