# label_service 的运行环境。构建上下文是仓库根目录（见 label_service/docker-compose.yml）。
#
# 只装依赖，不 COPY 代码——compose 把整个仓库挂进 /app：
#   - 改 label_service/*.py 重启容器就生效，不用重建镜像
#   - 训练产出（results/、data/、tmp/）直接落在宿主机仓库里，跟命令行跑出来的一样
#   - 模型文件（results/.../*.pkl）本来就在仓库里，不用另外拷进镜像
#
# 装的是 label_service/requirements-docker.txt（精简版，见那个文件里的说明），
# 不是仓库根目录那份完整的——服务只做推理/训练/牙齿检测，用不到 shap/pymc/
# gradio 那些研究工具。
#
# 构建慢的话调这两个源（默认已经指到国内镜像）：
#   docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
FROM python:3.11-slim

# 国内拉 PyPI 默认源经常只有几百 KB/s，这里默认走清华镜像；要用官方源就
# --build-arg PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# torch 的 CPU 版只有官方这个源有（PyPI 上的 linux 轮子是带 CUDA 的）。
# 国内可换 https://mirrors.aliyun.com/pytorch-wheels/cpu/
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# apt 也换国内源，不然装 ffmpeg 那一步就要几分钟
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libglib2.0-0 tzdata \
    && rm -rf /var/lib/apt/lists/*
# ffmpeg：src/extract_clips.py 在 import 时就会探测它
# libglib2.0-0：opencv-python-headless 的运行时依赖（headless 版不需要 libgl1）
# tzdata：容器时区跟宿主机一致（compose 里设 TZ）

WORKDIR /app

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# torch 先单独装 CPU 版：ultralytics 依赖它，但容器里没有 GPU，装默认版会连带
# 拉 nvidia_cu* 那一堆 CUDA 轮子（好几个 G，全是用不上的）。CPU 版约 200MB。
RUN pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch torchvision

# 依赖单独一层：仓库代码变了不会让 pip 重装一遍
COPY label_service/requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LABEL_SERVICE_PORT=8383

EXPOSE 8383

CMD ["python", "-m", "uvicorn", "label_service.app:app", "--host", "0.0.0.0", "--port", "8383"]
