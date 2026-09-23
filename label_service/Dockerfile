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

# 基础镜像自带的 pip 太老，没有 raw 进度模式（--progress-bar 只认 on/off）。先升级
RUN pip install --no-cache-dir -U "pip>=24.1"
# PIP_PROGRESS_BAR=raw：构建日志不是终端，pip 默认一个字都不打，几个 G 的 torch
# 下几分钟看起来像卡死了。raw 模式每隔一会儿打一行"下到多少 MB / 共多少"
ENV PIP_PROGRESS_BAR=raw

# torch 先单独装 CPU 版：ultralytics 依赖它，但容器里没有 GPU，装默认版会连带
# 拉 nvidia_cu* 那一堆 CUDA 轮子（好几个 G，全是用不上的）。CPU 版约 200MB。
# torch 本体从 torch 源拿（+cpu / +cu128 那种带后缀的版本号比 PyPI 上的高，pip 会
# 优先选它），它的依赖（filelock、sympy、nvidia-* …）从 PIP_INDEX_URL 拿。
# 只给 --index-url torch源 的话，依赖会跟着 torch 源的链接走到那个镜像自己的 pypi
# 上——南大那个 pypi 一个 2KB 的包都能超时（2026-09-24）。
# **这行别动**：改一个字符这一层就作废，几个 G 的 CUDA 包又要重下。要加东西加在最后那层 apt 里
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" --extra-index-url "${TORCH_INDEX_URL}" torch torchvision

# 依赖单独一层：仓库代码变了不会让 pip 重装一遍
COPY label_service/requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# gcc：端侧那条线（algo_tinyml）把板上那份 C 现场编成 .so 跑——端侧服务和
# 「导出到端侧」都在这个镜像里，没有 gcc 两样都起不来。
# **放在 pip 那两层之后、单独一层**：docker 的层缓存是按顺序的，加进上面那条
# apt 里的话，后面 torch（几百 MB 的 CUDA 包）和全部依赖都得重装一遍
# （2026-09-24 就是这么多等了十几分钟）。以后再加系统工具也加在这里
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LABEL_SERVICE_PORT=8383

EXPOSE 8383

CMD ["python", "-m", "uvicorn", "label_service.app:app", "--host", "0.0.0.0", "--port", "8383"]
