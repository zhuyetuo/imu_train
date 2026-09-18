#!/usr/bin/env bash
# 起 vision_service（SAM 辅助标注）。装不上 SAM 也能起来：/status 会如实说不可用，
# 平台那边把按钮置灰，其它功能不受影响。
#
# 用法：
#   ./vision_service/run.sh          前台跑（Ctrl-C 停）——原来的行为，没变
#   ./vision_service/run.sh -d       后台常驻，日志写到 vision_service/.run.log
#   ./vision_service/run.sh status   看在不在跑
#   ./vision_service/run.sh down     停
#
#   --no-install     不自动装缺的依赖（离线机器，或者你自己管环境）
#
# 为什么要有 -d：前台跑的话，关掉终端 / SSH 断了，服务就跟着没了，而平台那边
# 只会表现成「SAM 按钮灰了」，看不出是服务掉了还是模型没装上。label_service
# 是 docker 起的、天然后台常驻，这个是宿主机上的 python（SAM2 和权重都在
# conda 环境里，塞进容器要连 CUDA 一起搬），所以得自己管进程。
set -euo pipefail
cd "$(dirname "$0")/.."

# 本机私有配置（API key 之类）放 vision_service/.env，一行一个 KEY=VALUE，不进 git。
# 「画面找片段」要 ANTHROPIC_API_KEY，写在这里就不用每次起服务都 export 一遍。
if [ -f vision_service/.env ]; then
    set -a
    # shellcheck disable=SC1091
    . vision_service/.env
    set +a
fi

export VISION_SERVICE_PORT="${VISION_SERVICE_PORT:-8385}"
# 权重从 HuggingFace 下。国内直连 huggingface.co 经常卡住不动（不报错、也没速度），
# 默认走 hf-mirror.com 镜像；能直连的机器在 vision_service/.env 里写 HF_ENDPOINT=https://huggingface.co
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MATERIAL_ROOT="${MATERIAL_ROOT:-/home/toky/alg_material}"

PID_FILE="vision_service/.run.pid"
LOG_FILE="vision_service/.run.log"
# 起服务的命令。抽成变量是为了能被测试替换掉——不然要验证"后台起没起来、
# down 停不停得掉"就只能真去装 SAM 和权重
VISION_RUN_CMD="${VISION_RUN_CMD:-python -m vision_service.app}"
PY_BIN="${PY_BIN:-python}"
# pip 源：几个国内镜像 + 官方测速选最快的（结果缓存一天）。手动指定就不测：
#   PIP_INDEX_URL=https://pypi.org/simple ./up.sh deploy
source "$(dirname "${BASH_SOURCE[0]}")/pick_pip_mirror.sh"

# ── 起之前先把缺的依赖装上 ────────────────────────────────────────────
#
# 不装的话表现是"服务起来了、功能是灰的"——/status 里写着"没装 ultralytics"，
# 但人得先想到去看 /status。让部署命令自己管这件事，比写在文档里让人记着强。
#
# **torch 和 sam2 一律不自动装**：它们的版本取决于机器上的 CUDA，装错会把
# 现成环境搞坏（比如把 GPU 版 torch 覆盖成 CPU 版，SAM 就悄悄退回 CPU 跑，
# 慢十几倍还不报错）。requirements.txt 里本来就特意没钉它们。缺了就说清楚
# 该怎么装，让人自己来。
#
# VISION_NO_INSTALL=1 或 --no-install 跳过这一步（离线机器、或者你自己管环境）。
NO_INSTALL="${VISION_NO_INSTALL:-0}"

ensure_deps() {
    [ "$NO_INSTALL" = "1" ] && return 0
    local missing=()
    # 左边是 import 名，右边是给人看的说明。只列 requirements.txt 里有的——
    # torch/sam2 不在这儿，它们走下面那段只提示不安装
    for pair in "fastapi:fastapi" "uvicorn:uvicorn" "numpy:numpy" "cv2:opencv-python-headless" "ultralytics:ultralytics" "anthropic:anthropic" "httpx:httpx" "transformers:transformers" "PIL:pillow" "sentencepiece:sentencepiece" "rtmlib:rtmlib" "onnxruntime:onnxruntime"; do
        local mod="${pair%%:*}"
        "$PY_BIN" -c "import ${mod}" >/dev/null 2>&1 || missing+=("${pair##*:}")
    done

    if [ ${#missing[@]} -gt 0 ]; then
        echo "缺依赖：${missing[*]}，装一下（pip install -r vision_service/requirements.txt）..."
        if "$PY_BIN" -m pip install -r vision_service/requirements.txt; then
            echo "装好了"
        else
            echo "⚠ 装依赖失败。服务还是会起来，但相关功能是灰的——"
            echo "  手动装：$PY_BIN -m pip install -r vision_service/requirements.txt"
        fi
    fi

    # 姿态关键点走 onnxruntime：有显卡就换成 GPU 版（rtmpose-m 一帧 CPU 30ms、GPU 几毫秒）。
    # requirements 里写的是 CPU 版兜底；这里发现有 nvidia-smi 且当前 onnxruntime 没有 CUDA provider，
    # 就卸掉 CPU 版装 GPU 版（两个包不能共存）。装不上就留 CPU 版，功能不受影响
    if command -v nvidia-smi >/dev/null 2>&1 && ! "$PY_BIN" -c "import onnxruntime as o, sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" >/dev/null 2>&1; then
        echo "有显卡，把 onnxruntime 换成 GPU 版（姿态关键点快十倍）。包连 CUDA 运行库有几百 MB，看下面 pip 的进度..."
        "$PY_BIN" -m pip uninstall -y -q onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
        if "$PY_BIN" -m pip install --progress-bar on onnxruntime-gpu; then
            if "$PY_BIN" -c "import onnxruntime as o, sys; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" >/dev/null 2>&1; then
                echo "  onnxruntime-gpu 装好了，CUDA provider 可用"
            else
                echo "  ⚠ onnxruntime-gpu 装了但 CUDA provider 不可用（多半是 cuDNN / CUDA 运行库版本不对），姿态会退回 CPU 跑"
            fi
        else
            echo "  ⚠ onnxruntime-gpu 装不上，退回 CPU 版"
            "$PY_BIN" -m pip install onnxruntime || true
        fi
    fi

    # vllm（本地大模型，「模型服务」页一键起停）：几 GB，自带钉死版本的 torch——装它可能把
    # 现有 torch 换成它要的那个版本（一般更新，狗检测 / SigLIP 照常能跑）。有显卡才装；
    # 不想它动环境：VLLM_INSTALL=0 ./up.sh deploy
    if [ "${VLLM_INSTALL:-1}" != "0" ] && command -v nvidia-smi >/dev/null 2>&1 && ! "$PY_BIN" -c "import vllm" >/dev/null 2>&1; then
        echo "缺 vllm（本地大模型），装一下：几 GB，会连带装它要的 torch 版本，看下面 pip 的进度..."
        if "$PY_BIN" -m pip install --progress-bar on vllm; then
            echo "  vllm 装好了。「模型服务」页点「启动」拉起来（第一次要下模型权重）"
        else
            echo "  ⚠ vllm 装不上。本地大模型那一行会显示没装，别的功能不受影响；手动：$PY_BIN -m pip install vllm"
        fi
    fi

    # torch / torchaudio 的 CUDA 版本对不上（装 vllm 换了 torch，torchaudio 还是老的）：transformers 一 import
    # 就炸，SigLIP 整个不可用。我们不用 torchaudio——先试装跟 torch 同源的版本，不行就卸掉
    if ! "$PY_BIN" -c "import torch, torchaudio" >/dev/null 2>&1 && "$PY_BIN" -c "import torch" >/dev/null 2>&1; then
        if "$PY_BIN" -c "import torch, torchaudio" 2>&1 | grep -q "different CUDA versions"; then
            echo "torch 和 torchaudio 的 CUDA 版本对不上（装 vllm 换了 torch），修一下..."
            ta_ver="$("$PY_BIN" -c "import importlib.metadata as m; print(m.version('torchaudio').split('+')[0])" 2>/dev/null)"
            if [ -n "$ta_ver" ] && "$PY_BIN" -m pip install --force-reinstall --no-deps "torchaudio==$ta_ver" >/dev/null 2>&1 \
               && "$PY_BIN" -c "import torch, torchaudio" >/dev/null 2>&1; then
                echo "  重装了 torchaudio==$ta_ver，跟 torch 对上了"
            else
                "$PY_BIN" -m pip uninstall -y -q torchaudio >/dev/null 2>&1 || true
                echo "  卸掉了 torchaudio（这里用不到它）"
            fi
        fi
    fi

    # ffmpeg：系统包，有它解码抽帧快好几倍（还能走 NVDEC）；没有退回 cv2，慢但能用。
    # 是 apt 装的，要 sudo：能免密就直接装，不能就问一次密码（不想装就 --no-install）
    if ! command -v ffmpeg >/dev/null 2>&1; then
        if command -v apt-get >/dev/null 2>&1; then
            echo "缺 ffmpeg（解码快 3~5 倍），用 apt 装一下（可能要输 sudo 密码）..."
            if sudo -n true 2>/dev/null || [ -t 0 ]; then
                sudo apt-get install -y ffmpeg && echo "ffmpeg 装好了" \
                    || echo "⚠ ffmpeg 没装上，解码退回 cv2（慢）。手动：sudo apt-get install -y ffmpeg"
            else
                echo "⚠ 没有终端输不了 sudo 密码，跳过。手动装：sudo apt-get install -y ffmpeg"
            fi
        else
            echo "ℹ 没装 ffmpeg，解码退回 cv2（慢 3~5 倍）。装一下：sudo apt install ffmpeg"
        fi
    fi

    # torch / sam2：只提示，不动手
    if ! "$PY_BIN" -c "import torch" >/dev/null 2>&1; then
        echo "ℹ 没装 torch。SAM 和画面狗检测都要它，但**这里不自动装**——"
        echo "  版本取决于你机器上的 CUDA，装错会把现成环境搞坏（GPU 版被覆盖成 CPU 版，"
        echo "  SAM 会悄悄退回 CPU 跑、慢十几倍还不报错）。按 vision_service/README.md 自己装。"
    elif ! "$PY_BIN" -c "import sam2" >/dev/null 2>&1; then
        echo "ℹ 没装 sam2，SAM 点选会是灰的（其它功能不受影响）："
        echo "  $PY_BIN -m pip install git+https://github.com/facebookresearch/sam2.git"
    fi
}

running_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    # pid 文件可能是上次没停干净留下的，里面那个号码早被别的进程用了。
    # kill -0 只判断"这个 pid 还在不在"，判断不了"是不是我起的那个"——但
    # 配合下面 down 之后就删文件，已经够用了，不值得为此再存一份启动时间
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "$pid"; return 0; }
    rm -f "$PID_FILE"
    return 1
}

# 端口上是谁：pid 文件丢了/不对的时候（比如上次是手动起的、或者 kill -9 之后
# CUDA 进程还没把端口吐出来）靠这个兜底。只认命令行里带 vision_service 的进程，
# 别的东西占着端口不是我们能杀的
port_pid() {
    local pid
    pid="$(ss -ltnp 2>/dev/null | awk -v p=":$VISION_SERVICE_PORT" '$4 ~ p"$" {print $NF}' | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)"
    [ -z "$pid" ] && command -v fuser >/dev/null 2>&1 && pid="$(fuser -n tcp "$VISION_SERVICE_PORT" 2>/dev/null | awk '{print $1}')"
    [ -n "$pid" ] && tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q vision_service && { echo "$pid"; return 0; }
    return 1
}

port_free() { ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$VISION_SERVICE_PORT\$"; }

stop_service() {
    local pid
    pid="$(running_pid)" || pid="$(port_pid)" || { echo "没在跑"; return 0; }
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 20); do
        sleep 0.25
        kill -0 "$pid" 2>/dev/null || break
    done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    rm -f "$PID_FILE"
    # kill -9 之后进程没了，端口不一定立刻空出来（CUDA 进程收尾要一会儿）。
    # 不等的话紧接着的 -d 会撞上 "address already in use"，看着像起不来
    for _ in $(seq 60); do
        port_free && break
        sleep 0.25
    done
    port_free || echo "⚠ 端口 $VISION_SERVICE_PORT 还被占着（pid $(port_pid || echo ?)），起新的可能会失败"
    echo "已停（原 pid $pid）"
}

# --no-install 可以出现在任何位置，先摘出去再看子命令
ARGS=()
for a in "$@"; do
    case "$a" in
        --no-install) NO_INSTALL=1 ;;
        *) ARGS+=("$a") ;;
    esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

case "${1:-}" in
    -d|--daemon)
        if pid="$(running_pid)"; then
            echo "已经在跑了（pid $pid，端口 $VISION_SERVICE_PORT）。要重起先 ./vision_service/run.sh down"
            exit 0
        fi
        # pid 文件没了但端口还被我们自己的旧进程占着（上次是手动起的 / 没停干净）：
        # 直接起会 "address already in use"。是我们的就先收掉，不是我们的就明说
        if ! port_free; then
            if pid="$(port_pid)"; then
                echo "端口 $VISION_SERVICE_PORT 上还有一个旧的 vision_service（pid $pid），先停掉它"
                echo "$pid" > "$PID_FILE"
                stop_service
            else
                echo "端口 $VISION_SERVICE_PORT 被别的进程占着，起不了：ss -ltnp | grep $VISION_SERVICE_PORT 看看是谁"
                exit 1
            fi
        fi
        ensure_deps
        : > "$LOG_FILE"
        nohup $VISION_RUN_CMD >>"$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        # 等它自己起来或者自己死掉，别立刻就说"起好了"——端口被占、依赖缺一个，
        # 都是起到一半才报出来的，那时候人已经去干别的了
        for _ in $(seq 20); do
            sleep 0.25
            running_pid >/dev/null || break
        done
        if pid="$(running_pid)"; then
            echo "vision_service 已后台启动（pid $pid，端口 $VISION_SERVICE_PORT，素材库 $MATERIAL_ROOT）"
            echo "  日志: $LOG_FILE     停: ./vision_service/run.sh down"
        else
            echo "起不来，日志最后几行："
            tail -n 15 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
            exit 1
        fi
        ;;
    down|stop)
        stop_service
        ;;
    status)
        if pid="$(running_pid)"; then
            echo "在跑：pid $pid，端口 $VISION_SERVICE_PORT，素材库 $MATERIAL_ROOT"
            echo "  模型状态: curl -s localhost:$VISION_SERVICE_PORT/api/v1/sam/status"
        else
            echo "没在跑。起：./vision_service/run.sh -d"
            exit 1
        fi
        ;;
    "")
        ensure_deps
        echo "vision_service 监听 :$VISION_SERVICE_PORT    素材库 $MATERIAL_ROOT"
        echo "（前台跑，关掉终端就停了；要常驻用 ./vision_service/run.sh -d）"
        exec $VISION_RUN_CMD
        ;;
    *)
        echo "不认识的参数: $1"
        sed -n '4,10p' "$0"
        exit 1
        ;;
esac
