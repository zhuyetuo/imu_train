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
export MATERIAL_ROOT="${MATERIAL_ROOT:-/home/toky/alg_material}"

PID_FILE="vision_service/.run.pid"
LOG_FILE="vision_service/.run.log"
# 起服务的命令。抽成变量是为了能被测试替换掉——不然要验证"后台起没起来、
# down 停不停得掉"就只能真去装 SAM 和权重
VISION_RUN_CMD="${VISION_RUN_CMD:-python -m vision_service.app}"
PY_BIN="${PY_BIN:-python}"

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
    for pair in "fastapi:fastapi" "uvicorn:uvicorn" "numpy:numpy" "cv2:opencv-python-headless" "ultralytics:ultralytics" "anthropic:anthropic"; do
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
        if pid="$(running_pid)"; then
            kill "$pid" 2>/dev/null || true
            for _ in $(seq 20); do
                sleep 0.25
                running_pid >/dev/null || break
            done
            running_pid >/dev/null && kill -9 "$pid" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo "已停（原 pid $pid）"
        else
            echo "没在跑"
        fi
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
