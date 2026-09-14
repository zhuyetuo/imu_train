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
# 为什么要有 -d：前台跑的话，关掉终端 / SSH 断了，服务就跟着没了，而平台那边
# 只会表现成「SAM 按钮灰了」，看不出是服务掉了还是模型没装上。label_service
# 是 docker 起的、天然后台常驻，这个是宿主机上的 python（SAM2 和权重都在
# conda 环境里，塞进容器要连 CUDA 一起搬），所以得自己管进程。
set -euo pipefail
cd "$(dirname "$0")/.."

export VISION_SERVICE_PORT="${VISION_SERVICE_PORT:-8385}"
export MATERIAL_ROOT="${MATERIAL_ROOT:-/home/toky/alg_material}"

PID_FILE="vision_service/.run.pid"
LOG_FILE="vision_service/.run.log"
# 起服务的命令。抽成变量是为了能被测试替换掉——不然要验证"后台起没起来、
# down 停不停得掉"就只能真去装 SAM 和权重
VISION_RUN_CMD="${VISION_RUN_CMD:-python -m vision_service.app}"

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

case "${1:-}" in
    -d|--daemon)
        if pid="$(running_pid)"; then
            echo "已经在跑了（pid $pid，端口 $VISION_SERVICE_PORT）。要重起先 ./vision_service/run.sh down"
            exit 0
        fi
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
