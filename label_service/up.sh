#!/bin/bash
# 起 label_service（后台常驻），完成后打印访问地址。
#
# 用法：
#   bash label_service/up.sh        重建并启动（CPU）
#   bash label_service/up.sh -g     用 GPU 起（需要宿主机装了 nvidia-container-toolkit）
#   bash label_service/up.sh -p     先 git pull 再重建
#   bash label_service/up.sh -d     只重启，不重建镜像（改了 .py 用这个，最快）
#   bash label_service/up.sh down   停服务
#
# 几个开关可以叠：bash label_service/up.sh -p -g
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

COMPOSE=(docker compose -f docker-compose.yml)
GPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--pull)
      echo "▶ 拉取最新代码..."
      git -C "$(git rev-parse --show-toplevel)" pull --ff-only
      echo ""
      shift
      ;;
    -g|--gpu)
      GPU=1
      COMPOSE+=(-f docker-compose.gpu.yml)
      shift
      ;;
    -d|--restart)
      "${COMPOSE[@]}" restart
      echo "已重启（代码走挂载，不需要重建镜像）"
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if [[ "$GPU" == "1" ]]; then
  if ! docker info 2>/dev/null | grep -qi 'runtimes.*nvidia'; then
    echo "⚠ 没检测到 docker 的 nvidia runtime，GPU 可能起不来。"
    echo "  装一下：sudo apt install nvidia-container-toolkit && sudo systemctl restart docker"
    echo ""
  fi
  echo "▶ GPU 模式（只有牙齿检测/DL 模型吃 GPU；IMU 的随机森林类训练还是走 CPU）"
fi

# down 之类的子命令直接透传
if [[ "${1:-}" =~ ^(down|stop|ps|logs|restart)$ ]]; then
  exec "${COMPOSE[@]}" "$@"
fi

"${COMPOSE[@]}" up -d --build "$@"

[ -f .env ] && set -a && source .env && set +a
PORT="${LABEL_SERVICE_PORT:-8383}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-<服务器IP>}"

echo ""
echo "=== label_service 已启动$([ "$GPU" == "1" ] && echo "（GPU）") ==="
echo "健康检查: http://${HOST_IP}:${PORT}/health"
echo "接口文档: http://${HOST_IP}:${PORT}/docs"
echo "排队情况: http://${HOST_IP}:${PORT}/api/v1/label/queue"
echo "日志:     bash label_service/up.sh logs -f   （或看 label_service/logs/）"
echo ""
echo "注意：label_infra 的 ALGO_SERVICE_URL 要指到 http://${HOST_IP}:${PORT}"
