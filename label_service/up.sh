#!/bin/bash
# 起 label_service（后台常驻），完成后打印访问地址。
#
# 用法：
#   bash label_service/up.sh        重建并启动
#   bash label_service/up.sh -p     先 git pull 再重建
#   bash label_service/up.sh -d     只重启，不重建镜像（改了 .py 用这个，最快）
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

case "${1:-}" in
  -p|--pull)
    echo "▶ 拉取最新代码..."
    git -C "$(git rev-parse --show-toplevel)" pull --ff-only
    echo ""
    shift
    ;;
  -d|--restart)
    # 代码是挂载进去的，改 .py 只要重启，不用重建镜像
    docker compose restart
    echo "已重启（代码走挂载，不需要重建镜像）"
    exit 0
    ;;
esac

docker compose up -d --build "$@"

[ -f .env ] && set -a && source .env && set +a
PORT="${LABEL_SERVICE_PORT:-8383}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-<服务器IP>}"

echo ""
echo "=== label_service 已启动 ==="
echo "健康检查: http://${HOST_IP}:${PORT}/health"
echo "接口文档: http://${HOST_IP}:${PORT}/docs"
echo "排队情况: http://${HOST_IP}:${PORT}/api/v1/label/queue"
echo "日志:     docker compose -f label_service/docker-compose.yml logs -f   （或看 label_service/logs/）"
echo ""
echo "注意：label_infra 的 ALGO_SERVICE_URL 要指到 http://${HOST_IP}:${PORT}"
