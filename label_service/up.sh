#!/bin/bash
# 起 label_service（后台常驻），完成后打印访问地址。
#
# 用法：
#   bash label_service/up.sh -d     ← **平时用这个**。只重启，秒级
#   bash label_service/up.sh -u     改了环境变量（LABEL_MODELS 之类）用这个
#   bash label_service/up.sh        重建镜像再启动（**只有依赖变了才需要**）
#   bash label_service/up.sh -g     用 GPU 起（需要宿主机装了 nvidia-container-toolkit）
#   bash label_service/up.sh -p     先 git pull
#   bash label_service/up.sh down   停服务
#
# 三档的区别，选错了很费时间（默认那档在冷缓存的机器上要十几分钟：
# torch + 一堆科学计算的轮子）：
#
#   -d  docker compose restart   重启进程。**代码是挂载的**（Dockerfile 里
#                                没有 COPY 代码），所以改 .py 这就够了。
#                                但它用的是容器创建时的那套环境变量，
#                                **改了 LABEL_MODELS 之类不会生效**。
#   -u  docker compose up -d     配置变了就重建容器，镜像不动。
#                                改环境变量用这个。
#   默认 up -d --build           连镜像一起重建。只有 requirements-docker.txt
#                                或 Dockerfile 变了才需要。
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
      echo "注意：restart 用的是容器创建时的环境变量——改过 LABEL_MODELS"
      echo "      之类的话这样不生效，用 -u 重新创建容器。"
      exit 0
      ;;
    -u|--up)
      # 不 --build：镜像不动，只在 compose 配置/环境变量变了时重建容器。
      # 改 LABEL_MODELS 要走这条——restart 读不到新的环境变量，
      # 而"没生效"的表现是平台下拉里少一组，服务日志一切正常
      "${COMPOSE[@]}" up -d
      echo "已按当前配置起好（镜像没重建）"
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

# 构建前挑最快的 PyPI / torch 源，钉进 .env（挑过就不再测，见 pick_mirrors.sh）
bash pick_mirrors.sh || true
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
