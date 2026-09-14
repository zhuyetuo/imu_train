#!/usr/bin/env bash
# 一条命令把这台机器上的两个服务都拉起来。
#
#   ./up.sh              两个都起（label_service 重建镜像 + vision_service 后台）
#   ./up.sh -g           GPU 模式（透传给 label_service）
#   ./up.sh -d           只重启，不重建镜像（改了 .py 用这个，最快）
#   ./up.sh -p           先 git pull 再起
#   ./up.sh status       两个分别在不在跑
#   ./up.sh down         两个都停
#
#   ./up.sh --only label     只动 label_service
#   ./up.sh --only vision    只动 vision_service
#
# 开关可以叠：./up.sh -p -g
#
# ── 为什么本来是两条命令 ──────────────────────────────────────────────
#
# 它们是两个服务，而且是**故意分开**的：
#
#   label_service   docker compose 起，天然后台常驻   端口 8383   IMU 推理，纯 CPU
#   vision_service  宿主机上的 python                 端口 8385   SAM 分割，吃 GPU
#
# vision_service 没做成容器，是因为 SAM2 和权重都装在 conda 环境里，塞进容器
# 要把 CUDA 一起搬进去。端口也是刻意错开的：混进一个进程池的话，SAM 这种
# 长时 GPU 任务会让 IMU 推理排在它后面。
#
# 所以这个脚本只是个门面——它不改变上面那个结构，只是省得手敲两遍。真要单独
# 操作某一个，下面两条原样还在，这个脚本调的也就是它们：
#
#   bash label_service/up.sh -g -d
#   ./vision_service/run.sh -d
#
# ── vision_service 起不来不算失败 ─────────────────────────────────────
#
# 它是降级项：起不来的话平台把 SAM 按钮置灰，标注照常手画框，其它功能一概
# 不受影响。所以这里只警告、不返回非零——返回非零会让人以为整套都没起来，
# 反而去折腾已经好好跑着的 label_service。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ONLY=""
PASS=()          # 透传给 label_service/up.sh 的开关
RESTART_ONLY=0
SUB=""

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${2:-}"; shift 2 ;;
        -d|--restart)   RESTART_ONLY=1; PASS+=("$1"); shift ;;
        -g|--gpu|-p|--pull) PASS+=("$1"); shift ;;
        down|stop|status) SUB="$1"; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "不认识的参数: $1（-h 看用法）"; exit 1 ;;
    esac
done

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
line() { echo; echo "────────── $* ──────────"; }

case "$SUB" in
    status)
        # 有一个没跑就返回非零：这样 status 能直接拿去做监控/守护脚本的判据，
        # 而不是只能拿眼睛看输出
        srv=0
        want label && { line "label_service"; bash label_service/up.sh ps 2>&1 | tail -n 5; }
        want vision && { line "vision_service"; ./vision_service/run.sh status || srv=1; }
        exit $srv
        ;;
    down|stop)
        # 先停 vision 再停 label：反过来的话，label 停了而 vision 还在，平台会
        # 短暂处在"SAM 能用、IMU 推理不能用"的状态，看着像是坏了半边
        want vision && { line "vision_service"; ./vision_service/run.sh down; }
        want label && { line "label_service"; bash label_service/up.sh down; }
        exit 0
        ;;
esac

rc=0
if want label; then
    line "label_service（端口 8383）"
    bash label_service/up.sh ${PASS[@]+"${PASS[@]}"} || rc=1
fi

if want vision; then
    line "vision_service（端口 8385）"
    # -d 是"只重启"，对这个就是先停再起；不加 -d 时它没在跑才起，在跑就不动
    # （run.sh -d 自己会说"已经在跑了"）
    [ "$RESTART_ONLY" = "1" ] && ./vision_service/run.sh down >/dev/null 2>&1
    if ! ./vision_service/run.sh -d; then
        echo "  ⚠ vision_service 没起来。这是降级项，不影响 label_service 和标注："
        echo "    平台那边 SAM 按钮会置灰，手画框照常。日志: vision_service/.run.log"
        # 故意不置 rc=1，见文件头的说明
    fi
fi

echo
[ "$rc" = "0" ] && echo "完成。看状态: ./up.sh status    停: ./up.sh down" \
                || echo "label_service 那边出错了，往上翻看报错。"
exit $rc
