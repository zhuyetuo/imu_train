#!/usr/bin/env bash
# 一条命令把这台机器上的两个服务都拉起来。
#
#   ./up.sh              两个都起（label_service 重建镜像 + vision_service 后台）
#   ./up.sh -g           GPU 模式（透传给 label_service）
#   ./up.sh -d           只重启，不重建镜像（改了 .py 用这个，最快）
#   ./up.sh -p           先 git pull 再起
#   ./up.sh deploy       **算法服务更新到最新的一条命令**：git pull + 子模块 → label_service 依赖变了
#                        才重建镜像、否则重建容器+重启 → vision_service 停了再起 → 逐个探健康检查。
#                        只管这个仓库（IMU 推理、SAM、狗检测、找片段、向量索引、本地大模型）；
#                        web 平台是 label_infra 自己的 deploy_all.sh，两边各发各的
#   ./up.sh status       都在不在跑（label_service 那栏里有 edge-service）
#
#   端侧模型服务（端口 8900）跟 label_service 在同一个 compose 里，一起起停。
#   代码在 ~/algo_tinyml（deploy 会自动 clone / pull），那个仓库不单独起服务。
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

# deploy 会 git pull 到这个文件自己——bash 是边读边执行的，文件在跑到一半时被换掉
# 会执行到错位的行（第一次就这样：拉下来的新 up.sh 没生效，还得再跑一遍）。
# 所以 deploy 先把自己拷到临时文件再跑那一份
if [ "${1:-}" = "deploy" ] && [ -z "${UP_SH_REEXEC:-}" ]; then
    _TMP="$(mktemp /tmp/imu_up.XXXXXX.sh)"
    cp "${BASH_SOURCE[0]}" "$_TMP"
    UP_SH_REEXEC=1 UP_SH_HOME="$(pwd)" exec bash "$_TMP" "$@"
fi
[ -n "${UP_SH_HOME:-}" ] && cd "$UP_SH_HOME"

ONLY=""
PASS=()          # 透传给 label_service/up.sh 的开关
RESTART_ONLY=0
SUB=""

while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="${2:-}"; shift 2 ;;
        -d|--restart)   RESTART_ONLY=1; PASS+=("$1"); shift ;;
        -g|--gpu|-p|--pull) PASS+=("$1"); shift ;;
        down|stop|status|deploy) SUB="$1"; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "不认识的参数: $1（-h 看用法）"; exit 1 ;;
    esac
done

want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
line() { echo; echo "────────── $* ──────────"; }

# ── deploy：更新到最新，一条命令 ────────────────────────────────────────
# web 平台（label_infra）那边有自己的 deploy_all.sh，不会来调这里；两边解耦。
# DRY_RUN=1 只打印命令不执行。
deploy_run() { echo "  \$ $*"; [ "${DRY_RUN:-0}" = "1" ] && return 0; "$@"; }
deploy_probe() {   # $1 名字 $2 url $3 秒数
    local name=$1 url=$2 tries=${3:-30} i
    [ "${DRY_RUN:-0}" = "1" ] && { echo "  （DRY_RUN）$name $url"; return 0; }
    for ((i = 1; i <= tries; i++)); do
        curl -fsS -o /dev/null --max-time 2 "$url" 2>/dev/null && { echo "  ✓ $name"; return 0; }
        sleep 1
    done
    echo "  ✗ $name 等了 ${tries}s 还没通：$url"; return 1
}
deploy_field() {
    [ "${DRY_RUN:-0}" = "1" ] && { echo "?"; return; }
    curl -fsS --max-time 3 "$1" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('$2'))" 2>/dev/null || echo "?"
}
if [ "$SUB" = "deploy" ]; then
    FAILED=()
    GPU_FLAG=""
    for a in ${PASS[@]+"${PASS[@]}"}; do [ "$a" = "-g" ] || [ "$a" = "--gpu" ] && GPU_FLAG="-g"; done
    [ "${DEPLOY_GPU:-0}" = "1" ] && GPU_FLAG="-g"

    line "拉代码"
    OLD_REV="$(git rev-parse HEAD 2>/dev/null || echo none)"
    deploy_run git pull --ff-only || FAILED+=("git pull 失败（本地有改动？先 git stash）")
    # 采集端代码（witmotion_imu 子模块）跟着钉的版本走
    deploy_run git submodule update --init --recursive || true
    NEW_REV="$(git rev-parse HEAD 2>/dev/null || echo none)"

    if want label; then
        # 端侧那套代码。挂进容器给 edge-service 和「导出到端侧」用；不在就 clone
        line "algo_tinyml（端侧模型代码，~/algo_tinyml）"
        ALGO_TINYML_HOST="${ALGO_TINYML_HOST:-$HOME/algo_tinyml}"
        if [ -d "$ALGO_TINYML_HOST/.git" ]; then
            deploy_run git -C "$ALGO_TINYML_HOST" pull --ff-only || FAILED+=("algo_tinyml 拉不动（本地有改动？）")
        else
            deploy_run git clone "${ALGO_TINYML_REPO:-https://github.com/zhuyetuo/algo_tinyml.git}" "$ALGO_TINYML_HOST" \
                || FAILED+=("algo_tinyml clone 失败，端侧服务起不来")
        fi
        export ALGO_TINYML_HOST

        line "label_service（端口 8383）+ edge-service（端口 8900）"
        # 要不要重建镜像：看这次 pull 有没有动 Dockerfile/依赖；**另外直接探一下现有镜像**——
        # 人手动 pull 过、或者上次重建没成，git diff 看不出来，而镜像里缺东西的表现是
        # 「导出到端侧」报 No such file or directory: 'gcc'（2026-09-24 就是这样）
        IMG="imu-train-label-service$([ -n "$GPU_FLAG" ] && echo ':gpu')"
        image_lacks() {   # 镜像里没有 $1 这个命令（或镜像还不存在）→ 0
            [ "${DRY_RUN:-0}" = "1" ] && return 1
            ! docker run --rm --entrypoint sh "$IMG" -c "command -v $1" >/dev/null 2>&1
        }
        if { [ "$OLD_REV" != "$NEW_REV" ] && [ -n "$(git diff --name-only "$OLD_REV" "$NEW_REV" -- label_service/Dockerfile label_service/requirements-docker.txt 2>/dev/null)" ]; } \
           || image_lacks gcc; then
            echo "  依赖变了（或镜像里缺 gcc）→ 重建镜像（冷缓存十几分钟）"
            deploy_run bash label_service/up.sh $GPU_FLAG || FAILED+=("label_service 重建没成")
        else
            # -u：镜像不动，配置/环境变量变了就重建容器；-d 重启让挂载的新代码生效
            { deploy_run bash label_service/up.sh $GPU_FLAG -u && deploy_run bash label_service/up.sh $GPU_FLAG -d; } \
                || FAILED+=("label_service 重启没成")
        fi
    fi
    if want vision; then
        line "vision_service（端口 8385）"
        # 画面向量模型的权重：机器直连 HF 常卡死，先用 get_weights.sh 按国内源挨个试下到本地
        # （已经有了秒过）；三个源都不通它会说怎么从别的电脑拷。下不到不挡住起服务
        deploy_run ./vision_service/get_weights.sh || FAILED+=("向量模型权重没下到（看上面怎么拷）")
        # 姿态关键点模型（以图搜图第二路信号）：下不到不算失败，那一路自动关
        deploy_run ./vision_service/get_pose_weights.sh || true
        # 夜视这一摊默认关着（实测判不出动作，见 README），关着就别每次部署
        # 都去翻权重。想再打开：vision_service/.env 里写 LOWLIGHT_ENABLED=1
        if grep -qE "^LOWLIGHT_ENABLED=(1|true|yes|on)" vision_service/.env 2>/dev/null \
           && ls models/vision/lowlight/*.pth >/dev/null 2>&1; then
            deploy_run ./vision_service/get_lowlight_weights.sh || true
        fi
        deploy_run ./vision_service/run.sh down
        deploy_run ./vision_service/run.sh -d || FAILED+=("vision_service 起不来（看 vision_service/.run.log）")
    fi

    line "都通了吗"
    if want label; then
        deploy_probe "label_service " "http://127.0.0.1:${LABEL_SERVICE_PORT:-8383}/health" 30 || FAILED+=("label_service 不通")
        # 起来要把每个端侧模型的 C 编一遍再过自检，比 label_service 慢
        deploy_probe "edge-service  " "http://127.0.0.1:${EDGE_SERVICE_PORT:-8900}/health" 90 \
            || FAILED+=("edge-service 不通（bash label_service/up.sh logs edge-service 看自检哪里没过）")
    fi
    if want vision; then
        VB_PORT="${VISION_SERVICE_PORT:-8385}"
        VB="http://127.0.0.1:$VB_PORT"
        if deploy_probe "vision_service" "$VB/health" 60; then
            if [ "${DRY_RUN:-0}" != "1" ]; then
                echo "    狗检测   available=$(deploy_field "$VB/api/v1/dog/status" available)"
                echo "    找片段   available=$(deploy_field "$VB/api/v1/seek/status" available)   （false = 环境变量没 key；用平台「大模型 API」页的 key 不看这个）"
                # 向量模型是启动后后台加载的；第一次要从 HF 下约 400MB 权重。
                # 边下边画进度和预计时间，最多等 30 分钟（下载失败会立刻报错退出循环）
                for _ in $(seq 360); do
                    [ "$(deploy_field "$VB/api/v1/embed/status" loading)" = "True" ] || break
                    prog="$(curl -fsS --max-time 3 "$VB/api/v1/embed/status" 2>/dev/null | python3 -c '
import sys, json
p = (json.load(sys.stdin).get("progress") or {})
if not p: print("加载中…"); sys.exit()
bar = int(p["pct"] // 5)
eta = p.get("eta_s")
eta_s = "剩 %d 分 %02d 秒" % (eta // 60, eta % 60) if eta is not None else "估算中"
print("下载权重 [%s%s] %5.1f%%  %.0f/%.0f MB  %.1f MB/s  %s" % ("#" * bar, "." * (20 - bar), p["pct"], p["done_mb"], p["total_mb"], p["speed_mbps"], eta_s))
' 2>/dev/null || echo "加载中…")"
                    printf "\r    向量索引 %s          " "$prog"
                    sleep 5
                done
                printf "\r%80s\r" ""
                echo "    向量索引 available=$(deploy_field "$VB/api/v1/embed/status" available)   indexed=$(deploy_field "$VB/api/v1/embed/status" indexed_videos)   $( [ "$(deploy_field "$VB/api/v1/embed/status" available)" = "True" ] || echo "← $(deploy_field "$VB/api/v1/embed/status" error)" )"
                echo "    SAM      available=$(deploy_field "$VB/api/v1/sam/status" available)"
            fi
        else
            FAILED+=("vision_service 不通")
        fi
    fi
    echo
    if [ ${#FAILED[@]} -eq 0 ]; then echo "=== imu_train 这台全部更新完成 ==="; exit 0; fi
    echo "=== 有 ${#FAILED[@]} 项没成 ==="
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi

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
    # 不加 -d 会走 `docker compose up -d --build`，重建镜像要几分钟。代码是挂载
    # 进去的，改了 .py 根本不用重建——只有改了 Dockerfile / requirements-docker.txt
    # 才需要。默认行为跟 label_service/up.sh 保持一致（那边不加 -d 也是重建），
    # 但不能默默就开始烧几分钟：先说清楚，留三秒给人 Ctrl-C。
    if [ "$RESTART_ONLY" = "0" ]; then
        echo "⚠ 没加 -d：要重建镜像，几分钟起步。"
        echo "  只是改了 .py 的话不用重建（代码走挂载），用 ./up.sh${PASS[@]+ ${PASS[*]}} -d 秒起。"
        echo "  3 秒后开始重建，不想重建现在 Ctrl-C。"
        # 必须显式 trap：脚本是 `set -uo pipefail`（故意不开 -e），Ctrl-C 只会
        # 打断 sleep，然后若无其事地往下走去重建——等于这句"现在 Ctrl-C"是骗人的。
        # 实测过：不加这个 trap，等待期间按 Ctrl-C 重建照样启动。
        trap 'echo; echo "已取消，没有重建。要只重启用: ./up.sh -d"; exit 130' INT
        sleep 3
        trap - INT
    fi
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
