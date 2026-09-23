#!/usr/bin/env bash
# 构建镜像前挑最快的源：PyPI 源 + torch 轮子源，各测一遍下载速度，把最快的写进
# label_service/.env（PIP_INDEX_URL / TORCH_INDEX_URL / TORCH_CUDA_INDEX_URL），
# docker compose 构建时会读这个文件。
#
#   bash label_service/pick_mirrors.sh          没写过才测（**默认**，up.sh 构建前调）
#   bash label_service/pick_mirrors.sh --force  重测一遍、覆盖
#
# **挑一次就钉住，不是每次构建都换**：源地址是 docker 那一层缓存键的一部分，
# 换一个源就等于那一层作废，torch 几百 MB 的 CUDA 包又要重下——
# "自动选最快"如果每次都选，反而每次都慢。所以只在 .env 里没有的时候测。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_FILE=.env
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1

# 测速要下**真实的轮子**，不能只拿索引页测：索引页几十 KB，测出来的是延迟不是带宽
# （第一版就是这么测的，把 0.4 MB/s 的源当成了最快的，torch 下了一个小时）。
# 各源的轮子文件名一样，先从索引页里找到一个 cp311 的 linux 轮子，再下 10 秒看速度
speed() {   # $1 索引根  $2 包名  $3 文件名要含的模式 → 字节/秒，失败 0
    local idx="${1%/}/$2/" href
    href=$(curl -fsSL --max-time 12 "$idx" 2>/dev/null \
           | grep -o 'href="[^"]*' | sed 's/href="//' | grep -- "$3" | grep -v 'sha256=.*sha256' | tail -1)
    [ -n "$href" ] || { echo 0; return; }
    href="${href%%#*}"
    case "$href" in http*) ;; /*) href="$(echo "$1" | sed -E 's#(https?://[^/]+).*#\1#')$href" ;; *) href="$idx$href" ;; esac
    # 只下 10 秒就断，看这 10 秒的平均速度（超时退出码不影响 -w 的输出）
    curl -sSL -o /dev/null --max-time 10 -w '%{speed_download}' "$href" 2>/dev/null \
        | awk '{printf "%d", $1+0}' || echo 0
}
pick() {    # $1 变量名  $2 包名  $3 轮子文件名模式  $4.. 候选源
    local var=$1 pkg=$2 pat=$3; shift 3
    if [ "$FORCE" = "0" ] && grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
        echo "  $var 已钉住：$(grep "^${var}=" "$ENV_FILE" | cut -d= -f2-)（要重测加 --force）"
        return
    fi
    local best="" best_s=0 u s
    for u in "$@"; do
        s=$(speed "$u" "$pkg" "$pat")
        printf '  %-58s %6.1f MB/s\n' "$u" "$(awk "BEGIN{print $s/1048576}")"
        [ "$s" -gt "$best_s" ] && { best=$u; best_s=$s; }
    done
    [ -n "$best" ] || { echo "  $var：一个都不通，用默认值"; return; }
    touch "$ENV_FILE"
    sed -i "/^${var}=/d" "$ENV_FILE"
    echo "${var}=${best}" >> "$ENV_FILE"
    echo "  → $var=$best（写进 label_service/.env）"
}

echo "▶ 挑最快的源（只在没钉住时测）"
pick PIP_INDEX_URL numpy 'cp311-cp311-manylinux.*x86_64.whl' \
    https://pypi.tuna.tsinghua.edu.cn/simple \
    https://mirrors.aliyun.com/pypi/simple \
    https://mirrors.cloud.tencent.com/pypi/simple \
    https://mirrors.ustc.edu.cn/pypi/simple \
    https://pypi.org/simple
pick TORCH_INDEX_URL torch 'cpu-cp311-cp311-manylinux.*x86_64.whl' \
    https://mirror.nju.edu.cn/pytorch/whl/cpu \
    https://download.pytorch.org/whl/cpu
pick TORCH_CUDA_INDEX_URL torch 'cu128-cp311-cp311-manylinux.*x86_64.whl' \
    https://mirror.nju.edu.cn/pytorch/whl/cu128 \
    https://download.pytorch.org/whl/cu128
