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

# 各测一个几 MB 的真实文件，看 speed_download。只拿首页测的话全是延迟，看不出带宽
speed() {   # $1 url → 字节/秒（整数），失败 0
    curl -fsSL -o /dev/null --max-time 12 -w '%{speed_download}' "$1" 2>/dev/null \
        | awk '{printf "%d", $1+0}' || echo 0
}
pick() {    # $1 变量名  $2 测速用的相对路径  $3.. 候选源
    local var=$1 probe=$2; shift 2
    if [ "$FORCE" = "0" ] && grep -q "^${var}=" "$ENV_FILE" 2>/dev/null; then
        echo "  $var 已钉住：$(grep "^${var}=" "$ENV_FILE" | cut -d= -f2-)（要重测加 --force）"
        return
    fi
    local best="" best_s=0 u s
    for u in "$@"; do
        s=$(speed "${u%/}/$probe")
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
# 测速文件：numpy 的轮子（PyPI 各镜像路径一样）；torch 源测 torch 目录索引页
# 没法拿"同一个轮子"（各版本文件名带 hash），改测 filelock 这种小包的索引 + 首个轮子
pick PIP_INDEX_URL "numpy/" \
    https://pypi.tuna.tsinghua.edu.cn/simple \
    https://mirrors.aliyun.com/pypi/simple \
    https://mirrors.cloud.tencent.com/pypi/simple \
    https://mirrors.ustc.edu.cn/pypi/simple \
    https://pypi.org/simple
pick TORCH_INDEX_URL "torch/" \
    https://mirrors.aliyun.com/pytorch-wheels/cpu \
    https://mirror.nju.edu.cn/pytorch/whl/cpu \
    https://download.pytorch.org/whl/cpu
pick TORCH_CUDA_INDEX_URL "torch/" \
    https://mirrors.aliyun.com/pytorch-wheels/cu128 \
    https://mirror.nju.edu.cn/pytorch/whl/cu128 \
    https://download.pytorch.org/whl/cu128
