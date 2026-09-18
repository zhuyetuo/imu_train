#!/usr/bin/env bash
# 挑最快的 pip 源：几个国内镜像 + 官方各下同一个小包 3 秒，按速度选，结果缓存一天。
#
#   source vision_service/pick_pip_mirror.sh     → 设好 PIP_INDEX_URL（已经手动设了就不动）
#   ./vision_service/pick_pip_mirror.sh --force  → 重新测一遍并打印
#
# 为什么：清华源有时只有几百 KB/s，换个源就十几 MB/s；哪个快跟时段、跟运营商都有关，
# 写死一个不靠谱，每次装大包前测一下最省事。
_PPM_CACHE="${PIP_MIRROR_CACHE:-$HOME/.cache/imu_train_pip_mirror}"
_PPM_TTL=86400
# 拿来测速的文件：pip 自己的 wheel（1.7MB 左右，每个源都有）
_PPM_PROBE="packages/8a/6a/19e9fe04fca059ccf770861c7d5721ab4c2aebc539889e97c7977528a53b/pip-24.0-py3-none-any.whl"
_PPM_CANDIDATES=(
  "清华|https://pypi.tuna.tsinghua.edu.cn/simple|https://pypi.tuna.tsinghua.edu.cn"
  "阿里云|https://mirrors.aliyun.com/pypi/simple|https://mirrors.aliyun.com/pypi"
  "中科大|https://mirrors.ustc.edu.cn/pypi/simple|https://mirrors.ustc.edu.cn/pypi"
  "腾讯云|https://mirrors.cloud.tencent.com/pypi/simple|https://mirrors.cloud.tencent.com/pypi"
  "华为云|https://repo.huaweicloud.com/repository/pypi/simple|https://repo.huaweicloud.com/repository/pypi"
  "官方 PyPI|https://pypi.org/simple|https://files.pythonhosted.org"
)

_ppm_measure() {   # $1 下载根 → 打印 字节/秒（失败 0）
    local out
    out="$(curl -sL --max-time 3 --connect-timeout 3 -o /dev/null -w '%{size_download} %{time_total}' "$1/$_PPM_PROBE" 2>/dev/null || echo "0 1")"
    awk -v s="${out%% *}" -v t="${out##* }" 'BEGIN { if (t + 0 <= 0) t = 1; printf "%d", s / t }'
}

pick_pip_mirror() {
    local force="${1:-}"
    if [ "$force" != "--force" ] && [ -n "${PIP_INDEX_URL:-}" ] && [ "${PIP_INDEX_URL_AUTO:-}" != "1" ]; then
        return 0     # 人手动指定了，不动
    fi
    if [ "$force" != "--force" ] && [ -f "$_PPM_CACHE" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$_PPM_CACHE" 2>/dev/null || echo 0) ))
        if [ "$age" -lt "$_PPM_TTL" ]; then
            export PIP_INDEX_URL="$(cat "$_PPM_CACHE")" PIP_INDEX_URL_AUTO=1
            return 0
        fi
    fi
    echo "测各 pip 源速度（各 3 秒）..."
    local best_url="" best_bps=0 line name idx dl bps
    for line in "${_PPM_CANDIDATES[@]}"; do
        IFS='|' read -r name idx dl <<<"$line"
        bps="$(_ppm_measure "$dl")"
        printf "  %-8s %8.1f MB/s\n" "$name" "$(awk -v b="$bps" 'BEGIN { printf "%.1f", b / 1048576 }')"
        if [ "$bps" -gt "$best_bps" ]; then best_bps="$bps"; best_url="$idx"; fi
    done
    if [ -z "$best_url" ]; then
        best_url="https://pypi.tuna.tsinghua.edu.cn/simple"
        echo "  都测不到速度，先用清华源"
    fi
    echo "  → 用 $best_url"
    mkdir -p "$(dirname "$_PPM_CACHE")" && echo "$best_url" > "$_PPM_CACHE"
    export PIP_INDEX_URL="$best_url" PIP_INDEX_URL_AUTO=1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    pick_pip_mirror "${1:-}"
    echo "PIP_INDEX_URL=$PIP_INDEX_URL"
else
    pick_pip_mirror
fi
