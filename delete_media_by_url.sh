#!/usr/bin/env bash
# 根据Label Studio里看到的媒体文件URL，直接在服务器上删除对应的本地文件——
# 不用自己拼MEDIA_DIR路径，脚本会先探测Nginx容器实际挂载的媒体根目录，
# 删除前会先列一遍要删的文件，确认后才真的删，避免URL传错或路径算错
# 导致误删。
#
# 用法:
#   ./delete_media_by_url.sh "http://192.168.2.140:8182/transcoded/xxx.mp4" ["http://.../yyy.csv" ...]
#
#   # 也可以一次传多个（引号里空格分隔，或者每个URL单独一个参数都行）
#   ./delete_media_by_url.sh "url1" "url2" "url3"
#
# 环境变量:
#   NGINX_CONTAINER  Nginx容器名（默认 label_studio_nginx，跟docker ps看到的一致）
#   MEDIA_DIR        手动指定媒体根目录，跳过自动探测（自动探测失败时用这个）

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "用法: $0 <URL1> [URL2] [URL3] ..." >&2
    exit 1
fi

CONTAINER="${NGINX_CONTAINER:-label_studio_nginx}"

if [[ -z "${MEDIA_DIR:-}" ]]; then
    # 自动探测挂载的媒体根目录：docker inspect拿到所有bind mount的宿主机
    # 路径，找到下面真的有transcoded/子目录的那一个，就是MEDIA_DIR——
    # 跟run_review_bins_all_days.sh里cp CSV/MP4进去的那个目录是同一个
    if ! command -v docker >/dev/null 2>&1; then
        echo "[错误] 没有docker命令，请手动设置 MEDIA_DIR=... 环境变量后重跑" >&2
        exit 1
    fi
    if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
        echo "[错误] 找不到容器 $CONTAINER（docker ps看到的名字如果不是这个，" \
             "传 NGINX_CONTAINER=实际名字 重跑）" >&2
        exit 1
    fi
    while IFS= read -r src; do
        if [[ -n "$src" && -d "$src/transcoded" ]]; then
            MEDIA_DIR="$src"
            break
        fi
    done < <(docker inspect -f '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' "$CONTAINER")
fi

if [[ -z "${MEDIA_DIR:-}" ]]; then
    echo "[错误] 没有在容器 $CONTAINER 的挂载点里找到带transcoded/子目录的媒体根目录。" \
         "手动确认路径后传 MEDIA_DIR=实际路径 重跑" >&2
    exit 1
fi

echo "媒体根目录: $MEDIA_DIR"
echo ""

# 把URL转成本地路径：去掉协议+host:port，剩下的就是相对MEDIA_DIR的路径——
# 跟review_to_labelstudio.py生成URL的逻辑对应(csv_url_prefix/video_url_prefix
# 分别对应MEDIA_DIR根目录和MEDIA_DIR/transcoded/)
url_to_path() {
    local url="$1"
    local rel_path
    rel_path="$(echo "$url" | sed -E 's#^https?://[^/]+/##')"
    # URL可能带百分号编码（比如空格变%20），转回真实文件名
    rel_path="$(python3 -c "import sys, urllib.parse; print(urllib.parse.unquote(sys.argv[1]))" "$rel_path")"
    echo "$MEDIA_DIR/$rel_path"
}

declare -a to_delete=()
declare -a not_found=()

for url in "$@"; do
    path="$(url_to_path "$url")"
    if [[ -f "$path" ]]; then
        to_delete+=("$path")
        echo "[将删除] $path"
    else
        not_found+=("$path")
        echo "[跳过-文件不存在] $path"
    fi
done

echo ""
if [[ ${#to_delete[@]} -eq 0 ]]; then
    echo "没有找到任何要删除的文件，检查一下URL是不是对的、MEDIA_DIR探测得对不对"
    exit 0
fi

read -r -p "以上 ${#to_delete[@]} 个文件确认删除？输入 yes 确认: " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "已取消，没有删除任何文件"
    exit 0
fi

for path in "${to_delete[@]}"; do
    rm -v "$path"
done
echo ""
echo "删除完成，共删除 ${#to_delete[@]} 个文件"
if [[ ${#not_found[@]} -gt 0 ]]; then
    echo "另有 ${#not_found[@]} 个URL对应的文件本来就不存在，已跳过"
fi
echo ""
echo "⚠️ 这只删除了Nginx媒体目录($MEDIA_DIR)里的这一份拷贝，不会删除："
echo "  1. 原始完整录制视频（DATA_ROOT下的对应文件）"
echo "  2. extract_clips.py裁剪出来的本地副本（RESULT_ROOT/{day}/clips_*/下）"
echo "如果要彻底清除这段内容，这两处也要手动检查删除。"
