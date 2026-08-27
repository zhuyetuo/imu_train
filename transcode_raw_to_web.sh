#!/usr/bin/env bash
# 把原始录制视频转成浏览器兼容的编码，覆盖Nginx媒体目录里的同名文件——
# --ml_full_video模式下同步到媒体目录的完整录制视频是直接cp过去的，没有
# 走extract_clips.py裁剪片段时那道转码，浏览器HTML5 video标签放不了
# (常见原因：不是libx264 baseline、或者moov atom不在文件头没法流式播放)。
# 这里用跟extract_clips.py裁剪片段完全一样的ffmpeg参数重新编码，转完直接
# 覆盖Nginx媒体目录里的同名文件，Label Studio task JSON里的URL不用改。
#
# 用法:
#   ./transcode_raw_to_web.sh /path/to/multicam_..._cam1_imu1_raw.mp4 [更多文件...]
#
#   # 也可以给一整个目录，自动处理目录下所有.mp4
#   ./transcode_raw_to_web.sh data/raw_custom/data/2026_8_22/
#
# 环境变量:
#   NGINX_CONTAINER  Nginx容器名（默认label_studio_nginx，同delete_media_by_url.sh）
#   MEDIA_DIR        手动指定媒体根目录，跳过自动探测

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "用法: $0 <视频文件或目录> [更多...]" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[错误] 没有ffmpeg命令" >&2
    exit 1
fi

CONTAINER="${NGINX_CONTAINER:-label_studio_nginx}"

if [[ -z "${MEDIA_DIR:-}" ]]; then
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

echo "媒体根目录: $MEDIA_DIR/transcoded"
echo ""

# 展开参数：文件直接收，目录展开成目录下所有.mp4
files=()
for arg in "$@"; do
    if [[ -d "$arg" ]]; then
        while IFS= read -r -d '' f; do
            files+=("$f")
        done < <(find "$arg" -maxdepth 1 -name '*.mp4' -print0)
    elif [[ -f "$arg" ]]; then
        files+=("$arg")
    else
        echo "[跳过] $arg 不存在" >&2
    fi
done

if [[ ${#files[@]} -eq 0 ]]; then
    echo "[错误] 没有找到任何.mp4文件" >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

for src in "${files[@]}"; do
    basename_mp4="$(basename "$src")"
    dst="$MEDIA_DIR/transcoded/$basename_mp4"
    tmp_out="$tmp_dir/$basename_mp4"

    echo "▶ 转码: $basename_mp4"
    # 跟extract_clips.py裁剪片段时用的完全一样的参数：libx264 baseline，
    # faststart(moov atom放文件头，浏览器流式播放必需)，宽高转成偶数，
    # 原始视频无音频明确丢弃(-an)避免aac编码报错
    if ! ffmpeg -y -i "$src" \
        -c:v libx264 -profile:v baseline -level 3.1 -crf 23 -preset veryfast \
        -pix_fmt yuv420p -an -movflags +faststart \
        -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
        "$tmp_out" 2>"$tmp_dir/${basename_mp4}.log"; then
        echo "  ❌ 转码失败，日志：" >&2
        tail -20 "$tmp_dir/${basename_mp4}.log" >&2
        continue
    fi

    cp "$tmp_out" "$dst"
    echo "  → $dst  ($(du -h "$dst" | cut -f1))"
done

echo ""
echo "转码完成，直接刷新浏览器（Ctrl+Shift+R硬刷新，避免用到缓存的旧视频）再试一次播放"
