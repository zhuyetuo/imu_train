#!/usr/bin/env bash
# 一步做完"导出project(含未标注task) + 把completed_by换成目标实例的用户ID"，
# 只用传PROJECT_ID，不用再手动分两步敲导出命令、再敲remap_completed_by.py命令。
#
# 用法:
#   ./export_and_remap_project.sh 91 155 203
#
# LS_HOST/LS_PAT都写死在下面的默认值里——这个仓库是私有的，只有内网能访问
# 这台Label Studio，按用户的判断这个token写进git历史可以接受，不用每次
# 传参数。这个PAT过期/被revoke之后要换新的，直接改下面DEFAULT_LS_PAT这行
# 就行，不用去改调用方式。
#
# 环境变量（可以临时覆盖下面的默认值，比如给不同的原实例跑）:
#   LS_HOST        原实例(要导出数据的那个)地址，包含端口，不带末尾斜杠
#   LS_PAT         原实例的Personal Access Token
#   COMPLETED_BY_MAP  "原ID=目标ID"的映射，逗号分隔多个。默认值是按两边
#                  实例各自真实的/api/users/查出来对齐的（不是猜的顺序）：
#                    原1(zyt290386779@gmail.com)  → 新6(zyt290386779@gmail.com)
#                    原2(leon@hiccpet.com)        → 新2(leon@hiccpet.com)
#                    原3(466038089@qq.com)        → 新4(466038089@qq.com)
#                    原5(raven@hiccpet.com)       → 新5(raven@hiccpet.com)
#                    原6(ml@model.local)          → 新3(ml@model.local)
#                  原4(1270329394@qq.com)新实例还没注册，暂时没放进默认映射，
#                  这个人的project导入前要先让他用邀请链接注册，查到新ID后
#                  用COMPLETED_BY_MAP补上"4=<新ID>"，比如：
#                  COMPLETED_BY_MAP="1=6,2=2,3=4,4=7,5=5,6=3" ./export_and_remap_project.sh 200
#
# 输出: project{ID}_remapped.json（当前目录下，可以直接拿去目标实例导入）
#
# 遇到COMPLETED_BY_MAP里没覆盖到的ID，脚本会在输出里明确警告(不是静默漏改)，
# 需要你去两边实例的/api/users/确认对应关系，补上COMPLETED_BY_MAP重跑

set -euo pipefail

DEFAULT_LS_HOST="http://192.168.2.140:8181"
DEFAULT_LS_PAT="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA4OTQ2Njk2OSwiaWF0IjoxNzgyMjY2OTY5LCJqdGkiOiI2MGI4ZmI2MTUyZmY0OGJmODZlMTViZDhjMTEwMDgzNSIsInVzZXJfaWQiOiIxIn0.N5mD05XdF9RImY703henvvI9r5mKvmrF_IO0Sv_VrNY"

LS_HOST="${LS_HOST:-$DEFAULT_LS_HOST}"
LS_PAT="${LS_PAT:-$DEFAULT_LS_PAT}"

if [[ $# -eq 0 ]]; then
    echo "用法: $0 <project_id> [project_id2] ..." >&2
    exit 1
fi

MAP_STR="${COMPLETED_BY_MAP:-1=6,2=2,3=4,5=5,6=3}"
IFS=',' read -r -a MAP_PAIRS <<< "$MAP_STR"
MAP_ARGS=()
for pair in "${MAP_PAIRS[@]}"; do
    MAP_ARGS+=(--map "$pair")
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for project_id in "$@"; do
    echo "▶ project $project_id"

    access="$(curl -s -X POST "$LS_HOST/api/token/refresh" \
        -H "Content-Type: application/json" \
        -d "{\"refresh\": \"$LS_PAT\"}" | python3 -c "import sys,json; print(json.load(sys.stdin)['access'])")"

    if [[ -z "$access" ]]; then
        echo "  [错误] 换取access token失败，检查LS_HOST/LS_PAT是不是对的" >&2
        continue
    fi

    raw="project${project_id}_all_tasks.json"
    curl -s -H "Authorization: Bearer $access" \
        "$LS_HOST/api/projects/$project_id/export?exportType=JSON&download_all_tasks=true" \
        -o "$raw"

    size="$(wc -c < "$raw")"
    if [[ "$size" -lt 500 ]]; then
        echo "  ⚠️ $raw 只有${size}字节，大概率是错误响应不是真的导出数据：" >&2
        cat "$raw" >&2
        echo "" >&2
        continue
    fi
    echo "  → $raw  (${size} 字节)"

    remapped="project${project_id}_remapped.json"
    python3 "$SCRIPT_DIR/src/remap_completed_by.py" \
        --input "$raw" --output "$remapped" "${MAP_ARGS[@]}"
done
