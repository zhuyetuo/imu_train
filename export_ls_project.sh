#!/usr/bin/env bash
# 从Label Studio实例导出一个project的全部task(含未标注的)，输出文件名按
# project ID自动命名——一次要导出好几个project时不用每次手动改-o参数。
#
# 用法:
#   LS_HOST=http://192.168.2.140:8181 LS_PAT="<Personal Access Token>" \
#     ./export_ls_project.sh 91 155 203
#
#   # 也可以只传一个project ID
#   LS_HOST=http://192.168.2.140:8181 LS_PAT="<PAT>" ./export_ls_project.sh 91
#
# 输出: project{ID}_all_tasks.json（当前目录下，每个project一个文件）
#
# 环境变量:
#   LS_HOST  Label Studio实例地址，包含端口，不带末尾斜杠（必填）
#   LS_PAT   Personal Access Token，从该实例的My Account页面拿（必填）

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "用法: LS_HOST=... LS_PAT=... $0 <project_id> [project_id2] ..." >&2
    exit 1
fi

if [[ -z "${LS_HOST:-}" || -z "${LS_PAT:-}" ]]; then
    echo "[错误] 请先设置 LS_HOST 和 LS_PAT 环境变量" >&2
    exit 1
fi

for project_id in "$@"; do
    echo "▶ project $project_id"

    # access token只有约5分钟有效期，每个project单独换一次，避免导出
    # 慢的时候前面project耗光了后面project的token有效期
    access="$(curl -s -X POST "$LS_HOST/api/token/refresh" \
        -H "Content-Type: application/json" \
        -d "{\"refresh\": \"$LS_PAT\"}" | python3 -c "import sys,json; print(json.load(sys.stdin)['access'])")"

    if [[ -z "$access" ]]; then
        echo "  [错误] 换取access token失败，检查LS_HOST/LS_PAT是不是对的" >&2
        continue
    fi

    out="project${project_id}_all_tasks.json"
    curl -s -H "Authorization: Bearer $access" \
        "$LS_HOST/api/projects/$project_id/export?exportType=JSON&download_all_tasks=true" \
        -o "$out"

    size="$(wc -c < "$out")"
    echo "  → $out  (${size} 字节)"
    if [[ "$size" -lt 500 ]]; then
        echo "  ⚠️ 文件很小，大概率是错误响应不是真的导出数据，看看内容：" >&2
        cat "$out" >&2
        echo "" >&2
    fi
done
