#!/usr/bin/env bash
# 检查 data/raw_custom/ 下每个日期目录标注覆盖情况：project-*.json 数量、
# 合并后任务数（如果之前跑过train_custom.sh、merged_tmp.json已存在的话）。
# 用来确认"标注的数据是不是全部都用上了"——train_custom.sh的--date/
# --extra_date要手动一个个传，这里列出磁盘上实际存在的全部日期目录，
# 方便跟自己传的参数核对，看有没有漏传某个批次。
#
# 用法:
#   bash src/data/check_annotation_coverage.sh

set -e
ROOT="data/raw_custom"

if [[ ! -d "$ROOT" ]]; then
  echo "[错误] 找不到 $ROOT"
  exit 1
fi

echo "=================================================="
echo "  $ROOT 下的日期目录 标注覆盖情况"
echo "=================================================="
printf "%-28s %10s %12s %14s\n" "日期目录" "project数" "合并任务数" "merged CSV行数"

total_projects=0
total_tasks=0
for d in "$ROOT"/*/; do
  d="${d%/}"
  name=$(basename "$d")
  n_proj=$(find "$d" -maxdepth 1 -name "project-*.json" 2>/dev/null | wc -l | tr -d ' ')
  [[ "$n_proj" -eq 0 ]] && continue

  n_tasks="-"
  merged_json="$d/merged_tmp.json"
  if [[ -f "$merged_json" ]]; then
    n_tasks=$(python3 -c "import json; print(len(json.load(open('$merged_json'))))" 2>/dev/null || echo "?")
  fi

  n_rows="-"
  merged_csv=$(find "$d" -maxdepth 1 -name "merged_${name}.csv" 2>/dev/null | head -1)
  if [[ -n "$merged_csv" && -f "$merged_csv" ]]; then
    n_rows=$(($(wc -l < "$merged_csv") - 1))
  fi

  printf "%-28s %10s %12s %14s\n" "$name" "$n_proj" "$n_tasks" "$n_rows"
  total_projects=$((total_projects + n_proj))
done

echo "--------------------------------------------------"
echo "共 $total_projects 个 project-*.json（分布在上面这些日期目录里）"
echo ""
echo "⚠ 这里只统计了已经导出到 data/raw_custom/ 下的project-*.json。"
echo "  如果Label Studio里还有标注完但没导出到这里的project，这个脚本看不到，"
echo "  需要去Label Studio网页上核对项目列表/任务数是不是都导出全了。"
echo ""
echo "训练时记得把上面列出的每个日期目录都传给 train_custom.sh 的"
echo "--date（主批次）或 --extra_date DATE:HZ（其它批次），漏传的不会报错，"
echo "只是那批数据不会参与训练。"
