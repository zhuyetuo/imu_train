#!/bin/bash
# 无人值守队列：等冒烟测试结束 → 按 queue.txt 一个一个跑 → 每个跑完重写 summary.md。
# 单个实验最长 2.5 小时，超时就杀掉记成「超时」，接着跑下一个。断开 SSH 不影响（nohup + setsid 启动）。
cd ~/claude_exp/multiscale
PY=~/miniconda3/bin/python
while [ ! -f s2_done.txt ]; do sleep 20; done
echo "$(date '+%F %T')	-	队列开始" >> status.txt
grep -v '^#' queue.txt | while IFS='|' read -r tag desc args; do
  [ -z "$tag" ] && continue
  [ -f "res_$tag.json" ] && continue
  [ -f STOP ] && { echo "$(date '+%F %T')	-	看到 STOP 文件，停" >> status.txt; break; }
  spec=spec.json
  case "$args" in *--spec*) ;; *) args="$args --spec $spec";; esac
  echo "$(date '+%F %T')	$tag	运行中" >> status.txt
  $PY summary.py > /dev/null 2>&1 < /dev/null
  if timeout 9000 nice -n 10 $PY multiscale_exp.py $args --out "res_$tag.json" > "log_$tag.log" 2>&1 < /dev/null; then
    echo "$(date '+%F %T')	$tag	完成" >> status.txt
  else
    rc=$?; [ $rc -eq 124 ] && s="超时" || s="失败($rc)"
    echo "$(date '+%F %T')	$tag	$s" >> status.txt
  fi
  $PY summary.py > /dev/null 2>&1 < /dev/null
done
echo "$(date '+%F %T')	-	队列结束" >> status.txt
$PY summary.py > /dev/null 2>&1
