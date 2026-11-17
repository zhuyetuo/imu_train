cd ~/claude_exp/rate
while IFS='|' read -r tag spec; do
  [ -f "res_$tag.json" ] && continue
  echo "$(date '+%F %T') $tag 开始" >> status.txt
  timeout 7200 nice -n 10 ~/miniconda3/bin/python multiscale_exp.py --spec $spec --model cnn --epochs 30 --out res_$tag.json > log_$tag.log 2>&1 < /dev/null \
    && echo "$(date '+%F %T') $tag 完成" >> status.txt || echo "$(date '+%F %T') $tag 失败" >> status.txt
done < queue.txt
echo "$(date '+%F %T') 全部结束" >> status.txt
