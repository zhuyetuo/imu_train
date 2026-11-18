cd ~/claude_exp/rate
while pgrep -f "bash run2.sh" > /dev/null; do sleep 30; done
for q in 512 128; do
  for ax in 3 6; do
    tag=q${ax}_${q}
    [ -f res_$tag.json ] && continue
    echo "$(date '+%F %T') $tag 开始" >> status.txt
    timeout 7200 nice -n 10 ~/miniconda3/bin/python multiscale_exp_q.py --spec spec_${ax}ax_20.json --model cnn --epochs 30 --quant-eval $q --out res_$tag.json > log_$tag.log 2>&1 < /dev/null \
      && echo "$(date '+%F %T') $tag 完成" >> status.txt || echo "$(date '+%F %T') $tag 失败" >> status.txt
  done
done
echo "$(date '+%F %T') 第三批结束" >> status.txt
