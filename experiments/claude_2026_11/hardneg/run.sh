cd ~/claude_exp/hardneg
for v in base plusB plusB_pseudo; do
  [ -f res_$v.json ] && continue
  echo "$(date '+%F %T') $v 开始" >> status.txt
  timeout 10800 nice -n 10 ~/miniconda3/bin/python multiscale_exp.py --spec spec_$v.json --model cnn --epochs 30 --folds 0 \
     --heldout /home/toky/claude_exp/hardneg/B_ev.json --out res_$v.json > log_$v.log 2>&1 < /dev/null \
     && echo "$(date '+%F %T') $v 完成" >> status.txt || echo "$(date '+%F %T') $v 失败" >> status.txt
done
echo "$(date '+%F %T') 全部结束" >> status.txt
