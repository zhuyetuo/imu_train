#!/bin/bash
# 夜里跑：v1 / v2b 两路并行（错开 90 秒，避开读数据时的内存峰值），完了再跑 v3。8 点半还没跑完就停，留资源给白天的标注。
cd ~/claude_exp/imu_pages
export TRAIN_V2_TL_CACHE=/home/toky/claude_exp/tlcache TRAIN_V2_WIN_CACHE=/home/toky/claude_exp/wincache
one() {
  local n=$1
  TRAIN_V2_DUMP_SEG=/home/toky/claude_exp/evt/${n}_seg.pkl.gz ~/miniconda3/bin/python src/ml/train_v2.py \
    --spec /home/toky/claude_exp/evt/spec_${n}.json --workers 10 > ~/claude_exp/evt/${n}.log 2>&1 < /dev/null
  echo "rc=$?" >> ~/claude_exp/evt/${n}.log
}
guard() {
  while pgrep -f "spec_v[123]" >/dev/null; do
    [ "$(date +%H%M)" -ge 0830 ] && { pkill -f "spec_v[123]"; echo "8:30 到了，停" >> ~/claude_exp/evt/cmp.log; }
    sleep 30
  done
}
guard &
one v1 &
sleep 90
one v2b &
wait %2 %3 2>/dev/null; wait
pgrep -f "spec_v[12b]" >/dev/null || one v3
echo "全部跑完 $(date +%H:%M)" >> ~/claude_exp/evt/cmp.log
