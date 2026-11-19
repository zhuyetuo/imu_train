#!/bin/bash
# 等 6 轴大模型跑完，再依次跑 3 轴大模型、6 轴小模型、3 轴小模型（一个一个跑，避免内存不够）
cd ~/imu_train
while pgrep -f "train_v2_cnn_w8_6ax_v19neg" >/dev/null; do sleep 60; done
for t in w8_3ax w8s_6ax w8s_3ax; do
  [ -f STOP_v19neg ] && break
  L=logs/train_v19neg_${t}_$(date +%m%d_%H%M).log
  echo "$(date "+%F %T") $t 开始" >> logs/v19neg_queue.txt
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nice -n 19 ~/miniconda3/bin/python src/ml/train_v2.py --spec experiments/claude_2026_11/specs/train_v2_cnn_${t}_v19neg.json --workers 4 --n_jobs 8 > $L 2>&1 < /dev/null
  echo "$(date "+%F %T") $t 结束 rc=$?" >> logs/v19neg_queue.txt
done
echo "$(date "+%F %T") 队列结束" >> logs/v19neg_queue.txt
