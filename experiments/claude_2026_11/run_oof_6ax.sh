#!/bin/bash
# 等 v19neg 队列跑完，再跑一遍 6 轴大模型、存交叉验证逐窗预测（结果放 results_oof，不覆盖正式结果）
cd ~/imu_train
while pgrep -f "train_v2.py --spec" >/dev/null; do sleep 60; done
TRAIN_V2_DUMP_OOF=$HOME/claude_exp/oof_v19neg/w8_6ax.npz OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nice -n 19 \
  ~/miniconda3/bin/python src/ml/train_v2.py --spec experiments/claude_2026_11/specs/train_v2_cnn_w8_6ax_v19neg.json \
  --workers 4 --n_jobs 8 --results_dir results_oof > logs/train_oof_w8_6ax.log 2>&1 < /dev/null
echo "rc=$?" >> logs/train_oof_w8_6ax.log
