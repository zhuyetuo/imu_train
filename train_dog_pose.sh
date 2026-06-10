#!/bin/bash
# 狗姿态估计训练启动脚本
# 用法:
#   bash train_dog_pose.sh                                        # 默认配置
#   bash train_dog_pose.sh --batch 256                           # 指定 batch size
#   bash train_dog_pose.sh --data datasets/dog-pose-behavior/data.yaml  # 行为识别数据集
#   bash train_dog_pose.sh --resume                              # 从断点继续

set -e
cd "$(dirname "$0")"

# ── 退出时自动清理内存 ──────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[cleanup] 开始释放内存..."

    # 释放 /dev/shm 共享内存文件
    find /dev/shm -maxdepth 1 \( -name "torch_*" -o -name "*.shm" \) \
        -user "$(whoami)" -delete 2>/dev/null && \
        echo "[cleanup] /dev/shm 共享内存已清理" || true

    # 释放 SysV 共享内存段（PyTorch dataloader 用这个）
    ipcs -m 2>/dev/null | awk -v user="$(whoami)" '$3==user {print $2}' \
        | xargs -r -I{} ipcrm -m {} 2>/dev/null && \
        echo "[cleanup] SysV 共享内存已清理" || true

    # 释放页面缓存
    if sudo -n sync 2>/dev/null; then
        sudo sync
        echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 && \
            echo "[cleanup] 页面缓存已释放" || true
    fi

    echo "[cleanup] 当前内存状态:"
    free -h
    echo "[cleanup] 完成。"
}
trap cleanup EXIT

# ── 解析 --batch 参数（用户明确指定则跳过自动选择）────────────────────────
USER_BATCH=""
for arg in "$@"; do
    if [[ "$prev" == "--batch" ]]; then
        USER_BATCH="$arg"
    fi
    prev="$arg"
done

# ── 训练前检查内存 ──────────────────────────────────────────────────────────
AVAIL_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
echo "[train] 可用内存: ${AVAIL_GB}GB"

if [ "$AVAIL_GB" -lt 8 ]; then
    echo "[warn] 可用内存不足 8GB，尝试先释放缓存..."
    sudo sync 2>/dev/null || true
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true
    AVAIL_GB=$(awk '/MemAvailable/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
    echo "[train] 释放后可用内存: ${AVAIL_GB}GB"
fi

# ── 自动选 batch size（用户未指定时）────────────────────────────────────────
if [ -n "$USER_BATCH" ]; then
    echo "[train] 使用指定 batch=${USER_BATCH}"
    python train_dog_pose.py "$@"
else
    if [ "$AVAIL_GB" -ge 20 ]; then
        DEFAULT_BATCH=256
    elif [ "$AVAIL_GB" -ge 12 ]; then
        DEFAULT_BATCH=128
    else
        DEFAULT_BATCH=64
    fi
    echo "[train] 自动选择 batch=${DEFAULT_BATCH}（可用 ${AVAIL_GB}GB）"
    python train_dog_pose.py --batch "$DEFAULT_BATCH" "$@"
fi
