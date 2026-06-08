"""
CollarCNN: 复现论文架构
"Deep Learning Classification of Canine Behavior Using a Single
Collar-Mounted Accelerometer: Real-World Validation"
Animals 2021, 11(6), 1549. https://doi.org/10.3390/ani11061549

原论文:
  - 输入: 3通道加速度计, 20Hz, 20s窗口 (400点)
  - 架构: Conv(64)→Conv(128)→Conv(256)→FC, MaxPool=4, BN+ReLU
  - Dropout(0.5) 在全连接层前

本实现适配6通道(acc+gyr)及任意窗口大小/类别数。
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, pool_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class CollarCNN(nn.Module):
    def __init__(self, n_channels, window_size, n_classes, cfg):
        super().__init__()
        filters     = cfg.get("filters", [64, 128, 256])
        kernel_size = cfg.get("kernel_size", 5)
        pool_size   = cfg.get("pool_size", 4)
        fc_dim      = cfg.get("fc_dim", 256)
        dropout     = cfg.get("dropout", 0.5)

        # 卷积块
        blocks = []
        in_ch = n_channels
        for out_ch in filters:
            blocks.append(ConvBlock(in_ch, out_ch, kernel_size, pool_size))
            in_ch = out_ch
        self.conv = nn.Sequential(*blocks)

        # 计算展平维度
        out_len = window_size
        for _ in filters:
            out_len = out_len // pool_size
        flat_dim = in_ch * max(out_len, 1)

        # 全连接分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, n_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        return self.classifier(self.conv(x))
