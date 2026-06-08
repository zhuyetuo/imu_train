"""
FilterNet (many-to-one 适配版)

原论文: Chambers & Yoder (2020). FilterNet: A Many-to-Many Deep Learning
Architecture for Time Series Classification. Sensors, 20(9), 2498.
https://doi.org/10.3390/s20092498
GitHub: https://github.com/WhistleLabs/FilterNet

原架构是 many-to-many（每个时间步预测标签），本实现保留其
多尺度 CNN + LSTM encoder，末端改为全局平均池化 → FC，
适配本项目的 many-to-one 滑动窗口管线。

原始架构要点:
  - Pre-conv:  1x Conv(filters, k=5)
  - Downsample stack: 3x Conv(filters, stride=2)  → 1/8 时间分辨率
  - Interpolation stack: 多尺度特征逐步上采样拼接
  - Pre-LSTM conv → LSTM(hidden) → 每步输出预测
本版简化: Pre-conv → Downsample stack → LSTM → GAP → FC
"""

import torch
import torch.nn as nn


class FilterNet(nn.Module):
    def __init__(self, n_channels, window_size, n_classes, cfg):
        super().__init__()
        filters     = cfg.get("filters", 100)
        kernel_size = cfg.get("kernel_size", 5)
        n_downsample= cfg.get("n_downsample", 3)   # stride=2 下采样层数，时序缩短 2^n 倍
        lstm_units  = cfg.get("lstm_units", 100)
        dropout     = cfg.get("dropout", 0.3)

        # ── Pre-conv（对应原论文 Component A）──────────────────────────────
        self.pre_conv = nn.Sequential(
            nn.Conv1d(n_channels, filters, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
        )

        # ── Strided downsample stack（Component B）────────────────────────
        # 每层 stride=2，连续 n_downsample 层 → 时序长度缩至 1/(2^n)
        down_layers = []
        for _ in range(n_downsample):
            down_layers += [
                nn.Conv1d(filters, filters, kernel_size=kernel_size,
                          stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
            ]
        self.downsample = nn.Sequential(*down_layers)

        # ── LSTM 时序建模（Component F）───────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=lstm_units,
            batch_first=True,
            dropout=dropout if n_downsample > 1 else 0,
        )

        # ── 分类头：全局平均池化 → FC ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_units, n_classes),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.pre_conv(x)              # (B, filters, T)
        x = self.downsample(x)            # (B, filters, T/2^n)
        x = x.permute(0, 2, 1)           # (B, T', filters) for LSTM
        x, _ = self.lstm(x)              # (B, T', lstm_units)
        x = x.mean(dim=1)               # Global Average Pooling over time
        return self.classifier(x)        # (B, n_classes)
