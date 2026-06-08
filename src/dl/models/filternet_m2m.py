"""
FilterNet Many-to-Many (原版架构)

原论文: Chambers & Yoder (2020). FilterNet: A Many-to-Many Deep Learning
Architecture for Time Series Classification. Sensors, 20(9), 2498.
https://doi.org/10.3390/s20092498

架构流程（与原论文一致）:
  Pre-conv → Strided Downsample (×n_downsample, stride=2 每次)
  → LSTM → 线性插值上采样回原始 T → Conv(1×1) 输出逐帧概率

训练时对每个时间步计算 CrossEntropy（使用窗口内逐帧标签 y_seq）。
预测时取每帧 argmax，再多数投票得到窗口级别标签用于评估。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FilterNetM2M(nn.Module):
    def __init__(self, n_channels, window_size, n_classes, cfg):
        super().__init__()
        filters      = cfg.get("filters", 100)
        kernel_size  = cfg.get("kernel_size", 5)
        n_downsample = cfg.get("n_downsample", 3)
        lstm_units   = cfg.get("lstm_units", 100)
        lstm_layers  = cfg.get("lstm_layers", 2)
        dropout      = cfg.get("dropout", 0.3)

        self.window_size  = window_size
        self.n_downsample = n_downsample

        # Pre-conv（Component A）
        self.pre_conv = nn.Sequential(
            nn.Conv1d(n_channels, filters, kernel_size=kernel_size,
                      padding=kernel_size // 2),
            nn.BatchNorm1d(filters),
            nn.ReLU(),
        )

        # Strided Downsample（Component B）
        down_layers = []
        for _ in range(n_downsample):
            down_layers += [
                nn.Conv1d(filters, filters, kernel_size=kernel_size,
                          stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
            ]
        self.downsample = nn.Sequential(*down_layers)

        # LSTM（Component F）
        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # 1×1 Conv 输出头（Component G）— 对每个时间步输出 n_classes logits
        self.output_conv = nn.Conv1d(lstm_units, n_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        T = x.shape[2]
        x = self.pre_conv(x)              # (B, filters, T)
        x = self.downsample(x)            # (B, filters, T')，T' = T / 2^n
        x = x.permute(0, 2, 1)           # (B, T', filters)
        x, _ = self.lstm(x)              # (B, T', lstm_units)
        x = x.permute(0, 2, 1)           # (B, lstm_units, T')

        # 线性插值上采样回原始时间步 T
        x = F.interpolate(x, size=T, mode="linear", align_corners=False)
        # (B, lstm_units, T)

        return self.output_conv(x)        # (B, n_classes, T)
