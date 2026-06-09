"""序列 LSTM 分类器：输入关键点时序窗口 → 行为标签。"""

import torch
import torch.nn as nn


class PoseLSTM(nn.Module):
    def __init__(self, input_dim: int, n_classes: int,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        _, (h, _) = self.lstm(x)
        return self.classifier(h[-1])
