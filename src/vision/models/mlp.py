"""单帧 MLP 分类器：输入关键点向量 → 行为标签。"""

import torch
import torch.nn as nn


class PoseMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden is None:
            hidden = [256, 128]
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
