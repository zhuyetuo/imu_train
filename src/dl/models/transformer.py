import torch
import torch.nn as nn
import math


class TransformerClassifier(nn.Module):
    """
    参考 2024 IEEE 犬只行为分类论文架构（Transformer Encoder + 分类头）。
    输入: (B, C, T) → permute → (B, T, C) → 线性投影 → Transformer → CLS token → FC
    """

    def __init__(self, n_channels, window_size, n_classes, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        nhead = cfg["nhead"]
        num_layers = cfg["num_layers"]
        drop = cfg["dropout"]
        dim_ff = cfg["dim_feedforward"]

        self.input_proj = nn.Linear(n_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=drop, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, C, T)
        x = x.permute(0, 2, 1)               # (B, T, C)
        x = self.input_proj(x)               # (B, T, d_model)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)       # (B, T+1, d_model)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.encoder(x)
        x = self.norm(x[:, 0])              # CLS token
        return self.fc(x)
