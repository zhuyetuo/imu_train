import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_channels, window_size, n_classes, cfg):
        super().__init__()
        filters = cfg["filters"]
        k = cfg["kernel_size"]
        drop = cfg["dropout"]

        layers = []
        in_ch = n_channels
        for out_ch in filters:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(drop),
            ]
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        # 计算展平后的维度
        pool_times = len(filters)
        out_len = window_size // (2 ** pool_times)
        self.fc = nn.Linear(in_ch * max(out_len, 1), n_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)
