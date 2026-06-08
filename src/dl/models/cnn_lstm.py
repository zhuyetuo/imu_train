import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(self, n_channels, window_size, n_classes, cfg):
        super().__init__()
        filters = cfg["cnn_filters"]
        k = cfg["cnn_kernel_size"]
        lstm_units = cfg["lstm_units"]
        drop = cfg["dropout"]

        cnn_layers = []
        in_ch = n_channels
        for out_ch in filters:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)
        self.lstm = nn.LSTM(in_ch, lstm_units, batch_first=True, bidirectional=False)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(lstm_units, n_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.cnn(x)             # (B, C', T')
        x = x.permute(0, 2, 1)     # (B, T', C')
        _, (h, _) = self.lstm(x)
        x = self.drop(h[-1])        # (B, lstm_units)
        return self.fc(x)
