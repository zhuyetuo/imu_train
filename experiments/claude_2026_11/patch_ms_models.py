"""multiscale_exp.py 加几个端侧候选结构：valid（不补零，可流式增量计算）、valid_small、dsc_small（深度可分离）"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = 'WINDOW_MODELS = {"cnn": make_cnn, "dual": make_dual, "small": make_small, "tcn": make_tcn}'
new = '''def make_valid(C, W, n_classes, filters=(64, 128, 128), k=9):
    """卷积不补零（valid）：窗口里每个输出只看真实数据，滑动时中间层可以缓存复用（流式增量计算）。
    步长要是 2^层数 个采样点的整数倍（20 Hz、3 层池化 → 0.4 s / 0.8 s），池化才对得齐"""
    import torch.nn as nn
    layers, cin, t = [], C, W
    for c in filters:
        layers += [nn.Conv1d(cin, c, k), nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.2)]
        cin, t = c, (t - k + 1) // 2

    class V(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(cin * t, n_classes)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))
    return V()


def make_valid_small(C, W, n_classes):
    return make_valid(C, W, n_classes, (32, 64, 64))


def make_dsc_small(C, W, n_classes, k=9):
    """深度可分离卷积：每层 = 逐通道卷积（k×C）+ 1×1 混通道，参数和乘加大约是普通卷积的 1/k"""
    import torch.nn as nn

    def block(ci, co):
        return [nn.Conv1d(ci, ci, k, padding=k // 2, groups=ci), nn.Conv1d(ci, co, 1), nn.BatchNorm1d(co), nn.ReLU(),
                nn.MaxPool1d(2), nn.Dropout(0.2)]

    class D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(C, 32, k, padding=k // 2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                                      *block(32, 64), *block(64, 96))
            self.fc = nn.Linear(96 * (W // 8), n_classes)

        def forward(self, x):
            return self.fc(self.conv(x).flatten(1))
    return D()


WINDOW_MODELS = {"cnn": make_cnn, "dual": make_dual, "small": make_small, "tcn": make_tcn,
                 "valid": make_valid, "valid_small": make_valid_small, "dsc_small": make_dsc_small}'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('choices=["cnn", "dual", "small", "tcn", "filternet"]', 'choices=["cnn", "dual", "small", "tcn", "filternet", "valid", "valid_small", "dsc_small"]')
open(p, "w", encoding="utf-8").write(s)
print("ok")
