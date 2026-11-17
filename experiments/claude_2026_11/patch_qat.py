"""给 imu_train 的 train_v2 加量化感知训练（QAT），给 algo_tinyml 的 export_train 加大校准集。在仓库根目录跑。"""
import sys

which = sys.argv[1]

if which == "imu_train":
    p = "src/ml/train_v2.py"
    s = open(p, encoding="utf-8").read()

    # 1) 参数：model_params.qat_epochs（默认 0 = 不做，行为跟原来一样）
    old = '''    if params.get("dropout") not in (None, ""):
        cfg["dropout"] = float(params["dropout"])
    return cfg, int(params.get("epochs") or 20)'''
    new = '''    if params.get("dropout") not in (None, ""):
        cfg["dropout"] = float(params["dropout"])
    if params.get("qat_epochs") not in (None, "", 0, "0"):
        cfg["qat_epochs"] = int(params["qat_epochs"])
    return cfg, int(params.get("epochs") or 20)'''
    assert old in s
    s = s.replace(old, new)

    # 2) QAT 的几个函数，放在 fit_cnn 前面
    anchor = "def fit_cnn(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: dict, epochs: int, seed: int = 0):"
    qat = '''# ── 量化感知训练（QAT）────────────────────────────────────────────────────────
# 板上跑的是 algo_tinyml 训练后量化（tinyml/net.py quantize）出来的 int8：BN 折进卷积，
# 权重逐输出通道对称量化（max|w|/127），输入和每层输出按分位数范围做非对称 int8（256 格）。
# 直接训出来的 float 模型对这一步很敏感：2026-11-17 那版换个分位数，int8 跟 float 的判决一致率
# 就在 52%~89% 之间跳。QAT 在 float 训完之后再微调几轮，前向里按**同一套方案**模拟量化
# （直通估计器传梯度），让模型学会在 int8 下也判得对。存下来的还是普通的 state_dict，
# 导出照旧走训练后量化，不改端侧任何代码。
QAT_PCT = 99.9   # 激活范围的分位数，跟导出时最常选中的那档一致


def _fq_weight(w):
    """逐输出通道对称 int8（跟 net.quantize 的 w_scale 一样），直通估计器"""
    s = w.detach().abs().flatten(1).max(1).values.clamp_min(1e-12) / 127.0
    s = s.view(-1, *([1] * (w.dim() - 1)))
    return w + ((w / s).round().clamp(-127, 127) * s - w).detach()


def _fq_act(x, lo: float, hi: float):
    """非对称 int8（跟 net._affine 一样：0 要能精确表示），直通估计器"""
    import torch

    lo, hi = min(lo, 0.0), max(hi, 0.0)
    sc = max((hi - lo) / 255.0, 1e-8)
    zp = int(np.clip(round(-128 - lo / sc), -128, 127))
    q = (torch.round(x / sc) + zp).clamp(-128, 127)
    return x + ((q - zp) * sc - x).detach()


def _qat_forward(net, x, ranges: dict, update: bool):
    """跟 tinyml 的 FloatNet → QNet 同一个计算图：BN 用滑动统计量折进卷积 → relu → 量化 → 池化"""
    import torch
    import torch.nn.functional as F

    def rng(key, t):
        if update:
            flat = t.detach().flatten()
            if flat.numel() > 200_000:
                flat = flat[torch.randint(0, flat.numel(), (200_000,), device=flat.device)]
            tail = (100.0 - QAT_PCT) / 200.0
            lo, hi = torch.quantile(flat.float(), torch.tensor([tail, 1 - tail], device=flat.device)).tolist()
            if key in ranges:
                lo, hi = 0.9 * ranges[key][0] + 0.1 * lo, 0.9 * ranges[key][1] + 0.1 * hi
            ranges[key] = (lo, hi)
        return ranges[key]

    x = _fq_act(x, *rng("in", x))
    mods = list(net.conv)
    for i in range(0, len(mods), 5):
        conv, bn, drop = mods[i], mods[i + 1], mods[i + 4]
        inv = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * inv.view(-1, 1, 1)
        b = (conv.bias - bn.running_mean) * inv + bn.bias
        x = F.relu(F.conv1d(x, _fq_weight(w), b, padding=conv.padding[0]))
        x = _fq_act(x, *rng(f"c{i}", x))
        x = F.max_pool1d(x, 2)
        x = F.dropout(x, drop.p, net.training)
    out = F.linear(x.flatten(1), _fq_weight(net.fc.weight), net.fc.bias)
    return _fq_act(out, *rng("fc", out))


def _qat_finetune(net, Xt, yt, lossf, dev, epochs: int, rng, shift: int):
    """float 训完之后接着微调：BN 统计量冻住（板上用的就是它们），只更新权重和 BN 的 gamma / beta"""
    import torch

    ranges: dict = {}
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-3)
    bs = 512
    nb = (len(yt) + bs - 1) // bs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs * nb))
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(yt))
        for b in range(nb):
            ix = perm[b * bs:(b + 1) * bs]
            xb = Xt[ix].to(dev)
            xb = xb * (1 + 0.1 * torch.randn(len(ix), 1, 1, device=dev)) + 0.02 * torch.randn_like(xb)
            xb = torch.roll(xb, int(rng.integers(-shift, shift + 1)), 2)
            loss = lossf(_qat_forward(net, xb, ranges, update=True), yt[ix].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    return ranges


'''
    assert anchor in s
    s = s.replace(anchor, qat + anchor, 1)

    # 3) fit_cnn 里 float 训完之后接 QAT
    old = '''            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    return net.cpu().eval(), mu.astype(np.float64), sd.astype(np.float64)'''
    new = '''            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    if int(cfg.get("qat_epochs") or 0) > 0:
        _qat_finetune(net, Xt, yt, lossf, dev, int(cfg["qat_epochs"]), rng, shift)
    return net.cpu().eval(), mu.astype(np.float64), sd.astype(np.float64)'''
    assert old in s
    s = s.replace(old, new)

    # 4) CNN 构造只认这三个键，qat_epochs 别传进去也行（多了不碍事），打印里带上
    open(p, "w", encoding="utf-8").write(s)

    p = "tests/test_train_v2.py"
    t = open(p, encoding="utf-8").read()
    t += '''

def test_qat_fake_quant_matches_int8_grid():
    """QAT 的模拟量化跟导出的 int8 方案一致：权重逐通道 ≤127 格、激活落在 256 格上，前向形状对"""
    import importlib.util as _u
    import torch
    _s = _u.spec_from_file_location("tv2q", os.path.join(ROOT, "src", "ml", "train_v2.py")); tv = _u.module_from_spec(_s); _s.loader.exec_module(tv)
    w = torch.randn(4, 3, 5)
    q = tv._fq_weight(w)
    s = w.abs().flatten(1).max(1).values / 127
    assert torch.allclose((q / s.view(-1, 1, 1)).round(), q / s.view(-1, 1, 1), atol=1e-4)
    x = torch.linspace(-1, 3, 1000)
    a = tv._fq_act(x, -1.0, 3.0)
    assert len(torch.unique(a)) <= 256
    sys.path.insert(0, os.path.join(ROOT, "src", "dl"))
    from models.cnn import CNN
    net = CNN(5, 32, 4, dict(filters=[8, 8, 8], kernel_size=5, dropout=0.1)).eval()
    out = tv._qat_forward(net, torch.randn(6, 5, 32), {}, update=True)
    assert out.shape == (6, 4)
'''
    open(p, "w", encoding="utf-8").write(t)
    print("imu_train patched")

elif which == "algo_tinyml":
    p = "service/export_train.py"
    s = open(p, encoding="utf-8").read()
    old = "    per = max(1, 256 // len(classes))"
    new = ("    # 校准集原来每类 ~50 窗（共 256），分位数在这么点数据上估得很飘：同一个模型换一档分位数，\n"
           "    # int8 跟 float 一致率能从 52% 跳到 89%。加到每类 ~200 窗（共 1024）\n"
           "    per = max(1, 1024 // len(classes))")
    assert old in s
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    print("algo_tinyml patched")
