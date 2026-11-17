"""同一批留出窗口：浮点模型（走同一个动作门）的逐类结果，跟导出日志里端侧 int8 的逐类结果对比。
导出脚本报的「跟 float 判决一致」是全部窗口算的，包括动作门直接判掉的未佩戴 / 静止（那些窗口 CNN 说什么都不算数），
所以这里另外算：只看动作门放行、真正交给 CNN 判的那些窗口，int8 跟 float 差多少。"""
import importlib.util
import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report

WT = os.path.expanduser("~/claude_exp/imu_train_wt")
sys.path[:0] = [os.path.join(WT, "src", "ml"), os.path.join(WT, "src", "data"), os.path.join(WT, "src", "dl")]
_s = importlib.util.spec_from_file_location("tv2", os.path.join(WT, "src", "ml", "train_v2.py"))
tv = importlib.util.module_from_spec(_s)
_s.loader.exec_module(tv)
import torch  # noqa: E402

# src/ml 下也有个 models 包，按名字 import 会撞上；按路径直接载 src/dl/models/cnn.py
_c = importlib.util.spec_from_file_location("dl_cnn", os.path.join(WT, "src", "dl", "models", "cnn.py"))
_cm = importlib.util.module_from_spec(_c)
_c.loader.exec_module(_cm)
CNN = _cm.CNN

TM = os.path.expanduser("~/algo_tinyml/service")
sys.path[:0] = [TM]
import serve  # noqa: E402

for run, gen in [(a, b) for a, b in (x.split("=") for x in sys.argv[1:])]:
    d = os.path.join(WT, "results", f"processed_{run}", "20hz_v2", "cnn")
    meta = json.load(open(os.path.join(d, "dl_cnn_best.json"), encoding="utf-8"))
    classes = meta["classes"] if isinstance(meta["classes"], list) else eval(meta["classes"])
    v = np.load(os.path.join(WT, "data", f"processed_{run}", "20hz", "val.npz"))
    X, y = v["X"].astype(np.float32), v["y"]
    if X.shape[1] != int(meta["window_size"]):
        X = X.transpose(0, 2, 1)
    cfg = meta["model_cfg"]
    net = CNN(X.shape[2], X.shape[1], len(classes), cfg)
    net.load_state_dict(torch.load(os.path.join(d, "dl_cnn_best.pt"), map_location="cpu", weights_only=True))
    net.eval()
    mu, sd = np.asarray(meta["ch_mean"], np.float64), np.asarray(meta["ch_std"], np.float64)
    P = tv.cnn_proba(net, mu, sd, X)
    gate = meta.get("motion_gate")
    mot = tv.motion_of(X)
    Pg = tv.apply_motion_gate(P, mot, gate) if gate else P
    f_pred = Pg.argmax(1)
    # 端侧：编出来的那份 C 逐窗口跑（serve.Engine 里已经带了动作门）
    eng = serve.Engine(serve.build(gen))
    Xc = np.ascontiguousarray(X.transpose(0, 2, 1))
    e_pred = np.concatenate([np.asarray(eng.infer(Xc[i:i + 512])[0]) for i in range(0, len(Xc), 512)])
    passed = np.ones(len(y), bool)
    if gate:
        i_nw, i_st = gate.get("notworn"), gate.get("still")
        gated = np.zeros(len(y), bool)
        for i in (i_nw, i_st):
            if i is not None and i >= 0:
                gated |= f_pred == i
        passed = ~gated
    print(f"== {run}  留出 {len(y)} 窗，动作门放行给 CNN 判的 {passed.sum()} 窗")
    print(f"   全部窗口 int8 跟 float 一致 {np.mean(e_pred == f_pred):.1%}；只看放行的 {np.mean(e_pred[passed] == f_pred[passed]):.1%}")
    for name, pr in (("float", f_pred), ("int8 ", e_pred)):
        rep = classification_report(y, pr, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
        print(f"   {name}: " + "  ".join(f"{c} {rep[c]['f1-score']:.3f}" for c in classes) + f"  | macro {rep['macro avg']['f1-score']:.3f}")
