"""宿主机上跑：把四个模型的结果 / 预处理目录拷进 ~/imu_train（容器里的 /app），日志放到容器看得见的地方，写 items.json"""
import json
import os
import shutil
import subprocess

WT = os.path.expanduser("~/claude_exp/imu_train_wt")
IT = os.path.expanduser("~/imu_train")
LOGS = os.path.expanduser("~/claude_exp/export_w8")
STAGE = os.path.join(IT, "tmp", "import_logs")
os.makedirs(STAGE, exist_ok=True)

MODELS = [
    # tag, run, kind, axes, window_s, edge_tag
    ("cnn_w8_3ax", "pool_20261116_all__cnn_w8_3ax", "cnn", 3, 8, "cnn_w8_3ax"),
    ("cnn_w8_6ax", "pool_20261116_all__cnn_w8_6ax", "cnn", 6, 8, "cnn_w8_6ax"),
    ("xgb_srv_3ax", "pool_20261116_all__xgb_srv_3ax", "xgb", 3, 10, None),
    ("xgb_srv_6ax", "pool_20261116_all__xgb_srv_6ax", "xgb", 6, 10, None),
]
items = []
for tag, run, kind, ax, win, edge_tag in MODELS:
    for top in ("results", "data"):
        src = os.path.join(WT, top, f"processed_{run}")
        dst = os.path.join(IT, top, f"processed_{run}")
        if not os.path.exists(dst):
            shutil.copytree(src, dst, symlinks=True)
    spec_file = json.load(open(os.path.join(WT, "tmp", f"train_v2_{tag}.json"), encoding="utf-8"))
    sub = "cnn/dl_cnn_best.pt" if kind == "cnn" else "xgb/ml_xgb.pkl"
    log = os.path.join(STAGE, f"{tag}.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"▶ 宿主机 GPU 上训练（同一份 src/ml/train_v2.py，spec 见 experiments/claude_2026_11/specs/train_v2_{tag}.json）\n\n")
        f.write(open(os.path.join(LOGS, f"train_{tag}.log" if kind == "xgb" else f"train_{ax}ax.log"), encoding="utf-8").read())
    edge = None
    if edge_tag:
        exp = open(os.path.join(LOGS, f"export_{ax}ax.log"), encoding="utf-8").read()
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n▶ 导出到端侧\n" + exp)
        line = next(l for l in reversed(exp.splitlines()) if l.startswith("EXPORT_RESULT "))
        edge = json.loads(line[len("EXPORT_RESULT "):])
        edge["gen"] = f"core/models/generated_{edge_tag}"
        edge["meta"] = f"core/models/generated_{edge_tag}/meta.json"
        edge["reloaded"], edge["reload_error"] = True, None
    secs = 1135 if kind == "cnn" else 600
    items.append({
        "tag": tag, "run": run, "model_type": kind, "secs": secs,
        "model_path": f"/app/results/processed_{run}/20hz_v2/{sub}",
        "log": f"/app/tmp/import_logs/{tag}.log",
        "command": f"python src/ml/train_v2.py --spec experiments/claude_2026_11/specs/train_v2_{tag}.json",
        "spec": {"axes": ax, "window_s": float(win), "stride_s": 1.0, "edge_size": kind == "cnn",
                 "model_params": spec_file["model_params"]},
        "edge": edge,
    })
json.dump(items, open(os.path.join(IT, "tmp", "import_items.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("items:", [i["tag"] for i in items])
