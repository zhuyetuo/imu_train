"""加了公开数据集（Vehkaoja 2022 + Marcato 2023）的三个模型：拷进 ~/imu_train，写 import_items_pub.json。
跟 mk_items.py 一样，只是多了 extra_datasets（平台上「一起训练的数据集」那一栏），平台上点重训也能复现"""
import json
import os
import shutil

WT = os.path.expanduser("~/claude_exp/imu_train_wt")
IT = os.path.expanduser("~/imu_train")
LOGS = os.path.expanduser("~/claude_exp/export_w8")
STAGE = os.path.join(IT, "tmp", "import_logs")
os.makedirs(STAGE, exist_ok=True)
EXTRA = [{"date": "vehkaoja2022", "export_json": "public_datasets/vehkaoja2022/export.json", "source_hz": 100.0, "exclude_classes": []},
         {"date": "marcato2023", "export_json": "public_datasets/marcato2023/export.json", "source_hz": 100.0, "exclude_classes": []}]
MODELS = [("cnn_w8_3ax_pub", "cnn", 3, 8, "cnn_w8_3ax_pub"),
          ("cnn_w8_6ax_pub", "cnn", 6, 8, "cnn_w8_6ax_pub"),
          ("xgb_srv_6ax_pub", "xgb", 6, 10, None)]
items = []
for tag, kind, ax, win, edge_tag in MODELS:
    run = f"pool_20261116_all__{tag}"
    for top in ("results", "data"):
        dst = os.path.join(IT, top, f"processed_{run}")
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(WT, top, f"processed_{run}"), dst, symlinks=True)
    spec_file = json.load(open(os.path.join(WT, "tmp", f"train_v2_{tag}.json"), encoding="utf-8"))
    log = os.path.join(STAGE, f"{tag}.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"▶ 宿主机 GPU 上训练（同一份 src/ml/train_v2.py，spec 见 experiments/claude_2026_11/specs/train_v2_{tag}.json）\n"
                "  数据：数据池全部 + 公开数据集 Vehkaoja 2022、Marcato 2023（颈部传感器，转换脚本 experiments/claude_2026_11/convert_public.py）\n\n")
        f.write(open(os.path.join(LOGS, f"train_{tag}.log"), encoding="utf-8").read())
    edge = None
    if edge_tag:
        exp = open(os.path.join(LOGS, f"export_{ax}ax_pub.log"), encoding="utf-8").read()
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n▶ 导出到端侧\n" + exp)
        line = next(l for l in reversed(exp.splitlines()) if l.startswith("EXPORT_RESULT "))
        edge = json.loads(line[len("EXPORT_RESULT "):])
        edge["reloaded"], edge["reload_error"] = True, None
    sub = "cnn/dl_cnn_best.pt" if kind == "cnn" else "xgb/ml_xgb.pkl"
    items.append({
        "tag": tag, "run": run, "model_type": kind, "secs": 1500 if kind == "cnn" else 968,
        "model_path": f"/app/results/processed_{run}/20hz_v2/{sub}",
        "log": f"/app/tmp/import_logs/{tag}.log",
        "command": f"python src/ml/train_v2.py --spec experiments/claude_2026_11/specs/train_v2_{tag}.json",
        "spec": {"axes": ax, "window_s": float(win), "stride_s": 1.0, "edge_size": kind == "cnn",
                 "model_params": spec_file["model_params"], "extra_datasets": EXTRA},
        "edge": edge,
    })
json.dump(items, open(os.path.join(IT, "tmp", "import_items_pub.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("items:", [i["tag"] for i in items])
