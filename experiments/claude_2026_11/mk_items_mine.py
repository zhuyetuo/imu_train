"""负例挖掘模型：拷进 ~/imu_train，写 import_items_mine.json（只有服务端版本，不导端侧）"""
import json
import os
import shutil

WT = os.path.expanduser("~/claude_exp/imu_train_wt")
IT = os.path.expanduser("~/imu_train")
LOGS = os.path.expanduser("~/claude_exp/export_w8")
STAGE = os.path.join(IT, "tmp", "import_logs")
os.makedirs(STAGE, exist_ok=True)
PUB = [{"date": "pub_vehkaoja2022", "export_json": "data_train/pub_vehkaoja2022/merged_tmp.json", "source_hz": 100.0, "exclude_classes": []},
       {"date": "pub_marcato2023", "export_json": "data_train/pub_marcato2023/merged_tmp.json", "source_hz": 100.0, "exclude_classes": []}]
tag = "cnn_mine_6ax"
run = f"pool_20261116_all__{tag}"
for top in ("results", "data"):
    dst = os.path.join(IT, top, f"processed_{run}")
    if not os.path.exists(dst):
        shutil.copytree(os.path.join(WT, top, f"processed_{run}"), dst, symlinks=True)
spec = json.load(open(os.path.join(WT, "tmp", f"train_v2_{tag}.json"), encoding="utf-8"))
log = os.path.join(STAGE, f"{tag}.log")
with open(log, "w", encoding="utf-8") as f:
    f.write("▶ 负例挖掘模型（服务端用，不上板）：进食 / 饮水 / 嗅闻 从「活动」里单独分出来。\n"
            "  项目里选它做预标注（只补候选），这几类的片段不进草稿，变成「疑似进食 / 饮水 / 嗅闻」候选给人确认。\n"
            f"  宿主机 GPU 上训练（同一份 src/ml/train_v2.py，spec 见 experiments/claude_2026_11/specs/train_v2_{tag}.json）\n\n")
    f.write(open(os.path.join(LOGS, f"train_{tag}.log"), encoding="utf-8").read())
items = [{
    "tag": tag, "run": run, "model_type": "cnn", "secs": 960,
    "model_path": f"/app/results/processed_{run}/20hz_v2/cnn/dl_cnn_best.pt",
    "log": f"/app/tmp/import_logs/{tag}.log",
    "command": f"python src/ml/train_v2.py --spec experiments/claude_2026_11/specs/train_v2_{tag}.json",
    "spec": {"axes": 6, "window_s": 8.0, "stride_s": 1.0, "edge_size": False,
             "model_params": spec["model_params"], "extra_datasets": PUB, "label_remap": spec["label_remap"]},
    "edge": None,
}]
json.dump(items, open(os.path.join(IT, "tmp", "import_items_mine.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("items:", [i["tag"] for i in items])
