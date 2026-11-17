"""宿主机上跑：把手动导出的端侧模型改名成 train<任务号>，挂回训练记录；训练记录里的公开数据集指向平台登记的那两份。
平台下拉靠端侧模型 meta.json 里 train.job_id 对回训练记录（启用 / 停用也靠它），名字不是 train<号> 就对不上。"""
import json
import os
import shutil
import sys

# 两段分开跑：algo_tinyml 那边在宿主机（文件归 toky），任务文件在 label-service 容器里（文件归 root）
PART = sys.argv[1] if len(sys.argv) > 1 else "edge"
TM = os.path.expanduser("~/algo_tinyml")
JOBS = "/app/label_service/jobs"
RENAME = {"cnn_w8_3ax": 32, "cnn_w8_6ax": 33, "cnn_w8_3ax_pub": 36, "cnn_w8_6ax_pub": 37}
PUB = [{"date": "pub_vehkaoja2022", "export_json": "data_train/pub_vehkaoja2022/merged_tmp.json", "source_hz": 100.0, "exclude_classes": []},
       {"date": "pub_marcato2023", "export_json": "data_train/pub_marcato2023/merged_tmp.json", "source_hz": 100.0, "exclude_classes": []}]

if PART == "edge":
    lj_path = os.path.join(TM, "edge_models.local.json")
    lj = json.load(open(lj_path, encoding="utf-8"))
    for old, jid in RENAME.items():
        new = f"train{jid}"
        src = os.path.join(TM, "core", "models", f"generated_{old}")
        dst = os.path.join(TM, "core", "models", f"generated_{new}")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
        mp = os.path.join(dst, "meta.json")
        meta = json.load(open(mp, encoding="utf-8"))
        meta.setdefault("train", {})
        meta["train"].update(job_id=jid, tag=new)
        json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        lj["models"] = [m for m in lj["models"] if m.get("tag") not in (old, new)]
        lj["models"].append({"tag": new, "kind": "cnn", "gen": f"core/models/generated_{new}",
                             "meta": f"core/models/generated_{new}/meta.json"})
        print(old, "->", new)
    json.dump(lj, open(lj_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    sys.exit(0)

for old, jid in RENAME.items():
    new = f"train{jid}"
    jp = os.path.join(JOBS, f"{jid}.json")
    job = json.load(open(jp, encoding="utf-8"))
    if job.get("edge"):
        job["edge"].update(tag=new, spec=f"edge:{new}", gen=f"core/models/generated_{new}",
                           meta=f"core/models/generated_{new}/meta.json")
    json.dump(job, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("job", jid, "edge ->", new)

for jid in (36, 37, 38):
    jp = os.path.join(JOBS, f"{jid}.json")
    job = json.load(open(jp, encoding="utf-8"))
    job["dataset_spec"]["extra_datasets"] = PUB
    json.dump(job, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("job", jid, "extra_datasets ->", [p["date"] for p in PUB])
