"""把转换好的公开数据集登记成平台上的训练数据集：NAS data_train/<名字>/merged_tmp.json + meta.json，
跟平台「导出数据集」写的是同一套文件，「模型训练」那里的数据集列表里就能选、能当「一起训练的数据集」。"""
import json
import os
from collections import Counter, defaultdict
from datetime import datetime

NAS = os.path.expanduser("~/ai_data")
SRC = {
    "pub_vehkaoja2022": dict(dir="public_datasets/vehkaoja2022", n_dogs=45, license="CC BY 4.0",
                             url="https://data.mendeley.com/datasets/vxhx934tbn/3",
                             cite="Vehkaoja et al. 2022, Data in Brief; Kumpulainen et al. 2021, Applied Animal Behaviour Science",
                             note="ActiGraph GT9X 颈部（项圈）传感器，加速度 + 陀螺仪，100 Hz。原数据还有背部传感器，这里只取颈部"),
    "pub_marcato2023": dict(dir="public_datasets/marcato2023", n_dogs=42, license="CC BY 4.0",
                            url="https://data.mendeley.com/datasets/mpph6bmn7g/1",
                            cite="Marcato et al. 2023, PLOS ONE 18(6): e0286311",
                            note="ActiGraph GT9X 颈部传感器，加速度 + 陀螺仪，100 Hz。原数据还有背 / 胸，这里只取颈部"),
}

for name, s in SRC.items():
    tasks = json.load(open(os.path.join(NAS, s["dir"], "export.json"), encoding="utf-8"))
    labels, secs = Counter(), defaultdict(float)
    tree = defaultdict(lambda: [0, 0.0, defaultdict(lambda: [0, 0.0])])
    n_seg, total = 0, 0.0
    for t in tasks:
        for r in t["annotations"][0]["result"]:
            v = r["value"]
            chain = v["timeserieslabels"]
            sec = (v["end_ms"] - v["start_ms"]) / 1000
            labels[chain[-1]] += 1
            secs[chain[-1]] += sec
            top = tree[chain[0]]
            top[0] += 1
            top[1] += sec
            if len(chain) > 1:
                top[2][chain[-1]][0] += 1
                top[2][chain[-1]][1] += sec
            n_seg += 1
            total += sec
    label_tree = [{"label": k, "n": v[0], "sec": round(v[1], 1), "depth": 1,
                   **({"children": [{"label": c, "n": cv[0], "sec": round(cv[1], 1), "depth": 2} for c, cv in v[2].items()]} if v[2] else {})}
                  for k, v in sorted(tree.items(), key=lambda x: -x[1][1])]
    out_dir = os.path.join(NAS, "data_train", name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "merged_tmp.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False)
    meta = {
        "name": name, "date_from": None, "date_to": None,
        "filter": {"date_from": None, "date_to": None, "project_ids": []}, "include_submitted": False,
        "scope": "public",
        "public": {"source": s["url"], "license": s["license"], "cite": s["cite"], "n_dogs": s["n_dogs"], "note": s["note"],
                   "converted_by": "imu_train experiments/claude_2026_11/convert_public.py"},
        "flatten": True, "tracks": {"behavior": n_seg}, "n_device_segments": 0,
        "n_tasks": len(tasks), "n_segments": n_seg, "total_hours": round(total / 3600, 2),
        "n_uncertain_excluded": 0, "missing_excluded_min": 0,
        "sample_rate_hz": {"100": len(tasks)}, "n_sample_rate_unknown": 0,
        "labels": dict(labels), "label_seconds": {k: round(v, 1) for k, v in secs.items()},
        "label_tree": label_tree, "label_tree_before": label_tree,
        "reduce": {"labels": [], "factor": 3.0, "targets": {}, "result": []},
        "warnings": [f"公开数据集（{s['license']}），不是平台标注导出；没有抓挠标注。用的时候请引用：{s['cite']}"],
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "export_json": f"data_train/{name}/merged_tmp.json",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(name, len(tasks), "段录制", n_seg, "个片段", round(total / 3600, 2), "小时", dict(labels))
