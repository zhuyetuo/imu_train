"""把 B 组（人工审过、不在训练池；只取 8~11 月原始数据，不要 16 Hz 老设备）做成 train_v2 能吃的导出 JSON：
  B_tr.json        训练那一半（按 天 × 项圈 分组，crc32 % 2 == 0）的全部人工片段
  B_tr_pseudo.json 同上 + 新模型多报、落在人工没标时间里的片段，当「活动」（抽查 14/14 都不是抓挠）
  B_ev.json        验证那一半的全部人工片段（只用来算命中 / 误报，不进训练）"""
import json
import os
import re
import zlib

OUT = os.path.expanduser("~/claude_exp/hardneg")
os.makedirs(OUT, exist_ok=True)
tasks = {t["task_id"]: t for t in json.load(open("groupB.json", encoding="utf-8"))}
labels = json.load(open("groupB_labels.json", encoding="utf-8"))
res = {json.loads(l)["task_id"]: json.loads(l) for l in open("groupB_res.jsonl", encoding="utf-8")}


def group(code):
    day = re.search(r"_(\d{8})_", code)
    imu = re.search(r"imu\d+", code, re.I)
    return f"{day.group(1) if day else '?'}_{imu.group(0).lower() if imu else code}"


def item(t, segs):
    return {"data": {"csv": t["csv"], "sample_code": t["sample_code"]},
            "annotations": [{"result": [{"value": {"start_ms": s, "end_ms": e, "timeserieslabels": ch}} for s, e, ch in segs]}]}


tr, trp, ev = [], [], []
n_pseudo = 0
for tid, t in tasks.items():
    if "16hz" in t["csv"] or t["date"] < "2026-08":
        continue
    hum = [(x["start_ms"], x["end_ms"], x["chain"]) for x in labels.get(str(tid), [])]
    if not hum:
        continue
    if zlib.crc32(group(t["sample_code"]).encode()) % 2:
        ev.append(item(t, hum))
        continue
    tr.append(item(t, hum))
    extra = []
    r = res.get(tid)
    if r and r.get("ok") and "new" in r:
        for a, b in r["new"]:
            s, e = int(a * 1000), int(b * 1000)
            if not any(min(e, he) - max(s, hs) > 0 for hs, he, _ in hum):
                extra.append((s, e, ["活动"]))
    n_pseudo += len(extra)
    trp.append(item(t, hum + extra))
for name, d in (("B_tr", tr), ("B_tr_pseudo", trp), ("B_ev", ev)):
    json.dump(d, open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"训练那一半 {len(tr)} 个任务，验证那一半 {len(ev)} 个任务；当负例的多报片段 {n_pseudo} 段")
