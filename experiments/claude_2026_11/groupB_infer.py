"""B 组（人工审过、不在训练池里的任务）用 train37 + 精准版在 label_service 上推理，只记每个任务的抓挠片段，
跟人工确认的、平台老 AI 预标注的比。结果写 groupB_res.jsonl（可断点续跑），不碰平台文件。"""
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

NAS = "/home/toky/ai_data"
URL = "http://127.0.0.1:8383/api/v1/label/infer_batch"
OUT = "groupB_res.jsonl"


def P(s):
    return datetime.strptime(s[:23], "%Y-%m-%d %H:%M:%S.%f") if "." in s else datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def merge(sp):
    out = []
    for a, b in sorted(sp):
        if out and a < out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def scratch_spans(segments, t0):
    o = []
    for lab, ss in (segments or {}).items():
        if lab.startswith("抓挠"):
            o += [[(P(x["start_ts"]) - t0).total_seconds(), (P(x["end_ts"]) - t0).total_seconds()] for x in ss]
    return merge(o)


def main():
    tasks = json.load(open("groupB.json", encoding="utf-8"))
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["task_id"] for l in open(OUT, encoding="utf-8")}
    todo = [t for t in tasks if t["task_id"] not in done]
    print("待跑", len(todo), flush=True)
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        body = json.dumps({"items": [{"path": t["csv"], "sample_id": t["sample_id"]} for t in batch],
                           "mode": "viterbi_precise", "model": "train37"}).encode()
        req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
        try:
            res = json.load(urllib.request.urlopen(req, timeout=3600))
        except Exception as e:  # noqa: BLE001
            print("批次失败", i, e, flush=True)
            time.sleep(30)
            continue
        with open(OUT, "a", encoding="utf-8") as f:
            for t, r in zip(batch, res):
                rec = {"task_id": t["task_id"], "date": t["date"], "imu": t["sample_code"].split("_")[-1].upper(), "ok": r["ok"]}
                if r["ok"]:
                    w = r["result"].get("windows") or []
                    if w:
                        t0 = P(w[0]["ts"])
                        lo, hi = 0.0, (P(w[-1]["ts"]) - t0).total_seconds() + 8
                        rec["hours"] = (hi - lo) / 3600
                        rec["new"] = scratch_spans(r["result"].get("segments"), t0)
                        # 人工片段是相对 CSV 首行的毫秒；窗口第一个时间戳就是 CSV 首行附近，差别在毫秒级
                        rec["human"] = merge([[a / 1000, b / 1000] for a, b in t["human"]])
                        p = os.path.join(NAS, t["ai_json"])
                        if os.path.exists(p):
                            try:
                                aj = json.load(open(p, encoding="utf-8"))
                                rec["old"] = scratch_spans(aj.get("segments"), t0)
                            except Exception:  # noqa: BLE001
                                pass
                else:
                    rec["error"] = (r.get("error") or "")[:200]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{min(i + 40, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
