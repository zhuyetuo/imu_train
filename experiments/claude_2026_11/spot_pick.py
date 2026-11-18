"""抽查新模型多报的片段：从 groupB_res.jsonl 里分层抽样，写 spot_items.json（片段 + 样本号），视频路径另查。"""
import json
import random

random.seed(7)
rs = [json.loads(l) for l in open("groupB_res.jsonl", encoding="utf-8")]
tasks = {t["task_id"]: t for t in json.load(open("groupB.json", encoding="utf-8"))}
ov = lambda a, b: min(a[1], b[1]) - max(a[0], b[0]) > 0  # noqa: E731
items = []
by_month = {}
for r in rs:
    if not r.get("ok") or "new" not in r:
        continue
    t = tasks[r["task_id"]]
    if not r["human"]:
        for s in r["new"]:
            by_month.setdefault(r["date"][:7], []).append(("X", r, t, s))
    else:
        for s in r["new"]:
            if not any(ov(s, h) for h in r["human"]):
                by_month.setdefault("Y", []).append(("Y", r, t, s))
        for h in r["human"]:
            by_month.setdefault("C", []).append(("C", r, t, h))
for k, n in (("2026-07", 6), ("2026-08", 6), ("2026-09", 6), ("2026-10", 6), ("2026-11", 4), ("Y", 6), ("C", 5)):
    pool = by_month.get(k, [])
    for cat, r, t, s in random.sample(pool, min(n, len(pool))):
        items.append({"cat": cat, "month": r["date"][:7], "task_id": r["task_id"], "sample_id": t["sample_id"], "csv": t["csv"],
                      "start": round(s[0], 2), "end": round(s[1], 2)})
print({k: len(v) for k, v in by_month.items()}, "抽了", len(items))
json.dump(items, open("spot_items.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
