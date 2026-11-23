"""api 容器里跑：11-18 项目随机挑几个文件，用 train60 分别跑稳定版v2 和抓挠严格版，对比抓挠片段。不写库。"""
import asyncio
import random
from datetime import datetime

import httpx
from sqlalchemy import text

from app.core.config import settings
from app.db.session import SessionLocal

URL = settings.algo_service_url.rstrip("/") + "/api/v1/label/infer"


def spans(res):
    out = []
    for lab, segs in (res.get("segments") or {}).items():
        if lab.startswith("抓挠"):
            for s in segs:
                a = datetime.fromisoformat(s["start_ts"].replace("Z", ""))
                b = datetime.fromisoformat(s["end_ts"].replace("Z", ""))
                out.append((a, b, s.get("conf_mean")))
    return sorted(out)


async def main():
    async with SessionLocal() as db:
        rows = (await db.execute(text(
            "select distinct s.id, s.imu_csv_path, s.sample_hz from samples s join tasks t on t.sample_id=s.id "
            "join projects p on p.id=t.project_id where p.name='2026-11-18' and s.imu_csv_path is not null"))).all()
    random.seed(1124)
    pick = random.sample(rows, min(30, len(rows)))
    tot = {"v2": 0, "strict": 0, "v2_s": 0.0, "strict_s": 0.0, "kept_same": 0, "cands_v2": 0, "cands_strict": 0}
    async with httpx.AsyncClient(timeout=600) as c:
        for sid, path, hz in pick:
            res = {}
            for mode in ("viterbi", "viterbi_strict"):
                r = await c.post(URL, json={"path": path, "sample_id": sid, "mode": mode, "device_hz": hz, "model": "train60"})
                r.raise_for_status()
                res[mode] = r.json()
            a, b = spans(res["viterbi"]), spans(res["viterbi_strict"])
            # 严格版的每一段都应该落在 v2 的某段里（只删不加）
            inside = sum(any(x[0] >= y[0] and x[1] <= y[1] for y in a) for x in b)
            tot["v2"] += len(a); tot["strict"] += len(b); tot["kept_same"] += inside
            tot["v2_s"] += sum((y[1] - y[0]).total_seconds() for y in a)
            tot["strict_s"] += sum((y[1] - y[0]).total_seconds() for y in b)
            tot["cands_v2"] += len(res["viterbi"].get("candidates") or [])
            tot["cands_strict"] += len(res["viterbi_strict"].get("candidates") or [])
            other_v2 = {k: len(v) for k, v in (res["viterbi"].get("segments") or {}).items() if not k.startswith("抓挠") and v}
            other_st = {k: len(v) for k, v in (res["viterbi_strict"].get("segments") or {}).items() if not k.startswith("抓挠") and v}
            print(f"样本{sid}: 抓挠 v2 {len(a)} → 严格 {len(b)}（落在 v2 段里 {inside}）；其他类别一样：{other_v2 == other_st}；"
                  f"疑似候选 {len(res['viterbi'].get('candidates') or [])} → {len(res['viterbi_strict'].get('candidates') or [])}", flush=True)
    print("合计", tot)


asyncio.run(main())
