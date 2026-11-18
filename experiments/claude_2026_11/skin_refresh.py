"""在 api 容器里跑：皮肤联动全量重算（跟「项目联动 → 重新拉取」按钮同一个函数、同一组默认参数：所有项目、不含草稿、ai_min_conf=0），
前后各存一份 (日期, IMU, 来源) 的 C 值 / 档 / 抓挠次数，好对比变了什么。"""
import asyncio
import json
import sys
import time
from datetime import date

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.skin_daily import SkinDailyStat  # noqa: E402
from app.services.skin_link_service import collect_link_stats  # noqa: E402

D_FROM, D_TO = date(2026, 8, 19), date(2026, 11, 17)


async def snap(path):
    async with SessionLocal() as db:
        rows = (await db.execute(select(SkinDailyStat))).scalars().all()
        out = {}
        for r in rows:
            s = json.loads(r.stats or "{}")
            out[f"{r.stat_date}|{r.imu}|{r.source}"] = {"c": r.c_value, "tier": r.c_tier, "n": s.get("event_count"),
                                                        "min": s.get("total_duration_min"), "updated": str(r.updated_at)}
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    return out


async def main():
    before = await snap("/tmp/skin_before.json")
    print("重算前", len(before), "行", flush=True)
    t0 = time.time()
    async with SessionLocal() as db:
        r = await collect_link_stats(db, D_FROM, D_TO, None, ai_min_conf=0.0, include_drafts=False)
    print(f"重算完 {time.time() - t0:.0f}s，行 {len(r.get('rows') or [])}，警告 {len(r.get('warnings') or [])}", flush=True)
    for w in (r.get("warnings") or [])[:8]:
        print("  ⚠", str(w)[:200])
    after = await snap("/tmp/skin_after.json")
    print("重算后", len(after), "行")


asyncio.run(main())
