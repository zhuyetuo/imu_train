"""不写库：拿几个真实任务拼成测试片段，看描述 / 配额表对不对。"""
import asyncio
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.model_test import ModelTestItem
from app.models.task import Task, TaskStatus
from app.services import model_test_service as mts


async def m():
    async with SessionLocal() as db:
        ts = (await db.execute(select(Task).where(Task.status == TaskStatus.APPROVED).order_by(Task.id.desc()).limit(6))).scalars().all()
        items = [ModelTestItem(id=i + 1, set_id=0, task_id=t.id, sample_id=t.sample_id,
                               start_ms=(0 if i % 2 else None), end_ms=(60000 if i % 2 else None), added_by=1)
                 for i, t in enumerate(ts)]
        rows = await mts.describe_items(db, items)
        for r in rows:
            print(r["task_id"], r["date"], r["site"], r["dog_name"], r["size"], r["dog_source"], r["kind"], r["dur_sec"], r["labels"])
        cov = mts.coverage(rows)
        print("配额", {k: {t: v["n"] for t, v in m.items()} for k, m in cov["matrix"].items()}, cov["kinds"], cov["warnings"])
asyncio.run(m())
