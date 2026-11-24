"""从测试集拿掉「窗口里什么都没有」的片段（没有标注、也没有待判的疑似片段）。只删测试集里的条目，不删任务。
加 --apply 才动手。"""
import asyncio
import sys
from collections import Counter

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.model_test import ModelTestItem, ModelTestSet  # noqa: E402
from app.services import model_test_service as mts  # noqa: E402

APPLY = "--apply" in sys.argv


async def main():
    async with SessionLocal() as db:
        n, tasks = Counter(), set()
        for s in (await db.execute(select(ModelTestSet).order_by(ModelTestSet.id))).scalars().all():
            if s.frozen_at is not None:
                print(f"{s.name} 已冻结，跳过")
                continue
            its = (await db.execute(select(ModelTestItem).where(ModelTestItem.set_id == s.id))).scalars().all()
            byid = {i.id: i for i in its}
            for r in await mts.describe_items(db, list(its)):
                if not r["labels"] and not any((v.get("pending") or 0) for v in r["cands"].values()) and not r["n_uncertain"]:
                    n[s.name] += 1
                    tasks.add(r["task_id"])
                    if APPLY:
                        await db.delete(byid[r["id"]])
        if APPLY:
            await db.commit()
        print("拿掉", dict(n), "合计", sum(n.values()), "条；涉及任务", len(tasks), "个：", sorted(tasks)[:80])
        print("已写库" if APPLY else "（试运行）")


asyncio.run(main())
