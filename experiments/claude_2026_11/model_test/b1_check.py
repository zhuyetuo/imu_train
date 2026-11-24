import asyncio
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.model_test import ModelTestSet, ModelTestItem
from app.api.v1 import model_tests as api
from app.services import model_test_service as mts
async def m():
    async with SessionLocal() as db:
        s = (await db.execute(select(ModelTestSet).where(ModelTestSet.name == "测试集 v1"))).scalar_one()
        d = (await api.get_items(s.id, db))["data"]
        c = d["coverage"]
        print("测试集", d["set"]["id"], d["set"]["n_items"], "段", d["set"]["n_samples"], "样本", "狗", c["n_dogs"], c["kinds"], c["sites"])
        print("提醒", c["warnings"][:3])
        excl, tasks = await mts.test_exclusion(db)
        print("训练要避开：样本", len(excl), "其中整个样本", sum(1 for v in excl.values() if v is None), "测试任务", len(tasks))
        r = (await api.seen(s.id, "56,61,62", db))["data"]
        print("见没见过", [(m["model_id"], m["n_seen"], m["n_unseen"]) for m in r["models"]])
asyncio.run(m())
