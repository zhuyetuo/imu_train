import asyncio
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.model_test import ModelTestSet
from app.api.v1 import model_tests as api
async def m():
    async with SessionLocal() as db:
        for s in (await db.execute(select(ModelTestSet).order_by(ModelTestSet.id))).scalars().all():
            d = (await api.get_items(s.id, db))["data"]
            r = (await api.seen(s.id, "56,61,62", db))["data"]
            print(s.id, s.name, d["set"]["n_items"], "段", "狗", d["coverage"]["n_dogs"], "见过/没见过", [(m["model_id"], m["n_seen"], m["n_unseen"]) for m in r["models"]])
asyncio.run(m())
