import asyncio
from sqlalchemy import text
from app.db.session import SessionLocal
from app.models.model_version import ModelVersion
from app.api.v1 import model_tests as api
from app.services import model_test_service as mts


async def m():
    async with SessionLocal() as db:
        print("表", (await db.execute(text("select count(*) from model_test_sets"))).all(),
              (await db.execute(text("select count(*) from model_test_items"))).all())
        print("列表接口", (await api.list_sets(db))["data"])
        for mid in (56, 61, 62):
            mv = await db.get(ModelVersion, mid)
            names = mts.datasets_of_spec(mv.dataset_spec)
            got = {}
            for n in names:
                c = await asyncio.to_thread(mts.dataset_sample_codes, n)
                got[n] = len(c) if c is not None else None
            print(f"模型 #{mid} 训练数据集 → 样本数", got)
asyncio.run(m())
