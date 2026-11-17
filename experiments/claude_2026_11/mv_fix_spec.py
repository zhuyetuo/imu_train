"""api 容器里跑：训练记录 #40~#42 的 dataset_spec 跟 label_service 任务对齐（公开数据集指向平台登记名），再按「刷新」拉一次"""
import asyncio
import json

from sqlalchemy import select

from app.api.v1.model_versions import _apply_status
from app.db.session import SessionLocal
from app.models.model_version import ModelVersion
from app.services import algo_client

PUB = [{"date": "pub_vehkaoja2022", "export_json": "data_train/pub_vehkaoja2022/merged_tmp.json", "source_hz": 100.0, "exclude_classes": []},
       {"date": "pub_marcato2023", "export_json": "data_train/pub_marcato2023/merged_tmp.json", "source_hz": 100.0, "exclude_classes": []}]


async def main():
    async with SessionLocal() as db:
        for jid in (36, 37, 38):
            r = (await db.execute(select(ModelVersion).where(ModelVersion.algo_job_id == jid))).scalars().first()
            spec = json.loads(r.dataset_spec)
            spec["extra_datasets"] = PUB
            r.dataset_spec = json.dumps(spec, ensure_ascii=False)
            _apply_status(r, await algo_client.poll_train(jid))
        await db.commit()
        rows = (await db.execute(select(ModelVersion).where(ModelVersion.algo_job_id.in_([32, 33, 34, 35, 36, 37, 38])))).scalars().all()
        for r in sorted(rows, key=lambda x: x.id):
            print("记录", r.id, "任务", r.algo_job_id, r.model_version, "启用" if r.listed else "停用")

asyncio.run(main())
