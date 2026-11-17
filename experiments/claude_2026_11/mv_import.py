"""给导入的 label_service 任务建平台的训练记录（model_versions），跟 POST /train 插的那行一样，
再按「刷新」的逻辑从 label_service 拉状态和指标写回；listed=True，项目的版本下拉里直接能选。
在 api 容器里跑：python mv_import.py items_done.json"""
import asyncio
import json
import sys

from sqlalchemy import select

from app.api.v1.model_versions import _apply_status
from app.db.session import SessionLocal
from app.models.model_version import ModelTrainStatus, ModelVersion
from app.models.user import User
from app.services import algo_client


async def main(path):
    items = json.load(open(path, encoding="utf-8"))
    async with SessionLocal() as db:
        user = (await db.execute(select(User).where(User.username == "toky"))).scalars().first()
        for it in items:
            old = (await db.execute(select(ModelVersion).where(ModelVersion.algo_job_id == it["job_id"]))).scalars().first()
            row = old or ModelVersion(algo_job_id=it["job_id"], status=ModelTrainStatus.queued, model_type=it["model_type"],
                                      dataset_spec=json.dumps(it["spec"], ensure_ascii=False), created_by=user.id)
            if not old:
                db.add(row)
            _apply_status(row, await algo_client.poll_train(it["job_id"]))
            row.listed = True
            await db.commit()
            await db.refresh(row)
            print("记录", row.id, "任务", row.algo_job_id, row.status, row.model_type, row.model_version, row.model_path)

asyncio.run(main(sys.argv[1]))
