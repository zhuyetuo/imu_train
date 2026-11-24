import asyncio
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.task import Task, TaskStatus
from app.models.user import User, UserRole
from app.services.task_service import heartbeat, TaskConflictError
async def m():
    async with SessionLocal() as db:
        admin = (await db.execute(select(User).where(User.role == UserRole.super_admin))).scalars().first()
        tid = (await db.execute(select(Task.id).where(Task.status == TaskStatus.PENDING_ASSIGN, Task.locked_by.is_(None)).limit(1))).scalar()
        await heartbeat(db, tid, admin)
        print("管理员心跳（任务", tid, "）：成功")
        ann = (await db.execute(select(User).where(User.role == UserRole.annotator))).scalars().first()
        try:
            await heartbeat(db, tid, ann)
            print("标注员心跳：竟然成功了（不对）")
        except TaskConflictError as e:
            print("标注员心跳：照旧报错 →", e)
asyncio.run(m())
