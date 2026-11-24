"""真库 + 异步会话：管理员对没认领的待分配任务发心跳应成功；标注员应照旧报错。不改数据（心跳失败就回滚，管理员这条不写）。"""
import asyncio
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.task import Task, TaskStatus
from app.models.user import User, UserRole
from app.services.task_service import heartbeat, TaskConflictError


async def m():
    async with SessionLocal() as db:
        admin = await db.get(User, 1)
        print("1 号账号角色", admin.role)
        tid = (await db.execute(select(Task.id).where(Task.status == TaskStatus.PENDING_ASSIGN, Task.locked_by.is_(None)).limit(1))).scalar()
        await heartbeat(db, tid, admin)
        print("管理员心跳（任务", tid, "）：成功")
    async with SessionLocal() as db:
        ann = (await db.execute(select(User).where(User.role == UserRole.annotator))).scalars().first()
        if ann is None:
            print("没有标注员账号，跳过")
            return
        try:
            await heartbeat(db, tid, ann)
            print("标注员心跳：竟然成功了（不对）")
        except TaskConflictError as e:
            print("标注员心跳：照旧报错 →", e)
asyncio.run(m())
