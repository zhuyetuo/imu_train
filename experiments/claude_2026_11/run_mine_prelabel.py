"""项目 325 里有人工标注、还在标注中的任务，用负例挖掘模型（srv:train39）走平台的批量预标注、只补候选。
只挑有人工痕迹的任务：这些走 candidates_only，草稿一条不动；纯 AI / 空白的任务不在名单里，不会写草稿。"""
import asyncio
import collections
import json

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.ai_candidate import AiCandidate
from app.models.annotation import AnnotationLabelItem, AnnotationRecord, LabelItemSource
from app.models.task import Task, TaskStatus
from app.services import ai_prelabel_service as aps

PID, MODE = 325, "srv:train39"


async def main():
    async with SessionLocal() as db:
        ts = (await db.execute(select(Task).where(Task.project_id == PID, Task.status.in_(
            [TaskStatus.PENDING_ASSIGN, TaskStatus.IN_PROGRESS])))).scalars().all()
        ids = [t.id for t in ts]
        items = (await db.execute(select(AnnotationLabelItem, AnnotationRecord.task_id).join(
            AnnotationRecord, AnnotationRecord.id == AnnotationLabelItem.annotation_record_id).where(
            AnnotationRecord.task_id.in_(ids)))).all()
        human = sorted({tid for it, tid in items if it.source_type == LabelItemSource.human_added or it.is_modified or it.ai_confirmed})
        before = collections.Counter((await db.execute(select(AiCandidate.reason).where(AiCandidate.task_id.in_(human)))).scalars().all())
    print("任务", human, flush=True)
    print("之前的候选", dict(before), flush=True)
    await aps._run(PID, human, False, MODE, True)
    async with SessionLocal() as db:
        rows = (await db.execute(select(AiCandidate).where(AiCandidate.task_id.in_(human)))).scalars().all()
        after = collections.Counter(r.reason for r in rows)
        neg = collections.Counter(r.label_name for r in rows if r.reason == "model_neg")
        per_task = collections.Counter(r.task_id for r in rows if r.reason == "model_neg")
    print("之后的候选", dict(after), flush=True)
    print("新的负例候选", dict(neg), "分布在", len(per_task), "个任务", flush=True)
    print("DONE", json.dumps({"tasks": human, "neg": dict(neg)}, ensure_ascii=False), flush=True)

asyncio.run(main())
