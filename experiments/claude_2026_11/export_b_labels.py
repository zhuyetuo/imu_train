"""在 api 容器里跑：B 组任务（人工审过、不在训练池）当前轮的全部人工片段（所有类别，不含待定），带类别链（父 > 子）。只读。"""
import asyncio
import json
import sys

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402
from app.models.task import Task  # noqa: E402


async def main():
    tids = [t["task_id"] for t in json.load(open("/tmp/groupB.json", encoding="utf-8"))]
    async with SessionLocal() as db:
        labs = {i: (n, p) for i, n, p in (await db.execute(select(LabelDefinition.id, LabelDefinition.display_name, LabelDefinition.parent_id))).all()}

        def chain(i):
            out = []
            while i is not None and i in labs and len(out) < 5:
                out.append(labs[i][0])
                i = labs[i][1]
            return list(reversed(out))

        rows = (await db.execute(
            select(AnnotationRecord.task_id, AnnotationLabelItem.label_id, AnnotationLabelItem.start_time_ms, AnnotationLabelItem.end_time_ms)
            .join(AnnotationLabelItem, AnnotationLabelItem.annotation_record_id == AnnotationRecord.id)
            .join(Task, Task.id == AnnotationRecord.task_id)
            .where(AnnotationRecord.task_id.in_(tids), AnnotationRecord.round_no == Task.round_no,
                   AnnotationLabelItem.uncertain.is_(False)))).all()
    out = {}
    for tid, lid, a, b in rows:
        if a is None or b is None or b <= a:
            continue
        out.setdefault(str(tid), []).append({"chain": chain(lid), "start_ms": int(a), "end_ms": int(b)})
    json.dump(out, open("/tmp/groupB_labels.json", "w", encoding="utf-8"), ensure_ascii=False)
    n = sum(len(v) for v in out.values())
    print("有人工片段的任务", len(out), "片段", n)


asyncio.run(main())
