"""在 api 容器里跑：导出「人工审过抓挠」的任务（跟皮肤联动人工版同一个口径：当前轮里有确认 / 修改 / 人工新增 / 待定的抓挠片段，
或任务已提交 / 已通过），每个任务带：样本 CSV 相对路径、人工确认的抓挠片段（不含待定）、平台老 AI 预标注 JSON 的路径。只读。"""
import asyncio
import json
import sys

from sqlalchemy import or_, select

sys.path.insert(0, "/app")
from app.db.session import SessionLocal  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord, LabelItemSource  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402
from app.services.ai_prelabel_service import ai_label_relpath  # noqa: E402


async def main():
    async with SessionLocal() as db:
        from app.services import label_tree
        base = (await db.execute(select(LabelDefinition.id).where(
            (LabelDefinition.display_name == "抓挠") | (LabelDefinition.code == "抓挠")))).scalars().all()
        scratch_ids = list(await label_tree.expand_ids(db, set(base)))
        q = (select(AnnotationRecord.task_id).join(AnnotationLabelItem, AnnotationLabelItem.annotation_record_id == AnnotationRecord.id)
             .join(Task, Task.id == AnnotationRecord.task_id)
             .where(AnnotationRecord.round_no == Task.round_no, AnnotationLabelItem.label_id.in_(scratch_ids),
                    or_(AnnotationLabelItem.ai_confirmed.is_(True), AnnotationLabelItem.is_modified.is_(True),
                        AnnotationLabelItem.uncertain.is_(True), AnnotationLabelItem.source_type == LabelItemSource.human_added)))
        tids = set((await db.execute(q)).scalars())
        tids |= set((await db.execute(select(Task.id).where(Task.status.in_([TaskStatus.SUBMITTED, TaskStatus.APPROVED])))).scalars())
        rows = (await db.execute(select(Task.id, Task.round_no, Sample.id, Sample.sample_code, Sample.session_date, Sample.imu_csv_path)
                                 .join(Sample, Sample.id == Task.sample_id).where(Task.id.in_(tids)))).all()
        out = []
        for tid, rnd, sid, code, d, csv in rows:
            if not csv or d is None:
                continue
            segs = (await db.execute(
                select(AnnotationLabelItem.start_time_ms, AnnotationLabelItem.end_time_ms)
                .join(AnnotationRecord, AnnotationLabelItem.annotation_record_id == AnnotationRecord.id)
                .where(AnnotationRecord.task_id == tid, AnnotationRecord.round_no == rnd,
                       AnnotationLabelItem.label_id.in_(scratch_ids), AnnotationLabelItem.uncertain.is_(False)))).all()
            out.append({"task_id": tid, "sample_id": sid, "sample_code": code, "date": d.isoformat(), "csv": csv,
                        "ai_json": ai_label_relpath(csv), "human": sorted([[int(a), int(b)] for a, b in segs if a is not None and b is not None])})
        json.dump(out, open("/tmp/reviewed_tasks.json", "w", encoding="utf-8"), ensure_ascii=False)
        print("人工审过的任务", len(out), "有人工抓挠的", sum(1 for o in out if o["human"]), "人工抓挠片段", sum(len(o["human"]) for o in out),
              "日期", min(o["date"] for o in out), "~", max(o["date"] for o in out))


asyncio.run(main())
