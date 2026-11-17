"""api 容器里跑：导出平台上所有「饮水」片段（含人确认 / 审核通过标记）所在任务的全部片段，给相似度实验用。
不写库。输出 /tmp/drink_items.json"""
import asyncio
import json

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.annotation import AnnotationLabelItem, AnnotationRecord, LabelItemSource
from app.models.label import LabelDefinition
from app.models.sample import Sample
from app.models.task import Task

TARGET = "饮水"


async def main():
    async with SessionLocal() as db:
        lab_ids = set((await db.execute(select(LabelDefinition.id).where(
            LabelDefinition.display_name == TARGET))).scalars().all())
        names = dict((await db.execute(select(LabelDefinition.id, LabelDefinition.display_name))).all())
        # 有饮水片段的任务（当前轮）
        task_ids = set((await db.execute(
            select(AnnotationRecord.task_id)
            .join(Task, (Task.id == AnnotationRecord.task_id) & (Task.round_no == AnnotationRecord.round_no))
            .join(AnnotationLabelItem, AnnotationLabelItem.annotation_record_id == AnnotationRecord.id)
            .where(AnnotationLabelItem.label_id.in_(lab_ids)))).scalars().all())
        out = []
        for tid in sorted(task_ids):
            t = await db.get(Task, tid)
            s = await db.get(Sample, t.sample_id)
            rec = (await db.execute(select(AnnotationRecord).where(
                AnnotationRecord.task_id == tid, AnnotationRecord.round_no == t.round_no))).scalars().first()
            items = (await db.execute(select(AnnotationLabelItem).where(
                AnnotationLabelItem.annotation_record_id == rec.id))).scalars().all()
            out.append({
                "task_id": tid, "project_id": t.project_id, "sample_code": s.sample_code,
                "csv": s.imu_csv_path, "hz": s.sample_hz,
                "items": [{
                    "label": names.get(i.label_id, "?"), "start_ms": i.start_time_ms, "end_ms": i.end_time_ms,
                    # 人看过的：人加的 / 确认过的 / 改过的 / 片段审核通过
                    "human": bool(i.source_type == LabelItemSource.human_added or i.ai_confirmed or i.is_modified
                                  or i.seg_status == "approved"),
                    "approved": i.seg_status == "approved",
                } for i in items],
            })
    json.dump(out, open("/tmp/drink_items.json", "w", encoding="utf-8"), ensure_ascii=False)
    n_d = sum(1 for t in out for i in t["items"] if i["label"] == TARGET)
    n_dh = sum(1 for t in out for i in t["items"] if i["label"] == TARGET and i["human"])
    print("任务", len(out), "饮水片段", n_d, "其中人看过", n_dh, "狗（imu）", len({t["sample_code"].split("_")[-1] for t in out}))

asyncio.run(main())
