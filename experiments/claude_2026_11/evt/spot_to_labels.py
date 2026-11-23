"""在 api 容器里跑：把「未标负例新模型·未标时间抽查」里人确认是抓挠的片段写进任务标注。
照平台「确认」的写法（candidates.py decide 非专项分支）：同名标签真正重叠就撑开那条，否则新加一条
human_added、from_candidate_id 指回抽查片段、带上起止调整记录、created_by 记判的人。
跳过：轮次对不上的、任务正在被人认领在标的（工作台存草稿会把不在它列表里的条目删掉）。
默认试运行；加 --apply 才写库。撤回：DELETE FROM annotation_label_items WHERE from_candidate_id IN (清单里的候选 id) AND created_at >= 写入时间"""
import asyncio
import json
import sys

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord, LabelItemSource, RecordSourceType  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402

MODEL = "未标负例新模型·未标时间抽查"
DRY = "--apply" not in sys.argv


async def main():
    out = []
    async with SessionLocal() as db:
        lab = dict((await db.execute(select(LabelDefinition.id, LabelDefinition.display_name))).all())
        cs = (await db.execute(select(AiCandidate).where(
            AiCandidate.model == MODEL, AiCandidate.status == CandidateStatus.confirmed))).scalars().all()
        for c in cs:
            name = lab.get(c.decided_label_id) or ""
            if not name.startswith("抓挠"):
                continue
            t = await db.get(Task, c.task_id)
            row = {"cand": c.id, "task": t.id, "s": c.start_time_ms, "e": c.end_time_ms, "label": name,
                   "task_status": str(t.status).split(".")[-1]}
            if c.round_no != t.round_no:
                row["do"] = "跳过·轮次变了"
                out.append(row)
                continue
            if t.status == TaskStatus.IN_PROGRESS and t.locked_by is not None:
                row["do"] = "跳过·有人在标"
                out.append(row)
                continue
            already = (await db.execute(select(AnnotationLabelItem.id).where(AnnotationLabelItem.from_candidate_id == c.id))).scalar()
            if already:
                row["do"] = "跳过·已经写过"
                out.append(row)
                continue
            record = (await db.execute(select(AnnotationRecord).where(
                AnnotationRecord.task_id == t.id, AnnotationRecord.round_no == t.round_no))).scalar_one_or_none()
            if record is None:
                record = AnnotationRecord(task_id=t.id, round_no=t.round_no, source_type=RecordSourceType.ai_revised)
                db.add(record)
                await db.flush()
            same = (await db.execute(select(LabelDefinition.id).where(
                LabelDefinition.project_id == t.project_id, LabelDefinition.display_name == name))).scalars().all()
            same = set(same) | {c.decided_label_id}
            dup = (await db.execute(select(AnnotationLabelItem).where(
                AnnotationLabelItem.annotation_record_id == record.id, AnnotationLabelItem.label_id.in_(same),
                AnnotationLabelItem.start_time_ms < c.end_time_ms, AnnotationLabelItem.end_time_ms > c.start_time_ms))).scalars().first()
            if dup is not None:
                row["do"] = f"撑开已有一条 {dup.start_time_ms/1000:.1f}-{dup.end_time_ms/1000:.1f}"
                dup.start_time_ms = min(dup.start_time_ms, c.start_time_ms)
                dup.end_time_ms = max(dup.end_time_ms, c.end_time_ms)
                if dup.from_candidate_id is None:
                    dup.from_candidate_id = c.id
                dup.ai_confirmed = True
            else:
                row["do"] = "新加"
                db.add(AnnotationLabelItem(
                    annotation_record_id=record.id, label_id=c.decided_label_id,
                    start_time_ms=c.start_time_ms, end_time_ms=c.end_time_ms,
                    source_type=LabelItemSource.human_added, is_modified=False,
                    ai_confidence=c.confidence, ai_confirmed=False, from_candidate_id=c.id,
                    boundary_edits=c.boundary_edits, created_by=c.decided_by))
            out.append(row)
        if DRY:
            await db.rollback()
        else:
            await db.commit()
    for r in out:
        print(f"任务{r['task']}({r['task_status']}) {r['s']/1000:.1f}-{r['e']/1000:.1f} {r['label']} → {r['do']}")
    from collections import Counter
    print("合计", len(out), dict(Counter(r["do"].split(" ")[0] for r in out)),
          "；所在任务已通过", sum(r["task_status"] == "APPROVED" for r in out))
    json.dump(out, open("/tmp/spot_to_labels.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("（试运行，没写库）" if DRY else "已写进任务标注")


asyncio.run(main())
