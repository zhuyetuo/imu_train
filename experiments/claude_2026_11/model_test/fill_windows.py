"""项目 #342「模型测试 v1」、#343「模型测试 场地」里还是空的短任务：把同一个样本原任务里、窗口内的片段
（以前人工标过的 / AI 预标的）拷进来，标注员只核对、调整，不用从零标。
· 每个窗口挑一个来源任务：已提交 / 通过的优先，其次窗口里片段覆盖最多的
· 片段裁到窗口里；类别按名字对到本项目的标签；拷成「AI 预标」来源（要人确认）
· 已经写了疑似片段（提示）的那一类，跟疑似片段重叠的不拷，免得同一段出现两次
· 已经有片段的任务不动
默认试运行；加 --apply 才写库。"""
import asyncio
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord, LabelItemSource  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402
from app.services.task_service import _get_or_create_record  # noqa: E402

PROJECTS = [342, 343]
APPLY = "--apply" in sys.argv


async def main():
    async with SessionLocal() as db:
        lrows = (await db.execute(select(LabelDefinition.id, LabelDefinition.display_name, LabelDefinition.parent_id,
                                         LabelDefinition.project_id))).all()
        lname = {i: n for i, n, _, _ in lrows}
        parent = {i: p for i, _, p, _ in lrows}
        by_proj_name = {(pid, n): i for i, n, _, pid in lrows}

        def top(l):
            while parent.get(l) is not None:
                l = parent[l]
            return lname.get(l)

        tasks = (await db.execute(select(Task).where(Task.project_id.in_(PROJECTS)))).scalars().all()
        stat, n_items, src_kind, skipped_lab = Counter(), Counter(), Counter(), Counter()
        for t in tasks:
            ws, we = t.segment_start_ms, t.segment_end_ms
            has = (await db.execute(select(AnnotationLabelItem.id)
                                    .join(AnnotationRecord, AnnotationLabelItem.annotation_record_id == AnnotationRecord.id)
                                    .where(AnnotationRecord.task_id == t.id, AnnotationRecord.round_no == t.round_no).limit(1))).first()
            if has:
                stat["已有片段，不动"] += 1
                continue
            rows = (await db.execute(
                select(Task.id, Task.status, AnnotationLabelItem.label_id, AnnotationLabelItem.start_time_ms,
                       AnnotationLabelItem.end_time_ms, AnnotationLabelItem.uncertain, AnnotationLabelItem.ai_confidence,
                       AnnotationLabelItem.source_type, AnnotationLabelItem.ai_confirmed, AnnotationLabelItem.is_modified)
                .join(AnnotationRecord, AnnotationRecord.task_id == Task.id)
                .join(AnnotationLabelItem, AnnotationLabelItem.annotation_record_id == AnnotationRecord.id)
                .where(Task.sample_id == t.sample_id, Task.segment_start_ms.is_(None), AnnotationRecord.round_no == Task.round_no,
                       AnnotationLabelItem.start_time_ms < we, AnnotationLabelItem.end_time_ms > ws))).all()
            if not rows:
                stat["原任务窗口里也没有片段"] += 1
                continue
            by_task = defaultdict(list)
            for r in rows:
                by_task[(r[0], r[1])].append(r)

            def cover(rs):
                return sum(min(r[4], we) - max(r[3], ws) for r in rs)
            (src_tid, src_st), rs = max(by_task.items(), key=lambda kv: (
                kv[0][1] in (TaskStatus.SUBMITTED, TaskStatus.APPROVED), cover(kv[1])))
            reviewed = src_st in (TaskStatus.SUBMITTED, TaskStatus.APPROVED)
            src_kind["来源任务已提交/通过" if reviewed else "来源任务还没审（多是 AI 预标）"] += 1
            hints = (await db.execute(select(AiCandidate.label_name, AiCandidate.start_time_ms, AiCandidate.end_time_ms)
                                      .where(AiCandidate.task_id == t.id, AiCandidate.round_no == t.round_no,
                                             AiCandidate.status == CandidateStatus.pending))).all()
            new = []
            for _tid, _st, lid, a, b, unc, conf, src, cf, mod in rs:
                tp = top(lid)
                a2, b2 = max(int(a), ws), min(int(b), we)
                if b2 <= a2:
                    continue
                if any(h[0] == tp and h[1] < b2 and h[2] > a2 for h in hints):
                    stat["跟疑似片段重叠，不拷"] += 1
                    continue
                tgt = by_proj_name.get((t.project_id, lname.get(lid)))
                if tgt is None:
                    skipped_lab[lname.get(lid)] += 1
                    continue
                new.append((tgt, a2, b2, bool(unc), conf))
                n_items["人工" if (reviewed or src != LabelItemSource.ai_generated or cf or mod) else "AI"] += 1
            stat["拷了"] += 1 if new else 0
            if APPLY and new:
                rec = await _get_or_create_record(db, t)
                for tgt, a2, b2, unc, conf in new:
                    db.add(AnnotationLabelItem(annotation_record_id=rec.id, label_id=tgt, start_time_ms=a2, end_time_ms=b2,
                                               source_type=LabelItemSource.ai_generated, ai_confidence=conf, uncertain=unc))
        if APPLY:
            await db.commit()
        print("任务", len(tasks), dict(stat))
        print("来源", dict(src_kind), "；拷的片段", dict(n_items))
        if skipped_lab:
            print("本项目没有这个标签、没拷：", dict(skipped_lab.most_common(8)))
        print("已写库" if APPLY else "（试运行，没写库）")


asyncio.run(main())
