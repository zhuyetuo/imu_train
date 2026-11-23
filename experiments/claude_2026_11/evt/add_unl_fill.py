"""在 api 容器里跑：把新模型在未标时间报的抓挠（抽查 50 段里约 75% 是真抓挠）写成普通疑似片段，让标注员补标。
reason=low_conf（普通来源：确认后直接写进任务标注），model 用单独的名字，整批可删：
  DELETE FROM ai_candidates WHERE model = 'MODEL 常量' AND status = 'pending'
跳过：抽查过的 50 段、现在已经有抓挠标注的、已有抓挠疑似片段（任何状态）压着的、有人工标注（非 AI 自动片段）压着的、同任务互相重叠的。
默认试运行；加 --apply 才写库。"""
import asyncio
import json
import sys
from collections import Counter

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402

MODEL = "未标负例新模型·未标时间补标"
DRY = "--apply" not in sys.argv
ST_CN = {"pending": "待判", "confirmed": "已确认", "rejected": "已排除", "uncertain": "待定"}


def ov(a0, a1, b0, b1):
    return a0 < b1 and a1 > b0


async def main():
    pool = json.load(open("/tmp/unl_pool.json", encoding="utf-8"))
    spot = json.load(open("/tmp/unl_spot_picked.json", encoding="utf-8"))
    skip, put, by_site, by_proj, tasks = Counter(), [], Counter(), Counter(), set()
    async with SessionLocal() as db:
        lab = dict((await db.execute(select(LabelDefinition.id, LabelDefinition.display_name))).all())
        for p in sorted(pool, key=lambda x: (x["sample_code"], x["start"])):
            s_ms, e_ms = int(p["start"] * 1000), int(p["end"] * 1000)
            t = (await db.execute(
                select(Task).join(Sample, Sample.id == Task.sample_id)
                .where(Sample.sample_code == p["sample_code"], Task.segment_start_ms.is_(None))
                .order_by(Task.id.desc()))).scalars().first()
            if t is None:
                skip["没有任务"] += 1
                continue
            if any(q["task_id"] == t.id and ov(s_ms, e_ms, q["s_ms"], q["e_ms"]) for q in spot):
                skip["抽查过"] += 1
                continue
            if any(q["task_id"] == t.id and ov(s_ms, e_ms, q["s_ms"], q["e_ms"]) for q in put):
                skip["同任务重叠"] += 1
                continue
            cands = (await db.execute(select(AiCandidate.status).where(
                AiCandidate.task_id == t.id, AiCandidate.round_no == t.round_no, AiCandidate.label_name.like("抓挠%"),
                AiCandidate.start_time_ms < e_ms, AiCandidate.end_time_ms > s_ms))).scalars().all()
            if cands:
                skip["已有抓挠疑似片段·" + "/".join(sorted({ST_CN.get(str(x).split('.')[-1], str(x)) for x in cands}))] += 1
                continue
            recs = (await db.execute(select(AnnotationRecord.id).where(
                AnnotationRecord.task_id == t.id, AnnotationRecord.round_no == t.round_no))).scalars().all()
            items = (await db.execute(select(AnnotationLabelItem).where(
                AnnotationLabelItem.annotation_record_id.in_(recs),
                AnnotationLabelItem.start_time_ms < e_ms, AnnotationLabelItem.end_time_ms > s_ms))).scalars().all() if recs else []
            if any((lab.get(it.label_id) or "").startswith("抓挠") for it in items):
                skip["已有抓挠标注"] += 1
                continue
            human = [it for it in items if str(it.source_type).split(".")[-1] != "ai_generated"]
            if human:
                skip["有人工标注压着"] += 1
                continue
            approved = t.status == TaskStatus.APPROVED
            proj = (await db.execute(select(Project.name).where(Project.id == t.project_id))).scalar()
            put.append({**p, "task_id": t.id, "s_ms": s_ms, "e_ms": e_ms, "project": proj, "approved": approved})
            by_site[p["site"]] += 1
            by_proj[proj] += 1
            tasks.add(t.id)
            if not DRY:
                ev = (("【已通过·先点「继续编辑」才能判】" if approved else "")
                      + "【补标】新模型报了抓挠，这段时间没人标过（同批抽查约 3/4 是真抓挠）：是抓挠就确认，起止可以调；"
                      f"模型 {float(p['p'] or 0):.0%} 把握，{p['end'] - p['start']:.0f} 秒"
                      + (f"，离最近的抓挠标注 {p['dist']:.0f} 秒" if p.get("dist") is not None else "，这个文件里没标过抓挠"))
                db.add(AiCandidate(task_id=t.id, round_no=t.round_no, label_name="抓挠", start_time_ms=s_ms, end_time_ms=e_ms,
                                   confidence=round(float(p["p"] or 0), 3), reason="low_conf", model=MODEL, evidence=ev[:300],
                                   status=CandidateStatus.pending))
        if not DRY:
            await db.commit()
    print("池子", len(pool), "；放", len(put), "；任务", len(tasks), "；已通过任务里的", sum(q["approved"] for q in put))
    print("按场地：", dict(by_site))
    print("按项目：", dict(by_proj.most_common()))
    print("跳过：", dict(skip))
    json.dump(put, open("/tmp/unl_fill_put.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("（试运行，没写库）" if DRY else f"已写入 {len(put)} 条补标疑似片段")


asyncio.run(main())
