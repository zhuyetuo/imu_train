"""在 api 容器里跑：把「未标时间当负例」新模型在没人标的时间里报的抓挠，随机抽 50 段写成平台抽查疑似片段（reason=spot）。
用来定真实查准：这些预报在评估里一律算错，真抓挠占多少，决定下一步改模型还是补标注。
只放整段任务；同一任务里的抽查片段不重叠；已审核通过的、跟已有抓挠疑似片段重叠的照放，但在说明最前面打标记。
默认试运行；加 --apply 才写库。要删：DELETE FROM ai_candidates WHERE model = 'MODEL 常量'"""
import asyncio
import json
import random
import sys

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402

MODEL = "未标负例新模型·未标时间抽查"
WANT = {"bolangwang": 30, "yingpeng": 11, "gouchang": 9}
DRY = "--apply" not in sys.argv
ST_CN = {"pending": "待判", "confirmed": "已确认", "rejected": "已排除", "uncertain": "待定"}


async def main():
    pool = json.load(open("/tmp/unl_pool.json", encoding="utf-8"))
    random.seed(20261124)
    picked, skip = [], {"没有任务": 0, "同任务重叠": 0}
    async with SessionLocal() as db:
        for site, n in WANT.items():
            cands = [p for p in pool if p["site"] == site]
            random.shuffle(cands)
            got = 0
            for p in cands:
                if got >= n:
                    break
                t = (await db.execute(
                    select(Task).join(Sample, Sample.id == Task.sample_id)
                    .where(Sample.sample_code == p["sample_code"], Task.segment_start_ms.is_(None))
                    .order_by(Task.id.desc()))).scalars().first()
                if t is None:
                    skip["没有任务"] += 1
                    continue
                s_ms, e_ms = int(p["start"] * 1000), int(p["end"] * 1000)
                if any(q["task_id"] == t.id and q["s_ms"] < e_ms and q["e_ms"] > s_ms for q in picked):
                    skip["同任务重叠"] += 1
                    continue
                approved = t.status == TaskStatus.APPROVED
                dups = (await db.execute(select(AiCandidate.status).where(
                    AiCandidate.task_id == t.id, AiCandidate.round_no == t.round_no, AiCandidate.label_name.like("抓挠%"),
                    AiCandidate.start_time_ms < e_ms, AiCandidate.end_time_ms > s_ms))).scalars().all()
                dup_txt = "、".join(sorted({ST_CN.get(str(x).split(".")[-1], str(x)) for x in dups}))
                tags = ("【已通过·先点「继续编辑」才能判】" if approved else "") + (f"【跟已有抓挠疑似片段重叠（{dup_txt}）】" if dups else "")
                proj = (await db.execute(select(Project.name).where(Project.id == t.project_id))).scalar()
                picked.append({**p, "task_id": t.id, "s_ms": s_ms, "e_ms": e_ms, "project": proj, "approved": approved,
                               "dup": dup_txt or None, "status": str(t.status).split(".")[-1]})
                got += 1
                if not DRY:
                    ev = (tags + f"新模型（未标时间当负例）报了抓挠，这段时间没人标过：请对着视频判是不是抓挠"
                          f"（模型 {float(p['p'] or 0):.0%} 把握，{p['end'] - p['start']:.0f} 秒"
                          + (f"，离最近的抓挠标注 {p['dist']:.0f} 秒" if p.get("dist") is not None else "，这个文件里没标过抓挠")
                          + "；2026-11-24 抽查）")
                    db.add(AiCandidate(task_id=t.id, round_no=t.round_no, label_name="抓挠", start_time_ms=s_ms, end_time_ms=e_ms,
                                       confidence=round(float(p["p"] or 0), 3), reason="spot", model=MODEL, evidence=ev[:300],
                                       status=CandidateStatus.pending))
            print(f"{site}: 要 {n}，放了 {got}（可选 {len(cands)}）")
        if not DRY:
            await db.commit()
    by = {}
    for q in picked:
        by[q["project"]] = by.get(q["project"], 0) + 1
    print("按项目：", by)
    print("任务状态：", {s: sum(1 for q in picked if q["status"] == s) for s in {q["status"] for q in picked}})
    print("跳过：", skip, "；已通过：", sum(q["approved"] for q in picked), "；跟已有抓挠疑似片段重叠：", sum(bool(q["dup"]) for q in picked))
    json.dump(picked, open("/tmp/unl_spot_picked.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("（试运行，没写库）" if DRY else f"已写入 {len(picked)} 条抽查疑似片段")


asyncio.run(main())
