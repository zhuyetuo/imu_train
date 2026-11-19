"""在 api 容器里跑：把「活动里疑似抓挠」专项片段写成平台疑似片段（reason=act_scr）。
输入 /tmp/act_scr_pool.json：[{sample_code, start, end（秒，相对样本开头）, p, dist, dataset}, ...]，已按优先级排好。
只放没审核通过的整段任务（通过的是只读、判不了）；跟已有抓挠疑似片段 / 已标抓挠重叠的跳过。
默认试运行；加 --apply 才写库。要删：清理候选里按来源「专项·活动里疑似抓挠」删，或 DELETE FROM ai_candidates WHERE reason=act_scr。"""
import asyncio, json, sys
sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402

MODEL = "v19neg 6轴·活动里疑似抓挠"
DRY = "--apply" not in sys.argv
N = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--n=")), 150))


async def main():
    pool = json.load(open("/tmp/act_scr_pool.json", encoding="utf-8"))
    picked, skip = [], {"没有任务": 0, "已通过": 0, "重叠": 0}
    async with SessionLocal() as db:
        for p in pool:
            if len(picked) >= N:
                break
            t = (await db.execute(
                select(Task).join(Sample, Sample.id == Task.sample_id)
                .where(Sample.sample_code == p["sample_code"], Task.segment_start_ms.is_(None))
                .order_by(Task.id.desc()))).scalars().first()
            if t is None:
                skip["没有任务"] += 1; continue
            s_ms, e_ms = int(p["start"] * 1000), int(p["end"] * 1000)
            if any(q["task_id"] == t.id and q["s_ms"] < e_ms and q["e_ms"] > s_ms for q in picked):
                skip["重叠"] += 1; continue
            # 已通过的、跟已有抓挠疑似片段重叠的也放，但在说明最前面打标记，一眼分得出来
            approved = t.status == TaskStatus.APPROVED
            dups = (await db.execute(select(AiCandidate.reason, AiCandidate.status).where(
                AiCandidate.task_id == t.id, AiCandidate.round_no == t.round_no, AiCandidate.label_name.like("抓挠%"),
                AiCandidate.reason != "act_scr",
                AiCandidate.start_time_ms < e_ms, AiCandidate.end_time_ms > s_ms))).all()
            st_cn = {"pending": "待判", "confirmed": "已确认", "rejected": "已排除", "uncertain": "待定"}
            dup_txt = "、".join(sorted({st_cn.get(str(x[1]).split(".")[-1], str(x[1])) for x in dups}))
            tags = ("【已通过·先点「继续编辑」才能判】" if approved else "") + (f"【跟已有低置信抓挠片段重叠（{dup_txt}）】" if dups else "")
            p = {**p, "approved": approved, "dup": dup_txt or None, "tags": tags}
            proj = (await db.execute(select(Project.name).where(Project.id == t.project_id))).scalar()
            picked.append({**p, "task_id": t.id, "s_ms": s_ms, "e_ms": e_ms, "project": proj,
                           "status": str(t.status).split(".")[-1]})
            if not DRY:
                db.add(AiCandidate(
                    task_id=t.id, round_no=t.round_no, label_name="抓挠", start_time_ms=s_ms, end_time_ms=e_ms,
                    confidence=round(float(p["p"]), 3), reason="act_scr", model=MODEL,
                    evidence=(tags + f"训练数据里标「活动」，模型 {float(p.get(chr(112))):.0%} 把握判成抓挠"
                              + (f"；离最近的抓挠标注 {float(p.get(chr(100)+chr(105)+chr(115)+chr(116)) or 0):.0f} 秒" if p.get("dist") is not None else "；这个文件里没标过抓挠"))[:300],
                    status=CandidateStatus.pending))
        if not DRY:
            await db.commit()
    by = {}
    for q in picked:
        by[q["project"]] = by.get(q["project"], 0) + 1
    print("按项目：", by)
    print("任务状态：", {s: sum(1 for q in picked if q["status"] == s) for s in {q["status"] for q in picked}})
    print("跳过：", skip)
    print("已通过：", sum(1 for q in picked if q["approved"]), "；跟已有片段重叠：", sum(1 for q in picked if q["dup"]), "；两样都是：", sum(1 for q in picked if q["approved"] and q["dup"]))
    json.dump(picked, open("/tmp/act_scr_picked.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("（试运行，没写库）" if DRY else f"已写入 {len(picked)} 条专项疑似片段")


asyncio.run(main())
