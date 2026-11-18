"""在 api 容器里跑：把抽查用的新模型片段写成平台疑似片段（reason=model_neg「模型判成这一类」，model 标「train37 精准版·抽查」）。
只放没审核通过的任务（通过的点进去可能是只读，判不了）；跟已有同类疑似片段重叠的跳过。
要删：DELETE FROM ai_candidates WHERE model = 'train37 精准版·抽查'（或「清理候选」里按来源删，但那会连负例挖掘的一起删）。"""
import asyncio
import json
import random
import sys

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402

MODEL = "train37 精准版·抽查"
EVID = "新模型（train37 精准版）报了、人工没标：抽查新模型多报是真抓挠还是误报（2026-11-18）"
DRY = "--apply" not in sys.argv


async def main():
    pool = json.load(open("/tmp/spot_pool.json", encoding="utf-8"))
    random.seed(11)
    want = {"2026-08": 8, "2026-09": 12, "2026-10": 10, "2026-11": 4, "Y": 6}
    picked, by_proj = [], {}
    async with SessionLocal() as db:
        for key, n in want.items():
            cands = [p for p in pool if (p["cat"] == "Y" if key == "Y" else (p["cat"] == "X" and p["month"] == key))]
            random.shuffle(cands)
            got = 0
            for p in cands:
                if got >= n:
                    break
                t = (await db.execute(select(Task).where(Task.id == p["task_id"]))).scalars().first()
                if t is None or t.status == TaskStatus.APPROVED:
                    continue
                s_ms, e_ms = int(p["start"] * 1000), int(p["end"] * 1000)
                dup = (await db.execute(select(AiCandidate.id).where(
                    AiCandidate.task_id == t.id, AiCandidate.label_name.like("抓挠%"),
                    AiCandidate.start_time_ms < e_ms, AiCandidate.end_time_ms > s_ms))).first()
                if dup:
                    continue
                if any(q["task_id"] == t.id and q["start"] < p["end"] and q["end"] > p["start"] for q in picked):
                    continue
                proj = (await db.execute(select(Project.name).where(Project.id == t.project_id))).scalar()
                picked.append({**p, "round_no": t.round_no, "status": str(t.status).split(".")[-1], "project": proj})
                by_proj[proj] = by_proj.get(proj, 0) + 1
                got += 1
                if not DRY:
                    db.add(AiCandidate(task_id=t.id, round_no=t.round_no, label_name="抓挠", start_time_ms=s_ms, end_time_ms=e_ms,
                                       confidence=None, reason="model_neg", model=MODEL, evidence=EVID, status=CandidateStatus.pending))
            print(f"{key}: 要 {n}，放了 {got}（可选 {len(cands)}）")
        if not DRY:
            await db.commit()
    print("按项目：", by_proj)
    print("任务状态：", {s: sum(1 for p in picked if p["status"] == s) for s in {p["status"] for p in picked}})
    json.dump(picked, open("/tmp/spot_picked.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("（试运行，没写库）" if DRY else f"已写入 {len(picked)} 条疑似片段")


asyncio.run(main())
