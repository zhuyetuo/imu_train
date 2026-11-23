"""api 容器里跑：把补标的 810 段挪到「补标·未标时间」来源（reason=unl_fill）。
  - 346 段：model=「未标负例新模型·未标时间补标」、reason=low_conf → reason=unl_fill
  - 早先「模型低置信」里跟新模型未标时间预报重叠、还没判的抓挠疑似片段（约 464 段）→ reason=unl_fill、同一个 model，
    说明前面加【补标·早先低置信】（原来 model 是空的，记在说明里）
默认试运行；加 --apply 才写库。"""
import asyncio
import json
import sys

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.models.task import Task  # noqa: E402

MODEL = "未标负例新模型·未标时间补标"
DRY = "--apply" not in sys.argv


async def main():
    pool = json.load(open("/tmp/unl_pool.json", encoding="utf-8"))
    spot = json.load(open("/tmp/unl_spot_picked.json", encoding="utf-8"))
    async with SessionLocal() as db:
        mine = (await db.execute(select(AiCandidate).where(AiCandidate.model == MODEL, AiCandidate.reason == "low_conf"))).scalars().all()
        old = {}
        for p in pool:
            s_ms, e_ms = int(p["start"] * 1000), int(p["end"] * 1000)
            t = (await db.execute(select(Task).join(Sample, Sample.id == Task.sample_id).where(
                Sample.sample_code == p["sample_code"], Task.segment_start_ms.is_(None)).order_by(Task.id.desc()))).scalars().first()
            if t is None or any(q["task_id"] == t.id and q["s_ms"] < e_ms and q["e_ms"] > s_ms for q in spot):
                continue
            for c in (await db.execute(select(AiCandidate).where(
                    AiCandidate.task_id == t.id, AiCandidate.round_no == t.round_no, AiCandidate.label_name.like("抓挠%"),
                    AiCandidate.reason == "low_conf", AiCandidate.model.is_(None), AiCandidate.status == CandidateStatus.pending,
                    AiCandidate.start_time_ms < e_ms, AiCandidate.end_time_ms > s_ms))).scalars().all():
                old[c.id] = c
        print("新放的", len(mine), "（其中还没判", sum(c.status == CandidateStatus.pending for c in mine), "）；早先低置信重叠待判", len(old))
        if DRY:
            print("（试运行，没写库）")
            return
        for c in mine:
            c.reason = "unl_fill"
        for c in old.values():
            c.reason, c.model = "unl_fill", MODEL
            c.evidence = ("【补标·早先低置信】新模型在这段没人标的时间里也报了抓挠（同批抽查约 3/4 是真抓挠）；" + (c.evidence or ""))[:300]
        await db.commit()
        n = (await db.execute(select(AiCandidate.status).where(AiCandidate.reason == "unl_fill"))).scalars().all()
        print("改完：unl_fill 共", len(n), "条，待判", sum(x == CandidateStatus.pending for x in n))


asyncio.run(main())
