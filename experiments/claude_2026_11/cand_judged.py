"""api 容器里跑：各来源的「疑似饮水 / 进食」候选，人判过的里确认了多少、排除了多少（不写库）"""
import asyncio
from collections import defaultdict

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.models.ai_candidate import AiCandidate


async def main():
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(AiCandidate.label_name, AiCandidate.reason, AiCandidate.status, func.count())
            .where(AiCandidate.label_name.in_(["饮水", "进食"]))
            .group_by(AiCandidate.label_name, AiCandidate.reason, AiCandidate.status))).all()
    t = defaultdict(lambda: defaultdict(int))
    for lab, reason, st, n in rows:
        t[(lab, reason)][st.value if hasattr(st, "value") else str(st)] += n
    print(f"{'类别':<4}{'来源':<12}{'确认':>6}{'排除':>6}{'待定':>6}{'没判':>7}{'确认率':>8}")
    for (lab, reason), c in sorted(t.items()):
        ok, no = c.get("confirmed", 0), c.get("rejected", 0)
        rate = f"{ok / (ok + no):.0%}" if ok + no else "-"
        print(f"{lab:<4}{reason:<12}{ok:>6}{no:>6}{c.get('uncertain', 0):>6}{c.get('pending', 0):>7}{rate:>8}")

asyncio.run(main())
