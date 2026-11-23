"""api 容器里跑：统计「未标负例新模型·未标时间抽查」的判定，含改起止 / 拆分（不写库）"""
import asyncio
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import SPECIAL_REASONS, AiCandidate  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402

MODEL = "未标负例新模型·未标时间抽查"
picked = json.load(open("/tmp/unl_spot_picked.json", encoding="utf-8"))


def site_of(task_id, s, e):
    for p in picked:
        if p["task_id"] == task_id and p["s_ms"] <= e and p["e_ms"] >= s:
            return p["site"], (p["s_ms"], p["e_ms"])
    return "?", None


async def main():
    print("spot 在 SPECIAL_REASONS 里：", "spot" in SPECIAL_REASONS)
    async with SessionLocal() as db:
        cs = (await db.execute(select(AiCandidate).where(AiCandidate.model == MODEL).order_by(AiCandidate.task_id, AiCandidate.start_time_ms))).scalars().all()
        lab = dict((await db.execute(select(LabelDefinition.id, LabelDefinition.display_name))).all())
        print("行数", len(cs), "（原 50，多出来的是拆分）")
        orig = defaultdict(list)
        for c in cs:
            site, o = site_of(c.task_id, c.start_time_ms, c.end_time_ms)
            orig[(c.task_id, o)].append(c)
        verdict = Counter()
        by_site = defaultdict(Counter)
        for (tid, o), rows in orig.items():
            site = site_of(tid, *o)[0] if o else "?"
            sts = []
            for c in rows:
                st = str(c.status).split(".")[-1]
                dl = lab.get(c.decided_label_id) if c.decided_label_id else None
                if st == "confirmed" and dl and not dl.startswith("抓挠"):
                    st = f"改成{dl}"
                sts.append(st)
                ne = (c.boundary_edits or "").count('"from"')
                print(f"任务{tid} {site:10s} 原[{o[0]/1000:.1f},{o[1]/1000:.1f}] 现[{c.start_time_ms/1000:.1f},{c.end_time_ms/1000:.1f}] {st:10s} 调整{ne}次"
                      + (f" 待定原因 {c.uncertain_reason}" if c.uncertain_reason else ""))
            # 一段原预报：拆开后只要有一块是抓挠，就算这次报对了
            if any(s == "confirmed" for s in sts):
                v = "是抓挠"
            elif all(s == "pending" for s in sts):
                v = "没判"
            elif any(s == "uncertain" for s in sts):
                v = "待定"
            else:
                v = "不是抓挠"
            verdict[v] += 1
            by_site[site][v] += 1
            if len(rows) > 1:
                print(f"   ↑ 拆成 {len(rows)} 段：{sts}")
        # 人在这些时间段里自己画的标注（不是从候选确认出来的）
        print("\n抽查时间段里的人工标注（非候选确认）：")
        for p in picked:
            recs = (await db.execute(select(AnnotationRecord.id).where(AnnotationRecord.task_id == p["task_id"]))).scalars().all()
            if not recs:
                continue
            its = (await db.execute(select(AnnotationLabelItem).where(
                AnnotationLabelItem.annotation_record_id.in_(recs),
                AnnotationLabelItem.start_time_ms < p["e_ms"], AnnotationLabelItem.end_time_ms > p["s_ms"]))).scalars().all()
            for it in its:
                print(f"  任务{p['task_id']} [{p['s_ms']/1000:.1f},{p['e_ms']/1000:.1f}] ← {lab.get(it.label_id)} [{it.start_time_ms/1000:.1f},{it.end_time_ms/1000:.1f}]"
                      f" 来源{str(it.source_type).split('.')[-1]} from_cand={it.from_candidate_id} 建于{it.created_at}")
    print("\n总：", dict(verdict))
    for s, c in by_site.items():
        print(s, dict(c))


asyncio.run(main())
