"""不写库：拿跑过评测的样本，比「不给窗口」和「窗口 = 整个样本」应一致；窗口缩小后真值/预测只剩窗口里的。"""
import asyncio
from collections import Counter
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.inference_run import SampleInferenceRun
from app.models.sample import Sample
from app.services.model_eval_service import evaluate


async def m():
    async with SessionLocal() as db:
        from app.models.task import Task
        from app.services.model_eval_service import _REVIEWED
        reviewed = select(Task.sample_id).where(Task.status.in_(_REVIEWED))
        rows = (await db.execute(select(SampleInferenceRun.sample_id, SampleInferenceRun.model_tag, SampleInferenceRun.mode)
                                 .where(SampleInferenceRun.sample_id.in_(reviewed))
                                 .order_by(SampleInferenceRun.id.desc()).limit(3000))).all()
        c = Counter((t, md) for _s, t, md in rows)
        (tag, mode), _ = c.most_common(1)[0]
        sids = list(dict.fromkeys(s for s, t, md in rows if (t, md) == (tag, mode)))[:40]
        dur = dict((await db.execute(select(Sample.id, Sample.video_duration_sec).where(Sample.id.in_(sids)))).all())
        v = [{"model_tag": tag, "mode": mode}]
        a = await evaluate(db, sids, v, label="抓挠")
        full = {s: [(0, int((dur.get(s) or 0) * 1000) + 10_000_000)] for s in sids}
        b = await evaluate(db, sids, v, label="抓挠", windows=full)
        half = {s: [(0, int((dur.get(s) or 0) * 500))] for s in sids}
        h = await evaluate(db, sids, v, label="抓挠", windows=half)
        f = lambda r: (r["n_truth"], *(r["versions"][0][k] for k in ("tp", "fp", "fn"))) if r["versions"] else r["warnings"][-1:]
        print("版本", tag, mode, "样本", len(sids))
        print("不给窗口  真值/tp/fp/fn", f(a))
        print("窗口=整段", f(b), "一致" if f(a) == f(b) else "不一致！")
        print("窗口=前一半", f(h), h.get("windowed"))
asyncio.run(m())
