import asyncio
from collections import Counter
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.sample import Sample
from app.models.dog import Dog
from app.services import dog_name_service as DN, site_layout as SL


async def m():
    async with SessionLocal() as db:
        idx = await DN.dog_index(db); prof = await DN.dog_profiles(db)
        names = {d.id: d.name for d in (await db.execute(select(Dog))).scalars().all()}
        rows = (await db.execute(select(Sample.sample_code, Sample.dog_id, Sample.video_cam1_path, Sample.session_date)
                                 .where(Sample.dog_id.is_not(None)))).all()
        c = Counter(); agree = Counter()
        for r in rows:
            g = await DN.sample_dog_id(db, r.sample_code, None, r.video_cam1_path, idx=idx, profiles=prof)
            site = SL.site_of(r.sample_code, r.video_cam1_path) or "认不出"
            mon = r.session_date.strftime("%m") if r.session_date else "?"
            if g != r.dog_id:
                c[(site, mon, f"{names.get(g)}→{names.get(r.dog_id)}")] += 1
            else:
                agree[(site, mon)] += 1
        print("一致", sorted(agree.items()))
        print("不一致", c.most_common(20))
asyncio.run(m())
