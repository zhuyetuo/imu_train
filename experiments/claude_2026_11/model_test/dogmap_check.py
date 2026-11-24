"""平台现有的 sample_dog_id（人关联优先，否则按 场地+IMU 推）能不能把每个样本都对上狗；已标样本按体型的分布。"""
import asyncio
from collections import Counter
from sqlalchemy import select, distinct
from app.db.session import SessionLocal
from app.models.sample import Sample
from app.models.task import Task
from app.models.dog import Dog
from app.services import dog_name_service as DN


async def m():
    async with SessionLocal() as db:
        idx = await DN.dog_index(db)
        prof = await DN.dog_profiles(db)
        dogs = {d.id: d for d in (await db.execute(select(Dog))).scalars().all()}
        tasked = set((await db.execute(select(distinct(Task.sample_id)))).scalars().all())
        rows = (await db.execute(select(Sample.id, Sample.sample_code, Sample.dog_id, Sample.video_cam1_path,
                                        Sample.session_date, Sample.video_duration_sec))).all()
        src = Counter(); miss = Counter(); by_dog = Counter(); hrs = Counter(); clash = 0
        for r in rows:
            guess = await DN.sample_dog_id(db, r.sample_code, None, r.video_cam1_path, idx=idx, profiles=prof)
            eff = r.dog_id or guess
            t = r.id in tasked
            if r.dog_id and guess and r.dog_id != guess:
                clash += 1
            src[("人工" if r.dog_id else ("推断" if guess else "对不上"), "已建任务" if t else "未建任务")] += 1
            if eff is None:
                miss[(r.session_date.strftime("%Y-%m") if r.session_date else None, r.sample_code.split("_")[-1])] += 1
            elif t:
                d = dogs[eff]
                by_dog[(d.site, d.name, d.size)] += 1
                hrs[d.size] += (r.video_duration_sec or 0) / 3600
        print("来源", sorted(src.items()))
        print("人工关联跟推断不一致", clash)
        print("对不上的（月, imu）", miss.most_common(12))
        print("已建任务样本 按狗", sorted(by_dog.items(), key=lambda kv: str(kv[0])))
        print("已建任务样本 按体型 小时", {k: round(v) for k, v in hrs.items()})
asyncio.run(m())
