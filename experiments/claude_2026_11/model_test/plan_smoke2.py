"""照「抓挠测试说明」：1 分钟版 + 7 分钟版，有/无各半，无抓挠四类 4:2:3:1。线上试挑（不写库）。"""
import asyncio
import time
from collections import Counter
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.dog import Dog
from app.services import model_test_service as mts

PLAN = [("佛山博朗旺狗场", "奥迪", [8, 6, 6, 7], [6, 8]), ("佛山博朗旺狗场", "小七", [7, 7, 7, 6], [8, 6]),
        ("佛山博朗旺狗场", "旺财1", [7, 7, 6, 7], [4, 9]), ("佛山博朗旺狗场", "平安", [6, 7, 8, 6], [8, 6]),
        ("佛山博朗旺狗场", "久久", [7, 6, 6, 8], [6, 7]), ("佛山博朗旺狗场", "小白", [6, 7, 7, 7], [7, 6]),
        ("公司影棚", "巴利", [3, 3, 3, 3], [4, 2]), ("公司影棚", "lulu", [1, 2, 2, 1], [2, 1]),
        ("深圳雷诺狗场", "年糕", [1, 2, 2, 1], [1, 2]), ("深圳雷诺狗场", "闪电", [2, 1, 1, 2], [2, 1]),
        ("公司影棚", "bibi", [1, 1, 1, 1], [1, 1]), ("公司影棚", "小满", [1, 1, 1, 1], [1, 1])]
CATS = [{"name": "像抓挠的动作", "anchors": ["舔", "啃", "蹭", "甩头/抖身"], "ratio": 4},
        {"name": "正常活动", "anchors": ["活动"], "ratio": 2},
        {"name": "休息睡眠", "anchors": ["静止/休息"], "ratio": 3},
        {"name": "未佩戴", "anchors": ["未佩戴"], "ratio": 1}]


async def m():
    async with SessionLocal() as db:
        dogs = {(d.site, d.name): d.id for d in (await db.execute(select(Dog))).scalars().all()}
        plan = {"target": "抓挠", "durations": [60, 420], "parts": ["抓挠-头颈耳", "抓挠-躯干"], "neg_cats": CATS,
                "dogs": [{"dog_id": dogs[(s, n)], "q": {"60": [v[0], v[1]], "420": [v[2], v[3]]},
                          "p": {"抓挠-头颈耳": p[0], "抓挠-躯干": p[1]}} for s, n, v, p in PLAN]}
        t0 = time.time()
        r = await mts.plan_run(db, 1, plan)
        print(f"用时 {time.time() - t0:.1f} 秒；计划 200 段，挑到 {r['n_picked']} 段，视频 {r['video_hours']} 小时")
        for d in ("60", "420"):
            rows = [v[1] if d == "60" else v[3] for s, n, v, p in PLAN]
            cols = mts.split_by_ratio(sum(rows), [c["ratio"] for c in CATS])
            want = Counter({c["name"]: x for c, x in zip(CATS, cols)})
            got = Counter()
            for rep in r["report"].values():
                got.update(rep.get("n", {}).get(d, {}))
            print(f"  {int(d) // 60} 分钟·无抓挠：" + "；".join(f"{c['name']} 计划 {want[c['name']]} 挑到 {got[c['name']]}" for c in CATS))
        short = []
        for s, n, v, p in PLAN:
            q = r["report"][str(dogs[(s, n)])]["q"]
            got = [q["60"][0], q["60"][1], q["420"][0], q["420"][1]]
            if got != v:
                short.append(f"{n} 计划{v} 挑到{got}")
        print("  有缺口的狗：", short or "无")
asyncio.run(m())
