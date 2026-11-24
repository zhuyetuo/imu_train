"""照参考方案的配额在线上试挑一次（不写库），看速度和每只狗缺几段。"""
import asyncio
import time
from sqlalchemy import select
from app.db.session import SessionLocal
from app.models.dog import Dog
from app.services import model_test_service as mts

# 场所, 名字, [1分有, 1分无, 10分有, 10分无], [头颈耳, 躯干]
PLAN = [("佛山博朗旺狗场", "奥迪", [8, 6, 6, 7], [6, 8]), ("佛山博朗旺狗场", "小七", [7, 7, 7, 6], [8, 6]),
        ("佛山博朗旺狗场", "旺财1", [7, 7, 6, 7], [4, 9]), ("佛山博朗旺狗场", "平安", [6, 7, 8, 6], [8, 6]),
        ("佛山博朗旺狗场", "久久", [7, 6, 6, 8], [6, 7]), ("佛山博朗旺狗场", "小白", [6, 7, 7, 7], [7, 6]),
        ("公司影棚", "巴利", [3, 3, 3, 3], [4, 2]), ("公司影棚", "lulu", [1, 2, 2, 1], [2, 1]),
        ("深圳雷诺狗场", "年糕", [1, 2, 2, 1], [1, 2]), ("深圳雷诺狗场", "闪电", [2, 1, 1, 2], [2, 1]),
        ("公司影棚", "bibi", [1, 1, 1, 1], [1, 1]), ("公司影棚", "小满", [1, 1, 1, 1], [1, 1])]


async def m():
    async with SessionLocal() as db:
        dogs = {(d.site, d.name): d.id for d in (await db.execute(select(Dog))).scalars().all()}
        plan = {"target": "抓挠", "durations": [60, 600], "parts": ["抓挠-头颈耳", "抓挠-躯干"], "source": "any",
                "dogs": [{"dog_id": dogs[(s, n)], "q": {"60": [v[0], v[1]], "600": [v[2], v[3]]},
                          "p": {"抓挠-头颈耳": p[0], "抓挠-躯干": p[1]}} for s, n, v, p in PLAN]}
        t0 = time.time()
        r = await mts.plan_run(db, 1, plan)
        print(f"用时 {time.time() - t0:.1f} 秒；计划 {sum(sum(v) for _, _, v, _ in PLAN)} 段，挑到 {r['n_picked']} 段，"
              f"视频 {r['video_hours']} 小时，{r['by_source']}")
        for s, n, v, p in PLAN:
            rep = r["report"][str(dogs[(s, n)])]
            g = rep["q"]
            got = [g["60"][0], g["60"][1], g["600"][0], g["600"][1]]
            gp = [rep["p"].get("抓挠-头颈耳", 0), rep["p"].get("抓挠-躯干", 0)]
            mark = "" if got == v and gp == p else "  ← 缺"
            print(f"  {n:4s} 计划 {v} 部位 {p}  挑到 {got} 部位 {gp}{mark}")
        print("提醒", r["warnings"][:3])
asyncio.run(m())
