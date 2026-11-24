"""模型测试第一批：从 11-19~11-23（所有现有模型都没见过）里按狗挑窗口。
加 --apply 才动手：建项目「模型测试 v1」（套模板 3、从零标注、不预标、不指派）、每个窗口一个短任务、
建测试集「测试集 v1」（不冻结）把这些短任务加进去。

每只狗：
  · 抓挠提示 ≤10 段：train60 预标报的抓挠，凑够 60 秒（找得到事件，但会偏向模型报得出来的）
  · 未佩戴边界 3 段：train60 预标的未佩戴开始处 ±30 秒（考边界）
  · 随机 1 分钟 12 段：白天 / 夜里各半，5 天轮着来（不看模型，召回率靠它才不偏）
  · 随机 10 分钟 3 段：白天 2、夜里 1（长片段，考连续事件和分钟数）
同一个样本里的窗口不重叠。随机种子固定，重跑结果一样。
"""
import asyncio
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord, LabelItemSource  # noqa: E402
from app.models.dog import Dog  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.models.task import Task  # noqa: E402
from app.services import dog_name_service as DN  # noqa: E402
from app.services import model_test_service as mts  # noqa: E402

APPLY = "--apply" in sys.argv
PROJECTS = [337, 338, 339, 340, 341]
PROJECT_NAME = "模型测试 v1"
SET_NAME = "测试集 v1"
TEMPLATE = 3
N_SCR, N_NW, N_RAND, N_LONG = 10, 3, 12, 3
SHORT, LONG = 60_000, 600_000
rng = random.Random(20261123)


def hour_of(code: str) -> int | None:
    m = re.search(r"_(\d{8})_(\d{2})\d{4}", code or "")
    return int(m.group(2)) if m else None


def is_day(h: int | None) -> bool:
    return h is not None and 8 <= h < 20


async def main():
    async with SessionLocal() as db:
        rows = (await db.execute(select(Task, Sample).join(Sample, Sample.id == Task.sample_id)
                                 .where(Task.project_id.in_(PROJECTS)))).all()
        idx = await DN.dog_index(db); prof = await DN.dog_profiles(db)
        dogs = {d.id: d for d in (await db.execute(select(Dog))).scalars().all()}
        lrows = (await db.execute(select(LabelDefinition.id, LabelDefinition.display_name, LabelDefinition.parent_id))).all()
        lname = {i: n for i, n, _ in lrows}; parent = {i: p for i, _, p in lrows}

        def top(l):
            while parent.get(l) is not None:
                l = parent[l]
            return lname.get(l)

        smp = {}          # sample_id → (sample, dog, dur_ms, hour, date)
        by_dog = defaultdict(list)
        for t, s in rows:
            if s.id in smp:
                continue
            did = await DN.sample_dog_id(db, s.sample_code, s.dog_id, s.video_cam1_path, idx=idx, profiles=prof)
            dur = int((s.video_duration_sec or 0) * 1000)
            if not did or dur < 10 * 60_000:
                continue
            smp[s.id] = (s, dogs[did], dur, hour_of(s.sample_code), s.session_date)
            by_dog[did].append(s.id)
        task_of_sample = {s.id: t.id for t, s in rows}
        it = AnnotationLabelItem
        ai = defaultdict(list)   # (dog, 大类) → [(sample, start, end)]
        for tid, lid, a, b, sid in (await db.execute(
            select(AnnotationRecord.task_id, it.label_id, it.start_time_ms, it.end_time_ms, Task.sample_id)
            .join(AnnotationRecord, it.annotation_record_id == AnnotationRecord.id)
            .join(Task, Task.id == AnnotationRecord.task_id)
            .where(Task.project_id.in_(PROJECTS), AnnotationRecord.round_no == Task.round_no,
                   it.source_type == LabelItemSource.ai_generated))).all():
            if sid in smp and top(lid) in ("抓挠", "未佩戴"):
                ai[(smp[sid][1].id, top(lid))].append((sid, int(a), int(b)))

        chosen = []        # (sample_id, start, end, 来源)
        taken = defaultdict(list)

        def free(sid, s, e):
            return all(e <= a or s >= b for a, b in taken[sid])

        def take(sid, s, e, why):
            taken[sid].append((s, e))
            chosen.append((sid, s, e, why))

        for did, sids in sorted(by_dog.items()):
            days = sorted({smp[x][4] for x in sids})
            # 抓挠提示：按天轮着挑
            evs = ai.get((did, "抓挠"), [])
            rng.shuffle(evs)
            per_day = defaultdict(list)
            for ev in evs:
                per_day[smp[ev[0]][4]].append(ev)
            n = 0
            while n < N_SCR and any(per_day.values()):
                for d in days:
                    if n >= N_SCR or not per_day[d]:
                        continue
                    sid, a, b = per_day[d].pop()
                    s, e = mts.pad_window(a, b, SHORT, 0, smp[sid][2])
                    if free(sid, s, e):
                        take(sid, s, e, "抓挠提示（train60 预标报的）")
                        n += 1
            # 未佩戴边界
            evs = [x for x in ai.get((did, "未佩戴"), []) if x[1] >= 30_000]
            rng.shuffle(evs)
            n, used_days = 0, set()
            for sid, a, b in evs:
                if n >= N_NW:
                    break
                if smp[sid][4] in used_days and len(used_days) < len(days):
                    continue
                s, e = a - 30_000, a + 30_000
                if e <= smp[sid][2] and free(sid, s, e):
                    take(sid, s, e, "未佩戴开始边界（train60 预标报的）")
                    used_days.add(smp[sid][4])
                    n += 1

            # 随机窗：白天 / 夜里交替，日子轮着来
            def rand_windows(k, length, why, day_seq):
                got = 0
                for i in range(k * 20):
                    if got >= k:
                        break
                    want_day = day_seq[got % len(day_seq)]
                    d = days[(got + i) % len(days)]
                    pool = [x for x in sids if smp[x][4] == d and is_day(smp[x][3]) == want_day and smp[x][2] >= length]
                    if not pool:
                        continue
                    sid = rng.choice(pool)
                    s = rng.randrange(0, smp[sid][2] - length + 1, 1000)
                    if free(sid, s, s + length):
                        take(sid, s, s + length, f"{why}·{'白天' if want_day else '夜里'}")
                        got += 1
            rand_windows(N_RAND, SHORT, "随机 1 分钟", [True, False])
            rand_windows(N_LONG, LONG, "随机 10 分钟", [True, True, False])

        # 汇总
        print(f"共 {len(chosen)} 段，视频 {sum(e - s for _, s, e, _ in chosen) / 3600_000:.1f} 小时，样本 {len({c[0] for c in chosen})} 个")
        cnt = Counter()
        for sid, s, e, why in chosen:
            d = smp[sid][1]
            cnt[(f"{d.site[:2]}{d.name}({d.size})", why.split("（")[0].split("·")[0])] += 1
        for dn in sorted({k[0] for k in cnt}):
            print(" ", dn, {k[1]: v for k, v in cnt.items() if k[0] == dn})
        print("白天/夜里", Counter("白天" if is_day(smp[c[0]][3]) else "夜里" for c in chosen))
        print("日期", sorted(Counter(str(smp[c[0]][4]) for c in chosen).items()))
        if not APPLY:
            print("（试运行，什么都没建）")
            return

        from app.api.v1.label_templates import apply_template
        from app.models.model_test import ModelTestItem, ModelTestSet
        from app.models.project import Project
        from app.models.task import TaskStatus, TaskType
        from app.models.user import User
        admin = await db.get(User, 1)
        if (await db.execute(select(Project.id).where(Project.name == PROJECT_NAME))).scalar():
            sys.exit(f"停：已经有项目「{PROJECT_NAME}」")
        if (await db.execute(select(ModelTestSet.id).where(ModelTestSet.name == SET_NAME))).scalar():
            sys.exit(f"停：已经有测试集「{SET_NAME}」")
        from app.api.v1.projects import create_project
        from app.schemas.project import ProjectCreate
        await create_project(ProjectCreate(
            name=PROJECT_NAME,
            description="模型测试第一批（11-19~11-23）：每个任务是一个窗口，从零标、窗口里全标（休息、活动也标），标完要第二个人对着视频复核。不预标。"),
            db, admin)
        p = (await db.execute(select(Project).where(Project.name == PROJECT_NAME))).scalar_one()
        await apply_template(TEMPLATE, p.id, db, admin)
        tset = ModelTestSet(name=SET_NAME, created_by=admin.id,
                            note="第一批：11-19~11-23，10 只狗；抓挠提示 / 未佩戴边界 / 随机 1 分钟 / 随机 10 分钟。标注在项目「模型测试 v1」")
        db.add(tset)
        await db.flush()
        for sid, s, e, why in sorted(chosen, key=lambda c: (str(smp[c[0]][4]), smp[c[0]][0].sample_code, c[1])):
            t = Task(project_id=p.id, sample_id=sid, task_type=TaskType.from_scratch, status=TaskStatus.PENDING_ASSIGN,
                     segment_start_ms=s, segment_end_ms=e, created_by=admin.id)
            db.add(t)
            await db.flush()
            db.add(ModelTestItem(set_id=tset.id, task_id=t.id, sample_id=sid, note=why, added_by=admin.id))
        await db.commit()
        print(f"建好了：项目 #{p.id}「{PROJECT_NAME}」{len(chosen)} 个短任务；测试集 #{tset.id}「{SET_NAME}」")


asyncio.run(main())
