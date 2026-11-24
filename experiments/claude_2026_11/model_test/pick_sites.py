"""按场地的三个测试集，每个分「训练没用过」「训练用过」两部分（相对 #56/#61/#62 的训练数据集）：
  · 公司影棚 / 佛山博朗旺：没用过 = 「测试集 v1」里这个场地的（11-19~23）；用过 = 从训练数据集里的样本另挑窗口
  · 深圳雷诺（只有 9 月）：没用过 = 训练数据集里没有的样本挑窗口；用过 = 训练数据集里的样本挑窗口
新挑的窗口都建在项目「模型测试 场地」（套模板 3、从零标、不指派），每个窗口一个短任务。
提示窗口（抓挠 / 未佩戴）：优先用人工标过的（任务已提交/通过，或片段人加/人确认/人改过），没有再用 AI 预标；
这些段写成疑似片段（model=测试窗口提示·场地），随机窗口从零标。
影棚只用 9 月中以后文件名带场地的数据（更早的按设备号认狗常认错）。
加 --apply 才动手。随机种子固定。"""
import asyncio
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord, LabelItemSource  # noqa: E402
from app.models.dog import Dog  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402
from app.models.model_test import ModelTestItem, ModelTestSet  # noqa: E402
from app.models.model_version import ModelVersion  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402
from app.services import dog_name_service as DN  # noqa: E402
from app.services import model_test_service as mts  # noqa: E402
from app.services import site_layout as SL  # noqa: E402

APPLY = "--apply" in sys.argv
PROJECT = "模型测试 场地"
CAND_MODEL = "测试窗口提示·场地"
SITES = {"yingpeng": "公司影棚", "bolangwang": "佛山博朗旺狗场", "gouchang": "深圳雷诺狗场"}
SET_NAME = {"yingpeng": "测试集·公司影棚", "bolangwang": "测试集·佛山博朗旺", "gouchang": "测试集·深圳雷诺"}
FULL = dict(scr=10, nw=3, rand=12, long=3)      # 雷诺没用过
SEEN = dict(scr=6, nw=2, rand=6, long=1)        # 各场地用过（省人手）
SHORT, LONG = 60_000, 600_000
rng = random.Random(20261124)
SEEN_TAG, UNSEEN_TAG = "【训练用过】", "【训练没用过】"


def hour_of(code):
    m = re.search(r"_(\d{8})_(\d{2})\d{4}", code or "")
    return int(m.group(2)) if m else None


def is_day(h):
    return h is not None and 8 <= h < 20


async def main():
    async with SessionLocal() as db:
        from app.models.user import User
        admin = await db.get(User, 1)
        v1 = (await db.execute(select(ModelTestSet).where(ModelTestSet.name == "测试集 v1"))).scalar_one()
        v1_items = (await db.execute(select(ModelTestItem).where(ModelTestItem.set_id == v1.id))).scalars().all()
        v1_rows = {r["id"]: r for r in await mts.describe_items(db, list(v1_items))}
        v1_samples = {x.sample_id for x in v1_items}

        seen = set()
        for mid in (56, 61, 62):
            mv = await db.get(ModelVersion, mid)
            for n in mts.datasets_of_spec(mv.dataset_spec):
                c = await asyncio.to_thread(mts.dataset_sample_codes, n)
                if c:
                    seen |= c
        idx = await DN.dog_index(db); prof = await DN.dog_profiles(db)
        dogs = {d.id: d for d in (await db.execute(select(Dog))).scalars().all()}
        smp = {}
        pools = defaultdict(lambda: defaultdict(list))   # (site, 用过?) → dog → [sample]
        for s in (await db.execute(select(Sample))).scalars().all():
            site = SL.site_of(s.sample_code, s.imu_csv_path, s.video_cam1_path)
            if site not in SITES or s.id in v1_samples:
                continue
            dur = int((s.video_duration_sec or 0) * 1000)
            did = await DN.sample_dog_id(db, s.sample_code, s.dog_id, s.video_cam1_path, idx=idx, profiles=prof)
            if not did or dur < 10 * 60_000:
                continue
            smp[s.id] = (s, dogs[did], dur, hour_of(s.sample_code), s.session_date)
            pools[(site, s.sample_code in seen)][did].append(s.id)

        lrows = (await db.execute(select(LabelDefinition.id, LabelDefinition.display_name, LabelDefinition.parent_id))).all()
        lname = {i: n for i, n, _ in lrows}; parent = {i: p for i, _, p in lrows}

        def top(l):
            while parent.get(l) is not None:
                l = parent[l]
            return lname.get(l)

        it = AnnotationLabelItem
        ev = defaultdict(list)   # (sample, 大类) → [(人工?, start, end, conf)]
        want_sids = list(smp)
        for i in range(0, len(want_sids), 2000):
            for st, src, conf_, mod, conf, lid, a, b, sid in (await db.execute(
                select(Task.status, it.source_type, it.ai_confirmed, it.is_modified, it.ai_confidence, it.label_id,
                       it.start_time_ms, it.end_time_ms, Task.sample_id)
                .join(AnnotationRecord, it.annotation_record_id == AnnotationRecord.id)
                .join(Task, Task.id == AnnotationRecord.task_id)
                .where(Task.sample_id.in_(want_sids[i:i + 2000]), AnnotationRecord.round_no == Task.round_no,
                       Task.segment_start_ms.is_(None), it.uncertain.is_(False)))).all():
                tp = top(lid)
                if tp in ("抓挠", "未佩戴"):
                    human = st in (TaskStatus.SUBMITTED, TaskStatus.APPROVED) or src != LabelItemSource.ai_generated or conf_ or mod
                    ev[(sid, tp)].append((bool(human), int(a), int(b), conf))

        chosen = []   # (site, 用过?, sample, s, e, why, hint 大类)
        taken = defaultdict(list)

        def free(sid, s, e):
            return all(e <= a or s >= b for a, b in taken[sid])

        def pick(site, used, did, sids, q):
            days = sorted({smp[x][4] for x in sids})
            tag = SEEN_TAG if used else UNSEEN_TAG
            for tp, n_want, why in (("抓挠", q["scr"], "抓挠提示"), ("未佩戴", q["nw"], "未佩戴开始边界")):
                evs = [(h, sid, a, b, c) for sid in sids for (h, a, b, c) in ev.get((sid, tp), [])]
                rng.shuffle(evs)
                evs.sort(key=lambda x: not x[0])
                per_day = defaultdict(list)
                for x in evs:
                    per_day[smp[x[1]][4]].append(x)
                for d in per_day:
                    per_day[d].reverse()
                n = 0
                while n < n_want and any(per_day.values()):
                    for d in days:
                        if n >= n_want or not per_day.get(d):
                            continue
                        human, sid, a, b, conf = per_day[d].pop()
                        if tp == "抓挠":
                            s, e = mts.pad_window(a, b, SHORT, 0, smp[sid][2])
                        else:
                            if a < 30_000:
                                continue
                            s, e = a - 30_000, a + 30_000
                        if e <= smp[sid][2] and free(sid, s, e):
                            taken[sid].append((s, e))
                            chosen.append((site, used, sid, s, e, f"{tag}{why}（{'人工标过' if human else 'AI 预标报的'}）", tp))
                            n += 1
            for k, length, why, seq in ((q["rand"], SHORT, "随机 1 分钟", [True, False]),
                                         (q["long"], LONG, "随机 10 分钟", [True, True, False])):
                got = 0
                for i in range(k * 30):
                    if got >= k:
                        break
                    want_day = seq[got % len(seq)]
                    d = days[(got + i) % len(days)]
                    pool = [x for x in sids if smp[x][4] == d and is_day(smp[x][3]) == want_day and smp[x][2] >= length]
                    if not pool:
                        continue
                    sid = rng.choice(pool)
                    s = rng.randrange(0, smp[sid][2] - length + 1, 1000)
                    if free(sid, s, s + length):
                        taken[sid].append((s, s + length))
                        chosen.append((site, used, sid, s, s + length, f"{tag}{why}·{'白天' if want_day else '夜里'}", None))
                        got += 1

        for (site, used), by_dog in sorted(pools.items()):
            if not used and site != "gouchang":
                continue                      # 影棚 / 博朗旺 的「没用过」用 v1
            for did, sids in sorted(by_dog.items()):
                pick(site, used, did, sids, SEEN if used else FULL)

        for site in SITES:
            v1n = sum(1 for r in v1_rows.values() if r["site"] == SITES[site])
            for used in (False, True):
                cs = [c for c in chosen if c[0] == site and c[1] == used]
                if not cs and not (not used and v1n):
                    continue
                kinds = Counter(c[5].split("（")[0].split("·")[0].replace(SEEN_TAG, "").replace(UNSEEN_TAG, "") for c in cs)
                extra = f"（另有 v1 {v1n} 段）" if (not used and site != "gouchang") else ""
                print(f"{SITES[site]} {'用过' if used else '没用过'}：新挑 {len(cs)} 段{extra}，"
                      f"{sum(c[4] - c[3] for c in cs) / 3600_000:.1f} 小时，狗 {len({smp[c[2]][1].id for c in cs})} 只，{dict(kinds)}，"
                      f"日期 {min((str(smp[c[2]][4]) for c in cs), default='-')}~{max((str(smp[c[2]][4]) for c in cs), default='-')}")
        print(f"新挑合计 {len(chosen)} 段，{sum(c[4] - c[3] for c in chosen) / 3600_000:.1f} 小时视频")
        if not APPLY:
            print("（试运行，什么都没建）")
            return

        from app.api.v1.label_templates import apply_template
        from app.api.v1.projects import create_project
        from app.models.project import Project
        from app.models.task import TaskType
        from app.schemas.project import ProjectCreate
        for n in SET_NAME.values():
            if (await db.execute(select(ModelTestSet.id).where(ModelTestSet.name == n))).scalar():
                sys.exit(f"停：已经有测试集「{n}」")
        if (await db.execute(select(Project.id).where(Project.name == PROJECT))).scalar():
            sys.exit(f"停：已经有项目「{PROJECT}」")
        await create_project(ProjectCreate(
            name=PROJECT,
            description="模型测试·按场地（影棚 / 深圳雷诺 / 佛山博朗旺），分训练用过 / 没用过：每个任务是一个窗口，窗口里全标（休息、活动也标），"
                        "标完第二个人复核。提示窗口有疑似片段（以前人工标过的 / AI 预标的），随机窗口从零标。"), db, admin)
        p = (await db.execute(select(Project).where(Project.name == PROJECT))).scalar_one()
        await apply_template(3, p.id, db, admin)
        sets = {}
        for site, n in SET_NAME.items():
            st_ = ModelTestSet(name=n, created_by=admin.id,
                               note=f"{SITES[site]}：片段备注里【训练用过】/【训练没用过】是相对 #56/#61/#62。"
                                    + ("没用过的部分就是「测试集 v1」里这个场地的（项目「模型测试 v1」）；" if site != "gouchang" else "")
                                    + f"新挑的在项目「{PROJECT}」")
            db.add(st_)
            await db.flush()
            sets[site] = st_
            if site != "gouchang":
                for x in v1_items:
                    if v1_rows[x.id]["site"] == SITES[site]:
                        db.add(ModelTestItem(set_id=st_.id, task_id=x.task_id, sample_id=x.sample_id, start_ms=x.start_ms,
                                             end_ms=x.end_ms, note=UNSEEN_TAG + (x.note or ""), added_by=admin.id))
        n_c = Counter()
        for site, used, sid, s, e, why, tp in sorted(chosen, key=lambda c: (c[0], not c[1], str(smp[c[2]][4]), smp[c[2]][0].sample_code, c[3])):
            t = Task(project_id=p.id, sample_id=sid, task_type=TaskType.from_scratch, status=TaskStatus.PENDING_ASSIGN,
                     segment_start_ms=s, segment_end_ms=e, created_by=admin.id)
            db.add(t)
            await db.flush()
            db.add(ModelTestItem(set_id=sets[site].id, task_id=t.id, sample_id=sid, note=why, added_by=admin.id))
            if tp:
                for human, a, b, conf in ev.get((sid, tp), []):
                    if b <= s or a >= e:
                        continue
                    a2, b2 = max(a, s), min(b, e)
                    db.add(AiCandidate(task_id=t.id, round_no=t.round_no, label_name=tp, start_time_ms=a2, end_time_ms=b2,
                                       confidence=round(float(conf), 3) if conf is not None else None, reason="low_conf",
                                       model=CAND_MODEL, status=CandidateStatus.pending,
                                       evidence=(f"【模型测试提示】以前{'人工标过' if human else 'AI 预标报了'}「{tp}」（{(b2 - a2) / 1000:.1f} 秒）。"
                                                 "对着视频判：是就确认（起止可以调），不是就排除；窗口里别的时间也要标全")[:300]))
                    n_c[(tp, "人工" if human else "AI")] += 1
        await db.commit()
        print(f"建好：项目 #{p.id}「{PROJECT}」{len(chosen)} 个短任务；测试集 " + "、".join(f"#{s.id}{s.name}" for s in sets.values())
              + f"；疑似片段 {dict(n_c)}")


asyncio.run(main())
