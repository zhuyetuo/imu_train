"""api 容器里跑：照项目页「按日期新建」的做法建 11-19 ~ 11-23 五个项目（套模板 3「犬行为全量」、
任务类型 AI 预标注+人工修改、指派给 1 号），用 srv:train60@viterbi_strict 批量预标，等预标跑完再退出
（预标是这个进程里的后台任务，进程退出就断了）。加 --apply 才动手。"""
import asyncio
import sys
from datetime import date

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.api.v1.label_templates import apply_template  # noqa: E402
from app.api.v1.projects import create_project  # noqa: E402
from app.api.v1.tasks import bulk_create_tasks  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models.project import Project  # noqa: E402
from app.models.sample import Sample  # noqa: E402
from app.models.task import Task  # noqa: E402
from app.models.user import User  # noqa: E402
from app.schemas.project import ProjectCreate  # noqa: E402
from app.schemas.task import BulkTaskCreate  # noqa: E402
from app.services import ai_prelabel_service as aps, algo_client  # noqa: E402

MODE = "srv:train60@viterbi_strict"
TEMPLATE = 3
ASSIGNEE = 1
DATES = [date(2026, 11, d) for d in range(19, 24)]
APPLY = "--apply" in sys.argv


async def main():
    assert algo_client.post_mode_of(MODE) == "viterbi_strict" and algo_client.model_of(MODE) == "train60"
    tags = [m["tag"] for m in await algo_client.models()]
    assert "train60" in tags, tags
    async with SessionLocal() as db:
        admin = await db.get(User, 1)
        for d in DATES:
            name = d.isoformat()
            sids = (await db.execute(select(Sample.id).where(Sample.session_date == d, Sample.imu_csv_path.is_not(None)))).scalars().all()
            busy = (await db.execute(select(Task.id).where(Task.sample_id.in_(sids)).limit(1))).scalar()
            exists = (await db.execute(select(Project.id).where(Project.name == name))).scalar()
            print(f"{name}: 样本 {len(sids)}，已有项目 {exists}，样本已在别的任务里 {bool(busy)}", flush=True)
            if not APPLY or exists or busy or not sids:
                continue
            p = (await create_project(ProjectCreate(name=name, description=""), db, admin))
            pid = (p.body if hasattr(p, "body") else p)
            pid = (await db.execute(select(Project.id).where(Project.name == name))).scalar()
            await apply_template(TEMPLATE, pid, db, admin)
            r = await bulk_create_tasks(BulkTaskCreate(project_id=pid, sample_ids=list(sids), task_type="ai_assisted",
                                                       assigned_to=ASSIGNEE, infer_mode=MODE), db, admin)
            print(f"  建了项目 {pid}，{r.body.decode()[:120] if hasattr(r, 'body') else r}", flush=True)
    if not APPLY:
        print("（试运行，什么都没建）")
        return
    while aps._running:
        await asyncio.sleep(30)
        print("预标进行中：", {pid: f"{p.processed}/{p.total}" for pid, p in aps._progress.items()
                              if getattr(p, "status", "") == "running"}, flush=True)
    for pid, p in aps._progress.items():
        print("完成", pid, p.status, f"成功 {p.succeeded} 失败 {p.failed} 跳过 {p.skipped}", flush=True)


asyncio.run(main())
