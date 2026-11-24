"""删掉项目 #342/#343 里那 53 个空短任务（已从测试集拿掉）。删前逐个核对：在这两个项目、待分配、
没有标注片段、没有疑似片段、不在任何测试集里；有一条不符就整批不删。走平台自己的 _delete_tasks。
加 --apply 才删。"""
import asyncio
import sys

sys.path.insert(0, "/app")
from sqlalchemy import func, select  # noqa: E402

from app.api.v1.tasks import _delete_tasks  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord  # noqa: E402
from app.models.model_test import ModelTestItem  # noqa: E402
from app.models.task import Task, TaskStatus  # noqa: E402

IDS = [74163, 74173, 74286, 74300, 74344, 74491, 74499, 74500, 74503, 74520, 74553, 74557, 74573, 74579, 74580, 74583,
       74589, 74590, 74591, 74593, 74599, 74602, 74604, 74606, 74611, 74613, 74619, 74626, 74628, 74631, 74632, 74634,
       74636, 74640, 74646, 74647, 74648, 74649, 74651, 74652, 74654, 74655, 74656, 74657, 74669, 74680, 74687, 74691,
       74694, 74705, 74714, 74726, 74728]
APPLY = "--apply" in sys.argv


async def main():
    assert len(IDS) == len(set(IDS)) == 53
    async with SessionLocal() as db:
        bad = []
        for tid in IDS:
            t = await db.get(Task, tid)
            if t is None:
                bad.append((tid, "不存在"))
                continue
            n_items = (await db.execute(select(func.count(AnnotationLabelItem.id))
                                        .join(AnnotationRecord, AnnotationLabelItem.annotation_record_id == AnnotationRecord.id)
                                        .where(AnnotationRecord.task_id == tid))).scalar()
            n_c = (await db.execute(select(func.count(AiCandidate.id)).where(AiCandidate.task_id == tid))).scalar()
            n_t = (await db.execute(select(func.count(ModelTestItem.id)).where(ModelTestItem.task_id == tid))).scalar()
            if t.project_id not in (342, 343) or t.status != TaskStatus.PENDING_ASSIGN or t.segment_start_ms is None or n_items or n_c or n_t:
                bad.append((tid, t.project_id, t.status.value, n_items, n_c, n_t))
        if bad:
            sys.exit(f"停：有不符合的，整批不删：{bad}")
        print("53 个都核对过：在 #342/#343、待分配、短任务、没片段、没疑似片段、不在测试集")
        if not APPLY:
            print("（试运行，没删）")
            return
        n = await _delete_tasks(db, IDS)
        print("删了", n, "个")
        left = (await db.execute(select(func.count(Task.id)).where(Task.id.in_(IDS)))).scalar()
        print("复查：还剩", left, "个")


asyncio.run(main())
