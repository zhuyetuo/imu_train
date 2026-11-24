"""模型测试 v1：「抓挠提示」「未佩戴边界」两类窗口，把 train60 在原项目（337~341）预标的那几段
（抓挠 / 未佩戴，跟窗口重叠的部分）写成疑似片段，标注员确认 / 排除 / 调起止。随机窗口不加，保持从零标。
reason=low_conf（确认后写进任务标注），model 用单独的名字，整批可删：
  DELETE FROM ai_candidates WHERE model = 'train60·测试窗口提示' AND status = 'pending'
默认试运行；加 --apply 才写库。"""
import asyncio
import sys
from collections import Counter

sys.path.insert(0, "/app")
from sqlalchemy import select  # noqa: E402

from app.db.session import SessionLocal  # noqa: E402
from app.models.ai_candidate import AiCandidate, CandidateStatus  # noqa: E402
from app.models.annotation import AnnotationLabelItem, AnnotationRecord, LabelItemSource  # noqa: E402
from app.models.label import LabelDefinition  # noqa: E402
from app.models.model_test import ModelTestItem, ModelTestSet  # noqa: E402
from app.models.task import Task  # noqa: E402

MODEL = "train60·测试窗口提示"
SRC_PROJECTS = [337, 338, 339, 340, 341]
WANT = {"抓挠提示": "抓挠", "未佩戴开始边界": "未佩戴"}
APPLY = "--apply" in sys.argv


async def main():
    async with SessionLocal() as db:
        s = (await db.execute(select(ModelTestSet).where(ModelTestSet.name == "测试集 v1"))).scalar_one()
        items = (await db.execute(select(ModelTestItem).where(ModelTestItem.set_id == s.id))).scalars().all()
        lrows = (await db.execute(select(LabelDefinition.id, LabelDefinition.display_name, LabelDefinition.parent_id))).all()
        lname = {i: n for i, n, _ in lrows}; parent = {i: p for i, _, p in lrows}

        def top(l):
            while parent.get(l) is not None:
                l = parent[l]
            return lname.get(l)

        if (await db.execute(select(AiCandidate.id).where(AiCandidate.model == MODEL).limit(1))).first():
            sys.exit(f"停：已经有 model={MODEL} 的疑似片段，别重复写")
        n_win, n_cand, kinds, empty = Counter(), Counter(), Counter(), Counter()
        for it in items:
            kind = next((k for k in WANT if (it.note or "").startswith(k)), None)
            if kind is None:
                continue
            t = await db.get(Task, it.task_id)
            ws, we = t.segment_start_ms, t.segment_end_ms
            want = WANT[kind]
            rows = (await db.execute(
                select(AnnotationLabelItem.label_id, AnnotationLabelItem.start_time_ms, AnnotationLabelItem.end_time_ms,
                       AnnotationLabelItem.ai_confidence)
                .join(AnnotationRecord, AnnotationLabelItem.annotation_record_id == AnnotationRecord.id)
                .join(Task, Task.id == AnnotationRecord.task_id)
                .where(Task.sample_id == t.sample_id, Task.project_id.in_(SRC_PROJECTS),
                       AnnotationRecord.round_no == Task.round_no,
                       AnnotationLabelItem.source_type == LabelItemSource.ai_generated,
                       AnnotationLabelItem.start_time_ms < we, AnnotationLabelItem.end_time_ms > ws))).all()
            segs = [(lname.get(lid), max(a, ws), min(b, we), c) for lid, a, b, c in rows if top(lid) == want]
            n_win[kind] += 1
            if not segs:
                empty[kind] += 1
            for name, a, b, c in segs:
                n_cand[kind] += 1
                kinds[name] += 1
                if APPLY:
                    db.add(AiCandidate(
                        task_id=t.id, round_no=t.round_no, label_name=name, start_time_ms=int(a), end_time_ms=int(b),
                        confidence=round(float(c), 3) if c is not None else None, reason="low_conf", model=MODEL,
                        evidence=f"【模型测试提示】train60 在原项目预标里报了「{name}」（{(b - a) / 1000:.1f} 秒）。"
                                 "对着视频判：是就确认（起止可以调），不是就排除；窗口里别的时间也要标全"[:300],
                        status=CandidateStatus.pending))
        if APPLY:
            await db.commit()
        print("窗口", dict(n_win), "；疑似片段", dict(n_cand), "；窗口里没找到的", dict(empty))
        print("类别", dict(kinds.most_common()))
        print("已写库" if APPLY else "（试运行，没写库）")


asyncio.run(main())
