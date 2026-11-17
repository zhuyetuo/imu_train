"""项目 325 那 18 个任务：疑似抓挠按 2026-11-16 那次预标注（稳定版 v2 = viterbi，默认模型）存下的结果原样取回。
结果 JSON 按 (样本, 模型, 版本) 存在 NAS（sample_inference_runs.json_path），里面有当时的 candidates。
只换 low_conf / spectral 且没判过的；负例候选、别的规则候选、判过的、人工片段不碰。"""
import asyncio
import collections
import json
import os

from sqlalchemy import select

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.ai_candidate import AiCandidate, CandidateStatus
from app.models.annotation import AnnotationLabelItem, AnnotationRecord
from app.models.inference_run import SampleInferenceRun
from app.models.label import LabelDefinition
from app.models.sample import Sample
from app.models.task import Task
from app.services import ai_prelabel_service as aps

T = [67756, 67758, 67760, 67762, 67768, 67769, 67776, 67778, 67781, 67785, 67793, 67813, 67817, 67825, 67830, 67845, 67878, 67922]
SCRATCH = ("low_conf", "spectral")
MODEL_PATH = "/app/results/processed_2026_8_11-2026_8_27_raw_missing_drop_window/16hz_remap_custom_3class/rf/ml_rf.pkl"


async def main():
    removed = written = 0
    for tid in T:
        async with SessionLocal() as db:
            task = await db.get(Task, tid)
            sample = await db.get(Sample, task.sample_id)
            runs = (await db.execute(select(SampleInferenceRun).where(
                SampleInferenceRun.sample_id == sample.id, SampleInferenceRun.mode == "viterbi"))).scalars().all()
        run = next((r for r in runs if r.model_path == MODEL_PATH), None) or (runs[0] if len(runs) == 1 else None)
        if run is None:
            print(f"任务 {tid}: 找不到 11-16 那次的结果（viterbi 有 {[(r.model_tag, r.model_path) for r in runs]}），不动", flush=True)
            continue
        raw = json.load(open(os.path.join(settings.nas_root, run.json_path), encoding="utf-8"))
        csv_start = await aps._csv_start_of(sample)
        new = [c for c in aps.flatten_candidates(raw.get("candidates") or [], csv_start) if c.reason in SCRATCH]
        async with SessionLocal() as db:
            task = await db.get(Task, tid)
            rows = (await db.execute(select(AiCandidate).where(
                AiCandidate.task_id == tid, AiCandidate.round_no == task.round_no))).scalars().all()
            gone = [o for o in rows if o.status == CandidateStatus.pending and o.reason in SCRATCH]
            for o in gone:
                await db.delete(o)
            taken = [(o.label_name, o.start_time_ms, o.end_time_ms) for o in rows if o.status != CandidateStatus.pending]
            rec = (await db.execute(select(AnnotationRecord).where(
                AnnotationRecord.task_id == tid, AnnotationRecord.round_no == task.round_no))).scalars().first()
            if rec is not None:
                items = (await db.execute(select(AnnotationLabelItem).where(
                    AnnotationLabelItem.annotation_record_id == rec.id))).scalars().all()
                ids = {i.label_id for i in items}
                names = dict((await db.execute(select(LabelDefinition.id, LabelDefinition.display_name).where(
                    LabelDefinition.id.in_(ids)))).all()) if ids else {}
                taken += [(names.get(i.label_id, ""), i.start_time_ms, i.end_time_ms) for i in items]
            n = 0
            for c in new:
                if any(aps._same_kind(c.label_name, lab) and c.start_time_ms < e and c.end_time_ms > s for lab, s, e in taken):
                    continue
                db.add(AiCandidate(task_id=tid, round_no=task.round_no, label_name=c.label_name,
                                   start_time_ms=c.start_time_ms, end_time_ms=c.end_time_ms,
                                   confidence=c.confidence, spec=c.spec, reason=c.reason))
                n += 1
            await db.commit()
        removed += len(gone)
        written += n
        print(f"任务 {tid}: 结果 {run.json_path.split('/')[-1]}（{run.updated_at}），换掉 {len(gone)} 条，取回 {n} 条", flush=True)
    async with SessionLocal() as db:
        rows = (await db.execute(select(AiCandidate).where(AiCandidate.task_id.in_(T)))).scalars().all()
        print("合计：换掉", removed, "取回", written, flush=True)
        print("现在", dict(collections.Counter((r.reason, r.status.value) for r in rows)), flush=True)
    print("DONE", flush=True)

asyncio.run(main())
