"""label-service 容器里跑：把宿主机上训好的 v2b（8 s / 1 s，池子 + 老数据只取抓挠，不加公开数据）登记成训练任务（done）。
配置用 #56（job56，同一组数据集、同一窗口）的 dataset_spec 做底，跟平台上点训练出来的一样的字段。"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, "/app")
from label_service import jobs as J  # noqa: E402

RUN = "evt_cmp_v2b"
TAG = "v2b_8s_evt_20261124"
MP = f"/app/results/processed_{RUN}/20hz_v2/cnn/dl_cnn_best.pt"
LOG = f"/app/tmp/import_logs/{TAG}.log"
if not os.path.exists(MP):
    sys.exit(f"模型不在：{MP}")
if any(j.get("model_path") == MP for j in (J.get_job(i) or {} for i in range(1, J._next_job_id()))):
    sys.exit("这个模型已经登记过了")
spec = dict(J.get_job(56)["dataset_spec"])
jid = J._next_job_id()
now = int(time.time())
job = {
    "job_id": jid, "run_date": RUN, "status": J.STATUS_DONE, "dataset_spec": spec,
    "model_type": "cnn", "tag": TAG, "model_version": TAG, "model_path": MP,
    "command": "python src/ml/train_v2.py --spec experiments/claude_2026_11/evt/spec_v2b.json --workers 10",
    "metrics": None, "error": None, "log_path": J._log_path(jid),
    "created_at": now - 264, "started_at": now - 264, "finished_at": now,
    "imported": "宿主机 GPU 训练后导入（同一份 train_v2）。2026-11-24 抓挠事件级对比里最好的一版：同召回下确定误报比 #61 少约 1/4–1/3",
}
metrics = J._load_metrics(MP)
try:
    from label_service import dataset_stats
    metrics["dataset"] = dataset_stats.compute(spec, [RUN], *J.v2_stats_geom(spec))
except Exception as e:  # noqa: BLE001
    print(f"  训练集统计跳过：{e}")
job["metrics"] = metrics
shutil.copy(LOG, J._log_path(jid))
J._save(job)
json.dump([{"job_id": jid, "tag": TAG, "model_type": "cnn", "spec": spec}], open("/app/tmp/import_v2b_done.json", "w", encoding="utf-8"),
          ensure_ascii=False)
print(f"#{jid} {TAG} {MP}  macro F1 {metrics.get('macro_f1')}")
