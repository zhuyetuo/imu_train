"""把宿主机 GPU 上用同一份 train_v2 训好的模型，登记成 label_service 的训练任务（状态 done）。
跟平台上点「训练」跑出来的任务一模一样：同一套 json 字段、模型在 /app/results 下、日志在 jobs/{id}.log。
在 label-service 容器里跑：python import_jobs.py items.json"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, "/app")
from label_service import jobs as J  # noqa: E402

items = json.load(open(sys.argv[1], encoding="utf-8"))
base = J.get_job(31)["dataset_spec"]          # 平台上那次 cnn 训练（数据池全部）的参数做底
now = int(time.time())
out = []
for it in items:
    jid = J._next_job_id()
    spec = {**base, **it["spec"]}
    mp = it["model_path"]
    if not os.path.exists(mp):
        sys.exit(f"模型不在：{mp}")
    job = {
        "job_id": jid, "run_date": it["run"], "status": J.STATUS_DONE, "dataset_spec": spec,
        "model_type": it["model_type"], "tag": it["tag"], "command": it["command"],
        "model_version": it["tag"], "model_path": mp, "metrics": None, "error": None,
        "log_path": J._log_path(jid), "created_at": now - it["secs"], "started_at": now - it["secs"],
        "finished_at": now, "imported": "宿主机 GPU 训练后导入（同一份 train_v2，容器里只有 CPU）",
    }
    metrics = J._load_metrics(mp)
    try:
        from label_service import dataset_stats
        metrics["dataset"] = dataset_stats.compute(spec, [it["run"]], *J.v2_stats_geom(spec))
    except Exception as e:  # noqa: BLE001
        print(f"  训练集统计跳过：{e}")
    job["metrics"] = metrics
    if it.get("edge"):
        job["edge"] = it["edge"]
    shutil.copy(it["log"], J._log_path(jid))
    J._save(job)
    out.append({"job_id": jid, "tag": it["tag"], "model_type": it["model_type"], "spec": spec})
    print(f"#{jid} {it['tag']}  {mp}")
json.dump(out, open(sys.argv[1].replace(".json", "_done.json"), "w", encoding="utf-8"), ensure_ascii=False)
