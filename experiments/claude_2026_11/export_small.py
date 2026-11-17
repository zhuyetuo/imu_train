"""在 label-service 容器里跑：把 #40 / #41（宿主机 GPU 训的 GR5513 小模型）导成端侧模型 train40 / train41，
结果写进任务的 edge 字段，再让端侧服务挂上。跟 jobs.export_edge 做的一样，只是处理目录直接给（不走 train_custom 的目录约定）。"""
import json
import os
import subprocess
import sys

sys.path.insert(0, "/app")
from label_service import config, jobs as J  # noqa: E402

for jid, run in ((40, "pool_20261116_all__cnn_w8s_6ax_pub"), (41, "pool_20261116_all__cnn_w8s_3ax_pub")):
    job = J.get_job(jid)
    script = os.path.join(config.ALGO_TINYML_DIR, "service", "export_train.py")
    cmd = [sys.executable, script, "--tag", J.model_tag(jid), "--model", job["model_path"],
           "--processed-dir", f"/app/data/processed_{run}", "--imu-train", config.REPO_ROOT]
    print("▶", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=config.ALGO_TINYML_DIR, timeout=1800)
    with open(J._log_path(jid), "a", encoding="utf-8") as f:
        f.write("\n▶ 导出到端侧\n" + r.stdout + r.stderr)
    print(r.stdout[-1500:], r.stderr[-800:], flush=True)
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("EXPORT_RESULT ")), None)
    if r.returncode != 0 or not line:
        sys.exit(f"#{jid} 导出失败")
    res = json.loads(line[len("EXPORT_RESULT "):])
    res["reloaded"], res["reload_error"] = J.reload_edge_service()
    job = J.get_job(jid)
    job["edge"] = res
    J._save(job)
    print(f"#{jid} → {res.get('spec')}  reloaded={res['reloaded']} {res['reload_error'] or ''}", flush=True)
