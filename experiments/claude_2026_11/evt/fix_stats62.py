"""label-service 容器里跑：#62 的训练集统计按 job56 同一组 merged_tmp.json 重算（数据集一样），写回任务。"""
import os
import sys

sys.path.insert(0, "/app")
from label_service import config, dataset_stats, jobs as J  # noqa: E402

job = J.get_job(62)
spec = job["dataset_spec"]
raw = os.path.join(config.REPO_ROOT, "data", "raw_custom")
names = sorted(n for n in os.listdir(raw) if n.endswith("__job56") and os.path.isfile(os.path.join(raw, n, "merged_tmp.json")))
# 主数据集只取一份（__job56 那份就是同一个导出拷过去的）
if "pool_latest_20261120_165839__job56" in names and "pool_latest_20261120_165839" in names:
    names.remove("pool_latest_20261120_165839")
print("用这几份：", names)
st = dataset_stats.compute(spec, names, *J.v2_stats_geom(spec))
job["metrics"]["dataset"] = st
J._save(job)
print({k: st.get(k) for k in ("n_tasks", "n_days", "total_seconds", "window_s", "stride_s")})
j56 = (J.get_job(56).get("metrics") or {}).get("dataset") or {}
print("对照 #56：", {k: j56.get(k) for k in ("n_tasks", "n_days", "total_seconds")})
