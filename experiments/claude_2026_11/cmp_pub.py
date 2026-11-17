"""加公开数据前后对比：总的片段级指标 + 只看我们自己数据池那份（per_dataset）"""
import json
import os
import sys

R = os.path.expanduser("~/claude_exp/imu_train_wt/results")
for b in sys.argv[1:]:
    kind = "xgb" if "xgb" in b else "cnn"
    fn = "ml_xgb.json" if kind == "xgb" else "dl_cnn.json"
    for tag in (b, b + "_pub"):
        p = f"{R}/processed_pool_20261116_all__{tag}/20hz_v2/{kind}/{fn}"
        if not os.path.exists(p):
            print(tag, "（还没训完）")
            continue
        r = json.load(open(p, encoding="utf-8"))
        ps = r.get("postprocess_segments") or {}
        sc, sh = ps.get("抓挠", {}), ps.get("甩头/抖身", {})
        pool = (r.get("per_dataset") or {}).get("pool_20261116_all", {})
        print(f"{tag:<18} 全部: 抓挠 命中 {sc.get('recall_hit')} 查准 {sc.get('precision')} 误报/h {sc.get('fp_per_h_other')} "
              f"| 甩头 n={sh.get('n_gt')} 命中 {sh.get('recall_hit')} 查准 {sh.get('precision')}")
        keep = {k: v for k, v in pool.items() if k not in ("confusion",)}
        print(f"{'':<18} 只看数据池: {json.dumps(keep, ensure_ascii=False)[:600]}")
