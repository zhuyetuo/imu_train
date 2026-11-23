"""从 spec_base.json（= 训练记录 #61 / job60 的配置）派生 v1 / v2b / v3 的配置：只改窗口和数据集，别的一模一样。"""
import copy
import json

base = json.load(open("/home/toky/claude_exp/evt/spec_base.json", encoding="utf-8"))
pool = [d for d in base["datasets"] if d["name"].startswith("pool_")]
old = [d for d in base["datasets"] if d["name"].startswith("ds_old")]
pub = [d for d in base["datasets"] if d["name"].startswith("pub_")]
assert len(pool) == 1 and len(old) == 3 and len(pub) == 2, [d["name"] for d in base["datasets"]]
for name, ds in [("v1", pool), ("v2b", pool + old), ("v3", pool + old + pub)]:
    s = copy.deepcopy(base)
    s["datasets"] = copy.deepcopy(ds)
    s["window"] = int(8.0 * s["hz"])
    s["stride_s"] = 1.0
    s["run_name"] = f"evt_cmp_{name}"
    json.dump(s, open(f"/home/toky/claude_exp/evt/spec_{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(name, s["window"], s["stride_s"], [d["name"] for d in ds])
