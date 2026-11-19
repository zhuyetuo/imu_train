"""挑一只狗的一天：每分钟走了多少秒、三档智能打点和每 5 分钟固定打点各在什么时候，输出给网页画图。"""
import json, os, sys, importlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
res = json.load(open(os.path.expanduser("~/claude_exp/gnss/sim_result_省电.json")))
res = [r for r in res if r["worn_h"] >= 23]
res.sort(key=lambda r: r["moving_min"])
pick = res[int(len(res) * 0.75)]           # 偏活跃的一天（四分之三位）
print("挑的：", pick["day"], pick["dog"], pick["moving_min"], "分钟在走")
out = {"day": pick["day"], "dog": pick["dog"], "moving_min": pick["moving_min"], "presets": {}}
for preset in ("省电", "标准", "精细"):
    os.environ["PRESET"] = preset
    import sim_days; importlib.reload(sim_days)
    day = pick["day"]; pair = tuple(int(x[3:]) for x in pick["dog"].split("+"))
    sim_days.dog_day((day, pair))
    fixes, sus = sim_days._LAST["fixes"], sim_days._LAST["sus"]
    out["presets"][preset] = [[int(t), r] for t, r in fixes]
    if "moving_per_min" not in out:
        out["moving_per_min"] = [int(x) for x in sus[: 1440 * 60].reshape(1440, 60).sum(1)]
print({k: len(v) for k, v in out["presets"].items()})
json.dump(out, open(os.path.expanduser("~/claude_exp/gnss/timeline_day.json"), "w"), ensure_ascii=False)
