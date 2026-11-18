"""B 组统计：按（日期, IMU）汇总人工 / 新模型 / 老模型的抓挠次数；再看人工没标抓挠的任务里，新老模型每小时报几次（误报的上限）"""
import json
import statistics as st
from collections import defaultdict

rs = [json.loads(l) for l in open("groupB_res.jsonl", encoding="utf-8")]
ok = [r for r in rs if r.get("ok") and "new" in r and r.get("old") is not None]
fail = [r for r in rs if not r.get("ok")]
print(f"已跑 {len(rs)}，可比 {len(ok)}，失败 {len(fail)}" + (f"（例：{fail[0].get('error', '')[:80]}）" if fail else ""))
day = defaultdict(lambda: defaultdict(float))
for r in ok:
    k = (r["date"], r["imu"])
    day[k]["h"] += len(r["human"]); day[k]["new"] += len(r["new"]); day[k]["old"] += len(r["old"]); day[k]["hours"] += r.get("hours", 0)
days = [v for v in day.values() if v["h"] > 0]
print(f"（日期, IMU）{len(day)} 个，有人工抓挠的 {len(days)} 个")
for name in ("old", "new"):
    rn = [v[name] / v["h"] for v in days]
    q = st.quantiles(rn, n=4)
    print(f"  {'老模型' if name == 'old' else '新模型（train37 精准版）'}：每天次数 / 人工 中位 {st.median(rn):.2f}（四分位 {q[0]:.2f} ~ {q[2]:.2f}），±50% 以内 {sum(1 for x in rn if 0.67 <= x <= 1.5) / len(rn):.0%}")
zero = [r for r in ok if not r["human"]]
hz = sum(r.get("hours", 0) for r in zero)
print(f"人工没标抓挠的任务 {len(zero)} 个、{hz:.0f} h：新模型报 {sum(len(r['new']) for r in zero)} 段（{sum(len(r['new']) for r in zero) / max(hz, 1e-9):.2f}/h），"
      f"老模型 {sum(len(r['old']) for r in zero)} 段（{sum(len(r['old']) for r in zero) / max(hz, 1e-9):.2f}/h）")
withh = [r for r in ok if r["human"]]
hh = sum(r.get("hours", 0) for r in withh)
print(f"有人工抓挠的任务 {len(withh)} 个、{hh:.0f} h：人工 {sum(len(r['human']) for r in withh)} 段，新 {sum(len(r['new']) for r in withh)}，老 {sum(len(r['old']) for r in withh)}")
