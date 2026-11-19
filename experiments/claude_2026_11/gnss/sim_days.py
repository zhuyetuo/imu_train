"""在狗场真实全天数据上模拟「IMU 智能打点」，跟固定间隔打点比打点次数 / 估算耗电 / 最大漏追距离。
一只狗两个项圈轮换：每秒取戴着的那个。只用加速度 3 轴（不读陀螺仪）。"""
import glob, json, os, sys, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from locomotion import steps_and_loco, HZ

ROOT = os.path.expanduser("~/ai_data/data_raw")
DAYS = ["2026_10_27", "2026_10_28", "2026_10_29", "2026_10_30", "2026_10_31", "2026_11_1"]
DOGS = [(9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20)]

# 打点策略参数：三档，用户可调，默认「省电」。板上就是一张同样字段的表
PRESETS = {
    # 只想看大概去了哪：少打点、最省电
    "省电": dict(depart_min_m=20, move_dist_m=100, move_max_s=300, arrive_min_m=20, heartbeat_s=4 * 3600),
    "标准": dict(depart_min_m=10, move_dist_m=50, move_max_s=120, arrive_min_m=10, heartbeat_s=2 * 3600),
    # 想看小公园里怎么跑的：走 15 米就打一个、最长 30 秒一个
    "精细": dict(depart_min_m=0, move_dist_m=15, move_max_s=30, arrive_min_m=5, heartbeat_s=3600),
}
P = dict(sus_win=15, sus_min=10,          # 持续移动：15 秒里至少 10 秒在走/跑
         rest_before_depart=60,           # 静止至少 60 秒后再动起来，算「出发」
         dist_step_m=0.6,                 # 每检测到一步估多少米（要用真实 GNSS 标定）
         stop_s=60,                       # 停下 60 秒算到达
         **PRESETS[os.environ.get("PRESET", "省电")])
# 耗电假设（换成实际模组手册的数）：离上次定位越久，热 / 温 / 冷启动越慢
FIX_MA = 25.0
def fix_seconds(gap_s):
    return 5 if gap_s < 300 else (25 if gap_s < 4 * 3600 else 35)

def per_second(path):
    """一小时文件 → (起始秒, 每秒是否在走, 每秒步数, 每秒是否戴着)"""
    try:
        df = pd.read_csv(path, usecols=["timestamp", "acc_x", "acc_y", "acc_z"])
    except Exception:
        return None
    t = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[t.notna()]; t = t[t.notna()]
    if len(df) < HZ * 60:
        return None
    df = df.assign(t=t).drop_duplicates("t")
    ts = (df["t"] - df["t"].iloc[0].normalize()).dt.total_seconds().to_numpy()
    g = np.arange(np.ceil(ts[0]), ts[-1], 1 / HZ)
    acc = np.stack([np.interp(g, ts, df[c].to_numpy()) for c in ("acc_x", "acc_y", "acc_z")], 1)
    # 全 0 占位（蓝牙断了）当缺数据
    bad = (np.abs(acc).sum(1) == 0)
    step, loco, cad = steps_and_loco(acc)
    n = len(loco)
    m = np.sqrt((acc ** 2).sum(1))[: n * HZ].reshape(n, HZ)
    worn = (m.std(1) > 1e-4) & ~bad[: n * HZ].reshape(n, HZ).any(1)
    # 没戴着：放桌上 / 充电，几秒窗口内几乎没有起伏。放宽到 60 秒都没起伏才算没戴
    w = np.convolve(worn.astype(int), np.ones(60, int), mode="same") > 0
    steps = step[: n * HZ].reshape(n, HZ).sum(1)
    return int(g[0]), loco, steps, w

def dog_day(args):
    day, pair = args
    L = 86400
    loco = np.zeros(L, bool); steps = np.zeros(L); worn = np.zeros(L, bool); act = np.full(L, -1.0)
    for imu in pair:
        for f in sorted(glob.glob(f"{ROOT}/{day}_bolangwang/*_imu{imu}_raw.csv")):
            r = per_second(f)
            if r is None: continue
            s0, lo, st, w = r
            s1 = min(L, s0 + len(lo)); k = s1 - s0
            if k <= 0: continue
            sc = st[:k] + 10 * lo[:k] + 100 * w[:k]   # 两个项圈同一秒都有数据：取「戴着、在动」的那个
            better = sc > act[s0:s1]
            idx = np.arange(s0, s1)[better]
            loco[idx] = lo[:k][better]; steps[idx] = st[:k][better]; worn[idx] = w[:k][better]; act[idx] = sc[better]
    have = act >= 0
    sus = np.convolve(loco.astype(int), np.ones(P["sus_win"], int))[:L] >= P["sus_min"]
    sus &= worn
    # ── 智能打点 ──
    fixes = []  # (秒, 原因)
    last_fix = None; dist_since = 0.0; still_run = 10 ** 9; moving = False; stop_run = 0; pending_depart = False; depart_dist = 0.0; departed = False
    for s in range(L):
        if not have[s] or not worn[s]:
            still_run += 1; continue
        if sus[s]:
            dist_since += steps[s] * P["dist_step_m"]
            if not moving:
                moving = True; stop_run = 0; departed = False
                pending_depart = still_run >= P["rest_before_depart"]
                depart_dist = 0.0
            if pending_depart:
                depart_dist += steps[s] * P["dist_step_m"]
                if depart_dist >= P["depart_min_m"]:
                    fixes.append((s, "出发")); last_fix = s; dist_since = 0.0; pending_depart = False; departed = True
            elif not pending_depart and dist_since >= P["move_dist_m"] or (last_fix is not None and s - last_fix >= P["move_max_s"]):
                fixes.append((s, "移动中")); last_fix = s; dist_since = 0.0
            still_run = 0
        else:
            still_run += 1
            if moving:
                stop_run += 1
                if stop_run >= P["stop_s"]:
                    moving = False; pending_depart = False
                    if departed and dist_since >= P["arrive_min_m"]:
                        fixes.append((s, "到达")); last_fix = s; dist_since = 0.0
            if not moving and (last_fix is None or s - last_fix >= P["heartbeat_s"]):
                fixes.append((s, "保底")); last_fix = s
    def energy(times):
        e, prev = 0.0, None
        for t in times:
            e += FIX_MA * fix_seconds(10 ** 9 if prev is None else t - prev) / 3600; prev = t
        return e  # mAh
    worn_s = int((have & worn).sum())
    out = {"day": day, "dog": f"IMU{pair[0]}+IMU{pair[1]}", "worn_h": round(worn_s / 3600, 1),
           "moving_min": round(int(sus.sum()) / 60, 1), "est_dist_m": round(float((steps * sus).sum() * P["dist_step_m"])),
           "smart_fixes": len(fixes), "smart_by": dict(collections.Counter(r for _, r in fixes)),
           "smart_mAh": round(energy([t for t, _ in fixes]), 2)}
    worn_idx = np.flatnonzero(have & worn)
    cum = np.cumsum(steps * sus * P["dist_step_m"])
    for iv in (60, 300, 600):
        ts = worn_idx[::iv] if len(worn_idx) else []
        out[f"fixed{iv // 60}m_fixes"] = len(ts); out[f"fixed{iv // 60}m_mAh"] = round(energy(list(ts)), 2)
        gaps = np.diff(cum[ts]) if len(ts) > 1 else np.array([0])
        out[f"fixed{iv // 60}m_max_gap_m"] = round(float(gaps.max()) if len(gaps) else 0)
    ft = [t for t, _ in fixes]
    g = np.diff(cum[ft]) if len(ft) > 1 else np.array([0])
    out["smart_max_gap_m"] = round(float(g.max()) if len(g) else 0)
    return out

if __name__ == "__main__":
    jobs = [(d, p) for d in DAYS for p in DOGS]
    with ProcessPoolExecutor(int(os.environ.get("NJ", 6))) as ex:
        res = list(ex.map(dog_day, jobs))
    json.dump(res, open(os.path.expanduser(f"~/claude_exp/gnss/sim_result_{os.environ.get('PRESET', '省电')}.json"), "w"), ensure_ascii=False, indent=1)
    df = pd.DataFrame(res); df = df[df.worn_h > 6]
    print(f"{len(df)} 个狗·天（戴着 >6 小时）；参数 {P}")
    cols = ["worn_h", "moving_min", "est_dist_m", "smart_fixes", "smart_mAh", "smart_max_gap_m",
            "fixed1m_fixes", "fixed1m_mAh", "fixed5m_fixes", "fixed5m_mAh", "fixed5m_max_gap_m", "fixed10m_fixes", "fixed10m_mAh", "fixed10m_max_gap_m"]
    print(df[cols].describe().loc[["mean", "50%", "min", "max"]].round(1).to_string())
    by = collections.Counter()
    for r in df.smart_by: by.update(r)
    print("智能打点按原因合计：", dict(by))
