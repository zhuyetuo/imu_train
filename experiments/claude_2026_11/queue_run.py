"""Experiment queue: runs jobs with resource limits (GPU slots / CPU slots), each job = compare3 (+ eval_events).
Keeps the GPU and CPU busy for hours without supervision; status in queue/status.json, logs in queue/<name>.log,
summary (re-written after every finished job) in queue/summary.md.

Usage: nohup python queue_run.py > queue/queue.log 2>&1 &
"""
import json, os, subprocess, time

HOME = os.path.expanduser("~/claude_exp")
Q = os.path.join(HOME, "queue")
PY = os.path.expanduser("~/miniconda3/bin/python")
os.makedirs(Q, exist_ok=True)
GPU_SLOTS, CPU_SLOTS = 3, 2

BASE = "--axes 3 --epochs 30"
K7W = "--cnn_filters 64,128,128 --cnn_kernel 7"
jobs = []


def cnn(name, extra, init=None, deps=()):
    j = {"name": name, "res": "gpu", "deps": list(deps),
         "cmd": f"{PY} compare3.py {BASE} --methods cnn --out q_{name} {extra}" + (f" --cnn_init {init}" if init else "")
                + f" && {PY} eval_events.py --dir q_{name} --methods cnn --workers 6"}
    jobs.append(j)


def xgb(name, extra):
    jobs.append({"name": name, "res": "cpu", "deps": [],
                 "cmd": f"{PY} compare3.py {BASE} --methods xgb --out q_{name} {extra}"
                        f" && {PY} eval_events.py --dir q_{name} --methods xgb --workers 6"})


# 1) repeatability of the current default (k7 wide) and the old default — differences of 2–3 points need this
for s in (1, 2, 3):
    cnn(f"k7wide_r{s}", K7W)
for s in (1, 2):
    cnn(f"base_r{s}", "")
# 2) longer context: 6 s and 8 s windows (may separate sustained scratching from intermittent licking/chewing)
for w, tag in ((120, "w6"), (160, "w8")):
    for s in (1, 2):
        cnn(f"k7wide_{tag}_r{s}", f"{K7W} --window {w}")
# 3) architecture neighbours
cnn("k9wide", "--cnn_filters 64,128,128 --cnn_kernel 9")
cnn("k7w96", "--cnn_filters 96,128,128 --cnn_kernel 7")
cnn("k7w192", "--cnn_filters 64,128,192 --cnn_kernel 7")
cnn("k9wide_w6", "--cnn_filters 64,128,128 --cnn_kernel 9 --window 120")
# 4) larger self-supervised pretraining matched to the new architecture, then fine-tune (repeat twice)
jobs.append({"name": "ssl_k7wide", "res": "gpu", "deps": [],
             "cmd": f"{PY} ssl_pretrain.py --exclude_ds pool_20261116_all --n_files 12000 --axes 3 --out queue/ssl_k7wide.pt"
                    f" --workers 10 --epochs 30 --filters 64,128,128 --kernel 7"})
for s in (1, 2):
    cnn(f"k7wide_ssl_r{s}", K7W, init="queue/ssl_k7wide.pt", deps=["ssl_k7wide"])
# 5) XGBoost reference variants on CPU
xgb("xgb50d6", "")
xgb("xgb100d6", "--xgb_rounds 100")
xgb("xgb150d6", "--xgb_rounds 150")
xgb("xgb50d6_w6", "--window 120")


def summarize():
    import numpy as np
    rows = []
    for j in jobs:
        f = os.path.join(HOME, f"q_{j['name']}", "events.json")
        if not os.path.exists(f):
            continue
        ev = json.load(open(f))
        for m in ("cnn", "xgb"):
            pts = sorted([(v["cv_FP_other_h"], v["cv_recall_hit"]) for k, v in ev.items() if k.startswith(m + "@")])
            if not pts:
                continue
            fp, rc = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            r3, r5 = np.interp(3, fp, rc), np.interp(5, fp, rc)
            fp90 = float(np.interp(0.9, rc, fp)) if rc.max() >= 0.9 else float("nan")
            w = json.load(open(os.path.join(HOME, f"q_{j['name']}", "window_metrics.json")))[m]
            rows.append((j["name"], m, r3, r5, fp90, w["scratch_window"]["P"], w["scratch_window"]["R"], w["acc"]))
    rows.sort(key=lambda r: -r[3])
    lines = ["# 实验队列汇总（自动生成）", "",
             "只用加速度、数据池全部（pool_20261116_all）、按（天, 项圈）分组 5 折；事件级 = 连续推理 + 稳定版 v2 后处理。",
             "检测率@误报 x = 误报 x 次/小时（人标别的动作的时间里）时，人标抓挠被检测到的比例（插值）。", "",
             "| 实验 | 方法 | 检测率@误报3 | 检测率@误报5 | 检测率90%时误报 | 窗口抓挠 P | 窗口抓挠 R | 窗口准确率（无门） |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.4f} |")
    st = json.load(open(os.path.join(Q, "status.json"))) if os.path.exists(os.path.join(Q, "status.json")) else {}
    lines += ["", f"状态：{sum(1 for v in st.values() if v.get('state') == 'done')} 完成 / "
                  f"{sum(1 for v in st.values() if v.get('state') == 'failed')} 失败 / {len(jobs)} 个任务；更新于 {time.strftime('%m-%d %H:%M')}"]
    open(os.path.join(Q, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    status = {j["name"]: {"state": "pending"} for j in jobs}
    running = {}
    def save():
        json.dump(status, open(os.path.join(Q, "status.json"), "w"), ensure_ascii=False, indent=1)
    save()
    while True:
        for n, (p, j, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                status[n] = {"state": "done" if rc == 0 else "failed", "rc": rc, "minutes": round((time.time() - t0) / 60, 1)}
                del running[n]; save()
                try:
                    summarize()
                except Exception as e:  # noqa: BLE001
                    print("summary failed", e, flush=True)
                print(time.strftime("%H:%M"), n, status[n], flush=True)
        used = {"gpu": sum(1 for _, j, _ in running.values() if j["res"] == "gpu"),
                "cpu": sum(1 for _, j, _ in running.values() if j["res"] == "cpu")}
        for j in jobs:
            n = j["name"]
            if status[n]["state"] != "pending":
                continue
            if any(status[d]["state"] != "done" for d in j["deps"]):
                if any(status[d]["state"] == "failed" for d in j["deps"]):
                    status[n] = {"state": "failed", "rc": "dep"}; save()
                continue
            lim = GPU_SLOTS if j["res"] == "gpu" else CPU_SLOTS
            if used[j["res"]] >= lim:
                continue
            log = open(os.path.join(Q, f"{n}.log"), "w")
            p = subprocess.Popen(j["cmd"], shell=True, cwd=HOME, stdout=log, stderr=subprocess.STDOUT)
            running[n] = (p, j, time.time()); used[j["res"]] += 1
            status[n] = {"state": "running", "started": time.strftime("%H:%M")}; save()
            print(time.strftime("%H:%M"), "start", n, flush=True)
        if not running and all(v["state"] in ("done", "failed") for v in status.values()):
            break
        time.sleep(20)
    summarize()
    print("queue finished", flush=True)


if __name__ == "__main__":
    main()
