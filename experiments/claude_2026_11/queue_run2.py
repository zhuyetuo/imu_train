"""Phase 2 of the experiment queue: waits for phase 1 (queue_run.py) to finish, then runs more jobs with the same
resource limits. Summary covers every q_* result dir (both phases). Usage: nohup python queue_run2.py > queue/queue2.log &"""
import json, os, subprocess, time
import numpy as np

import queue_run as Q1

HOME, Q, PY = Q1.HOME, Q1.Q, Q1.PY
K7W = Q1.K7W
Q1.jobs.clear()
cnn, xgb, jobs = Q1.cnn, Q1.xgb, Q1.jobs

# gyro value with the new architecture (6-axis: new timeline cache)
for s in (1, 2):
    jobs.append({"name": f"k7wide_6ax_r{s}", "res": "gpu", "deps": [],
                 "cmd": f"{PY} compare3.py --axes 6 --epochs 30 --methods cnn --out q_k7wide_6ax_r{s} {K7W}"
                        f" && {PY} eval_events.py --dir q_k7wide_6ax_r{s} --methods cnn --workers 6"})
# longer training, more repeats of the neighbours
for s in (1, 2):
    cnn(f"k7wide_e50_r{s}", f"{K7W} --epochs 50")
cnn("k7wide_w6_r3", f"{K7W} --window 120")
cnn("k9wide_r2", "--cnn_filters 64,128,128 --cnn_kernel 9")
cnn("k7w192_r2", "--cnn_filters 64,128,192 --cnn_kernel 7")
cnn("k7w96_r2", "--cnn_filters 96,128,128 --cnn_kernel 7")
xgb("xgb150d6_w8", "--xgb_rounds 150 --window 160")
jobs.append({"name": "xgb50d6_6ax", "res": "cpu", "deps": [],
             "cmd": f"{PY} compare3.py --axes 6 --epochs 30 --methods xgb --out q_xgb50d6_6ax"
                    f" && {PY} eval_events.py --dir q_xgb50d6_6ax --methods xgb --workers 6"})
# big self-supervised pretraining on (almost) all never-labeled files, then fine-tune 4 s and 6 s windows
jobs.append({"name": "ssl_big", "res": "gpu", "deps": [],
             "cmd": f"{PY} ssl_pretrain.py --exclude_ds pool_20261116_all --n_files 16000 --axes 3 --out queue/ssl_big.pt"
                    f" --workers 12 --epochs 60 --filters 64,128,128 --kernel 7"})
for s in (1, 2):
    cnn(f"k7wide_sslbig_r{s}", K7W, init="queue/ssl_big.pt", deps=["ssl_big"])
cnn("k7wide_w6_sslbig", f"{K7W} --window 120", init="queue/ssl_big.pt", deps=["ssl_big"])


def summarize_all():
    names = sorted(d[2:] for d in os.listdir(HOME) if d.startswith("q_") and os.path.exists(os.path.join(HOME, d, "events.json")))
    Q1.jobs[:] = [{"name": n} for n in names]
    Q1.summarize()


Q1.summarize_orig = Q1.summarize


def main():
    # wait for phase 1
    while subprocess.run(["pgrep", "-f", "queue_run.py"], capture_output=True).stdout.strip() and \
            any(l.split()[-1].endswith("queue_run.py") for l in
                subprocess.run(["pgrep", "-af", "queue_run.py"], capture_output=True, text=True).stdout.splitlines()):
        time.sleep(60)
    print(time.strftime("%H:%M"), "phase 1 done, starting phase 2", flush=True)
    status = {j["name"]: {"state": "pending"} for j in jobs}
    running = {}
    phase2 = list(jobs)
    def save():
        json.dump(status, open(os.path.join(Q, "status2.json"), "w"), ensure_ascii=False, indent=1)
    save()
    while True:
        for n, (p, j, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                status[n] = {"state": "done" if rc == 0 else "failed", "rc": rc, "minutes": round((time.time() - t0) / 60, 1)}
                del running[n]; save()
                try:
                    summarize_all()
                except Exception as e:  # noqa: BLE001
                    print("summary failed", e, flush=True)
                jobs[:] = phase2
                print(time.strftime("%H:%M"), n, status[n], flush=True)
        used = {"gpu": sum(1 for _, j, _ in running.values() if j["res"] == "gpu"),
                "cpu": sum(1 for _, j, _ in running.values() if j["res"] == "cpu")}
        for j in phase2:
            n = j["name"]
            if status[n]["state"] != "pending":
                continue
            if any(status[d]["state"] != "done" for d in j["deps"]):
                if any(status[d]["state"] == "failed" for d in j["deps"]):
                    status[n] = {"state": "failed", "rc": "dep"}; save()
                continue
            if used[j["res"]] >= (Q1.GPU_SLOTS if j["res"] == "gpu" else Q1.CPU_SLOTS):
                continue
            log = open(os.path.join(Q, f"{n}.log"), "w")
            p = subprocess.Popen(j["cmd"], shell=True, cwd=HOME, stdout=log, stderr=subprocess.STDOUT)
            running[n] = (p, j, time.time()); used[j["res"]] += 1
            status[n] = {"state": "running", "started": time.strftime("%H:%M")}; save()
            print(time.strftime("%H:%M"), "start", n, flush=True)
        if not running and all(v["state"] in ("done", "failed") for v in status.values()):
            break
        time.sleep(20)
    summarize_all()
    print("phase 2 finished", flush=True)


if __name__ == "__main__":
    main()
