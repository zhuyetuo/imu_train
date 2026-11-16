"""Resource watcher: every 30 s looks at GPU utilization / memory and CPU load; when the GPU has been under-used
for 2 minutes (and there is memory headroom) it starts the next GPU job from the server-side pool; when the CPU has
headroom it starts the next CPU job. Server-side pool = best possible model, no edge constraint.
Log: queue/filler.log; results land in q_<name>/ like the other queues (summary.md covers them).
Usage: nohup python filler.py > queue/filler.log 2>&1 &"""
import os, subprocess, time

HOME = os.path.expanduser("~/claude_exp"); Q = os.path.join(HOME, "queue")
PY = os.path.expanduser("~/miniconda3/bin/python")
NCPU = os.cpu_count() or 32


def job(name, args, methods, res):
    return {"name": name, "res": res,
            "cmd": f"{PY} compare3.py --out q_{name} --methods {methods} {args}"
                   f" && {PY} eval_events.py --dir q_{name} --methods {methods} --workers 6"}


GPU_POOL, CPU_POOL = [], []
# server-side best model: deep dilated residual CNN, 6-axis and acc-only, long windows, repeats
for ax in (6, 3):
    for w, tag in ((160, "w8"), (200, "w10"), (120, "w6"), (240, "w12")):
        for ch, nb in ((128, 8), (192, 8), (128, 12)):
            for s in (1, 2):
                GPU_POOL.append(job(f"srv_res{ch}x{nb}_{ax}ax_{tag}_s{s}",
                                    f"--axes {ax} --window {w} --epochs 40 --cnn_arch res --res_ch {ch} --res_blocks {nb}",
                                    "cnn", "gpu"))
# server-side XGBoost: deeper / more rounds, 6-axis and acc-only, long windows
for ax in (6, 3):
    for w, tag in ((160, "w8"), (120, "w6"), (200, "w10")):
        for r, d in ((400, 8), (800, 10)):
            CPU_POOL.append(job(f"srv_xgb{r}d{d}_{ax}ax_{tag}", f"--axes {ax} --window {w} --xgb_rounds {r} --xgb_depth {d}",
                                "xgb", "cpu"))


def gpu_stats():
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().split(",")
    return float(out[0]), float(out[1]), float(out[2])


def main():
    hist, running = [], {}
    gi = ci = 0
    while gi < len(GPU_POOL) or ci < len(CPU_POOL) or running:
        util, mem, tot = gpu_stats()
        load = os.getloadavg()[0]
        hist = (hist + [util])[-4:]                      # last 2 minutes
        for n, p in list(running.items()):
            if p.poll() is not None:
                print(time.strftime("%H:%M"), "done", n, "rc", p.returncode, flush=True); del running[n]
        n_gpu = sum(1 for n in running if n.startswith("srv_res"))
        n_cpu = sum(1 for n in running if n.startswith("srv_xgb"))
        if gi < len(GPU_POOL) and len(hist) == 4 and max(hist) < 70 and mem < 0.75 * tot and n_gpu < 4:
            j = GPU_POOL[gi]; gi += 1
            running[j["name"]] = subprocess.Popen(j["cmd"], shell=True, cwd=HOME, stdout=open(os.path.join(Q, j["name"] + ".log"), "w"),
                                                  stderr=subprocess.STDOUT)
            print(time.strftime("%H:%M"), f"start {j['name']} (gpu util {hist}, mem {mem:.0f}/{tot:.0f} MiB)", flush=True)
            hist = []                                    # give it time to ramp up before judging again
        if ci < len(CPU_POOL) and load < 0.6 * NCPU and n_cpu < 2:
            j = CPU_POOL[ci]; ci += 1
            running[j["name"]] = subprocess.Popen(j["cmd"], shell=True, cwd=HOME, stdout=open(os.path.join(Q, j["name"] + ".log"), "w"),
                                                  stderr=subprocess.STDOUT)
            print(time.strftime("%H:%M"), f"start {j['name']} (load {load:.1f}/{NCPU})", flush=True)
        time.sleep(30)
    print("filler finished", flush=True)


if __name__ == "__main__":
    main()
