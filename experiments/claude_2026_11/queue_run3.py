"""Phase 3: longer windows (the strongest lever so far), repeats, and SSL with the best shape. Waits for queue_run2.py.
Usage: nohup python queue_run3.py > queue/queue3.log &"""
import queue_run2 as Q2

Q1 = Q2.Q1
PY = Q1.PY
jobs = Q2.jobs
jobs.clear()
cnn = Q1.cnn
K9W = "--cnn_filters 64,128,128 --cnn_kernel 9"

for w, tag in ((120, "w6"), (160, "w8"), (200, "w10"), (240, "w12")):
    for s in (1, 2, 3):
        cnn(f"k9wide_{tag}_s{s}", f"{K9W} --window {w}")
for w, tag in ((160, "w8"), (200, "w10")):
    for s in (1, 2):
        cnn(f"k9w192_{tag}_s{s}", f"--cnn_filters 64,128,192 --cnn_kernel 9 --window {w}")
        cnn(f"k11wide_{tag}_s{s}", f"--cnn_filters 64,128,128 --cnn_kernel 11 --window {w}")
        cnn(f"k9wide_{tag}_e45_s{s}", f"{K9W} --window {w} --epochs 45")
jobs.append({"name": "ssl_k9wide", "res": "gpu", "deps": [],
             "cmd": f"{PY} ssl_pretrain.py --exclude_ds pool_20261116_all --n_files 16000 --axes 3 --out queue/ssl_k9wide.pt"
                    f" --workers 12 --epochs 60 --filters 64,128,128 --kernel 9"})
for w, tag in ((120, "w6"), (160, "w8"), (200, "w10")):
    for s in (1, 2):
        cnn(f"k9wide_{tag}_ssl_s{s}", f"{K9W} --window {w}", init="queue/ssl_k9wide.pt", deps=["ssl_k9wide"])


def main():
    import subprocess, time
    while any(l.split()[-1].endswith("queue_run2.py") for l in
              subprocess.run(["pgrep", "-af", "queue_run2.py"], capture_output=True, text=True).stdout.splitlines()):
        time.sleep(60)
    # reuse phase-2's scheduler loop with this job list (it waits for phase 1 first: already finished)
    Q2.main.__globals__["jobs"] = jobs
    Q2.main()


if __name__ == "__main__":
    main()
