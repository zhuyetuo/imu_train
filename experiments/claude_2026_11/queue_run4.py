"""Phase 4: self-training with pseudo-labels from never-labeled NAS data, an unconstrained big-CNN ceiling, extra repeats.
Waits for queue_run3.py. Usage: nohup python queue_run4.py > queue/queue4.log &"""
import queue_run2 as Q2

Q1 = Q2.Q1
PY = Q1.PY
jobs = Q2.jobs
jobs.clear()
cnn = Q1.cnn
K9W = "--cnn_filters 64,128,128 --cnn_kernel 9"

# unlabeled window pools (CPU / NAS bound)
for w in (120, 160):
    jobs.append({"name": f"unl_w{w}", "res": "cpu", "deps": [],
                 "cmd": f"{PY} make_unlabeled.py --window {w} --n_files 4000 --max_windows 400000 --out queue/unl_w{w}.npz --workers 12"})
# self-training (teacher = this fold's model): thresholds 0.95 / 0.9, 6 s and 8 s windows, 2 repeats
for w, tag in ((120, "w6"), (160, "w8")):
    for thr in (0.95, 0.9):
        for s in (1, 2):
            cnn(f"k9wide_{tag}_pl{int(thr*100)}_s{s}", f"{K9W} --window {w} --pseudo_npz queue/unl_w{w}.npz --pseudo_thr {thr}",
                deps=[f"unl_w{w}"])
# ceiling: big CNN not constrained by the board (is the bottleneck the model or the data?)
for w, tag in ((120, "w6"), (160, "w8")):
    for s in (1, 2):
        cnn(f"big_{tag}_s{s}", f"--cnn_filters 128,256,256 --cnn_kernel 9 --window {w} --epochs 40")
# more repeats of the leading shapes
for w, tag in ((120, "w6"), (160, "w8"), (200, "w10")):
    for s in (4, 5):
        cnn(f"k9wide_{tag}_s{s}", f"{K9W} --window {w}")


def main():
    import subprocess, time
    while any(l.split()[-1].endswith("queue_run3.py") for l in
              subprocess.run(["pgrep", "-af", "queue_run3.py"], capture_output=True, text=True).stdout.splitlines()):
        time.sleep(60)
    Q2.main()


if __name__ == "__main__":
    main()
