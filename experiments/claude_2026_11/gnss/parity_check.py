import os, subprocess, numpy as np, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locomotion import steps_and_loco
Z = np.load(os.path.expanduser("~/claude_exp/gnss/vehk_neck20.npz"), allow_pickle=True)
agree = tot = pos_py = pos_c = 0
for a in Z["acc"][:12]:
    a = np.asarray(a, float)
    mg = np.round(a * 1000).astype(int)
    out = subprocess.run(["./parity"], input="\n".join(f"{x} {y} {z}" for x, y, z in mg), capture_output=True, text=True).stdout.split("\n")
    c = np.array([int(l.split()[0]) for l in out if l.strip()])
    _, py, _ = steps_and_loco(mg / 1000.0)
    n = min(len(c), len(py))
    agree += int((c[:n] == py[:n]).sum()); tot += n; pos_py += int(py[:n].sum()); pos_c += int(c[:n].sum())
print(f"逐秒一致 {agree / tot:.3%}（{tot} 秒）；判成走/跑：Python {pos_py} 秒，C {pos_c} 秒")
