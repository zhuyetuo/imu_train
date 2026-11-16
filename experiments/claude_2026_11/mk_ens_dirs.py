"""Average the same method's continuous probabilities across several compare3 output dirs (same files/windows).
Usage: python mk_ens_dirs.py OUT_DIR METHOD DIR1 DIR2 ..."""
import json, os, shutil, sys
import numpy as np
out, m, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(dirs[0], "files.json"), os.path.join(out, "files.json"))
Zs = [np.load(os.path.join(d, f"cont_{m}.npz")) for d in dirs]
res = {}
for k in Zs[0].files:
    if k.startswith("s"):
        i = k[1:]
        assert all((Z[k] == Zs[0][k]).all() for Z in Zs)
        res[k] = Zs[0][k]
        res[f"p{i}"] = np.mean([Z[f"p{i}"].astype(np.float32) for Z in Zs], 0).astype(np.float16)
np.savez_compressed(os.path.join(out, f"cont_{m}.npz"), **res)
print("ok", len(res) // 2)
