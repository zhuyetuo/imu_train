"""
批量实验并行启动器

用法:
  # 默认：ML 4进程并行，DL 1个GPU顺序跑
  python run_experiments.py

  # 指定数据集和采样率
  python run_experiments.py --datasets processed_a processed_b --hz 25 50

  # 只跑 ML
  python run_experiments.py --skip_dl

  # 只跑 DL
  python run_experiments.py --skip_ml

  # ML 并行度（进程数）
  python run_experiments.py --ml_workers 8

  # DL 并行（多 GPU 或显存足够时）
  python run_experiments.py --dl_workers 2
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


ML_MODELS = ["rf", "xgb", "lgbm", "catboost"]
DL_MODELS = ["cnn", "collar_cnn", "cnn_lstm", "transformer", "filternet", "filternet_m2m"]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


def ts():
    return datetime.now().strftime("%H:%M:%S")


def run_job(cmd: list[str], label: str) -> tuple[str, bool, str]:
    """运行单个训练命令，返回 (label, success, output_tail)。"""
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start
        ok = result.returncode == 0
        # 取最后几行作为摘要
        out_lines = (result.stdout + result.stderr).strip().splitlines()
        tail = "\n    ".join(out_lines[-6:]) if out_lines else ""
        status = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  [{ts()}] {status} {label}  ({elapsed:.0f}s)")
        if not ok:
            print(f"    {RED}{tail}{RESET}")
        return label, ok, tail
    except Exception as e:
        return label, False, str(e)


def build_jobs(datasets, hz_list, ml_models, dl_models, skip_ml, skip_dl):
    ml_jobs, dl_jobs = [], []
    for ds in datasets:
        for hz in hz_list:
            if not skip_ml:
                for model in ml_models:
                    cmd = [
                        sys.executable, "src/ml/train.py",
                        "--hz", str(hz),
                        "--model", model,
                        "--processed_dir", f"data/{ds}",
                    ]
                    label = f"ML/{model:<10} {ds} {hz}hz"
                    ml_jobs.append((cmd, label))
            if not skip_dl:
                for model in dl_models:
                    cmd = [
                        sys.executable, "src/dl/train.py",
                        "--hz", str(hz),
                        "--model", model,
                        "--processed_dir", f"data/{ds}",
                    ]
                    label = f"DL/{model:<14} {ds} {hz}hz"
                    dl_jobs.append((cmd, label))
    return ml_jobs, dl_jobs


def run_pool(jobs, workers, section_name):
    if not jobs:
        return [], []
    print(f"\n{CYAN}{'─'*60}{RESET}")
    print(f"{CYAN}  {section_name}  ({len(jobs)} 个任务，{workers} 进程并行){RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")

    failed = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, cmd, label): label for cmd, label in jobs}
        for fut in as_completed(futures):
            label, ok, _ = fut.result()
            done += 1
            if not ok:
                failed.append(label)
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["processed_a", "processed_b", "processed_custom"])
    parser.add_argument("--hz", nargs="+", type=int, default=[5, 10, 25, 50])
    parser.add_argument("--ml_models", nargs="+", default=ML_MODELS)
    parser.add_argument("--dl_models", nargs="+", default=DL_MODELS)
    parser.add_argument("--ml_workers", type=int, default=4,
                        help="ML 并行进程数（默认 4）")
    parser.add_argument("--dl_workers", type=int, default=1,
                        help="DL 并行进程数，单 GPU 建议保持 1（默认 1）")
    parser.add_argument("--skip_ml", action="store_true")
    parser.add_argument("--skip_dl", action="store_true")
    args = parser.parse_args()

    ml_jobs, dl_jobs = build_jobs(
        args.datasets, args.hz,
        args.ml_models, args.dl_models,
        args.skip_ml, args.skip_dl,
    )

    total = len(ml_jobs) + len(dl_jobs)
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  批量实验启动器{RESET}")
    print(f"  数据集: {args.datasets}")
    print(f"  采样率: {args.hz}")
    print(f"  总任务: {total}  (ML={len(ml_jobs)}, DL={len(dl_jobs)})")
    print(f"{CYAN}{'='*60}{RESET}")

    t0 = time.time()
    all_failed = []

    all_failed += run_pool(ml_jobs, args.ml_workers, "机器学习")
    all_failed += run_pool(dl_jobs, args.dl_workers, "深度学习")

    elapsed = time.time() - t0
    print(f"\n{CYAN}{'='*60}{RESET}")
    succeeded = total - len(all_failed)
    print(f"  完成: {succeeded}/{total}  总耗时: {elapsed/60:.1f} 分钟")
    if all_failed:
        print(f"\n  {RED}失败任务:{RESET}")
        for label in all_failed:
            print(f"    {RED}✗ {label}{RESET}")
    else:
        print(f"  {GREEN}全部成功！{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
