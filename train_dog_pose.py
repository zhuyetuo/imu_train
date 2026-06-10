"""
狗姿态估计训练脚本（带内存自动清理）

用法:
  python train_dog_pose.py                        # 默认配置
  python train_dog_pose.py --batch 64             # 调整 batch size
  python train_dog_pose.py --model yolo26m-pose.pt  # 更大模型
  python train_dog_pose.py --resume               # 从上次中断处继续
"""

import argparse
import gc
import os
import signal
import sys


def cleanup_memory():
    """释放 GPU 和共享内存"""
    print("\n[cleanup] 开始释放内存...")

    # Python GC
    gc.collect()

    # 释放 CUDA 显存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved  = torch.cuda.memory_reserved()  / 1024**3
            print(f"[cleanup] GPU 显存: allocated={allocated:.1f}GB  reserved={reserved:.1f}GB")
    except Exception as e:
        print(f"[cleanup] GPU 清理跳过: {e}")

    # 释放 PyTorch dataloader 共享内存（/dev/shm 下的 torch_* 文件）
    try:
        import glob
        shm_files = glob.glob("/dev/shm/torch_*") + glob.glob("/dev/shm/*.shm")
        for f in shm_files:
            try:
                os.remove(f)
                print(f"[cleanup] 删除共享内存文件: {f}")
            except Exception:
                pass
    except Exception as e:
        print(f"[cleanup] shm 清理跳过: {e}")

    print("[cleanup] 完成。当前内存:")
    os.system("free -h")


def signal_handler(sig, frame):
    print(f"\n[train] 收到信号 {sig}，正在退出...")
    cleanup_memory()
    sys.exit(0)


def train(args):
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT,  signal_handler)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装: pip install ultralytics")
        sys.exit(1)

    print(f"[train] 模型: {args.model}")
    print(f"[train] batch={args.batch}  imgsz={args.imgsz}  epochs={args.epochs}")
    print(f"[train] compile={not args.no_compile}  half={not args.no_half}")
    print()

    model = YOLO(args.model)

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            cache=False,              # 内存紧张时不缓存
            workers=args.workers,
            amp=True,
            half=not args.no_half,
            compile=not args.no_compile,
            cos_lr=True,
            patience=30,
            exist_ok=True,
            resume=args.resume,
        )
        print(f"\n[train] 训练完成！最优权重: {results.save_dir}/weights/best.pt")

    except MemoryError as e:
        print(f"\n[train] 内存不足: {e}")
        print("[train] 建议: 降低 --batch 或 --workers，或关闭其他占内存的程序")
    except Exception as e:
        print(f"\n[train] 训练中断: {e}")
        raise
    finally:
        cleanup_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="dog-pose.yaml",
                        help="数据集 yaml（默认 dog-pose.yaml）")
    parser.add_argument("--model",   default="yolo26n-pose.pt",
                        help="模型权重（默认 yolo26n-pose.pt）")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=64,
                        help="batch size（内存紧张时用 32）")
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--workers", type=int,   default=4,
                        help="数据加载线程数（内存紧张时用 2）")
    parser.add_argument("--no_compile", action="store_true",
                        help="禁用 torch.compile（兼容性问题时用）")
    parser.add_argument("--no_half",    action="store_true",
                        help="禁用 FP16")
    parser.add_argument("--resume",     action="store_true",
                        help="从上次中断处继续训练")
    train(parser.parse_args())
