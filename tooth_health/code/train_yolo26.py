"""
用ultralytics训练狗狗口腔正常/异常检测模型。

用法：
    python3 tooth_health/code/train_yolo26.py \
        --data tooth_health/data/yolo_dataset/data.yaml \
        --model yolo26n.pt --epochs 100

先跑 merge_labelstudio_yolo_exports.py 生成 data.yaml，再跑这个脚本。
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="merge_labelstudio_yolo_exports.py生成的data.yaml")
    ap.add_argument("--model", default="yolo26n.pt",
                    help="基础权重，n/s/m/l/x对应不同模型大小，数据量小(几十到几百张)"
                         "建议从最小的n开始，不容易过拟合，训练也快")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="tooth_health/data/runs")
    ap.add_argument("--name", default="tooth_detect")
    args = ap.parse_args()

    # project传相对路径时，实测某些ultralytics版本会把它当成"name"的一部分拼在
    # 默认的runs/detect/下面(变成 <cwd>/runs/detect/tooth_health/data/runs/
    # tooth_detect这种嵌套错误路径)，不是我们要的tooth_health/data/runs/
    # tooth_detect。这里显式转成绝对路径，绕开这个歧义，保证结果一定存在
    # 工作目录里，不会跑到repo根目录下多出来的runs/detect/
    project_abs = str(Path(args.project).resolve())

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=project_abs,
        name=args.name,
        # 数据量小的时候适当加强数据增强，减少过拟合；具体参数先用ultralytics默认值
        # 跑一版看效果，不要一上来就手调一堆增强参数，先有基线结果再说
    )
    print(f"\n训练结果保存在: {results.save_dir}")
    print(f"最优权重: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
