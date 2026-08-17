"""
用ultralytics训练狗狗口腔正常/异常检测模型。

用法：
    python3 tooth_health/code/train_yolo26.py \
        --data tooth_health/data/yolo_dataset/data.yaml \
        --model yolo26n.pt --epochs 100

先跑 merge_labelstudio_yolo_exports.py 生成 data.yaml，再跑这个脚本。
"""
import argparse

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

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        # 数据量小的时候适当加强数据增强，减少过拟合；具体参数先用ultralytics默认值
        # 跑一版看效果，不要一上来就手调一堆增强参数，先有基线结果再说
    )


if __name__ == "__main__":
    main()
