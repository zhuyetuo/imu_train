"""
从网络摄像头（比如手机IP Webcam这类app推的MJPEG流）实时拉视频，跑训练好的
YOLO模型，把检测框实时画在画面上显示出来。

用法：
    python3 tooth_health/code/live_predict.py \
        --source http://192.168.33.190:8080/video \
        --weights tooth_health/data/runs/tooth_detect/weights/best.pt

按 q 退出，按 s 保存当前帧（连同检测框）到 tooth_health/data/live_snapshots/。

如果是没有显示器/没有GUI环境（比如SSH远程跑），加 --headless，不弹窗口，
终端打印检测结果 + 可选 --save_video 存成mp4文件供之后回看。
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="视频流地址，比如 http://192.168.33.190:8080/video"
                         "（手机IP Webcam app默认的MJPEG流地址），也可以传0/1这种"
                         "本地摄像头编号")
    ap.add_argument("--weights", default="tooth_health/data/runs/tooth_detect/weights/best.pt",
                    help="训练好的模型权重路径")
    ap.add_argument("--conf", type=float, default=0.5, help="置信度阈值，低于这个不显示框")
    ap.add_argument("--imgsz", type=int, default=960, help="推理时的图片尺寸，建议跟训练时一致")
    ap.add_argument("--device", default=None, help="cuda:0 / cpu，不传自动选（有GPU用GPU）")
    ap.add_argument("--headless", action="store_true", help="没有显示器/GUI环境时用，不弹窗口")
    ap.add_argument("--save_video", default=None,
                    help="传路径的话会把带检测框的画面存成mp4，比如"
                         "tooth_health/data/live_snapshots/session1.mp4")
    ap.add_argument("--snapshot_dir", default="tooth_health/data/live_snapshots")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到模型权重 {weights_path}——先跑 train_yolo26.py 训练出一版模型，"
            "或者用 --weights 指定别的权重路径")

    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(
            f"打不开视频流 {args.source}——检查：(1) 手机/摄像头和这台电脑是不是在"
            "同一个局域网；(2) 浏览器直接打开这个地址能不能看到画面；(3) 如果是"
            "IP Webcam这类app，确认地址是/video结尾（MJPEG流），不是/shot.jpg"
            "这种单张截图接口")

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_video:
        save_path = Path(args.save_video)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"开始读取 {args.source}，模型 {weights_path}，按q退出" +
          ("" if args.headless else "（画面窗口里按s保存当前帧）"))

    frame_count, t_start = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[警告] 读不到新帧，视频流可能断了，重试中...")
                time.sleep(0.5)
                continue

            results = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                    device=args.device, verbose=False)
            annotated = results[0].plot()  # 画好检测框+类别+置信度的画面

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                boxes = results[0].boxes
                names = [model.names[int(c)] for c in boxes.cls] if boxes is not None else []
                print(f"  已处理{frame_count}帧，平均{frame_count/elapsed:.1f} FPS，"
                      f"当前帧检测到: {names or '无'}")

            if writer is not None:
                writer.write(annotated)

            if not args.headless:
                cv2.imshow("tooth detection - press q to quit, s to save", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    fname = snapshot_dir / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(fname), annotated)
                    print(f"  已保存截图: {fname}")
    except KeyboardInterrupt:
        print("\n手动中断")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
