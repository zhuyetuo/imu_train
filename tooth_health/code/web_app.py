"""
局域网Web服务（Gradio版）：用户在网页上传图片/视频，服务器跑YOLO检测，
把带检测框的结果显示/播放给用户看。

用法：
    python3 tooth_health/code/web_app.py \
        --weights tooth_health/data/runs/tooth_detect/weights/best.pt

启动后同一局域网内的设备用浏览器打开 http://<这台机器的局域网IP>:6688
（不是6666——Chrome把6666-6669这几个端口列进了"不安全端口"黑名单会
直接拒绝访问，跟这个服务本身没关系，见README里的说明，想用6666可以
用Firefox打开，或者--port自己指定别的）。
"""
import argparse
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

MODEL = None
RESULT_DIR = None
CONF = 0.5
IMGSZ = 960


def detect_image(image_path):
    if image_path is None:
        return None, "没有上传图片"
    results = MODEL.predict(image_path, conf=CONF, imgsz=IMGSZ, verbose=False)
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)  # gradio的Image组件按RGB显示

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        detail = "没有检测到口腔区域（可能是构图问题，或者置信度阈值0.5下没有把握判断）"
    else:
        lines = [f"{MODEL.names[int(c)]}  置信度 {cf:.2f}"
                for c, cf in zip(boxes.cls.tolist(), boxes.conf.tolist())]
        detail = "\n".join(lines)
    return annotated_rgb, detail


def detect_video(video_path):
    if video_path is None:
        return None, "没有上传视频"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = RESULT_DIR / f"{uuid.uuid4().hex[:10]}_result.mp4"
    # avc1(H.264)是浏览器能直接播的编码，某些机器上ffmpeg/opencv编译没带这个
    # 编码器会打开失败，退回mp4v（不一定所有浏览器都能直接播放，但Gradio自己
    # 托管这个文件、前端播放器兼容性一般比原生<video>标签好一些）
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    frame_skip = 2
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / max(1, frame_skip), (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps / max(1, frame_skip), (w, h))
    if not writer.isOpened():
        cap.release()
        return None, f"视频写入器打不开(avc1和mp4v都试过了)，检查本机ffmpeg/opencv编码器支持"

    class_counts = {}
    frame_idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_skip == 0:
            results = MODEL.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)
            annotated = results[0].plot()
            writer.write(annotated)
            written += 1
            boxes = results[0].boxes
            if boxes is not None:
                for cls_id in boxes.cls.tolist():
                    name = MODEL.names[int(cls_id)]
                    class_counts[name] = class_counts.get(name, 0) + 1
        frame_idx += 1
    cap.release()
    writer.release()

    if written == 0:
        return None, "视频没有读到任何帧，确认文件没有损坏"
    if not class_counts:
        detail = f"处理了{written}帧，没有一帧检测到口腔区域"
    else:
        lines = [f"处理了{written}帧（每{frame_skip}帧抽1帧检测）："]
        for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {name}: {count}帧检测到")
        detail = "\n".join(lines)
    return str(out_path), detail


def build_app():
    with gr.Blocks(title="狗狗牙齿检测") as demo:
        gr.Markdown("# 狗狗牙齿检测（正常/异常）")
        with gr.Tab("图片检测"):
            with gr.Row():
                # sources只留"upload"——默认还会带"webcam"(摄像头拍照)/"clipboard"
                # 这两个选项，摄像头选项会挤占主要显示位置(变成"Click to Access
                # Webcam"这种大按钮)，反而把"点击上传文件"这个真正要用的功能
                # 挤到不显眼的地方，这里明确只保留上传
                img_in = gr.Image(type="filepath", label="上传口腔照片", sources=["upload"])
                img_out = gr.Image(label="检测结果")
            img_detail = gr.Textbox(label="检测详情", lines=3)
            img_btn = gr.Button("开始检测", variant="primary")
            img_btn.click(detect_image, inputs=img_in, outputs=[img_out, img_detail])
        with gr.Tab("视频检测"):
            with gr.Row():
                vid_in = gr.Video(label="上传口腔视频", sources=["upload"])
                vid_out = gr.Video(label="检测结果")
            vid_detail = gr.Textbox(label="检测详情", lines=5)
            vid_btn = gr.Button("开始检测（视频较长时会等一会）", variant="primary")
            vid_btn.click(detect_video, inputs=vid_in, outputs=[vid_out, vid_detail])
    return demo


def main():
    global MODEL, RESULT_DIR, CONF, IMGSZ
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="tooth_health/data/runs/tooth_detect/weights/best.pt")
    ap.add_argument("--port", type=int, default=6688,
                    help="默认6688不是6666——Chrome把6666-6669列进不安全端口黑名单，"
                         "直接用6666大概率打不开，想强行用6666可以传这个参数覆盖，"
                         "配合Firefox打开（Firefox不拦截）")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--data_dir", default="tooth_health/data/web_uploads")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到模型权重 {weights_path}——先跑 train_yolo26.py 训练出一版模型，"
            "或者用 --weights 指定别的权重路径")

    MODEL = YOLO(str(weights_path))
    CONF, IMGSZ = args.conf, args.imgsz
    RESULT_DIR = Path(args.data_dir) / "results"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"模型: {weights_path}")
    print(f"局域网访问地址: http://<这台机器的局域网IP>:{args.port}")

    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
