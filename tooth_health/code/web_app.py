"""
局域网Web服务（Gradio版）：用户上传图片/视频，或者用浏览器摄像头实时流式
检测，服务器跑YOLO，把带检测框的结果显示给用户看。

用法：
    python3 tooth_health/code/web_app.py \
        --weights tooth_health/data/runs/tooth_detect/weights/best.pt

默认自动生成自签名HTTPS证书并启用HTTPS（局域网内其他设备用浏览器摄像头
必须走HTTPS，这是浏览器安全策略，不是这个服务的限制），启动后浏览器打开
https://<这台机器的局域网IP>:6688（不是6666——Chrome把6666-6669这几个
端口列进了"不安全端口"黑名单会直接拒绝访问，见README里的说明）。首次
访问浏览器会提示"证书不受信任"，点"高级->继续前往"即可，这是自签名证书
的正常现象。只用图片/视频上传功能、不需要摄像头的话可以传--no_https
用回普通http。
"""
import argparse
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

from ssl_utils import ensure_self_signed_cert

MODEL = None
RESULT_DIR = None
CONF = 0.5
IMGSZ = 960
EXAMPLES_DIR = Path("tooth_health/data/examples")


def _find_examples(subdir: str, exts: tuple) -> list:
    """在 tooth_health/data/examples/<subdir>/ 下找文件名以normal/abnormal
    开头的示例文件，不管具体叫什么(normal_1.jpg/normal_狗名.jpg都行)，
    没放文件时返回空列表——Gradio的Examples组件传空列表不会报错，只是
    不显示示例区，不影响正常上传检测功能。"""
    d = EXAMPLES_DIR / subdir
    if not d.exists():
        return []
    files = [p for p in sorted(d.iterdir())
            if p.suffix.lower() in exts and p.name.lower().startswith(("normal", "abnormal"))]
    return [str(p) for p in files]


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


LIVE_IMGSZ = 640  # 摄像头实时流用更小尺寸（图片/视频检测的IMGSZ默认960更看重精度，
# 这里更看重速度）——GPU推理本身已经是零点几毫秒级别不是瓶颈，但imgsz越大，
# 浏览器端JPEG编码、网络传输、服务端解码这几步跟着都会变慢，尺寸小一圈能
# 直接压缩这几步的耗时，对"看着流畅"的帮助比压榨GPU推理时间更明显


def detect_frame(frame):
    """摄像头流式检测用——frame是Gradio webcam传过来的RGB numpy数组，
    跟detect_image/detect_video统一走BGR给ultralytics（cv2生态默认BGR），
    出来再转回RGB给Gradio显示，两头颜色顺序对齐，不然画面颜色会不对。"""
    if frame is None:
        return None
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = MODEL.predict(frame_bgr, conf=CONF, imgsz=LIVE_IMGSZ, verbose=False)
    annotated_bgr = results[0].plot()
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


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
            img_examples = _find_examples("images", (".jpg", ".jpeg", ".png"))
            if img_examples:
                gr.Examples(examples=img_examples, inputs=img_in,
                           label="示例图片（点击直接试，文件名normal开头=正常样本，abnormal开头=异常样本）")
        with gr.Tab("视频检测"):
            with gr.Row():
                vid_in = gr.Video(label="上传口腔视频", sources=["upload"])
                vid_out = gr.Video(label="检测结果")
            vid_detail = gr.Textbox(label="检测详情", lines=5)
            vid_btn = gr.Button("开始检测（视频较长时会等一会）", variant="primary")
            vid_btn.click(detect_video, inputs=vid_in, outputs=[vid_out, vid_detail])
            vid_examples = _find_examples("videos", (".mp4", ".mov", ".avi", ".mkv"))
            if vid_examples:
                gr.Examples(examples=vid_examples, inputs=vid_in,
                           label="示例视频（点击直接试，文件名normal开头=正常样本，abnormal开头=异常样本）")
        with gr.Tab("摄像头实时检测"):
            gr.Markdown(
                "允许浏览器访问摄像头后，画面持续传到服务器跑检测，结果实时显示在"
                "右侧（已经调过刷新频率/丢帧策略，正常局域网下应该是接近实时的"
                "体验，不是逐帧毫秒级但肉眼看不出明显延迟）。\n\n"
                "**局域网内其他设备（手机/笔记本）要用这个功能，访问地址必须是"
                "`https://`开头，不能是`http://`**——这是浏览器自己的摄像头权限"
                "安全策略，不是这个服务的限制。第一次用HTTPS访问会提示\"证书不"
                "受信任\"（因为是自签名证书，不是权威机构签发的），点\"高级->"
                "继续前往\"就行，只需要点一次。"
            )
            with gr.Row():
                cam_in = gr.Image(sources=["webcam"], streaming=True, label="摄像头画面")
                cam_out = gr.Image(label="实时检测结果")
            cam_in.stream(
                detect_frame, inputs=cam_in, outputs=cam_out,
                # stream_every默认0.5秒，相当于浏览器强制限速在2FPS，这才是延迟感
                # 明显的真正原因(不是算力/网络)，调到0.05(~20FPS)。concurrency_limit=1
                # 保证同一时刻只处理一帧，不会因为上一帧还没处理完、下一帧又来了导致
                # 排队积压(积压是"延迟感随时间越来越大"的典型成因)。trigger_mode=
                # "always_last"配合并发限制：处理忙不过来时，只保留最新一帧等着处理，
                # 中间来不及处理的旧帧直接丢弃，保证画面永远是"当前能处理的最新帧"，
                # 而不是"排在队列里的旧帧"，这是实时流场景该有的丢帧策略，不是bug
                stream_every=0.05, concurrency_limit=1, trigger_mode="always_last",
            )
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
    ap.add_argument("--no_https", action="store_true",
                    help="默认开HTTPS(自动生成自签名证书)，因为局域网其他设备"
                         "(手机/笔记本)要用浏览器摄像头必须走HTTPS，这是浏览器"
                         "安全策略决定的。只有服务器自己用自己摄像头/根本不用"
                         "摄像头功能时，可以传这个参数关掉HTTPS图省事(用http)")
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

    launch_kwargs = {"server_name": "0.0.0.0", "server_port": args.port}
    scheme = "http"
    if not args.no_https:
        cert_path, key_path = ensure_self_signed_cert(Path(args.data_dir) / "ssl")
        launch_kwargs["ssl_certfile"] = str(cert_path)
        launch_kwargs["ssl_keyfile"] = str(key_path)
        launch_kwargs["ssl_verify"] = False  # 自签名证书，跳过Gradio自己的校验
        scheme = "https"

    print(f"模型: {weights_path}")
    print(f"局域网访问地址: {scheme}://<这台机器的局域网IP>:{args.port}")
    if scheme == "https":
        print("（首次访问浏览器会提示证书不受信任，点\"高级->继续前往\"即可，"
              "这是自签名证书的正常现象）")

    demo = build_app()
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
