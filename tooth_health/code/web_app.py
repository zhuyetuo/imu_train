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
import warnings
from pathlib import Path

import cv2
import gradio as gr
from ultralytics import YOLO

from ssl_utils import ensure_self_signed_cert, get_lan_ip

# 示例视频/detect_video产出的结果都是mp4v软件编码，Gradio的<video>预览组件
# 检测到不是浏览器原生支持的编码时会自动转一遍(行为本身没问题，转完照样能看)，
# 但每次都刷一条警告到终端，纯噪音，压掉这一条，不影响其他警告正常显示
warnings.filterwarnings("ignore", message="Video does not have browser-compatible container or codec.*")

MODEL = None
RESULT_DIR = None
CONF = 0.5
IMGSZ = 960
EXAMPLES_DIR = Path("tooth_health/data/examples")


def _find_examples(subdir: str, exts: tuple) -> dict:
    """在 tooth_health/data/examples/<subdir>/ 下找文件名以normal/abnormal
    开头的示例文件，不管具体叫什么(normal_1.jpg/normal_狗名.jpg都行)。
    按"正常"/"异常"分组返回，方便上层各自打标签展示，不是混在一起的
    一个大列表（混在一起就是之前那版拥挤、看不出哪个是哪个的问题）。
    没放文件时对应分组是空列表，不影响正常上传检测功能。"""
    d = EXAMPLES_DIR / subdir
    if not d.exists():
        return {"正常样本": [], "异常样本": []}
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    return {
        "正常样本": [str(p) for p in files if p.name.lower().startswith("normal")],
        "异常样本": [str(p) for p in files if p.name.lower().startswith("abnormal")],
    }


def _build_example_picker(component_cls, groups: dict, target_input, height=220):
    """给每个示例文件单独起一列：标签(正常样本/异常样本) + 预览(实际渲染
    图片/视频，不是Gradio Examples那种小缩略图画廊，视频缩略图之前那版
    经常显示不出来，这样直接用真实的Image/Video组件展示，一定能看到内容)
    + 一个"用这个测试"按钮点了直接把这个文件灌进主输入框。没有任何示例
    文件时这个函数不渲染任何东西，不留空区块。"""
    total = sum(len(v) for v in groups.values())
    if total == 0:
        return
    gr.Markdown("**示例（点预览下面的按钮直接加载去检测，不用自己找文件）**")
    with gr.Row():
        for label, paths in groups.items():
            for i, path in enumerate(paths):
                col_label = label if len(paths) == 1 else f"{label} {i + 1}"
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown(f"<div style='text-align:center'>{col_label}</div>")
                    component_cls(value=path, interactive=False, height=height,
                                 show_label=False)
                    btn = gr.Button(f"用这个测试", size="sm")
                    btn.click(lambda p=path: p, outputs=target_input)


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
    # 直接用mp4v软件编码，不再先试avc1(H.264)硬件编码——avc1走的是V4L2 M2M
    # 接口，这是给树莓派这类带硬件视频编码芯片的嵌入式设备用的，普通x86服务器
    # (包括这台带5090的机器)根本没有这个硬件接口，每次都会失败，纯粹刷屏+
    # 浪费一次失败尝试的时间。而且不管我们这里写出来的是什么编码，Gradio
    # 自己收到文件后都会再检查一遍、不是浏览器兼容格式就自动转一遍mp4
    # (日志里"Converting to mp4"那行)，所以在avc1和mp4v之间纠结没有意义，
    # 直接用软件编码最简单
    frame_skip = 2
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(1, frame_skip), (w, h))
    if not writer.isOpened():
        cap.release()
        return None, f"视频写入器打不开(mp4v)，检查本机ffmpeg/opencv编码器支持"

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
    出来再转回RGB给Gradio显示，两头颜色顺序对齐，不然画面颜色会不对。

    这里包了try/except——流式场景下，如果某一帧处理时抛异常且没兜住，
    Gradio可能会认为这个"事件"卡住没结束，后续帧因为concurrency_limit=1
    (同一时刻只处理一帧)排在后面永远等不到轮到自己，表现出来就是画面
    整个卡死不动，不是变慢。异常时直接把原始画面原样返回(不叠加检测框)，
    保证流不会因为单帧出错就整体卡住，同时把错误打到终端方便排查。"""
    if frame is None:
        return None
    try:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = MODEL.predict(frame_bgr, conf=CONF, imgsz=LIVE_IMGSZ, verbose=False)
        annotated_bgr = results[0].plot()
        return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"  [摄像头帧处理出错，跳过这一帧] {e}")
        return frame


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
            _build_example_picker(gr.Image, _find_examples("images", (".jpg", ".jpeg", ".png")), img_in)
        with gr.Tab("视频检测"):
            with gr.Row():
                vid_in = gr.Video(label="上传口腔视频", sources=["upload"])
                vid_out = gr.Video(label="检测结果")
            vid_detail = gr.Textbox(label="检测详情", lines=5)
            vid_btn = gr.Button("开始检测（视频较长时会等一会）", variant="primary")
            vid_btn.click(detect_video, inputs=vid_in, outputs=[vid_out, vid_detail])
            _build_example_picker(gr.Video, _find_examples("videos", (".mp4", ".mov", ".avi", ".mkv")), vid_in)
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
                # stream_every默认0.5秒，相当于浏览器强制限速在2FPS，这是延迟感
                # 明显的主因(不是算力/网络)，调到0.1(~10FPS)。之前还加过
                # trigger_mode="always_last"，实测在这个Gradio版本的webcam流式
                # 组件上反而会导致整个画面卡死不动(不是变慢，是完全卡住)，
                # 怀疑是这个参数跟streaming=True的内部队列机制冲突，已去掉，
                # 只保留concurrency_limit=1(避免同一时刻处理多帧导致乱序/资源
                # 争抢)，stream_every调低这一项本身已经能明显改善延迟感
                stream_every=0.1, concurrency_limit=1,
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
    ap.add_argument("--share", action="store_true",
                    help="额外开一个公网可访问的临时链接(通过Gradio自己的隧道"
                         "服务器转发，不需要你自己有公网IP/做端口转发)，适合人"
                         "在外面连不上公司局域网时用，局域网地址同时还能正常用，"
                         "两个不冲突。这个链接一般72小时后失效，需要这台机器"
                         "能连外网；默认不开，因为链接本身是公开的，只要有人拿到"
                         "链接不用连公司网络也能访问，只在确实需要远程访问时开")
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

    lan_ip = get_lan_ip()
    print(f"模型: {weights_path}")
    print(f"局域网访问地址（可以直接发给同局域网的其他人）: {scheme}://{lan_ip}:{args.port}")
    if scheme == "https":
        print("（首次访问浏览器会提示证书不受信任，点\"高级->继续前往\"即可，"
              "这是自签名证书的正常现象）")
    if args.share:
        print("正在申请公网临时链接（需要这台机器能连外网，稍等几秒）..."
              "启动完成后下面会额外打印一个https://xxx.gradio.live这样的"
              "链接，人在外面时用这个，不受限于局域网")

    demo = build_app()
    demo.launch(share=args.share, **launch_kwargs)


if __name__ == "__main__":
    main()
