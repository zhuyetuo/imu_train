"""
局域网Web服务：用户在网页上传图片/视频，服务器跑YOLO检测，把带检测框的
结果显示/播放给用户看。

用法：
    python3 tooth_health/code/web_app.py \
        --weights tooth_health/data/runs/tooth_detect/weights/best.pt

启动后同一局域网内的设备用浏览器打开 http://<这台机器的局域网IP>:6666
就能上传图片/视频。

图片：秒级出结果，直接在页面上显示标好框的图。
视频：逐帧跑检测，处理完整段视频后转成网页能播放的mp4返回——视频越长
处理越久（跟视频秒数、帧率、这台机器的算力成正比，5090这种卡跑一分钟
视频大概率比视频本身时长快很多，但没有实时进度条，页面上传后会转圈
等，处理完才刷新出结果，不是逐帧实时推流）。
"""
import argparse
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB上限，防止意外传超大文件把服务拖死

MODEL = None  # 启动时加载一次，避免每个请求都重新加载权重（很慢）
UPLOAD_DIR = None
RESULT_DIR = None

PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>狗狗牙齿检测</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ font-size: 22px; }}
  .box {{ border: 1px solid #ccc; border-radius: 8px; padding: 20px; margin-top: 16px; }}
  .result-img, .result-video {{ max-width: 100%; border-radius: 6px; margin-top: 12px; }}
  .detail {{ color: #555; font-size: 14px; margin-top: 8px; white-space: pre-wrap; }}
  button {{ padding: 8px 20px; font-size: 15px; cursor: pointer; }}
  .warn {{ color: #b00; }}
</style>
</head>
<body>
<h1>狗狗牙齿检测（正常/异常）</h1>
<div class="box">
  <form method="post" action="/predict" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*,video/*" required>
    <button type="submit">上传并检测</button>
  </form>
  <p class="detail">支持图片(jpg/png等)和视频(mp4/mov等)。视频文件较大/较长时处理会慢一些，上传后请耐心等页面刷新。</p>
</div>
{result_html}
</body>
</html>
"""


def render_page(result_html=""):
    return PAGE_TEMPLATE.format(result_html=result_html)


@app.route("/")
def index():
    return render_page()


@app.route("/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)


def predict_image(src_path: Path, dst_path: Path, conf: float, imgsz: int):
    results = MODEL.predict(str(src_path), conf=conf, imgsz=imgsz, verbose=False)
    annotated = results[0].plot()
    ok = cv2.imwrite(str(dst_path), annotated)
    if not ok:
        # cv2.imwrite失败时不抛异常、只返回False，之前没检查这个返回值，导致
        # 页面显示"检测完成"但图片文件其实没写出来，/results/xxx.jpg后续访问
        # 404——现在显式检查，写失败就直接报错，报错信息里带完整路径方便诊断
        # （常见原因：目标目录权限不对、磁盘满、文件名带特殊字符）
        raise RuntimeError(f"结果图片写入失败: {dst_path.resolve()}（检查目录写权限/磁盘空间）")
    print(f"  [已保存] {dst_path.resolve()}")
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return "没有检测到口腔区域（可能是构图问题，或者置信度阈值0.5下没有把握判断）"
    lines = []
    for cls_id, conf_val in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        lines.append(f"{MODEL.names[int(cls_id)]}  置信度 {conf_val:.2f}")
    return "\n".join(lines)


def predict_video(src_path: Path, dst_path: Path, conf: float, imgsz: int, frame_skip: int):
    cap = cv2.VideoCapture(str(src_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # avc1(H.264)是浏览器<video>标签能直接播的编码，mp4v不一定能播（取决于浏览器/系统
    # 解码器），优先试avc1，不支持就退回mp4v并提示用户可能需要下载后用本地播放器看
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps / max(1, frame_skip), (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dst_path), fourcc, fps / max(1, frame_skip), (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"视频写入器打不开(avc1和mp4v都试过了): {dst_path.resolve()}，"
            "本机ffmpeg/opencv编码器支持可能有问题，检查目录写权限/磁盘空间")

    class_counts = {}
    frame_idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_skip == 0:
            results = MODEL.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
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
    print(f"  [已保存] {dst_path.resolve()}")

    if written == 0:
        return "视频没有读到任何帧，确认文件没有损坏"
    if not class_counts:
        return f"处理了{written}帧，没有一帧检测到口腔区域"
    lines = [f"处理了{written}帧（frame_skip={frame_skip}）："]
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {name}: {count}帧检测到")
    return "\n".join(lines)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_page('<div class="box warn">没有选文件</div>')

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    uid = uuid.uuid4().hex[:10]
    src_path = UPLOAD_DIR / f"{uid}_{filename}"
    file.save(src_path)

    conf = float(request.form.get("conf", 0.5))
    imgsz = int(request.form.get("imgsz", 960))

    t0 = time.time()
    try:
        if ext in IMAGE_EXTS:
            dst_name = f"{uid}_result.jpg"
            dst_path = RESULT_DIR / dst_name
            detail = predict_image(src_path, dst_path, conf, imgsz)
            media_html = f'<img class="result-img" src="/results/{dst_name}">'
        elif ext in VIDEO_EXTS:
            dst_name = f"{uid}_result.mp4"
            dst_path = RESULT_DIR / dst_name
            detail = predict_video(src_path, dst_path, conf, imgsz, frame_skip=2)
            media_html = (f'<video class="result-video" src="/results/{dst_name}" controls>'
                          f'浏览器不支持播放，<a href="/results/{dst_name}">点这里下载</a></video>')
        else:
            return render_page(f'<div class="box warn">不支持的文件类型: {ext}</div>')
    except Exception as e:
        # 之前这里没有try/except，检测过程中任何报错(比如结果文件写失败)都会让
        # Flask直接返回500，而且完整错误信息只在服务器终端能看到、网页上一片空白
        # 不知道发生了什么。现在把错误直接摊在页面上，同时终端也照常打印完整
        # traceback（用print_exc，方便你本地调试时对照）
        import traceback
        traceback.print_exc()
        return render_page(f'<div class="box warn">处理失败: {e}</div>')

    elapsed = time.time() - t0
    result_html = f"""
<div class="box">
  <h3>检测结果（耗时{elapsed:.1f}秒）</h3>
  {media_html}
  <div class="detail">{detail}</div>
</div>
"""
    return render_page(result_html)


def main():
    global MODEL, UPLOAD_DIR, RESULT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="tooth_health/data/runs/tooth_detect/weights/best.pt")
    ap.add_argument("--port", type=int, default=6666)
    ap.add_argument("--data_dir", default="tooth_health/data/web_uploads",
                    help="上传原文件和检测结果存放的目录")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"找不到模型权重 {weights_path}——先跑 train_yolo26.py 训练出一版模型，"
            "或者用 --weights 指定别的权重路径")

    MODEL = YOLO(str(weights_path))

    data_dir = Path(args.data_dir)
    UPLOAD_DIR = data_dir / "uploads"
    RESULT_DIR = data_dir / "results"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"模型: {weights_path}")
    print(f"局域网访问地址: http://<这台机器的局域网IP>:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
