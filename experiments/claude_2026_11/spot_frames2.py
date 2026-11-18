"""放大版：每个片段两个机位各截 2 帧（片段 30% / 70% 处），640 像素宽，标上这个项圈戴在哪只狗身上。只做原始数据（非 16 Hz）的片段。"""
import json
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

NAS = "/home/toky/ai_data"
items = json.load(open("spot_items.json", encoding="utf-8"))
vids = json.load(open("spot_videos.json", encoding="utf-8"))
dogs = json.load(open("imu_dog.json", encoding="utf-8"))
os.makedirs("spot2", exist_ok=True)
font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 26) if os.path.exists("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc") else ImageFont.load_default()
W = 640
for i, it in enumerate(items):
    if "16hz" in it["csv"]:
        continue
    imu = "IMU" + it["csv"].split("_imu")[1].split("_")[0]
    dog = dogs.get(imu, "?")
    v = vids.get(str(it["sample_id"])) or {}
    cams = [(c, p) for c, p in (("cam1", v.get("cam1")), ("cam2", v.get("cam2")), ("cam3", v.get("cam3"))) if p][:2]
    offs = v.get("offsets") or {}
    dur = it["end"] - it["start"]
    ts = [it["start"] + dur * f for f in (0.3, 0.7)]
    tiles = []
    for cam, p in cams:
        off = (offs.get(cam) or 0) / 1000 if isinstance(offs, dict) else 0
        for t in ts:
            tmp = f"/tmp/spot2_{i}_{cam}_{t:.1f}.jpg"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{max(0, t - off):.2f}", "-i", os.path.join(NAS, p),
                            "-frames:v", "1", "-vf", f"scale={W}:-2", tmp], capture_output=True)
            tiles.append(Image.open(tmp) if os.path.exists(tmp) else Image.new("RGB", (W, 360), "gray"))
    if not tiles:
        continue
    h = max(im.size[1] for im in tiles)
    sheet = Image.new("RGB", (W * 2, h * ((len(tiles) + 1) // 2) + 36), "white")
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % 2) * W, 36 + (k // 2) * h))
    ImageDraw.Draw(sheet).text((5, 3), f"#{i:02d} {it['cat']} {it['month']} {imu}={dog}  {it['start']:.1f}-{it['end']:.1f}s ({dur:.1f}s)  上排 cam1 / 下排 cam2",
                                fill="black", font=font)
    sheet.save(f"spot2/{i:02d}_{it['cat']}.jpg", quality=82)
print("done")
