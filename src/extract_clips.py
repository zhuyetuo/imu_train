"""
读取推理 JSON，把每段抓挠裁剪成与 witmotion_imu 完全一致的三件套：
  {stem}_clip{N:02d}_{HHMMSS}-{HHMMSS}.mp4   ← cam1 视频片段
  {cam2_stem}_clip{N:02d}_{HHMMSS}-{HHMMSS}.mp4  ← cam2 视频片段
  {stem}_clip{N:02d}_{HHMMSS}-{HHMMSS}.csv   ← cam1 CSV 裁剪

文件按置信度分桶放入 clips_{bin}/ 子目录，同时写 scratch_log_review_{bin}.txt。

用法:
  python src/extract_clips.py \
    --infer_dir  infer_result/2026_7_17/_infer \
    --video_dir  data/raw_custom/data/2026_7_17 \
    --output_dir infer_result/2026_7_17 \
    --context_s  3
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def get_bin(conf: float) -> str:
    for i in range(len(BINS) - 1):
        if BINS[i] <= conf < BINS[i + 1]:
            lo = BINS[i]
            hi = BINS[i + 1] if BINS[i + 1] <= 1.0 else 1.0
            return f"{lo:.1f}-{hi:.1f}"
    return "0.9-1.0"


def ts_to_sec(ts_str: str) -> float:
    if not ts_str:
        return 0.0
    dt = datetime.strptime(ts_str[:23], "%Y-%m-%d %H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


def ts_to_hhmmss(ts_str: str) -> str:
    """'2026-07-17 18:59:07.681' → '185907'"""
    if not ts_str:
        return "000000"
    return ts_str[11:13] + ts_str[14:16] + ts_str[17:19]


HAS_CUDA = subprocess.run(
    ["ffmpeg", "-hide_banner", "-hwaccels"],
    capture_output=True, text=True
).stdout.find("cuda") >= 0


def find_video(stem: str, video_dir: str) -> str:
    for ext in (".mp4", ".MP4", ".avi", ".mov"):
        p = os.path.join(video_dir, stem + ext)
        if os.path.exists(p):
            return p
    return ""


def csv_start_sec(csv_path: str) -> float:
    """读取 CSV 第一行数据的时间戳，作为视频 t=0 的绝对时间（秒）。"""
    try:
        with open(csv_path, encoding="utf-8") as f:
            header = f.readline()
            first  = f.readline()
        cols = [c.strip().strip('"') for c in header.split(",")]
        try:
            ts_idx = next(i for i, c in enumerate(cols) if "timestamp" in c.lower())
        except StopIteration:
            ts_idx = 0
        ts_str = first.split(",")[ts_idx].strip().strip('"')
        return ts_to_sec(ts_str)
    except Exception:
        return 0.0


def sibling_stem(stem: str, target_cam: int) -> str:
    """cam1_imu1 → cam2_imu2（或反向），target_cam=1 或 2"""
    return re.sub(r"cam\d_imu\d", f"cam{target_cam}_imu{target_cam}", stem)


_WINDOWS_CACHE = {}
_WINDOWS_CACHE_LOCK = threading.Lock()


def load_windows_for_stem(infer_dir: str, stem: str) -> list:
    """加载某个 stem（比如 xxx_cam1_imu1）对应的 {stem}_infer.json 里的逐窗口预测
    （ts/label/conf/probs），每个stem只读一次盘，同一批clip任务共用缓存。
    找不到文件时返回空列表（比如只有一路IMU跑了推理，另一路没跑）。"""
    with _WINDOWS_CACHE_LOCK:
        if stem in _WINDOWS_CACHE:
            return _WINDOWS_CACHE[stem]
    path = os.path.join(infer_dir, f"{stem}_infer.json")
    windows = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                windows = json.load(f).get("windows", [])
        except Exception as e:
            print(f"  [警告] 读取 {path} 失败: {e}")
    with _WINDOWS_CACHE_LOCK:
        _WINDOWS_CACHE[stem] = windows
    return windows


def _ass_time(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: conf,Noto Sans CJK SC,46,&H0000D7FF,&H00000000,&H80000000,1,2,1,9,10,10,24
[Events]
Format: Layer, Start, End, Style, Text
"""


def build_conf_subtitle(windows: list, video_t0: float, rel_start: float, clip_duration: float,
                        event_start_abs: float, event_end_abs: float,
                        out_ass_path: str, label_text: str, target_label: str = "抓挠") -> bool:
    """把某一路IMU的逐窗口置信度，转成跟裁剪后视频时间轴对齐的ASS字幕文件，
    烧录到右上角(Alignment=9)。

    只在真正的事件时间范围[event_start_abs, event_end_abs]内显示字幕——
    clip本身前后会用context_s多留一段缓冲（给人看前因后果），但那段缓冲不是
    事件本身，缓冲区间里模型的真实置信度往往偏低，如果整段clip全程都显示
    数字，容易让人误以为缓冲区间也是"检测到抓挠"的一部分，所以缓冲段不烧
    字幕，只在真实事件区间内显示，跟真实标注/检测范围严格对应。

    时间换算：cut_clip()里视频用 -ss rel_start 从原始视频（未裁剪）的
    rel_start秒处开始截取，输出clip自己的0点对应原始视频的rel_start秒。
    某个时间点的绝对秒数，先换算成"原始视频相对秒数" = abs_sec - video_t0，
    再减掉rel_start，就是这个时间点在裁剪后clip里的位置。
    """
    event_rel_start = (event_start_abs - video_t0) - rel_start
    event_rel_end   = (event_end_abs   - video_t0) - rel_start
    event_rel_start = max(0.0, min(event_rel_start, clip_duration))
    event_rel_end   = max(0.0, min(event_rel_end,   clip_duration))
    if event_rel_end <= event_rel_start:
        return False

    events = []
    filtered = []
    for w in windows:
        ts_str = w.get("ts")
        if not ts_str:
            continue
        abs_sec = ts_to_sec(ts_str)
        rel_sec = (abs_sec - video_t0) - rel_start
        # 只保留落在真实事件时间范围内的窗口（留一点点容差，覆盖窗口本身
        # 跨越事件边界的情况，避免边界上刚好差一帧就整个不显示）
        if (event_rel_start - 0.5) <= rel_sec <= (event_rel_end + 0.5):
            conf = w.get("probs", {}).get(target_label)
            if conf is None:
                conf = w.get("conf", 0.0) if w.get("label") == target_label else 0.0
            filtered.append((rel_sec, conf))
    if not filtered:
        return False
    filtered.sort(key=lambda x: x[0])

    for i, (t0, conf) in enumerate(filtered):
        t1 = filtered[i + 1][0] if i + 1 < len(filtered) else t0 + 0.5
        t0c = max(event_rel_start, min(t0, event_rel_end))
        t1c = max(t0c + 0.02, min(t1, event_rel_end))
        if t0c >= event_rel_end:
            continue
        text = f"{label_text} {target_label}:{conf:.2f}"
        events.append(f"Dialogue: 0,{_ass_time(t0c)},{_ass_time(t1c)},conf,{text}")

    if not events:
        return False
    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(ASS_HEADER)
        f.write("\n".join(events) + "\n")
    return True


def cut_clip(video_path: str, start_abs: float, end_abs: float,
             out_path: str, context_s: float, video_t0: float = 0.0,
             windows: list = None, label_text: str = "") -> bool:
    """
    start_abs / end_abs: 午夜起始的绝对秒数
    video_t0: 视频第一帧对应的绝对秒数（从 CSV 第一行读取）
    windows: 若提供该路IMU自己的逐窗口预测（{stem}_infer.json里的windows数组），
      会在右上角烧录实时抓挠置信度字幕，方便复查时直接看置信度曲线，不用
      对照日志猜哪一段模型信心高/低。传 None 或空列表则不烧字幕，跟以前
      行为一致。
    """
    rel_start = max(0.0, (start_abs - video_t0) - context_s)
    rel_end   = (end_abs - video_t0) + context_s
    duration  = rel_end - rel_start
    if duration <= 0:
        print(f"    [跳过] 时间偏移为负（video_t0={video_t0:.1f}, start={start_abs:.1f}）")
        return False

    # 浏览器兼容优先：libx264 baseline，faststart，yuv420p，无音频
    # nvenc 输出某些浏览器不支持，统一用软编码保证兼容性
    #
    # 提速两点（复查用的片段，不是最终交付质量，preset变快、体积略增没关系）：
    #   1. preset从fast改成veryfast——libx264速度阶梯里veryfast比fast快不少，
    #      画质肉眼基本看不出差别，复查用完全够。
    #   2. 显式限制-threads——不限制的话libx264默认会尝试用满所有CPU核心，
    #      而这里本来就是many-workers并行跑多个ffmpeg进程，每个进程再各自
    #      抢满所有核心，会严重过度订阅（16个进程 x 各自想用24核），互相
    #      抢核反而更慢。显式限制每个进程用少量线程，让"并行进程数"而不是
    #      "单进程内部线程数"来吃满CPU，这是many-parallel-encodes场景的
    #      标准优化方式。
    vf_chain = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    ass_path = None
    if windows:
        ass_path = out_path + ".ass"
        ok_sub = build_conf_subtitle(windows, video_t0, rel_start, duration,
                                     start_abs, end_abs, ass_path, label_text)
        if ok_sub:
            # subtitles滤镜的路径里冒号/反斜杠有特殊含义，需要转义；
            # 单引号包住整个路径，路径本身的单引号也要转义（正常路径不会有，防御一下）
            escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            vf_chain = f"subtitles='{escaped}'," + vf_chain
        else:
            ass_path = None  # 没有落在这段时间范围内的窗口，没必要烧字幕

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{rel_start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-crf", "23",
        "-preset", "veryfast",
        "-threads", "2",
        "-pix_fmt", "yuv420p",
        "-an",                       # 原始视频无音频，明确丢弃避免 aac 编码报错
        "-movflags", "+faststart",   # moov atom 放文件头，浏览器流式播放必需
        "-vf", vf_chain,             # 确保宽高为偶数，可选叠加置信度字幕
        out_path,
    ]
    ret = subprocess.run(cmd, capture_output=True)
    if ass_path and os.path.exists(ass_path):
        os.remove(ass_path)  # 字幕已经烧进视频画面了，临时ass文件不用留
    if ret.returncode != 0:
        err = ret.stderr.decode("utf-8", errors="replace")[-400:]
        print(f"    [ffmpeg错误] {err}")
    return ret.returncode == 0


def slice_csv(src_csv: str, dst_csv: str,
              pad_start: float, pad_end: float) -> bool:
    """裁剪 CSV 到 [pad_start, pad_end] 秒范围内的行。"""
    try:
        with open(src_csv, encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return False
        header = lines[0]
        cols = [c.strip().strip('"') for c in header.split(",")]
        try:
            ts_idx = next(i for i, c in enumerate(cols) if "timestamp" in c.lower())
        except StopIteration:
            ts_idx = 0
        kept = [header]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= ts_idx:
                continue
            try:
                sec = ts_to_sec(parts[ts_idx].strip().strip('"'))
            except Exception:
                continue
            if pad_start <= sec <= pad_end:
                kept.append(line)
        if len(kept) <= 1:
            return False
        with open(dst_csv, "w", encoding="utf-8") as f:
            f.writelines(kept)
        return True
    except Exception as e:
        print(f"    [CSV裁剪错误] {e}")
        return False


def _symlink_into(src_path: str, dst_dir: str) -> None:
    """在 dst_dir 下建一个指向 src_path 的相对软链接，同名文件已存在则跳过。
    用于 --bin_by both：物理文件只裁剪一次，第二套分桶目录里放软链接，
    不重复编码、不占双倍磁盘。"""
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    if os.path.exists(dst_path) or os.path.islink(dst_path):
        return
    rel_target = os.path.relpath(src_path, dst_dir)
    try:
        os.symlink(rel_target, dst_path)
    except OSError as e:
        print(f"    [软链接失败] {dst_path}: {e}")


def _process_one(task: dict) -> dict:
    """处理单个 clip 任务（在线程池中运行）。"""
    args       = task["args"]
    has_ffmpeg = task["has_ffmpeg"]
    stem1      = task["stem1"]
    stem2      = task["stem2"]
    src_csv1   = task["src_csv1"]
    cam1_mp4   = task["cam1_mp4"]
    cam2_mp4   = task["cam2_mp4"]
    video_t0   = task["video_t0"]
    clip_dir   = task["clip_dir"]
    suffix     = task["suffix"]
    seg        = task["seg"]
    secondary_dir = task.get("secondary_dir")

    start_sec = ts_to_sec(seg["start_ts"])
    end_sec   = ts_to_sec(seg["end_ts"]) if seg["end_ts"] else start_sec + 2.0
    pad_start = max(0.0, start_sec - args.context_s)
    pad_end   = end_sec + args.context_s

    detected_stem = task["detected_stem"]
    # run_tag（一般是RESULT_ROOT的名字，比如"infer_result_majority_syn"）拼进
    # 文件名末尾：不同模型/不同次运行的推理结果，文件名本来只由"CSV名+第几段+
    # 起止时间"决定，如果两次运行检测到时间重叠的事件，文件名会完全一样，
    # 复制到共享的Nginx媒体目录时后一次会静默覆盖前一次，Label Studio里的
    # 复查任务链接的还是旧文件名但内容已经被换掉，数据会错乱。加这个标签
    # 保证跨运行文件名互不冲突，而且是确定性的、能看出源头，不用随机UUID
    # （UUID虽然也能保证唯一，但看不出是哪次运行产出的，出问题不好排查）。
    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    cam1_clip_mp4 = os.path.join(clip_dir, stem1 + suffix + run_suffix + ".mp4")
    cam2_clip_mp4 = os.path.join(clip_dir, stem2 + suffix + run_suffix + ".mp4")
    cam1_clip_csv = os.path.join(clip_dir, detected_stem + suffix + run_suffix + ".csv")  # 检测狗的 CSV

    windows1 = task.get("windows1")  # cam1自己的imu1逐窗口置信度（可能为空=没有对应_infer.json）
    windows2 = task.get("windows2")  # cam2自己的imu2逐窗口置信度
    ok_cam1 = cut_clip(cam1_mp4, start_sec, end_sec, cam1_clip_mp4, args.context_s, video_t0,
                       windows=windows1, label_text="IMU1") if (cam1_mp4 and has_ffmpeg) else False
    ok_cam2 = cut_clip(cam2_mp4, start_sec, end_sec, cam2_clip_mp4, args.context_s, video_t0,
                       windows=windows2, label_text="IMU2") if (cam2_mp4 and has_ffmpeg) else False
    ok_csv  = slice_csv(src_csv1, cam1_clip_csv, pad_start, pad_end) if os.path.exists(src_csv1) else False

    if secondary_dir:
        if ok_cam1:
            _symlink_into(cam1_clip_mp4, secondary_dir)
        if ok_cam2:
            _symlink_into(cam2_clip_mp4, secondary_dir)
        if ok_csv:
            _symlink_into(cam1_clip_csv, secondary_dir)

    return {**task, "ok_cam1": ok_cam1, "ok_cam2": ok_cam2, "ok_csv": ok_csv}


def main():
    parser = argparse.ArgumentParser(description="按置信度区间裁剪抓挠视频片段（witmotion_imu 格式）")
    parser.add_argument("--infer_dir",  required=True)
    parser.add_argument("--video_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_s",  type=float, default=3.0)
    parser.add_argument("--workers",    type=int,   default=4,
                        help="并行裁剪线程数（默认 4）。裁剪固定用CPU软编码libx264"
                             "（见下方说明），不是GPU瓶颈，核数多的机器可以调大，"
                             "比如16核以上的机器可以试试 8~16")
    parser.add_argument("--bin_by", default="conf_max", choices=["conf_max", "conf_mean", "both"],
                        help="置信度分桶依据：conf_max=片段内最高置信度（默认），conf_mean=片段内平均置信度，"
                             "both=两套分桶同时保留（conf_max找漏检更方便、conf_mean误报少更稳，"
                             "两个都想看的时候用这个，不用跑两遍）。both模式下视频只真正裁剪一次，"
                             "第二套分桶目录里放的是软链接，不会重复编码、不会占用双倍磁盘")
    parser.add_argument("--no_conf_overlay", action="store_true",
                        help="不烧录右上角实时置信度字幕（默认会烧，读取每路IMU自己的"
                             "{stem}_infer.json里的逐窗口置信度）。传这个可以跳过读取/"
                             "烧字幕的开销，适合只是想快速看效果、不需要置信度曲线的场景")
    parser.add_argument("--run_tag", default="",
                        help="拼进裁剪出的mp4/csv文件名末尾的标识（建议传这次运行用的"
                             "RESULT_ROOT名字，比如'infer_result_majority_syn'）。"
                             "不同模型/不同次运行如果检测到时间重叠的事件，文件名默认"
                             "只由CSV名+第几段+起止时间决定，会完全一样，复制到共享的"
                             "Nginx媒体目录时后一次会静默覆盖前一次，导致Label Studio里"
                             "复查任务链接的文件名对不上实际内容。留空则不加（跟以前"
                             "行为一致），多次运行结果会共用同一个媒体目录时强烈建议传")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer_jsons = sorted(glob.glob(os.path.join(args.infer_dir, "*_infer.json")))
    if not infer_jsons:
        print(f"[错误] {args.infer_dir} 下没有 *_infer.json")
        sys.exit(1)

    has_ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    if not has_ffmpeg:
        print("[警告] 找不到 ffmpeg，只裁剪 CSV")

    # 注意：即使检测到有 CUDA(h264_nvenc)，cut_clip() 里实际固定用的是
    # CPU软编码 libx264（nvenc 输出在部分浏览器兼容性有问题，见 cut_clip() 里
    # 的注释），之前这里打印"编码器: CUDA"是误导性的——GPU利用率显示0%不是
    # 没生效，是压根没用GPU，这是纯CPU瓶颈的活，加并行线程数才有用，加GPU没用。
    print(f"编码器: CPU (libx264，固定软编码，即使有CUDA也不用——nvenc输出兼容性有问题)"
          f"  并行线程: {args.workers}"
          f"{'（检测到有CUDA硬件加速但没用上，纯CPU编码，是有意为之）' if HAS_CUDA else ''}")

    # ── 收集所有 clip 任务 ────────────────────────────────────────────────────
    bin_dirs  = {}
    lock      = threading.Lock()
    all_tasks = []

    for infer_path in infer_jsons:
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)
        csv_basename = data["csv_basename"]
        segs = data.get("scratch_segments", [])
        if not segs:
            continue

        # 判断是哪只狗的 IMU：cam1_imu1=狗1，cam2_imu2=狗2
        detected_stem = os.path.splitext(csv_basename)[0]
        is_cam2 = bool(re.search(r"cam2_imu2", csv_basename))
        if is_cam2:
            # 狗2检测到抓挠：cam1=视角1(兄弟), cam2=检测狗
            stem2 = detected_stem                  # cam2_imu2（检测狗）
            stem1 = sibling_stem(detected_stem, 1) # cam1_imu1（视角1兄弟）
        else:
            # 狗1检测到抓挠：cam1=检测狗, cam2=视角2(兄弟)
            stem1 = detected_stem                  # cam1_imu1（检测狗）
            stem2 = sibling_stem(detected_stem, 2) # cam2_imu2（视角2兄弟）
        src_csv1 = os.path.join(args.video_dir, csv_basename)
        cam1_mp4 = find_video(stem1, args.video_dir)
        cam2_mp4 = find_video(stem2, args.video_dir)
        video_t0 = csv_start_sec(src_csv1) if os.path.exists(src_csv1) else 0.0

        # 各自IMU自己的逐窗口置信度，用于烧录右上角实时置信度字幕；
        # stem1/stem2各自有独立的_infer.json（两路IMU分别跑的推理），
        # 不是检测触发的那一路也应该有自己的推理结果文件，除非那一路
        # 压根没跑推理，这时优雅降级成不烧字幕（cut_clip里windows为空会跳过）
        windows1 = load_windows_for_stem(args.infer_dir, stem1) if not args.no_conf_overlay else []
        windows2 = load_windows_for_stem(args.infer_dir, stem2) if not args.no_conf_overlay else []

        for idx, seg in enumerate(segs, 1):
            t0_str = seg.get("start_ts", "") or ""
            t1_str = seg.get("end_ts",   "") or ""
            if not t0_str:
                continue

            conf_mean = seg.get("conf_mean", 0.0)
            conf_max  = seg.get("conf_max",  0.0)
            is_both   = args.bin_by == "both"
            bin_conf  = conf_max if args.bin_by in ("conf_max", "both") else conf_mean
            bin_label = get_bin(bin_conf)  # both模式下这是canonical(conf_max)分桶，实际裁剪落在这里
            bin_label_mean = get_bin(conf_mean) if is_both else None
            suffix    = f"_clip{idx:02d}_{ts_to_hhmmss(t0_str)}-{ts_to_hhmmss(t1_str or t0_str)}"

            root_max = os.path.join(args.output_dir, "by_conf_max") if is_both else args.output_dir
            with lock:
                if bin_label not in bin_dirs:
                    clip_dir = os.path.join(root_max, f"clips_{bin_label}")
                    os.makedirs(clip_dir, exist_ok=True)
                    bin_dirs[bin_label] = clip_dir
                clip_dir = bin_dirs[bin_label]

            secondary_dir = None
            if is_both:
                secondary_dir = os.path.join(args.output_dir, "by_conf_mean", f"clips_{bin_label_mean}")
                with lock:
                    os.makedirs(secondary_dir, exist_ok=True)

            all_tasks.append({
                "args": args, "has_ffmpeg": has_ffmpeg,
                "stem1": stem1, "stem2": stem2,
                "detected_stem": detected_stem,  # CSV clip 用检测狗的 stem
                "bin_label_mean": bin_label_mean, "secondary_dir": secondary_dir,
                "src_csv1": src_csv1, "cam1_mp4": cam1_mp4, "cam2_mp4": cam2_mp4,
                "video_t0": video_t0, "clip_dir": clip_dir, "suffix": suffix,
                "windows1": windows1, "windows2": windows2,
                "seg": {"start_ts": t0_str, "end_ts": t1_str,
                        "conf_mean": conf_mean, "conf_max": conf_max},
                "csv_basename": csv_basename, "bin_label": bin_label,
                "bin_by": args.bin_by,
            })

    print(f"共 {len(all_tasks)} 个 clip 任务，开始并行处理...")

    # ── 并行执行 ─────────────────────────────────────────────────────────────
    from tqdm import tqdm
    is_both = args.bin_by == "both"
    bin_logs      = {bl: [] for bl in bin_dirs}       # conf_max（both模式下是canonical分桶）
    bin_logs_mean = {}                                 # 仅both模式用：conf_mean那一套
    done = 0
    pbar = tqdm(total=len(all_tasks), desc="裁剪片段", unit="clip")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one, t): t for t in all_tasks}
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            pbar.update(1)
            seg       = res["seg"]
            t0_str    = seg["start_ts"]
            t1_str    = seg["end_ts"]
            conf_mean = seg["conf_mean"]
            conf_max  = seg["conf_max"]
            bin_label = res["bin_label"]
            bin_label_mean = res.get("bin_label_mean")
            stem1     = res["stem1"]
            suffix    = res["suffix"]
            bin_by    = res["bin_by"]

            parts = [
                "cam1✅" if res["ok_cam1"] else ("cam1⚠️" if not res["cam1_mp4"] else "cam1❌"),
                "cam2✅" if res["ok_cam2"] else ("cam2⚠️" if not res["cam2_mp4"] else "cam2❌"),
                "csv✅"  if res["ok_csv"]  else "csv❌",
            ]
            status = " ".join(parts)
            if is_both:
                conf_str = f"max={conf_max:.2f} mean={conf_mean:.2f}"
            else:
                conf_str = f"{bin_by}={(conf_max if bin_by == 'conf_max' else conf_mean):.2f}"
            tqdm.write(f"  [{done}/{len(all_tasks)}] [{bin_label}] {res['csv_basename']}  "
                       f"{t0_str[11:19]}→{t1_str[11:19] if t1_str else '?'}  "
                       f"{conf_str}  {status}")

            line = (f"{res['csv_basename']}\t{t0_str[11:19]}\t{t1_str[11:19] if t1_str else '?'}"
                    f"\tconf_mean={conf_mean:.3f}\tconf_max={conf_max:.3f}"
                    f"\t{stem1 + suffix + '.mp4'}\t{status}")
            with lock:
                bin_logs[bin_label].append(line)
                if is_both:
                    bin_logs_mean.setdefault(bin_label_mean, []).append(line)
    pbar.close()

    max_root = os.path.join(args.output_dir, "by_conf_max") if is_both else args.output_dir
    for bin_label, lines in sorted(bin_logs.items()):
        log_path = os.path.join(max_root, f"scratch_log_review_{bin_label}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("csv_file\tstart\tend\tconf_mean\tconf_max\tclip_file\tstatus\n")
            f.write("\n".join(lines) + "\n")
        print(f"  → {log_path}  ({len(lines)} 条)")

    if is_both:
        mean_root = os.path.join(args.output_dir, "by_conf_mean")
        for bin_label, lines in sorted(bin_logs_mean.items()):
            log_path = os.path.join(mean_root, f"scratch_log_review_{bin_label}.txt")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("csv_file\tstart\tend\tconf_mean\tconf_max\tclip_file\tstatus\n")
                f.write("\n".join(lines) + "\n")
            print(f"  → {log_path}  ({len(lines)} 条)")

    total = sum(len(v) for v in bin_logs.values())
    both_note = ""
    if is_both:
        both_note = ("（按conf_max分桶，实际裁剪落在 by_conf_max/；"
                      "by_conf_mean/ 下是同样这些片段按conf_mean重新分桶后的软链接视图）")
    print(f"\n共 {total} 段抓挠，分布在 {len(bin_logs)} 个置信度区间{both_note}")
    print(f"  按 conf_max 分桶（by_conf_max/）：")
    for bl in sorted(bin_logs):
        print(f"    clips_{bl}/: {len(bin_logs[bl])} 段")

    if is_both:
        print(f"  按 conf_mean 分桶（by_conf_mean/）：")
        for bl in sorted(bin_logs_mean):
            print(f"    clips_{bl}/: {len(bin_logs_mean[bl])} 段")


if __name__ == "__main__":
    main()
