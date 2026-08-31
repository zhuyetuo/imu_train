"""
把推理结果或裁剪片段转换为 Label Studio 任务。

两种模式:
  --use_clips   扫描 clips_*/ 目录，每个 clip 若干件套(cam1.mp4 + cam2.mp4
                [+ cam3.mp4 ...] + cam1.csv) 生成一个 Label Studio 任务
                （推荐，与 witmotion_imu 格式一致）。机位数量不固定——
                有几个机位就生成video1..videoN几个字段，2机位的场次只有
                video1/video2，3机位（或以后更多）的场次会自动多出
                video3/video4...，不用为新增机位改代码。
  默认          从 *_infer.json 生成任务（引用完整录制文件）

Label Studio 项目配置:
  video1 = cam1 视频, video2 = cam2 视频, video3 = cam3 视频（如果有）,
  csv1 = cam1 IMU CSV, from_name="label", to_name="ts"
  ——新增机位时，Label Studio项目的Labeling Interface配置里要记得加上
  对应的 <Video name="video3" value="$video3"/> 才能实际显示出来，这个
  是Label Studio项目配置本身的事，这份脚本只负责把URL放进task JSON。

用法:
  # 裁剪片段模式（推荐）
  python src/review_to_labelstudio.py \
    --infer_dir infer_result/2026_7_17 \
    --output infer_result/2026_7_17/labelstudio_review.json \
    --use_clips \
    --csv_url_prefix http://192.168.2.140:8182

  # 全录制模式
  python src/review_to_labelstudio.py \
    --infer_dir infer_result/2026_7_17/_infer \
    --output infer_result/2026_7_17/labelstudio_review.json \
    --csv_url_prefix http://192.168.2.140:8182
"""

import argparse
import glob
import json
import os
import re


def _cam_mode_type(value: str) -> str:
    """--cam_mode的取值校验："auto"或任意正整数字符串（"2"/"3"/"4"/...）。
    之前用choices=["auto","2","3"]写死只认2或3，机位数以后涨到4个的话
    就得先改代码——机位数量本来就是会变的东西（cam3是这次会话里才新增
    的），这里的校验逻辑不该跟着写死，改成"auto或任意正整数"就不用再为
    新增机位改这个文件了。"""
    if value == "auto":
        return value
    if value.isdigit() and int(value) >= 1:
        return value
    raise argparse.ArgumentTypeError(f'必须是 "auto" 或正整数（比如 "2"/"3"/"4"），收到: {value!r}')


# ── 文件名解析 ────────────────────────────────────────────────────────────────

def parse_cam(basename: str) -> str:
    """返回 'cam1' / 'cam2' / ''"""
    m = re.search(r"(cam\d)", basename, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def parse_imu_label(basename: str) -> str:
    """提取触发检测的IMU标识，比如
    ..._cam1_imu1_raw_byIMU3_clip01_...mp4 → 'IMU3'。
    mp4文件名带"_by{IMU}"（见extract_clips.py的imu_tag），csv文件名不带
    （csv的stem里已经有实际IMU编号），取不到就返回''（比如老产出的文件名
    没有这个标识，或者传进来的是csv文件名）。"""
    m = re.search(r"_by([A-Za-z0-9]+)", basename)
    return m.group(1) if m else ""


def parse_imu_num(stem: str):
    """从CSV的stem里取狗身上IMU的编号，比如
    multicam_20260814_000000560_cam1_imu3_raw → 3。取不到返回None。

    注意这跟机位号(camN)是两回事：房间固定就2个机位，但可以同时挂4条狗，
    同一个cam1下面会有 cam1_imu1 / cam1_imu3 / cam1_imu4 三条不同狗的CSV
    （见extract_clips.py的session_prefix注释）。按cam号当IMU号用的话，
    同一个cam下的多条狗会互相覆盖。"""
    m = re.search(r"_imu(\d+)(?:_|$)", stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def camera_video_stem_of(stem: str, cam_num: int) -> str:
    """把任意一条这个session的CSV stem，换算成第cam_num个机位的视频文件
    stem。机位视频的文件名后缀固定是 camN_imuN（cam1_imu1/cam2_imu2/...，
    见extract_clips.camera_video_stem），不管这个session实际挂了几条狗，
    所以把末尾的 _cam*_imu* 替换掉就行。替换不了(文件名不是这个格式)时
    原样返回，行为不会比以前更差。"""
    new_stem, n = re.subn(r"_cam\d+_imu\d+", f"_cam{cam_num}_imu{cam_num}", stem, count=1)
    return new_stem if n else stem


def session_key(basename: str) -> str:
    """提取会话前缀（cam_tag 之前的部分）"""
    stem = os.path.splitext(basename)[0]
    m = re.search(r"(cam\d)", stem, re.IGNORECASE)
    if not m:
        return stem
    return stem[: m.start()].rstrip("_")


def clip_key(basename: str) -> str:
    """提取 clip 唯一键（去掉 cam1/cam2 差异，以及触发检测的IMU标识）
    multicam_20260717_185620_cam1_imu1_resampled16hz_clip01_185907-185910
    → multicam_20260717_185620_clip01_185907-185910

    extract_clips.py 现在支持3条以上的狗共用固定2机位视频：mp4文件名会带
    "_by{IMU编号}"标识是哪条狗触发的检测（比如_byIMU3），但csv文件名不带
    这个标识（csv本身的stem里已经有实际IMU编号了，不需要再加）。这里必须
    把"_byIMU..."也去掉，不然同一个clip的mp4和csv算出来的key不一样，
    会被误判成两个不同的clip，导致有的任务缺视频、有的缺csv。
    """
    stem = os.path.splitext(basename)[0]
    stem = re.sub(r"_cam\d_imu\d", "", stem)
    stem = re.sub(r"_by[A-Za-z0-9]+", "", stem)
    return stem


# ── URL 构建 ──────────────────────────────────────────────────────────────────

def parse_bin_lo(bin_name: str) -> float:
    """从meta.bin(比如'clips_0.7-0.8')解析出这个置信度桶的下界0.7。
    extract_clips.py按BINS=[0.0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01]分桶，
    桶边界本身就是常用阈值(0.3/0.5/0.7等)，用"桶下界>=阈值"筛选等价于
    "这个clip的置信度>=阈值"，不需要额外去读scratch_log_review_*.txt里
    逐clip的精确conf_max/conf_mean数值。解析不出来时返回-1（保守地当成
    "置信度未知"，任何正数阈值筛选都会把它排除掉，不会误留进结果里）。"""
    m = re.search(r"(\d+\.\d+)-", bin_name)
    return float(m.group(1)) if m else -1.0


def csv_url(basename: str, csv_prefix: str) -> str:
    return f"{csv_prefix.rstrip('/')}/{basename}"


def video_url(basename: str, video_prefix: str) -> str:
    stem = os.path.splitext(basename)[0]
    return f"{video_prefix.rstrip('/')}/{stem}.mp4"


# ── 标注生成 ──────────────────────────────────────────────────────────────────

def make_annotation(start_ts, end_ts, label):
    return {
        "type": "timeserieslabels",
        "from_name": "label",
        "to_name": "ts",
        "value": {
            "start": start_ts or "",
            "end":   end_ts   or "",
            "timeserieslabels": [label],
        }
    }


# ── 模式 1：clips_*/ 目录扫描 ─────────────────────────────────────────────────

def build_tasks_from_clips(infer_dir, csv_url_prefix, video_url_prefix, label_name, cam_mode="auto"):
    """
    扫描 infer_dir/clips_*/ 目录，按 clip 键分组：
      camN mp4 → videoN（N按机位号动态生成，不写死1/2两个）
      camN csv（优先cam1，没有就取分组里任意一个有csv的机位）→ csv1 + 标注

    cam_mode 控制每个任务固定生成几个videoN字段：
      "auto"（默认）：每个clip组按自己实际探测到的最高机位号生成，
                     组跟组之间video字段数量可能不一样
      "2" / "3"：不管这个clip组实际找到几个机位的视频，固定生成
                video1..videoN这N个字段——Label Studio项目如果配置成
                固定N个Video组件，任务JSON里缺哪个key就会导入报错，
                所以要按项目实际配置传这个参数，保证每个任务的字段
                数量一致。真的缺失的机位（比如原始只拍了2个视角但
                cam_mode=3）该字段值给空字符串""占位，不是拿别的
                机位视频顶替（避免误导复查人员"这就是cam3的画面"）。
    """
    tasks = []
    task_id = 1

    clip_dirs = sorted(glob.glob(os.path.join(infer_dir, "clips_*")))
    if not clip_dirs:
        print(f"[警告] {infer_dir} 下没有 clips_*/ 目录，请先运行 extract_clips.py")
        return tasks

    for clip_dir in clip_dirs:
        # 按 clip_key 分组，每组里 mp4_by_cam/csv_by_cam 存 {"cam1": fname, "cam2": fname, ...}
        # 机位号不固定——遇到cam3/cam4的文件也会自动分进去，不需要为新机位改代码
        groups = {}

        for fname in sorted(os.listdir(clip_dir)):
            fpath = os.path.join(clip_dir, fname)
            if not os.path.isfile(fpath):
                continue
            cam = parse_cam(fname) or "cam1"  # 老文件名没有cam标识时按cam1处理，兼容以前的产出
            key = clip_key(fname)
            g = groups.setdefault(key, {"mp4": {}, "csv": {}, "imu_label": ""})
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".mp4", ".avi", ".mov"):
                g["mp4"][cam] = fname
                if not g["imu_label"]:
                    g["imu_label"] = parse_imu_label(fname)  # mp4文件名带_by{IMU}，csv不带
            elif ext == ".csv":
                g["csv"][cam] = fname

        for key in sorted(groups):
            g = groups[key]
            mp4_by_cam = g["mp4"]
            csv_by_cam = g["csv"]

            # csv1 = 检测狗的 CSV，优先取cam1，没有就取分组里随便哪个机位有的
            cam1_csv_name = csv_by_cam.get("cam1") or next(iter(csv_by_cam.values()), "")

            if not mp4_by_cam and not cam1_csv_name:
                continue

            # 映射成video1/video2/video3...：cam_mode="auto"时字段数量跟着
            # 这个clip组实际探测到的最高机位号走；cam_mode="2"/"3"时不管
            # 实际找到几个机位，固定生成这么多个videoN字段，保证同一批任务
            # 的字段数量一致，匹配Label Studio项目固定配置的Video组件个数。
            # 真正缺失的机位给空字符串""占位，不拿别的机位视频顶替。
            cam_nums_present = {int(re.search(r"\d+", c).group()): c for c in mp4_by_cam}
            if cam_mode == "auto":
                slot_count = max(cam_nums_present) if cam_nums_present else 1
            else:
                slot_count = int(cam_mode)
            urls_by_num = {n: video_url(mp4_by_cam[c], video_url_prefix)
                          for n, c in cam_nums_present.items()}
            task_data = {f"video{n}": urls_by_num.get(n, "")
                        for n in range(1, slot_count + 1)}
            task_data["csv1"] = csv_url(cam1_csv_name, csv_url_prefix) if cam1_csv_name else ""

            # 从 key 解析日期和时间（如 multicam_20260717_185620_clip03_190408-190410）
            # 日期从会话前缀中提取：multicam_YYYYMMDD_...
            date_m = re.search(r"(\d{4})(\d{2})(\d{2})_\d{6}", key)
            # 锚定在"_clipNN_"后面，而不是用$锚定字符串结尾——extract_clips.py的
            # --run_tag会在文件名末尾追加运行标识（比如_infer_result_majority_syn），
            # 时间戳段就不再是key的结尾了，用$锚定会匹配失败、导致annotations
            # 拿不到起止时间、整个标注是空的（这个bug的直接表现）
            time_m = re.search(r"_clip\d+_(\d{6})-(\d{6})", key)
            if date_m and time_m:
                date_str = f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}"
                def hms_to_ts(hms: str) -> str:
                    return f"{date_str} {hms[:2]}:{hms[2:4]}:{hms[4:6]}.000"
                start_ts = hms_to_ts(time_m.group(1))
                end_ts   = hms_to_ts(time_m.group(2))
                results = [make_annotation(start_ts, end_ts, label_name)]
            else:
                results = []

            tasks.append({
                "id":   task_id,
                "data": task_data,
                "annotations": [{"result": results}] if results else [],
                "meta": {
                    "clip_key":  key,
                    "bin":       os.path.basename(clip_dir),
                    "imu_label": g["imu_label"],  # 触发检测的IMU，比如"IMU3"，可能是""（老产出/取不到）
                    "note":      "模型检测到抓挠片段，请核实",
                }
            })
            task_id += 1

    return tasks


# ── 模式 2：_infer.json 扫描（全录制文件）────────────────────────────────────

def build_tasks_from_infer(infer_jsons, csv_url_prefix, video_url_prefix,
                           mode, low_threshold, high_threshold, label_name):
    sessions = {}
    for infer_path in sorted(infer_jsons):
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)
        csv_basename = data["csv_basename"]
        sess = session_key(csv_basename)
        cam  = parse_cam(csv_basename)
        if sess not in sessions:
            sessions[sess] = {}
        sessions[sess][cam or "cam1"] = data

    tasks = []
    task_id = 1

    for sess in sorted(sessions):
        cams = sessions[sess]
        main_data = cams.get("cam1") or next(iter(cams.values()))
        main_csv  = main_data["csv_basename"]
        stem1     = os.path.splitext(main_csv)[0]
        stem2     = re.sub(r"cam1_imu1", "cam2_imu2", stem1)

        c1_url = csv_url(main_csv, csv_url_prefix)
        v1_url = f"{video_url_prefix.rstrip('/')}/{stem1}.mp4"
        v2_url = f"{video_url_prefix.rstrip('/')}/{stem2}.mp4" if "cam2" in cams else v1_url

        task_data = {"csv1": c1_url, "video1": v1_url, "video2": v2_url}

        scratch_segs = main_data.get("scratch_segments", [])
        windows      = main_data.get("windows", [])
        results      = []

        if mode in ("scratch_only", "all") and scratch_segs:
            for seg in scratch_segs:
                if seg.get("start_ts") and seg.get("end_ts"):
                    results.append(make_annotation(seg["start_ts"], seg["end_ts"], label_name))

        if mode in ("uncertain", "all"):
            for w in windows:
                prob = w.get("probs", {}).get(label_name, 0.0)
                if w.get("label") != label_name and low_threshold < prob <= high_threshold:
                    if w.get("ts"):
                        results.append(make_annotation(w["ts"], w["ts"], f"{label_name}?"))

        if not results:
            continue

        tasks.append({
            "id":   task_id,
            "data": task_data,
            "annotations": [{"result": results}],
            "meta": {
                "session":  sess,
                "csv_file": main_csv,
                "note":     f"模型检测到 {len(scratch_segs)} 段抓挠，请核实",
            }
        })
        task_id += 1

    return tasks


# ── 模式 3：全录制文件 + 高置信度自动预标注(标注人=ML) ───────────────────────

def build_tasks_from_infer_ml(infer_jsons, csv_url_prefix, video_url_prefix, label_name,
                              min_conf=0.8, conf_field="conf_max", cam_mode="auto"):
    """
    在原始完整录制视频上直接生成"模型预标注"任务，不裁剪clip：
      - 每个session(不管有没有检测到达标片段)都生成一个task，让复查的人
        能看到全部录制数据、不只是模型觉得有抓挠的那部分——目的是让人
        能顺便核查"模型是不是有漏检"，只导出命中的片段做不到这一点
      - 只保留置信度(conf_field，默认conf_max)>=min_conf的抓挠片段进
        annotations，其余片段不出现在标注结果里(不是标成"低置信度"，
        是直接不标)；一个片段都没达标时，这个task的annotations是空
        列表(未标注状态)，不是塞一个"result为空的annotation"进去——
        那样Label Studio会显示成"已完成"，容易让复查人误以为这段已经
        看过了，直接跳过，跟"提醒核查漏检"这个目的正好相反
      - 机位处理跟build_tasks_from_clips一样支持cam_mode(auto/2/3)
      - 标注人固定标成"ML"（completed_by），跟人工标注区分开
    """
    # 一个session里：每条狗(IMU)有自己的CSV，机位(cam)有自己的视频，两者
    # 不是一回事——房间固定就2个机位，但可以同时挂4条狗，文件名会出现
    # cam1_imu1 / cam1_imu3 / cam1_imu4 / cam2_imu2 这样的组合。所以要
    # 分开收集：imus按IMU编号存各自的CSV和检测结果，cam_nums_seen只是
    # 记录这个session里出现过哪些机位编号(给cam_mode=auto估机位数用)。
    # 之前这里按cam分组当IMU用，同一个cam下的多条狗会互相覆盖，只剩最后
    # 读到的那一条。
    sessions = {}
    all_cam_nums_seen = set()  # 整个数据集里出现过的机位号，不分session——
                               # 用来判断某个机位是"结构性根本不存在"还是
                               # "只是这个session凑巧数据缺失"，见下面的说明
    for infer_path in sorted(infer_jsons):
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)
        csv_basename = data["csv_basename"]
        sess = session_key(csv_basename)
        stem = os.path.splitext(csv_basename)[0]

        entry = sessions.setdefault(sess, {"imus": {}, "cam_nums_seen": set(), "any_stem": stem})

        imu_num = parse_imu_num(stem)
        if imu_num is not None:
            entry["imus"][imu_num] = data

        cam = parse_cam(csv_basename)
        if cam:
            cam_num = int(re.search(r"\d+", cam).group())
            entry["cam_nums_seen"].add(cam_num)
            all_cam_nums_seen.add(cam_num)

    tasks = []
    task_id = 1

    for sess in sorted(sessions):
        imus_data = sessions[sess]["imus"]
        any_stem = sessions[sess]["any_stem"]

        if cam_mode == "auto":
            slot_count = max(sessions[sess]["cam_nums_seen"]) if sessions[sess]["cam_nums_seen"] else 1
        else:
            slot_count = int(cam_mode)

        # video1..videoN 是这个session共享的，跟具体用哪条狗的CSV无关，
        # 所以只算一次，下面每个IMU的task都复用同一份。机位视频的文件名
        # 固定是camN_imuN(见extract_clips.camera_video_stem)，用any_stem
        # (这个session里随便哪个IMU的CSV都行，前缀+后缀是共享的)就能推算
        # 出来，不要求camN自己这个session的数据必须存在——之前按"cam_num
        # 是否在这个session读到的数据里"决定要不要生成video{n}，如果那个
        # 机位designated的IMU(比如cam2的imu2)这次推理没跑出结果(文件
        # 损坏/为空之类)，即使摄像头本身正常录了像，也会导致video2被漏填
        # 成空字符串——摄像头视频存不存在，跟某一路IMU传感器数据完不完整，
        # 是两件不该混在一起判断的事。
        #
        # 但"这个session缺数据"和"这个机位整个数据集里压根不存在"也要分开：
        # 用all_cam_nums_seen(整个数据集，不分session)判断——camN只要在
        # 别的session出现过，就说明这个机位真实存在，这个session虽然缺数据
        # 也照样按规律推算视频名(上面那种情况)；但如果camN在整个数据集里
        # 一次都没出现过(比如CAM_MODE传的机位数比实际录制的机位数还多，
        # 这次只有3个机位却传了CAM_MODE=4)，说明这个机位是"从来就不存在"，
        # 那就该照老规矩给空字符串占位，不能瞎猜一个大概率404的URL出来
        video_urls = {}
        for n in range(1, slot_count + 1):
            if n in all_cam_nums_seen:
                cam_stem = camera_video_stem_of(any_stem, n)
                video_urls[f"video{n}"] = f"{video_url_prefix.rstrip('/')}/{cam_stem}.mp4"
            else:
                video_urls[f"video{n}"] = ""

        # 每条狗(IMU)自己的CSV、自己的检测结果各生成一个task——它们各有
        # 独立的CSV和抓挠检测，混在一起或者互相顶掉都会导致复查的时候
        # 看不到某条狗的检测结果
        for imu_num in sorted(imus_data):
            imu_data = imus_data[imu_num]
            cam_csv = imu_data["csv_basename"]
            imu_label = f"IMU{imu_num}"

            task_data = {"csv1": csv_url(cam_csv, csv_url_prefix)}
            task_data.update(video_urls)

            scratch_segs = imu_data.get("scratch_segments", [])
            # 只保留置信度达标的片段——不是"标出来但打个低分标签"，是这批
            # 片段压根不出现在标注结果里，跟clips模式的--min_conf是同一个
            # "额外筛一份高置信度子集"的思路，只是这里作用在整段视频的标注
            # 内容上，不是文件层面的筛选
            kept = [seg for seg in scratch_segs
                   if seg.get(conf_field, 0.0) >= min_conf and seg.get("start_ts") and seg.get("end_ts")]
            results = [make_annotation(seg["start_ts"], seg["end_ts"], label_name) for seg in kept]

            # 不管results是不是空都生成task——让复查人能看到全部录制数据，
            # 用来核查漏检，不是只有命中的才值得导出。但"没有达标片段"跟
            # "标注了、标注内容是空"是两回事：没检测到就是没有标注，
            # annotations给空列表(未标注状态)，不能塞一个"result为空的
            # annotation"进去冒充"已经标注完、判定为无"，那样Label Studio
            # 里会显示成"已完成"，反而让人误以为这段已经复查过，不会再点
            # 开看，跟"提醒核查漏检"这个目的正好相反。
            if results:
                note = f"模型ML自动检测，{conf_field}>={min_conf}，共{len(results)}段（原始{len(scratch_segs)}段中筛出）"
                annotations = [{
                    "completed_by": {"email": "ml@model.local", "first_name": "ML", "last_name": "Model"},
                    "result": results,
                }]
            elif scratch_segs:
                note = (f"模型记录到{len(scratch_segs)}段疑似抓挠，但{conf_field}都低于{min_conf}阈值，"
                       f"未生成标注，请人工核实是否漏检")
                annotations = []
            else:
                note = "模型ML完全没有检测到抓挠片段，未生成标注，请人工核实是否漏检"
                annotations = []
            tasks.append({
                "id": task_id,
                "data": task_data,
                "annotations": annotations,
                "meta": {
                    "session": sess, "csv_file": cam_csv,
                    "imu_label": imu_label,
                    "labeled_by": "ML",
                    "note": note,
                }
            })
            task_id += 1

    return tasks


def build_tasks_from_infer_ml_multi(infer_root, labels, csv_url_prefix, video_url_prefix,
                                    min_conf=0.8, conf_field="conf_max", cam_mode="auto"):
    """
    跟build_tasks_from_infer_ml是同一个"全录制视频+ML预标注"的思路，区别是
    这个函数一次性合并多个类别（比如活动/睡觉/抓挠/未佩戴）的检测结果到
    同一批task里，每个类别的片段各自打上自己的timeserieslabels——之前
    build_tasks_from_infer_ml一次只认一个label_name，多类别时每个类别
    各自输出一份独立的_full_ml_IMU*.json，同一个IMU要点开4个文件才能
    分别看到4个类别的检测结果；这个函数让同一条狗的同一份录制在Label
    Studio里一个task、一条时间轴上就能同时看到4个类别各自标出来的片段，
    不用来回切文件对照。

    infer_root: RESULT_ROOT/{day}这一级目录，下面按类别各有一个子目录
    （RESULT_ROOT/{day}/{label}/_infer/*.json，跟run_review_bins_all_days.sh
    的目录结构一致），不是某一个类别自己的_infer目录。
    labels: 要合并的类别列表，比如["活动","睡觉","抓挠","未佩戴"]。
    """
    # 每个session+每条IMU收集来自多个类别的infer数据：
    # sessions[sess]["imus"][imu_num][label] = 该类别这条IMU的infer json内容
    sessions = {}
    all_cam_nums_seen = set()

    for label in labels:
        label_dir = os.path.join(infer_root, label, "_infer")
        infer_jsons = sorted(glob.glob(os.path.join(label_dir, "*_infer.json")))
        for infer_path in infer_jsons:
            with open(infer_path, encoding="utf-8") as f:
                data = json.load(f)
            csv_basename = data["csv_basename"]
            sess = session_key(csv_basename)
            stem = os.path.splitext(csv_basename)[0]

            entry = sessions.setdefault(sess, {"imus": {}, "cam_nums_seen": set(), "any_stem": stem})

            imu_num = parse_imu_num(stem)
            if imu_num is not None:
                entry["imus"].setdefault(imu_num, {})[label] = data

            cam = parse_cam(csv_basename)
            if cam:
                cam_num = int(re.search(r"\d+", cam).group())
                entry["cam_nums_seen"].add(cam_num)
                all_cam_nums_seen.add(cam_num)

    tasks = []
    task_id = 1

    for sess in sorted(sessions):
        imus_data = sessions[sess]["imus"]
        any_stem = sessions[sess]["any_stem"]

        if cam_mode == "auto":
            slot_count = max(sessions[sess]["cam_nums_seen"]) if sessions[sess]["cam_nums_seen"] else 1
        else:
            slot_count = int(cam_mode)

        video_urls = {}
        for n in range(1, slot_count + 1):
            if n in all_cam_nums_seen:
                cam_stem = camera_video_stem_of(any_stem, n)
                video_urls[f"video{n}"] = f"{video_url_prefix.rstrip('/')}/{cam_stem}.mp4"
            else:
                video_urls[f"video{n}"] = ""

        for imu_num in sorted(imus_data):
            by_label = imus_data[imu_num]
            # 同一条IMU的csv_basename在各个类别下应该完全一样(同一份原始
            # 录制文件)，随便取一个类别的就行，不需要每个类别各自核对
            cam_csv = next(iter(by_label.values()))["csv_basename"]
            imu_label = f"IMU{imu_num}"

            task_data = {"csv1": csv_url(cam_csv, csv_url_prefix)}
            task_data.update(video_urls)

            results = []
            note_parts = []
            for label in labels:
                data = by_label.get(label)
                scratch_segs = data.get("scratch_segments", []) if data else []
                kept = [seg for seg in scratch_segs
                       if seg.get(conf_field, 0.0) >= min_conf and seg.get("start_ts") and seg.get("end_ts")]
                results.extend(make_annotation(seg["start_ts"], seg["end_ts"], label) for seg in kept)
                note_parts.append(f"{label}:{len(kept)}/{len(scratch_segs)}段达标")

            note = f"模型ML自动检测(多类别合并)，{conf_field}>={min_conf}，" + "  ".join(note_parts)
            annotations = [{
                "completed_by": {"email": "ml@model.local", "first_name": "ML", "last_name": "Model"},
                "result": results,
            }] if results else []

            tasks.append({
                "id": task_id,
                "data": task_data,
                "annotations": annotations,
                "meta": {
                    "session": sess, "csv_file": cam_csv,
                    "imu_label": imu_label,
                    "labeled_by": "ML",
                    "note": note,
                }
            })
            task_id += 1

    return tasks


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="推理结果 → Label Studio 复查任务")
    parser.add_argument("--infer_dir",  required=True,
                        help="包含 *_infer.json 或 clips_*/ 的目录")
    parser.add_argument("--output",     required=True,
                        help="输出 Label Studio JSON 路径")
    parser.add_argument("--csv_url_prefix", default="http://192.168.2.140:8182",
                        help="CSV 文件 URL 前缀")
    parser.add_argument("--video_url_prefix", default="",
                        help="MP4 文件 URL 前缀（默认 csv_url_prefix/transcoded）")
    parser.add_argument("--use_clips",  action="store_true",
                        help="扫描 clips_*/ 目录（推荐），而非 _infer.json")
    parser.add_argument("--cam_mode", default="auto", type=_cam_mode_type,
                        help="每个任务固定生成几个videoN字段（--use_clips和--ml_full_video"
                             "两种模式都吃这个参数）。auto（默认）=按每个session/clip组"
                             "实际探测到的机位数量走，不同组之间字段数量可能不一样；传"
                             "2/3/4/...任意正整数可以强制固定字段数量（要跟Label Studio"
                             "项目里配置的Video组件个数一致，否则导入会报错缺字段）——"
                             "比如项目按3机位配置好了，但某天原始视频只拍了2个视角，传"
                             "--cam_mode 3能保证那天的任务也带上video3字段（值是空字符串"
                             "占位，不是拿别的机位画面顶替），不会导致那天的任务导入失败。"
                             "机位数以后增加时（比如4机位）直接传--cam_mode 4就行，不需要"
                             "改代码")
    parser.add_argument("--mode", default="scratch_only",
                        choices=["scratch_only", "uncertain", "all"])
    parser.add_argument("--low_threshold",  type=float, default=0.3)
    parser.add_argument("--high_threshold", type=float, default=0.65)
    parser.add_argument("--label", default="抓挠")
    parser.add_argument("--split_by_imu", action="store_true",
                        help="除了生成混合所有IMU的主JSON，额外按触发检测的IMU"
                             "（meta.imu_label，比如'IMU3'）各自拆分出一份"
                             "{output去掉.json}_{IMU}.json，方便只导入某一条狗"
                             "的复查任务，跟混合版互不影响，两种用法都能用。"
                             "仅--use_clips模式生效（_infer.json模式的task本身"
                             "就是按单个IMU的CSV生成的，不需要再拆）")
    parser.add_argument("--min_conf", default=None,
                        help="除了生成完整的一批JSON，额外按置信度下限再筛一份"
                             "——比如传0.7，会另外生成{output去掉.json}_0.7.json"
                             "（以及--split_by_imu时对应的{IMU}_0.7.json），只保留"
                             "meta.bin置信度桶下界>=0.7的任务（即extract_clips.py"
                             "分到clips_0.7-0.8/、clips_0.8-0.9/等桶的clip，桶下界"
                             "低于0.7的排除）。完整版JSON照常生成，这个是在它基础上"
                             "额外多出的筛选版，两者并存、不是互相替代。可以传多个"
                             "阈值，逗号分隔，比如'0.7,0.9'会各自生成一份。仅"
                             "--use_clips模式生效（bin这个字段只有clips模式的task"
                             "才有）")
    parser.add_argument("--ml_full_video", action="store_true",
                        help="独立的第三种模式(跟--use_clips/默认全录制模式不冲突)："
                             "在原始完整录制视频上直接标注模型检测到的高置信度抓挠"
                             "片段，不裁剪clip，标注人标成'ML'(completed_by)，用来"
                             "区分这批是模型自动预标注的、不是人工标的。每个session"
                             "(不管有没有检测到达标片段)都会生成一个task，让复查的人"
                             "能看到全部录制数据，不只是命中的部分——方便顺便核查"
                             "模型有没有漏检。--infer_dir要指向包含*_infer.json的"
                             "目录(通常是out_dir/_infer)")
    parser.add_argument("--ml_min_conf", type=float, default=0.8,
                        help="--ml_full_video模式下，只标注置信度>=这个值的抓挠片段"
                             "（默认0.8，跟--min_conf是两回事：--min_conf是给clips"
                             "模式按bin下界筛选任务文件用的，这个是给--ml_full_video"
                             "模式筛选具体标注哪些片段用的）")
    parser.add_argument("--ml_conf_field", default="conf_max", choices=["conf_max", "conf_mean"],
                        help="--ml_full_video模式下用哪个置信度字段判断，默认conf_max")
    parser.add_argument("--ml_multi_labels", default="",
                        help="传逗号分隔的多个类别（比如'活动,睡觉,抓挠,未佩戴'）时，"
                             "--ml_full_video会切换成合并模式：同一条IMU、同一条时间轴"
                             "上同时标出这几个类别各自检测到的片段，而不是每个类别各自"
                             "生成一份独立文件。这时--infer_dir要指向RESULT_ROOT/{day}"
                             "这一级目录（下面按类别各有一个子目录，run_review_bins_"
                             "all_days.sh的标准结构），不是某个类别自己的_infer目录。"
                             "--label参数在这个模式下不生效")
    args = parser.parse_args()

    video_prefix = args.video_url_prefix or f"{args.csv_url_prefix.rstrip('/')}/transcoded"

    if args.ml_full_video:
        multi_labels = [l.strip() for l in args.ml_multi_labels.split(",") if l.strip()]

        if multi_labels:
            print(f"模式: 全录制文件+ML自动预标注(多类别合并: {multi_labels})，"
                  f"置信度阈值({args.ml_conf_field})>={args.ml_min_conf}")
            ml_tasks = build_tasks_from_infer_ml_multi(
                args.infer_dir, multi_labels, args.csv_url_prefix, video_prefix,
                min_conf=args.ml_min_conf, conf_field=args.ml_conf_field, cam_mode=args.cam_mode)
            suffix = "_full_ml_multi_"
        else:
            infer_jsons = glob.glob(os.path.join(args.infer_dir, "**", "*_infer.json"), recursive=True)
            infer_jsons += glob.glob(os.path.join(args.infer_dir, "*_infer.json"))
            infer_jsons = sorted(set(infer_jsons))
            if not infer_jsons:
                print(f"[错误] {args.infer_dir} 下没有找到 *_infer.json")
                return
            print(f"模式: 全录制文件+ML自动预标注，找到 {len(infer_jsons)} 个推理结果，"
                  f"置信度阈值({args.ml_conf_field})>={args.ml_min_conf}")
            ml_tasks = build_tasks_from_infer_ml(
                infer_jsons, args.csv_url_prefix, video_prefix, args.label,
                min_conf=args.ml_min_conf, conf_field=args.ml_conf_field, cam_mode=args.cam_mode)
            suffix = "_full_ml_"

        if not ml_tasks:
            print(f"[错误] 没有生成任何task，检查--infer_dir路径是不是对的"
                  f"{'（多类别合并模式--infer_dir要指向day这一级目录）' if multi_labels else ''}")
            return

        base, ext = os.path.splitext(args.output)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        # 按imu_label拆成多个文件——每个IMU自己的CSV/检测结果各自一份，
        # 不要混在一个文件里互相顶掉
        by_imu = {}
        for t in ml_tasks:
            by_imu.setdefault(t["meta"]["imu_label"], []).append(t)

        for imu_label in sorted(by_imu):
            sub_tasks = by_imu[imu_label]
            out_path = f"{base}{suffix}{imu_label}{ext}"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(sub_tasks, f, ensure_ascii=False, indent=2)
            n_hit = sum(1 for t in sub_tasks if t["annotations"])
            print(f"  → {out_path}  （{len(sub_tasks)} 个任务，其中{n_hit}个有达标片段、"
                  f"{len(sub_tasks) - n_hit}个模型未检出，含全部录制数据供核查漏检）")
        return

    if args.use_clips:
        print("模式: clips 裁剪片段")
        tasks = build_tasks_from_clips(
            args.infer_dir, args.csv_url_prefix, video_prefix, args.label, args.cam_mode)
    else:
        infer_jsons = glob.glob(os.path.join(args.infer_dir, "**", "*_infer.json"), recursive=True)
        infer_jsons += glob.glob(os.path.join(args.infer_dir, "*_infer.json"))
        infer_jsons = sorted(set(infer_jsons))
        if not infer_jsons:
            print(f"[错误] {args.infer_dir} 下没有找到 *_infer.json")
            return
        print(f"模式: 全录制文件，找到 {len(infer_jsons)} 个推理结果")
        tasks = build_tasks_from_infer(
            infer_jsons, args.csv_url_prefix, video_prefix,
            args.mode, args.low_threshold, args.high_threshold, args.label)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"共生成 {len(tasks)} 个 Label Studio 任务")
    print(f"已保存: {args.output}")

    def _write(path, sub_tasks, note):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sub_tasks, f, ensure_ascii=False, indent=2)
        print(f"  → {path}  ({len(sub_tasks)} 个任务{note})")

    base, ext = os.path.splitext(args.output)

    by_imu = {}
    if args.split_by_imu and args.use_clips:
        # 按meta.imu_label分组，另外各写一份——跟主JSON(混合全部IMU)是两套
        # 独立文件，同一批tasks各自完整复制一份，不是从主JSON里挪走，导入
        # 主JSON看全部、导入某个IMU的JSON只看那一条狗，互不冲突。
        for t in tasks:
            label = t.get("meta", {}).get("imu_label") or "unknown"
            by_imu.setdefault(label, []).append(t)
        for label, sub_tasks in sorted(by_imu.items()):
            _write(f"{base}_{label}{ext}", sub_tasks, f"，仅{label}")

    if args.min_conf and args.use_clips:
        # 按置信度下限再筛一份——完整版JSON（以及上面按IMU拆分的版本）照常
        # 生成，这个是在它们基础上额外多出的筛选版，两者并存，不是互相替代。
        thresholds = [t.strip() for t in args.min_conf.split(",") if t.strip()]
        print(f"\n按置信度筛选（阈值: {', '.join(thresholds)}）...")
        for th_str in thresholds:
            th = float(th_str)
            filtered = [t for t in tasks if parse_bin_lo(t.get("meta", {}).get("bin", "")) >= th]
            _write(f"{base}_{th_str}{ext}", filtered, f"，置信度>={th_str}")
            if args.split_by_imu:
                for label, sub_tasks in sorted(by_imu.items()):
                    sub_filtered = [t for t in sub_tasks
                                   if parse_bin_lo(t.get("meta", {}).get("bin", "")) >= th]
                    _write(f"{base}_{label}_{th_str}{ext}", sub_filtered,
                          f"，仅{label}且置信度>={th_str}")

    print(f"\n导入方式: Label Studio → Import → 选择上面的 JSON 文件")


if __name__ == "__main__":
    main()
