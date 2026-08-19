"""
把一天的原始录制目录（比如 data/raw_custom/data/2026_8_18），按"触发检测的
IMU编号"重新组织成 imu1/ imu2/ imu3/... 子目录，每个子目录下放：
  - 这条狗自己的 CSV（比如 imu1/ 下放 ..._cam1_imu1_..._raw.csv）
  - 这个场次全部机位的 MP4（cam1/cam2/cam3...，同一份视频复制到每个
    imu 子目录下，不是软链接——上游有些工具（比如浏览器直接播放、
    某些不认软链接的复查/标注流程）需要的是实体文件）

为什么要复制成多份，不是直接引用原文件：房间固定几个机位拍全景，同一批
mp4是所有狗共用的，但复查/标注工具经常是"按一个目录一个任务"的思路，
把"这条狗的CSV + 全部机位视频"放进同一个目录，能直接把这个目录整个丢给
那类工具，不用工具自己再去猜"这条狗对应哪几个机位视频"。

用法：
    python src/split_day_by_imu.py --day_dir data/raw_custom/data/2026_8_18

    # 输出到别的目录，不在原始day_dir里创建imu*/子目录（比如不想污染
    # 原始数据目录）：
    python src/split_day_by_imu.py \
        --day_dir data/raw_custom/data/2026_8_18 \
        --output_dir data/raw_custom/data_by_imu/2026_8_18

    # 只想看会怎么处理、不真的复制文件：
    python src/split_day_by_imu.py --day_dir data/raw_custom/data/2026_8_18 --dry_run
"""
import argparse
import glob
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_clips import KNOWN_STEM_SUFFIXES, camera_video_stem, session_prefix  # noqa: E402

MAX_CAM_NUM = 6  # 探测机位号的上限，够用就行，不用配置成参数


def find_file(stem: str, day_dir: str, exts: tuple) -> str:
    for ext in exts:
        p = os.path.join(day_dir, stem + ext)
        if os.path.exists(p):
            return p
    return ""


def discover_sessions(day_dir: str) -> dict:
    """扫描day_dir下所有CSV，按(session, suffix)分组，每组记录这个场次里
    实际有CSV的IMU编号列表（== 触发检测的狗的编号，不是机位编号）。
    返回 {(session, suffix): [imu_num, ...]}。"""
    sessions = {}
    for csv_path in sorted(glob.glob(os.path.join(day_dir, "*.csv"))):
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        m = re.search(r"_cam(\d+)_imu(\d+)(_resampled16hz|_raw)$", stem)
        if not m:
            continue
        imu_num = int(m.group(2))
        session, suffix = session_prefix(stem)
        sessions.setdefault((session, suffix), set()).add(imu_num)
    return {k: sorted(v) for k, v in sessions.items()}


def discover_cam_videos(session: str, suffix: str, day_dir: str) -> list:
    """返回这个场次实际存在的机位视频列表 [(cam_num, mp4_path), ...]。"""
    found = []
    for cam_num in range(1, MAX_CAM_NUM + 1):
        stem = camera_video_stem(session, cam_num, suffix)
        p = find_file(stem, day_dir, (".mp4", ".MP4", ".avi", ".mov"))
        if p:
            found.append((cam_num, p))
    return found


def main():
    ap = argparse.ArgumentParser(description="按触发IMU把一天的原始数据拆分成imu1/imu2/.../子目录")
    ap.add_argument("--day_dir", required=True, help="一天的原始录制目录，比如 data/raw_custom/data/2026_8_18")
    ap.add_argument("--output_dir", default=None,
                    help="imu*/子目录建在哪（默认跟--day_dir同一个目录，直接在里面建imu1/imu2/...）")
    ap.add_argument("--dry_run", action="store_true", help="只打印会怎么处理，不真的复制文件")
    args = ap.parse_args()

    day_dir = args.day_dir
    output_dir = args.output_dir or day_dir
    if not os.path.isdir(day_dir):
        print(f"[错误] {day_dir} 不是一个目录")
        sys.exit(1)

    sessions = discover_sessions(day_dir)
    if not sessions:
        print(f"[警告] {day_dir} 下没有找到任何 *_camN_imuN(_raw|_resampled16hz).csv")
        return

    print(f"共探测到 {len(sessions)} 个录制场次，逐个处理...")

    n_copied = 0
    n_skipped = 0
    imu_dirs_used = set()

    for (session, suffix), imu_nums in sorted(sessions.items()):
        cam_videos = discover_cam_videos(session, suffix, day_dir)
        if not cam_videos:
            print(f"  [跳过] {session}{suffix}：没有找到任何机位视频")
            continue

        for imu_num in imu_nums:
            csv_stem = camera_video_stem(session, imu_num, suffix)  # csv命名跟机位视频同一套规则(camN_imuN)
            csv_path = find_file(csv_stem, day_dir, (".csv",))
            if not csv_path:
                print(f"  [跳过] {session}{suffix} IMU{imu_num}：没找到对应CSV({csv_stem}.csv)")
                continue

            imu_dir = os.path.join(output_dir, f"imu{imu_num}")
            imu_dirs_used.add(imu_dir)
            if not args.dry_run:
                os.makedirs(imu_dir, exist_ok=True)

            targets = [csv_path] + [p for _, p in cam_videos]
            for src in targets:
                dst = os.path.join(imu_dir, os.path.basename(src))
                if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                    n_skipped += 1
                    continue
                if args.dry_run:
                    print(f"    [dry_run] {src} → {dst}")
                else:
                    shutil.copy2(src, dst)
                n_copied += 1

    print(f"\n完成：{'（dry_run，未真正复制）' if args.dry_run else ''}")
    print(f"  涉及 {len(imu_dirs_used)} 个imu目录: {sorted(os.path.basename(d) for d in imu_dirs_used)}")
    print(f"  复制 {n_copied} 个文件，跳过 {n_skipped} 个（已存在且大小一致，视为已复制过）")


if __name__ == "__main__":
    main()
