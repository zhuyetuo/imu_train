"""
TF版IMU设备（无蓝牙，离线导出TXT日志）→ 训练/标注用CSV。

做两件事：
  1. convert：TXT日志 → CSV（跟 hicc_offline_to_labelstudio.py 逻辑一致，
     处理设备记录异常导致的时间戳小幅倒退、真正跨午夜、文件名猜日期等）
  2. slice：从已转换的CSV里，按起止时间截取一段，另存为新CSV
     （比如手动核实某一段可疑数据，或者补充标注某个时间窗）

设备导出格式（逗号分隔文本，示例见 26080712_tf2.TXT）:
    HH:MM:SS.MS,AX,AY,AZ,GX,GY,GZ
    12:36:16.000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
    ...
只有"时:分:秒.毫秒"，没有年月日。日期按以下优先级确定：
    1. --date 显式指定（YYYY-MM-DD）
    2. 文件名形如 YYMMDDHH 的前8位数字（例如 26080712_tf2.TXT
       -> 2026-08-07，其中第7-8位 12 是小时，需要跟文件内第一行的
       时间小时数一致，用来交叉验证文件名确实是这种编码）
    3. 都识别不到就用今天日期，并打印警告

用法:
  # 单个文件转换
  python src/data/tf_offline_to_custom.py convert data/26080712_tf2.TXT

  # 批量转换目录下所有 .TXT
  python src/data/tf_offline_to_custom.py convert data/raw_tf/ -o data/raw_tf_csv/

  # 显式指定日期（文件名猜不出来，或者猜错了的时候用）
  python src/data/tf_offline_to_custom.py convert data/26080712_tf2.TXT --date 2026-08-07

  # 按起止时间从CSV里截取一段
  python src/data/tf_offline_to_custom.py slice data/26080712_tf2.csv \\
    --start "2026-08-07 12:36:20.000" --end "2026-08-07 12:36:30.000" \\
    -o data/26080712_tf2_clip.csv
"""

import argparse
import csv
import os
import re
import sys
from datetime import date, datetime, timedelta

CSV_HEADER = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

_FNAME_DATE_RE = re.compile(r"(\d{2})(\d{2})(\d{2})(\d{2})")


# ── 第一件事：TXT → CSV ──────────────────────────────────────────────────────

def guess_date_from_filename(path: str, first_row_hour: int = None):
    """尝试从文件名解析日期，形如 YYMMDDHH（前8位数字，第7-8位是小时）。
    如果 first_row_hour 提供且与文件名的小时不一致，视为识别失败。"""
    name = os.path.basename(path)
    m = _FNAME_DATE_RE.search(name)
    if not m:
        return None
    yy, mm, dd, hh = m.groups()
    try:
        year = 2000 + int(yy)
        d = date(year, int(mm), int(dd))
    except ValueError:
        return None
    if first_row_hour is not None and int(hh) != first_row_hour:
        return None
    return d


def parse_tf_offline_txt(path: str):
    """读取TF设备离线TXT（HH:MM:SS.MS,AX,AY,AZ,GX,GY,GZ），返回逐行dict列表。"""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for line in reader:
            if not line or len(line) < 7:
                continue
            rows.append({
                "time_str": line[0].strip(),
                "acc_x": line[1].strip(), "acc_y": line[2].strip(), "acc_z": line[3].strip(),
                "gyro_x": line[4].strip(), "gyro_y": line[5].strip(), "gyro_z": line[6].strip(),
            })
    return rows


_MIDNIGHT_WRAP_THRESHOLD = timedelta(hours=12)


def build_csv_rows(rows, base_date: date):
    """把 HH:MM:SS.MS + 日期 拼成完整 timestamp。

    时间戳倒退时区分两种情况：
      1. 真正跨午夜（倒退幅度接近一整天，比如 23:59 -> 00:00）：日期 +1。
      2. 设备日志自身的小毛刺（倒退幅度很小）：不改日期，直接丢弃这一行
         （保证 timestamp 严格递增），并统计丢弃数量提示用户。
    """
    out = []
    prev_dt = None
    day_offset = 0
    dropped = 0
    for r in rows:
        h, m, rest = r["time_str"].split(":")
        s, ms = rest.split(".")
        t = datetime(base_date.year, base_date.month, base_date.day,
                      int(h), int(m), int(s), int(ms) * 1000) + timedelta(days=day_offset)

        if prev_dt is not None and t < prev_dt:
            backward = prev_dt - t
            if backward >= _MIDNIGHT_WRAP_THRESHOLD:
                day_offset += 1
                t += timedelta(days=1)
            else:
                dropped += 1
                continue

        prev_dt = t
        ts_str = t.strftime("%Y-%m-%d %H:%M:%S.") + f"{t.microsecond // 1000:03d}"
        out.append([ts_str, r["acc_x"], r["acc_y"], r["acc_z"], r["gyro_x"], r["gyro_y"], r["gyro_z"]])

    if dropped:
        print(f"警告: 发现 {dropped} 行时间戳小幅倒退（设备日志自身的记录异常，不是跨午夜），"
              f"已丢弃这些行以保证 timestamp 严格递增。")

    report_time_gaps(out)
    return out


def report_time_gaps(out_rows, gap_ratio: float = 5.0):
    """检测输出结果里是否存在真实的时间缺口（设备本身没记录到数据，不是脚本丢弃造成的）。"""
    if len(out_rows) < 3:
        return
    ts_list = [datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f") for row in out_rows]
    diffs_ms = [(ts_list[i] - ts_list[i - 1]).total_seconds() * 1000 for i in range(1, len(ts_list))]
    diffs_ms.sort()
    median_ms = diffs_ms[len(diffs_ms) // 2]
    if median_ms <= 0:
        return

    threshold_ms = median_ms * gap_ratio
    gaps = []
    for i in range(1, len(ts_list)):
        gap_ms = (ts_list[i] - ts_list[i - 1]).total_seconds() * 1000
        if gap_ms > threshold_ms:
            gaps.append((out_rows[i - 1][0], out_rows[i][0], gap_ms))

    if gaps:
        print(f"提示: 发现 {len(gaps)} 处真实数据缺口（设备本身没有记录到这段时间的数据）：")
        for before, after, gap_ms in gaps[:20]:
            print(f"  {before}  →  {after}  （缺口约 {gap_ms:.0f} ms）")
        if len(gaps) > 20:
            print(f"  ...（其余 {len(gaps) - 20} 处从略）")


def convert_one(input_path: str, output_path: str = None, date_override: str = None) -> bool:
    """转换单个文件，成功返回 True。"""
    rows = parse_tf_offline_txt(input_path)
    if not rows:
        print(f"[跳过] 未解析到任何数据行: {input_path}")
        return False

    if date_override:
        try:
            base_date = datetime.strptime(date_override, "%Y-%m-%d").date()
        except ValueError:
            print(f"--date 格式错误，应为 YYYY-MM-DD: {date_override}")
            return False
    else:
        first_hour = int(rows[0]["time_str"].split(":")[0])
        base_date = guess_date_from_filename(input_path, first_row_hour=first_hour)
        if base_date is None:
            base_date = datetime.now().date()
            print(f"警告: 无法从文件名识别日期，使用今天日期 {base_date}"
                  f"（同一份文件内的相对时间顺序仍正确，但绝对日期可能不对；"
                  f"可用 --date YYYY-MM-DD 显式指定）。")
        else:
            print(f"从文件名识别日期: {base_date}")

    csv_rows = build_csv_rows(rows, base_date)

    out_path = output_path or (os.path.splitext(input_path)[0] + ".csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(csv_rows)

    print(f"已生成: {out_path}（{len(csv_rows)} 行）")
    return True


def cmd_convert(args):
    if not os.path.exists(args.input):
        print(f"路径不存在: {args.input}")
        sys.exit(1)

    if os.path.isdir(args.input):
        txt_files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(".txt"))
        if not txt_files:
            print(f"目录下没有找到 .TXT 文件: {args.input}")
            sys.exit(1)

        out_dir = args.output
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print(f"批量转换模式: {args.input} 下共 {len(txt_files)} 个 .TXT 文件")
        ok_count = 0
        for fname in txt_files:
            in_path = os.path.join(args.input, fname)
            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".csv") if out_dir else None
            print(f"\n── {fname} ──")
            if convert_one(in_path, output_path=out_path, date_override=args.date):
                ok_count += 1

        print(f"\n批量转换完成: 成功 {ok_count}/{len(txt_files)} 个文件")
        return

    if not convert_one(args.input, output_path=args.output, date_override=args.date):
        sys.exit(1)


# ── 第二件事：按起止时间截取CSV ────────────────────────────────────────────

def slice_csv_by_time(csv_path: str, start_ts: str, end_ts: str, output_path: str) -> int:
    """从已转换的CSV里截取 [start_ts, end_ts] 闭区间内的行，写到output_path。
    start_ts/end_ts 格式: 'YYYY-MM-DD HH:MM:SS' 或 'YYYY-MM-DD HH:MM:SS.mmm'。
    返回截取到的行数（0表示范围内没有任何数据，调用方应该视为失败）。"""
    def parse_flexible(ts_str):
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"无法解析时间: {ts_str}（应为 'YYYY-MM-DD HH:MM:SS' 或 '....mmm'）")

    t_start = parse_flexible(start_ts)
    t_end = parse_flexible(end_ts)
    if t_end < t_start:
        raise ValueError(f"--end ({end_ts}) 比 --start ({start_ts}) 还早")

    kept = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for line in reader:
            if not line:
                continue
            try:
                t = parse_flexible(line[0].strip())
            except ValueError:
                continue
            if t_start <= t <= t_end:
                kept.append(line)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(kept)
    return len(kept)


def cmd_slice(args):
    if not os.path.exists(args.input):
        print(f"路径不存在: {args.input}")
        sys.exit(1)
    out_path = args.output or (
        os.path.splitext(args.input)[0] +
        f"_clip_{args.start.replace(' ', '_').replace(':', '')}-{args.end.replace(' ', '_').replace(':', '')}.csv"
    )
    try:
        n = slice_csv_by_time(args.input, args.start, args.end, out_path)
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)
    if n == 0:
        print(f"[警告] {args.start} ~ {args.end} 范围内没有截到任何数据"
              f"（时间范围是否写错了，或者跟CSV实际覆盖的时间对不上？）")
    else:
        print(f"已生成: {out_path}（{n} 行）")


def main():
    ap = argparse.ArgumentParser(description="TF版IMU设备离线数据处理：TXT转CSV / 按时间截取CSV")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_convert = sub.add_parser("convert", help="TXT日志 → CSV")
    p_convert.add_argument("input", help="TXT文件路径，也可以传目录批量转换")
    p_convert.add_argument("-o", "--output", default=None,
                            help="单文件模式：输出路径（默认同名同目录，扩展名改.csv）。"
                                 "批量模式：输出目录（默认跟输入目录相同）")
    p_convert.add_argument("--date", default=None,
                            help="显式指定日期 YYYY-MM-DD，覆盖文件名自动识别")
    p_convert.set_defaults(func=cmd_convert)

    p_slice = sub.add_parser("slice", help="按起止时间从CSV截取一段")
    p_slice.add_argument("input", help="已转换好的CSV文件路径")
    p_slice.add_argument("--start", required=True, help="起始时间，'YYYY-MM-DD HH:MM:SS[.mmm]'")
    p_slice.add_argument("--end", required=True, help="结束时间，'YYYY-MM-DD HH:MM:SS[.mmm]'")
    p_slice.add_argument("-o", "--output", default=None,
                          help="输出路径（默认在原文件名后加上起止时间后缀）")
    p_slice.set_defaults(func=cmd_slice)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
