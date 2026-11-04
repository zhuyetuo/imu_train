"""
将 Label Studio 导出的 IMU 标注 JSON 转换为训练用 CSV。

Label Studio 导出格式（时间序列标注）:
  每个 task 对应一个 CSV 文件（data.csv 字段），
  annotations 里包含多个时间段 + 标签（timeserieslabels）。

输出格式（与 loader_custom.py 兼容）:
  record_id, label, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

  注：record_id 不是真实的狗ID，是"task编号+传感器路数"拼出来的分组键
  （比如 task496_imu1），一次录制/一路传感器算一个record_id，跟实际有
  多少条不同的狗没有直接对应关系，只是用来做数据集划分时的分组单位。

用法:
  # CSV 从 URL 下载（需要设备在线）
  python src/data/labelstudio_to_custom.py \\
    --json data/raw_custom/labelstudio_export.json \\
    --output data/raw_custom/data.csv

  # CSV 在本地目录（文件名与 URL 里的文件名一致）
  python src/data/labelstudio_to_custom.py \\
    --json data/raw_custom/labelstudio_export.json \\
    --csv_dir data/raw_wit/ \\
    --output data/raw_custom/data.csv

  # WitMotion acc 单位是 g，转换成 m/s²
  python src/data/labelstudio_to_custom.py \\
    --json data/raw_custom/labelstudio_export.json \\
    --output data/raw_custom/data.csv \\
    --acc_unit g
"""

import argparse
import json
import os
import sys
import urllib.request
from urllib.parse import urlparse

import numpy as np
import pandas as pd

G = 9.81

# ── 自动检测列名 ──────────────────────────────────────────────────────────────
ACC_CANDIDATES  = [
    ["acc_x", "acc_y", "acc_z"],
    ["AccX",  "AccY",  "AccZ"],
    ["AX",    "AY",    "AZ"],
    ["ax",    "ay",    "az"],
    ["Ax",    "Ay",    "Az"],
]
GYRO_CANDIDATES = [
    ["gyro_x", "gyro_y", "gyro_z"],
    ["GyroX",  "GyroY",  "GyroZ"],
    ["GX",     "GY",     "GZ"],
    ["gx",     "gy",     "gz"],
    ["Gx",     "Gy",     "Gz"],
    ["wx",     "wy",     "wz"],
]
TS_KEYWORDS = ["time", "timestamp", "datetime", "date", "chip_time"]


def _find_cols(df_cols, candidates):
    for grp in candidates:
        if all(c in df_cols for c in grp):
            return grp
    return None


def _find_ts_col(df_cols):
    low = [c.lower() for c in df_cols]
    for kw in TS_KEYWORDS:
        for i, cl in enumerate(low):
            if kw in cl:
                return df_cols[i]
    return None


# ── CSV 加载 ──────────────────────────────────────────────────────────────────
def load_csv(path_or_url: str, csv_dir: str) -> pd.DataFrame:
    """从本地目录或 URL 加载 CSV。"""
    fname = os.path.basename(urlparse(path_or_url).path)

    if csv_dir:
        local = os.path.join(csv_dir, fname)
        if os.path.exists(local):
            return pd.read_csv(local)

    tmp = f"/tmp/_ls_imu_{fname}"
    urllib.request.urlretrieve(path_or_url, tmp)
    return pd.read_csv(tmp)


# ── 时间戳解析 ────────────────────────────────────────────────────────────────
def parse_ts(df: pd.DataFrame, ts_col: str) -> pd.Series:
    """把时间戳列解析为 datetime，容忍多种格式。"""
    return pd.to_datetime(df[ts_col], errors="coerce")


# ── 主转换逻辑 ────────────────────────────────────────────────────────────────
def _load_sensor_df(url_or_none: str, csv_dir: str, sensor_name: str, missing_strategy: str = "drop"):
    """加载单个传感器 CSV，返回 (df, acc_cols, gyro_cols, ts_col) 或 None。

    missing_strategy：acc/gyro里蓝牙断联留下的缺失值(NaN)怎么处理：
      "drop"（默认）——丢掉缺失的那几行，不编造数据。真实数据不该有缺失，
        训练数据（不像实时推理要求时间连续不能有空隙）能承受丢几行造成
        的小缺口。
      "ffill"——用前后值填充（旧版本一度用过这个，跟infer_csv_scratch.py
        实时推理那边保持一致），会"编造"出这几个点本来没采集到的数值。
      "none"——完全不处理，保留原始NaN（这是最早、改动之前一直用的方式，
        NaN会一路带进CSV/预处理/最终npz。RF/XGBoost/LightGBM大多原生
        兼容NaN不会报错，但神经网络完全不能容忍，训练会直接loss变nan）。
    做成可选项是因为不确定NaN会不会影响树模型的效果——之前一直是"none"
    这个状态训出来的树模型，万一效果反而更好，能重新跑一版对比，而不是
    直接替换掉、没法回头验证。"""
    if not url_or_none:
        return None
    try:
        df = load_csv(url_or_none, csv_dir)
    except Exception as e:
        print(f"  [错误] {sensor_name} CSV 无法加载: {e}")
        return None
    df.columns = [c.strip() for c in df.columns]
    acc_cols  = _find_cols(df.columns.tolist(), ACC_CANDIDATES)
    gyro_cols = _find_cols(df.columns.tolist(), GYRO_CANDIDATES)
    ts_col    = _find_ts_col(df.columns.tolist())
    if acc_cols is None or ts_col is None:
        print(f"  [错误] {sensor_name}: 找不到加速度列或时间戳列，现有: {list(df.columns)}")
        return None
    df["_ts"] = parse_ts(df, ts_col)
    if df["_ts"].isna().all():
        print(f"  [错误] {sensor_name}: 时间戳解析失败")
        return None
    # 蓝牙断联等原因，原始CSV里acc/gyro偶尔会有缺失行（NaN），处理方式见
    # missing_strategy——丢弃的是整行(该行所有acc/gyro列)，不是只丢NaN的
    # 那一列，避免同一行里部分列是真实值、部分列因为其它列缺失被迫也丢掉
    # 这种不必要的浪费：ACC_CANDIDATES/GYRO_CANDIDATES匹配到的每组列本来
    # 就是同一颗传感器的同一次采样，缺一个通道这一行本来就不完整。
    all_cols = acc_cols + (gyro_cols or [])
    n_nan_before = int(df[all_cols].isna().any(axis=1).sum())
    if n_nan_before and missing_strategy != "none":
        action = "丢弃这些行（不做前后值填充，避免编造数据）" if missing_strategy == "drop" \
                 else "前后值填充"
        print(f"  [警告] {sensor_name}: acc/gyro有{n_nan_before}行缺失（蓝牙断联？），{action}")
        if missing_strategy == "drop":
            df = df.dropna(subset=all_cols)
        elif missing_strategy == "ffill":
            df[acc_cols] = df[acc_cols].ffill().bfill()
            if gyro_cols:
                df[gyro_cols] = df[gyro_cols].ffill().bfill()
    elif n_nan_before:
        print(f"  [警告] {sensor_name}: acc/gyro有{n_nan_before}行缺失（蓝牙断联？），"
              f"--missing_strategy=none，原样保留NaN不处理")
    return df, acc_cols, gyro_cols


_OUT_COLS = ["record_id", "label", "timestamp", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def _extract_frame(df, acc_cols, gyro_cols, label, t_start_str, t_end_str,
                   subject_id, acc_unit, keep_labels):
    """从 df 中切出时间段，返回 DataFrame（列见 _OUT_COLS），没有就 None。

    **整段向量化，不逐行建 dict。** 以前每一行一个 dict：670 小时 × 50Hz ≈ 1.2 亿行，
    每个 dict 几百字节，几十 GB，跑到四分之三被系统 OOM 杀掉（2026-10-29 训练 #15）。"""
    if keep_labels and label not in keep_labels:
        return None
    t_start = pd.to_datetime(t_start_str)
    t_end   = pd.to_datetime(t_end_str)
    mask    = (df["_ts"] >= t_start) & (df["_ts"] <= t_end)
    seg_df  = df[mask]
    if len(seg_df) == 0:
        print(f"  [警告] {label} {t_start_str}→{t_end_str} 无匹配行")
        return None
    acc = seg_df[acc_cols].values.astype(np.float64)
    if acc_unit == "g":
        acc = acc * G
    gyro = seg_df[gyro_cols].values.astype(np.float64) if gyro_cols else np.zeros((len(seg_df), 3))
    return pd.DataFrame({
        "record_id": subject_id,
        "label": label,
        "timestamp": seg_df["_ts"].values,
        "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
        "gyr_x": gyro[:, 0], "gyr_y": gyro[:, 1], "gyr_z": gyro[:, 2],
    }, columns=_OUT_COLS)


def _extract_short_event(df, acc_cols, gyro_cols, label, t_start_str, t_end_str, subject_id, k,
                         acc_unit, others, window_s, stats):
    """短事件（比如甩动，一两秒）补成训练窗口：以事件为中心、带上前后**真实的**上下文切一段，
    整段标成这个类别，单独成一个 record（{subject}_short{k}）。

    为什么要补：训练按 window_s（默认 2 秒）切窗。比窗口短的片段——
      · 按片段切（label_concat）时整段被丢掉；
      · 按记录滑窗时跟前后别的片段拼在一起，多数投票投成别的类，
    结果甩动几乎进不了训练集，模型没见过「甩动」长什么样，一甩就判成抓挠。

    长度取 S = 2W - L（L = 事件长）：这样在这一段里怎么滑 W 长的窗口，**每个窗口都完整
    包含这次事件**——窗口里有一次完整的甩动，就叫它甩动（推理时看到的也是这样的窗口）。
    上下文碰到前后别的类别的标注就截住（不把抓挠的数据标成甩动）；截短了不够 S，
    用剩下那段真实数据镜像补齐，不编造新的信号。"""
    a, b = pd.to_datetime(t_start_str), pd.to_datetime(t_end_str)
    L = (b - a).total_seconds()
    W = float(window_s)
    if L >= W or L <= 0:
        return None
    S = 2 * W - L
    c = a + (b - a) / 2
    lo, hi = c - pd.Timedelta(seconds=S / 2), c + pd.Timedelta(seconds=S / 2)
    clipped = False
    for o_lab, o0, o1 in others:
        if o_lab == label:
            continue
        if o1 <= a and o1 > lo:
            lo, clipped = o1, True
        if o0 >= b and o0 < hi:
            hi, clipped = o0, True
    seg_df = df[(df["_ts"] >= lo) & (df["_ts"] <= hi)]
    if len(seg_df) < 2:
        return None
    acc = seg_df[acc_cols].values.astype(np.float64)
    if acc_unit == "g":
        acc = acc * G
    gyro = seg_df[gyro_cols].values.astype(np.float64) if gyro_cols else np.zeros((len(seg_df), 3))
    ts = seg_df["_ts"].values
    dt = np.median(np.diff(ts).astype("timedelta64[ns]").astype(np.int64)) if len(ts) > 1 else 0
    if dt <= 0:
        return None
    need = int(np.ceil(S * 1e9 / dt)) + 1
    padded = False
    if len(seg_df) < need:
        # 不够长：两头镜像补（用的是这段里真实的数据），不插值、不造信号
        lack = need - len(seg_df)
        left = lack // 2
        right = lack - left
        vals = np.concatenate([acc, gyro], axis=1)
        vals = np.pad(vals, ((left, right), (0, 0)), mode="symmetric")
        acc, gyro = vals[:, :3], vals[:, 3:]
        t0 = ts[0] - np.timedelta64(int(dt) * left, "ns")
        ts = t0 + np.arange(len(vals)) * np.timedelta64(int(dt), "ns")
        padded = True
    stats["n"] = stats.get("n", 0) + 1
    stats["clipped"] = stats.get("clipped", 0) + int(clipped)
    stats["padded"] = stats.get("padded", 0) + int(padded)
    stats.setdefault("by_label", {}).setdefault(label, 0)
    stats["by_label"][label] += 1
    return pd.DataFrame({
        "record_id": f"{subject_id}_short{k}",
        "label": label,
        "timestamp": ts,
        "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
        "gyr_x": gyro[:, 0], "gyr_y": gyro[:, 1], "gyr_z": gyro[:, 2],
    }, columns=_OUT_COLS)


def _extract_rows(df, acc_cols, gyro_cols, label, t_start_str, t_end_str,
                  subject_id, acc_unit, keep_labels):
    """同 _extract_frame，返回 row dict 列表（给 analyze_missing 这种小量分析用）。"""
    fr = _extract_frame(df, acc_cols, gyro_cols, label, t_start_str, t_end_str,
                        subject_id, acc_unit, keep_labels)
    return [] if fr is None else fr.to_dict("records")


def convert(tasks: list, csv_dir: str, acc_unit: str,
            keep_labels: list = None, missing_strategy: str = "drop",
            output=None, short_labels: list | None = None, window_s: float = 2.0):
    """转换全部任务。

    给了 output 就**按任务边转边追加写**，返回 (总行数, 各类别行数)：内存里同时只有
    一个任务的数据，几百小时的数据集也不会撑爆。不给 output 就照旧返回整张 DataFrame
    （小数据 / 测试用）。

    排序：同一个 record_id 只来自一个任务，所以在任务内部按 (record_id, timestamp)
    排好就够了——下游按 record_id 分组，记录之间的先后无所谓。"""
    keep_set = set(keep_labels) if keep_labels else None
    # 短事件补齐（见 _extract_short_event）；统计打在最后
    short_stats: dict = {"_labels": set(short_labels or ())}
    short_k = [0]
    rows: list = []          # 当前这个任务的段（DataFrame）
    frames: list = []        # 不写文件时攒全部
    n_total = 0
    counts: dict = {}
    wrote_header = False
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        tmp_out = output + ".part"
        if os.path.exists(tmp_out):
            os.remove(tmp_out)

    def _flush():
        nonlocal wrote_header, n_total
        if not rows:
            return
        chunk = pd.concat(rows, ignore_index=True)
        rows.clear()
        chunk = chunk.sort_values(["record_id", "timestamp"], kind="stable")
        n_total += len(chunk)
        for lab, c in chunk["label"].value_counts().items():
            counts[lab] = counts.get(lab, 0) + int(c)
        if output:
            chunk.to_csv(tmp_out, mode="a", header=not wrote_header, index=False)
            wrote_header = True
        else:
            frames.append(chunk)

    def _add(fr):
        if fr is not None and len(fr):
            rows.append(fr)

    def _emit(df, acc_cols, gyro_cols, lab, t0, t1, subject_id, same_sensor_segs, orig=()):
        """一段标注 → 训练行。短事件类别且比窗口短：换成带上下文的独立 record；否则照旧。
        认短事件看现在的类别名，也看归并前的原名（甩头/抖身 并进「活动」当负样本时也要补）"""
        if keep_set and lab not in keep_set:
            return
        if lab in short_stats["_labels"] or any(o in short_stats["_labels"] for o in (orig or ())):
            others = [(l2, pd.to_datetime(a2), pd.to_datetime(b2)) for l2, a2, b2 in same_sensor_segs
                      if not (l2 == lab and a2 == t0 and b2 == t1)]
            fr = _extract_short_event(df, acc_cols, gyro_cols, lab, t0, t1, subject_id, short_k[0],
                                      acc_unit, others, window_s, short_stats)
            if fr is not None:
                short_k[0] += 1
                _add(fr)
                return
        _add(_extract_frame(df, acc_cols, gyro_cols, lab, t0, t1, subject_id, acc_unit, keep_set))

    from tqdm import tqdm
    for task in tqdm(tasks, desc="转换标注", unit="task"):
        _flush()
        task_id = task["id"]
        data    = task.get("data", {})

        annotations = task.get("annotations", [])
        if not annotations:
                continue

        # ── 判断格式：单 csv 还是双 csv1/csv2 ───────────────────────
        is_multi = "csv1" in data or "csv2" in data

        if is_multi:
            # 双传感器：label1→csv1, label2→csv2
            sensor_map = {}
            for idx in ("1", "2"):
                url = data.get(f"csv{idx}", "")
                if url:
                    res = _load_sensor_df(url, csv_dir, f"imu{idx}", missing_strategy)
                    if res:
                        sensor_map[f"label{idx}"] = (res[0], res[1], res[2],
                                                      f"task{task_id}_imu{idx}")
            if not sensor_map:
                continue
            # 任务只挂了一个传感器（csv1/csv2只填了一个）时，Label Studio 标注界面
            # 用的是通用的 from_name="label"，不是 "label1"/"label2"——实测显示这种
            # from_name="label" + is_multi 的标注段100%来自只有一个csv链接的任务，
            # 给这唯一存在的传感器加一个 "label" 别名，不然这些标注会被判定成"对不上
            # 传感器"整段跳过（531段抓挠标注里有342段、即64%，就是这么丢的）
            if len(sensor_map) == 1:
                sensor_map["label"] = next(iter(sensor_map.values()))

            segs = []
            for ann in annotations:
                for seg in ann.get("result", []):
                    val    = seg.get("value", {})
                    labels = val.get("timeserieslabels", [])
                    t0     = val.get("start", "")
                    t1     = val.get("end",   "")
                    fn     = seg.get("from_name", "")
                    if not labels or not t0 or not t1 or fn not in sensor_map:
                        continue
                    segs.append((sensor_map[fn], labels[0], t0, t1, val.get("orig_labels") or labels))
            for (df, acc_cols, gyro_cols, subject_id), lab, t0, t1, orig in segs:
                _emit(df, acc_cols, gyro_cols, lab, t0, t1, subject_id,
                      [(l2, t2a, t2b) for (s2, l2, t2a, t2b, _o) in segs if s2[3] == subject_id], orig)
        else:
            # 旧格式：单 csv
            csv_url = data.get("csv", "")
            if not csv_url:
                    continue
            res = _load_sensor_df(csv_url, csv_dir, "imu", missing_strategy)
            if not res:
                continue
            df, acc_cols, gyro_cols = res
            subject_id = f"task{task_id}"
            segs = []
            for ann in annotations:
                for seg in ann.get("result", []):
                    val    = seg.get("value", {})
                    labels = val.get("timeserieslabels", [])
                    t0     = val.get("start", "")
                    t1     = val.get("end",   "")
                    if not labels or not t0 or not t1:
                        continue
                    segs.append((labels[0], t0, t1, val.get("orig_labels") or labels))
            for lab, t0, t1, orig in segs:
                _emit(df, acc_cols, gyro_cols, lab, t0, t1, subject_id, [x[:3] for x in segs], orig)

    _flush()
    if short_stats.get("n"):
        print(f"\n[短事件补齐] {short_stats['n']} 段比 {window_s:g} 秒窗口短的片段补成了带上下文的独立片段 "
              f"{short_stats['by_label']}；其中 {short_stats.get('clipped', 0)} 段前后碰到别的类别被截住、"
              f"{short_stats.get('padded', 0)} 段不够长用真实数据镜像补齐")
    elif short_stats["_labels"]:
        print(f"\n[短事件补齐] {sorted(short_stats['_labels'])} 里没有比 {window_s:g} 秒短的片段")
    if not n_total:
        raise RuntimeError("没有任何有效数据，请检查输入 JSON 和 CSV 路径")
    if output:
        # 写完才改名：中途被杀的话不留半份 CSV（train_custom.sh 见到文件在就会跳过这一步）
        os.replace(tmp_out, output)
        return n_total, counts
    # 标注段是按 Label Studio 导出顺序拼接的，不一定按时间先后，
    # 必须按 (record_id, timestamp) 重新排序，否则同一个 record_id 内部的
    # "局部秒数"跟真实时间顺序对不上（窗口切分、事件时间戳换算都会错）
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["record_id", "timestamp"], kind="stable").reset_index(drop=True)


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Label Studio IMU 标注 → 训练 CSV")
    parser.add_argument("--json",    required=True, help="Label Studio 导出的 JSON 文件")
    parser.add_argument("--output",  default="",   help="输出 CSV 路径（默认: 与 JSON 同目录同名，扩展名改 .csv）")
    parser.add_argument("--csv_dir", default="",   help="本地 CSV 目录（不填则从 URL 下载）")
    parser.add_argument("--acc_unit", default="ms2", choices=["ms2", "g"],
                        help="加速度单位：ms2=m/s²（默认），g=重力单位（自动×9.81）")
    parser.add_argument("--keep_labels", nargs="*", default=["活动", "睡觉", "抓挠"],
                        help="只保留这些标签（默认: 活动 睡觉 抓挠）")
    parser.add_argument("--missing_strategy", default="drop", choices=["drop", "ffill", "none"],
                        help="acc/gyro缺失值(蓝牙断联)怎么处理：drop(默认)=丢掉缺失的行，"
                             "不编造数据；ffill=前后值填充；none=不处理，保留原始NaN"
                             "（这是最早改动之前一直用的方式）。想对比不同处理方式训出来的"
                             "模型效果，改这个参数配合--tag分别跑")
    parser.add_argument("--short_labels", default="",
                        help="短促动作类别（逗号分隔，比如 甩动,甩头/抖身）：比训练窗口短的片段不再被丢掉 / "
                             "被多数投票吞掉，而是以事件为中心带上前后真实上下文补成一段，每个训练窗口都完整包含这次事件")
    parser.add_argument("--window_s", type=float, default=2.0,
                        help="训练窗口长度（秒），配合 --short_labels：要跟 preprocess 的 --window_s 一致")
    args = parser.parse_args()

    json_stem = os.path.splitext(os.path.basename(args.json))[0]
    json_dir  = os.path.dirname(os.path.abspath(args.json))
    output    = args.output if args.output else os.path.join(json_dir, f"{json_stem}.csv")
    processed_dir = f"data/processed_{json_stem}"

    with open(args.json, encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"共 {len(tasks)} 个 task")
    print(f"只保留标签: {args.keep_labels}")

    short = [x.strip() for x in args.short_labels.split(",") if x.strip()]
    if short:
        print(f"短事件补齐: {short}（窗口 {args.window_s:g} 秒）")
    n_total, counts = convert(tasks, args.csv_dir, args.acc_unit, keep_labels=args.keep_labels,
                              missing_strategy=args.missing_strategy, output=output,
                              short_labels=short, window_s=args.window_s)

    print(f"\n── 汇总 ──")
    print(f"总行数: {n_total}")
    for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {cnt} 行")
    print(f"\n已保存: {output}")
    print(f"\n下一步（预处理）:")
    print(f"  python src/data/preprocess.py --dataset custom --raw_csv_custom {output} --output_dir {processed_dir} --config configs/data.yaml")
    print(f"\n下一步（训练）:")
    print(f"  python src/ml/train.py --hz 16 --model rf --processed_dir {processed_dir}")


if __name__ == "__main__":
    main()
