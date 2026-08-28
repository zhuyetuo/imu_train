"""
人工核对工具：把训练/标注数据里的每一段标注，跟模型预测结果对照，输出
一张表——project是哪个、task是哪个、record_id是什么、人工标的类别和
起止时间、模型预测的类别，一致打勾、不一致打叉。用来排查"是不是有些
数据标注错了"（比如把甩身体标成抓挠、活动标成抓挠）。

跟labelstudio_to_custom.py读同样的Label Studio导出JSON+原始传感器CSV，
按标注段（不是整条record）切出片段，每段独立滑窗、提取特征、跑模型
预测，多数投票得到这一段的预测类别，再跟人工标签（remap后）比较。

用法（对单个日期目录跑，因为每个日期目录的原始采样率可能不一样，见
下面--source_hz）:
  python src/ml/review_predictions.py \\
    --project_glob "data/raw_custom/2026_8_11-2026_8_27_raw/project-*.json" \\
    --csv_dir data/raw_wit/ \\
    --model_dir results/processed_2026_8_11-2026_8_27_raw_merged_majority/16hz_remap_custom_3class \\
    --source_hz 50 \\
    --log tmp/review_2026_8_11-2026_8_27_raw.log

  # 结果直接打印+落到--log文本文件，一行一段标注，方便直接翻log看；
  # 多个日期目录分别跑完，几份log文件可以直接cat到一起看
"""

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import joblib

class _Tee:
    """把print()同时写到终端和日志文件——跑一堆project/task下来终端刷太快，
    没log文件的话出了问题翻不回去看，输出结果时已经找不到当时哪段报了警告。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_THIS_DIR, "..", "data")
sys.path.insert(0, _DATA_DIR)
sys.path.insert(0, _THIS_DIR)

from labelstudio_to_custom import _load_sensor_df, _extract_rows  # noqa: E402
from preprocess import downsample  # noqa: E402
from gravity_align import gravity_align_batch, append_raw_tilt_batch  # noqa: E402
from features import extract_features  # noqa: E402

PROJECT_ID_RE = re.compile(r"project-(\d+)-")
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def _disp_width(s):
    """中文/全角字符在终端里占2格，Python的:<N对齐是按字符数算的（不是显示
    宽度），中英文混排时列会错位——这里按east_asian_width算实际显示宽度。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s, width):
    s = str(s)
    return s + " " * max(0, width - _disp_width(s))


def _fmt_row(cols):
    """cols: [(值, 目标宽度), ...]，用｜分隔，中英文混排也不会错位。"""
    return " | ".join(_pad(v, w) for v, w in cols)


def _ls_url(base, project_id, task_id):
    if not base:
        return ""
    return f"  {base.rstrip('/')}/projects/{project_id}/data?task={task_id}"


def _fmt_time(t):
    """时间戳只保留到0.1秒——微秒级精度对人工核对没意义，反而占地方看着乱。"""
    t = pd.to_datetime(t)
    return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]


def _window_data(data, window_size, stride):
    """按窗口切data（一个标注段内label是单一值，不需要preprocess.sliding_window
    那套多数投票/整数标签逻辑），返回 (N, window_size, n_channels)。"""
    n = len(data)
    windows = [data[start:start + window_size]
               for start in range(0, n - window_size + 1, stride)]
    if not windows:
        return np.empty((0, window_size, data.shape[1]), dtype=np.float32)
    return np.array(windows, dtype=np.float32)


def _load_remap(path):
    import yaml
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    return {k: v for k, v in cfg.items() if not str(k).startswith("#")}


def _iter_task_segments(task, csv_dir, acc_unit):
    """复用labelstudio_to_custom.py里convert()的task解析逻辑，但保留每段的
    独立身份（不摊平成行），yield (subject_id, label, t0, t1, seg_rows_df)。"""
    task_id = task["id"]
    data = task.get("data", {})
    annotations = task.get("annotations", [])
    if not annotations:
        return

    is_multi = "csv1" in data or "csv2" in data
    if is_multi:
        sensor_map = {}
        for idx in ("1", "2"):
            url = data.get(f"csv{idx}", "")
            if url:
                res = _load_sensor_df(url, csv_dir, f"imu{idx}")
                if res:
                    sensor_map[f"label{idx}"] = (res[0], res[1], res[2], f"task{task_id}_imu{idx}")
        if not sensor_map:
            return
        if len(sensor_map) == 1:
            sensor_map["label"] = next(iter(sensor_map.values()))

        for ann in annotations:
            for seg in ann.get("result", []):
                val = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                t0, t1 = val.get("start", ""), val.get("end", "")
                fn = seg.get("from_name", "")
                if not labels or not t0 or not t1 or fn not in sensor_map:
                    continue
                df, acc_cols, gyro_cols, subject_id = sensor_map[fn]
                rows = _extract_rows(df, acc_cols, gyro_cols, labels[0], t0, t1,
                                      subject_id, acc_unit, None)
                if rows:
                    yield task_id, subject_id, labels[0], t0, t1, pd.DataFrame(rows)
    else:
        csv_url = data.get("csv", "")
        if not csv_url:
            return
        res = _load_sensor_df(csv_url, csv_dir, "imu")
        if not res:
            return
        df, acc_cols, gyro_cols = res
        subject_id = f"task{task_id}"
        for ann in annotations:
            for seg in ann.get("result", []):
                val = seg.get("value", {})
                labels = val.get("timeserieslabels", [])
                t0, t1 = val.get("start", ""), val.get("end", "")
                if not labels or not t0 or not t1:
                    continue
                rows = _extract_rows(df, acc_cols, gyro_cols, labels[0], t0, t1,
                                      subject_id, acc_unit, None)
                if rows:
                    yield task_id, subject_id, labels[0], t0, t1, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_glob", required=True,
                     help='匹配project-*.json的glob，例如 "data/raw_custom/2026_7_17-2026_7_29/project-*.json"')
    ap.add_argument("--csv_dir", default="data/raw_wit/")
    ap.add_argument("--model_dir", required=True,
                     help="包含ml_rf.pkl和ml_rf.json的目录（train.py的--results_dir输出）")
    ap.add_argument("--model_name", default="rf")
    ap.add_argument("--source_hz", type=int, required=True,
                     help="这批project json对应原始传感器CSV的真实采样率")
    ap.add_argument("--remap", default="configs/remap_custom_3class.yaml")
    ap.add_argument("--acc_unit", default="ms2", choices=["ms2", "g"])
    ap.add_argument("--log", default="tmp/review_predictions.log",
                     help="结果打印+落盘到这个文件（不是CSV，直接翻log看）")
    ap.add_argument("--ls_url_base", default="",
                     help="Label Studio地址（例如 http://192.168.2.140:8181），填了的话每行"
                          "后面会附一个可以直接点开跳转到对应task的链接，方便改标注。"
                          "不带tab参数（tab是网页上视图的编号，从导出的JSON里拿不到），"
                          "跳转后如果没落到对应视图，手动在Label Studio里点一下就行")
    ap.add_argument("--only_wrong", action="store_true",
                     help="逐段明细只打印预测跟人工标注不一致的（✗），太短/一致的不打印，"
                          "方便直接盯着有问题的段看，不用在全量输出里翻")
    ap.add_argument("--window_detail", action="store_true",
                     help="预测错误的段，额外打印每个窗口自己对每个类别的概率、自己投给谁——"
                          "标注本身没错但模型学不会时，从这里能看出是整段窗口都判错（模型"
                          "系统性地不认识这类样本），还是只有部分窗口判错、内部本来就有分歧"
                          "（可能是窗口切分刚好切在动作边界上）")
    args = ap.parse_args()

    log_path = args.log
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    print(f"[review] 日志: {log_path}")

    model_path = os.path.join(args.model_dir, f"ml_{args.model_name}.pkl")
    meta_path = os.path.join(args.model_dir, f"ml_{args.model_name}.json")
    model = joblib.load(model_path)
    meta = json.load(open(meta_path, encoding="utf-8"))
    classes = meta["classes"]
    target_hz = int(meta["hz"])
    window_size = int(meta["window_size"])
    stride = int(meta["stride"])
    gravity_aligned = meta.get("gravity_aligned", True)
    print(f"[review] 模型: {model_path}  classes={classes}  "
          f"hz={target_hz} window={window_size} stride={stride}")

    remap = _load_remap(args.remap)

    files = sorted(glob.glob(args.project_glob))
    if not files:
        print(f"[错误] {args.project_glob} 没匹配到任何文件")
        sys.exit(1)
    print(f"[review] 匹配到 {len(files)} 个project文件")

    has_proba = hasattr(model, "predict_proba")
    if not has_proba:
        print("[提示] 模型不支持predict_proba，置信度列会显示为'-'")

    COLS = [("project", 7), ("task", 7), ("record_id", 17), ("raw_label", 10),
            ("映射标签", 10), ("窗口数", 6), ("pred_label", 10), ("一致", 4),
            ("最大置信度", 8), ("平均置信度", 8), ("全窗口平均置信度", 8),
            ("seg_start", 22), ("seg_end", 22), ("pred_start", 22), ("pred_end", 22)]
    print("\n" + _fmt_row(COLS))

    stats = Counter()  # (true_label, correct) -> count，跑完打统计用
    wrong_rows = []  # 预测错的段，跑完按置信度从低到高排一份"最可疑"清单
    for fp in files:
        m = PROJECT_ID_RE.search(os.path.basename(fp))
        project_id = m.group(1) if m else "?"
        tasks = json.load(open(fp, encoding="utf-8"))
        for task in tasks:
            for task_id, subject_id, raw_label, t0, t1, seg_df in _iter_task_segments(
                    task, args.csv_dir, args.acc_unit):
                data = seg_df[SENSOR_COLS].to_numpy(dtype=np.float64)
                labels = seg_df["label"].to_numpy()
                if args.source_hz != target_hz:
                    data, labels = downsample(data, labels, args.source_hz, target_hz)

                if len(data) < window_size:
                    if not args.only_wrong:
                        print(_fmt_row([
                            (project_id, 7), (task_id, 7), (subject_id, 17), (raw_label, 10),
                            (remap.get(raw_label, "(未映射)"), 10), (0, 6), ("(片段太短)", 10), ("", 4),
                            ("", 8), ("", 8), ("", 8),
                            (_fmt_time(t0), 22), (_fmt_time(t1), 22), ("", 22), ("", 22),
                        ]))
                    continue

                true_label = remap.get(raw_label)
                if true_label is None:
                    # 这个原始标签没在remap里，模型训练时压根没见过这个类别，
                    # 没法比较预测对不对，跳过（不是bug，是remap故意排除的类别）
                    continue

                X = _window_data(data, window_size, stride)
                if len(X) == 0:
                    continue
                tilt = append_raw_tilt_batch(X)[:, :, 6:8]
                if gravity_aligned:
                    X = gravity_align_batch(X)
                X = np.concatenate([X, tilt], axis=2)
                feats = extract_features(X, target_hz, show_progress=False)
                pred_ids = np.array(model.predict(feats)).flatten().astype(int)
                pred_names = [classes[i] for i in pred_ids]
                pred_label = Counter(pred_names).most_common(1)[0][0]
                agree = sum(1 for p in pred_names if p == pred_label)
                correct = pred_label == true_label

                # 每个窗口i覆盖的是重采样后第[i*stride, i*stride+window_size)个
                # 采样点，按target_hz换算回真实时间（重采样是按固定hz等间隔插值/
                # 抽点的，t0+样本序号/target_hz就是这个窗口对应的真实时刻）——
                # 取"被判成pred_label的那些窗口"里最早的起点、最晚的终点，就是
                # 模型自己认为这段事件的起止时间，拿去跟人工标注的seg_start/
                # seg_end比，能看出模型判断的边界跟人工标的差多少。
                t0_dt = pd.to_datetime(t0)
                pred_match_idx = [i for i, p in enumerate(pred_names) if p == pred_label]
                pred_start_t = t0_dt + pd.Timedelta(seconds=pred_match_idx[0] * stride / target_hz)
                pred_end_t = t0_dt + pd.Timedelta(
                    seconds=(pred_match_idx[-1] * stride + window_size) / target_hz)

                # 置信度：最大/平均置信度只看跟多数票pred_label一致的那些窗口，
                # 跟"这段判成pred_label"这件事直接相关，但会把不同意的窗口整个
                # 忽略掉，看不出段内部有没有分歧（比如3个窗口投抓挠都很自信，
                # 另外2个窗口其实倾向活动，这两个只看"获胜方"的指标完全体现不
                # 出来）。全窗口平均置信度是不管每个窗口自己投给谁，统一看
                # 全部窗口对pred_label这个类别打了多少概率再取平均，不同意的
                # 窗口会把这个数字拉低，能反映出段内部的分歧程度。
                if has_proba:
                    proba = model.predict_proba(feats)
                    proba_of_pred = proba[np.arange(len(pred_ids)), pred_ids]
                    match_confs = proba_of_pred[pred_match_idx]
                    max_conf, mean_conf = f"{match_confs.max():.2f}", f"{match_confs.mean():.2f}"
                    pred_label_col = classes.index(pred_label)
                    all_conf = f"{proba[:, pred_label_col].mean():.2f}"
                else:
                    max_conf, mean_conf, all_conf = "-", "-", "-"

                if not correct:
                    window_details = []
                    if args.window_detail and has_proba:
                        for i, p in enumerate(pred_names):
                            w_start = t0_dt + pd.Timedelta(seconds=i * stride / target_hz)
                            w_end = w_start + pd.Timedelta(seconds=window_size / target_hz)
                            window_details.append({
                                "idx": i, "start": w_start, "end": w_end, "pred": p,
                                "probs": {c: float(proba[i, j]) for j, c in enumerate(classes)},
                            })
                    wrong_rows.append({
                        "project_id": project_id, "task_id": task_id, "subject_id": subject_id,
                        "raw_label": raw_label, "true_label": true_label, "pred_label": pred_label,
                        "n_windows": len(pred_names), "max_conf": float(max_conf) if has_proba else -1,
                        "mean_conf": float(mean_conf) if has_proba else -1,
                        "all_conf": float(all_conf) if has_proba else -1,
                        "t0": t0, "t1": t1, "pred_start_t": pred_start_t, "pred_end_t": pred_end_t,
                        "agree": agree, "window_details": window_details,
                    })

                if not (correct and args.only_wrong):
                    print(_fmt_row([
                        (project_id, 7), (task_id, 7), (subject_id, 17), (raw_label, 10),
                        (true_label, 10), (len(pred_names), 6), (pred_label, 10),
                        ("✓" if correct else "✗", 4), (max_conf, 8), (mean_conf, 8), (all_conf, 8),
                        (_fmt_time(t0), 22), (_fmt_time(t1), 22),
                        (_fmt_time(pred_start_t), 22), (_fmt_time(pred_end_t), 22),
                    ]) + f"  ({agree}/{len(pred_names)})" + _ls_url(args.ls_url_base, project_id, task_id))
                stats[(true_label, correct)] += 1

    n_total = sum(stats.values())
    n_wrong = sum(v for (_, correct), v in stats.items() if not correct)
    print("")
    if n_total:
        print(f"[review] 共 {n_total} 段有效对比，其中 {n_wrong} 段预测跟人工标注不一致"
              f"（{n_wrong/n_total*100:.1f}%）")
        print("\n[review] 按人工标签统计不一致占比:")
        by_label = Counter()
        wrong_by_label = Counter()
        for (lbl, correct), v in stats.items():
            by_label[lbl] += v
            if not correct:
                wrong_by_label[lbl] += v
        for lbl, total in sorted(by_label.items(), key=lambda kv: -kv[1]):
            wrong = wrong_by_label.get(lbl, 0)
            print(f"  {lbl}: {wrong}/{total} 段不一致 ({wrong/total*100:.1f}%)")
    else:
        print("[review] 没有可对比的段")

    if wrong_rows and has_proba:
        print("\n[review] 预测错误的段，按最大置信度从低到高排序（越靠前越可疑，优先人工核对）:")
        print(_fmt_row(COLS))
        for r in sorted(wrong_rows, key=lambda r: r["max_conf"]):
            print(_fmt_row([
                (r["project_id"], 7), (r["task_id"], 7), (r["subject_id"], 17),
                (r["raw_label"], 10), (r["true_label"], 10), (r["n_windows"], 6),
                (r["pred_label"], 10), ("✗", 4),
                (f"{r['max_conf']:.2f}", 8), (f"{r['mean_conf']:.2f}", 8), (f"{r['all_conf']:.2f}", 8),
                (_fmt_time(r["t0"]), 22), (_fmt_time(r["t1"]), 22),
                (_fmt_time(r["pred_start_t"]), 22), (_fmt_time(r["pred_end_t"]), 22),
            ]) + f"  ({r['agree']}/{r['n_windows']})"
                  + _ls_url(args.ls_url_base, r["project_id"], r["task_id"]))

    if args.window_detail and wrong_rows and has_proba:
        print("\n[review] 逐窗口置信度明细（跟上面按置信度排序的顺序一致）:")
        for r in sorted(wrong_rows, key=lambda r: r["max_conf"]):
            if not r["window_details"]:
                continue
            print(f"\n  {r['project_id']} | {r['task_id']} | {r['subject_id']} | "
                  f"{r['raw_label']}→{r['pred_label']}"
                  + _ls_url(args.ls_url_base, r["project_id"], r["task_id"]))
            wcols = [("窗口", 4)] + [(c, 8) for c in classes] + [("自己投给", 10),
                     ("窗口起", 14), ("窗口止", 14)]
            print("    " + _fmt_row(wcols))
            for w in r["window_details"]:
                print("    " + _fmt_row(
                    [(w["idx"], 4)] + [(f"{w['probs'][c]:.2f}", 8) for c in classes]
                    + [(w["pred"], 10), (w["start"].strftime("%H:%M:%S.%f")[:-5], 14),
                       (w["end"].strftime("%H:%M:%S.%f")[:-5], 14)]))

    print(f"\n已保存: {log_path}")
    sys.stdout = sys.__stdout__
    log_f.close()


if __name__ == "__main__":
    main()
