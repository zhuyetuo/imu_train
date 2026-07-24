"""
把推理结果 JSON 转换为 Label Studio 任务，用于人工复查和重新标注。

输出三类任务：
  1. 检测到抓挠的片段（需人工确认是否真实）
  2. 置信度低但疑似抓挠的窗口（抓挠概率 > low_threshold，但未达最高）
  3. 可选：所有窗口全量上传（用于首次从零标注）

用法:
  python src/review_to_labelstudio.py \
    --infer_dir infer_result/2026_7_15 \
    --output infer_result/2026_7_15/labelstudio_review.json \
    --csv_url_prefix http://labelstudio.local:8080/data/local-files/?d=raw_wit

  # 只上传检测到的抓挠片段（最常用）
  python src/review_to_labelstudio.py \
    --infer_dir infer_result/2026_7_15 \
    --output infer_result/2026_7_15/labelstudio_review.json \
    --mode scratch_only
"""

import argparse
import glob
import json
import os


def ts_to_label_studio(ts_str: str) -> str:
    """确保时间戳格式为 Label Studio 接受的格式（毫秒精度）。"""
    if ts_str is None:
        return ""
    # 已经是 %Y-%m-%d %H:%M:%S.mmm 格式
    return ts_str


def make_annotation(start_ts, end_ts, label, from_name="label", to_name="ts"):
    """生成一条 Label Studio timeserieslabels 标注结果。"""
    return {
        "type": "timeserieslabels",
        "from_name": from_name,
        "to_name": to_name,
        "value": {
            "start": ts_to_label_studio(start_ts),
            "end":   ts_to_label_studio(end_ts),
            "timeserieslabels": [label],
        }
    }


def build_tasks(infer_jsons, csv_url_prefix, mode, low_threshold, high_threshold,
                context_s, label_name):
    """
    mode:
      scratch_only  - 只上传检测到抓挠的片段（含前后 context_s 秒上下文）
      uncertain     - 只上传低置信度疑似抓挠窗口
      all           - 两者都上传
    """
    tasks = []
    task_id = 1

    for infer_path in sorted(infer_jsons):
        with open(infer_path, encoding="utf-8") as f:
            data = json.load(f)

        csv_basename = data["csv_basename"]
        csv_url      = f"{csv_url_prefix.rstrip('/')}/{csv_basename}"
        windows      = data.get("windows", [])
        scratch_segs = data.get("scratch_segments", [])

        # ── 模式 1：检测到的抓挠片段 ────────────────────────────────
        if mode in ("scratch_only", "all") and scratch_segs:
            results = []
            for seg in scratch_segs:
                if seg["start_ts"] and seg["end_ts"]:
                    results.append(make_annotation(seg["start_ts"], seg["end_ts"], label_name))
            if results:
                tasks.append({
                    "id":   task_id,
                    "data": {"csv": csv_url},
                    "annotations": [{"result": results}],
                    "meta": {
                        "source":   "infer_scratch",
                        "csv_file": csv_basename,
                        "note":     f"模型检测到 {len(scratch_segs)} 段抓挠，请核实",
                    }
                })
                task_id += 1

        # ── 模式 2：低置信度疑似抓挠（未达最高但 scratch_prob > low_threshold）─
        if mode in ("uncertain", "all"):
            uncertain_windows = []
            for w in windows:
                scratch_prob = w["probs"].get(label_name, 0.0)
                is_predicted_scratch = (w["label"] == label_name)
                # 预测为非抓挠，但抓挠概率超过低阈值 → 可能漏检
                if not is_predicted_scratch and low_threshold < scratch_prob <= high_threshold:
                    uncertain_windows.append(w)

            if uncertain_windows:
                # 按时间段聚合连续的疑似窗口
                results = []
                for w in uncertain_windows:
                    if w["ts"]:
                        results.append(make_annotation(w["ts"], w["ts"], f"{label_name}?"))
                if results:
                    tasks.append({
                        "id":   task_id,
                        "data": {"csv": csv_url},
                        "annotations": [{"result": results}],
                        "meta": {
                            "source":   "infer_uncertain",
                            "csv_file": csv_basename,
                            "note":     f"{len(uncertain_windows)} 个窗口抓挠概率在 [{low_threshold:.0%}, {high_threshold:.0%}] 之间，请人工判断",
                        }
                    })
                    task_id += 1

    return tasks


def main():
    parser = argparse.ArgumentParser(description="推理结果 → Label Studio 复查任务")
    parser.add_argument("--infer_dir",  required=True,
                        help="包含 *_infer.json 的目录（run_review_bins_all_days.sh 输出）")
    parser.add_argument("--output",     required=True,
                        help="输出 Label Studio JSON 路径")
    parser.add_argument("--csv_url_prefix", default="http://localhost:8080/data/local-files/?d=raw_wit",
                        help="Label Studio 中 CSV 文件的 URL 前缀")
    parser.add_argument("--mode", default="scratch_only",
                        choices=["scratch_only", "uncertain", "all"],
                        help="生成任务类型（默认 scratch_only）")
    parser.add_argument("--low_threshold",  type=float, default=0.3,
                        help="疑似抓挠下限概率（默认 0.3）")
    parser.add_argument("--high_threshold", type=float, default=0.65,
                        help="疑似抓挠上限概率（超过则应该已经被预测为抓挠，默认 0.65）")
    parser.add_argument("--context_s", type=float, default=5.0,
                        help="抓挠片段前后的上下文秒数（默认 5s）")
    parser.add_argument("--label", default="抓挠",
                        help="要复查的目标类别（默认 抓挠）")
    args = parser.parse_args()

    infer_jsons = glob.glob(os.path.join(args.infer_dir, "**", "*_infer.json"), recursive=True)
    infer_jsons += glob.glob(os.path.join(args.infer_dir, "*_infer.json"))
    infer_jsons = sorted(set(infer_jsons))

    if not infer_jsons:
        print(f"[错误] {args.infer_dir} 下没有找到 *_infer.json 文件")
        return

    print(f"找到 {len(infer_jsons)} 个推理结果文件")

    tasks = build_tasks(
        infer_jsons,
        csv_url_prefix=args.csv_url_prefix,
        mode=args.mode,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        context_s=args.context_s,
        label_name=args.label,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    n_scratch   = sum(1 for t in tasks if t["meta"]["source"] == "infer_scratch")
    n_uncertain = sum(1 for t in tasks if t["meta"]["source"] == "infer_uncertain")
    print(f"共生成 {len(tasks)} 个 Label Studio 任务")
    print(f"  检测到抓挠: {n_scratch} 个文件")
    print(f"  疑似漏检:   {n_uncertain} 个文件")
    print(f"已保存: {args.output}")
    print(f"\n导入方式: Label Studio → Import → 选择上面的 JSON 文件")


if __name__ == "__main__":
    main()
