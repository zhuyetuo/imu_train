"""
接真实数据，每天出一份 SBS 报告——衔接 run_infer_tf.sh 的推理结果和
scratch_burden.py 的打分引擎，供每天例行跑、跟兽医当天的目测评估对照。

一只狗对应一组 CSV_DIR（一个设备的原始数据目录）+ 对应的 _infer.json 目录。
两只狗分别跑两次（--pet_id 不同），各自独立算基线/评分，不要混在一起跑。

有效佩戴时长的估算：按当天实际记录到的样本行数 / 设备采样率算，这个值天然会
把设备缺口/佩戴不足的时段排除掉（缺口期间没有采样行，自然不计入），不需要
额外的佩戴检测逻辑（松动/未佩戴检测那部分工作先不接，见对话里"先不做"的决定）。

用法（每天跑一次，会用当前 CSV_DIR/infer_json_dir 下能看到的全部历史数据重算）：
    python src/eval/daily_skin_report.py \
        --pet_id dog1 \
        --csv_dir data/raw_tf_csv \
        --infer_json_dir infer_result_tf_majority/csv/_infer \
        --device_hz 50 \
        --out_csv results/skin_health/dog1_daily.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # for infer_csv_scratch (src/)
sys.path.insert(0, os.path.dirname(__file__))  # for scratch_burden (src/eval/)

from infer_csv_scratch import load_csv  # noqa: E402
from scratch_burden import load_events_from_infer_json, run_pipeline  # noqa: E402


def compute_wear_hours(csv_dir, pet_id, device_hz):
    """按当天实际记录到的行数/设备采样率估算每日有效佩戴小时数。"""
    rows = []
    for path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        acc, _, ts, valid_mask, null_ratio = load_csv(path)
        if ts is None or ts.isna().all():
            print(f"  [警告] {os.path.basename(path)} 没有可用时间戳，跳过佩戴时长统计")
            continue
        df = pd.DataFrame({"date": ts.dt.date, "valid": valid_mask})
        rows.append(df[df["valid"]])
    if not rows:
        return pd.DataFrame(columns=["pet_id", "date", "valid_wear_hours"])
    all_rows = pd.concat(rows, ignore_index=True)
    counts = all_rows.groupby("date").size()
    wear_hours = (counts / device_hz / 3600.0).reset_index()
    wear_hours.columns = ["date", "valid_wear_hours"]
    wear_hours.insert(0, "pet_id", pet_id)
    return wear_hours


_TIER_LABEL = {"C0": "C0 正常", "C1": "C1 需要关注", "C2": "C2 建议兽医检查",
               "insufficient_data": "数据不足"}


def write_vet_markdown(merged, pet_id, out_path):
    """跟 docs/skin_health_daily_template.md 同样的表格格式，
    '兽医目测评估'/'备注' 两列留空给人工填，算法这边只填自己算出来的那几列。"""
    lines = [
        f"# {pet_id} 皮肤评级日报（算法计算 vs 兽医目测，对照用）",
        "",
        "> 皮肤评级：C0 正常 / C1 需要关注 / C2 建议兽医检查。"
        "`是否引导期`=是 的行，评级是靠绝对秒数阈值兜底算的（还没有这只狗自己的21天历史基线可比），"
        "不影响兽医正常打分，两边独立判断即可。",
        "",
        "| 日期 | 算法-抓挠次数 | 算法-抓挠总时长(分钟) | 算法-皮肤评级 | 算法-触发红旗 | "
        "是否引导期(bootstrap_mode) | 兽医目测评估 | 备注 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in merged.iterrows():
        tier = _TIER_LABEL.get(row.get("tier"), row.get("tier", ""))
        red_flags = row.get("red_flags")
        red_flags_str = ", ".join(red_flags) if isinstance(red_flags, list) and red_flags else "-"
        bootstrap = "是" if row.get("bootstrap_mode") else "否"
        lines.append(
            f"| {row['date']} | {row.get('event_count', 0)} | {row.get('total_duration_min', 0):.2f} | "
            f"{tier} | {red_flags_str} | {bootstrap} | | |"
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pet_id", required=True)
    ap.add_argument("--csv_dir", required=True, help="该狗的原始CSV目录（用于估算每日有效佩戴时长）")
    ap.add_argument("--infer_json_dir", required=True, help="run_infer_tf.sh 产出的 _infer JSON 目录")
    ap.add_argument("--device_hz", type=int, required=True)
    ap.add_argument("--out_csv", default=None, help="把每日明细追加保存成CSV，方便跟兽医评分对照")
    ap.add_argument("--out_md", default=None,
                    help="导出成给兽医看的markdown表格（跟 docs/skin_health_daily_template.md "
                         "同样的格式），'兽医目测评估'/'备注' 两列"
                         "留空，每次重跑会覆盖整份文件，兽医已经填的内容记得先另存")
    args = ap.parse_args()

    json_paths = sorted(glob.glob(os.path.join(args.infer_json_dir, "*_infer.json")))
    if not json_paths:
        print(f"[错误] {args.infer_json_dir} 下没找到 *_infer.json，先跑 run_infer_tf.sh")
        sys.exit(1)

    print(f"读取 {len(json_paths)} 份推理结果...")
    events = load_events_from_infer_json(json_paths, args.pet_id)
    print(f"  共 {len(events)} 段抓挠事件")

    print(f"从 {args.csv_dir} 估算每日有效佩戴时长...")
    wear_hours = compute_wear_hours(args.csv_dir, args.pet_id, args.device_hz)
    if wear_hours.empty:
        print("[错误] 算不出任何一天的佩戴时长，检查CSV是否有有效时间戳")
        sys.exit(1)
    for _, row in wear_hours.iterrows():
        flag = "good" if row.valid_wear_hours >= 12 else ("partial" if row.valid_wear_hours > 0 else "insufficient")
        print(f"  {row.date}: {row.valid_wear_hours:.1f}小时 ({flag})")

    result = run_pipeline(events, wear_hours)
    result = result[result.pet_id == args.pet_id].sort_values("date").reset_index(drop=True)

    print("\n" + "=" * 100)
    print(f"{args.pet_id} 每日报告")
    print("=" * 100)
    cols = ["date", "event_count" if "event_count" in result.columns else None]
    # event_count/total_duration_sec 不在 score_day 的返回值里，需要从 daily 特征里带出来，
    # 这里直接从 events 表按天重新聚合一份，跟评分结果拼在一起展示，保证"次数/时长"这两个
    # 兽医最容易感知的原始指标每天都单独打印出来，不用只看抽象的分数。
    if len(events):
        ev = events.copy()
        ev["date"] = ev["start"].dt.date
        daily_raw = ev.groupby("date").agg(
            event_count=("start", "count"),
            total_duration_min=("duration_sec", lambda s: s.sum() / 60.0),
        ).reset_index()
    else:
        daily_raw = pd.DataFrame(columns=["date", "event_count", "total_duration_min"])

    merged = result.merge(daily_raw, on="date", how="left")
    merged["event_count"] = merged["event_count"].fillna(0).astype(int)
    merged["total_duration_min"] = merged["total_duration_min"].fillna(0).round(2)

    show_cols = ["date", "event_count", "total_duration_min", "total", "tier",
                 "delta_score", "cluster_score", "persistence_score", "interrupt_score",
                 "red_flags", "bootstrap_mode"]
    show_cols = [c for c in show_cols if c in merged.columns]
    print(merged[show_cols].to_string(index=False))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
        merged.to_csv(args.out_csv, index=False)
        print(f"\n已保存到 {args.out_csv}（每天重跑会用最新全部历史覆盖，不是追加）")

    if args.out_md:
        write_vet_markdown(merged, args.pet_id, args.out_md)
        print(f"已生成兽医对照表: {args.out_md}（每次重跑会整份覆盖，兽医已经填的'兽医目测评估'/'备注'"
              "记得先另存，这个脚本不会帮你保留）")

    print("\n把上面这张表的 date/event_count/total_duration_min/tier 这几列，跟兽医当天的目测评分按日期对齐，"
          "看趋势是否一致。bootstrap_mode=True 的行，是靠绝对阈值兜底算的分（还没有个人基线），"
          "不是靠这只狗自己的历史对比算的，见 docs/skin_health.md §2.3.1。")


if __name__ == "__main__":
    main()
