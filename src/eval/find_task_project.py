"""
反查 record_id（形如 task496_imu1）对应的 Label Studio project 文件和 task id，
方便去 Label Studio 网页版按 project + task 定位具体标注核实。

背景: labelstudio_to_custom.py 把多个 project 导出的 JSON 合并转换成一份
训练用 CSV 时，record_id 只保留了 task_id（Label Studio 任务的全局唯一ID），
丢失了"来自哪个 project 导出文件"这个信息。这个脚本反过来在原始的
project-*.json 文件里搜索，找到 task_id 属于哪个文件（文件名里一般带有
project 编号，如 project-23-at-2026-07-30-06-04-c4825aac.json → project 23）。

用法:
  # record_id 形如 task496_imu1，只需要传数字部分
  python src/eval/find_task_project.py \\
    --task_ids 496 539 567 \\
    --json_dir data/raw_custom/2026_7_30

  # 也可以直接传 record_id 字符串，脚本会自动提取数字
  python src/eval/find_task_project.py \\
    --task_ids task496_imu1 task539_imu1 task567_imu2 \\
    --json_dir data/raw_custom/2026_7_30
"""

import argparse
import glob
import json
import os
import re


def extract_task_id(s):
    """从 'task496_imu1' 或 '496' 里提取纯数字 task id"""
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


def extract_project_no(filename):
    """从 'project-23-at-2026-07-30-06-04-c4825aac.json' 提取 project 编号"""
    m = re.search(r"project-(\d+)-at-", os.path.basename(filename))
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description="反查 task_id 属于哪个 Label Studio project 文件")
    ap.add_argument("--task_ids", nargs="+", required=True,
                     help="task id 列表，支持纯数字或 task496_imu1 这种 record_id 格式")
    ap.add_argument("--json_dir", required=True, help="存放 project-*.json 的目录")
    ap.add_argument("--pattern", default="project-*.json", help="项目导出文件名匹配模式")
    args = ap.parse_args()

    target_ids = {extract_task_id(t) for t in args.task_ids}
    target_ids.discard(None)
    if not target_ids:
        print("[错误] 没有解析出任何有效的 task id")
        return

    files = sorted(glob.glob(os.path.join(args.json_dir, args.pattern)))
    if not files:
        print(f"[错误] {args.json_dir} 下没有匹配 {args.pattern} 的文件")
        return

    print(f"共 {len(files)} 个 project 文件，查找 task id: {sorted(target_ids)}\n")

    found = {}  # task_id -> (filename, project_no, task_data_or_None)
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                tasks = json.load(f)
        except Exception as e:
            print(f"[跳过] {fpath}: 读取失败 ({e})")
            continue
        project_no = extract_project_no(fpath) or (
            str(tasks[0].get("project", "")) if tasks and isinstance(tasks[0], dict) else ""
        )
        for task in tasks:
            tid = task.get("id")
            if tid in target_ids and tid not in found:
                found[tid] = (fpath, project_no, task)

    print(f"{'='*90}")
    print(f"  查找结果")
    print(f"{'='*90}")
    for tid in sorted(target_ids):
        if tid in found:
            fpath, project_no, task = found[tid]
            proj_field = task.get("project", "")
            print(f"  task_id={tid}")
            print(f"    文件: {os.path.basename(fpath)}")
            print(f"    project 编号（从文件名解析）: {project_no or '未知'}"
                  + (f"  |  task内project字段: {proj_field}" if proj_field else ""))
            data = task.get("data", {})
            for k in ("csv", "video1", "video2", "cam1", "cam2"):
                if k in data:
                    print(f"    data.{k}: {data[k]}")
            print()
        else:
            print(f"  task_id={tid}  [未在任何 project 文件里找到，检查 --json_dir 是否正确]\n")

    print(f"{'='*90}")
    print("  在 Label Studio 网页版查找方式：")
    print("  打开对应 project 编号的项目 → Data Manager → 右上角搜索/筛选，")
    print("  按 task id 过滤（部分版本 URL 可直接拼: /projects/<project编号>/data?task=<task_id>）")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
