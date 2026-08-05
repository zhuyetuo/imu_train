"""
临时诊断脚本：统计标注JSON里 from_name 字段的分布，尤其是双传感器(is_multi)任务里的
from_name 取值——用于排查 synthesize_scratch.py / labelstudio_to_custom.py 里
"from_name 必须是 label1/label2 才能对应 csv1/csv2" 这个假设是否成立。

用法:
  python src/data/inspect_from_name.py data/raw_custom/2026_7_30/merged_tmp.json
"""
import json
import sys
from collections import Counter

with open(sys.argv[1], encoding="utf-8") as f:
    tasks = json.load(f)

fn_counter = Counter()
fn_by_multi = Counter()
examples = {}

# 针对 from_name=='label' 的 is_multi 任务，看它实际有几个csv链接（只有一个 还是 两个都有）
label_fn_csv_count = Counter()

for task in tasks:
    data = task.get("data", {})
    is_multi = "csv1" in data or "csv2" in data
    n_csv = sum(1 for idx in ("1", "2") if data.get(f"csv{idx}", ""))
    for ann in task.get("annotations", []):
        for seg in ann.get("result", []):
            fn = seg.get("from_name", "")
            fn_counter[fn] += 1
            if is_multi:
                fn_by_multi[fn] += 1
                if fn not in examples:
                    examples[fn] = task["id"]
                if fn == "label":
                    label_fn_csv_count[n_csv] += 1

print("全部 from_name 分布:")
for fn, cnt in fn_counter.most_common():
    print(f"  {fn!r}: {cnt}")

print("\n双传感器(is_multi)任务里的 from_name 分布:")
for fn, cnt in fn_by_multi.most_common():
    print(f"  {fn!r}: {cnt}  (例: task{examples[fn]})")

print("\nfrom_name=='label' 且 is_multi 的标注段，所在task实际有几个csv链接:")
for n_csv, cnt in sorted(label_fn_csv_count.items()):
    print(f"  csv链接数={n_csv}: {cnt} 段")
