"""
把Label Studio导出的project JSON里，annotations[].completed_by的数字用户ID
换成邮箱地址，再导入另一个Label Studio实例——两个实例是完全独立的数据库，
同一个人在实例A可能是用户ID 5，在实例B可能是ID 4或者压根还没注册，数字ID
互不通用；但邮箱是跨实例稳定的，Label Studio导入校验"completed_by"字段时
邮箱和数字ID都认，所以换成邮箱就能绕开"not a valid annotator's email or ID"
这个报错。

只改annotations[].completed_by这一个字段——drafts[].user本来就是邮箱字符串
不用改；updated_by/last_created_by这些字段导入时不校验，留着数字ID也没事，
不去动它们，改动范围越小越安全。

用法:
  # 先去原实例的Django admin（/django-admin/htx_user/user/）或调用
  # GET /api/users/ （带Authorization token）确认数字ID对应的邮箱，
  # 然后：
  python src/remap_completed_by.py \
    --input project91at202608250329a0084b99.json \
    --output project91_remapped.json \
    --map "5=raven@hiccpet.com" --map "6=leon@hiccpet.com"

  # 有几个人的annotation就传几个--map，格式都是 数字ID=邮箱
"""
import argparse
import json


def remap(tasks: list, id_to_email: dict) -> tuple:
    """返回(改好的tasks, 统计信息)。统计信息用来在命令行里报告改了几处、
    遇到了哪些没在映射表里的ID（漏传了--map的话，这些annotation的
    completed_by会原样保留数字ID，导入时大概率还是会报同一个错，命令行
    输出里会明确列出来，不是静默漏改）。"""
    changed = 0
    unmapped_ids = set()
    for task in tasks:
        for ann in task.get("annotations", []):
            cb = ann.get("completed_by")
            if isinstance(cb, int):
                if cb in id_to_email:
                    ann["completed_by"] = id_to_email[cb]
                    changed += 1
                else:
                    unmapped_ids.add(cb)
    return tasks, {"changed": changed, "unmapped_ids": sorted(unmapped_ids)}


def parse_map_args(map_args: list) -> dict:
    result = {}
    for item in map_args:
        if "=" not in item:
            raise ValueError(f'--map格式应该是"数字ID=邮箱"，收到: {item!r}')
        id_str, email = item.split("=", 1)
        result[int(id_str.strip())] = email.strip()
    return result


def main():
    parser = argparse.ArgumentParser(description="把Label Studio导出JSON的completed_by数字ID换成邮箱")
    parser.add_argument("--input", required=True, help="从原实例导出的project JSON")
    parser.add_argument("--output", required=True, help="转换后的输出路径")
    parser.add_argument("--map", action="append", default=[],
                        help='数字ID=邮箱，比如 --map "5=raven@hiccpet.com"，可以传多次')
    args = parser.parse_args()

    id_to_email = parse_map_args(args.map)
    if not id_to_email:
        print("[错误] 至少要传一个 --map \"数字ID=邮箱\"")
        return

    with open(args.input, encoding="utf-8") as f:
        tasks = json.load(f)

    tasks, stats = remap(tasks, id_to_email)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"→ {args.output}  共改了{stats['changed']}处completed_by")
    if stats["unmapped_ids"]:
        print(f"⚠️ 还有这些数字ID没在--map里给对应邮箱，原样保留，导入时大概率还会报同一个错: "
              f"{stats['unmapped_ids']}")
        print("   去原实例确认这些ID对应哪个邮箱，补上--map参数重新跑一遍")


if __name__ == "__main__":
    main()
