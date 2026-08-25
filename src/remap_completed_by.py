"""
把Label Studio导出的project JSON里，annotations[].completed_by换成目标实例
（要导入进去的那个实例）里对应用户的标识，解决"两个Label Studio实例是完全
独立的数据库，同一个人在实例A可能是用户ID 5，在实例B可能是ID 4或者压根还
没注册，数字ID互不通用"这个问题。

--map的值优先用目标实例里对应用户的数字ID（Label Studio官方文档写明
completed_by就是数字ID这一种格式，最稳）；传邮箱字符串也支持（部分版本/
路径可能认，官方错误提示文案里写了"email or ID"），但实测1.23.0版本邮箱
字符串这条路走不通(报"not a valid annotator's email or ID")，能拿到目标
实例的数字ID的话优先用数字ID，邮箱只是备选，不保证每个版本都生效。

只改annotations[].completed_by这一个字段——drafts[].user本来就是邮箱字符串
不用改；updated_by/last_created_by这些字段导入时不校验，留着原样也没事，
不去动它们，改动范围越小越安全。

用法:
  # 拿目标实例(要导入进去的那个)的API token，查GET /api/users/确认每个
  # 邮箱对应的数字ID，然后用"原实例的数字ID=目标实例的数字ID"这样传：
  python src/remap_completed_by.py \
    --input project91at202608250329a0084b99.json \
    --output project91_remapped.json \
    --map "5=12" --map "6=13"

  # 邮箱字符串也支持(备选，不保证生效)：
  #   --map "5=raven@hiccpet.com"
"""
import argparse
import json


def remap(tasks: list, id_map: dict) -> tuple:
    """返回(改好的tasks, 统计信息)。id_map的value可能是int(目标实例的数字ID)
    或str(邮箱)，直接原样赋值——数字ID赋成JSON整数，不是字符串"12"，这点
    很重要，Label Studio原本这个字段就是整数类型，赋成字符串"12"未必能正确
    匹配到用户。统计信息用来在命令行里报告改了几处、遇到了哪些没在映射表
    里的ID（漏传了--map的话，这些annotation的completed_by原样保留，导入
    时大概率还是会报同一个错，命令行输出里会明确列出来，不是静默漏改）。"""
    changed = 0
    unmapped_ids = set()
    for task in tasks:
        for ann in task.get("annotations", []):
            cb = ann.get("completed_by")
            if isinstance(cb, int):
                if cb in id_map:
                    ann["completed_by"] = id_map[cb]
                    changed += 1
                else:
                    unmapped_ids.add(cb)
    return tasks, {"changed": changed, "unmapped_ids": sorted(unmapped_ids)}


def parse_map_args(map_args: list) -> dict:
    """value是纯数字字符串就转成int(目标实例的数字ID)，否则原样当邮箱
    字符串处理——这样同一个--map参数格式，数字ID和邮箱两种写法都能自动
    识别，不用额外传一个"这是ID还是邮箱"的开关。"""
    result = {}
    for item in map_args:
        if "=" not in item:
            raise ValueError(f'--map格式应该是"原ID=目标ID"或"原ID=邮箱"，收到: {item!r}')
        id_str, value = item.split("=", 1)
        value = value.strip()
        result[int(id_str.strip())] = int(value) if value.isdigit() else value
    return result


def main():
    parser = argparse.ArgumentParser(description="把Label Studio导出JSON的completed_by换成目标实例的用户ID/邮箱")
    parser.add_argument("--input", required=True, help="从原实例导出的project JSON")
    parser.add_argument("--output", required=True, help="转换后的输出路径")
    parser.add_argument("--map", action="append", default=[],
                        help='原实例的数字ID=目标实例的数字ID或邮箱，比如 --map "5=12"，可以传多次')
    args = parser.parse_args()

    id_map = parse_map_args(args.map)
    if not id_map:
        print('[错误] 至少要传一个 --map "原ID=目标ID或邮箱"')
        return

    with open(args.input, encoding="utf-8") as f:
        tasks = json.load(f)

    tasks, stats = remap(tasks, id_map)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"→ {args.output}  共改了{stats['changed']}处completed_by")
    if stats["unmapped_ids"]:
        print(f"⚠️ 还有这些数字ID没在--map里给对应邮箱，原样保留，导入时大概率还会报同一个错: "
              f"{stats['unmapped_ids']}")
        print("   去原实例确认这些ID对应哪个邮箱，补上--map参数重新跑一遍")


if __name__ == "__main__":
    main()
