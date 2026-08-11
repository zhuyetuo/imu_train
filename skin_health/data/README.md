# 数据存放说明

这个目录放皮肤评估任务的**产出物**（每日报告、跟兽医对照的表格、合成验证结果），不放原始IMU数据（原始CSV在仓库根目录的 `data/` 下，比如 `data/raw_tf_csv/`）。

内容默认不入 git（见根目录 `.gitignore`），因为：
- 每日报告/兽医对照表会频繁重新生成、包含兽医手填的内容，不适合用git追踪
- 合成验证脚本产出的CSV是可随时重新生成的临时产物

典型产出物：
- `dog1_daily.csv` / `dog1_daily.md`：`daily_skin_report.py` 每天跑出来的报告
- `synthetic_skin_health_sbs_report.csv`：`gen_synthetic_scratch_scenarios.py` 的合成验证结果

如果某天的报告需要长期保留存档（比如要给兽医/产品同事看的正式版本），手动复制一份改名（比如加上日期），不要依赖这个目录里的文件不被下次运行覆盖——脚本每次都是整份重写，不是追加。
