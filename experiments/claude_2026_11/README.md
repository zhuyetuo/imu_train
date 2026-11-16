# experiments/claude_2026_11

Claude 这一轮（2026-11）用来做方法对比、自动改标、自监督预训练的脚本。过程和结论见 `docs/claude_worklog.md`。
都在 GPU 服务器上跑（~/miniconda3 的 python，需要 xgboost / torch），默认读 `~/imu_train`（train_v2 的数据读取和特征）
和 NAS `~/ai_data`。

| 脚本 | 做什么 |
|---|---|
| `compare3.py` | 规则 / RF / XGB / 1D-CNN × 6 轴或只加速度，按（天, 项圈）分组 5 折；输出窗口级指标和每个文件在标注范围内的连续概率（没见过这个文件的模型）。`--train_ds` 用另一份导出的标签训练、仍按原标注评估；`--cnn_init` 用预训练的卷积层初始化 CNN |
| `eval_events.py` | 对 compare3 的连续概率做稳定版 v2 后处理（viterbi），跟人标抓挠片段配对：检测率、IoU 配对率、误报 / 小时，并扫抓挠概率系数，方便在相同误报下比较 |
| `baseline_stable.py` | 线上稳定版 v2（models/stable_v2_rf）在同一批文件上推理，作为基准 |
| `relabel.py` | 自动改标：不改原始数据，另存新数据集 + relabel_log.csv（每处改动的依据） |
| `ssl_pretrain.py` | NAS 未标数据上自监督预训练板上 CNN 的卷积层（SimCLR 式），排除跟评估数据同一天同一项圈的文件 |

示例：

```bash
python experiments/claude_2026_11/compare3.py --axes 3 --out ~/claude_exp/pool3
python experiments/claude_2026_11/eval_events.py --dir ~/claude_exp/pool3 --methods rule,rf,xgb,cnn
```
