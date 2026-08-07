# IMU 犬只 / 猫咪行为识别

基于项圈 IMU（加速度计 + 陀螺仪）数据，对比机器学习与深度学习在不同采样率、不同数据集下的行为分类效果。

---

## 文档

| 文档 | 内容 |
|------|------|
| [datasets.md](docs/datasets.md) | 数据集介绍、下载、预处理 |
| [training.md](docs/training.md) | ML/DL 训练、批量实验、实验结果、SHAP 分析 |
| [inference.md](docs/inference.md) | 离线推理、规则推理、实时 BLE 推理 |
| [vision.md](docs/vision.md) | 视觉行为识别（CLIP / 豆包 / YOLO26） |
| [features.md](docs/features.md) | 特征工程说明（每个特征对应的物理/行为意义）、行为分类体系（当前3分类 + 规划中的瘙痒细分类） |
| [skin_health.md](docs/skin_health.md) | 基于抓挠行为的皮肤健康评估方案（草案）：趋势报告 + 异常检测 |

---

## 项目结构

```
imu_train/
├── data/
│   ├── raw/                     ← 数据集A原始文件（不入 git）
│   ├── raw_b/                   ← 数据集B原始文件
│   ├── raw_custom/              ← 自采数据集（按日期子目录存放 JSON）
│   ├── raw_cat_dunford2024/     ← 猫咪数据集
│   ├── infer/                   ← 待推理的无标签 TXT/CSV 文件
│   └── processed_*/             ← 预处理结果（自动生成）
├── src/
│   ├── data/          ← 数据加载与预处理
│   ├── ml/            ← 机器学习
│   ├── dl/            ← 深度学习
│   ├── eval/          ← 结果对比与 SHAP 分析
│   ├── vision/        ← 视觉识别模块
│   ├── infer.py       ← ML/DL 离线推理
│   ├── infer_rule.py  ← 规则离线推理
│   └── infer_rule_live.py ← 实时 BLE 推理
├── witmotion_imu/     ← git submodule（BLE 解析）
├── configs/           ← 超参数配置
├── results/           ← 训练结果（自动生成，不入 git）
├── setup.sh           ← 数据预处理脚本
└── requirements.txt
```

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化 submodule（实时推理需要）
git submodule update --init

# 预处理数据集 A
bash setup.sh --dataset a

# 训练（ML + DL，并行）
python run_experiments.py --ml_workers 8 --dl_workers 4
```

### 自采数据训练

每次标注完成后从 Label Studio 导出 JSON，**按日期放入对应子目录**，方便管理多次导出：

```
data/raw_custom/
├── 2026_7_15/
│   ├── project-24-at-2026-07-15-09-20-4a6a29c1.json
│   └── project-9-at-2026-07-15-....json
├── 2026_7_23/
│   └── project-25-at-2026-07-23-....json
└── ...
```

以下示例以 `DATE=2026_7_23` 为当天日期，替换为实际值即可。

#### 步骤 0：分析标注质量

```bash
DATE=2026_7_23

# 分析当天所有 JSON（合并后分析）
python -c "
import json, glob, sys
files = glob.glob(sys.argv[1])
merged = []
for f in sorted(files):
    merged += json.load(open(f))
    print(f'  加载: {f}')
json.dump(merged, open(sys.argv[2], 'w'), ensure_ascii=False)
print(f'合并完成，共 {len(merged)} 条任务')
" \
  "data/raw_custom/${DATE}/*.json" \
  "data/raw_custom/${DATE}/merged_tmp.json"

python src/data/analyze_annotations.py \
  --json "data/raw_custom/${DATE}/merged_tmp.json"
# 输出：各类别片段数、总时长、占比、估算可用窗口数及均衡性警告
```

#### 步骤 1：转换标注 JSON → 训练 CSV

```bash
DATE=2026_7_23

python src/data/labelstudio_to_custom.py \
  --json "data/raw_custom/${DATE}/merged_tmp.json" \
  --output "data/raw_custom/${DATE}/merged_${DATE}.csv" \
  --csv_dir data/raw_wit/
# 脚本执行完会打印步骤 2、3 的完整命令，直接复制运行即可
```

#### 步骤 2：预处理

```bash
DATE=2026_7_23

python src/data/preprocess.py \
  --dataset custom \
  --raw_csv_custom "data/raw_custom/${DATE}/merged_${DATE}.csv" \
  --output_dir "data/processed_${DATE}" \
  --config configs/data.yaml \
  --split_strategy label_concat \
  --hz 16

# 查看类别分布（确认各类样本数是否均衡）
python src/data/analyze_dataset.py \
  --processed_dir "data/processed_${DATE}" \
  --hz 16
```

#### 步骤 3（可选）：合成少数类别数据

某个类别窗口数偏少时，从当天 JSON 标注自动提取该类别的所有片段进行数据增强：

```bash
DATE=2026_7_23

# 合成抓挠数据
# 加上 --processed_dir 后脚本会自动从训练集推算合理的 target_windows，无需手动填写
python src/data/synthesize_scratch.py \
  --json "data/raw_custom/${DATE}/merged_tmp.json" \
  --csv_dir data/raw_wit/ \
  --output "data/synthetic/scratch_${DATE}.npz" \
  --processed_dir "data/processed_${DATE}" \
  --remap configs/remap_custom_3class.yaml \
  --label 抓挠 --hz 16 --n_aug 50

# 合成其他少数类别（同理）
python src/data/synthesize_scratch.py \
  --json "data/raw_custom/${DATE}/merged_tmp.json" \
  --csv_dir data/raw_wit/ \
  --output "data/synthetic/sleep_${DATE}.npz" \
  --processed_dir "data/processed_${DATE}" \
  --remap configs/remap_custom_3class.yaml \
  --label 睡觉 --hz 16 --n_aug 50

# 验证生成数量
python -c "import numpy as np; d=np.load('data/synthetic/scratch_${DATE}.npz'); print('合成窗口数:', d['X'].shape)"
```

> 增强方式：加高斯噪声、幅值缩放（±15%）、轴向翻转、循环时移、时间拉伸。自动从 JSON 读取标注时间段，无需硬编码。

#### 步骤 3.5：确认数据分布

先不带合成数据跑 `--dry_run`，看各类别的原始训练窗口数：

```bash
DATE=2026_7_23

python src/ml/train.py --hz 16 --model rf \
  --processed_dir "data/processed_${DATE}" \
  --remap configs/remap_custom_3class.yaml \
  --dry_run
```

如果某个类别偏少（<其他类别 1/3），再加合成数据重新确认：

```bash
python src/ml/train.py --hz 16 --model rf \
  --processed_dir "data/processed_${DATE}" \
  --remap configs/remap_custom_3class.yaml \
  --synthetic "data/synthetic/scratch_${DATE}.npz" \
  --synthetic_label 抓挠 \
  --dry_run
```

输出示例：
```
[ml/train] ── 原始类别分布（映射前）──
  类别              训练      验证      测试      合计
  --------------------------------------------
  活动               800       100       100      1000
  睡觉               600        75        75       750
  抓挠                64         8         8        80
  甩身体              96        12        12       120
  ...

[ml/train] ── 数据集类别分布（含合成数据）──
  类别            训练      验证      测试      合计
  ------------------------------------------
  抓挠            1340       167       167      1674
  活动            3200       400       400      4000
  睡觉            1800       225       225      2250
  ------------------------------------------
  合计            6340       792       792      7924
```

各类别比例差距在 3 倍以内视为可接受，差距过大时可调整 `--n_aug` 重新生成合成数据。

#### 步骤 4：训练

**一键训练（推荐）**：预处理完成后直接跑脚本，自动并行训练两个模型并打印对比结果：

```bash
DATE=2026_7_23
bash train_custom.sh --date $DATE
```

可选参数：
```
--hz              16      采样率（默认 16）
--n_aug           50      每段原始片段的增强倍数（默认 50）
--label           抓挠    要合成的少数类别（默认 抓挠）
--split_strategy  random  训练/验证/测试划分策略（默认 random，见下）
--train_ratio     0.9        训练集比例（默认 0.9）
--val_ratio       0.1        验证集比例（默认 0.1）
--test_ratio      0.0        测试集比例（默认 0.0=无测试集）
--label_mode      majority   窗口标签怎么定（默认 majority，见下）
--stride_s        （留空）    训练窗口步长秒数，留空则用 configs/data.yaml 的默认值（通常1秒）
```

**`--label_mode` 说明**

| 值 | 行为 |
|---|---|
| `majority`（默认） | 窗口标签取窗口内多数投票的结果，原有行为不变 |
| `center` | 窗口标签取窗口正中心那一帧的标签，不做多数投票压制。要发挥效果需要配合更密的 `--stride_s`（比如 0.25 或 0.0625），否则中心点之间会有大段没有预测覆盖的空隙 |

> `label_concat` 划分策略下窗口本来就纯净（不会跨标注边界），`--label_mode` 对它没有影响。

**关于步长变密后的数据泄漏（已修复）**：训练/验证/测试划分现在是**按连续标注片段分组**，不是按窗口随机——同一次抓挠事件产生的所有窗口（不管重叠率多高、步长多密）保证被整体分进同一个集合，不会一部分进训练集、一部分进验证集导致验证分数虚高。这个修复对默认的 `majority` 模式同样生效，不只是给 `center` 模式用的。

**`--split_strategy` 说明**

> **"窗口纯净/不纯净"是什么意思**：每个训练窗口固定2秒，标签是这2秒内的行为类别。"纯净"指这2秒内自始至终只发生了一种行为（比如全程都是"活动"），窗口的标签能准确反映窗口内容；"不纯净"指这2秒中途发生了行为切换（比如前1.2秒"活动"、后0.8秒"抓挠"），此时窗口只能被打上一个标签，必然跟部分内容对不上——本质是"一个标签只能描述一段2秒混合信号里的一部分"。

| 值 | 行为 | 窗口是否纯净 | 适用场景 |
|---|---|---|---|
| `random`（**默认**） | 把所有狗的窗口混合后，按比例随机划分 | ❌ **不纯净**——滑窗用多数投票定标签（见下方⚠️说明），横跨标注边界的窗口会被强行标成占比更多的那一类，标签"看起来纯净"但窗口内原始信号可能是混合的 | 狗的数量少（<10条）时推荐。验证/测试集的类别分布与训练集接近，指标更稳定 |
| `subject` | 按狗 ID 划分，同一条狗只出现在一个集合里 | ❌ **不纯净**——划分逻辑跟 `random` 不同（按狗分），但窗口切分用的还是同一套多数投票滑窗，一样会产生混合窗口 | 狗的数量足够多时推荐。能真实评估模型对**未见过的新狗**的泛化能力，但若某几条狗的行为分布偏斜，验证集指标会出现类别严重不均的情况 |
| `label_concat` | 先把每条狗每种标签的**连续片段**单独抠出来（过滤掉太短的片段），再各自在片段内部滑窗——窗口不会跨越标注边界 | ✅ **纯净**——窗口内的原始信号只属于一个标签，不存在多数投票强行贴标签的情况 | 数据量小、片段短时用这个能更充分利用短片段，且训练数据没有边界噪声 |

> 狗的数量少时用 `subject` 会导致验证集某些类别样本极少（例如只有15个活动窗口），指标失去参考价值，因此默认 `random`。数据积累到覆盖20条以上的狗时，建议切换到 `subject` 以获得更真实的泛化评估。

> ⚠️ **`random` 和 `subject` 两种策略的窗口都可能跨越标注边界**：滑窗时用的是窗口内**多数投票**决定标签（`sliding_window()` 里 `Counter(frame_labels).most_common(1)`），一个横跨"活动→抓挠"边界的窗口，会被强行标成占比更多的那一类，标签本身"看起来纯净"（每条样本只有一个标签），但窗口内的原始信号其实是混合的。这是当前**默认训练流程（`random`）真实存在的情况**，不是笔误。只有 `label_concat` 从设计上完全避免了这个问题。
>
> **具体例子**（假设一个2秒窗口横跨"活动→抓挠"的边界）：
>
> | 窗口内容占比 | 多数投票结果 | 说明 |
> |---|---|---|
> | 90% 活动 + 10% 抓挠 | 标为 **活动** | 抓挠成分占比太小，直接被多数投票"抹掉"，这几帧抓挠信号混进了一条"活动"训练样本里 |
> | 60% 活动 + 40% 抓挠 | 标为 **活动** | 依然标活动，但窗口里近一半信号其实是抓挠——这条样本对模型来说是较强的噪声 |
> | 50% 活动 + 50% 抓挠（正好平局） | 标为**窗口内先出现的那个标签** | `Counter.most_common(1)` 平局时按插入顺序取第一个，也就是这2秒里先发生的那个行为胜出，不是随机、也不是固定偏向某一类，取决于窗口恰好从哪里切下去 |
> | 40% 活动 + 60% 抓挠 | 标为 **抓挠** | 同理，占比多的一方获胜，活动成分混入抓挠样本 |
>
> 越接近50/50的窗口，样本"名不副实"的程度越高，这类窗口在边界附近天然会持续出现（每次真实事件开始/结束都会产生一批），数量不算多但会持续污染训练数据。

**`--train_ratio` / `--val_ratio` 说明**

`test = 1 - train_ratio - val_ratio`，设为 0 即无测试集。

| 阶段 | 推荐比例 | 原因 |
|---|---|---|
| 纠错循环阶段（默认） | `--train_ratio 0.9 --val_ratio 0.1 --test_ratio 0.0` | 数据量少，纠错样本应尽量全部进训练集；测试集不参与训练也不影响训练决策，此阶段意义不大 |
| 模型成熟、准备上线 | `--train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1` | 留出 10% 测试集做最终无偏评估，评估结果完全独立于训练过程 |

输出：
```
results/processed_<DATE>/16hz_remap_custom_3class/ml_rf.pkl      ← 纯标注
results/processed_<DATE>/16hz_remap_custom_3class_syn/ml_rf.pkl  ← 带合成
```

---

**逐步训练（分析用）**：两种模型都训练，方便对比效果：

```bash
DATE=2026_7_23

# ── 方案 A：不带合成数据（纯标注）──
python src/ml/train.py --hz 16 --model rf \
  --processed_dir "data/processed_${DATE}" \
  --remap configs/remap_custom_3class.yaml
# 模型保存至 results/processed_${DATE}/16hz_remap_custom_3class/ml_rf.pkl

# ── 方案 B：带合成数据 ──
# 先生成合成数据（--processed_dir 自动推算目标窗口数）
python src/data/synthesize_scratch.py \
  --json "data/raw_custom/${DATE}/merged_tmp.json" \
  --csv_dir data/raw_wit/ \
  --output "data/synthetic/scratch_${DATE}.npz" \
  --processed_dir "data/processed_${DATE}" \
  --remap configs/remap_custom_3class.yaml \
  --label 抓挠 --hz 16 --n_aug 50

# 再训练，自动在目录名后加 _syn 后缀，不覆盖方案 A
python src/ml/train.py --hz 16 --model rf \
  --processed_dir "data/processed_${DATE}" \
  --remap configs/remap_custom_3class.yaml \
  --synthetic "data/synthetic/scratch_${DATE}.npz" \
  --synthetic_label 抓挠
# 模型保存至 results/processed_${DATE}/16hz_remap_custom_3class_syn/ml_rf.pkl
```

> - 采样率（`--hz`）必须与设备一致，推理时也要用同一个值
> - 数据集采集用 25Hz、部署用 16Hz 时：预处理加 `--source_hz 25`，训练和推理都用 `--hz 16`
> - 补充新数据后只需换 JSON 文件名重跑，旧版本数据完整保留
> - 抓挠数据足够多后可去掉 `--synthetic`，直接用标注数据训练（方案 A 即为此情况）

### TF 版 IMU（无蓝牙）离线数据转换

TF 版项圈没有蓝牙，只能离线导出 TXT 日志（`HH:MM:SS.MS,AX,AY,AZ,GX,GY,GZ`），用 `src/data/tf_offline_to_custom.py` 转成训练/标注用的 CSV：

```bash
# 单文件转换（文件名按 YYMMDDHH 猜日期，比如 26080712_tf2.TXT）
python src/data/tf_offline_to_custom.py convert data/raw_tf/26080712_tf2.TXT

# 整目录批量转换
python src/data/tf_offline_to_custom.py convert data/raw_tf/ -o data/raw_tf_csv/

# 从已转换的 CSV 按起止时间截取一段（核实可疑数据、单独标注某段时用）
python src/data/tf_offline_to_custom.py slice data/raw_tf_csv/26080712_tf2.csv \
  --start "2026-08-07 12:36:20.000" --end "2026-08-07 12:36:30.000" \
  -o data/raw_tf_csv/26080712_tf2_clip.csv
```

> 转换逻辑复用 `hicc_offline_to_labelstudio.py` 已验证过的处理方式：区分"真正跨午夜"（时间戳倒退接近一整天，日期+1）和"设备记录异常的小幅倒退"（丢弃该行保证 timestamp 严格递增），转换完会提示真实数据缺口。

### 标签映射（15类 → 3类）

Label Studio 中支持全部标签正常标注，训练时通过 `configs/remap_custom_3class.yaml` 自动合并：

| Label Studio 标签 | 训练类别 | 说明 |
|------------------|---------|------|
| 活动 | 活动 | |
| 睡觉 | 睡觉 | |
| 抓挠 | 抓挠 | |
| 甩身体 | 活动 | |
| 跳跃 | 活动 | |
| 舔身体 | 活动 | |
| 啃身体 | 活动 | |
| 奔跑 | 活动 | |
| 行走 | 活动 | |
| 进食 | 活动 | |
| 饮水 | 活动 | |
| 蹭擦身体 | 活动 | |
| 嗅闻 | 活动 | |
| 戴摘项圈 | 活动 | |
| 伸懒腰 | 活动 | |

> 数据积累到足够量后，删除 remap 文件中对应行即可将该类别拆出独立训练，无需修改其他代码。

### 实时 BLE 推理

```bash
# 扫描附近设备，获取 MAC 地址
python src/infer_rule_live.py --scan

# ── HICC_PetCollar ───────────────────────────────────────────
# 仅规则算法
python src/infer_rule_live.py --device hicc

# 仅 ML 模型
python src/infer_rule_live.py --device hicc --algo ml \
  --model results/processed_custom/20hz/ml_rf.pkl

# 指定 MAC 地址 + ML 模型（自己设备）
python src/infer_rule_live.py --device hicc --algo ml --model results/processed_custom/20hz/ml_rf.pkl --address EA:CB:3E:CF:00:1A --hz 20

# 规则 + ML 并排对比
python src/infer_rule_live.py --device hicc --algo rule ml \
  --model results/processed_custom/20hz/ml_rf.pkl

# 指定 MAC 地址（新设备或地址变了时用）
python src/infer_rule_live.py --device hicc --address AA:BB:CC:DD:EE:FF

# ── WitMotion WT901SDCL-BT50 ─────────────────────────────────
# 仅规则算法（自动扫描）
python src/infer_rule_live.py --device wit --hz 20

# 仅 ML 模型（设备 16Hz，模型也 16Hz）
python src/infer_rule_live.py --device wit --hz 16 --algo ml \
  --model results/processed_merged_all/16hz/ml_rf.pkl

# 设备 100Hz，模型训练用 16Hz（自动降采样）
python src/infer_rule_live.py --device wit --hz 100 --model_hz 16 --algo ml \
  --model results/processed_merged_all/16hz/ml_rf.pkl

# 设备 50Hz，模型训练用 16Hz（自动降采样）
python src/infer_rule_live.py --device wit --hz 50 --model_hz 16 --algo ml \
  --model results/processed_merged_all/16hz/ml_rf.pkl

# 规则 + ML 并排对比
python src/infer_rule_live.py --device wit --hz 50 --model_hz 16 --algo rule ml \
  --model results/processed_merged_all/16hz/ml_rf.pkl

# 指定 MAC 地址
python src/infer_rule_live.py --device wit --hz 50 --model_hz 16 --address AA:BB:CC:DD:EE:FF \
  --algo ml --model results/processed_merged_all/16hz/ml_rf.pkl
```

> `--hz` 是设备实际采样率，`--model_hz` 是模型训练时的采样率，不同时自动降采样，无需修改设备配置。

### 离线 CSV 推理（验证历史采集数据）

对已录制的 CSV 文件批量推理，输出每个时间窗口的预测结果和抓挠片段汇总：

```bash
# 单个 CSV（设备采样率与模型一致）
python src/infer_csv_scratch.py \
  --csv data/raw_wit/multicam_20260715_084939_cam1_imu1_resampled16hz.csv \
  --model results/processed_merged_all/16hz/ml_rf.pkl

# 设备 100Hz CSV，模型训练用 16Hz（自动降采样）
python src/infer_csv_scratch.py \
  --csv data/raw_wit/rec_wit_20260629.csv \
  --model results/processed_merged_all/16hz/ml_rf.pkl \
  --device_hz 100 --model_hz 16

# 批量处理目录下所有 imu1 CSV
python src/infer_csv_scratch.py \
  --csv_dir data/raw_wit/ \
  --pattern "*imu1*.csv" \
  --model results/processed_merged_all/16hz/ml_rf.pkl \
  --device_hz 16
```

输出示例：
```
  时间                   预测    置信度
  --------------------------------------
  2026-07-15 08:52:37    活动     0.91
  2026-07-15 08:52:39    抓挠     0.87  ⬅ 抓挠
  2026-07-15 08:52:41    抓挠     0.92  ⬅ 抓挠

  【抓挠片段】
    08:52:39 → 08:52:43
```

详细用法见各文档页。

**片段边界怎么算，取决于模型的 `label_mode`（自动从模型 `.json` 元数据读取，无需手动指定）**

| 模型的 label_mode | 片段边界 |
|---|---|
| `majority`（没有 `.json` 元数据的旧模型，退化为这个） | 窗口整段跨度——连续正例窗口的第一个窗口起点 → 最后一个窗口起点+窗口长度 |
| `center` | 窗口中心点 ±步长/2——第一个正例窗口的"中心点-半步长" → 最后一个正例窗口的"中心点+半步长"，紧贴事件真实范围，比 majority 窄很多 |

用 `--label_mode center` 训练出来的模型，如果这里不识别 `label_mode`，片段边界会按 majority 语义重建（整段窗口跨度），比真实事件明显偏宽——训练时中心点标注带来的边界精度收益，在生成裁剪片段这一步就会被浪费掉。

**`infer_csv_scratch.py` 片段过滤参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--merge_gap` | 3 | 相邻抓挠片段间隔 ≤ N 秒时合并为一段 |
| `--min_windows` | 1 | 片段包含的窗口数不足时丢弃（1=不过滤） |
| `--no_keep_isolated` | 不传 | 传此参数则丢弃**孤立单窗口片段**（见下方说明） |

**孤立单窗口片段**：前一个窗口和后一个窗口均不是抓挠，只有中间这1个窗口被判为抓挠。由于相邻窗口有1秒重叠，真实抓挠（通常持续2秒以上）几乎必然触发连续多个窗口；孤立单窗口通常意味着动作持续时间不足1秒，可能是瞬时晃动误报，也可能是真实的极短抓挠。默认**保留**，依靠置信度分桶来区分优先级。如果发现低置信度孤立窗口大量出现，可加 `--no_keep_isolated` 过滤。

---

### 批量推理 + 视频裁剪 + Label Studio 复查

```bash
DATA_ROOT=data/raw_custom/data \
MODEL=results/processed_2026_7_23/16hz_remap_custom_3class_syn/ml_rf.pkl \
WORKERS=16 RESULT_ROOT=infer_result \
PATTERN="*resampled16hz.csv" DEVICE_HZ=16 \
./run_review_bins_all_days.sh
```

只处理特定日期：
```bash
INCLUDE_DAYS="2026_7_24" \
DATA_ROOT=... MODEL=... \
./run_review_bins_all_days.sh
```

**`run_review_bins_all_days.sh` 环境变量**

| 变量 | 默认 | 说明 |
|---|---|---|
| `MERGE_GAP` | 1 | 合并相邻片段的最大间隔秒数（`event_eval.py` 用真实数据验证过，3s会把约一半真实抓挠事件错误合并，1s已能消除碎片化且合并更少，见下方"事件级评估"） |
| `MIN_WINDOWS` | 1 | 片段最少窗口数（1=不过滤） |
| `KEEP_ISOLATED` | 1 | 1=保留孤立单窗口片段，0=丢弃 |
| `BIN_BY` | conf_max | 置信度分桶依据（见下方说明） |
| `CONTEXT_S` | 3 | 裁剪片段前后保留秒数 |
| `CLIP_WORKERS` | 4 | 视频裁剪并行线程数 |
| `INCLUDE_DAYS` | （空=全部） | 空格分隔的白名单，只处理列出的日期 |
| `EXCLUDE_DAYS` | （空） | 空格分隔的跳过日期列表 |

> 底层调用的是 `infer_csv_scratch.py`，片段边界会按模型的 `label_mode` 自动切换（见上方"片段边界怎么算"），不用额外传参数。

**`BIN_BY`：置信度分桶依据**

每段抓挠片段同时记录 `conf_max`（段内最高置信度）和 `conf_mean`（段内平均置信度）。

| 值 | 行为 | 适用场景 |
|---|---|---|
| `conf_max`（默认） | 按段内**最高**置信度分桶 | 复查优先：只要片段内有一个窗口模型非常确定，就放入高优先级桶，不会因为前后犹豫窗口被降级 |
| `conf_mean` | 按段内**平均**置信度分桶 | 保守优先：要求整段都持续高置信度才进高桶，适合已知误报率较低时使用 |

两个值都会保存在 `_infer.json` 里，无论用哪个分桶，另一个仍可在日志中查看。

**复查优先级建议**（适用于两种 `BIN_BY`）：

```
clips_0.8-1.0/  ← 先看，模型最确定，漏掉真实抓挠代价最高
clips_0.6-0.8/  ← 次看，边界案例，修正后对模型提升最大
clips_0.3-0.6/  ← 数量多时可抽样，主要用于捡漏和分析误报类型
```

---

## 事件级评估与参数自动调优（`event_eval.py`）

`ml/train.py` 打印的准确率/F1 是**逐窗口**指标，看不出"一次完整的抓挠有没有被切碎成好几段"或"两次独立抓挠有没有被错误粘成一段"这类问题——这类信息只有把窗口预测拼回连续事件、跟真实标注按事件对比才能看到。`src/eval/event_eval.py` 用 [ward-metrics](https://pypi.org/project/ward-metrics/) 库做这件事，同时会自动网格搜索出当前模型最合适的 `--confidence_threshold` 和 `--merge_gap`。

```bash
pip install ward-metrics   # import 名是 wardmetrics，注意不一致

python src/eval/event_eval.py \
  --labeled_csv data/raw_custom/2026_7_30/merged_all_labels_2026_7_30.csv \
  --model results/processed_2026_7_30/16hz_remap_custom_3class_syn/ml_rf.pkl \
  --hz 16 --target_label 抓挠 \
  --scan_mode full \
  --json_dir data/raw_custom/2026_7_30 \
  --log_file logs/event_eval_2026_7_30.log
```

`--labeled_csv` 要用**未按 `--keep_labels` 过滤的全量标注CSV**（用 `labelstudio_to_custom.py --keep_labels` 不传值生成，见上文"标注分析"一节），这样窗口级分类报告和事件提取才有完整的真实标签可用。

`labelstudio_to_custom.py` 生成的CSV现在带一列 `timestamp`（原始CSV里解析出来的绝对时间），`event_eval.py` 会自动识别这一列，在逐条对应表里除了相对秒数还会多打印一行"绝对时间戳"，方便直接去原始CSV/视频里定位对应位置。用旧版脚本生成、没有这一列的CSV仍然能跑，只是不显示绝对时间戳（自动降级）。

### 输出内容（按顺序）

1. **窗口级分类报告**：标准的多分类 precision/recall/f1-score，只统计真实标签在模型已知类别内的窗口（比如标注CSV里的"啃身体"模型没训练过，会被跳过，不会拉低指标）
2. **窗口级置信度分布**：预测为目标行为的窗口，有多少落在各置信度阈值以上
3. **网格搜索**：扫描 `(置信度阈值, merge_gap)` 组合，用论文定义的 **F1e**（把碎片化F、合并M都算作错误，跟 ward-metrics 库自带、把"合并"当命中处理的 precision/recall 不同）挑出最优组合
4. **推荐配置详细拆解**：真实事件总数 = 精确匹配C + 漏检D + 碎片F + 合并M（+FM），带验算
5. **全部真实事件逐条对应表**：每条事件的类别、真实时间、匹配到的预测区间，按 `record_id` 分块，方便逐条去 Label Studio 复查
6. **被合并的真实事件组**：单独摘出"合并"这类问题，附上间隔秒数——间隔很短（<1秒）通常是标注切碎了同一个连续动作；间隔一两秒甚至更长，更可能是模型/参数层面的问题，也可能是真实的"多次独立动作间隔很近"（需要人工核实，见下方案例）

### 网格搜索两种模式

| `--scan_mode` | 行为 | 适用场景 |
|---|---|---|
| `auto`（默认） | 先用 `--confidence_threshold`/`--merge_gap` 列表跑一遍粗网格，取 Top5 后在其范围附近（默认阈值±0.05、gap±0.5s）自动做精细网格 | 想快速拿到一个还不错的参数；数据量大、粗网格能覆盖到真实最优附近时用 |
| `full` | 直接在 `--full_thr_start/stop/step`、`--full_gap_start/stop/step` 指定的完整范围内做一次精细网格（默认阈值0~1步长0.01、gap 0.5~5s步长0.1，约4600组合） | 想保证找到全局最优，不想因为粗网格没覆盖到真实最优所在区间而错过。模型推理只做一次、缓存复用，网格本身是纯Python比对，通常几秒到几十秒能跑完，代价不大，**推荐默认用这个** |

网格搜索会显示 `tqdm` 进度条（走 stderr，不会污染 `--log_file` 保存的日志内容）。

### 其他参数

| 参数 | 说明 |
|---|---|
| `--json_dir` | Label Studio `project-*.json` 所在目录，传了的话逐条对应表/被合并事件组会顺带打印每个事件对应的 project 文件、video/csv 链接，不用再单独跑 `find_task_project.py` |
| `--log_file` | 把完整输出另存一份到文件（每次覆盖写，不追加），方便复查时对照 |

### 反查 record_id 对应的 Label Studio project（`find_task_project.py`）

事件表里的 `record_id`（形如 `task496_imu1`）只保留了 Label Studio 的 task_id，丢失了来自哪个 project 导出文件这个信息。`--json_dir` 已经把这一步自动接进 `event_eval.py` 了；如果只是想单独查几个 task_id，也可以直接跑：

```bash
python src/eval/find_task_project.py \
  --task_ids task496_imu1 task539_imu1 \
  --json_dir data/raw_custom/2026_7_30
```

---

## 参考论文

- Kumpulainen et al. (2021). *Dog behaviour classification with movement sensors placed on the harness and the collar.* Applied Animal Behaviour Science. https://doi.org/10.1016/j.applanim.2021.105393
- Chambers & Yoder (2020). *FilterNet: A Many-to-Many Deep Learning Architecture for Time Series Classification.* Sensors. https://doi.org/10.3390/s20092498
  （开放获取: MDPI https://www.mdpi.com/1424-8220/20/9/2498 ｜ PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC7249062/）
- van Herwijnen et al. (2021). *Deep Learning Classification of Canine Behavior Using a Single Collar-Mounted Accelerometer: Real-World Validation.* Animals. https://doi.org/10.3390/ani11061549
- Dunford et al. (2024). *Predicting cat behaviour using accelerometer data.* Ecology and Evolution. https://doi.org/10.1002/ece3.11368
- Smit et al. (2023). *Behaviour Classification of Extensively Kept Goats and Sheep Using Raw Accelerometer Data.* Sensors. https://doi.org/10.3390/s23052404

## 参考代码仓库

- [WhistleLabs/FilterNet](https://github.com/WhistleLabs/FilterNet) — 上面 FilterNet 论文的官方仓库（Whistle/Pet Insight Project 发布，组织现显示为 "Former Whistle Labs" 但仓库仍可访问）。`pip install -e .` 安装，`scripts/` 复现论文实验，`notebooks/` 复现论文图表。当前项目未直接使用其深度学习模型，但预处理/窗口切分思路和事件级评估方法论值得借鉴（见下方讨论）。
  - 非官方衍生版本（仅供参考，命名容易混淆，注意区分）：
    [vlainic/FilterNet-Keras](https://github.com/vlainic/FilterNet-Keras)（第三方 Keras 复现，仅模型结构）、
    [Mikata-Project/FilterNet](https://github.com/Mikata-Project/FilterNet)（同名但不同项目，PyTorch + fastai 的 1D CNN，灵感来自 WaveNet，与 Whistle 的 FilterNet 无关）
- [ward-metrics](https://pypi.org/project/ward-metrics/) — 事件级别（而非逐帧）分类评估指标库，实现 Ward et al. 提出的事件对应关系分类法（正确匹配/漏检/碎片化/合并/插入误报等）。与模型架构无关，可直接用在当前的随机森林输出上。项目里 `src/eval/event_eval.py` 已经接入，用法见该脚本头部注释；需要 `pip install ward-metrics`（import 名是 `wardmetrics`，注意不一致）。
