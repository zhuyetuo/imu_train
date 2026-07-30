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
--train_ratio     0.9     训练集比例（默认 0.9）
--val_ratio       0.1     验证集比例（默认 0.1，test = 1 - train - val）
```

**`--split_strategy` 说明**

| 值 | 行为 | 适用场景 |
|---|---|---|
| `random`（默认） | 把所有狗的窗口混合后，按比例随机划分 | 狗的数量少（<10条）时推荐。验证/测试集的类别分布与训练集接近，指标更稳定 |
| `subject` | 按狗 ID 划分，同一条狗只出现在一个集合里 | 狗的数量足够多时推荐。能真实评估模型对**未见过的新狗**的泛化能力，但若某几条狗的行为分布偏斜，验证集指标会出现类别严重不均的情况 |

> 狗的数量少时用 `subject` 会导致验证集某些类别样本极少（例如只有15个活动窗口），指标失去参考价值，因此默认 `random`。数据积累到覆盖20条以上的狗时，建议切换到 `subject` 以获得更真实的泛化评估。

**`--train_ratio` / `--val_ratio` 说明**

`test = 1 - train_ratio - val_ratio`，设为 0 即无测试集。

| 阶段 | 推荐比例 | 原因 |
|---|---|---|
| 纠错循环阶段（默认） | `--train_ratio 0.9 --val_ratio 0.1` | 数据量少，纠错样本应尽量全部进训练集；测试集意义不大，验证集用于监控过拟合即可 |
| 模型成熟、准备上线 | `--train_ratio 0.8 --val_ratio 0.1` | 留出 10% 测试集做最终无偏评估 |

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
| `MERGE_GAP` | 3 | 合并相邻片段的最大间隔秒数 |
| `MIN_WINDOWS` | 1 | 片段最少窗口数（1=不过滤） |
| `KEEP_ISOLATED` | 1 | 1=保留孤立单窗口片段，0=丢弃 |
| `BIN_BY` | conf_max | 置信度分桶依据（见下方说明） |
| `CONTEXT_S` | 3 | 裁剪片段前后保留秒数 |
| `CLIP_WORKERS` | 4 | 视频裁剪并行线程数 |
| `INCLUDE_DAYS` | （空=全部） | 空格分隔的白名单，只处理列出的日期 |
| `EXCLUDE_DAYS` | （空） | 空格分隔的跳过日期列表 |

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

## 参考论文

- Kumpulainen et al. (2021). *Dog behaviour classification with movement sensors placed on the harness and the collar.* Applied Animal Behaviour Science. https://doi.org/10.1016/j.applanim.2021.105393
- Chambers & Yoder (2020). *FilterNet: A Many-to-Many Deep Learning Architecture for Time Series Classification.* Sensors. https://doi.org/10.3390/s20092498
- van Herwijnen et al. (2021). *Deep Learning Classification of Canine Behavior Using a Single Collar-Mounted Accelerometer: Real-World Validation.* Animals. https://doi.org/10.3390/ani11061549
- Dunford et al. (2024). *Predicting cat behaviour using accelerometer data.* Ecology and Evolution. https://doi.org/10.1002/ece3.11368
- Smit et al. (2023). *Behaviour Classification of Extensively Kept Goats and Sheep Using Raw Accelerometer Data.* Sensors. https://doi.org/10.3390/s23052404
