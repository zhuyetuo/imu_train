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
│   ├── raw_custom/              ← 自采数据集
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

每次在 Label Studio 补充标注后导出一个新 JSON，流程自动以 JSON 文件名命名数据集，不会覆盖旧版本。

```bash
# 以下示例用 project-24-at-2026-07-15-09-20-4a6a29c1.json，替换为你的实际文件名

# 步骤 0：分析标注质量（不需要下载 CSV，秒出结果）
python src/data/analyze_annotations.py \
  --json data/raw_custom/project-24-at-2026-07-15-09-20-4a6a29c1.json
# 输出：各类别片段数、总时长、占比、估算可用窗口数及均衡性警告

# 步骤 1：转换标注 JSON → 训练 CSV
# --output 可省略，自动保存为 data/raw_custom/<json文件名>.csv
python src/data/labelstudio_to_custom.py \
  --json data/raw_custom/project-24-at-2026-07-15-09-20-4a6a29c1.json \
  --csv_dir data/raw_wit/
# 脚本执行完会打印步骤 2、3 的完整命令，直接复制运行即可

# 步骤 2：预处理（将 <json文件名> 替换为实际值）
python src/data/preprocess.py \
  --dataset custom \
  --raw_csv_custom data/raw_custom/project-24-at-2026-07-15-09-20-4a6a29c1.csv \
  --output_dir data/processed_project-24-at-2026-07-15-09-20-4a6a29c1 \
  --config configs/data.yaml

# 步骤 2.5：查看数据集类别分布
python src/data/analyze_dataset.py \
  --processed_dir data/processed_project-24-at-2026-07-15-09-20-4a6a29c1 \
  --hz 16

# 步骤 3：训练（按设备实际采样率选 hz）
python src/ml/train.py --hz 16 --model rf \
  --processed_dir data/processed_project-24-at-2026-07-15-09-20-4a6a29c1
```

> - 采样率（`--hz`）必须与设备一致，推理时也要用同一个值
> - 数据集采集用 25Hz、部署用 16Hz 时：预处理加 `--source_hz 25`，训练和推理都用 `--hz 16`
> - 补充新数据后只需换 JSON 文件名重跑，旧版本数据完整保留

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

---

## 参考论文

- Kumpulainen et al. (2021). *Dog behaviour classification with movement sensors placed on the harness and the collar.* Applied Animal Behaviour Science. https://doi.org/10.1016/j.applanim.2021.105393
- Chambers & Yoder (2020). *FilterNet: A Many-to-Many Deep Learning Architecture for Time Series Classification.* Sensors. https://doi.org/10.3390/s20092498
- van Herwijnen et al. (2021). *Deep Learning Classification of Canine Behavior Using a Single Collar-Mounted Accelerometer: Real-World Validation.* Animals. https://doi.org/10.3390/ani11061549
- Dunford et al. (2024). *Predicting cat behaviour using accelerometer data.* Ecology and Evolution. https://doi.org/10.1002/ece3.11368
- Smit et al. (2023). *Behaviour Classification of Extensively Kept Goats and Sheep Using Raw Accelerometer Data.* Sensors. https://doi.org/10.3390/s23052404
