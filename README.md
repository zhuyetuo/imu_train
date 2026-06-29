# IMU 犬只 / 猫咪行为识别

基于项圈 IMU（加速度计 + 陀螺仪）数据，对比机器学习与深度学习在不同采样率、不同数据集下的行为分类效果。

---

## 数据集

### 犬只数据集

| 标识 | 数据集 | 下载 | 说明 |
|------|--------|------|------|
| `a` | Movement Sensor Dataset for Dog Behavior Classification | [Mendeley vxhx934tbn](https://data.mendeley.com/datasets/vxhx934tbn/2) | 45条狗，100Hz，6通道，6类行为 |
| `b` | Assistance Dog Activity Dataset | [Mendeley mpph6bmn7g](https://data.mendeley.com/datasets/mpph6bmn7g/1) | 42条狗，100Hz，6通道，4类行为 |
| `custom` | 自采数据集 | — | 格式见下文 |

数据集 A / B 只使用**项圈**传感器数据（加速度计 + 陀螺仪，共 6 通道）。

### 猫咪数据集（仅加速度计，3 通道）

| 标识 | 数据集 | 下载 | 说明 |
|------|--------|------|------|
| `cat_dunford2024` | Dunford et al. 2024 | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83sx) | 9只猫，40Hz，原始时间序列，推荐优先使用 |
| `cat_smit2024` | Smit et al. 2024 | [FigShare 24848292](https://figshare.com/articles/dataset/24848292) | 28只猫，25Hz，8类行为 |
| `cat_smit2023` | Smit et al. 2023 | [FigShare 23605842](https://figshare.com/articles/dataset/23605842) | 12只猫，30Hz，R 格式需转换 |

---

## 项目结构

```
imu_train/
├── data/
│   ├── raw/                     ← 数据集A原始文件（不入 git）
│   ├── raw_b/                   ← 数据集B原始文件（不入 git）
│   ├── raw_custom/              ← 自采数据集（不入 git）
│   ├── raw_cat_dunford2024/     ← 猫咪数据集（不入 git）
│   ├── raw_cat_smit2024/
│   ├── raw_cat_smit2023/
│   ├── infer/                   ← 待推理的无标签 TXT/CSV 文件放这里
│   ├── processed_a/             ← 预处理结果（自动生成）
│   ├── processed_b/
│   ├── processed_custom/
│   └── processed_cat_*/
├── src/
│   ├── data/          ← 数据加载与预处理（ML/DL 共用）
│   ├── ml/            ← 机器学习（独立，不依赖 dl/）
│   ├── dl/            ← 深度学习（独立，不依赖 ml/）
│   └── eval/          ← 结果对比与 SHAP 分析
├── configs/           ← 超参数配置
├── results/           ← 训练结果（自动生成，不入 git）
├── setup.sh           ← 数据预处理脚本
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备与预处理

#### 数据集 A（45条狗）

从 [Mendeley](https://data.mendeley.com/datasets/vxhx934tbn/2) 下载，放到 `data/raw/`：

```
data/raw/
├── DogMoveData_csv_format.zip
└── DogInfo.csv
```

```bash
bash setup.sh --dataset a
```

#### 数据集 B（42条狗）

从 [Mendeley](https://data.mendeley.com/datasets/mpph6bmn7g/1) 下载 `df_raw.csv`，放到 `data/raw_b/`：

```bash
bash setup.sh --dataset b
```

#### 猫咪数据集（Dunford 2024，推荐）

从 [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83sx) 下载 CSV，放到 `data/raw_cat_dunford2024/`：

```
data/raw_cat_dunford2024/
└── Dunford_et_al._Cats_calibrated_data.csv
```

```bash
bash setup.sh --dataset cat_dunford2024
```

#### 猫咪数据集（Smit 2024）

从 [FigShare](https://figshare.com/articles/dataset/24848292) 下载，放到 `data/raw_cat_smit2024/cat_smit2024.csv`：

```bash
bash setup.sh --dataset cat_smit2024
```

#### 猫咪数据集（Smit 2023，需转换 R 格式）

```bash
# 先将 .RDATA 转为 CSV（需安装 R 或 rpy2）
python src/data/convert_cat_rdata.py --dataset smit2023 \
  --rdata data/raw_cat_smit2023/accel_data.RDATA \
  --annot data/raw_cat_smit2023/anno_data.RDATA \
  --out   data/raw_cat_smit2023/cat_smit2023.csv

bash setup.sh --dataset cat_smit2023
```

#### 自采数据集

将数据整理成 CSV，放到 `data/raw_custom/data.csv`，每行一个采样点：

```
dog_id, label, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
dog1,   Walk,  0.12,  -0.03, 9.81,  0.01,  0.02,  -0.01
...
```

编辑 `configs/data.yaml` 匹配实际列名和采样率，然后：

```bash
bash setup.sh --dataset custom
```

每个数据集独立预处理，互不影响，分别保存到 `data/processed_<tag>/`，每个采样率一个子目录。

预处理默认启用**重力轴对齐**（每个窗口将加速度均值对齐到 +Z 轴，使模型对项圈安装方向鲁棒）。如需禁用：

```bash
python src/data/preprocess.py --dataset a --output_dir data/processed_a --no_gravity_align
```

> 训练和推理必须使用相同的对齐设置，`setup.sh` 使用默认设置（启用对齐）。

---

## 训练

通过 `--processed_dir` 指定数据集，每个数据集独立训练，结果互不覆盖。

### 机器学习

支持模型：`rf`（随机森林）、`xgb`（XGBoost）、`lgbm`（LightGBM）、`catboost`（CatBoost）

```bash
python src/ml/train.py --hz 50 --model rf       --processed_dir data/processed_a
python src/ml/train.py --hz 50 --model xgb      --processed_dir data/processed_a
python src/ml/train.py --hz 50 --model lgbm     --processed_dir data/processed_a
python src/ml/train.py --hz 50 --model catboost --processed_dir data/processed_a
```

首次提取特征会自动缓存到 `data/processed_a/50hz/ml_features.npz`，下次直接加载。

### 深度学习

支持模型：

| 模型 | 类型 | 说明 |
|------|------|------|
| `cnn` | many-to-one | 通用 1D CNN（MaxPool=2，每层 Dropout） |
| `collar_cnn` | many-to-one | 复现 [Animals 2021](https://doi.org/10.3390/ani11061549) 架构（64→128→256，MaxPool=4，Dropout 只在 FC 前） |
| `cnn_lstm` | many-to-one | CNN 提取局部特征 + LSTM 建模时序依赖 |
| `transformer` | many-to-one | 基于自注意力机制的时序分类器 |
| `filternet` | many-to-one | 复现 [FilterNet (Sensors 2020)](https://doi.org/10.3390/s20092498) encoder，stride 下采样 + LSTM + GAP |
| `filternet_m2m` | **many-to-many** | FilterNet 原版逐帧预测，插值上采样回原始 T，多数投票得窗口标签 |

```bash
python src/dl/train.py --hz 50 --model cnn           --processed_dir data/processed_a
python src/dl/train.py --hz 50 --model collar_cnn    --processed_dir data/processed_a
python src/dl/train.py --hz 50 --model cnn_lstm      --processed_dir data/processed_a
python src/dl/train.py --hz 50 --model transformer   --processed_dir data/processed_a
python src/dl/train.py --hz 50 --model filternet     --processed_dir data/processed_a
python src/dl/train.py --hz 50 --model filternet_m2m --processed_dir data/processed_a
```

> **注意**：`filternet_m2m` 需要 npz 文件包含 `y_seq` 字段。如果之前用旧版本预处理过数据，需重新运行 `bash setup.sh --dataset <tag>`。

GPU 可用时自动使用，否则回退到 CPU。

### 切换数据集

把 `--processed_dir` 换成对应目录：

```bash
python src/ml/train.py --hz 50 --model xgb --processed_dir data/processed_b
python src/ml/train.py --hz 10 --model xgb --processed_dir data/processed_cat_dunford2024
```

结果自动保存到对应的 `results/<dataset_tag>/<hz>hz/`。

---

## 批量实验（并行）

推荐使用并行启动器，比逐个运行快 5-10 倍：

```bash
# ML 8进程并行，DL 4进程并行（5090 显存充足）
python run_experiments.py --ml_workers 8 --dl_workers 4

# 只跑指定数据集和采样率
python run_experiments.py --datasets processed_a processed_b --hz 25 50

# 只跑 ML 或只跑 DL
python run_experiments.py --skip_dl --ml_workers 8
python run_experiments.py --skip_ml --dl_workers 4
```

启动器会自动：
- 按 `cpu_count / ml_workers` 分配每个 ML 任务的核数，避免多进程竞争
- 跳过 `data/` 目录不存在的数据集（如尚未准备自采数据）
- 每 15 秒打印一次进行中的任务及耗时

每次运行约 20 分钟（2 个数据集 × 4 采样率 × 10 模型，RTX 5090）。

### 手动逐个运行

```bash
for ds in processed_a processed_b; do
  for hz in 5 10 25 50; do
    for model in rf xgb lgbm catboost; do
      python src/ml/train.py --hz $hz --model $model --processed_dir data/$ds
    done
    for model in cnn collar_cnn cnn_lstm transformer filternet filternet_m2m; do
      python src/dl/train.py --hz $hz --model $model --processed_dir data/$ds
    done
  done
done
```

---

## 实验结果参考

以下为在数据集 A（45条狗）和数据集 B（42条狗）上的实测结果：

### 数据集 A — Accuracy

| 模型 | 5Hz | 10Hz | 25Hz | 50Hz |
|------|-----|------|------|------|
| ML/rf | 0.779 | 0.805 | **0.812** | 0.800 |
| ML/xgb | 0.772 | 0.802 | 0.806 | 0.806 |
| ML/lgbm | 0.768 | 0.797 | 0.802 | 0.805 |
| ML/catboost | 0.768 | 0.796 | 0.799 | 0.797 |
| DL/cnn | 0.768 | 0.788 | 0.775 | 0.795 |
| DL/collar_cnn | 0.725 | 0.762 | 0.779 | 0.799 |
| DL/cnn_lstm | 0.762 | 0.774 | 0.782 | 0.775 |
| DL/transformer | 0.723 | 0.761 | 0.748 | 0.735 |
| DL/filternet | 0.740 | 0.778 | 0.803 | 0.784 |
| DL/filternet_m2m | 0.730 | 0.781 | 0.776 | 0.803 |

**结论**：ML 模型整体略优；最佳为 `RF 25Hz`（Acc 0.812）。DL 中 `filternet 25Hz` F1 最高（0.754），`filternet_m2m 50Hz` 与 ML 持平。

### 数据集 B — Accuracy

| 模型 | 5Hz | 10Hz | 25Hz | 50Hz |
|------|-----|------|------|------|
| ML/rf | 0.653 | 0.680 | 0.691 | **0.692** |
| ML/lgbm | 0.661 | 0.674 | 0.691 | 0.685 |
| DL/cnn_lstm | 0.631 | 0.646 | 0.643 | 0.644 |
| DL/transformer | 0.619 | 0.609 | 0.637 | 0.624 |

**结论**：数据集 B 整体比 A 低约 12%；ML 明显优于 DL，说明 DL 需要更多数据才能发挥优势。

> 猫咪数据集（Dunford 2024，9只猫）样本量过小，测试集仅 1 只猫，结果不具统计意义，需更多数据。

---

## 结果对比

```bash
# 查看所有数据集的结果
python src/eval/compare.py

# 只看某个数据集
python src/eval/compare.py --dataset processed_a

# 同时展示每类 Precision / Recall / F1
python src/eval/compare.py --per_class

# 只看 XGB 50Hz 的每类指标
python src/eval/compare.py --per_class --hz 50 --model xgb
```

`--per_class` 输出格式：

```
  [ML/xgb  50Hz]
  类别              Precision    Recall        F1
  ─────────────────────────────────────────────────
  Lying chest          0.6700    0.8100    0.7300
  Sitting              0.6300    0.4600    0.5300
  Sniffing             0.9700    0.9900    0.9800
  Standing             0.4200    0.2400    0.3000
  Trotting             0.9900    0.9400    0.9600
  Walking              0.8700    0.9500    0.9100
```

---

## SHAP 特征重要性分析

训练完 ML 模型后，可分析哪些传感器特征对分类贡献最大：

```bash
python src/eval/explain.py --hz 50 --model xgb --processed_dir data/processed_a
```

输出图表保存到 `results/processed_a/50hz/shap_xgb/`：

| 文件 | 内容 |
|------|------|
| `summary_bar.png` | Top-20 全局特征重要性 |
| `summary_dot.png` | 全局 SHAP 值分布（蜂群图） |
| `class_<行为>.png` | 各行为类别的 SHAP 值分布 |

可选参数：

```bash
python src/eval/explain.py \
  --hz 50 \
  --model xgb \
  --processed_dir data/processed_a \
  --max_samples 1000    # 用于分析的样本上限，越多越慢
```

---

## 推理（无标签数据）

将待测试的 TXT/CSV 文件放到 `data/infer/` 目录，文件格式与数据集A项圈传感器一致：

```
HH:MM:SS.MS,AX,AY,AZ,GX,GY,GZ
16:00:00.000,0.137362,1.149347,8.603002,0.032180,0.012545,0.003272
16:00:00.020,0.140873,1.149698,8.603353,0.032180,0.012545,0.003272
...
```

> 采样率固定为 **50Hz**（每行间隔 20ms）。

### 推荐配置（家养犬，项圈方向固定）

经测试，**RF 25Hz 不对齐**在实际推理中表现最符合预期（Lying chest 识别更准确）：

```bash
# 步骤 1：预处理（不对齐）
bash setup.sh --dataset a --no_gravity_align

# 步骤 2：训练
python run_experiments.py --datasets processed_a --skip_dl

# 步骤 3：推理
python src/infer.py \
  --model_type ml \
  --model_path results/processed_a/25hz/ml_rf.pkl \
  --processed_dir data/processed_a \
  --hz 25 \
  --confidence_threshold 0
```

> 重力对齐适合项圈安装方向差异较大的场景（如多设备、多狗）；方向固定时关闭反而更好。

---

### 常用示例

```bash
# ML 推理，标准（重力对齐跟训练一致，置信度阈值 0.6）
python src/infer.py \
  --model_type ml \
  --model_path results/processed_a/50hz/ml_xgb.pkl \
  --processed_dir data/processed_a \
  --hz 50

# ML 推理，关闭 Unknown 过滤（所有窗口都给出预测）
python src/infer.py \
  --model_type ml \
  --model_path results/processed_a/50hz/ml_xgb.pkl \
  --processed_dir data/processed_a \
  --hz 50 \
  --confidence_threshold 0

# ML 推理，关闭重力对齐 + 关闭 Unknown 过滤（训练时也需 --no_gravity_align）
python src/infer.py \
  --model_type ml \
  --model_path results/processed_a/50hz/ml_xgb.pkl \
  --processed_dir data/processed_a \
  --hz 50 \
  --no_gravity_align \
  --confidence_threshold 0

# DL 推理
python src/infer.py \
  --model_type dl \
  --model_name cnn_lstm \
  --model_path results/processed_a/50hz/dl_cnn_lstm_best.pt \
  --processed_dir data/processed_a \
  --hz 50

# 推理单个文件
python src/infer.py \
  --model_type ml \
  --model_path results/processed_a/50hz/ml_xgb.pkl \
  --processed_dir data/processed_a \
  --hz 50 \
  --input_file data/infer/26060316.TXT
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_type` | 必填 | `ml` 或 `dl` |
| `--model_path` | 必填 | 训练好的模型文件（ML: `.pkl`，DL: `.pt`） |
| `--model_name` | `cnn_lstm` | DL 模型名称（`cnn` / `collar_cnn` / `cnn_lstm` / `transformer` / `filternet` / `filternet_m2m`） |
| `--processed_dir` | 必填 | 对应数据集的预处理目录（用于读取类别名和窗口参数） |
| `--hz` | 必填 | 目标采样率（5 / 10 / 25 / 50） |
| `--input_dir` | `data/infer` | 待推理文件目录 |
| `--input_file` | — | 单个文件路径（与 `--input_dir` 二选一） |
| `--output_dir` | `results/infer` | 结果输出目录 |
| `--confidence_threshold` | `0.6` | 置信度阈值，低于此值标记为 `Unknown`（设为 `0` 禁用） |
| `--gravity_align` | 自动 | 强制启用重力轴对齐 |
| `--no_gravity_align` | — | 强制禁用重力轴对齐 |

#### 重力轴对齐

推理默认跟随训练时的设置（从预处理元数据自动读取），无需手动指定。如果训练时启用了对齐但推理时强制关闭（或反之），会打印警告：

```
[infer] ⚠️  警告：训练时重力对齐=是，当前设置=否，可能影响精度
```

若项圈在不同设备上方向不同（如传感器朝向各异），建议训练和推理都启用重力对齐（默认行为）。

#### 置信度与 Unknown 标签

模型对每个窗口输出各类别的概率，最高概率低于 `--confidence_threshold` 时标记为 `Unknown`，用于检测训练类别外的行为（如抓挠、打滚等）。

```bash
# 提高阈值，更严格地过滤低置信度预测
python src/infer.py ... --confidence_threshold 0.8

# 禁用过滤，所有窗口都给出预测
python src/infer.py ... --confidence_threshold 0
```

### 输出

结果保存为 CSV，每行一个预测窗口：

```
file,window_idx,time_start_s,time_end_s,prediction,confidence
26060316.TXT,0,0.0,2.0,Walking,0.9312
26060316.TXT,1,1.0,3.0,Walking,0.8754
26060316.TXT,2,2.0,4.0,Unknown,0.4821
...
```

同时打印每个文件的行为分布摘要：

```
26060316.TXT: 312 窗口  [Walking:45%  Standing:23%  Trotting:18%  Unknown:14%]
```

---

## 实时 BLE 推理

直接从 HICC_PetCollar 或 WitMotion WT901 设备实时接收 IMU 数据并预测行为，无需录制文件。

### 前置准备

`witmotion_imu` 已作为 git submodule 内嵌在本项目中，clone 时一并初始化：

```bash
# 全新 clone（推荐）
git clone --recurse-submodules https://github.com/zhuyetuo/imu_train

# 已经 clone 但未初始化 submodule
git submodule update --init
```

然后安装 BLE 依赖：

```bash
pip install bleak
```

> 注意：设备同一时间只能连一个应用，运行实时推理时请关闭其他 BLE 连接程序。

### 扫描附近设备

```bash
python src/infer_rule_live.py --scan
```

### HICC_PetCollar（默认地址 EA:CB:3E:CF:00:1B，25Hz）

```bash
# 规则算法
python src/infer_rule_live.py --device hicc

# ML 模型
python src/infer_rule_live.py --device hicc --algo ml \
  --model results/processed_a/25hz/ml_rf.pkl

# 规则 + ML 并排对比（推荐）
python src/infer_rule_live.py --device hicc --algo rule ml \
  --model results/processed_a/25hz/ml_rf.pkl

# 指定地址
python src/infer_rule_live.py --device hicc --address EA:CB:3E:CF:00:1B
```

### WitMotion WT901SDCL-BT50（自动扫描，采样率可调）

```bash
# 规则算法，自动扫描名含 WTSDCL 的设备（默认 50Hz）
python src/infer_rule_live.py --device wit

# 设备调成 25Hz 时
python src/infer_rule_live.py --device wit --hz 25

# 规则 + ML 并排对比
python src/infer_rule_live.py --device wit --hz 50 --algo rule ml \
  --model results/processed_a/50hz/ml_rf.pkl

# 指定 MAC 地址
python src/infer_rule_live.py --device wit --address AA:BB:CC:DD:EE:FF
```

### 输出格式

每隔 `--stride_s` 秒输出一行（颜色区分行为类别）：

```
[2026-06-24 15:14:10]  规则=睡觉  ML=睡觉
[2026-06-24 15:14:11]  规则=抓挠  ML=活动
[2026-06-24 15:14:12]  规则=活动  ML=活动
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--device` | `hicc` | 设备类型：`hicc` 或 `wit` |
| `--address` | — | BLE MAC 地址（不填自动扫描） |
| `--name` | `WTSDCL` | WitMotion 设备名关键字 |
| `--hz` | hicc=25, wit=50 | 采样率，与设备设置一致 |
| `--window_s` | `2.0` | 判断窗口长度（秒） |
| `--stride_s` | `1.0` | 判断间隔（秒） |
| `--algo` | `rule` | 算法：`rule`、`ml`，可同时指定多个 |
| `--model` | — | ML 模型路径（`.pkl`），`--algo ml` 时必填 |
| `--scan` | — | 扫描附近设备后退出 |

---

## 规则推理（离线文件）

无需训练模型，直接用信号特征阈值判断行为（睡觉 / 活动 / 抓挠）：

```bash
# 本地文件
python src/infer_rule.py --input_file data/infer/rec.csv

# URL（直接从设备下载）
python src/infer_rule.py --input_url "http://192.168.2.140:8182/rec_wit.csv"

# 打印每窗口特征值，用于调整阈值
python src/infer_rule.py --input_file data/infer/rec.csv --show_features

# 调整阈值
python src/infer_rule.py --input_file data/infer/rec.csv \
  --sleep_std 0.05 --scratch_power 0.15
```

输出保存为 `results/infer/<文件名>_rule.csv`（每窗口）和 `_rule_segments.csv`（合并连续相同行为段，含绝对时间戳）。

---

## 配置说明

| 文件 | 说明 |
|------|------|
| `configs/data.yaml` | 采样率列表、窗口大小、各数据集标签过滤、自采数据集列名、猫咪数据集路径 |
| `configs/ml.yaml` | RF / XGBoost / LightGBM / CatBoost 超参数 |
| `configs/dl.yaml` | 网络结构、epochs、batch_size、early stopping patience |

---

## 视觉行为识别模块

基于视频/图片的狗行为识别，支持三种方式：零样本（CLIP）、大模型API（豆包）、关键点分类器（YOLO26 + MLP）。

### 项目结构补充

```
imu_train/
├── src/vision/
│   ├── clip_infer.py          ← CLIP 零样本推理（无需训练）
│   ├── llm_infer.py           ← 火山引擎豆包多模态推理
│   ├── label_with_llm.py      ← 批量用 LLM 标注数据集行为标签
│   ├── relabel_for_yolo.py    ← 将行为标签写入 YOLO label 文件
│   ├── build_behavior_dataset.py ← 生成关键点+行为标签 npz
│   ├── quick_validate.py      ← 可视化 YOLO 关键点检测效果
│   ├── train.py               ← MLP/LSTM 行为分类器训练
│   ├── dataset.py             ← 数据集加载
│   └── models/
│       ├── mlp.py             ← PoseMLP
│       └── lstm_classifier.py ← PoseLSTM
├── configs/vision.yaml        ← 视觉模块配置
├── train_dog_pose.py          ← YOLO26 训练脚本（含内存自动清理）
└── train_dog_pose.sh          ← 训练启动脚本（自动选 batch，清理共享内存）
```

---

### 方式一：CLIP 零样本（最快，无需训练，无需 API Key）

预期准确率约 60-75%，适合快速验证。

```bash
pip install transformers torch opencv-python pillow

# 单张图片
python src/vision/clip_infer.py --image 你的图片.jpg

# 视频
python src/vision/clip_infer.py --video 你的视频.mp4 --fps 2 --no_video

# 批量目录
python src/vision/clip_infer.py --video data/raw_vision/videos/
```

模型首次运行自动下载并缓存到 `models/clip/`，之后离线使用。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--image` | — | 图片文件或目录 |
| `--video` | — | 视频文件或目录 |
| `--fps` | 5 | 视频采样帧率 |
| `--no_video` | — | 只输出 CSV，不生成标注视频 |
| `--model` | `clip-vit-base-patch32` | 也可用 `clip-vit-large-patch14`（更准） |
| `--model_dir` | `models/clip` | 本地模型缓存目录 |

结果保存到 `results/vision/clip/`。

---

### 方式二：大模型 API（最准，需要 API Key）

使用火山引擎豆包多模态模型，支持自然语言描述行为细节。

```bash
pip install openai

export ARK_API_KEY="your_key"   # 从 https://console.volcengine.com/ark 获取

# 单张图片
python src/vision/llm_infer.py --image 你的图片.jpg

# 视频（默认 1fps，省钱）
python src/vision/llm_infer.py --video 你的视频.mp4 --fps 1
```

输出示例：
```
行为: Lying  (置信度 95%)
描述: The dog is lying down on the ground, resting with its body flat and relaxed.
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | `doubao-seed-1-6-flash-250828` | 豆包模型名称 |
| `--fps` | 1 | 视频采样帧率（API 有费用，不宜过高） |

结果保存到 `results/vision/llm/`。

---

### 方式三：YOLO26 + 行为分类（训练后最准）

#### 步骤 1：训练 YOLO26 关键点检测

```bash
# 自动下载 dog-pose 数据集（24 关键点，6773 训练图）并训练
bash train_dog_pose.sh
# 或指定 batch size
bash train_dog_pose.sh --batch 256
```

训练完成后权重保存至 `runs/pose/train/weights/best.pt`。

指标参考（YOLO26n-pose，100 epoch，RTX 5090）：
- Box mAP50: **0.988**
- Pose mAP50: **0.908**

#### 步骤 2：用 LLM 给数据集打行为标签

```bash
export ARK_API_KEY="your_key"

python src/vision/label_with_llm.py --split val   --workers 8
python src/vision/label_with_llm.py --split train --workers 8 --resume
```

约 50 分钟标注全部 8476 张图片，支持断点续跑（`--resume`）。

标注分布参考：
```
Standing  39%   Sitting  26%   Lying  22%
Walking    7%   Trotting  5%   Sniffing  2%
```

#### 步骤 3：生成带行为类别的 YOLO 数据集并重新训练

```bash
# 将行为标签写入 YOLO label 文件，生成新数据集
python src/vision/relabel_for_yolo.py

# 从已训练的关键点模型继续训练（加入行为分类头）
bash train_dog_pose.sh \
    --data datasets/dog-pose-behavior/data.yaml \
    --model runs/pose/train/weights/best.pt \
    --batch 256
```

训练后 YOLO 单次推理同时输出：
- **24 个关键点**（body pose）
- **行为类别**（Lying / Sitting / Standing / Walking / Trotting / Sniffing）

#### 验证关键点效果

```bash
python src/vision/quick_validate.py \
    --video 你的视频.mp4 \
    --model runs/pose/train/weights/best.pt \
    --conf 0.3
```

输出标注骨骼点的视频 + 检测率统计。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--fps` | 10 | 输出帧率（同时控制推理密度，降低可加快速度） |
| `--conf` | 0.5 | 检测置信度阈值，降低可提高检测率 |
| `--imgsz` | 1280 | 推理分辨率，4K 视频建议 1280 或 1920 |
| `--max_dogs` | 1 | 每帧最多保留几只狗（默认 1，防止误检多框；0=不限制） |
| `--no_skeleton` | — | 只画关键点，不绘制骨骼连线 |

推理使用多线程加速（帧读取线程 + GPU 推理 + 视频写入线程并行），并启用 FP16 和 BN 融合。

#### 进一步提速：TensorRT 导出

使用 TensorRT 可比 PyTorch FP16 再提速 2-3 倍：

```bash
# 导出（只需一次，耗时约 2-5 分钟）
yolo export model=runs/pose/train/weights/best.pt format=engine half=True imgsz=1280

# 使用 TensorRT 引擎推理（与 PyTorch 用法相同，自动识别 .engine 后缀）
python src/vision/quick_validate.py \
    --video 你的视频.mp4 \
    --model runs/pose/train/weights/best.engine \
    --conf 0.3 --fps 30
```

> 注意：`.engine` 文件绑定当前机器的 GPU 型号和 TensorRT 版本，换机器需重新导出。导出时的 `imgsz` 要与推理时一致。

---

## 参考论文

- Kumpulainen et al. (2021). *Dog behaviour classification with movement sensors placed on the harness and the collar.* Applied Animal Behaviour Science. https://doi.org/10.1016/j.applanim.2021.105393
- Chambers & Yoder (2020). *FilterNet: A Many-to-Many Deep Learning Architecture for Time Series Classification.* Sensors. https://doi.org/10.3390/s20092498
- van Herwijnen et al. (2021). *Deep Learning Classification of Canine Behavior Using a Single Collar-Mounted Accelerometer.* Animals. https://doi.org/10.3390/ani11061549
- Dunford et al. (2024). *Predicting cat behaviour using accelerometer data.* Ecology and Evolution. https://doi.org/10.1002/ece3.11368
- Smit et al. (2023). *Behaviour Classification of Extensively Kept Goats and Sheep Using Raw Accelerometer Data.* Sensors. https://doi.org/10.3390/s23052404
