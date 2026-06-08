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
  --input data/raw_cat_smit2023/smit2023.RDATA \
  --output data/raw_cat_smit2023/cat_smit2023.csv

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

## 批量实验

```bash
for ds in processed_a processed_b processed_custom; do
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

## 结果对比

```bash
# 查看所有数据集的结果
python src/eval/compare.py

# 只看某个数据集
python src/eval/compare.py --dataset processed_a
```

输出示例：

```
======================================================================
  数据集: processed_a
======================================================================

  (Accuracy)
模型                    5Hz      10Hz      25Hz      50Hz
----------------------------------------------------------
ML/rf                   —         —         —    0.8058
ML/xgb                  —         —         —    0.7366
ML/lgbm                 —         —         —    0.8031
ML/catboost             —         —         —    0.8012
DL/cnn                  —         —         —    0.8123
DL/filternet_m2m        —         —         —    0.7921
...
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

## 配置说明

| 文件 | 说明 |
|------|------|
| `configs/data.yaml` | 采样率列表、窗口大小、各数据集标签过滤、自采数据集列名、猫咪数据集路径 |
| `configs/ml.yaml` | RF / XGBoost / LightGBM / CatBoost 超参数 |
| `configs/dl.yaml` | 网络结构、epochs、batch_size、early stopping patience |

---

## 参考论文

- Kumpulainen et al. (2021). *Dog behaviour classification with movement sensors placed on the harness and the collar.* Applied Animal Behaviour Science. https://doi.org/10.1016/j.applanim.2021.105393
- Chambers & Yoder (2020). *FilterNet: A Many-to-Many Deep Learning Architecture for Time Series Classification.* Sensors. https://doi.org/10.3390/s20092498
- van Herwijnen et al. (2021). *Deep Learning Classification of Canine Behavior Using a Single Collar-Mounted Accelerometer.* Animals. https://doi.org/10.3390/ani11061549
- Dunford et al. (2024). *Predicting cat behaviour using accelerometer data.* Ecology and Evolution. https://doi.org/10.1002/ece3.11368
- Smit et al. (2023). *Behaviour Classification of Extensively Kept Goats and Sheep Using Raw Accelerometer Data.* Sensors. https://doi.org/10.3390/s23052404
