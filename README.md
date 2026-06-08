# IMU 犬只行为识别

基于项圈加速度计 + 陀螺仪数据，对比机器学习与深度学习在不同采样率、不同数据集下的犬只行为分类效果。

---

## 数据集

| 标识 | 数据集 | 下载 | 说明 |
|------|--------|------|------|
| `a` | Movement Sensor Dataset for Dog Behavior Classification | [Mendeley vxhx934tbn](https://data.mendeley.com/datasets/vxhx934tbn/2) | 45条狗，100Hz，6类行为 |
| `b` | Assistance Dog Activity Dataset | [Mendeley mpph6bmn7g](https://data.mendeley.com/datasets/mpph6bmn7g/1) | 42条狗，100Hz，5类行为 |
| `custom` | 自采数据集 | — | 格式见下文 |

本项目**只使用项圈**传感器数据（加速度计 + 陀螺仪，共6通道）。

---

## 项目结构

```
imu_train/
├── data/
│   ├── raw/               ← 数据集A原始文件（不入 git）
│   ├── raw_b/             ← 数据集B原始文件（不入 git）
│   ├── raw_custom/        ← 自采数据集（不入 git）
│   ├── processed_a/       ← 数据集A预处理结果（自动生成）
│   ├── processed_b/       ← 数据集B预处理结果（自动生成）
│   └── processed_custom/  ← 自采数据集预处理结果（自动生成）
├── src/
│   ├── data/              ← 数据加载与预处理（ML/DL 共用）
│   ├── ml/                ← 机器学习（独立，不依赖 dl/）
│   ├── dl/                ← 深度学习（独立，不依赖 ml/）
│   └── eval/              ← 结果对比与 SHAP 分析
├── configs/               ← 超参数配置
├── results/               ← 训练结果（自动生成，不入 git）
├── setup.sh               ← 数据预处理脚本
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

#### 数据集 A

从 [Mendeley](https://data.mendeley.com/datasets/vxhx934tbn/2) 下载以下文件，放到 `data/raw/`：

```
data/raw/
├── DogMoveData_csv_format.zip
└── DogInfo.csv
```

```bash
bash setup.sh --dataset a
```

#### 数据集 B

从 [Mendeley](https://data.mendeley.com/datasets/mpph6bmn7g/1) 下载 `df_raw.csv`，放到 `data/raw_b/`：

```
data/raw_b/
└── df_raw.csv
```

```bash
bash setup.sh --dataset b
```

#### 自采数据集

将数据整理成 CSV 文件，放到 `data/raw_custom/data.csv`。

**默认列名**（可在 `configs/data.yaml` 的 `custom` 节修改）：

```
dog_id, label, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
dog1,   Walk,  0.12,  -0.03, 9.81,  0.01,  0.02,  -0.01
dog1,   Walk,  0.11,  -0.02, 9.80,  0.01,  0.01,  -0.01
...
```

- `dog_id`：每条狗的唯一标识，用于 leave-some-dogs-out 划分
- `label`：行为标签
- 多条狗的数据放在同一个文件里，按 `dog_id` 自动拆分

编辑 `configs/data.yaml` 匹配你的列名和采样率：

```yaml
custom:
  csv_path: data/raw_custom/data.csv
  source_hz: 100        # 实际采样率
  dog_id_col: dog_id    # 改成你的列名
  label_col: label
  sensor_cols: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
  keep_labels: []       # 留空保留所有标签
```

```bash
bash setup.sh --dataset custom
```

每个数据集的预处理结果独立保存，互不影响：

```
data/processed_a/    data/processed_b/    data/processed_custom/
├── 5hz/             ├── 5hz/             ├── 5hz/
├── 10hz/            ├── 10hz/            ├── 10hz/
├── 25hz/            ├── 25hz/            ├── 25hz/
└── 50hz/            └── 50hz/            └── 50hz/
    train.npz            train.npz            train.npz
    val.npz              val.npz              val.npz
    test.npz             test.npz             test.npz
```

---

## 训练

每个数据集独立训练，通过 `--processed_dir` 指定。

### 机器学习

支持模型：`rf`（随机森林）、`xgb`（XGBoost）

```bash
python src/ml/train.py --hz 50 --model rf  --processed_dir data/processed_a
python src/ml/train.py --hz 50 --model xgb --processed_dir data/processed_a
```

首次提取特征会自动缓存到 `data/processed_a/50hz/ml_features.npz`，下次直接加载。

### 深度学习

支持模型：`cnn`、`cnn_lstm`、`transformer`

```bash
python src/dl/train.py --hz 50 --model cnn         --processed_dir data/processed_a
python src/dl/train.py --hz 50 --model cnn_lstm    --processed_dir data/processed_a
python src/dl/train.py --hz 50 --model transformer --processed_dir data/processed_a
```

GPU 可用时自动使用，否则回退到 CPU。

### 切换数据集

把 `--processed_dir` 换成对应目录即可：

```bash
python src/ml/train.py --hz 50 --model xgb --processed_dir data/processed_b
python src/ml/train.py --hz 50 --model xgb --processed_dir data/processed_custom
```

结果自动保存到 `results/processed_a/`、`results/processed_b/`、`results/processed_custom/`。

---

## 批量实验

```bash
for ds in processed_a processed_b processed_custom; do
  for hz in 5 10 25 50; do
    for model in rf xgb; do
      python src/ml/train.py --hz $hz --model $model --processed_dir data/$ds
    done
    for model in cnn cnn_lstm transformer; do
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
...

======================================================================
  数据集: processed_b
======================================================================
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
| `class_Walking.png` | Walking 类别的 SHAP 值分布 |
| `class_Standing.png` | Standing 类别的 SHAP 值分布 |
| ... | 每个行为类别一张 |

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
| `configs/data.yaml` | 采样率列表、窗口大小、各数据集标签过滤、自采数据集列名 |
| `configs/ml.yaml` | RF / XGBoost 超参数 |
| `configs/dl.yaml` | 网络结构、epochs、batch_size、early stopping |

---

## 参考论文

- Kumpulainen et al. (2021). *Dog behaviour classification with movement sensors placed on the harness and the collar.* Applied Animal Behaviour Science. https://doi.org/10.1016/j.applanim.2021.105393
- Chambers & Yoder (2020). *FilterNet: A Many-to-Many Deep Learning Architecture for Time Series Classification.* Sensors. https://doi.org/10.3390/s20092498
- Transformer-based Dog Behavior Classification with Motion Sensors. IEEE Sensors Journal (2024). https://doi.org/10.1109/JSEN.2024.3455383
