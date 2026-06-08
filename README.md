# IMU 犬只行为识别

基于项圈加速度计 + 陀螺仪数据，对比机器学习与深度学习在不同采样率下的犬只行为分类效果。

## 数据集

[Movement Sensor Dataset for Dog Behavior Classification](https://data.mendeley.com/datasets/vxhx934tbn/2)

- 45 条狗，100Hz，3轴加速度计 + 3轴陀螺仪（项圈 + 胸背带）
- 本项目**只使用项圈**数据：`ANeck_x/y/z`、`GNeck_x/y/z`
- 行为标签（20类）：Standing、Walking、Trotting、Galloping、Sitting、Lying chest、Sniffing 等

## 项目结构

```
imu_train/
├── data/
│   ├── raw/              ← 原始数据放这里（不入 git）
│   └── processed/        ← 预处理结果（自动生成，不入 git）
├── src/
│   ├── data/             ← 数据加载与预处理（ML/DL 共用）
│   ├── ml/               ← 机器学习（独立，不依赖 dl/）
│   ├── dl/               ← 深度学习（独立，不依赖 ml/）
│   └── eval/             ← 结果对比
├── configs/              ← 超参数配置
├── results/              ← 训练结果（自动生成，不入 git）
├── setup.sh              ← 一键数据整理脚本
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

从 [Mendeley](https://data.mendeley.com/datasets/vxhx934tbn/2) 下载以下文件，放到 `data/raw/`：

```
data/raw/
├── DogMoveData_csv_format.zip   (421 MB)
└── DogInfo.csv                  (1.38 KB)
```

### 3. 一键整理数据

```bash
bash setup.sh
```

脚本会自动完成：解压 → 按狗ID划分 train/val/test（36/4/5）→ 降采样到 5/10/25/50Hz → 滑动窗口切分 → 保存 `.npz`

完成后 `data/processed/` 结构如下：
```
data/processed/
├── 5hz/   train.npz  val.npz  test.npz
├── 10hz/  train.npz  val.npz  test.npz
├── 25hz/  train.npz  val.npz  test.npz
└── 50hz/  train.npz  val.npz  test.npz
```

---

## 训练

### 机器学习

支持模型：`rf`（随机森林）、`svm`、`xgb`（XGBoost）

```bash
python src/ml/train.py --hz 50 --model rf
python src/ml/train.py --hz 50 --model svm
python src/ml/train.py --hz 50 --model xgb
```

### 深度学习

支持模型：`cnn`、`cnn_lstm`、`transformer`

```bash
python src/dl/train.py --hz 50 --model cnn
python src/dl/train.py --hz 50 --model cnn_lstm
python src/dl/train.py --hz 50 --model transformer
```

GPU 可用时自动使用，否则回退到 CPU。

---

## 多采样率批量实验

跑完所有 hz × 模型组合后，输出汇总对比表：

```bash
# 批量跑 ML（示例）
for hz in 5 10 25 50; do
  for model in rf svm xgb; do
    python src/ml/train.py --hz $hz --model $model
  done
done

# 批量跑 DL
for hz in 5 10 25 50; do
  for model in cnn cnn_lstm transformer; do
    python src/dl/train.py --hz $hz --model $model
  done
done

# 汇总对比
python src/eval/compare.py
```

输出示例：

```
======================================================================
  IMU 犬只行为识别 — ML vs DL × 采样率 对比结果 (Test Accuracy)
======================================================================
模型                    5Hz       10Hz      25Hz      50Hz
--------------------------------------------------------------
ML/rf                  0.8234    0.8701    0.9123    0.9341
ML/svm                 0.7891    0.8412    0.8934    0.9102
ML/xgb                 0.8412    0.8923    0.9234    0.9456
DL/cnn                 0.8601    0.9012    0.9312    0.9523
DL/cnn_lstm            0.8734    0.9123    0.9434    0.9612
DL/transformer         0.8689    0.9089    0.9401    0.9589
======================================================================
```

结果文件保存在 `results/{hz}hz/` 下，格式为 JSON。

---

## 配置说明

| 文件 | 说明 |
|------|------|
| `configs/data.yaml` | 窗口大小、采样率列表、train/val/test 比例 |
| `configs/ml.yaml` | RF/SVM/XGBoost 超参数、手工特征开关 |
| `configs/dl.yaml` | 网络结构、epochs、batch_size、early stopping |

---

## 参考论文

- Kumpulainen et al. (2021). *Dog behaviour classification with movement sensors placed on the harness and the collar.* Applied Animal Behaviour Science. https://doi.org/10.1016/j.applanim.2021.105393
- Chambers & Yoder (2020). *FilterNet: A Many-to-Many Deep Learning Architecture for Time Series Classification.* Sensors. https://doi.org/10.3390/s20092498
- Transformer-based Dog Behavior Classification with Motion Sensors. IEEE Sensors Journal (2024). https://doi.org/10.1109/JSEN.2024.3455383
