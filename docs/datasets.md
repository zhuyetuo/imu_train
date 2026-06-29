# 数据集与预处理

## 支持的数据集

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

## 数据准备

### 数据集 A（45条狗）

从 [Mendeley](https://data.mendeley.com/datasets/vxhx934tbn/2) 下载，放到 `data/raw/`：

```
data/raw/
├── DogMoveData_csv_format.zip
└── DogInfo.csv
```

```bash
bash setup.sh --dataset a
```

### 数据集 B（42条狗）

从 [Mendeley](https://data.mendeley.com/datasets/mpph6bmn7g/1) 下载 `df_raw.csv`，放到 `data/raw_b/`：

```bash
bash setup.sh --dataset b
```

### 猫咪数据集（Dunford 2024，推荐）

从 [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.q2bvq83sx) 下载 CSV，放到 `data/raw_cat_dunford2024/`：

```
data/raw_cat_dunford2024/
└── Dunford_et_al._Cats_calibrated_data.csv
```

```bash
bash setup.sh --dataset cat_dunford2024
```

### 猫咪数据集（Smit 2024）

从 [FigShare](https://figshare.com/articles/dataset/24848292) 下载，放到 `data/raw_cat_smit2024/cat_smit2024.csv`：

```bash
bash setup.sh --dataset cat_smit2024
```

### 猫咪数据集（Smit 2023，需转换 R 格式）

```bash
# 先将 .RDATA 转为 CSV（需安装 R 或 rpy2）
python src/data/convert_cat_rdata.py --dataset smit2023 \
  --rdata data/raw_cat_smit2023/accel_data.RDATA \
  --annot data/raw_cat_smit2023/anno_data.RDATA \
  --out   data/raw_cat_smit2023/cat_smit2023.csv

bash setup.sh --dataset cat_smit2023
```

### 自采数据集

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

---

## 重力轴对齐

预处理默认启用**重力轴对齐**（每个窗口将加速度均值对齐到 +Z 轴，使模型对项圈安装方向鲁棒）。如需禁用：

```bash
python src/data/preprocess.py --dataset a --output_dir data/processed_a --no_gravity_align
```

> 训练和推理必须使用相同的对齐设置，`setup.sh` 使用默认设置（启用对齐）。

每个数据集独立预处理，互不影响，分别保存到 `data/processed_<tag>/`，每个采样率一个子目录。

---

## 配置文件

| 文件 | 说明 |
|------|------|
| `configs/data.yaml` | 采样率列表、窗口大小、各数据集标签过滤、自采数据集列名、猫咪数据集路径 |
| `configs/ml.yaml` | RF / XGBoost / LightGBM / CatBoost 超参数 |
| `configs/dl.yaml` | 网络结构、epochs、batch_size、early stopping patience |
