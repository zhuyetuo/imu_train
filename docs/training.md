# 训练与实验结果

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

| 模型 | 类型 | 说明 |
|------|------|------|
| `cnn` | many-to-one | 通用 1D CNN（MaxPool=2，每层 Dropout） |
| `collar_cnn` | many-to-one | 复现 [Animals 2021 — Real-World Validation](https://doi.org/10.3390/ani11061549) 架构（64→128→256，MaxPool=4，Dropout 只在 FC 前） |
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

```bash
python src/eval/explain.py \
  --hz 50 \
  --model xgb \
  --processed_dir data/processed_a \
  --max_samples 1000    # 用于分析的样本上限，越多越慢
```
