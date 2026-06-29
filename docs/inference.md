# 推理

## ML/DL 模型推理（离线文件）

将待测试的 TXT/CSV 文件放到 `data/infer/` 目录，文件格式与数据集A项圈传感器一致：

```
HH:MM:SS.MS,AX,AY,AZ,GX,GY,GZ
16:00:00.000,0.137362,1.149347,8.603002,0.032180,0.012545,0.003272
16:00:00.020,0.140873,1.149698,8.603353,0.032180,0.012545,0.003272
...
```

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

### 常用示例

```bash
# ML 推理，置信度阈值 0.6
python src/infer.py \
  --model_type ml \
  --model_path results/processed_a/50hz/ml_xgb.pkl \
  --processed_dir data/processed_a \
  --hz 50

# 关闭 Unknown 过滤（所有窗口都给出预测）
python src/infer.py \
  --model_type ml \
  --model_path results/processed_a/50hz/ml_xgb.pkl \
  --processed_dir data/processed_a \
  --hz 50 \
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
| `--model_name` | `cnn_lstm` | DL 模型名称 |
| `--processed_dir` | 必填 | 对应数据集的预处理目录 |
| `--hz` | 必填 | 目标采样率（5 / 10 / 25 / 50） |
| `--input_dir` | `data/infer` | 待推理文件目录 |
| `--input_file` | — | 单个文件路径（与 `--input_dir` 二选一） |
| `--output_dir` | `results/infer` | 结果输出目录 |
| `--confidence_threshold` | `0.6` | 置信度阈值，低于此值标记为 `Unknown`（设为 `0` 禁用） |
| `--no_gravity_align` | — | 强制禁用重力轴对齐 |

### 输出

结果保存为 CSV，每行一个预测窗口：

```
file,window_idx,time_start_s,time_end_s,prediction,confidence
26060316.TXT,0,0.0,2.0,Walking,0.9312
26060316.TXT,1,1.0,3.0,Walking,0.8754
26060316.TXT,2,2.0,4.0,Unknown,0.4821
```

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

## 实时 BLE 推理

直接从 HICC_PetCollar 或 WitMotion WT901 设备实时接收 IMU 数据并预测行为，无需录制文件。

### 前置准备

`witmotion_imu` 已作为 git submodule 内嵌在本项目中：

```bash
# 全新 clone（推荐）
git clone --recurse-submodules https://github.com/zhuyetuo/imu_train

# 已经 clone 但未初始化 submodule
git submodule update --init
```

安装 BLE 依赖：

```bash
pip install bleak
```

> 设备同一时间只能连一个应用，运行实时推理时请关闭其他 BLE 连接程序。

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
[2026-06-24 15:14:10]  规则=睡觉  ML=睡觉(92%)
[2026-06-24 15:14:11]  规则=抓挠  ML=活动(78%)
[2026-06-24 15:14:12]  规则=活动  ML=活动(85%)
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
| `--confidence_threshold` | `0.0` | ML 置信度阈值（0=不过滤），低于此值显示 Unknown |
| `--scan` | — | 扫描附近设备后退出 |
