# 狗狗皮肤健康评估

基于项圈IMU抓挠识别结果，评估狗狗皮肤/瘙痒健康状况的完整方案——代码、文档、数据都在这个目录下，跟项目其他通用的IMU数据处理基础设施（`src/`，TF转换、降采样、设备诊断等，给整个项目用，不属于皮肤评估任务本身）分开放。

## 目录结构

```
skin_health/
├── README.md              ← 本文件
├── docs/
│   ├── skin_health.md               ← 主设计文档（评分公式、贝叶斯模型、待验证参数、开放问题）
│   └── skin_health_daily_template.md ← 算法评级 vs 兽医目测 每日对照表格模板
├── code/
│   ├── scratch_burden.py                  ← SBS评分引擎（按PM文档Wardyn V0.4定义实现）
│   ├── gen_synthetic_scratch_scenarios.py ← SBS合成场景验证
│   ├── daily_skin_report.py               ← 接真实推理结果，每天出报告，跟兽医评分对照用
│   ├── bayesian_skin_model.py             ← 分层贝叶斯模型（对兽医打的有序标签建模）
│   ├── validate_bayesian_skin_model.py    ← 上面这个模型的合成数据验证
│   ├── bhm_scratch_count.py               ← 分层贝叶斯模型（直接对抓挠次数建模，不需要兽医标签）
│   └── validate_bhm_scratch_count.py      ← 上面这个模型的合成数据验证
└── data/                   ← 真实数据/报告产出物放这里（不入git，见 data/README.md）
```

## 先看这个

完整设计、公式、已验证/待验证的内容，都在 [`docs/skin_health.md`](docs/skin_health.md)，从头看即可，里面按章节记录了：

- §0-8：SBS评分引擎（按产品Wardyn V0.4定义实现的规则系统），已用合成数据验证过
- §9：分层贝叶斯模型（对兽医有序标签建模），需要兽医标签才能训练
- §10：分层贝叶斯计数模型（直接对抓挠次数建模），**不需要兽医标签就能跑**，适合现在就接真实数据

要给贝叶斯模型选特征时看 [`docs/feature_catalog.md`](docs/feature_catalog.md)——所有讨论过的候选特征（频率/时长/时间分布/聚集/睡眠/相对基线六大类），标了现在实际用没用上、数据类型（计数/连续/比例/二值，选似然分布用得上），以及选特征时的几条实用建议。

## 快速上手

```bash
# 1. SBS机制本身的合成数据验证（不需要任何真实数据，随时能跑）
python skin_health/code/gen_synthetic_scratch_scenarios.py

# 2. 接真实推理结果，每天出报告（需要先用 run_infer_tf.sh 等工具跑出 *_infer.json）
python skin_health/code/daily_skin_report.py \
  --pet_id dog1 \
  --csv_dir data/raw_tf_csv \
  --infer_json_dir infer_result_tf_majority/csv/_infer \
  --device_hz 50 \
  --out_csv skin_health/data/dog1_daily.csv \
  --out_md skin_health/data/dog1_daily.md

# 3. 分层贝叶斯模型的合成数据验证（需要先 pip install pymc arviz，已加进 requirements.txt）
python skin_health/code/validate_bayesian_skin_model.py       # 有序标签版
python skin_health/code/validate_bhm_scratch_count.py          # 抓挠计数版（不需要兽医标签）
```

## 跟项目其他部分的关系

这里的代码依赖项目根目录 `src/` 下的通用推理工具（比如 `daily_skin_report.py` 会导入 `src/infer_csv_scratch.py` 里的 `load_csv`），是单向依赖——`src/` 不依赖这里的任何东西。TF设备转换、降采样算法、设备信号诊断这些通用IMU处理工具留在 `src/` 和仓库根目录（`tf_offline_to_custom.py`、`run_infer_tf.sh`、`resample_training_match.py`、`diagnose_device_signal.py` 等），不属于皮肤评估任务，不搬进来。

松动检测/未佩戴检测（`docs/wear_state_detection.md`、`src/data/wear_state.py`）是另一个独立任务，跟皮肤评估无关，也不在这个目录下。
