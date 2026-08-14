# 模型A（行为严重度分类器）训练报告——合成数据

训练表：2857行，86个场景狗

交叉验证(8折，按场景狗分组)宏F1：均值=0.886 标准差=0.030

## 特征重要性(permutation importance，全量数据)

| 特征 | 重要性均值 | 标准差 |
|---|---|---|
| sleep_disruption_count | 0.2952 | 0.0040 |
| rolling_mean_30d | 0.1728 | 0.0059 |
| history_days_available | 0.0649 | 0.0036 |
| total_duration_min | 0.0540 | 0.0021 |
| interval_mean | 0.0184 | 0.0024 |
| night_ratio | 0.0053 | 0.0011 |
| rolling_mean_3d | 0.0026 | 0.0007 |
| rolling_mean_14d | 0.0022 | 0.0002 |
| rolling_std_14d | 0.0018 | 0.0008 |
| interval_std | 0.0017 | 0.0009 |
| rolling_std_30d | 0.0016 | 0.0005 |
| event_rate_per_wear_hour | 0.0010 | 0.0004 |
| rolling_std_7d | 0.0004 | 0.0004 |
| rolling_std_3d | 0.0003 | 0.0003 |
| duration_std | 0.0001 | 0.0003 |
| max_event_duration_sec | 0.0000 | 0.0000 |
| duration_median | 0.0000 | 0.0000 |
| duration_mean | 0.0000 | 0.0000 |
| duration_rate_per_wear_hour | 0.0000 | 0.0000 |
| event_count | 0.0000 | 0.0000 |
| wear_completeness_ratio | 0.0000 | 0.0000 |
| breed_or_size_class | 0.0000 | 0.0000 |
| rolling_mean_7d | 0.0000 | 0.0000 |
