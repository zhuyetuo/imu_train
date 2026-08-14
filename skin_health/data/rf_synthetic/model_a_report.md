# 模型A（行为严重度分类器）训练报告——合成数据

训练表：2857行，86个场景狗

交叉验证(8折，按场景狗分组)宏F1：均值=0.932 标准差=0.022

## 特征重要性(permutation importance，全量数据)

| 特征 | 重要性均值 | 标准差 |
|---|---|---|
| sleep_disruption_count | 0.2867 | 0.0039 |
| baseline_ratio_count | 0.0568 | 0.0050 |
| total_duration_min | 0.0477 | 0.0027 |
| night_ratio | 0.0159 | 0.0016 |
| rolling_mean_3d | 0.0071 | 0.0010 |
| baseline_delta_count | 0.0065 | 0.0010 |
| rolling_std_3d | 0.0008 | 0.0005 |
| interval_mean | 0.0004 | 0.0001 |
| baseline_ratio_duration | 0.0004 | 0.0005 |
| duration_mean | 0.0000 | 0.0000 |
| duration_rate_per_wear_hour | 0.0000 | 0.0000 |
| event_rate_per_wear_hour | 0.0000 | 0.0000 |
| event_count | 0.0000 | 0.0000 |
| wear_completeness_ratio | 0.0000 | 0.0000 |
| interval_std | 0.0000 | 0.0000 |
| max_event_duration_sec | 0.0000 | 0.0000 |
| duration_std | 0.0000 | 0.0000 |
| duration_median | 0.0000 | 0.0000 |
| baseline_delta_duration | 0.0000 | 0.0000 |
| history_days_available | 0.0000 | 0.0000 |
| breed_or_size_class | 0.0000 | 0.0000 |
| z_score_vs_self | 0.0000 | 0.0000 |
| has_baseline | 0.0000 | 0.0000 |
| rolling_mean_14d | 0.0000 | 0.0000 |
| rolling_mean_7d | 0.0000 | 0.0000 |
| rolling_mean_30d | 0.0000 | 0.0000 |
| rolling_std_7d | 0.0000 | 0.0000 |
| rolling_std_14d | 0.0000 | 0.0000 |
| rolling_std_30d | 0.0000 | 0.0000 |
