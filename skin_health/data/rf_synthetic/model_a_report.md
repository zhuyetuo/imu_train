# 模型A（行为严重度分类器）训练报告——合成数据

训练表：2857行，86个场景狗

交叉验证(5折，按场景狗分组)宏F1：均值=0.925 标准差=0.012

## 特征重要性(permutation importance，全量数据)

| 特征 | 重要性均值 | 标准差 |
|---|---|---|
| sleep_disruption_count | 0.2882 | 0.0039 |
| baseline_ratio_count_excl_recent14 | 0.1126 | 0.0048 |
| total_duration_min | 0.0449 | 0.0024 |
| night_ratio | 0.0119 | 0.0015 |
| rolling_mean_3d | 0.0047 | 0.0010 |
| interval_mean | 0.0007 | 0.0008 |
| baseline_ratio_duration_excl_recent14 | 0.0003 | 0.0004 |
| event_count | 0.0000 | 0.0000 |
| duration_std | 0.0000 | 0.0000 |
| max_event_duration_sec | 0.0000 | 0.0000 |
| duration_median | 0.0000 | 0.0000 |
| duration_mean | 0.0000 | 0.0000 |
| duration_rate_per_wear_hour | 0.0000 | 0.0000 |
| event_rate_per_wear_hour | 0.0000 | 0.0000 |
| wear_completeness_ratio | 0.0000 | 0.0000 |
| history_days_available | 0.0000 | 0.0000 |
| has_any_baseline | 0.0000 | 0.0000 |
| z_score_vs_self | 0.0000 | 0.0000 |
| baseline_ratio_count_recent3 | 0.0000 | 0.0000 |
| consecutive_days_above_baseline | 0.0000 | 0.0000 |
| baseline_ratio_count_recent7 | 0.0000 | 0.0000 |
| baseline_ratio_count_recent14 | 0.0000 | 0.0000 |
| breed_or_size_class | 0.0000 | 0.0000 |
| interval_std | 0.0000 | 0.0000 |
| baseline_ratio_count_recent30 | 0.0000 | 0.0000 |
| baseline_ratio_count_recent21 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent7 | 0.0000 | 0.0000 |
| baseline_ratio_count_excl_recent30 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent14 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent21 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent30 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent3 | 0.0000 | 0.0000 |
| baseline_ratio_duration_excl_recent30 | 0.0000 | 0.0000 |
| rolling_mean_7d | 0.0000 | 0.0000 |
| rolling_mean_14d | 0.0000 | 0.0000 |
| rolling_mean_30d | 0.0000 | 0.0000 |
| rolling_std_3d | 0.0000 | 0.0000 |
| rolling_std_7d | 0.0000 | 0.0000 |
| rolling_std_14d | 0.0000 | 0.0000 |
| rolling_std_30d | 0.0000 | 0.0000 |
