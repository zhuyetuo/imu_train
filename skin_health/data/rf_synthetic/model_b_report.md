# 模型B（综合严重度分类器）训练报告——合成数据

训练表：602行，45个场景狗（仅C1/C2触发问答且用户回答的天）

交叉验证(5折，按场景狗分组)宏F1：均值=0.902 标准差=0.030

## 特征重要性(permutation importance，全量数据)

| 特征 | 重要性均值 | 标准差 |
|---|---|---|
| skin_lesion_severity | 0.0359 | 0.0055 |
| odor_level | 0.0235 | 0.0046 |
| model_a_proba_C2 | 0.0160 | 0.0035 |
| skin_redness_level | 0.0150 | 0.0041 |
| hair_loss_spot_count_level | 0.0140 | 0.0033 |
| hair_loss_max_diameter_level | 0.0090 | 0.0021 |
| coat_quality_level | 0.0063 | 0.0026 |
| rolling_mean_30d | 0.0040 | 0.0026 |
| model_a_proba_C1 | 0.0005 | 0.0009 |
| rolling_mean_7d | 0.0005 | 0.0009 |
| event_rate_per_wear_hour | 0.0000 | 0.0000 |
| event_count | 0.0000 | 0.0000 |
| duration_mean | 0.0000 | 0.0000 |
| total_duration_min | 0.0000 | 0.0000 |
| max_event_duration_sec | 0.0000 | 0.0000 |
| duration_std | 0.0000 | 0.0000 |
| night_ratio | 0.0000 | 0.0000 |
| duration_rate_per_wear_hour | 0.0000 | 0.0000 |
| rolling_mean_14d | 0.0000 | 0.0000 |
| rolling_mean_3d | 0.0000 | 0.0000 |
| breed_or_size_class | 0.0000 | 0.0000 |
| history_days_available | 0.0000 | 0.0000 |
| sleep_disruption_count | 0.0000 | 0.0000 |
| wear_completeness_ratio | 0.0000 | 0.0000 |
| interval_std | 0.0000 | 0.0000 |
| interval_mean | 0.0000 | 0.0000 |
| duration_median | 0.0000 | 0.0000 |
| owner_reported_scratch_count_bucket | 0.0000 | 0.0000 |
| rolling_std_7d | 0.0000 | 0.0000 |
| rolling_std_14d | 0.0000 | 0.0000 |
| rolling_std_3d | 0.0000 | 0.0000 |
| rolling_std_30d | 0.0000 | 0.0000 |
| skin_pigment_abnormal | 0.0000 | 0.0000 |
| owner_reported_scratch_duration_bucket | 0.0000 | 0.0000 |
