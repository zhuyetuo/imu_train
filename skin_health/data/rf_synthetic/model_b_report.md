# 模型B（综合严重度分类器）训练报告——合成数据

训练表：602行，45个场景狗（仅C1/C2触发问答且用户回答的天）

交叉验证(5折，按场景狗分组)宏F1：均值=0.892 标准差=0.013

## 特征重要性(permutation importance，全量数据)

| 特征 | 重要性均值 | 标准差 |
|---|---|---|
| model_a_proba_C1 | 0.0861 | 0.0118 |
| skin_redness_level | 0.0748 | 0.0101 |
| coat_quality_level | 0.0442 | 0.0054 |
| odor_level | 0.0384 | 0.0052 |
| skin_lesion_severity | 0.0228 | 0.0036 |
| hair_loss_spot_count_level | 0.0073 | 0.0019 |
| hair_loss_max_diameter_level | 0.0042 | 0.0014 |
| duration_rate_per_wear_hour | 0.0000 | 0.0000 |
| night_ratio | 0.0000 | 0.0000 |
| interval_mean | 0.0000 | 0.0000 |
| interval_std | 0.0000 | 0.0000 |
| sleep_disruption_count | 0.0000 | 0.0000 |
| duration_mean | 0.0000 | 0.0000 |
| duration_median | 0.0000 | 0.0000 |
| max_event_duration_sec | 0.0000 | 0.0000 |
| duration_std | 0.0000 | 0.0000 |
| event_count | 0.0000 | 0.0000 |
| event_rate_per_wear_hour | 0.0000 | 0.0000 |
| total_duration_min | 0.0000 | 0.0000 |
| flare_episode_count_90d | 0.0000 | 0.0000 |
| consecutive_days_above_baseline | 0.0000 | 0.0000 |
| z_score_vs_self | 0.0000 | 0.0000 |
| has_any_baseline | 0.0000 | 0.0000 |
| breed_or_size_class | 0.0000 | 0.0000 |
| history_days_available | 0.0000 | 0.0000 |
| wear_completeness_ratio | 0.0000 | 0.0000 |
| days_since_last_flare_end | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent3 | 0.0000 | 0.0000 |
| baseline_ratio_count_recent3 | 0.0000 | 0.0000 |
| baseline_ratio_count_recent7 | 0.0000 | 0.0000 |
| baseline_ratio_count_recent14 | 0.0000 | 0.0000 |
| baseline_ratio_count_recent21 | 0.0000 | 0.0000 |
| baseline_ratio_count_recent30 | 0.0000 | 0.0000 |
| baseline_ratio_count_excl_recent14 | 0.0000 | 0.0000 |
| baseline_ratio_count_excl_recent30 | 0.0000 | 0.0000 |
| rolling_mean_3d | 0.0000 | 0.0000 |
| baseline_ratio_duration_excl_recent30 | 0.0000 | 0.0000 |
| baseline_ratio_duration_excl_recent14 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent30 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent21 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent14 | 0.0000 | 0.0000 |
| baseline_ratio_duration_recent7 | 0.0000 | 0.0000 |
| rolling_mean_7d | 0.0000 | 0.0000 |
| rolling_std_30d | 0.0000 | 0.0000 |
| rolling_std_14d | 0.0000 | 0.0000 |
| rolling_std_7d | 0.0000 | 0.0000 |
| rolling_std_3d | 0.0000 | 0.0000 |
| rolling_mean_14d | 0.0000 | 0.0000 |
| rolling_mean_30d | 0.0000 | 0.0000 |
| skin_pigment_abnormal | 0.0000 | 0.0000 |
| model_a_proba_C2 | 0.0000 | 0.0000 |
