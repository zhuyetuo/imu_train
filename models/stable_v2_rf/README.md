# 稳定版 v2 在用的 RF 模型（钉住不动）

线上 label_service 和标注平台「稳定版 v2」用的就是这份。效果好，
钉在这里防止被后续训练覆盖——results/ 是 gitignored 的，存不住。

    来源：results/processed_2026_8_11-2026_8_27_raw_missing_drop_window/16hz_remap_custom_3class/rf/ml_rf.pkl
    钉住时间：2026-09-16T02:00:11+08:00
    类别：活动、睡觉、抓挠、未佩戴、甩身体（5 类）
    几何：16Hz，窗口 2.0s，步长 1.0s，重力对齐
    推理时：device_hz=50 → 重采样到 16Hz，算法 training_match

**不要改这个目录。**换模型新建目录，别覆盖。
