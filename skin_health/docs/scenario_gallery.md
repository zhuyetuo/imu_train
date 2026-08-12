# 皮肤评级合成场景画廊

> 一共 10 个合成场景，每个对应 `gen_synthetic_scratch_scenarios.py` 里的一个 `scenario_*` 函数，目的是验证 **SBS 打分机制本身**（不是抓挠事件识别准确率）——也就是给定一组已知的抓挠事件，SBS引擎算出来的评级是否符合预期，用来在改动打分公式后快速回归测试。下面每个场景的标题都带编号（场景1/10、场景2/10……），跟目录一一对应。

## 名词解释（看不懂图例的先看这里）

- **SBS（Scratch Burden Score，抓挠负担分）**：算法给每只狗每天算出来的0~100分，分数越高代表当天的抓挠行为越偏离正常，由四个子项（变化幅度/聚集程度/持续程度/正常行为中断）加总得到，达到不同门槛对应下面的C0/C1/C2三档。
- **C0 正常 / C1 需要关注 / C2 建议兽医检查**：SBS总分 <30 是C0，30~50是C1，≥50是C2，对应下图绿/黄/红三种背景色。
- **红旗（red flag）**：不管总分多少，只要触发了预定义的严重信号（比如单次连续抓挠超过1分钟、1小时内聚集抓挠≥3次、影响夜间睡眠等），就在图上标黑色星号，代表这天有值得直接关注的具体行为，不是单纯靠算分数判断。
- **引导期（bootstrap_mode）**：这只狗历史数据还不满21天，没法算出"跟自己比"的个人基线，这期间改用文献（Whistle FIT）给的固定秒数阈值兜底评分，图上标空心方块，跟基线建立后的正常评分是两套逻辑，不能直接比较分数高低。

- **横轴（日期）**：每一格代表**一整天**（不是一天内的时间段），从合成数据第1天（2026-06-01）到第35天，前21天是给算法建立个人基线用的观察窗口，后14天是真正评估/触发问题的观察期，具体每个场景的"异常"通常安排在后14天里发生。

每张图分两部分：**上图**是当天实际发生的抓挠情况（原始数据，蓝柱=次数，红线=总时长），**下图**是算法根据上图数据算出来的SBS评分结果（total/tier），两张图上下对齐、同一天可以直接对照看"当天发生了什么 → 算法打了多少分"。每张图内部都带完整图例，不需要再回来看这段文字。

## 场景1/10：dog_stable_low

**预期**：全程应保持 C0（低基线、无明显偏离）

![dog_stable_low](assets/scenario_dog_stable_low.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 2 | 0 | C0 | - | 是 |
| 2026-06-02 | 2 | 0 | C0 | - | 是 |
| 2026-06-03 | 2 | 0 | C0 | - | 是 |
| 2026-06-04 | 3 | 0 | C0 | - | 是 |
| 2026-06-05 | 3 | 0 | C0 | - | 是 |
| 2026-06-06 | 3 | 0 | C0 | - | 是 |
| 2026-06-07 | 2 | 0 | C0 | - | 是 |
| 2026-06-08 | 2 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_stable_low_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_stable_low_daily_result.csv)

---

## 场景2/10：dog_sudden_spike

**预期**：后14天从每天3次开始，应在基线建立后逐步触发 C1/C2（验证低基线狗的相对变化是否能被捕捉到）

![dog_sudden_spike](assets/scenario_dog_sudden_spike.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 0 | 0 | C0 | - | 是 |
| 2026-06-02 | 0 | 0 | C0 | - | 是 |
| 2026-06-03 | 0 | 0 | C0 | - | 是 |
| 2026-06-04 | 0 | 0 | C0 | - | 是 |
| 2026-06-05 | 1 | 0 | C0 | - | 是 |
| 2026-06-06 | 0 | 0 | C0 | - | 是 |
| 2026-06-07 | 0 | 0 | C0 | - | 是 |
| 2026-06-08 | 0 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_sudden_spike_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_sudden_spike_daily_result.csv)

---

## 场景3/10：dog_gradual_worsening

**预期**：应观察到变化幅度分逐日上升、持续程度分在偏离持续>=3天后达到20

![dog_gradual_worsening](assets/scenario_dog_gradual_worsening.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 2 | 0 | C0 | - | 是 |
| 2026-06-02 | 2 | 0 | C0 | - | 是 |
| 2026-06-03 | 2 | 0 | C0 | - | 是 |
| 2026-06-04 | 1 | 0 | C0 | - | 是 |
| 2026-06-05 | 2 | 0 | C0 | - | 是 |
| 2026-06-06 | 2 | 0 | C0 | - | 是 |
| 2026-06-07 | 2 | 0 | C0 | - | 是 |
| 2026-06-08 | 2 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_gradual_worsening_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_gradual_worsening_daily_result.csv)

---

## 场景4/10：dog_high_baseline_modest

**预期**：PM文档示例五（30→35，倍数1.17）：应判定为无明显变化，变化幅度分=0

![dog_high_baseline_modest](assets/scenario_dog_high_baseline_modest.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 3 | 0 | C0 | - | 是 |
| 2026-06-02 | 2 | 0 | C0 | - | 是 |
| 2026-06-03 | 3 | 0 | C0 | - | 是 |
| 2026-06-04 | 3 | 0 | C0 | - | 是 |
| 2026-06-05 | 3 | 0 | C0 | - | 是 |
| 2026-06-06 | 3 | 0 | C0 | - | 是 |
| 2026-06-07 | 3 | 0 | C0 | - | 是 |
| 2026-06-08 | 3 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_high_baseline_modest_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_high_baseline_modest_daily_result.csv)

---

## 场景5/10：dog_clustered

**预期**：第25天(day25)应触发聚集程度=10分（1个聚集时段），其他天不触发

![dog_clustered](assets/scenario_dog_clustered.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 1 | 0 | C0 | - | 是 |
| 2026-06-02 | 2 | 0 | C0 | - | 是 |
| 2026-06-03 | 3 | 0 | C0 | - | 是 |
| 2026-06-04 | 2 | 0 | C0 | - | 是 |
| 2026-06-05 | 1 | 0 | C0 | - | 是 |
| 2026-06-06 | 1 | 0 | C0 | - | 是 |
| 2026-06-07 | 3 | 0 | C0 | - | 是 |
| 2026-06-08 | 2 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_clustered_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_clustered_daily_result.csv)

---

## 场景6/10：dog_night_disruption

**预期**：第28天应识别出2次睡眠中断（23:00/23:20间隔20min不合并，02:00算第2或第3次），触发中断分>=20

![dog_night_disruption](assets/scenario_dog_night_disruption.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 2 | 0 | C0 | - | 是 |
| 2026-06-02 | 1 | 0 | C0 | - | 是 |
| 2026-06-03 | 2 | 0 | C0 | - | 是 |
| 2026-06-04 | 3 | 0 | C0 | - | 是 |
| 2026-06-05 | 2 | 0 | C0 | - | 是 |
| 2026-06-06 | 2 | 0 | C0 | - | 是 |
| 2026-06-07 | 2 | 0 | C0 | - | 是 |
| 2026-06-08 | 2 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_night_disruption_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_night_disruption_daily_result.csv)

---

## 场景7/10：dog_long_scratch

**预期**：第30天应触发红旗 interrupt_or_long_scratch，正常行为影响分=30

![dog_long_scratch](assets/scenario_dog_long_scratch.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 2 | 0 | C0 | - | 是 |
| 2026-06-02 | 2 | 0 | C0 | - | 是 |
| 2026-06-03 | 2 | 0 | C0 | - | 是 |
| 2026-06-04 | 1 | 0 | C0 | - | 是 |
| 2026-06-05 | 1 | 0 | C0 | - | 是 |
| 2026-06-06 | 2 | 0 | C0 | - | 是 |
| 2026-06-07 | 1 | 0 | C0 | - | 是 |
| 2026-06-08 | 2 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_long_scratch_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_long_scratch_daily_result.csv)

---

## 场景8/10：dog_missing_wear

**预期**：缺失日应标记 insufficient，不应计入基线，也不应被当成'0次抓挠=正常'

![dog_missing_wear](assets/scenario_dog_missing_wear.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 3 | 0 | C0 | - | 是 |
| 2026-06-02 | 1 | 0 | C0 | - | 是 |
| 2026-06-03 | 2 | 0 | C0 | - | 是 |
| 2026-06-04 | 3 | 0 | C0 | - | 是 |
| 2026-06-05 | 2 | 0 | C0 | - | 是 |
| 2026-06-06 | 1 | 0 | C0 | - | 是 |
| 2026-06-07 | 2 | 0 | C0 | - | 是 |
| 2026-06-08 | 3 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_missing_wear_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_missing_wear_daily_result.csv)

---

## 场景9/10：dog_cold_start

**预期**：前21天历史不足，bootstrap_mode=True；轻度抓挠远低于120秒/天，应保持 C0，不强行凑基线打分

![dog_cold_start](assets/scenario_dog_cold_start.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 2 | 0 | C0 | - | 是 |
| 2026-06-02 | 2 | 0 | C0 | - | 是 |
| 2026-06-03 | 2 | 0 | C0 | - | 是 |
| 2026-06-04 | 3 | 0 | C0 | - | 是 |
| 2026-06-05 | 2 | 0 | C0 | - | 是 |
| 2026-06-06 | 1 | 0 | C0 | - | 是 |
| 2026-06-07 | 2 | 0 | C0 | - | 是 |
| 2026-06-08 | 2 | 0 | C0 | - | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_cold_start_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_cold_start_daily_result.csv)

---

## 场景10/10：dog_severe_bootstrap

**预期**：基线还没建立（bootstrap_mode=True），但当日总时长远超300秒/天，应从第1天起就靠绝对阈值兜底触发高分/C2，而不是等21天基线建立后才发现

![dog_severe_bootstrap](assets/scenario_dog_severe_bootstrap.png)

关键天数（非C0 / 触发红旗 / 引导期，完整35天数据见CSV）：

| 日期 | 抓挠次数 | 总分 | 分档 | 红旗 | 引导期 |
|---|---|---|---|---|---|
| 2026-06-01 | 4 | 60 | C2 | bootstrap_absolute_severe, interrupt_or_long_scratch | 是 |
| 2026-06-02 | 4 | 60 | C2 | bootstrap_absolute_severe, interrupt_or_long_scratch | 是 |
| 2026-06-03 | 4 | 60 | C2 | bootstrap_absolute_severe, interrupt_or_long_scratch | 是 |
| 2026-06-04 | 4 | 60 | C2 | bootstrap_absolute_severe, interrupt_or_long_scratch | 是 |
| 2026-06-05 | 4 | 60 | C2 | bootstrap_absolute_severe, interrupt_or_long_scratch | 是 |
| 2026-06-06 | 4 | 60 | C2 | bootstrap_absolute_severe, interrupt_or_long_scratch | 是 |
| 2026-06-07 | 4 | 60 | C2 | bootstrap_absolute_severe, interrupt_or_long_scratch | 是 |
| 2026-06-08 | 4 | 60 | C2 | bootstrap_absolute_severe, interrupt_or_long_scratch | 是 |

原始数据：[抓挠事件明细](../data/synthetic_scenarios/dog_severe_bootstrap_events.csv) | [每日SBS结果](../data/synthetic_scenarios/dog_severe_bootstrap_daily_result.csv)

---
