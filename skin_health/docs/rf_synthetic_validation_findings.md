# RF合成数据验证结果——模型A（行为严重度分类器）

> 用86个合成场景（workflow多agent设计，覆盖急性发作/渐进恶化/波动复发/
> 数据质量冷启动/噪声基线多样性/问答相关六大主题，含大量卡在真实业务阈值
> 边界的场景）训练模型A，验证`rf_feature_spec.md`的特征集。**标签来自
> `scratch_burden.py`当前的SBS规则**（没有真实兽医标签），所以这次结果
> 验证的是"特征管道本身能不能学出SBS想表达的规律"，不是真实临床准确率——
> 等真实标签攒够后要重新训练评估，但这次结果已经能指出特征集哪里需要调整。

## 复现方式

```bash
python3 skin_health/code/gen_rf_synthetic_scenarios.py \
    --scenarios skin_health/data/rf_synthetic/scenarios.json \
    --out_dir skin_health/data/rf_synthetic
python3 skin_health/code/train_rf_model_a.py \
    --data_dir skin_health/data/rf_synthetic --n_folds 8
```

场景定义（`scenarios.json`）、生成的事件/佩戴CSV（`all_events.csv`/
`all_wear.csv`，以及每个场景单独一份）、场景元信息（`scenario_meta.csv`）
全部保存在`skin_health/data/rf_synthetic/`，跟代码一起提交，同一份
scenarios.json重跑会得到完全一样的结果（每个场景用`scenario_id`哈希出
固定随机种子）。

## 整体结果

- 86个场景，2857行(狗,天)训练样本，标签分布 C0:1819 / C1:709 / C2:329
- 8折`GroupKFold`（按场景狗分组）交叉验证：**宏F1均值0.886，标准差0.030**
- 混淆矩阵显示C1/C2之间有一定混淆（51+53=104行C1被误判到C0或C2，这是
  三分类里最难区分的中间档，符合预期，不是代码问题）

完整报告见`model_a_report.md`，特征重要性明细见`model_a_feature_importance.csv`。

## 特征重要性发现与调整建议

### 1. `sleep_disruption_count`重要性异常高——已验证是合成数据生成器的耦合artifact，不代表真实价值

首次训练permutation importance显示`sleep_disruption_count`（0.295）远超
第二名`rolling_mean_30d`（0.173）。做了消融实验验证：**去掉这个特征重新
训练，宏F1只从0.886掉到0.880**，几乎没有影响——说明这个特征携带的信息
在其他特征（主要是`rolling_mean_30d`/`total_duration_min`）里已经有
冗余覆盖，permutation importance显示的"重要性"只是这次模型训练时**恰好
依赖了这一个特征作为主要判别路径**，不代表这是唯一或不可替代的信息源。

根因：合成数据生成器里，异常期的`night_bias`参数（很多场景设在0.3-0.6）
让异常期新增事件也更容易落在夜间，`sleep_disruption_count`因此变成了
"是否处于异常期"的一个近似代理——这是生成器设计的副作用，不是真实业务
规律的必然结果（真实狗夜间抓挠增多不一定总是这么强的伴随关系）。

**调整建议**：不删除这个特征（真实数据里它仍然是有意义的信号，
`rf_feature_spec.md`里已经论证过它是瘙痒强度的强代理指标），但训练时
要注意监控这类"单特征依赖过重"的情况，等真实数据后重新检查这个特征的
重要性是否还是这么突出——如果突出，需要确认是真实规律还是数据本身的
某种偏差。

### 2. 时长统计量（`duration_mean`/`duration_median`/`max_event_duration_sec`/`duration_rate_per_wear_hour`）重要性持续为0

这几个在这次合成数据上完全没有边际贡献。分析原因：`total_duration_min`
已经是"次数×平均时长"的汇总量，`rolling_mean_30d`（基于`event_rate_
per_wear_hour`的滚动均值）进一步吸收了趋势信息，这几个更细粒度的时长
统计量在SBS标签定义下没有提供增量信息——**这跟`rf_feature_spec.md`
之前的猜测吻合**（`duration_mean`/`duration_median`当时就标注了"跟中位数
可能存在部分冗余"），但`max_event_duration_sec`归零比较意外，因为它
理论上应该是"长时间抓挠红旗"的直接依据。

**调整建议**：保留这几个特征（清零可能是这批合成数据里`long_scratch_
injection`触发的场景太少、真实数据里也可能因为标签细分不同而恢复重要性），
但降低优先级——如果后续要精简特征集，这几个是候选清单里的第一梯队。

### 3. `wear_completeness_ratio`重要性为0——符合预期，不是问题

SBS当前的标签定义（`data_quality_flag`只做二值过滤，`insufficient`的天
直接被排除出训练集）本身不依赖"佩戴覆盖率是多少"这个连续量，只要过了
`good`/`partial`的门槛，覆盖率具体多少不影响SBS怎么打分——所以这个特征
对"预测SBS标签"这个任务边际贡献为0是完全符合预期的，**不代表这个特征
本身没用**，等换成真实兽医标签（可能确实会因为佩戴覆盖率不同而影响
诊断可信度）时需要重新评估。

### 4. `breed_or_size_class`重要性为0——符合预期，不是问题

SBS的评分公式本身完全不看品种（只用个体自己的历史基线），所以拿SBS
标签当训练目标时，品种特征天然不会有预测力——这是"训练目标本身不依赖
品种"导致的，不是"品种信息没用"。等模型B或者未来真实标签里品种确实
带来差异时，这个特征才有机会体现价值（呼应`two_stage_rf_architecture.md`
里品种特征主要服务于跟BHM品种先验对接的设计意图）。

## 结论：这次合成数据验证证明了什么、没证明什么

**证明了**：
- 特征计算管道（`rf_features.py`）+ 训练管道（`train_rf_model_a.py`）
  代码没有明显bug，能在合理的场景多样性下达到还不错的分类效果
- 多窗口滚动特征（尤其`rolling_mean_30d`）是目前最重要的单一特征类别，
  验证了`rf_feature_spec.md`里"多窗口滚动特征值得优先实现"的判断
- 一部分候选冗余特征（时长细粒度统计量）在这次验证里确实边际贡献很低

**没有证明、也不该拿这次结果当结论的**：
- 真实业务准确率（标签来自SBS规则本身，不是真实兽医诊断）
- 品种/佩戴覆盖率这两个特征"没用"——只是在这个特定的合成标签定义下
  没有边际贡献，换成真实标签后需要重新评估
- `sleep_disruption_count`的真实重要性——消融实验证明它在这批合成数据上
  可替代，但这是数据设计的副作用，不是真实世界的结论
