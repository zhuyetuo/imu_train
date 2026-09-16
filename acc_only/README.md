# 只用加速计的模型（3 轴验证）

回答一个问题：**如果项圈上只有加速度计、没有陀螺仪，效果掉多少。**

这个目录里全是新增文件，仓库原有的任何东西都没有动。

---

## 一句话用法

```bash
cd ~/imu_train
bash acc_only/train_acc_only.sh \
  --processed_dir data/processed_2026_8_11-2026_8_27_raw_missing_drop_window
```

`--processed_dir` 给的是**基线那份已经预处理好的目录**（跑过 `train_custom.sh`
之后就有）。不确定是哪个的话：

```bash
ls -d data/processed_*
```

训完会直接打印「基线 vs 只用加速计」的对比表（总体准确率、macro F1、逐类 F1）。

---

## 做法：保留 8 通道的形状，把陀螺仪置零

不是「只留 acc 三列」。这一点是整件事的关键，说清楚为什么。

### 只留 3 列会连带砍掉一堆本来算得出来的特征

`src/ml/features.py` 的 `_extract_one` 里，**全局特征、模长特征、jerk 特征
全都挂在 `window.shape[1] >= 6` 这个条件下**。喂 3 通道进去，这一整块直接跳过：

- acc 三轴模长的时域 + 频域统计（19 维）
- acc jerk 模长（11 维）
- acc 三轴相关系数、SMA

**而这些只靠加速计就能算。** 另外 `preprocess.py` 里取姿态角写的是
`[:, :, 6:8]`，3 通道时那是个空切片——**不报错，静默变成没有 pitch/roll**。
可 pitch/roll 本来就是**从 acc 三列算出来的**（`gravity_align.raw_tilt` 只读
`w[:, :3]`），一个 3 轴设备完全给得出来。

最后只剩 57 维，其中一大半是被实现细节砍掉的，不是"加速计本来就没有"。
拿它跟 193 维的基线比，比出来的是两套特征工程的差距。

### 置零之后，剩下的正好是"加速计能给的全部信息"

193 维里**正好 80 维变成常数**：

| 变成常数的 | 维数 |
|---|---|
| gyr_x / gyr_y / gyr_z（时域 11 + 频域 8） | 57 |
| gyro_mag（时域 11 + 频域 8） | 19 |
| sma_gyro | 1 |
| corr_gyro_xy / yz / xz | 3 |

剩下 **113 维**全部只依赖加速计和它派生的 pitch/roll、acc 模长、jerk。
反过来也验过：把陀螺仪换成任意随机值，**只有那 80 维在动，其余逐位不变**。

常数特征的信息增益恒为 0，树模型永远不会选它——所以这个模型跟真正
只有加速计的设备是等价的。`acc_only/selfcheck.py` 把这两件事都验一遍，
还会拿训好的真模型跑一次"陀螺仪换随机值，预测必须逐条不变"。

### 另外两个好处

**窗口跟基线逐位相同。** 同一份 npz 复制出来的，窗口切分、训练/验证划分、
标签、重力对齐全一样，只有陀螺仪那 6 列不同。所以准确率的差值**只能**
归因于陀螺仪。重跑一遍预处理的话，划分的随机性就混进对比里了。

**训出来的模型能直接在现有服务上跑。** 特征还是 193 维，几何还是 16 点
@16Hz，5 个类别——label_service / algo_service / 端侧服务一行都不用改。
而 57 维的模型喂进去会当场报
`X has 193 features, but RandomForestClassifier is expecting 57`。

---

## ⚠ 这一版不回答"端侧省多少"

板子上真做 3 轴的话只算 57 维（`core/tm_features.c` 的 `n_ch < 6` 分支），
而这里仍然是 193 维的计算量。**这一版回答的是效果，不是 flash / ms。**

效果如果能接受，再谈导出成 C：那时候才需要处理 57 维那条路
（`export_rf.py --channels 3` 已经支持，`n_features(3) = 57` 对得上）。

---

## 在平台上验证效果

训完之后把模型挂到端侧服务（algo_tinyml），平台的「版本」下拉里就会多一项。

```bash
# 1. 拷过去
mkdir -p ~/algo_tinyml/models/acc_only_rf
cp ~/imu_train/results_acc_only/*_acc_only/16hz_remap_custom_3class/rf/ml_rf.{pkl,json} \
   ~/algo_tinyml/models/acc_only_rf/

# 2. 重启端侧服务
cd ~/algo_tinyml && ./serve.sh -d

# 3. 平台上「版本」下拉里选  端侧 · acc_only_rf · 稳定版 v2
```

`edge_models.json` 里已经有这一项了（`kind: sk`，跑服务器上的 sklearn，
不是板上那份 C），标了 `optional` ——没训之前服务照常起，只是少这一个选项。

**后处理跟线上「稳定版 v2」是同一份代码**，所以并排比的时候差的只有模型本身。

---

## 文件

| 文件 | 干什么 |
|---|---|
| `make_acc_only_npz.py` | 复制预处理好的窗口，陀螺仪三轴置零。会主动删掉目标目录里的旧特征缓存——那份缓存维度一样（193），`train.py` 的"维度对不上就重建"保护**不会触发**，复制过去的话训出来的模型跟基线一模一样而日志毫无异常 |
| `train_acc_only.sh` | 置零 + 训练 + 自检 + 对比，一条命令 |
| `selfcheck.py` | 验"置零 ≡ 没有陀螺仪"，以及训好的模型确实没用到陀螺仪 |
