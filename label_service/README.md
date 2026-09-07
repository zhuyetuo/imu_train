# label_service — 数据标注平台用的 AI 推理/训练服务

给 `label_infra`（smart-label 后端）调用的 HTTP 服务，**跑在 imu_train 仓库里，跟命令行共用同一份代码**：

- `/infer` 底层就是 `src/infer_csv_scratch.infer_file()`——跟 `run_review_bins_all_days.sh` 调的同一个函数、同一套参数，出来的片段跟 `infer_result_majority/` 下的 `*_infer.json` 一致
- `/train` 底层就是后台跑 `train_custom.sh`

跟线上的 `algo_service` 没有关系，那个是生产项圈服务，不要混用。

## 启动

```bash
cd ~/imu_train
bash label_service/run.sh
```

所有默认值都在 `label_service/config.py` 里，就是你常用那条推理命令的配置，不用传任何环境变量；要换模型改 `config.py` 里的 `LABEL_MODEL` 默认值，或者临时 `LABEL_MODEL="..." bash label_service/run.sh`：

| 环境变量 | 默认 | 对应命令行 |
|---|---|---|
| `LABEL_MODEL` | `results/processed_2026_8_11-2026_8_27_raw_missing_drop_window/16hz_remap_custom_3class/rf/*.pkl` | `MODEL=...`（通配符规则一样：必须恰好匹配一个文件） |
| `DEVICE_HZ` | `50` | `DEVICE_HZ=50` |
| `RESAMPLE_METHOD` | `training_match` | `RESAMPLE_METHOD=training_match` |
| `TARGET_LABELS` | `活动,睡觉,抓挠,未佩戴,甩身体` | `TARGET_LABELS=...` |
| `NAS_ROOT` | `/home/toky/ai_data` | `/infer` 里传的 `path` 相对这个目录 |
| `LABEL_SERVICE_PORT` | `8383` | |
| `LABEL_JOBS_DIR` | `label_service/jobs/` | 训练任务状态 + 日志落盘处 |
| `LABEL_INFER_WORKERS` | CPU 核数-2 | `WORKERS=-1`（推理进程池大小，按文件并行） |
| `LABEL_LOG_DIR` | `label_service/logs/` | 日志目录 |
| `MATERIAL_ROOT` | `/home/toky/alg_material` | 素材库 NAS 挂载点，牙齿照片在 `口腔验证/` 下 |
| `TOOTH_WEIGHTS` | `tooth_health/data/runs/tooth_detect/weights/best.pt` | 牙齿 YOLO 权重（不在仓库里，没有则 /tooth 接口不可用，不影响 IMU 推理） |
| `TOOTH_CONF` / `TOOTH_IMGSZ` | `0.5` / `960` | 跟 tooth_health/code/web_app.py 默认一致 |

依赖：`pip install -r label_service/requirements.txt`（只多装 fastapi/uvicorn，其余复用仓库已有依赖）。

起来之后：`curl http://localhost:8383/health`，接口文档 `http://localhost:8383/docs`。

label_infra 那边只需要配 `ALGO_SERVICE_URL=http://<这台机器IP>:8383`。

## 日志

`label_service/logs/` 下（按天切、留 14 天）：
- `label_service.log` — 启动参数、每次推理（文件、耗时、各类别几段）、批量推理起止、训练任务提交/完成/失败、报错堆栈
- `access.log` — HTTP 访问日志（谁调了什么接口、状态码）
- `../jobs/{job_id}.log` — 每个训练任务 `train_custom.sh` 的完整输出

```bash
tail -f label_service/logs/label_service.log
```

## 接口

**`POST /api/v1/label/infer`** — 同步
```json
{"path": "data_raw/2026_8_28/xxx_imu1_raw.csv", "sample_id": 123}
```
返回 `segments`（按类别分组的片段，每段 `start_ts/end_ts/conf_max/conf_mean/n_windows`）+ `windows`（逐窗口预测和全类别概率）。字段跟 `*_infer.json` 一样。

**`POST /api/v1/label/infer_batch`** — 批量，进程池并行（几百个样本用这个）
```json
{"items": [{"path": "data_raw/.../a_imu1_raw.csv", "sample_id": 1}, {"path": "...", "sample_id": 2}]}
```
返回跟 items 一一对应的列表，每项 `{sample_id, path, ok, error, result}`，`result` 跟 `/infer` 的返回一样；单个文件失败不影响其它的。一个请求就能把 CPU 吃满（`LABEL_INFER_WORKERS` 个进程同时跑），不用调用方自己控制并发。单个 `/infer` 也走同一个进程池，并发发多个请求同样并行，但串行一个个发就只用得上 1 个核。

RF 本身预测很快，慢的是特征提取（纯 CPU 的 Python 循环，10 分钟的 50Hz 数据单核约 12 秒），所以提速靠多进程按文件并行，跟 `run_review_bins_all_days.sh` 的 `WORKERS=-1` 一个道理。

**`POST /api/v1/label/train`** — 提交训练，立刻返回 job
```json
{"dataset": {"date": "2026_8_28", "extra_date": ["2026_8_11-2026_8_27:50"],
             "missing_strategy": "drop_window", "skip_syn": false, "feat_workers": -1},
 "model_type": "rf", "tag": "可选"}
```
参数原样透传给 `train_custom.sh`。

**`GET /api/v1/label/train/{job_id}`** — 轮询状态 `queued/running/done/failed`，完成后带 `model_path/metrics`；日志在 `label_service/jobs/{job_id}.log`。

训练完**不会自动替换** `/infer` 正在用的模型——要换就改 `LABEL_MODEL` 重启服务。

**`POST /api/v1/tooth/detect`** — 牙齿/口腔照片 YOLO 检测
```json
{"path": "口腔验证/2026-09-02-ok/Bali/微信图片_xxx.jpg", "conf": 0.5, "with_image": true}
```
`path` 相对 `MATERIAL_ROOT`（或 `NAS_ROOT`）。返回 `detections[]`（`class_name/confidence/box[x1,y1,x2,y2]`）、`class_names`、原图尺寸、`annotated_jpeg_b64`（带框图）。`GET /api/v1/tooth/status` 看权重在不在、加载了没。

## 皮肤评估接口（`/api/v1/skin/*`，给 label_infra「皮肤评估」页用）

PM 规则全部从 `pm_skin_scoring/code/questionnaire_app.py` 逐字抽到 `label_service/skin_rules.py`（`python label_service/tools/sync_skin_rules.py` 重新生成 + 10 万组输入一致性校验），Gradio 那个 app 照旧独立能用。

| 接口 | 作用 |
|---|---|
| `GET /skin/options` | 题目/选项/分值、狗名、IMU→狗默认映射、档位阈值、周报表列定义 |
| `POST /skin/questionnaire-score` | 问答分（分项、两组小计、红旗项、缺答） |
| `POST /skin/c-score` | C 值（变化幅度/聚集/持续/中断四项、红旗、档位） |
| `POST /skin/s-total` | S 总分（C×40% + 皮肤组×35% + 毛发组×25%、档位、红旗） |
| `POST /skin/stats/scan` | 扫 `{root}/{day}/抓挠/imu_daily_scratch_stats.csv`（IMU_STATS=1 产出） |
| `POST /skin/stats/to-c-inputs` | 一行日统计 → C 值计算的输入 + 警示 |
| `POST /skin/ml/scan` `/ml/preview` `/ml/predict-c` `/ml/predict-s` | ML 模型 A/B（skin_health/code/rf_infer.py），兼容 `{day}/_infer` 和 `{day}/抓挠/_infer` 两种目录结构 |

记录/周报表的存储在 label_infra 的数据库里，这边只负责算。

## 推理模式：调试版 / 稳定版

`/infer`、`/infer_batch` 都接受 `mode`：

- `raw`（默认，调试版）：模型逐窗口 argmax 的原始输出，活动/睡觉会来回闪、抓挠常有单窗口噪声，适合看模型到底说了什么。
- `stable`（稳定版）：同一次推理结果做后处理（`postprocess.py`）：状态类概率滑动平均 + 短片段并入邻居；事件类（抓挠/甩身体）双阈值滞回（≥0.5 进入、≥0.25 维持）+ 间隔 ≤4s 合并成 bout + 抓挠前后 3s 内的甩身体并入抓挠，窗口数/平均概率不够的丢掉。参数见 `config.py` 的 `STABLE_*`。
- `viterbi`（稳定版 v2）：各类概率当发射概率、切换类别付固定代价，动态规划解码整条时间轴；事件类同样走合并/过滤。参数 `STABLE_VITERBI_SWITCH`。

每个窗口和片段都带 `spec`（陀螺仪 4–8 Hz 能量占比，抓挠的独立物理证据）。`STABLE_SPECTRAL_MIN` 设成 >0 后，平均 spec 低于它的抓挠 bout 会被丢掉；默认 0 不启用，先在 `data_labeled_ai/` 的 JSON 里看真/假抓挠的 spec 分布再定阈值。

几个模式用的是同一次模型推理，稳定版不会多花时间（频谱多读一次 CSV，约 1 秒）。

## 疑似抓挠候选、边界微调、训练闭环

- 稳定版/v2 的响应多一个 `candidates`：低门槛（`CAND_ENTER/CAND_STAY`）滞回抽出来、或频谱占比 ≥ `CAND_SPEC_MIN` 但模型没判抓挠的段，不进正式片段，给人工审核找漏检。
- `REFINE_ENABLED=1`（默认）时，抓挠/甩身体片段和候选的起止用陀螺仪能量包络（10 Hz）在 ±`REFINE_MARGIN_S` 内精确到 0.1 秒。
- `/train` 的 `dataset.export_json`：label_infra 从审核通过的任务导出的 Label Studio 格式 JSON（NAS 相对路径），服务整理成 `data/raw_custom/<date>/merged_tmp.json` 并把 CSV 软链进 `data/raw_wit/`；`source_hz`/`hz`/`clean` 直接透传给 `train_custom.sh`。
- `POST /api/v1/label/model/switch {model_path}`：运行时切换推理模型（重建进程池），重启后回到 `LABEL_MODEL`。

## 排队与并发

- 批量预标注最多占 `LABEL_INFER_WORKERS - LABEL_INFER_RESERVE` 个槽位（默认留 2 个），
  剩下的永远留给工作台点「AI预标注」的交互式请求——批量慢一点没关系，点一下要马上有反应。
- `GET /api/v1/label/queue`：几个在算、几个在等、平均一个文件多久、预计多久消化完。
  批量推理开始/结束的日志里也会带一份。
- worker 里 `load_csv` 带一个只存最近一个文件的缓存：一次推理原来要把 CSV 读两遍
  （特征一遍、频谱一遍），现在第二次直接命中。频谱也改成所有窗口一次批量 FFT。
