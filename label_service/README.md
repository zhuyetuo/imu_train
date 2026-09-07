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
| `MATERIAL_ROOT` | `/home/toky/算法任务素材库` | 素材库 NAS 挂载点，牙齿照片在 `口腔验证/` 下 |
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
