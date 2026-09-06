# label_service — 数据标注平台用的 AI 推理/训练服务

给 `label_infra`（smart-label 后端）调用的 HTTP 服务，**跑在 imu_train 仓库里，跟命令行共用同一份代码**：

- `/infer` 底层就是 `src/infer_csv_scratch.infer_file()`——跟 `run_review_bins_all_days.sh` 调的同一个函数、同一套参数，出来的片段跟 `infer_result_majority/` 下的 `*_infer.json` 一致
- `/train` 底层就是后台跑 `train_custom.sh`

跟线上的 `algo_service` 没有关系，那个是生产项圈服务，不要混用。

## 启动

```bash
cd ~/imu_train
LABEL_MODEL="results/processed_2026_8_11-2026_8_27_raw_missing_drop_window/16hz_remap_custom_3class/rf/*.pkl" \
bash label_service/run.sh
```

`LABEL_MODEL` 是必填的（支持通配符，规则跟 `run_review_bins_all_days.sh` 的 `MODEL` 一样：必须恰好匹配一个文件）。其余默认值就是你常用那条推理命令的配置：

| 环境变量 | 默认 | 对应命令行 |
|---|---|---|
| `DEVICE_HZ` | `50` | `DEVICE_HZ=50` |
| `RESAMPLE_METHOD` | `training_match` | `RESAMPLE_METHOD=training_match` |
| `TARGET_LABELS` | `活动,睡觉,抓挠,未佩戴,甩身体` | `TARGET_LABELS=...` |
| `NAS_ROOT` | `/home/toky/ai_data` | `/infer` 里传的 `path` 相对这个目录 |
| `LABEL_SERVICE_PORT` | `8383` | |
| `LABEL_JOBS_DIR` | `label_service/jobs/` | 训练任务状态 + 日志落盘处 |

依赖：`pip install -r label_service/requirements.txt`（只多装 fastapi/uvicorn，其余复用仓库已有依赖）。

起来之后：`curl http://localhost:8383/health`，接口文档 `http://localhost:8383/docs`。

label_infra 那边只需要配 `ALGO_SERVICE_URL=http://<这台机器IP>:8383`。

## 接口

**`POST /api/v1/label/infer`** — 同步
```json
{"path": "data_raw/2026_8_28/xxx_imu1_raw.csv", "sample_id": 123}
```
返回 `segments`（按类别分组的片段，每段 `start_ts/end_ts/conf_max/conf_mean/n_windows`）+ `windows`（逐窗口预测和全类别概率）。字段跟 `*_infer.json` 一样。同一时刻只跑一个推理（内部加了锁），多人同时点会排队。

**`POST /api/v1/label/train`** — 提交训练，立刻返回 job
```json
{"dataset": {"date": "2026_8_28", "extra_date": ["2026_8_11-2026_8_27:50"],
             "missing_strategy": "drop_window", "skip_syn": false, "feat_workers": -1},
 "model_type": "rf", "tag": "可选"}
```
参数原样透传给 `train_custom.sh`。

**`GET /api/v1/label/train/{job_id}`** — 轮询状态 `queued/running/done/failed`，完成后带 `model_path/metrics`；日志在 `label_service/jobs/{job_id}.log`。

训练完**不会自动替换** `/infer` 正在用的模型——要换就改 `LABEL_MODEL` 重启服务。
