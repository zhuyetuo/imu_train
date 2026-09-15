# vision_service

给标注平台用的 GPU 视觉服务。目前只有一件事：**SAM 2.1 交互式分割**——标注员在照片上
点一下，出一个掩膜，转成框和多边形回给平台，省掉手画框。

## 跟现有服务的关系

- **algo_service**：线上服务，一行不碰。
- **label_service**：IMU 推理，纯 CPU、同步 `/infer` + 进程池。也不碰，只是端口错开
  （它 8383，这里 8385）。混进去的话，SAM 这种长时 GPU 任务会让 IMU 推理排在后面。
- **平台调不通这个服务不是事故，是降级**：SAM 按钮置灰，其它功能一概不受影响。
  这也是为什么模型不可用时返回 503 而不是 500——平台据此置灰，而 500 会被当成 bug 弹红叉。

## 起服务

```bash
# 装依赖（torch 按机器上的 CUDA 版本单独装，这里不钉版本——钉错了会把现成环境搞坏）
pip install -r vision_service/requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

# 下 SAM 2.1 权重（Apache-2.0），放 vision_service/weights/
mkdir -p vision_service/weights && cd vision_service/weights
curl -LO https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# 起
./vision_service/run.sh
```

**没装 SAM 也能起**：`/api/v1/sam/status` 会如实说不可用和原因，平台把按钮置灰。
所以可以先把服务起起来，权重慢慢下。

## 配置（全走环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MATERIAL_ROOT` | `/home/toky/alg_material` | 素材库挂载点，请求里的路径相对它 |
| `VISION_SERVICE_PORT` | `8385` | 监听端口 |
| `SAM_CHECKPOINT` | `vision_service/weights/sam2.1_hiera_base_plus.pt` | 权重，不在仓库里 |
| `SAM_MODEL_CFG` | `configs/sam2.1/sam2.1_hiera_b+.yaml` | sam2 包自带的 config 名 |
| `SAM_DEVICE` | `cuda` | 没有卡会自动退回 cpu |

平台那边配 `VISION_SERVICE_URL=http://<这台机器>:8385` 才会去调它，不配就是关着的。

## 接口

```
GET  /health                    # 任何时候都能回，带 sam 状态
GET  /api/v1/sam/status         # {available, checkpoint, device, error}
POST /api/v1/sam/segment        # {path, points:[{x,y,label}], box?} → {bbox, polygon, score, width, height}
```

坐标**一律归一化到 0-1**，不传像素。前端拿到的图是缩放过的，传像素就得两边都知道
原图尺寸，迟早错一次。`label` 1=正点（要这块）0=负点（不要这块）。

## 为什么是 SAM 2.1，不是 SAM 3

SAM 2/2.1 的代码和权重是 **Apache-2.0**，可商用、无 copyleft；SAM 3（2025-11）和
3.1（2026-03）走 Meta 自定的 SAM License，衍生分发要沿用同一许可。而 SAM 3 的增量是
**文本概念提示**——"第 108 号牙"没有任何语义能跟邻牙区分，对这个场景零增益。

漂移严重时可以换 DAM4SAM / SAMURAI 的记忆策略（SAM 2.1 之上免训练 drop-in，正好
针对邻牙这种长得差不多的干扰物）。

## 已知的坑

- **口腔视频/照片上 SAM 的三大失败因子**：牙釉质和唾液的镜面高光（掩膜塌缩或跳到
  高光区）、嘴唇/舌遮挡后重识别错到邻牙、邻牙外观高度相似导致跟错。前两个能靠
  掩膜面积突变发现，**第三个"跟错了但跟得很稳"发现不了**，所以人工抽检不能省。
- **单卡上不要并发跑**：本来就串行，并发只买到显存峰值翻倍和碎片化。代码里加了锁。
- 一个点是有歧义的（牙面 / 整颗牙 / 一排牙），所以 `multimask_output=True` 出三个再按
  score 挑。挑错了就让标注员补一个负点。

## 测试

```bash
python -m pytest vision_service/tests -q
```

不需要 GPU、不需要 sam2、不需要权重。测的是掩膜换算、路径沙箱、以及**没有 SAM 时的
降级行为**——后者恰恰是新机器上最常遇到的状态。SAM 本身测不了，这里也没假装测。

## 画面里有没有狗（第一步）

不训练任何模型，用 COCO 预训练权重里现成的 `dog` 类。要回答的只有一个问题：
**这段视频里到底有没有狗**。整段没狗的片段，平台上直接标出来，人不用点进去
看波形才发现这半小时狗根本不在画面里。

```bash
./vision_service/run.sh -d                    # 缺的依赖它自己会装
curl -s localhost:8385/api/v1/dog/status      # available / loaded_weights / dog_class
```

起之前会检查依赖，缺了就按 requirements.txt 装上。**torch 和 sam2 不自动装**——
它们的版本取决于机器上的 CUDA，装错会把现成环境搞坏（GPU 版被覆盖成 CPU 版，
SAM 会悄悄退回 CPU 跑、慢十几倍还不报错）。缺了会提示怎么装。
不想让它自己装：加 `--no-install`，或者设 `VISION_NO_INSTALL=1`。

扫一段：

```bash
curl -s localhost:8385/api/v1/dog/scan -H 'Content-Type: application/json' \
  -d '{"path":"data_raw/2026_9_4/xxx_cam1_imu1_raw.mp4","every_sec":10}' \
  | python -m json.tool | head -20
```

`verdict` 三档：`no_dog`（整段没看见狗）/ `mostly_empty`（八成时间空镜）/
`has_dog`。**第四档是 `unknown`——一帧都没采到**，那是「没看成」不是「确认没狗」，
两者后果相反，不能混。

### 验一下 nano 够不够

默认权重是 `yolo26n.pt`（最小那档）。这是个**赌**：赌新架构的小模型已经够用。

按道理这一步该选大模型——它要的是**召回**不是速度（漏一只狗 = 人跳过一整段
真有素材的视频；多报一只 = 人点进去看一眼），而且按时间采样、一小时才 720 帧，
最大的模型也就几十秒跑完，速度几乎不花钱。

所以这个赌要拿真实素材验。同一批片段用两个型号各扫一遍，比 `verdict`：

```bash
for w in yolo26n.pt yolo26m.pt; do        # 第二个换成这个版本有的更大一档
  DOG_WEIGHTS=$w ./vision_service/run.sh down >/dev/null
  DOG_WEIGHTS=$w ./vision_service/run.sh -d >/dev/null
  echo "== $w"
  for f in <挑十来段真实的相对路径>; do
    curl -s localhost:8385/api/v1/dog/scan -H 'Content-Type: application/json' \
      -d "{\"path\":\"$f\",\"every_sec\":10}" \
      | python -c 'import sys,json;d=json.load(sys.stdin);print(d["verdict"], d["no_dog_ratio"], d["max_dogs"])'
  done
done
```

只要出现 **「n 说 no_dog、大的说 has_dog」** 的片段，就是赌输了——把
`DOG_WEIGHTS` 换成大一档。反过来（大的说没狗、n 说有狗）不要紧，那是误报，
人点进去看一眼就排掉了。

挑片段的时候要**专门挑难的**：夜里红外的、笼子栏杆挡着的、狗蜷成一团睡觉的、
白毛比熊贴浅色背景的。全挑白天大场面的话，两个型号都会全对，等于没验。
