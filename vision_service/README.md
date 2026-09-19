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

### nano 不够用——实测过了（2026-09-15）

默认是 `yolo26x.pt`（最大那档）。这**不是**按经验选的，是量出来的：

```
素材：data_raw/2026_9_12/multicam_20260912_230430647_cam1_imu1_raw.mp4
      3329 秒（影棚，cam1），每 10 秒采一个点，共 333 个点

  yolo26n   mostly_empty     5/333 个点有狗（1.5%）
  yolo26x   has_dog        273/333 个点有狗（82%）
```

nano 漏掉了 **268 个**有狗的采样点。

**难点不是画面暗。** 同一批的 `230020516` 抽帧看过：室内开着灯、彩色正常、
平均亮度 123/255，人一眼就能看见沙发上躺着一只、门口趴着一只，nano 照样报零。
真正难的是**俯拍 + 目标小 + 蜷成一团**——摄像头吊在天花板上，沙发上那只只占
约 100×50 像素（画幅 1280×720，0.5% 面积），而 COCO 里的狗绝大多数是侧面/正面、
占画幅很大。这种视角和姿态是分布外的，小模型首先在这儿垮。

这正好是最坏的那种错：nano 会让平台显示「这一小时大部分是空镜」，人就跳过了
——实际上 82% 的时间画面里有狗。这一步**要的是召回不是速度**（漏一只狗 =
人跳过一整段真有素材的视频；多报一只 = 人点进去看一眼），而按时间采样、
不逐帧，5090 上 x 也就一两分钟，速度几乎不花钱。

### 想换小模型省显存的话，先重验

方法就是上面那次做的：同一段素材、两个型号各扫一遍，比 `verdict`。

```bash
VID="data_raw/2026_9_12/multicam_20260912_230430647_cam1_imu1_raw.mp4"
for W in yolo26n.pt yolo26x.pt; do
  echo "== $W"
  DOG_WEIGHTS=$W ./vision_service/run.sh down >/dev/null
  DOG_WEIGHTS=$W ./vision_service/run.sh -d >/dev/null
  # 权重要现下，固定 sleep 不够——轮询到 available 为止
  for i in $(seq 120); do
    curl -s localhost:8385/api/v1/dog/status | grep -q '"available":true' && break
    sleep 5
  done
  curl -s --max-time 1800 localhost:8385/api/v1/dog/scan -H 'Content-Type: application/json' \
    -d "{\"path\":\"$VID\",\"every_sec\":10}" \
    | python -c 'import sys,json;d=json.load(sys.stdin);print(d["verdict"], d["frames_with_dog"],"/",d["sampled"])'
done
```

**素材要专挑难的。** 按已经量到的，难度从高到低是：

1. **俯拍 + 狗占画幅小**（天花板机位、狗在远处沙发/垫子上）——这一条最致命，
   nano 就是栽在这儿
2. **蜷成一团睡着**——轮廓不像 COCO 里那种四条腿站着的狗
3. 笼子栏杆挡着、白毛狗贴浅色背景

全挑"狗在画面中央、站着、占半个屏"的话，两个型号都会全对，等于没验。

判据：只要出现「小的说 no_dog/mostly_empty、大的说 has_dog」，就是小的不够。
反过来不要紧（误报，人点进去看一眼就排掉）。


## 画面找片段（视觉大模型走 API）

每加一个新类别（舔、啃、蹭……）都得有人从 24 小时视频里翻片段。这里让模型来翻：
一句话描述要的行为，把视频里像的那几秒挑出来，平台按视频时间写成候选，人只看这几段。

```
画面里有狗 + 狗在动（YOLO 框 + 帧差）      本地，不花钱，把空镜和睡觉筛掉
  → 切成 6 秒一窗，裁出狗那一块             720p 俯拍狗只占 100x50 像素，不裁模型看不清
  → 每窗抽 6 帧问 Claude（API）             按段计费，max_clips 封顶
  → 相邻同类合并成片段，带类别/部位/置信度
```

**用哪家、哪个模型、key，在平台「大模型 API」页配**（Claude / GPT / 豆包 / Gemini / 本地 vLLM），
每次请求随 `llm` 字段带过来，这边不存 key。老方式（只配环境变量里的 Claude key）也还能用：

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> vision_service/.env    # 不进 git
./vision_service/run.sh down && ./vision_service/run.sh -d
curl -s localhost:8385/api/v1/seek/status                     # available 要是 true
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 空 | 不配 = 这一项关着，别的不受影响 |
| `SEEK_MODEL` | `claude-opus-5` | 想省钱可换 `claude-sonnet-5` |
| `SEEK_CONCURRENCY` | `4` | 同时问几段 |

```
GET  /api/v1/seek/status
POST /api/v1/llm/test  {llm:{provider, model, api_key, base_url}}   → {ok, latency_ms, reply, error}
POST /api/v1/seek   {path, labels:[{name, description, parts}], llm?, max_clips, dry_run, start_s, end_s, ...}
                    → {segments:[{start_s, end_s, label, body_part, confidence, note}], windows, stats}
```

`dry_run=true` 只做本地筛选、不调 API，`stats.clips_candidate` 就是会送多少段。
`stats.usage.est_usd` 是按 token 数估的花费，数量级用，账以 Anthropic 后台为准。

平台那边：项目行「画面找片段」按钮，先「预览」看会送多少段，再真跑；结果进「疑似片段」
（reason=vision），确认走跟疑似抓挠一样的通道。时间对齐靠"视频 0 秒 = IMU CSV 第一行"。

一小时 720p 视频的量级：本地筛选一两分钟（狗检测每秒一帧）；狗在场且在动的窗
一般一两百个，默认 `max_clips=120` 封顶；每段 6 张 512px 图约 3000 token，
Opus 5 一段不到 2 美分，一个视频最多一两美元。

### 本地起模型（5090 32G 单卡）

`provider=local` 就是一个 OpenAI 兼容口，vLLM / SGLang / Ollama 都行。建议 vLLM：

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --port 8386 --max-model-len 8192 --gpu-memory-utilization 0.6
```

`--gpu-memory-utilization 0.6` 给 SAM/YOLO 留显存（它们在同一张卡上）。换模型就 Ctrl-C 重起一个，
显存立刻释放；平台「大模型 API」页 local 那一行改成对应的模型名即可。

## 画面向量索引：以图搜图 / 一句话搜（免费、瞬间）

跟找片段互补：找片段每个窗都要问一次大模型，贵；索引建一次，之后任何新问题都是向量比对。
**先框狗再算向量**（整帧算出来的是房间不是狗），每秒一帧过 SigLIP，每个视频一个 npz
放 `vision_service/index/`（一小时约 5MB）。模型只在本地跑。

```bash
pip install transformers pillow          # run.sh 会自动装；torch 同上自己装
curl -s localhost:8385/api/v1/embed/status   # available / model / indexed_videos
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `EMBED_MODEL` | `google/siglip-base-patch16-224` | `./up.sh deploy` 会先跑 `vision_service/get_weights.sh` 把权重下到 `models/vision/` 并自动写进 .env；三个国内外源都不通就按它提示的从别的电脑拷 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内直连 huggingface.co 常卡死，默认走镜像；能直连就在 .env 里改回官方 |
| `EMBED_INDEX_DIR` | `vision_service/index` | 索引文件放哪 |
| `EMBED_DEVICE` | 跟 `SAM_DEVICE` | 没卡退回 cpu |
| `EMBED_MASK_BG` | `1` | 算向量前先把狗**抠出来**、背景涂灰（见下） |
| `SEG_WEIGHTS` | `models/vision/yolo/yolo26x-seg.pt` | 抠狗用的分割权重，跟检测一样 ultralytics 按文件名自动下 |

```
POST /api/v1/embed/build    {path, every_sec, force}     → {n, cached, seconds}
POST /api/v1/embed/indexed  {paths}                      → {path: bool}
POST /api/v1/embed/search   {text | ref:{path,t}, paths, top_k, min_score, gap_s, exclude_self_s}
                            → {hits:[{path,t,score}], segments:[{path,start_s,end_s,score,n}], missing}
```

平台那边：项目页「建画面索引」→ 工作台「疑似片段」里「找相似」（当前帧或一句英文）→
候选 reason=similar。

### 先抠狗再算向量（2026-09-19）

狗只占画面一角，按框裁出来的那一块一大半是花砖地、门框、笼子。向量里这些
"共同背景"的分量比狗的姿态还大——去均值能压一部分，压不干净：狗挪到另一块
地砖上分数就乱。所以建索引和查询都先用 YOLO 分割版把狗抠出来，背景涂 114 灰
（边缘羽化几个像素），再送 SigLIP。检测框 / 姿态照旧用原图。

- 分割和检测同一家模型，一帧几十毫秒；不用 SAM（一帧几百毫秒，一路几千帧扛不住）
- 分割模型没加载（权重下不动）→ 自动退回不抠，`/embed/status` 的 `mask` 里报出来
- 索引 meta 记着抠没抠；开关变了、模型上线了，`建画面索引` 会自动重建（不用勾"重建"）
- 「先看命中」的缩略图（狗框那一块）显示的就是抠完的那张——看到的和拿去比的是同一张

检测那边同时改了：几个 COCO 类都算"狗"（dog / cat / bear …），模型默认按类分开做
NMS，一只狗会框出 dog + bear 两个框。现在跨类 NMS，再把几乎套在一起的框合掉。

### 建索引 / 找片段的速度

一小时 720p 视频原来 45 秒左右，瓶颈是 cv2 逐帧 grab（CPU）和狗检测一张张送 GPU。现在：

- 有 ffmpeg 就用它解码抽帧（多线程，有卡走 NVDEC），`DECODE_FFMPEG=0` 退回 cv2，`DECODE_HWACCEL=0` 不用 NVDEC
- 狗检测一批 `DETECT_BATCH`（默认 16）帧一起送 GPU
- 平台那边建索引默认 3 路并行送（`VISION_INDEX_CONCURRENCY`），GPU 有锁，解码各自并行

`sudo apt install ffmpeg` 装上就生效，不用改配置。

### 找片段走索引

建过画面索引的视频，「画面找片段」不再解码、不再检测：每秒有没有狗、动没动直接从索引拿
（动作量 = 相邻两秒向量距离，门槛 `SEEK_INDEX_MOTION_MIN`，默认 0.06），预览秒出；
真跑时只把选中的窗那几秒用 ffmpeg 定位抽出来、按索引里的框裁狗再送模型。
所以顺序是：先「建画面索引」，再「画面找片段」。
