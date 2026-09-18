"""
vision_service 的配置，全部走环境变量，风格跟 label_service/config.py 一致。

  MATERIAL_ROOT       素材库 NAS 挂载点（默认 /home/toky/alg_material），照片路径相对它
  VISION_SERVICE_PORT 监听端口（默认 8385；label_service 是 8383，别撞上）
  SAM_CHECKPOINT      SAM 2.1 权重（默认 vision_service/weights/sam2.1_hiera_base_plus.pt，不在仓库里）
  SAM_MODEL_CFG       SAM 2.1 的 config 名（默认 configs/sam2.1/sam2.1_hiera_b+.yaml，由 sam2 包提供）
  SAM_DEVICE          cuda / cpu（默认 cuda，没有卡会自动退回 cpu）
  VISION_WARMUP       启动时预热模型（默认 1；设 0 退回懒加载，第一刀要多等十几秒）
  VIDEO_ROOT          采集视频的 NAS 挂载点（默认 /home/toky/ai_data），扫描路径相对它
  DOG_WEIGHTS         画面狗检测的 COCO 预训练权重（默认 models/vision/yolo/yolo26x.pt，实测 nano 在夜里红外上漏 98%）
  VISION_LOG_DIR      日志目录（默认 vision_service/logs）
  ANTHROPIC_API_KEY   「画面找片段」用的 Claude API key（不配 = 这一项关着，别的不受影响）
  SEEK_MODEL          用哪个模型（默认 claude-opus-5）
  SEEK_CONCURRENCY    同时问几段（默认 4）
  EMBED_MODEL         画面向量索引用的模型（默认 google/siglip-base-patch16-224）
  EMBED_INDEX_DIR     索引文件放哪（默认 vision_service/index，每个视频一个 npz）

为什么单独起一个服务而不是加进 label_service：那边是 IMU 推理，纯 CPU、同步
/infer + 进程池排队，进程数按 CPU 核数配的。SAM 是长时 GPU 任务，混进去会让
IMU 推理排在 GPU 任务后面。algo_service 更是线上服务，一行都不碰。
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


MATERIAL_ROOT  = _env("MATERIAL_ROOT", "/home/toky/alg_material")
PORT           = int(_env("VISION_SERVICE_PORT", "8385"))
SAM_CHECKPOINT = _env("SAM_CHECKPOINT", os.path.join(HERE, "weights", "sam2.1_hiera_base_plus.pt"))
SAM_MODEL_CFG  = _env("SAM_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_b+.yaml")
SAM_DEVICE     = _env("SAM_DEVICE", "cuda")
# 启动时预热（加载权重 + 空跑一次推理）。设 0 退回原来的懒加载：
# 第一次调用才加载，那一刀要多等十几秒。
WARMUP         = _env("VISION_WARMUP", "1") not in ("0", "false", "False", "")

# ── 画面狗检测 ────────────────────────────────────────────────────────
# 视频在另一个 NAS 上（采集数据），跟素材库（照片）不是一棵树，所以单独一个根。
# 跟 smart-label 的 nas_root 指同一个目录。
VIDEO_ROOT     = _env("VIDEO_ROOT", "/home/toky/ai_data")
# COCO 预训练权重，不训练、不微调——只要现成的 dog 类。ultralytics 找不到会自己下。
#
# 为什么是最大那档（x）——**这是实测出来的，不是选出来的**。
#
# 2026-09-14 先按"新架构小模型提升最大"的想法默认用了 yolo26n，并说好拿真实
# 素材验。2026-09-15 在狗场夜里那段上验完，赌输了，而且不是边界差异：
#
#   素材：data_raw/2026_9_12/multicam_20260912_230430647_cam1_imu1_raw.mp4
#         3329 秒（影棚，cam1），每 10 秒采一个点，共 333 个点
#
#           yolo26n   mostly_empty    5/333 个点有狗（1.5%）
#           yolo26x   has_dog       273/333 个点有狗（82%）
#
# nano 漏掉了 268 个有狗的采样点。
#
# **不是画面暗的问题**。同一批的 230020516 那段抽帧看过：室内开着灯、彩色正常、
# 平均亮度 123/255，人一眼就能看见沙发上躺着一只、门口趴着一只——nano 照样报
# 零。难的是**俯拍 + 目标小 + 蜷成一团**：摄像头吊在天花板上，沙发上那只只占
# 约 100x50 像素（画幅 1280x720，0.5% 面积），而 COCO 里的狗绝大多数是侧面/
# 正面、占画幅很大。这种姿态和视角是分布外的，小模型首先在这儿垮。
#
# 而这正好是最坏的那种错：nano 会让平台显示"这一小时大部分是空镜"，人就跳过了
# ——实际上 82% 的时间画面里有狗。这一步**要的是召回不是速度**：
#
#   漏掉一只狗 → 这段被标成「没狗」→ 人直接跳过一整段真有素材的视频
#   多报一只   → 人点进去看一眼发现是空的
#
# 代价可以忽略：按时间采样、不逐帧（一小时 720 帧），5090 上 x 也就一两分钟。
#
# 换型号只要改这个环境变量，代码一行不用动——ultralytics 按文件名解析，
# 本地没有就自己下。省显存想换小的话，**先拿夜里的素材重验一遍**再换。
# 放仓库的 models/vision/yolo/ 下跟别的模型一起管；文件不在时 ultralytics 会按文件名自动下到这个路径
DOG_WEIGHTS    = _env("DOG_WEIGHTS", os.path.join(HERE, "..", "models", "vision", "yolo", "yolo26x.pt"))
LOG_DIR        = _env("VISION_LOG_DIR", os.path.join(HERE, "logs"))

# ── 画面找片段（视觉大模型走 API，不本地起） ────────────────────────────
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "")
SEEK_MODEL        = _env("SEEK_MODEL", "claude-opus-5")
SEEK_CONCURRENCY  = int(_env("SEEK_CONCURRENCY", "4"))
# 估算花费用，$/百万 token（输入, 输出）。只是给人看个数量级，账以 Anthropic 后台为准
SEEK_PRICE_PER_M  = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# ── 画面向量索引（以图搜图 / 一句话搜）──────────────────────────────────
# SigLIP：图像和文本同一个向量空间，一份索引两种查法。权重约 400MB，HF 自动下；
# 下不动就先在能上网的机器上下好，EMBED_MODEL 指向本地目录
EMBED_MODEL     = _env("EMBED_MODEL", "google/siglip-base-patch16-224")
EMBED_DEVICE    = _env("EMBED_DEVICE", SAM_DEVICE)
EMBED_BATCH     = int(_env("EMBED_BATCH", "128"))
EMBED_INDEX_DIR = _env("EMBED_INDEX_DIR", os.path.join(HERE, "index"))

# ── 姿态关键点（以图搜图的第二路信号）────────────────────────────────
# RTMPose-m AP-10K 的 ONNX 一个文件，rtmlib + onnxruntime 跑。没权重 / 没装就自动关，
# 索引里不存姿态、搜索只用画面。get_weights.sh 会下并写进 .env
POSE_ONNX  = _env("POSE_ONNX", os.path.join(HERE, "..", "models", "vision", "pose", "rtmpose_ap10k.onnx"))
POSE_INPUT = int(_env("POSE_INPUT", "256"))
POSE_DEVICE = _env("POSE_DEVICE", SAM_DEVICE)
# 搜索时姿态相似占多少（0 = 只看画面，1 = 只看姿态）。前端可调
POSE_W = float(_env("POSE_W", "0.5"))

# ── 本地大模型（vLLM）────────────────────────────────────────────────
# 「模型服务」页一键起停。权重放 models/vision/llm/<模型名>，docker 跑时挂进容器
VLLM_MODEL      = _env("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
VLLM_PORT       = int(_env("VLLM_PORT", "8386"))
VLLM_LOCAL_ROOT = _env("VLLM_LOCAL_ROOT", os.path.join(HERE, "..", "models", "vision", "llm"))
VLLM_MAX_LEN    = int(_env("VLLM_MAX_LEN", "8192"))
# 显存留一半给别的模型（狗检测 / SigLIP / 姿态 / SAM 都在同一张卡上）
VLLM_GPU_UTIL   = float(_env("VLLM_GPU_UTIL", "0.5"))
# 额外参数，原样拼到命令行后面（比如 --quantization awq --dtype half）
VLLM_ARGS       = _env("VLLM_ARGS", "")
# 怎么跑：docker（默认，官方镜像自带 CUDA 工具链，不动主机 python）/ process（pip 装的 vllm 直接起进程）
VLLM_BACKEND    = _env("VLLM_BACKEND", "docker")
VLLM_IMAGE      = _env("VLLM_IMAGE", "vllm/vllm-openai:latest")
VLLM_CONTAINER  = _env("VLLM_CONTAINER", "imu_vllm")

# ── 解码 / 检测的速度开关 ─────────────────────────────────────────────
# 有 ffmpeg 就用它解码抽帧（多线程 + NVDEC），比 cv2 逐帧 grab 快好几倍；DECODE_FFMPEG=0 退回 cv2
DECODE_FFMPEG   = _env("DECODE_FFMPEG", "1") not in ("0", "false", "False", "")
DECODE_HWACCEL  = _env("DECODE_HWACCEL", "1") not in ("0", "false", "False", "")
# 狗检测一批送几帧（GPU 上一批 16~32 比一张张送快好几倍；显存紧就调小）
DETECT_BATCH    = int(_env("DETECT_BATCH", "32"))
# 半精度推理（GPU 上快近一倍，框差别在小数点后）
DETECT_HALF     = _env("DETECT_HALF", "1") not in ("0", "false", "False", "")
# 画面没变（狗睡着 / 空房间）就不再送检测，直接沿用上一次的框：24 小时里大半时间是静止的。
# 整帧缩到 64x36 灰度后的平均像素差（0~1），低于它算没变；0 = 关掉这个优化
STATIC_SKIP_THR = float(_env("STATIC_SKIP_THR", "0.01"))

# 找片段走画面索引时的动作量门槛（相邻两秒向量距离，0~1；跟像素帧差不是一个刻度）
SEEK_INDEX_MOTION_MIN = float(_env("SEEK_INDEX_MOTION_MIN", "0.06"))
SEEK_INDEX_MOTION_MAX = float(_env("SEEK_INDEX_MOTION_MAX", "1.0"))
