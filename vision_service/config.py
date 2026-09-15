"""
vision_service 的配置，全部走环境变量，风格跟 label_service/config.py 一致。

  MATERIAL_ROOT       素材库 NAS 挂载点（默认 /home/toky/alg_material），照片路径相对它
  VISION_SERVICE_PORT 监听端口（默认 8385；label_service 是 8383，别撞上）
  SAM_CHECKPOINT      SAM 2.1 权重（默认 vision_service/weights/sam2.1_hiera_base_plus.pt，不在仓库里）
  SAM_MODEL_CFG       SAM 2.1 的 config 名（默认 configs/sam2.1/sam2.1_hiera_b+.yaml，由 sam2 包提供）
  SAM_DEVICE          cuda / cpu（默认 cuda，没有卡会自动退回 cpu）
  VISION_WARMUP       启动时预热模型（默认 1；设 0 退回懒加载，第一刀要多等十几秒）
  VIDEO_ROOT          采集视频的 NAS 挂载点（默认 /home/toky/ai_data），扫描路径相对它
  DOG_WEIGHTS         画面狗检测的 COCO 预训练权重（默认 yolo26x.pt，实测 nano 在夜里红外上漏 98%）
  VISION_LOG_DIR      日志目录（默认 vision_service/logs）

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
DOG_WEIGHTS    = _env("DOG_WEIGHTS", "yolo26x.pt")
LOG_DIR        = _env("VISION_LOG_DIR", os.path.join(HERE, "logs"))
