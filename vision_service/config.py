"""
vision_service 的配置，全部走环境变量，风格跟 label_service/config.py 一致。

  MATERIAL_ROOT       素材库 NAS 挂载点（默认 /home/toky/alg_material），照片路径相对它
  VISION_SERVICE_PORT 监听端口（默认 8385；label_service 是 8383，别撞上）
  SAM_CHECKPOINT      SAM 2.1 权重（默认 vision_service/weights/sam2.1_hiera_base_plus.pt，不在仓库里）
  SAM_MODEL_CFG       SAM 2.1 的 config 名（默认 configs/sam2.1/sam2.1_hiera_b+.yaml，由 sam2 包提供）
  SAM_DEVICE          cuda / cpu（默认 cuda，没有卡会自动退回 cpu）
  VISION_WARMUP       启动时预热模型（默认 1；设 0 退回懒加载，第一刀要多等十几秒）
  VIDEO_ROOT          采集视频的 NAS 挂载点（默认 /home/toky/ai_data），扫描路径相对它
  DOG_WEIGHTS         画面狗检测的 COCO 预训练权重（默认 yolo26n.pt；换型号改这个就行）
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
# 为什么是 yolo26 的 nano（2026-09-14 定）：
#
# 一句话——新架构在小模型这一档提升最大，yolo26n 很可能比上一代的 x 还强，
# 同时又省显存（这张卡还要同时挂着 SAM）。
#
# 这里有个要记下来的取舍。这一步**要的是召回不是速度**：
#
#   漏掉一只狗 → 这段被标成「没狗」→ 人直接跳过一整段真有素材的视频
#   多报一只   → 人点进去看一眼发现是空的
#
# 而且是按时间采样、不是逐帧（一小时才 720 帧），最大的模型也就几十秒跑完——
# 速度在这个场景里几乎不花钱。按这个道理本来该选最大那档。选 n 是赌新架构
# 的小模型已经够用。**这个赌要拿真实素材验**，验法见 vision_service/README.md
# 「验一下 nano 够不够」那一节：同一批片段用 n 和 m/l 各扫一遍，比 verdict。
# 出现"n 说没狗、大的说有狗"的片段，就说明赌输了，把这里换成大一档。
#
# 换型号只要改这个环境变量，代码一行不用动——ultralytics 按文件名解析，
# 本地没有就自己下。
DOG_WEIGHTS    = _env("DOG_WEIGHTS", "yolo26n.pt")
LOG_DIR        = _env("VISION_LOG_DIR", os.path.join(HERE, "logs"))
