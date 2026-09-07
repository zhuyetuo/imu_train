"""
label_service 的配置——全部走环境变量，跟 run_review_bins_all_days.sh 用的
那套变量名尽量对齐，命令行怎么配、服务就怎么配，两边一致：

  LABEL_MODEL       模型路径，支持通配符（跟 run_review_bins_all_days.sh 的 MODEL 一样），
                    默认是当前在用的 drop_window rf 模型，换模型改这里或者传环境变量
  DEVICE_HZ         样本 CSV 的采样率（默认 50）
  RESAMPLE_METHOD   降采样算法 poly / training_match（默认 training_match）
  TARGET_LABELS     逗号分隔（默认 活动,睡觉,抓挠,未佩戴,甩身体）
  NAS_ROOT          NAS 根目录，/infer 里的 path 是相对它的相对路径（默认 /home/toky/ai_data）
  LABEL_SERVICE_PORT  监听端口（默认 8383）
  LABEL_JOBS_DIR    训练任务状态/日志落盘目录（默认 label_service/jobs，gitignore）
  LABEL_LOG_DIR     日志目录（默认 label_service/logs，按天切、留 14 天，见 logging_setup.py）
  MATERIAL_ROOT     算法任务素材库 NAS 挂载点（默认 /home/toky/算法任务素材库），牙齿照片在它的
                    口腔验证/ 子目录下；/tooth/detect 里的 path 可以相对它，也可以相对 NAS_ROOT
  TOOTH_WEIGHTS     牙齿 YOLO 权重（默认 tooth_health/data/runs/tooth_detect/weights/best.pt，不在仓库里）
  TOOTH_CONF / TOOTH_IMGSZ  检测阈值 0.5 / 输入尺寸 960，跟 tooth_health/code/web_app.py 默认一致
  LABEL_INFER_WORKERS  推理进程数（默认 CPU 核数-2，跟 run_review_bins_all_days.sh 的 WORKERS=-1 一个意思）
"""

import glob
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


MODEL_GLOB       = _env("LABEL_MODEL", "results/processed_2026_8_11-2026_8_27_raw_missing_drop_window/16hz_remap_custom_3class/rf/*.pkl")
DEVICE_HZ        = int(_env("DEVICE_HZ", "50"))
RESAMPLE_METHOD  = _env("RESAMPLE_METHOD", "training_match")
TARGET_LABELS    = [t.strip() for t in _env("TARGET_LABELS", "活动,睡觉,抓挠,未佩戴,甩身体").split(",") if t.strip()]
NAS_ROOT         = _env("NAS_ROOT", "/home/toky/ai_data")
PORT             = int(_env("LABEL_SERVICE_PORT", "8383"))
JOBS_DIR         = _env("LABEL_JOBS_DIR", os.path.join(REPO_ROOT, "label_service", "jobs"))
LOG_DIR          = _env("LABEL_LOG_DIR", os.path.join(REPO_ROOT, "label_service", "logs"))
MATERIAL_ROOT    = _env("MATERIAL_ROOT", "/home/toky/算法任务素材库")
TOOTH_WEIGHTS    = _env("TOOTH_WEIGHTS", os.path.join(REPO_ROOT, "tooth_health", "data", "runs", "tooth_detect", "weights", "best.pt"))
TOOTH_CONF       = float(_env("TOOTH_CONF", "0.5"))
TOOTH_IMGSZ      = int(_env("TOOTH_IMGSZ", "960"))
INFER_WORKERS    = int(_env("LABEL_INFER_WORKERS", "0")) or max(1, (os.cpu_count() or 2) - 2)


def resolve_model_path() -> str:
    """跟 run_review_bins_all_days.sh 里 MODEL 通配符的规则一样：
    必须恰好匹配一个文件，0 个或多个都报错，不猜。"""
    pattern = MODEL_GLOB if os.path.isabs(MODEL_GLOB) else os.path.join(REPO_ROOT, MODEL_GLOB)
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise RuntimeError(f"LABEL_MODEL 通配符 {pattern} 没有匹配到任何文件")
    if len(matches) > 1:
        raise RuntimeError(f"LABEL_MODEL 通配符 {pattern} 匹配到多个文件，请写具体一点: {matches}")
    return matches[0]
