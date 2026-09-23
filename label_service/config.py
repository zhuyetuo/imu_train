"""
label_service 的配置——全部走环境变量，跟 run_review_bins_all_days.sh 用的
那套变量名尽量对齐，命令行怎么配、服务就怎么配，两边一致：

  LABEL_MODEL       模型路径，支持通配符（跟 run_review_bins_all_days.sh 的 MODEL 一样），
                    默认是当前在用的 drop_window rf 模型，换模型改这里或者传环境变量
  DEVICE_HZ         样本 CSV 的采样率（默认 50）
  RESAMPLE_METHOD   降采样算法 poly / training_match（默认 training_match）
  TARGET_LABELS     逗号分隔。**留空（默认）= 跟着当前模型自己的类别走**，
                    换了模型不用改这里。写死一串的话，模型输出「抓挠-头颈耳」
                    这种新类别时一个片段都不会出来
  NAS_ROOT          NAS 根目录，/infer 里的 path 是相对它的相对路径（默认 /home/toky/ai_data）
  LABEL_SERVICE_PORT  监听端口（默认 8383）
  LABEL_JOBS_DIR    训练任务状态/日志落盘目录（默认 label_service/jobs，gitignore）
  LABEL_LOG_DIR     日志目录（默认 label_service/logs，按天切、留 14 天，见 logging_setup.py）
  MATERIAL_ROOT     算法任务素材库 NAS 挂载点（默认 /home/toky/alg_material），牙齿照片在它的
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
# 额外挂几个模型给平台做对比，`标签=路径` 逗号分隔，路径支持通配符。
# 不配 = 只有 LABEL_MODEL 那一个，跟以前完全一样。见 registry.py。
#   LABEL_MODELS='acc3=results_acc3/*_acc3/16hz_remap_custom_3class/rf/ml_rf.pkl'
# 找不到文件的条目**跳过不报错**（实验模型在别的机器上可能没训），
# 但启动日志里会吵一句——线上标注不能因为一个实验模型没训就起不来。
EXTRA_MODELS     = _env("LABEL_MODELS", "")
DEVICE_HZ        = int(_env("DEVICE_HZ", "50"))
RESAMPLE_METHOD  = _env("RESAMPLE_METHOD", "training_match")
# 空列表 = 跟着模型的 classes 走（见 pool.py）。**不再写死**：写死的那串里
# 「甩身体」当前模型根本不会输出（等于白列），而模型真能输出的新类别
# （抓挠-头颈耳这种拆了二级的）反倒不在里面，一个片段都出不来
TARGET_LABELS    = [t.strip() for t in _env("TARGET_LABELS", "").split(",") if t.strip()]
NAS_ROOT         = _env("NAS_ROOT", "/home/toky/ai_data")
PORT             = int(_env("LABEL_SERVICE_PORT", "8383"))
JOBS_DIR         = _env("LABEL_JOBS_DIR", os.path.join(REPO_ROOT, "label_service", "jobs"))
LOG_DIR          = _env("LABEL_LOG_DIR", os.path.join(REPO_ROOT, "label_service", "logs"))
MATERIAL_ROOT    = _env("MATERIAL_ROOT", "/home/toky/alg_material")
TOOTH_WEIGHTS    = _env("TOOTH_WEIGHTS", os.path.join(REPO_ROOT, "tooth_health", "data", "runs", "tooth_detect", "weights", "best.pt"))
TOOTH_CONF       = float(_env("TOOTH_CONF", "0.5"))
TOOTH_IMGSZ      = int(_env("TOOTH_IMGSZ", "960"))
INFER_WORKERS    = int(_env("LABEL_INFER_WORKERS", "0")) or max(1, (os.cpu_count() or 2) - 2)
# 给交互式推理（工作台点「AI预标注」）留几个槽位，批量预标注最多占 INFER_WORKERS - 这个数，
# 免得标注员点一下要排在几十个批量文件后面
INFER_RESERVE    = int(_env("LABEL_INFER_RESERVE", "2"))

# 稳定版后处理参数（见 postprocess.py）。调试版 = 模型逐窗口原始输出，不受这些影响
STABLE_EVENT_LABELS      = [t.strip() for t in _env("STABLE_EVENT_LABELS", "抓挠,甩身体").split(",") if t.strip()]
STABLE_SMOOTH_WINDOWS    = int(_env("STABLE_SMOOTH_WINDOWS", "7"))
STABLE_MIN_STATE_S       = float(_env("STABLE_MIN_STATE_S", "10"))
STABLE_EVENT_ENTER       = float(_env("STABLE_EVENT_ENTER", "0.5"))    # 滞回：进入事件的概率
STABLE_EVENT_STAY        = float(_env("STABLE_EVENT_STAY", "0.25"))    # 滞回：维持在同一段的概率
STABLE_EVENT_GAP_S       = float(_env("STABLE_EVENT_GAP_S", "4"))
STABLE_SHAKE_ABSORB_S    = float(_env("STABLE_SHAKE_ABSORB_S", "3"))   # 抓挠前后多少秒内的甩身体并入抓挠
STABLE_EVENT_MIN_WINDOWS = int(_env("STABLE_EVENT_MIN_WINDOWS", "2"))
STABLE_EVENT_MIN_MEAN    = float(_env("STABLE_EVENT_MIN_MEAN", "0.45"))
STABLE_EVENT_SINGLE_CONF = float(_env("STABLE_EVENT_SINGLE_CONF", "0.85"))
# 抓挠 bout 陀螺仪 4–8 Hz 能量占比下限（0 = 不启用）。每个片段都带 spec 字段，先看
# 一批真/假抓挠的分布再定阈值，经验上真抓挠 > 0.3、误报 < 0.15
STABLE_SPECTRAL_MIN      = float(_env("STABLE_SPECTRAL_MIN", "0"))
# viterbi（稳定版 v2）切换类别的代价，对数单位；越大越不爱切换
STABLE_VITERBI_SWITCH    = float(_env("STABLE_VITERBI_SWITCH", "3.0"))
# 疑似抓挠候选（低门槛，给人工找漏检）与边界微调
# 门槛太松会失去意义：0.2 进入时一小时能抽出三百多条、绝大多数是 20~30% 的噪声，
# 人审不过来，真正值得看的那几条反而被淹掉。目标是"一小时能看完的清单"
CAND_ENTER               = float(_env("CAND_ENTER", "0.3"))
CAND_STAY                = float(_env("CAND_STAY", "0.25"))  # 低于背景噪声上沿会把边界拖长、稀释置信度
CAND_MIN_WINDOWS         = int(_env("CAND_MIN_WINDOWS", "2"))
CAND_MIN_MEAN            = float(_env("CAND_MIN_MEAN", "0.3"))   # 整段平均概率下限
# 只靠频谱拎候选：默认 0 = 关掉。实测这批设备本底在 4–8 Hz 就偏高，抽出来的几百条
# 里绝大多数模型置信度是 0%、频谱却有 0.5~0.68，阈值定在哪都是噪声。spec 照常算、
# 照常显示，只是不再单独作为候选来源；换设备想试再设成 >0
CAND_SPEC_MIN            = float(_env("CAND_SPEC_MIN", "0"))
CAND_MAX                 = int(_env("CAND_MAX", "40"))           # 每个文件最多给几条
# 一个窗口里"六轴全 0 / MISSING"的样本点占比超过这个数，就当这段没有数据：
# 预测结果里抠掉、不算有效佩戴、不进训练集。半个窗口都是空的就已经没法判了
# 疑似舔/啃候选（只靠加速度姿态，不靠模型；见 grooming.py）。跟疑似抓挠走同一条
# 确认/排除通道，label_name 是 GROOM_LABEL。目的是给「舔/啃哪个部位」攒标注：
# 人只看候选那几段，不用翻 24 小时视频。
GROOM_ENABLED            = _env("GROOM_ENABLED", "1") == "1"
GROOM_LABEL              = _env("GROOM_LABEL", "舔身体")
GROOM_MAX                = int(_env("GROOM_MAX", "40"))           # 每个文件最多给几条
GROOM_TILT_MIN_DEG       = float(_env("GROOM_TILT_MIN_DEG", "25"))  # 头偏离平时姿态多少度起算
GROOM_MOTION_MIN         = float(_env("GROOM_MOTION_MIN", "0.03"))  # 动作量下限（低于 = 趴着不动）
GROOM_MOTION_MAX         = float(_env("GROOM_MOTION_MAX", "0.35"))  # 动作量上限（高于 = 走/跑）
GROOM_MIN_S              = float(_env("GROOM_MIN_S", "3"))          # 短于这个的不要
GROOM_GAP_S              = float(_env("GROOM_GAP_S", "2"))          # 断开不超过这么久算同一段
MISSING_MIN_RATIO        = float(_env("MISSING_MIN_RATIO", "0.5"))
REFINE_MARGIN_S          = float(_env("REFINE_MARGIN_S", "1.0"))
REFINE_RATIO             = float(_env("REFINE_RATIO", "0.3"))
REFINE_ENABLED           = _env("REFINE_ENABLED", "1") == "1"


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
