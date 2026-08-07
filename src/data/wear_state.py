"""
佩戴状态判断：松动检测 + 未佩戴/静置检测。设计见 docs/wear_state_detection.md。

两条独立支路，都是规则/信号处理判断，不是训练出来的分类模型：

1. 松动检测（loose collar）：只在"稳定步态"窗口内看朝向相对重力的稳定性有没有随时间
   漂移/变宽——项圈松动的信号是"相对于狗真实运动的额外晃动"，脱离稳定的参考运动
   （走路/小跑）就没法判断"晃动是不是异常"。

2. 未佩戴/静置检测（off-body）：在"宏观运动能量接近0"的候选窗口内，看有没有心跳/
   呼吸引起的微幅周期性振动——脖子佩戴天然贴近颈动脉/气管，即使睡得很沉也该有微振动；
   完全没有则怀疑设备被摘下静置（放桌上/充电）。

所有阈值都是待用真实数据校准的经验值，标了 TODO_CALIBRATE。
"""
import numpy as np
from scipy.signal import butter, filtfilt

from gravity_align import raw_tilt

# ── 松动检测参数 ────────────────────────────────────────────────────────
GAIT_SUBSEG_SEC = 2.0                    # 稳定步态窗口内切子段算漂移的子段长度
LOOSE_WOBBLE_STD_THRESHOLD_RAD = 0.06    # TODO_CALIBRATE：子段间朝向漂移标准差超过此值判松动
LOOSE_EVENT_WINDOW_MIN = 10              # PM文档：10分钟窗口
LOOSE_EVENT_MIN_MINUTES = 3              # PM文档：窗口内松动信号累计>3分钟记1次事件

# ── 未佩戴/静置检测参数 ──────────────────────────────────────────────────
MACRO_STILL_WINDOW_SEC = 30              # 判断"宏观运动能量接近0"的观察窗口
MACRO_STILL_ACC_STD_THRESHOLD = 0.05     # TODO_CALIBRATE：窗口内加速度标准差低于此值才算候选静止
BIO_BAND_LOW_HZ = 0.15                   # 呼吸(~0.25-0.5Hz)+心率(~1-2.3Hz)合并频段下限
BIO_BAND_HIGH_HZ = 3.0                   # 合并频段上限，留一点余量覆盖体型偏小、心率偏快的狗
BIO_SIGNAL_ENERGY_THRESHOLD = 3e-5       # TODO_CALIBRATE：带通后信号方差超过此值判定"有生物信号"


# ── 松动检测 ──────────────────────────────────────────────────────────

def tilt_drift(acc, hz, subseg_sec=GAIT_SUBSEG_SEC):
    """
    给一段"稳定步态"窗口的原始加速度，返回子段平均朝向随时间漂移的程度（pitch/roll 标准差，弧度）。
    项圈松动时，狗每完成几个步态周期，项圈相对颈部的贴合角度就可能悄悄变化，
    表现为子段平均朝向逐渐漂移/扩散，比正常佩戴时更分散。
    """
    pitch, roll = raw_tilt(acc)[:, 0], raw_tilt(acc)[:, 1]
    subseg_len = max(1, int(subseg_sec * hz))
    n_sub = len(acc) // subseg_len
    if n_sub < 2:
        return 0.0
    pitch_means = [pitch[i * subseg_len:(i + 1) * subseg_len].mean() for i in range(n_sub)]
    roll_means = [roll[i * subseg_len:(i + 1) * subseg_len].mean() for i in range(n_sub)]
    return float(np.std(pitch_means) + np.std(roll_means))


def detect_loose_window(acc, hz, is_stable_gait):
    """单个窗口的松动判定。is_stable_gait：外部传入，标记这段是否是稳定行走/小跑（PM文档要求
    只在这类窗口里判断，静止/剧烈运动/转身时的朝向变化不能算松动信号）。"""
    if not is_stable_gait:
        return False, None
    drift = tilt_drift(acc, hz)
    return drift > LOOSE_WOBBLE_STD_THRESHOLD_RAD, drift


def aggregate_loose_events(loose_flags_per_window, window_sec, hz_windows_per_min=None):
    """
    loose_flags_per_window: 按时间顺序排好的 (timestamp_sec, is_loose) 列表。
    PM规则：10分钟滑窗内松动信号累计时长>3分钟，记1次松动事件。
    简化实现：按 window_sec 累计"松动窗口"的总时长，落在任意10分钟滑窗内超过3分钟即触发。
    """
    if not loose_flags_per_window:
        return []
    times = np.array([t for t, _ in loose_flags_per_window])
    flags = np.array([f for _, f in loose_flags_per_window])
    events = []
    win = LOOSE_EVENT_WINDOW_MIN * 60
    min_loose = LOOSE_EVENT_MIN_MINUTES * 60
    i = 0
    n = len(times)
    while i < n:
        win_end = times[i] + win
        j = i
        while j < n and times[j] < win_end:
            j += 1
        loose_sec = flags[i:j].sum() * window_sec
        if loose_sec > min_loose:
            events.append((float(times[i]), float(win_end), float(loose_sec)))
            i = j  # 跳到窗口外，避免同一段被反复计数成多个事件
        else:
            i += 1
    return events


# ── 未佩戴/静置检测（心跳/呼吸微振动） ──────────────────────────────────

def is_macro_still(acc, hz, window_sec=MACRO_STILL_WINDOW_SEC):
    """宏观运动能量是否接近0——只有这类窗口才需要进一步做生物信号判断，明显在动的窗口不用跑。"""
    if len(acc) < hz * 2:
        return False
    mag = np.linalg.norm(acc, axis=1)
    return float(np.std(mag)) < MACRO_STILL_ACC_STD_THRESHOLD


def bio_signal_energy(acc, hz):
    """带通滤波后，呼吸/心率频段内的信号方差——衡量有没有周期性微振动。"""
    mag = np.linalg.norm(acc, axis=1)
    mag = mag - mag.mean()
    nyq = hz / 2.0
    low = min(BIO_BAND_LOW_HZ / nyq, 0.99)
    high = min(BIO_BAND_HIGH_HZ / nyq, 0.999)
    if high <= low:
        return 0.0  # 采样率太低，覆盖不到目标频段
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, mag)
    return float(np.var(filtered))


def classify_wear_state(acc, hz, is_stable_gait=False):
    """
    单个窗口的佩戴状态判断。返回 (wear_state, detail)。
    wear_state ∈ {"worn_active", "worn_resting", "off_body_or_static", "uncertain"}
    （对应 PM 文档 schema 里的 wear_state 字段，off_body_or_static 覆盖"未佩戴/静置/充电"，
    这三者目前单靠IMU区分不开，等温湿度传感器上线后再细分——见文档。）
    """
    if not is_macro_still(acc, hz):
        return "worn_active", {"macro_still": False}

    energy = bio_signal_energy(acc, hz)
    has_bio_signal = energy > BIO_SIGNAL_ENERGY_THRESHOLD
    state = "worn_resting" if has_bio_signal else "off_body_or_static"
    return state, {"macro_still": True, "bio_signal_energy": energy, "has_bio_signal": has_bio_signal}
