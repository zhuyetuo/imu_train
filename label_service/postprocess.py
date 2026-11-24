"""
"稳定版"后处理：把逐窗口的原始预测（调试版，模型逐窗口 argmax 的真实输出）
整理成人看着可信的片段。原始输出里活动/睡觉这类"状态"经常一秒一换来回闪，
抓挠/甩身体这类"事件"又常常是孤立的单窗口噪声，或者一次完整抓挠中间被一两个
跳成活动/甩身体的窗口切碎——这里分两套规则处理。

两种算法（算法名 = /infer 的 mode）：

stable —— 规则版
  状态类（EVENT_LABELS 之外的全部）
    1. 状态类概率向量做滑动平均（SMOOTH_WINDOWS 个窗口）再取 argmax；
    2. 连续同状态短于 MIN_STATE_S 的段并入相邻更长的那段，反复直到没有碎片。
  事件类（抓挠 / 甩身体）
    1. 双阈值滞回：该事件概率 ≥ EVENT_ENTER 进入，进入后只要 ≥ EVENT_STAY 就继续
       算同一段（中间 argmax 跳成活动的窗口抓挠概率通常还有 0.3，不会被切断）；
    2. 同类事件间隔 ≤ EVENT_GAP_S 的合成一个 bout；
    3. 抓挠 bout 前后 SHAKE_ABSORB_S 内的甩身体窗口并进抓挠（抓完常甩一下，
       模型也爱把抓挠的剧烈段判成甩身体）；
    4. bout 要么窗口数 ≥ EVENT_MIN_WINDOWS 且平均概率 ≥ EVENT_MIN_MEAN，要么最高
       概率 ≥ EVENT_SINGLE_CONF，否则丢掉；
    5. 可选：窗口自带的陀螺仪 4–8 Hz 能量占比（spec，见 pool.py）低于 SPECTRAL_MIN
       的抓挠 bout 丢掉——抓挠是后腿高频往复，频谱上有明显峰，模型之外的独立证据。

viterbi —— 稳定版 v2
  把每个窗口的各类概率当发射概率，切换类别付一个固定代价（VITERBI_SWITCH，
  对数单位），动态规划求整条时间轴代价最小的标签序列。对所有类别统一生效，
  参数只有一个；事件类出来之后同样走上面 2–5 的合并/过滤。

输入就是 /infer 返回的 windows（ts/label/conf/probs/spec），不碰模型、不碰特征，
调试版原样保留，几个版本用同一次推理的结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

_TS_FMT = "%Y-%m-%d %H:%M:%S.%f"


#: 版本串后缀：加上它就是"同样的解码，但不让抓挠吞并甩身体"。
#
# 为什么要有这个：那条吞并规则（shake_absorb_s）是为了修"模型爱把抓挠的
# 剧烈段判成甩身体"而加的，但它反过来会把真的甩身体并进抓挠。到底哪个
# 更划算，只能拿真实样本并排跑一遍看——所以做成两个版本，而不是一个配置。
NOSHAKE_SUFFIX = "_noshake"

#: 版本串后缀：同样的解码（viterbi），**只改片段起止**——先平移掉 v2 的系统偏差，再按抓挠节律能量对齐。
#: 片段的个数、类别一个都不变，所以跟 v2 并排比时差的只有起止。做成版本（不是配置）的理由同 _noshake。
PRECISE_SUFFIX = "_precise"


def is_precise(algo: str | None) -> bool:
    return bool(algo) and str(algo).endswith(PRECISE_SUFFIX)


#: 版本串后缀：同样的解码（viterbi），**只把抓挠的段内平均概率下限收紧**（数值见 StableParams.strict_*）。要的是"别把不是抓挠的报成抓挠"，起止、长短差一些都能接受。
#: 别的类别、甩身体、疑似抓挠候选一律不变——被严格版筛掉的抓挠段，会照常落进疑似抓挠候选里给人看。
#: 做成版本（不是改默认配置）的理由同 _noshake：结果按 (样本, 模型, 版本) 存，能跟 v2 并排比，老项目也不受影响。
STRICT_SUFFIX = "_strict"


def is_strict(algo: str | None) -> bool:
    return bool(algo) and str(algo).endswith(STRICT_SUFFIX)


@dataclass
class StableParams:
    event_labels: tuple[str, ...] = ("抓挠", "甩身体")
    scratch_label: str = "抓挠"
    shake_label: str = "甩身体"
    smooth_windows: int = 7        # 状态概率滑动平均的窗口数（stride 1s 时 ≈ 7 秒）
    min_state_s: float = 10.0      # 状态片段最短时长，短于这个并入邻居
    event_enter: float = 0.5       # 滞回：进入事件的概率门槛
    event_stay: float = 0.25       # 滞回：进入后维持在同一段的概率门槛
    event_gap_s: float = 4.0       # 同类事件之间隔多久以内合成一个 bout
    shake_absorb_s: float = 3.0    # 抓挠 bout 前后多少秒内的甩身体并进抓挠
    event_min_windows: int = 2     # bout 至少几个窗口
    event_min_mean: float = 0.45   # bout 内该事件的平均概率下限
    # 只对抓挠的平均概率下限（0 = 跟 event_min_mean 一样）。严格版用它，甩身体不跟着变
    scratch_min_mean: float = 0.0
    event_single_conf: float = 0.85  # 单窗口也保留的最高概率下限
    spectral_min: float = 0.0      # 抓挠 bout 的平均频谱占比下限，0 = 不启用
    # 「真的在挠」的物理证据门限（0 = 不启用）。CNN 只用三轴加速度：设备没戴、
    # 躺在笼子里被碰、或者信号掉点，加速度都会抖，很容易被判成抓挠。真实抓挠
    # 是身体在动：陀螺仪 50–360°/s、几乎不丢帧、模长 1.1–2.5g；而误报样本
    # 陀螺仪只有 ~1°/s（设备根本没转动）、30% 的行是 0、模长 0.6g（低于重力）。
    gyro_std_min: float = 0.0      # 窗口陀螺仪幅值 std 下限
    missing_max: float = 0.0       # 窗口缺数据（空值/六轴全 0）比例上限
    amag_min: float = 0.0          # 窗口加速度模长均值下限（佩戴时≈1g）
    amag_max: float = 0.0          # 窗口加速度模长均值上限（抓挠 1.1–2.5g）
    # 三轴加速度在 2–10 Hz 的能量占比下限。抓挠是后腿高频往复（真抓挠 0.37–0.93），
    # 信号不好/设备被碰的波动集中在 1 Hz 以下（实测误报 0.16）
    acc_rhythm_min: float = 0.0
    # 高置信度豁免 acc_rhythm_min：当模型段均值 ≥ rhythm_exempt_mean 或峰值 ≥ rhythm_exempt_max，
    # 且陀螺仪/掉点/模长门限都已通过（证明设备戴着且真的在动）时，不再被单一项频谱节律占比误杀。0 = 不豁免
    rhythm_exempt_mean: float = 0.85
    rhythm_exempt_max: float = 0.90
    # 严格版（viterbi_strict）的抓挠门槛。2026-11-24 在 train_v2 4 s 窗口模型（训练记录 #61，job60）的
    # 5 折留出预测上扫出来（experiments/claude_2026_11/evt/evt_eval.py）：事件级（重叠 ≥ 0.5 s 算对）
    # 查全 0.892 → 0.851，确定查准（只算人看过的时间）0.909 → 0.946，全部预测里报对的比例 0.595 → 0.734。
    # 陀螺仪 / 节律门槛取 max(线上值, 这里)，只收紧不放松。**默认 0 = 沿用线上值**：评估里的陀螺仪 std /
    # 节律占比是按模型的 20 Hz 算的，线上按设备原始采样率（50–220 Hz）算，节律占比整体偏低，评估里扫出来的
    # 0.45 搬到线上严得多（11-18 抽 12 个文件，8 段抓挠删了 7 段）。要用得先按线上口径重新量
    strict_scratch_min_mean: float = 0.7
    strict_gyro_std_min: float = 0.0
    strict_acc_rhythm_min: float = 0.0
    viterbi_switch: float = 3.0    # viterbi 切换类别的代价（对数单位）
    # 短的「未佩戴」碎片并回旁边的静止：狗睡着 / 趴着不动时，模型偶尔冒出几十秒未佩戴——
    # 戴着的项圈总有呼吸、心跳带来的细微晃动，跟放在桌上不完全一样，但模型分不太清。
    # 真摘下来（充电）一般是很长一段，所以短于这个秒数、且挨着静止类的未佩戴改成那段静止。
    # 0 = 不启用
    notworn_min_s: float = 300.0
    # 疑似抓挠候选（给人工审核找漏检用，不进正式片段）
    #
    # 门槛定得太松会失去意义：0.2 进入时，一小时能抽出三百多条，绝大多数是
    # 20%~30% 的噪声，人根本审不过来，真正值得看的那几条反而被淹掉。这里的
    # 目标不是"把所有可能都列出来"，而是"给人一份一小时能看完的清单"。
    cand_enter: float = 0.3        # 低门槛滞回：进入（比正式的 0.5 低，但不能太低）
    # 维持门槛不能比背景噪声低太多：背景在 0.05~0.28 晃时，stay=0.2 会让一段真抓挠
    # 顺着噪声一路延伸出去，边界拖长、整段平均被稀释（实测 0.57 掉到 0.42），
    # 反而在按置信度排序时沉下去。0.25 刚好卡在噪声上沿之上
    cand_stay: float = 0.25        # 低门槛滞回：维持
    cand_min_windows: int = 2
    cand_min_mean: float = 0.3     # 整段的平均概率也要够，挡掉"就一个窗口冒了一下"
    # 只靠频谱把一段拎成候选：默认 0 = 关掉（见 scratch_candidates 里的说明，
    # 实测这批设备的本底就偏高，抽出来几乎全是模型置信 0% 的噪声）
    cand_spec_min: float = 0.0
    # 每个文件最多给这么多条（在 app.py 里按置信度从高到低裁，并记日志说明丢了多少）
    cand_max: int = 40
    # 边界微调：在片段起止各 ±refine_margin_s 内按陀螺仪能量找真正的起止
    refine_margin_s: float = 1.0
    refine_ratio: float = 0.3      # 能量高于片段内中位数 × 这个比例才算"在动"
    # 精准版（viterbi_precise）起止：先平移校正 v2 的系统偏差，再按陀螺仪 3–9 Hz（抓挠节律）能量对齐。
    # 数值来自 2026-11-18 的 5 折交叉验证（experiments/claude_2026_11/boundary_eval.py），见 precise_boundaries
    precise_ratio: float = 0.5        # 3–9 Hz 能量高于片段内中位数 × 这个比例才算"在抓"
    precise_pre_s: float = 1.0        # 起点往前最多找多少秒
    precise_start_in_s: float = 2.0   # 起点往里最多收多少秒
    precise_end_in_s: float = 2.0     # 终点往里最多收多少秒
    precise_end_out_s: float = 1.0    # 终点往后最多找多少秒
    precise_smooth_bins: int = 3      # 包络先做几格（0.1 s 一格）滑动平均
    precise_sustain_bins: int = 3     # 连续几格高于阈值才算开始 / 结束


def _parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    for f in (_TS_FMT, "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return None


def _fmt_ts(t: datetime) -> str:
    return t.strftime(_TS_FMT)[:-3]


def _zones(ts: list[datetime], window_s: float, stride_s: float, label_mode: str) -> list[tuple[datetime, datetime]]:
    """每个窗口在时间轴上负责的区间，跟 infer_csv_scratch.window_zone 一致。

    label_mode=segment（新版流程 train_v2 训的模型）：整体往后挪 (window_s - stride_s)/2。
    这种模型的窗口是从标注片段**里面**切的，一个窗口判成抓挠 ≈ 这个窗口大半落在抓挠里，
    代表的是窗口中间那段时间，不是窗口起点。按起点算的话片段整体早半个窗口——
    pool_20261107 上连续推理 + 稳定版 v2 跟人标的抓挠比，开始时间中位数早 2.3 秒
    （src/eval/scratch_event_cv.py）。挪的是整条时间轴，窗口之间的间隔不变，
    合并 / 吞并 / 门槛这些判断一个都不受影响，只是片段的起止时间对上了。
    板上（algo_tinyml tm_post）同理：喂给 tm_post 的窗口时间戳要加同样的偏移。"""
    out = []
    n = len(ts)
    shift = timedelta(seconds=(window_s - stride_s) / 2) if label_mode == "segment" else timedelta(0)
    for k in range(n):
        if label_mode == "center":
            c = ts[k] + timedelta(seconds=window_s / 2)
            out.append((c - timedelta(seconds=stride_s / 2), c + timedelta(seconds=stride_s / 2)))
        else:
            # 只有下一个窗口刚好差一个步长，才是同一段有效数据里相邻的窗口。
            # 窗口现在只在有效段内滑（先扔全 0/断连段再窗口化），跨段时下一个
            # 窗口的时间会跳很远，不能拿它当本段的结束边界，否则片段会横跨
            # 掉线段。最后一个窗口只负责到自己的数据结束
            if k + 1 < n and (ts[k + 1] - ts[k]).total_seconds() <= stride_s * 1.5:
                out.append((ts[k] + shift, ts[k + 1] + shift))
            else:
                out.append((ts[k] + shift, ts[k] + timedelta(seconds=window_s)))
    return out


def _runs(labels: list[str]) -> list[list]:
    """[[label, i0, i1], ...] 连续同标签的区间（闭区间，窗口下标）"""
    runs: list[list] = []
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])
    return runs


def _absorb_short_runs(labels: list[str], zones, min_s: float) -> list[str]:
    """把短于 min_s 的状态片段并进相邻更长的那段，直到没有碎片。"""
    labels = list(labels)
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        dur = [(zones[i1][1] - zones[i0][0]).total_seconds() for _, i0, i1 in runs]
        short = [k for k, d in enumerate(dur) if d < min_s]
        if not short:
            return labels
        k = min(short, key=lambda kk: dur[kk])
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        target = right if left is None else left if right is None else (left if dur[k - 1] >= dur[k + 1] else right)
        for i in range(runs[k][1], runs[k][2] + 1):
            labels[i] = target[0]


#: 「未佩戴」类别名里带这个就算（各版本模型叫法一致）
NOTWORN_KEY = "未佩戴"
#: 静止类：名字里带这些的算（各版本模型叫 睡觉 / 静止/休息 / 睡眠 / 卧…）
REST_KEYS = ("静止", "休息", "睡", "卧", "趴")


def _is_rest(lab: str | None) -> bool:
    return bool(lab) and NOTWORN_KEY not in lab and any(k in lab for k in REST_KEYS)


def _absorb_short_notworn(labels: list[str], zones, min_s: float) -> list[str]:
    """短于 min_s、且左右至少一边是静止类的「未佩戴」段，改成那段静止（两边都是就跟更长的那边）。
    两边都不是静止（比如夹在活动中间）的不动——那种更可能是真的摘了，交给人判断。"""
    if min_s <= 0:
        return labels
    labels = list(labels)
    runs = _runs(labels)
    dur = [(zones[i1][1] - zones[i0][0]).total_seconds() for _, i0, i1 in runs]
    for k, (lab, i0, i1) in enumerate(runs):
        if NOTWORN_KEY not in (lab or "") or dur[k] >= min_s:
            continue
        cands = [(dur[j], runs[j][0]) for j in (k - 1, k + 1) if 0 <= j < len(runs) and _is_rest(runs[j][0])]
        if not cands:
            continue
        target = max(cands)[1]
        for i in range(i0, i1 + 1):
            labels[i] = target
    return labels


def _smooth_states(probs: list[dict], states: list[str], n: int, k: int) -> list[str]:
    half = max(0, k // 2)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        best, best_v = None, -1.0
        for lab in states:
            v = sum(probs[j].get(lab, 0.0) for j in range(lo, hi)) / (hi - lo)
            if v > best_v:
                best, best_v = lab, v
        out.append(best or (states[0] if states else ""))
    return out


def _viterbi(probs: list[dict], classes: list[str], switch_cost: float) -> list[str]:
    """发射 = log(prob)，切换类别扣 switch_cost，求整条序列代价最小的标签。"""
    n, m = len(probs), len(classes)
    if n == 0 or m == 0:
        return []
    eps = 1e-6
    emit = [[math.log(max(eps, p.get(c, 0.0))) for c in classes] for p in probs]
    score = list(emit[0])
    back: list[list[int]] = []
    for i in range(1, n):
        best_prev = max(range(m), key=lambda j: score[j])
        new_score, bp = [], []
        for c in range(m):
            stay = score[c]
            switch = score[best_prev] - switch_cost
            if stay >= switch:
                new_score.append(stay + emit[i][c]); bp.append(c)
            else:
                new_score.append(switch + emit[i][c]); bp.append(best_prev)
        score, back = new_score, back + [bp]
    cur = max(range(m), key=lambda j: score[j])
    path = [cur]
    for bp in reversed(back):
        cur = bp[cur]
        path.append(cur)
    path.reverse()
    return [classes[c] for c in path]


def _bouts_hysteresis(pv: list[float], enter: float, stay: float) -> list[list[int]]:
    """双阈值滞回：≥enter 进入，≥stay 维持。返回 [[i0..i1 的下标列表], ...]"""
    bouts: list[list[int]] = []
    cur: list[int] | None = None
    for i, v in enumerate(pv):
        if cur is None:
            if v >= enter:
                cur = [i]
        else:
            if v >= stay:
                cur.append(i)
            else:
                bouts.append(cur); cur = None
    if cur:
        bouts.append(cur)
    return bouts


def _merge_gaps(bouts: list[list[int]], zones, gap_s: float) -> list[list[int]]:
    merged: list[list[int]] = []
    for b in bouts:
        if merged and (zones[b[0]][0] - zones[merged[-1][-1]][1]).total_seconds() <= gap_s:
            merged[-1] = list(range(merged[-1][0], b[-1] + 1))
        else:
            merged.append(list(range(b[0], b[-1] + 1)))
    return merged


# ── 掉数据（MISSING）────────────────────────────────────────────────────
#
# 采集端蓝牙断了会把六个轴全写 0 来占帧对齐的位置（也可能写 MISSING → NaN）。
# 真实佩戴时六轴不可能同时精确为 0——重力总落在某个轴上——所以"六轴全 0"就是
# 这一段根本没有数据。这种窗口喂给模型只会得到一个凭空的类别（实测多半判成
# 睡觉），既不能算进有效佩戴，也绝不能进训练集。
#
# 处理办法是"挖洞"而不是"丢行"：CSV 里的行照留（不然工作台上看视频会跳帧、
# 波形和时间轴对不上），只是把这些时间段从预测结果里抠掉。
def missing_spans(windows: list[dict], window_s: float, stride_s: float, label_mode: str,
                  min_ratio: float = 0.5) -> list[tuple]:
    """窗口里缺数据比例 ≥ min_ratio 的，合并成一组不相交的 (start, end) 时间区间。"""
    ts = [_parse_ts(w.get("ts")) for w in windows]
    if not ts or any(t is None for t in ts):
        return []
    zones = _zones(ts, window_s, stride_s, label_mode)  # type: ignore[arg-type]
    out: list[list] = []
    for w, (a, b) in zip(windows, zones):
        m = w.get("missing")
        if m is None or m < min_ratio:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def drop_missing(segments: list[dict], holes: list[tuple]) -> list[dict]:
    """把片段里跟 holes 重叠的部分抠掉；一段被从中间挖开就拆成两段。"""
    if not holes:
        return segments
    out: list[dict] = []
    for seg in segments:
        a, b = _parse_ts(seg.get("start_ts")), _parse_ts(seg.get("end_ts"))
        if a is None or b is None:
            out.append(seg)
            continue
        pieces = [(a, b)]
        for h0, h1 in holes:
            nxt = []
            for x, y in pieces:
                if h1 <= x or h0 >= y:
                    nxt.append((x, y))
                    continue
                if h0 > x:
                    nxt.append((x, h0))
                if h1 < y:
                    nxt.append((h1, y))
            pieces = nxt
        for x, y in pieces:
            # 挖剩不足一个窗口的碎片没有意义，丢掉
            if (y - x).total_seconds() < 1.0:
                continue
            piece = dict(seg)
            piece["start_ts"] = _fmt_ts(x)
            piece["end_ts"] = _fmt_ts(y)
            out.append(piece)
    return out


def scratch_candidates(windows: list[dict], final_segments: dict[str, list[dict]],
                       window_s: float, stride_s: float, label_mode: str,
                       params: StableParams | None = None) -> list[dict]:
    """
    疑似抓挠候选：正式片段（稳定版/v2）为了准把 20% 左右的真抓挠也滤掉了，这里用
    低门槛再抽一遍给人工看——两类来源：
      low_conf：抓挠概率 ≥ cand_enter 进入、≥ cand_stay 维持，≥ cand_min_windows 个窗口；
      spectral：陀螺仪 4–8 Hz 占比 ≥ cand_spec_min 且连续 ≥ cand_min_windows 个窗口，
                模型没判成抓挠（模型漏检但物理信号像抓挠）。
    跟正式抓挠片段重叠的去掉。返回结构同 segments 元素，多 reason 字段。
    """
    p = params or StableParams()
    n = len(windows)
    if n == 0:
        return []
    ts = [_parse_ts(w.get("ts")) for w in windows]
    if any(t is None for t in ts):
        return []
    zones = _zones(ts, window_s, stride_s, label_mode)  # type: ignore[arg-type]
    sl = p.scratch_label
    probs = [w.get("probs") or {} for w in windows]
    spec = [w.get("spec") for w in windows]
    pv = [probs[i].get(sl, 0.0) for i in range(n)]

    taken = [False] * n
    for seg in final_segments.get(sl) or []:
        s0, e0 = _parse_ts(seg["start_ts"]), _parse_ts(seg["end_ts"])
        for i in range(n):
            if s0 and e0 and zones[i][0] < e0 and zones[i][1] > s0:
                taken[i] = True

    cands: list[tuple[list[int], str]] = []
    for b in _merge_gaps(_bouts_hysteresis(pv, p.cand_enter, p.cand_stay), zones, p.event_gap_s):
        b = [i for i in b if not taken[i]]
        if len(b) >= p.cand_min_windows:
            cands.append((b, "low_conf"))
    # 频谱这一路默认关掉（cand_spec_min <= 0）。当初的设想是"模型之外的独立证据"，
    # 但实测下来对这批设备完全不具备区分度：抽出来的几百条里绝大多数模型置信度是
    # 0%、频谱占比却有 0.5~0.68——这只狗的陀螺仪本底在 4–8 Hz 就偏高，阈值定在哪
    # 都是一片噪声。spec 值仍然照常算、照常在界面上显示，可以用来辅助判断某一条
    # 候选像不像；只是不再单独凭它把一段拎出来。哪天换了设备想试，把
    # CAND_SPEC_MIN 设成 >0 即可。
    if p.cand_spec_min > 0:
        covered = {i for b, _ in cands for i in b}
        run: list[int] = []
        for i in range(n + 1):
            ok = i < n and not taken[i] and i not in covered and spec[i] is not None and spec[i] >= p.cand_spec_min
            if ok:
                run.append(i)
            else:
                if len(run) >= p.cand_min_windows:
                    cands.append((list(run), "spectral"))
                run = []

    out = []
    for b, reason in cands:
        i0, i1 = b[0], b[-1]
        vals = [pv[i] for i in range(i0, i1 + 1)]
        mean_c = sum(vals) / len(vals)
        # 低置信那一路要求整段平均够高；频谱那一路本来就是靠物理证据进来的，不看概率
        if reason == "low_conf" and mean_c < p.cand_min_mean:
            continue
        # 候选是低门槛找漏检，但也得是"可能漏掉的抓挠"：设备没戴/躺着被碰、
        # 信号掉点的波动不成立（陀螺仪不动、模长不对、没有 2–10 Hz 节律）
        if not _scratch_quality_ok(b, windows, p, conf_mean=mean_c, conf_max=max(vals)):
            continue
        sv = [spec[i] for i in range(i0, i1 + 1) if spec[i] is not None]
        out.append({
            "start_ts": _fmt_ts(zones[i0][0]),
            "end_ts": _fmt_ts(zones[i1][1]),
            "conf_max": float(max(vals)),
            "conf_mean": float(sum(vals) / len(vals)),
            "n_windows": i1 - i0 + 1,
            "spec": round(sum(sv) / len(sv), 3) if sv else None,
            "reason": reason,
        })
    # 置信度高的排前面：那些多半是被平滑抹掉的真抓挠，最值得先看。
    # 数量上限不在这里砍——裁剪和"丢了多少条"的日志一起放在调用方（app.py），
    # 免得这里静悄悄少给几条、外面还不知道
    out.sort(key=lambda c: c["conf_mean"], reverse=True)
    return out


def refine_boundaries(segments: list[dict], envelope: dict | None, params: StableParams | None = None) -> None:
    """
    用陀螺仪能量包络（pool.py 算的，10 Hz）把片段起止从 1 秒窗口对齐精确到 0.1 秒：
    在原起点 ±margin 内找第一个能量高于阈值的点当新起点，终点同理找最后一个。
    阈值 = 片段内能量中位数 × refine_ratio。原地改 segments 的 start_ts/end_ts。
    """
    p = params or StableParams()
    if not envelope or not segments:
        return
    t0 = _parse_ts(envelope.get("t0"))
    hz = float(envelope.get("hz") or 0)
    energy = envelope.get("energy") or []
    if t0 is None or hz <= 0 or not energy:
        return
    n = len(energy)

    def idx(t: datetime) -> int:
        return int(round((t - t0).total_seconds() * hz))

    def at(i: int) -> datetime:
        return t0 + timedelta(seconds=i / hz)

    m = int(round(p.refine_margin_s * hz))
    for seg in segments:
        s, e = _parse_ts(seg["start_ts"]), _parse_ts(seg["end_ts"])
        if s is None or e is None:
            continue
        si, ei = idx(s), idx(e)
        inner = sorted(energy[max(0, si):min(n, ei)])
        if not inner:
            continue
        thr = inner[len(inner) // 2] * p.refine_ratio
        lo, hi = max(0, si - m), min(n - 1, si + m)
        new_s = next((i for i in range(lo, hi + 1) if energy[i] >= thr), None)
        lo, hi = max(0, ei - m), min(n - 1, ei + m)
        new_e = next((i for i in range(hi, lo - 1, -1) if energy[i] >= thr), None)
        if new_s is not None and new_e is not None and new_e > new_s:
            seg["start_ts"], seg["end_ts"] = _fmt_ts(at(new_s)), _fmt_ts(at(new_e + 1))


# v2（8 s 窗口、label_mode=segment）片段起止的系统偏差，秒：(起点, 终点)，正 = 偏晚。
# 2026-11-18 5 折交叉验证（6 轴 8 s CNN，跟 #41 同一份数据），只算我们自己的项圈数据，中位数：
# 抓挠起点早 0.76 s、终点晚 1.69 s；甩头 / 抖身起点早 0.52 s、终点晚 1.44 s。
# 8 s 窗口判「里面有抓挠」，动作停了之后窗口还要再滑一阵才不报，所以终点总是偏晚。
# 平移 + 3–9 Hz 能量对齐之后：抓挠起止误差 0.98 / 1.75 s → 0.60 / 0.58 s，起止对齐（IoU≥0.5）46.6% → 53.5%；
# 甩头 9.8% → 53.4%；命中、查准不变（experiments/claude_2026_11/boundary_eval.py）。
# 别的窗口长度没量过，不平移，只做能量对齐。
PRECISE_BIAS_8S = {"抓挠": (-0.75, 1.7), "_other": (-0.5, 1.45)}


def is_event_for_precise(label: str) -> bool:
    """精准起止只管事件类（抓挠、甩头 / 抖身 / 甩身体及其二级），状态类（静止、活动……）不动"""
    return label.startswith("抓挠") or label.startswith("甩")


def precise_bias(label: str, window_s: float, stride_s: float, label_mode: str,
                 scratch_label: str = "抓挠") -> tuple[float, float]:
    if label_mode != "segment" or abs(window_s - 8.0) > 0.5:
        return 0.0, 0.0
    key = scratch_label if (label == scratch_label or label.startswith(scratch_label + "-")) else "_other"
    return PRECISE_BIAS_8S[key]


def precise_boundaries(segments: list[dict], envelope: dict | None, bias: tuple[float, float],
                       params: StableParams | None = None) -> None:
    """精准版起止，原地改 start_ts / end_ts：
    1. 平移掉系统偏差（bias，见 PRECISE_BIAS_8S）；
    2. 用陀螺仪 3–9 Hz 能量包络（pool.py 算的 envelope["band"]，10 Hz）在附近找真正开始 / 停下的点：
       起点在 [-pre, +start_in] 里找第一个「连续 sustain 格高于阈值」的点，终点在 [-end_in, +end_out] 里倒着找。
       阈值 = 片段内能量中位数 × precise_ratio。只看抓挠节律那段频率，转身、走动这类低频大动作不会把起点带偏
       ——平台原来的 refine_boundaries 用的是总能量，实测起点反而早了 1.4 s。
    找不到、或收完不到 0.5 s 的，保留平移后的起止。没有陀螺仪（没 band）就只平移。"""
    p = params or StableParams()
    if not segments:
        return
    ds, de = bias
    for seg in segments:
        s, e = _parse_ts(seg["start_ts"]), _parse_ts(seg["end_ts"])
        if s is None or e is None:
            continue
        s2, e2 = s - timedelta(seconds=ds), e - timedelta(seconds=de)
        if (e2 - s2).total_seconds() < 0.5:      # 短片段别被平移收没了：留中间 0.5 s
            mid = s + (e - s) / 2
            s2, e2 = mid - timedelta(seconds=0.25), mid + timedelta(seconds=0.25)
        seg["start_ts"], seg["end_ts"] = _fmt_ts(s2), _fmt_ts(e2)
    band = (envelope or {}).get("band") or []
    t0 = _parse_ts((envelope or {}).get("t0"))
    hz = float((envelope or {}).get("hz") or 0)
    if not band or t0 is None or hz <= 0:
        return
    k = max(1, int(p.precise_smooth_bins))
    raw = band
    n = len(raw)
    # 滑动平均（same 长度）
    csum = [0.0]
    for v in raw:
        csum.append(csum[-1] + v)
    half = k // 2
    sm = [(csum[min(n, i + half + 1)] - csum[max(0, i - half)]) / (min(n, i + half + 1) - max(0, i - half)) for i in range(n)]
    sus = max(1, int(p.precise_sustain_bins))
    for seg in segments:
        s, e = _parse_ts(seg["start_ts"]), _parse_ts(seg["end_ts"])
        si = int(round((s - t0).total_seconds() * hz))
        ei = int(round((e - t0).total_seconds() * hz))
        core = sorted(sm[max(0, si):min(n, ei)])
        if not core:
            continue
        thr = core[len(core) // 2] * p.precise_ratio
        on = [v >= thr for v in sm]
        lo, hi = max(0, si - int(p.precise_pre_s * hz)), min(n - sus, si + int(p.precise_start_in_s * hz))
        new_s = next((i for i in range(lo, hi + 1) if all(on[i:i + sus])), None)
        lo, hi = max(sus - 1, ei - int(p.precise_end_in_s * hz)), min(n - 1, ei + int(p.precise_end_out_s * hz))
        new_e = next((i for i in range(hi, lo - 1, -1) if all(on[i - sus + 1:i + 1])), None)
        if new_s is not None and new_e is not None and (new_e + 1 - new_s) / hz >= 0.5:
            seg["start_ts"] = _fmt_ts(t0 + timedelta(seconds=new_s / hz))
            seg["end_ts"] = _fmt_ts(t0 + timedelta(seconds=(new_e + 1) / hz))


def strict_params(p: StableParams) -> StableParams:
    """严格版的参数：抓挠平均概率下限换成 strict_scratch_min_mean，陀螺仪 / 节律门槛只往严里收"""
    return replace(p, scratch_min_mean=p.strict_scratch_min_mean,
                   gyro_std_min=max(p.gyro_std_min, p.strict_gyro_std_min),
                   acc_rhythm_min=max(p.acc_rhythm_min, p.strict_acc_rhythm_min))


def _scratch_quality_ok(b: list[int], windows: list[dict], p: StableParams,
                        *, conf_mean: float | None = None, conf_max: float | None = None) -> bool:
    """抓挠 bout 的「身体真的在动」证据。特征算不出来时不拦（宁可多留，不误杀）。"""
    gs = [windows[i]["gstd"] for i in b if windows[i].get("gstd") is not None]
    if p.gyro_std_min > 0 and gs and sum(gs) / len(gs) < p.gyro_std_min:
        return False
    ms = [windows[i]["missing"] for i in b if windows[i].get("missing") is not None]
    if p.missing_max > 0 and ms and sum(ms) / len(ms) > p.missing_max:
        return False
    am = [windows[i]["amag"] for i in b if windows[i].get("amag") is not None]
    if am:
        mean_am = sum(am) / len(am)
        if p.amag_min > 0 and mean_am < p.amag_min:
            return False
        if p.amag_max > 0 and mean_am > p.amag_max:
            return False
    rhythm_exempt = (
        (p.rhythm_exempt_mean > 0 and conf_mean is not None and conf_mean >= p.rhythm_exempt_mean)
        or (p.rhythm_exempt_max > 0 and conf_max is not None and conf_max >= p.rhythm_exempt_max)
    )
    ar = [windows[i]["arhythm"] for i in b if windows[i].get("arhythm") is not None]
    if not rhythm_exempt and p.acc_rhythm_min > 0 and ar and sum(ar) / len(ar) < p.acc_rhythm_min:
        return False
    return True


def stabilize(windows: list[dict], classes: list[str], target_labels: list[str],
              window_s: float, stride_s: float, label_mode: str,
              params: StableParams | None = None, algo: str = "stable") -> dict[str, list[dict]]:
    """返回 {label: [{start_ts, end_ts, conf_max, conf_mean, n_windows, spec}]}，跟调试版 segments 同结构。

    algo 认这几个：
      stable / viterbi                    两套解码，见模块顶部
      stable_noshake / viterbi_noshake    同上，但**不让抓挠吞并甩身体**
      viterbi_precise                     解码跟 viterbi 一样，起止在 app.py 里按 precise_boundaries 调
      viterbi_strict                      解码跟 viterbi 一样，只把抓挠的门槛换成 strict_*（见 STRICT_SUFFIX）

    `_noshake` 后缀单独做成一个 algo、而不是让调用方改 params，是为了让它
    在平台上是一个**独立的版本**：结果按 (样本, 模型, 版本) 存，版本串不同
    才不会互相覆盖，两边才能并排对比。改配置的话两次跑出来的 mode 都是
    "viterbi"，后一次直接把前一次顶掉。
    """
    p = params or StableParams()
    if algo.endswith(STRICT_SUFFIX):
        p = strict_params(p)
        algo = algo[: -len(STRICT_SUFFIX)]
    if algo.endswith(PRECISE_SUFFIX):
        # 解码跟不带后缀的完全一样；起止的调整在 app.py 里拿到能量包络之后做（precise_boundaries）
        algo = algo[: -len(PRECISE_SUFFIX)]
    if algo.endswith(NOSHAKE_SUFFIX):
        # 只改这一个参数，别的一律照旧——要比的就是"吞不吞"这一件事
        p = replace(p, shake_absorb_s=0.0)
        algo = algo[: -len(NOSHAKE_SUFFIX)]
    n = len(windows)
    if n == 0:
        return {lab: [] for lab in target_labels}
    ts = [_parse_ts(w.get("ts")) for w in windows]
    if any(t is None for t in ts):
        return {lab: [] for lab in target_labels}
    zones = _zones(ts, window_s, stride_s, label_mode)  # type: ignore[arg-type]

    # 哪些类别按"事件"处理（滞回、最短窗口数那一套），哪些按"状态"。
    # **按前缀认，不是精确相等**：把抓挠拆成二级之后类别叫「抓挠-头颈耳」，
    # 精确匹配的话它就掉进 states，走的是状态那套平滑——一段几秒的抓挠会被
    # 直接抹掉，而且看起来就像模型没检出来
    def _is_event(lab: str) -> bool:
        return any(lab == e or lab.startswith(e + "-") for e in p.event_labels)

    events = [lab for lab in classes if _is_event(lab)]
    states = [lab for lab in classes if lab not in events]
    probs = [w.get("probs") or {} for w in windows]
    spec = [w.get("spec") for w in windows]
    raw_label = [w.get("label") for w in windows]

    # ── 状态时间轴 ──
    if algo == "viterbi":
        decoded = _viterbi(probs, classes, p.viterbi_switch)
        # 状态部分：把解码出的事件窗口先按邻居填掉，得到纯状态轴
        state_label = []
        last = next((d for d in decoded if d in states), states[0] if states else "")
        for d in decoded:
            if d in states:
                last = d
            state_label.append(last)
    else:
        decoded = None
        state_label = _absorb_short_runs(_smooth_states(probs, states, n, p.smooth_windows), zones, p.min_state_s)
    # 短的未佩戴碎片并回挨着的静止（两套解码都做）
    state_label = _absorb_short_notworn(state_label, zones, p.notworn_min_s)

    # ── 事件 bout ──
    final = list(state_label)
    bouts_by_event: dict[str, list[list[int]]] = {}
    for ev in events:
        pv = [probs[i].get(ev, 0.0) for i in range(n)]
        if decoded is not None:
            idx = [i for i in range(n) if decoded[i] == ev]
            raw_b = [[i] for i in idx]
        else:
            raw_b = _bouts_hysteresis(pv, p.event_enter, p.event_stay)
        bouts_by_event[ev] = _merge_gaps(raw_b, zones, p.event_gap_s)

    # 抓挠吞并前后的甩身体窗口
    #
    # ⚠ 这里的判据是**从抓挠边界往外扩了多少秒（累计）**，不是相邻窗口之间
    # 的间隔。曾经写成后者，结果这个参数形同虚设：
    # majority 模式下相邻窗口的 zone 首尾相接（见 _zones），间隔**恒等于 0**，
    # 于是 `间隔 <= absorb_s` 对任何非负的 absorb_s 都成立，while 循环
    # 一路链式吞到底——**一串 4.5 秒的甩身体会被 3 秒的参数整段吞掉**，
    # 而且 STABLE_SHAKE_ABSORB_S=0 也关不掉（0 <= 0 为真）。
    #
    # absorb_s <= 0 = 关闭，跟 spectral_min 一个约定。
    sl, sh = p.scratch_label, p.shake_label
    if sl in bouts_by_event and sh in classes and p.shake_absorb_s > 0:
        shake_idx = {i for i in range(n) if raw_label[i] == sh or (decoded is not None and decoded[i] == sh)}
        grown = []
        for b in bouts_by_event[sl]:
            i0, i1 = b[0], b[-1]
            lo_edge, hi_edge = zones[i0][0], zones[i1][1]
            while (i0 - 1 >= 0 and (i0 - 1) in shake_idx
                   and (lo_edge - zones[i0 - 1][0]).total_seconds() <= p.shake_absorb_s):
                i0 -= 1
            while (i1 + 1 < n and (i1 + 1) in shake_idx
                   and (zones[i1 + 1][1] - hi_edge).total_seconds() <= p.shake_absorb_s):
                i1 += 1
            grown.append(list(range(i0, i1 + 1)))
        bouts_by_event[sl] = _merge_gaps(grown, zones, p.event_gap_s)
        taken = {i for b in bouts_by_event[sl] for i in b}
        if sh in bouts_by_event:
            bouts_by_event[sh] = [[i for i in b if i not in taken] for b in bouts_by_event[sh]]
            bouts_by_event[sh] = [b for b in bouts_by_event[sh] if b]

    # 门槛过滤 + 写到时间轴（抓挠优先于甩身体）
    for ev in sorted(bouts_by_event, key=lambda e: 0 if e == sl else 1):
        for b in bouts_by_event[ev]:
            pv = [probs[i].get(ev, 0.0) for i in b]
            mean_c, max_c = sum(pv) / len(pv), max(pv)
            min_mean = p.scratch_min_mean if (ev == sl and p.scratch_min_mean > 0) else p.event_min_mean
            keep = (len(b) >= p.event_min_windows and mean_c >= min_mean) or max_c >= p.event_single_conf
            if keep and ev == sl and p.spectral_min > 0:
                sv = [spec[i] for i in b if spec[i] is not None]
                if sv and sum(sv) / len(sv) < p.spectral_min:
                    keep = False
            # 概率再高也要有「身体真的在动」的证据：陀螺仪不动 / 大量掉点 /
            # 模长偏离 1g 的，是设备没戴或被碰，不是抓挠
            if keep and ev == sl:
                keep = _scratch_quality_ok(b, windows, p, conf_mean=mean_c, conf_max=max_c)
            if not keep:
                continue
            for i in b:
                if final[i] not in events or ev == sl:
                    final[i] = ev

    # ── 按最终标签切片段 ──
    out: dict[str, list[dict]] = {lab: [] for lab in target_labels}
    for lab, i0, i1 in _runs(final):
        if lab not in out:
            continue
        pv = [probs[i].get(lab, 0.0) for i in range(i0, i1 + 1)]
        sv = [spec[i] for i in range(i0, i1 + 1) if spec[i] is not None]
        out[lab].append({
            "start_ts": _fmt_ts(zones[i0][0]),
            "end_ts": _fmt_ts(zones[i1][1]),
            "conf_max": float(max(pv)) if pv else 0.0,
            "conf_mean": float(sum(pv) / len(pv)) if pv else 0.0,
            "n_windows": i1 - i0 + 1,
            "spec": round(sum(sv) / len(sv), 3) if sv else None,
        })
    return out
