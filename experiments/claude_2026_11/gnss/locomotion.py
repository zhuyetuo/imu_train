"""只用 3 轴加速度的移动检测 + 计步（板上能跑的写法：整数友好、只有 EMA / 比较 / 计数）。
输入：20 Hz 的 acc_x/y/z（单位 g）。输出：逐样本的「这一步是不是脚步」和逐秒的「是不是在走 / 跑」。"""
import numpy as np

HZ = 20


def steps_and_loco(acc, p=None):
    p = {**DEFAULT, **(p or {})}
    m = np.sqrt((acc ** 2).sum(1))                      # |a|，跟项圈朝向无关
    # 去重力：慢 EMA（约 1 s）当基线；再 2 点平均去毛刺（板上就是两个累加器）
    from scipy.signal import lfilter
    ka = 1 / (p["base_s"] * HZ)
    base, _ = lfilter([ka], [1, -(1 - ka)], m, zi=[m[0] * (1 - ka)])
    d = m - base
    d = np.convolve(d, np.ones(p["smooth"]) / p["smooth"], mode="same")
    # 脚步 = 高于门限的局部最大值，相邻两步至少隔 min_gap 秒（抓挠 5–8 Hz 的峰挤在一起，大部分会被这条吃掉）
    gap = int(round(p["min_gap"] * HZ))
    step = np.zeros(len(d), bool); last = -10 ** 9
    cand = np.flatnonzero((d[1:-1] > p["peak_g"]) & (d[1:-1] >= d[:-2]) & (d[1:-1] > d[2:])) + 1
    for i in cand:
        if i - last >= gap:
            step[i] = True; last = i
    # 逐秒判断：看前 win 秒里的脚步数和节奏是否均匀
    n_sec = len(m) // HZ
    loco = np.zeros(n_sec, bool); cad = np.zeros(n_sec)
    idx = np.flatnonzero(step)
    W = p["win_s"]
    los = np.searchsorted(idx, np.maximum(0, (np.arange(n_sec) + 1 - W) * HZ))
    his = np.searchsorted(idx, (np.arange(n_sec) + 1) * HZ)
    for s in range(n_sec):
        k = idx[los[s]:his[s]]
        n = len(k)
        cad[s] = n / W
        if n >= p["min_steps"]:
            iv = np.diff(k) / HZ
            cv = iv.std() / max(iv.mean(), 1e-6)
            loco[s] = cv <= p["max_cv"] and cad[s] <= p["max_cad"]
    return step, loco, cad


DEFAULT = dict(base_s=1.0, smooth=2, min_gap=0.25, peak_g=0.06, win_s=4, min_steps=5, max_cv=0.45, max_cad=4.5)
