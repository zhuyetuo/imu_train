"""实验脚本提速（生成 multiscale_fast.py）：
1. 时间线缓存：每个 CSV 读一次、重采样一次，存成 .npz（按路径 + 修改时间 + 大小 + 采样率 + 轴数 + 代码版本做键），以后几秒载入；
   人工标签每次照常按当前数据集重新打（不进缓存）
2. CNN 不算 193 维手工特征：窗口模型只要方向无关变换后的原始窗口（map_channels），手工特征是给 XGB / RF 的"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    if s.count(old) != count:
        sys.exit(f"出现 {s.count(old)} 次（要 {count}）：{old[:80]!r}")
    s = s.replace(old, new)


helper = '''

TL_CACHE = os.path.expanduser("~/claude_exp/tlcache")
TL_VER = "v2-zero1"   # 改了 load_timeline（比如全 0 行当缺失）就换这个，旧缓存自动作废


def _timeline_cached(csv_path, hz, declared, axes, upsample):
    import hashlib
    st = os.stat(csv_path)
    key = hashlib.md5(f"{csv_path}|{st.st_mtime_ns}|{st.st_size}|{hz}|{declared}|{axes}|{upsample}|{TL_VER}".encode()).hexdigest()
    p = os.path.join(TL_CACHE, key[:2], key + ".npz")
    if os.path.exists(p):
        try:
            z = np.load(p, allow_pickle=False)
            if int(z["none"]):
                return None
            from datetime import datetime as _dt
            return {"x": z["x"], "bad": z["bad"], "t0": _dt.fromisoformat(str(z["t0"])), "src": float(z["src"])}
        except Exception:  # noqa: BLE001 坏了就重算
            pass
    tl = T.load_timeline(csv_path, hz, declared, axes, upsample)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".{os.getpid()}.tmp.npz"
    if tl is None:
        np.savez(tmp, none=1)
    else:
        np.savez(tmp, none=0, x=tl["x"], bad=tl["bad"], t0=tl["t0"].isoformat(), src=tl["src"])
    os.replace(tmp, p)
    return tl


def _build_cached(args):
    """跟 train_v2._build_one 一样，只是时间线走缓存；人工标签每次按当前数据集重新打"""
    csv_path, segs, hz, axes, upsample = args
    try:
        if not os.path.isfile(csv_path):
            return csv_path, None, f"CSV 不存在：{csv_path}"
        tl = _timeline_cached(csv_path, hz, segs[0].get("source_hz"), axes, upsample)
        if tl is None:
            return csv_path, None, None
        n = len(tl["x"])
        lab = np.full(n, -1, np.int32)
        for s_ in segs:
            a_ms, b_ms = s_["start_ms"], s_["end_ms"]
            if a_ms is None or b_ms is None:
                a, b = T._parse_dt(s_["start"]), T._parse_dt(s_["end"])
                if a is None or b is None:
                    continue
                a_ms = (a - tl["t0"]).total_seconds() * 1000
                b_ms = (b - tl["t0"]).total_seconds() * 1000
            a, b = max(int(np.ceil(a_ms / 1000 * hz)), 0), min(int(np.floor(b_ms / 1000 * hz)), n)
            if b > a:
                lab[a:b] = s_["idx"]
        tl = dict(tl)
        tl["seg"] = lab
        return csv_path, tl, None
    except Exception as e:  # noqa: BLE001
        return csv_path, None, f"{type(e).__name__}: {str(e)[:120]}"


def _fz(X, hz):
    """窗口模型只要方向无关变换后的原始窗口；手工特征（给 XGB / RF 的）不算"""
    from gravity_align import map_channels
    return map_channels(X, "inv", True), None

'''
anchor = "# ── 留出任务上的评估"
sub(anchor, helper + anchor)
n1 = s.count("ex.map(_safe_build,")
s = s.replace("ex.map(_safe_build,", "ex.map(_build_cached,")
n2 = s.count("T.featurize(")
s = s.replace("T.featurize(", "_fz(")
open(dst, "w", encoding="utf-8").write(s)
print(f"读数据换成缓存 {n1} 处，手工特征去掉 {n2} 处")
