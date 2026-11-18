"""multiscale_exp.py 加 --heldout <导出 JSON>：（可以 --folds 0 跳过交叉验证）用全部训练数据训一个模型，
在留出任务上连续推理 → 稳定版 v2 → 按片段算命中 / 查准 / 误报（人工标的别的动作的时间里每小时报几次）。"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    if s.count(old) != 1:
        sys.exit(f"出现 {s.count(old)} 次：{old[:80]!r}")
    s = s.replace(old, new)


sub('''    res = {"model": a.model, "epochs": a.epochs, "folds_run": min(a.folds, 5), "n_params": n_params,''', '''    heldout = None
    if a.heldout:
        heldout = eval_heldout(a, spec, hz, axes, W, S, classes, remap, Xtr_m, y_tr, nK, gate, merge)
        n_params = heldout.pop("n_params")
    res = {"model": a.model, "epochs": a.epochs, "folds_run": min(a.folds, 5), "n_params": n_params, "heldout": heldout,''')
sub('''           "all": T.segment_metrics(seg_files, classes),
           "own": T.segment_metrics([f for f in seg_files if f["dataset"] == OWN], classes),''', '''           "all": T.segment_metrics(seg_files, classes) if seg_files else None,
           "own": T.segment_metrics([f for f in seg_files if f["dataset"] == OWN], classes) if seg_files else None,''')
sub('''# ── 主流程 ──────────────────────────────────────────────────────────────────
''', '''# ── 留出任务上的评估 ─────────────────────────────────────────────────────────

def eval_heldout(a, spec, hz, axes, W, S, classes, remap, Xtr_m, y_tr, nK, gate, merge):
    """全部训练数据训一个，在 --heldout 的任务上算片段级命中 / 误报"""
    from concurrent.futures import ProcessPoolExecutor
    proba, n_params = fit_window(WINDOW_MODELS[a.model], Xtr_m, y_tr, nK, a.epochs, seed=0)
    ds = {"name": "heldout", "export_json": a.heldout, "source_hz": None}
    segs = T.collect_segments([ds], spec["nas_root"], remap, classes, [])
    by_csv: dict = {}
    for i, s_ in enumerate(segs):
        s_["idx"] = i
        by_csv.setdefault(s_["csv"], []).append(s_)
    files = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for p_csv, tl, err in ex.map(T._build_one, [(p, ss, hz, axes, False) for p, ss in by_csv.items()], chunksize=4):
            if tl is None or err:
                continue
            ss = by_csv[p_csv]
            cw = T.labeled_span_windows(tl, ss, W, S, hz)
            if cw is None:
                continue
            Xc_, st_c, _ = cw
            Xm_c, _ = T.featurize(Xc_, hz)
            P = merge(proba(Xm_c))
            P = T.apply_motion_gate(P / P.sum(1, keepdims=True), T.motion_of(Xm_c), gate)
            pred = T.postprocess_segments(P, st_c, tl["t0"], classes, W, S, hz, REPO)
            hs = [(x["start_ms"] / 1000, x["end_ms"] / 1000, x["coarse"]) for x in ss if x.get("start_ms") is not None]
            files.append({"gt": {c: [(x0, x1) for x0, x1, cc in hs if cc == c] for c in T.EVENT_EVAL},
                          "other": {c: [(x0, x1) for x0, x1, cc in hs if cc != c] for c in T.EVENT_EVAL}, "pred": pred})
    m = T.segment_metrics(files, classes)
    print(f"▶ 留出任务 {len(files)} 个：抓挠 {m['抓挠']}", flush=True)
    return {"files": len(files), "metrics": m, "n_params": n_params}


# ── 主流程 ──────────────────────────────────────────────────────────────────
''')
sub('''    p.add_argument("--fine", default="", help="单独训、再并回去的细类，逗号分隔")''',
    '''    p.add_argument("--fine", default="", help="单独训、再并回去的细类，逗号分隔")
    p.add_argument("--heldout", default="", help="留出任务的导出 JSON：全量训一个模型，在这些任务上算命中 / 误报")''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
