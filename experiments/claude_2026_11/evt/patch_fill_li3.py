"""label_infra：顶部工具栏加「补标·抓挠 待判 N」模式（跟「专项待判」一样：点了列表只剩有补标的项目，点开都只看补标任务）。在 smart-label/ 下跑。"""
import sys


def sub(path, old, new):
    s = open(path, encoding="utf-8").read()
    if s.count(old) != 1:
        sys.exit(f"{path}: 片段出现 {s.count(old)} 次：{old[:70]!r}")
    open(path, "w", encoding="utf-8").write(s.replace(old, new))


P = "frontend/src/pages/Projects.tsx"

# 1) 状态 + 切换函数（跟专项互斥）
sub(P, '''  const [maskOnly, setMaskOnly] = useState(() => {''', '''  // 补标模式：模型在没人标的时间里报的抓挠（unl_fill），跟专项一样做成顶部开关；记在这台浏览器里
  const [fillOnly, setFillOnly] = useState(() => {
    try { return localStorage.getItem("projects-fill-only") === "1"; } catch { return false; }
  });
  const FILL_FILTER = { fill: true, spot: false, cand: false, candLabels: [] as string[], labels: [] as number[],
                        unreviewedOnly: false, segReturned: false, status: "ALL" as const };
  const fillOf = (pid: number) => scratchReview?.fill?.[String(pid)] ?? 0;
  const toggleFillOnly = () => {
    const on = !fillOnly;
    try { localStorage.setItem("projects-fill-only", on ? "1" : "0"); } catch { /* 隐私模式 */ }
    const pids = Object.entries(scratchReview?.fill ?? {}).filter(([, n]) => n > 0).map(([k]) => Number(k));
    if (on) {
      // 跟专项模式互斥，免得两个过滤叠在一起什么都看不到
      if (specialOnly) { try { localStorage.setItem("projects-special-only", "0"); } catch { /* 隐私模式 */ } setSpecialOnly(false); }
      pids.forEach((pid) => setFilter(pid, FILL_FILTER));
      if (pids.length === 1) setExpandedKeys(() => pids);
    } else {
      pids.forEach((pid) => setFilter(pid, { fill: false }));
    }
    setFillOnly(on);
  };
  const [maskOnly, setMaskOnly] = useState(() => {''')

# 专项打开时关掉补标模式
sub(P, '''    if (!v) {
      // 打开：有专项的项目都切到「只看专项任务」，只有一个就直接展开''', '''    if (!v) {
      if (fillOnly) { try { localStorage.setItem("projects-fill-only", "0"); } catch { /* 隐私模式 */ } setFillOnly(false); }
      // 打开：有专项的项目都切到「只看专项任务」，只有一个就直接展开''')

# 2) 工具栏按钮 + 3) 模式提示条
sub(P, '''        ) : null}
      </Space>
      {specialOnly && (''', '''        ) : null}
        {/* 补标待判：模型在没人标过的时间里很有把握报的抓挠，抽查约 3/4 是漏标的。确认了会写进标注 */}
        {(scratchReview?.fill_total ?? 0) > 0 || fillOnly ? (
          <Tooltip title={<div style={{ maxWidth: 380 }}>
            「补标」：模型在<b>没人标过的时间</b>里很有把握报了抓挠，抽查下来约 3/4 是漏标的真抓挠。
            对着视频判：是抓挠就确认（起止可以先调），会写进标注；不是就拒绝。
            <br />点一下：项目列表只剩有补标的项目，点开哪个都只看有补标待判的任务；再点一下恢复。
          </div>}>
            <Button size="small" type={fillOnly ? "primary" : "default"} style={fillOnly ? { marginLeft: 8 } : { marginLeft: 8, color: "#2f54eb", borderColor: "#2f54eb" }}
              onClick={toggleFillOnly}>
              {fillOnly ? "退出补标" : "补标·抓挠 待判"} {scratchReview?.fill_total ?? 0}
              {fillOnly ? "" : `（${Object.values(scratchReview?.fill ?? {}).filter((n) => n > 0).length} 个项目）`}
            </Button>
          </Tooltip>
        ) : null}
      </Space>
      {fillOnly && (
        <Alert type="info" showIcon style={{ marginBottom: 8 }}
          message={<span>补标模式：只显示有「补标·抓挠」待判的项目（共 {scratchReview?.fill_total ?? 0} 段），点开哪个都只看有补标待判的任务。
            打开任务在「疑似片段」里判，是抓挠就确认（会写进标注）。已通过的任务要先点「继续编辑」。</span>}
          action={<Button size="small" onClick={toggleFillOnly}>退出补标</Button>} />
      )}
      {specialOnly && (''')

# 4) 项目列表过滤
sub(P, '''          .filter((p) => !specialOnly || specialOf(p.id) > 0)}''',
       '''          .filter((p) => !specialOnly || specialOf(p.id) > 0)
          .filter((p) => !fillOnly || fillOf(p.id) > 0)}''')

# 5) 补标模式下点开项目自动只看补标任务
sub(P, '''            if (expanded && specialOnly) setFilter(p.id, SPECIAL_FILTER);''',
       '''            if (expanded && specialOnly) setFilter(p.id, SPECIAL_FILTER);
            if (expanded && fillOnly) setFilter(p.id, FILL_FILTER);''')
print("ok")
