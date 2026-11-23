/* 行为识别页的图示与交互图：不依赖任何库，全部用 SVG 现画 */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  // 语义色（跟 site.css 的 --c-* / --lv-* 一致）：真值蓝、预测橙、命中绿、漏检红、多报紫；三层口径青 / 黄 / 粉
  const C = { gt: "#4f8cff", pred: "#ff9f1c", hit: "#2ecc71", miss: "#ff4d6d", fp: "#c084fc",
              win: "#22d3ee", seg: "#facc15", evt: "#f472b6", mute: "#8fa0bd", text: "#dbe4f3", cur: "#ffe14d" };
  function rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function box(p, x, y, w, h, col, a, extra) {
    return add(p, "rect", Object.assign({ x, y, width: Math.max(1, w), height: h, rx: 3,
      fill: rgba(col, a == null ? 0.35 : a), stroke: col, "stroke-width": 1.2 }, extra || {}));
  }
  // 图内小图例
  function legend(p, x, y, items) {
    items.forEach(([col, lab]) => {
      add(p, "rect", { x, y: y - 9, width: 14, height: 10, rx: 2, fill: col });
      const t = add(p, "text", { x: x + 19, y, class: "small" }, lab);
      x += 19 + lab.length * 12 + 18;
    });
  }

  function root(host, w, h) {
    const old = host.querySelector("svg");
    if (old) old.remove();
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", `0 0 ${w} ${h}`);
    s.setAttribute("role", "img");
    host.insertBefore(s, host.firstChild);
    return s;
  }
  function add(p, tag, attrs, text) {
    const n = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    p.appendChild(n);
    return n;
  }
  function line(p, x1, y1, x2, y2, cls) {
    return add(p, "line", { x1, y1, x2, y2, class: cls || "ax" });
  }

  // 一条固定的示意信号：两段抓挠（6–11 s、14–17 s）
  function bur(t, a, b) {
    if (t < a || t >= b) return 0;
    return Math.min(1, (t - a) / 0.6, (b - t) / 0.6);
  }
  function sig(t) {
    let v = 0.10 * Math.sin(2 * Math.PI * 0.55 * t) + 0.05 * Math.sin(2 * Math.PI * 1.7 * t + 1);
    v += bur(t, 6, 11) * (0.52 * Math.sin(2 * Math.PI * 4.2 * t) + 0.22 * Math.sin(2 * Math.PI * 7.1 * t));
    v += bur(t, 14, 17) * (0.46 * Math.sin(2 * Math.PI * 3.6 * t + 2) + 0.18 * Math.sin(2 * Math.PI * 6.4 * t));
    return v;
  }
  function wavePath(x0, x1, ymid, amp, tmax) {
    let d = "";
    const n = 600;
    for (let i = 0; i <= n; i++) {
      const t = (i / n) * tmax;
      const x = x0 + (t / tmax) * (x1 - x0);
      const y = ymid - sig(t) * amp;
      d += (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    }
    return d;
  }

  /* ── 图 1：同一段信号，三层口径 ── */
  function figLevels(host) {
    const W = 760, H = 352, x0 = 200, x1 = 736, T = 20;
    const X = (t) => x0 + (t / T) * (x1 - x0);
    const svg = root(host, W, H);
    add(svg, "text", { x: 8, y: 18, class: "ttl" }, "同一段 20 s 信号（6–11 s、14–17 s 两次抓挠），三种问法");
    // 每层左边一条口径色竖条 + 口径色标题
    function lvHead(y, col, t1, t2, t3) {
      add(svg, "rect", { x: 0, y: y - 16, width: 4, height: 56, rx: 2, fill: col });
      add(svg, "text", { x: 12, y, class: "ttl", style: `fill:${col};font-weight:700` }, t1);
      add(svg, "text", { x: 12, y: y + 16, class: "small" }, t2);
      add(svg, "text", { x: 12, y: y + 31, class: "small" }, t3);
    }

    // ① 窗口级：每个 4 s 窗一个判决，橙 = 判成抓挠，灰 = 判成活动；判错的窗红框
    lvHead(52, C.win, "① 窗口级", "每 4 s 窗一个样本", "看每窗判得对不对");
    const wy = 36, wh = 62;
    add(svg, "path", { d: wavePath(x0, x1, wy + wh / 2 + 6, 22, T), fill: "none", stroke: C.mute, "stroke-width": 1.1, opacity: 0.7 });
    const winPred = ["活动", "抓挠", "抓挠", "活动", "活动"];   // 第 4 个窗故意判错，示意窗口级怎么扣分
    for (let k = 0; k < 5; k++) {
      const t = k * 4;
      const sc = winPred[k] === "抓挠";
      const truth = Math.max(0, Math.min(t + 4, 11) - Math.max(t, 6)) + Math.max(0, Math.min(t + 4, 17) - Math.max(t, 14)) >= 2;
      const wrong = sc !== truth;
      add(svg, "rect", { x: X(t) + 1, y: wy, width: X(t + 4) - X(t) - 3, height: wh, rx: 3,
        fill: sc ? rgba(C.pred, 0.22) : rgba(C.mute, 0.08),
        stroke: wrong ? C.miss : (sc ? C.pred : rgba(C.mute, 0.5)), "stroke-width": wrong ? 2 : 1 });
      add(svg, "text", { x: X(t) + 7, y: wy + 14, class: "small", style: `fill:${sc ? C.pred : C.mute}` }, winPred[k] + (wrong ? " ✗" : " ✓"));
    }
    add(svg, "text", { x: x1, y: 30, "text-anchor": "end", class: "small", style: `fill:${C.win}` }, "实测 87.7% 窗口判对");

    // ② 片段级：人标段（蓝），段内平均后主类对就算对
    lvHead(140, C.seg, "② 片段级（宽松）", "人标一段取平均", "主类对就算整段对");
    const ry = 122;
    add(svg, "path", { d: wavePath(x0, x1, ry + 30, 18, T), fill: "none", stroke: C.mute, "stroke-width": 1, opacity: 0.45 });
    [[6, 11], [14, 17]].forEach(([a, b]) => {
      box(svg, X(a), ry, X(b) - X(a), 60, C.gt, 0.22);
      add(svg, "text", { x: (X(a) + X(b)) / 2, y: ry + 20, "text-anchor": "middle", class: "small", style: `fill:${C.text}` }, `人标抓挠 ${b - a} s`);
      add(svg, "text", { x: (X(a) + X(b)) / 2, y: ry + 46, "text-anchor": "middle", class: "small", style: `fill:${C.hit};font-weight:700` }, "✓ 判对");
    });
    add(svg, "text", { x: x1, y: 118, "text-anchor": "end", class: "small", style: `fill:${C.seg}` }, "实测抓挠 F1 0.939");

    // ③ 事件级：真值（蓝）对预测（橙），交集绿，漏检红虚框，多报紫
    lvHead(230, C.evt, "③ 事件级（严格）", "同类 + IoU≥0.5 + 一对一", "多报 / 切碎 / 边界差都扣分");
    const gy = 206, gh = 24, py = 246;
    add(svg, "text", { x: x0 - 6, y: gy + 16, "text-anchor": "end", class: "small", style: `fill:${C.gt}` }, "真值");
    add(svg, "text", { x: x0 - 6, y: py + 16, "text-anchor": "end", class: "small", style: `fill:${C.pred}` }, "预测");
    box(svg, X(6), gy, X(11) - X(6), gh, C.gt, 0.4);
    box(svg, X(14), gy, X(17) - X(14), gh, C.miss, 0.12, { "stroke-dasharray": "5 3", "stroke-width": 2 });
    add(svg, "text", { x: (X(14) + X(17)) / 2, y: gy + 16, "text-anchor": "middle", class: "small", style: `fill:${C.miss};font-weight:700` }, "漏检");
    box(svg, X(6.5), py, X(11.4) - X(6.5), gh, C.pred, 0.4);
    box(svg, X(13.1), py, X(13.9) - X(13.1), gh, C.pred, 0.4);
    box(svg, X(18.2), py, X(19.6) - X(18.2), gh, C.fp, 0.4);
    add(svg, "rect", { x: X(6.5), y: gy + gh, width: X(11) - X(6.5), height: py - gy - gh, fill: rgba(C.hit, 0.75) });
    add(svg, "text", { x: (X(6.5) + X(11)) / 2, y: gy - 6, "text-anchor": "middle", class: "small", style: `fill:${C.hit};font-weight:700` }, "IoU 0.87 → 命中");
    add(svg, "text", { x: (X(13.1) + X(13.9)) / 2, y: py + gh + 16, "text-anchor": "middle", class: "small", style: `fill:${C.miss}` }, "太短，IoU 不够");
    add(svg, "text", { x: (X(18.2) + X(19.6)) / 2, y: py + gh + 16, "text-anchor": "middle", class: "small", style: `fill:${C.fp};font-weight:700` }, "多报");
    add(svg, "text", { x: x1, y: 200, "text-anchor": "end", class: "small", style: `fill:${C.evt}` }, "实测抓挠 F1 0.544");
    // 时间轴
    const ay = 304;
    line(svg, x0, ay, x1, ay);
    for (let t = 0; t <= T; t += 5) {
      line(svg, X(t), ay, X(t), ay + 5);
      add(svg, "text", { x: X(t), y: ay + 18, "text-anchor": "middle", class: "small" }, t + " s");
    }
    legend(svg, 12, 344, [[C.gt, "真值"], [C.pred, "预测"], [C.hit, "交集 / 命中"], [C.miss, "漏检 / 判错"], [C.fp, "多报"]]);
  }

  /* ── 图 2：IoU 交互 ── */
  function figIou(host) {
    const W = 760, H = 196, x0 = 64, x1 = 736, T = 24;
    const X = (t) => x0 + (t / T) * (x1 - x0);
    const svg = root(host, W, H);
    add(svg, "text", { x: 8, y: 16, class: "ttl" }, "事件匹配：IoU = 交集 / 并集，同类且 ≥0.5 才算命中");

    const gy = 34, gh = 26, py = 84;
    add(svg, "text", { x: x0 - 6, y: gy + 17, "text-anchor": "end", class: "small", style: `fill:${C.gt}` }, "真值");
    add(svg, "text", { x: x0 - 6, y: py + 17, "text-anchor": "end", class: "small", style: `fill:${C.pred}` }, "预测");

    const g = add(svg, "rect", { y: gy, height: gh, rx: 3, fill: rgba(C.gt, 0.4), stroke: C.gt, "stroke-width": 1.2 });
    const p = add(svg, "rect", { y: py, height: gh, rx: 3, fill: rgba(C.pred, 0.4), stroke: C.pred, "stroke-width": 1.2 });
    const ig = add(svg, "rect", { y: gy, height: gh, fill: rgba(C.hit, 0.55) });
    const ip = add(svg, "rect", { y: py, height: gh, fill: rgba(C.hit, 0.55) });
    const inter = add(svg, "rect", { y: gy + gh, height: py - gy - gh, fill: rgba(C.hit, 0.7) });
    const gTxt = add(svg, "text", { y: gy + 17, "text-anchor": "middle", class: "small", style: `fill:${C.text}` });
    const pTxt = add(svg, "text", { y: py + 17, "text-anchor": "middle", class: "small", style: `fill:${C.text}` });
    const bandTxt = add(svg, "text", { "text-anchor": "middle", class: "small" });
    const ay = 132;
    line(svg, x0, ay, x1, ay);
    for (let t = 0; t <= T; t += 4) {
      line(svg, X(t), ay, X(t), ay + 5);
      add(svg, "text", { x: X(t), y: ay + 17, "text-anchor": "middle", class: "small" }, t + " s");
    }

    const wrap = document.createElement("div");
    wrap.className = "ctl";
    const defs = [["真值起点", "gs", 0, 22, 0.1, 4], ["真值时长", "gd", 1, 12, 0.1, 6],
                  ["预测起点", "ps", 0, 22, 0.1, 5], ["预测时长", "pd", 1, 14, 0.1, 6]];
    const inputs = {};
    defs.forEach(([lab, id, min, max, step, val]) => {
      const l = document.createElement("label");
      l.textContent = lab;
      const i = document.createElement("input");
      Object.assign(i, { type: "range", min, max, step, value: val, id });
      const o = document.createElement("output");
      l.appendChild(i); l.appendChild(o);
      wrap.appendChild(l);
      inputs[id] = { i, o };
    });
    const presets = document.createElement("div");
    presets.className = "ctl";
    const presetDefs = [["标准命中", 4, 6, 5, 6], ["边界偏差", 4, 6, 2.5, 5], ["长度翻倍", 4, 6, 2, 10], ["完全错位", 4, 6, 14, 5]];
    presetDefs.forEach(([lab, gs, gd, ps, pd]) => {
      const b = document.createElement("button");
      b.className = "btn";
      b.textContent = lab;
      b.onclick = () => {
        inputs.gs.i.value = gs; inputs.gd.i.value = gd; inputs.ps.i.value = ps; inputs.pd.i.value = pd; draw();
      };
      presets.appendChild(b);
    });
    const out = document.createElement("div");
    out.className = "readout";
    host.appendChild(wrap); host.appendChild(presets); host.appendChild(out);

    function draw() {
      const gs = +inputs.gs.i.value, gd = +inputs.gd.i.value;
      const ps = +inputs.ps.i.value, pd = +inputs.pd.i.value;
      const ge = gs + gd, pe = ps + pd;
      const ix = Math.max(0, Math.min(ge, pe) - Math.max(gs, ps));
      const un = (ge - gs) + (pe - ps) - ix;
      const iou = ix / Math.max(un, 1e-6);
      const hit = iou >= 0.5;

      g.setAttribute("x", X(gs)); g.setAttribute("width", Math.max(1, X(ge) - X(gs)));
      p.setAttribute("x", X(ps)); p.setAttribute("width", Math.max(1, X(pe) - X(ps)));
      const ia = Math.max(gs, ps);
      // 交集：过线绿、不过线红，一眼看出命中没有
      const ic = hit ? C.hit : C.miss;
      [[ig, 0.3], [ip, 0.3], [inter, 0.75]].forEach(([r, a]) => {
        r.setAttribute("x", X(ia));
        r.setAttribute("width", Math.max(0, X(ia + ix) - X(ia)));
        r.setAttribute("opacity", ix > 0 ? 1 : 0);
        r.setAttribute("fill", rgba(ic, a));
      });
      gTxt.setAttribute("x", X(gs + gd / 2)); gTxt.setAttribute("y", gy + 17); gTxt.textContent = `真值 ${gs.toFixed(1)}–${ge.toFixed(1)} s`;
      pTxt.setAttribute("x", X(ps + pd / 2)); pTxt.setAttribute("y", py + 17); pTxt.textContent = `预测 ${ps.toFixed(1)}–${pe.toFixed(1)} s`;
      bandTxt.setAttribute("x", X(Math.max(gs, ps) + ix / 2)); bandTxt.setAttribute("y", gy + gh + 17);
      bandTxt.textContent = ix > 0 ? `交集 ${ix.toFixed(1)} s` : "无交集";
      bandTxt.style.fill = hit ? C.hit : C.miss;
      bandTxt.style.fontWeight = 700;

      inputs.gs.o.textContent = gs.toFixed(1) + " s";
      inputs.gd.o.textContent = gd.toFixed(1) + " s";
      inputs.ps.o.textContent = ps.toFixed(1) + " s";
      inputs.pd.o.textContent = pd.toFixed(1) + " s";
      out.innerHTML = `IoU <b>${iou.toFixed(2)}</b>　并集 <b>${un.toFixed(1)} s</b>　` +
        (hit ? `<span class="pill ok">命中</span>` : `<span class="pill bad">不命中</span>`) +
        `　<span class="muted">同一个真值只能配一个预测；多出来的预测算误报。</span>`;
    }
    Object.values(inputs).forEach(({ i }) => i.addEventListener("input", draw));
    draw();
  }

  /* ── 图 3：窗口 / 步长交互 ── */
  function figWindow(host) {
    const W = 760, H = 318, x0 = 64, x1 = 736, T = 20;
    const X = (t) => x0 + (t / T) * (x1 - x0);
    const svg = root(host, W, H);
    add(svg, "text", { x: 8, y: 16, class: "ttl" }, "拖动滑杆：窗口 W 与步长 S 怎么影响切片、延迟和计算量");
    add(svg, "text", { x: 8, y: 34, class: "small" }, "每一行是一个窗口；行与行的错位就是重叠");

    const gWindows = add(svg, "g", {});
    const play = line(svg, 0, 40, 0, 300, "ax");
    play.setAttribute("stroke", C.cur);
    play.setAttribute("stroke-width", 2);
    play.setAttribute("opacity", 0.95);
    // 信号：抓挠的两段底下垫蓝色（真值），让窗口和动作的关系看得出来
    [[6, 11], [14, 17]].forEach(([a, b]) => add(svg, "rect", { x: X(a), y: 240, width: X(b) - X(a), height: 56, fill: rgba(C.gt, 0.16) }));
    add(svg, "path", { d: wavePath(x0, x1, 268, 24, T), fill: "none", stroke: C.mute, "stroke-width": 1.1, opacity: 0.85 });
    add(svg, "text", { x: 8, y: 230, class: "small" }, "信号");
    const ay = 300;
    line(svg, x0, ay, x1, ay);
    for (let t = 0; t <= T; t += 5) {
      line(svg, X(t), ay, X(t), ay + 5);
      add(svg, "text", { x: X(t), y: ay + 17, "text-anchor": "middle", class: "small" }, t + " s");
    }

    const ctl = document.createElement("div");
    ctl.className = "ctl";
    function mkSlider(lab, min, max, step, val) {
      const l = document.createElement("label");
      l.textContent = lab;
      const i = document.createElement("input");
      Object.assign(i, { type: "range", min, max, step, value: val });
      const o = document.createElement("output");
      l.appendChild(i); l.appendChild(o);
      ctl.appendChild(l);
      return { i, o, l };
    }
    const wCtl = mkSlider("窗口 W", 1, 10, 0.5, 4);
    const sCtl = mkSlider("步长 S", 0.25, 4, 0.25, 0.5);
    const btnPlay = document.createElement("button");
    btnPlay.className = "btn";
    btnPlay.textContent = "播放";
    const btn4 = document.createElement("button");
    btn4.className = "btn"; btn4.textContent = "4 s / 0.5 s";
    const btn8 = document.createElement("button");
    btn8.className = "btn"; btn8.textContent = "8 s / 1 s";
    ctl.appendChild(btnPlay); ctl.appendChild(btn4); ctl.appendChild(btn8);
    const out = document.createElement("div");
    out.className = "readout";
    host.appendChild(ctl); host.appendChild(out);

    let tplay = 0, playing = false, last = 0, raf = 0;

    function draw() {
      let w = +wCtl.i.value, s = +sCtl.i.value;
      if (s > w) { s = w; sCtl.i.value = w; }
      wCtl.o.textContent = w.toFixed(2).replace(/0$/, "") + " s";
      sCtl.o.textContent = s.toFixed(2).replace(/0$/, "") + " s";
      gWindows.innerHTML = "";
      const rows = Math.floor((T - w) / s) + 1;
      const shown = Math.min(rows, 34);
      const rowH = Math.min(5.4, (208 - 44) / Math.max(shown, 1));
      for (let k = 0; k < shown; k++) {
        const t = k * s;
        const y = 44 + k * rowH;
        const active = tplay >= t && tplay < t + s;
        // 相邻两行青 / 紫交替，重叠一眼看出；正在算的那个窗亮黄
        const col = active ? C.cur : (k % 2 ? C.fp : C.win);
        add(gWindows, "rect", { x: X(t), y: y, width: Math.max(1, X(t + w) - X(t)), height: Math.max(2, rowH - 1.2), rx: 1.5,
          fill: rgba(col, active ? 0.85 : 0.38), stroke: col, "stroke-width": active ? 1.2 : 0.5 });
      }
      if (rows > shown) add(gWindows, "text", { x: x1, y: 44 + shown * rowH + 12, "text-anchor": "end", class: "small" }, `只画前 ${shown} 个窗口（共 ${rows} 个）`);
      play.setAttribute("x1", X(Math.min(tplay, T)));
      play.setAttribute("x2", X(Math.min(tplay, T)));

      const ov = 1 - s / w;
      out.innerHTML =
        `重叠 <b>${(ov * 100).toFixed(1)}%</b>` +
        `　每秒窗口 <b>${(1 / s).toFixed(1)}</b>` +
        `　20 s 内 <b>${rows}</b> 个` +
        `　首次判决延迟 ≈ <b>${w.toFixed(1)} s</b>` +
        `　最短事件 ≈ <b>${(2 * s).toFixed(2)} s</b>` +
        `　同段信号重复计算 <b>${(w / s).toFixed(1)}×</b>`;
    }

    function tick(ts) {
      if (!playing) return;
      if (!last) last = ts;
      const dt = Math.min(0.1, (ts - last) / 1000) * 4;
      last = ts;
      tplay += dt;
      if (tplay > T) tplay = 0;
      draw();
      raf = requestAnimationFrame(tick);
    }
    function toggle() {
      playing = !playing;
      btnPlay.textContent = playing ? "暂停" : "播放";
      btnPlay.classList.toggle("on", playing);
      last = 0;
      if (playing) raf = requestAnimationFrame(tick); else cancelAnimationFrame(raf);
    }
    btnPlay.onclick = toggle;
    btn4.onclick = () => { wCtl.i.value = 4; sCtl.i.value = 0.5; draw(); };
    btn8.onclick = () => { wCtl.i.value = 8; sCtl.i.value = 1; draw(); };
    wCtl.i.addEventListener("input", draw);
    sCtl.i.addEventListener("input", draw);
    draw();
  }

  /* ── 图 4：按段长分桶（job60 实测） ── */
  function figLen(host) {
    const W = 760, H = 236, x0 = 250, x1 = 716;
    const data = [["< 3 s", 1027, 0.706, 0.318], ["3–6 s", 1768, 0.903, 0.631],
                  ["6–10 s", 1293, 0.956, 0.776], ["≥ 10 s", 1219, 0.964, 0.804]];
    const X = (v) => x0 + v * (x1 - x0);
    const svg = root(host, W, H);
    add(svg, "text", { x: 8, y: 16, class: "ttl" }, "同一份 4 s / 0.5 s 模型：真值越短，越难命中");
    legend(svg, 8, 38, [[C.gt, "报出来了（有重叠）"], [C.hit, "起止也对（同类且 IoU≥0.5）"]]);
    // 网格
    for (let v = 0; v <= 1.0001; v += 0.25) {
      line(svg, X(v), 52, X(v), 204, "ax");
      add(svg, "text", { x: X(v), y: 220, "text-anchor": "middle", class: "small" }, Math.round(v * 100) + "%");
    }
    data.forEach(([lab, n, hit, iou], i) => {
      const y = 60 + i * 36;
      add(svg, "text", { x: 8, y: y + 11, class: "small", style: `fill:${C.text};font-weight:700` }, lab);
      add(svg, "text", { x: 8, y: y + 26, class: "small" }, `${n} 段`);
      add(svg, "rect", { x: X(0), y: y, width: X(hit) - X(0), height: 12, rx: 2, fill: C.gt });
      add(svg, "rect", { x: X(0), y: y + 14, width: X(iou) - X(0), height: 12, rx: 2, fill: C.hit });
      // 两条之差 = 报了但起止不对，红色标出来
      add(svg, "rect", { x: X(iou), y: y + 14, width: X(hit) - X(iou), height: 12, rx: 2, fill: rgba(C.miss, 0.25), stroke: C.miss, "stroke-dasharray": "3 2", "stroke-width": 1 });
      add(svg, "text", { x: X(hit) + 6, y: y + 10, class: "small", style: `fill:${C.gt}` }, (hit * 100).toFixed(0) + "%");
      add(svg, "text", { x: X(hit) + 6, y: y + 25, class: "small", style: `fill:${C.hit};font-weight:700` }, (iou * 100).toFixed(0) + "%");
    });
    add(svg, "rect", { x: 8, y: 223, width: 14, height: 10, rx: 2, fill: rgba(C.miss, 0.25), stroke: C.miss, "stroke-dasharray": "3 2" });
    add(svg, "text", { x: 27, y: 232, class: "small" }, "红虚框 = 报出来了但起止不对。3 s 以下：报出 71%，起止对的只有 32%。");
  }

  /* ── 图 5：many-to-one vs many-to-many ── */
  function figMtm(host) {
    const W = 760, H = 214;
    const svg = root(host, W, H);
    add(svg, "text", { x: 8, y: 16, class: "ttl" }, "两种输出方式：每窗一个标签（左）vs 每个时间点一个标签（右）");
    // 左
    add(svg, "text", { x: 8, y: 40, class: "ttl" }, "many-to-one（我们现在）");
    const labs = ["活动", "抓挠", "抓挠", "抓挠", "活动", "活动"];
    labs.forEach((l, i) => {
      const x = 8 + i * 58;
      const sc = l === "抓挠";
      box(svg, x, 50, 52, 24, sc ? C.pred : C.mute, sc ? 0.35 : 0.1);
      add(svg, "text", { x: x + 26, y: 66, "text-anchor": "middle", class: "small", style: `fill:${sc ? C.text : C.mute}` }, l);
    });
    add(svg, "text", { x: 8, y: 92, class: "small" }, "每 0.5 s 一个窗，窗长 4 s");
    add(svg, "text", { x: 8, y: 116, class: "small" }, "↓ Viterbi + 最短时长 + 间隔合并");
    box(svg, 8, 126, 172, 22, C.hit, 0.3);
    add(svg, "text", { x: 94, y: 141, "text-anchor": "middle", class: "small", style: `fill:${C.text}` }, "事件：抓挠 6.0–11.0 s");
    add(svg, "text", { x: 8, y: 172, class: "small" }, "简单、好导出；边界受窗长限制");
    add(svg, "text", { x: 8, y: 188, class: "small" }, "相邻窗高度相关（重叠 87.5%）");

    // 分隔
    line(svg, 380, 34, 380, 196, "ax");

    // 右
    add(svg, "text", { x: 400, y: 40, class: "ttl" }, "many-to-many（逐点）");
    const cell = 13.5;
    for (let i = 0; i < 25; i++) {
      const on = i >= 10 && i < 18;
      add(svg, "rect", { x: 400 + i * cell, y: 50, width: cell - 1.2, height: 24,
        fill: on ? rgba(C.pred, 0.45) : rgba(C.mute, 0.1),
        stroke: on ? C.pred : rgba(C.mute, 0.3), "stroke-width": 0.6 });
    }
    add(svg, "text", { x: 400, y: 92, class: "small" }, "每个 0.5 s（或每个点）都输出类别");
    add(svg, "text", { x: 400, y: 116, class: "small" }, "↓ 直接取连续同类的段");
    box(svg, 400, 126, 172, 22, C.hit, 0.3);
    add(svg, "text", { x: 486, y: 141, "text-anchor": "middle", class: "small", style: `fill:${C.text}` }, "事件：抓挠 5.0–9.0 s");
    add(svg, "text", { x: 400, y: 172, class: "small" }, "边界准、上下文长；要稠密标签");
    add(svg, "text", { x: 400, y: 188, class: "small" }, "端侧要流式状态，算力更大");
  }

  function boot() {
    const f1 = document.getElementById("fig-levels"); if (f1) figLevels(f1);
    const f2 = document.getElementById("fig-iou"); if (f2) figIou(f2);
    const f3 = document.getElementById("fig-window"); if (f3) figWindow(f3);
    const f4 = document.getElementById("fig-len"); if (f4) figLen(f4);
    const f5 = document.getElementById("fig-mtm"); if (f5) figMtm(f5);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
