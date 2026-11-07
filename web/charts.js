/* Minimal SVG chart kit for the terminal. No external dependencies. */
const C = (() => {
  const NS = 'http://www.w3.org/2000/svg';
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const nice = (lo, hi, n = 4) => {
    if (!isFinite(lo) || !isFinite(hi)) return [0, 1];
    if (lo === hi) { lo -= 1; hi += 1; }
    const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n)));
    const err = (hi - lo) / n / step;
    const s = step * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const t = []; for (let v = Math.floor(lo / s) * s; v <= hi + 1e-9; v += s) t.push(+v.toFixed(10));
    return t;
  };
  const empty = (w, h, msg) => `<svg viewBox="0 0 ${w} ${h}" width="100%"><text x="${w/2}" y="${h/2}" text-anchor="middle">${esc(msg)}</text></svg>`;

  /* line(series=[{name,color,points:[[x,y]]}], opts) */
  function line(series, o = {}) {
    const w = o.w || 360, h = o.h || 150, m = {l: 34, r: 8, t: 8, b: 20};
    const pts = series.flatMap(s => s.points.filter(p => p[1] != null && isFinite(p[1])));
    if (!pts.length) return empty(w, h, o.empty || 'No data');
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    let y0 = Math.min(...ys, o.y0 ?? Infinity), y1 = Math.max(...ys, o.y1 ?? -Infinity);
    const yt = nice(y0, y1); y0 = yt[0]; y1 = yt[yt.length - 1];
    const x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
    const X = x => m.l + (x - x0) / (x1 - x0) * (w - m.l - m.r), Y = y => h - m.b - (y - y0) / (y1 - y0) * (h - m.t - m.b);
    let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
    yt.forEach(v => s += `<line x1="${m.l}" x2="${w - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1d2a58" stroke-width="1"/><text x="${m.l - 4}" y="${Y(v) + 3}" text-anchor="end">${o.fmt ? o.fmt(v) : v}</text>`);
    if (o.zero && y0 < 0 && y1 > 0) s += `<line x1="${m.l}" x2="${w - m.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="#3a4a85" stroke-dasharray="3 3"/>`;
    const xl = [...new Set(xs)].sort((a, b) => a - b); const every = Math.ceil(xl.length / 10);
    xl.forEach((x, i) => { if (i % every === 0) s += `<text x="${X(x)}" y="${h - 5}" text-anchor="middle">${o.xlab ? o.xlab(x) : x}</text>`; });
    series.forEach(se => {
      const p = se.points.filter(q => q[1] != null && isFinite(q[1]));
      if (!p.length) return;
      s += `<polyline fill="none" stroke="${se.color}" stroke-width="1.8" points="${p.map(q => `${X(q[0])},${Y(q[1])}`).join(' ')}"/>`;
      p.forEach(q => s += `<circle cx="${X(q[0])}" cy="${Y(q[1])}" r="2.4" fill="${se.color}"><title>${esc(se.name)} ${q[0]}: ${(+q[1]).toFixed(3)}</title></circle>`);
    });
    return s + '</svg>';
  }

  /* bars: stacked/side histogram. groups=[{name,color,values[]}], labels[] */
  function bars(labels, groups, o = {}) {
    const w = o.w || 360, h = o.h || 150, m = {l: 30, r: 6, t: 8, b: 22};
    const n = labels.length; if (!n) return empty(w, h, 'No data');
    const tot = labels.map((_, i) => groups.reduce((a, g) => a + (o.stack === false ? 0 : (g.values[i] || 0)), 0));
    const ymax = o.stack === false ? Math.max(...groups.flatMap(g => g.values)) : Math.max(...tot);
    const yt = nice(0, ymax || 1, 4); const y1 = yt[yt.length - 1];
    const bw = (w - m.l - m.r) / n;
    const Y = v => h - m.b - v / y1 * (h - m.t - m.b);
    let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
    yt.forEach(v => s += `<line x1="${m.l}" x2="${w - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1d2a58"/><text x="${m.l - 3}" y="${Y(v) + 3}" text-anchor="end">${o.fmt ? o.fmt(v) : v}</text>`);
    labels.forEach((lb, i) => {
      let acc = 0;
      groups.forEach((g, gi) => {
        const v = g.values[i] || 0;
        const x = m.l + i * bw + (o.stack === false ? gi * bw / groups.length : 0) + 1;
        const ww = (o.stack === false ? bw / groups.length : bw) - 2;
        const yTop = o.stack === false ? Y(v) : Y(acc + v), yBot = o.stack === false ? Y(0) : Y(acc);
        s += `<rect x="${x}" y="${yTop}" width="${Math.max(ww, 1)}" height="${Math.max(yBot - yTop, 0)}" fill="${g.color}"><title>${esc(g.name)} ${esc(lb)}: ${o.fmt ? o.fmt(v) : v}</title></rect>`;
        if (o.stack !== false) acc += v;
      });
      const every = Math.ceil(n / (o.maxLabels || 14));
      if (i % every === 0) s += `<text x="${m.l + i * bw + bw / 2}" y="${h - 6}" text-anchor="middle">${esc(lb)}</text>`;
    });
    if (o.vline != null) { const i = labels.findIndex(l => +l >= o.vline); if (i >= 0) s += `<line x1="${m.l + i * bw}" x2="${m.l + i * bw}" y1="${m.t}" y2="${h - m.b}" stroke="#cfd6ff" stroke-dasharray="3 3"/>`; }
    return s + '</svg>';
  }

  /* density curves from histogram counts (two series) */
  function density(edges, series, o = {}) {
    const w = o.w || 360, h = o.h || 150, m = {l: 30, r: 6, t: 8, b: 22};
    const mids = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
    const tot = series.reduce((a, s) => a + s.values.reduce((x, y) => x + y, 0), 0) || 1;
    const ymax = Math.max(...series.flatMap(s => s.values)) / tot;
    const yt = nice(0, ymax, 4), y1 = yt[yt.length - 1];
    const X = x => m.l + (x - edges[0]) / (edges[edges.length - 1] - edges[0]) * (w - m.l - m.r);
    const Y = v => h - m.b - v / y1 * (h - m.t - m.b);
    let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
    yt.forEach(v => s += `<line x1="${m.l}" x2="${w - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1d2a58"/><text x="${m.l - 3}" y="${Y(v) + 3}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`);
    series.forEach(se => {
      const p = mids.map((x, i) => [X(x), Y(se.values[i] / tot)]);
      s += `<path d="M${X(mids[0])},${Y(0)} L${p.map(q => q.join(',')).join(' L')} L${X(mids[mids.length - 1])},${Y(0)} Z" fill="${se.color}" fill-opacity=".35" stroke="${se.color}" stroke-width="1.6"/>`;
    });
    nice(edges[0], edges[edges.length - 1], 8).forEach(v => { if (v >= edges[0] && v <= edges[edges.length - 1]) s += `<text x="${X(v)}" y="${h - 6}" text-anchor="middle">${v}</text>`; });
    if (o.vline != null) s += `<line x1="${X(o.vline)}" x2="${X(o.vline)}" y1="${m.t}" y2="${h - m.b}" stroke="#cfd6ff" stroke-dasharray="3 3"/>`;
    return s + '</svg>';
  }

  /* calibration scatter with 45-degree line */
  function calib(points, o = {}) {
    const w = o.w || 260, h = o.h || 190, m = {l: 30, r: 8, t: 8, b: 26};
    if (!points || !points.length) return empty(w, h, 'No stored calibration');
    const X = v => m.l + v * (w - m.l - m.r), Y = v => h - m.b - v * (h - m.t - m.b);
    let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
    [0, .2, .4, .6, .8, 1].forEach(v => s += `<line x1="${X(0)}" x2="${X(1)}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1d2a58"/><text x="${m.l - 3}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(1)}</text><text x="${X(v)}" y="${h - 12}" text-anchor="middle">${v.toFixed(1)}</text>`);
    s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="#5967a8" stroke-dasharray="4 3"/>`;
    s += `<polyline fill="none" stroke="${o.color || '#7b5cff'}" stroke-width="1.6" points="${points.map(p => `${X(p.pred)},${Y(p.obs)}`).join(' ')}"/>`;
    points.forEach(p => s += `<circle cx="${X(p.pred)}" cy="${Y(p.obs)}" r="${2 + Math.min(4, Math.sqrt(p.n) / 4)}" fill="${o.color || '#7b5cff'}"><title>pred ${p.pred.toFixed(3)} obs ${p.obs.toFixed(3)} n=${p.n}</title></circle>`);
    s += `<text x="${w / 2}" y="${h - 1}" text-anchor="middle">Predicted probability</text>`;
    return s + '</svg>';
  }

  /* radar of percentiles: axes=[{label,value(0-100)}] for up to two series */
  function radar(axes, series, o = {}) {
    const w = o.w || 260, h = o.h || 230, cx = w / 2, cy = h / 2 + 4, R = Math.min(w, h) / 2 - 34;
    const n = axes.length; if (!n) return empty(w, h, 'No data');
    const pt = (i, v) => [cx + R * v / 100 * Math.sin(2 * Math.PI * i / n), cy - R * v / 100 * Math.cos(2 * Math.PI * i / n)];
    let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-height:${o.maxh || 300}px">`;
    [25, 50, 75, 100].forEach(r => s += `<polygon fill="none" stroke="#1d2a58" points="${axes.map((_, i) => pt(i, r).join(',')).join(' ')}"/>`);
    axes.forEach((a, i) => { const [x, y] = pt(i, 118); s += `<line x1="${cx}" y1="${cy}" x2="${pt(i, 100)[0]}" y2="${pt(i, 100)[1]}" stroke="#1d2a58"/><text x="${x}" y="${y + 3}" text-anchor="middle">${esc(a)}</text>`; });
    series.forEach(se => s += `<polygon fill="${se.color}" fill-opacity=".22" stroke="${se.color}" stroke-width="1.5" points="${se.values.map((v, i) => pt(i, v ?? 0).join(',')).join(' ')}"><title>${esc(se.name)}</title></polygon>`);
    return s + '</svg>';
  }

  /* percentile bar html */
  function pbar(p) {
    if (p == null) return '<div class="bar"></div>';
    const col = p >= 80 ? 'var(--green)' : p >= 60 ? '#5fc98f' : p >= 40 ? 'var(--blue)' : p >= 20 ? 'var(--amber)' : 'var(--red)';
    return `<div class="bar" title="${p.toFixed(0)}th pctl"><i style="width:${Math.max(2, p)}%;background:${col}"></i></div>`;
  }

  /* depth heat grid for throw chart */
  function heatColor(v, lo, hi) {
    if (v == null) return '#141c3d';
    const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    const r = Math.round(20 + 120 * t), g = Math.round(60 + 150 * t), b = Math.round(60 - 20 * t);
    return `rgb(${r},${g},${b})`;
  }
  return {line, bars, density, calib, radar, pbar, heatColor, esc, css};
})();
