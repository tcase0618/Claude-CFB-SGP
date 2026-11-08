/* CFB Quant Intelligence Terminal — vanilla JS SPA over the FastAPI backend. */
const S = {meta: null, season: null, conf: '', teams: null, cache: {}};
const $ = s => document.querySelector(s);
const E = C.esc;
const HOME = '#7b5cff', AWAY = '#e0466e', OFFC = '#ff4d6d', DEFC = '#4da3ff';

async function api(path) {
  if (S.cache[path]) return S.cache[path];
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}: ${(await r.text()).slice(0, 200)}`);
  const j = await r.json();
  if (!path.startsWith('/api/forecast')) S.cache[path] = j;
  return j;
}
const qs = o => Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
const f = (v, d = 2) => v == null || !isFinite(v) ? '—' : (+v).toFixed(d);
const fs = (v, d = 3) => v == null || !isFinite(v) ? '—' : `${v > 0 ? '+' : ''}${(+v).toFixed(d)}`;
const pc = (v, d = 1) => v == null || !isFinite(v) ? '—' : `${(100 * v).toFixed(d)}%`;
const ord = p => { if (p == null || !isFinite(p)) return '—'; const n = Math.round(p); const s = ['th', 'st', 'nd', 'rd'], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
const cls = (v, dir = 1) => v == null ? '' : (v * dir > 0 ? 'pos' : v * dir < 0 ? 'neg' : '');
const tag = c => { const m = {observed: 'observed', calculated: 'calculated', estimated: 'estimated', unavailable_without_charting: 'unavailable', predicted: 'predicted'}; const k = m[c] || c; return `<span class="tag t-${k}">${E(k)}</span>`; };
const fmtMetric = (v, id = '') => {
  if (v == null) return '—';
  const a = Math.abs(v);
  if (/rate|neutral_pass|early_pass|_sr|pct|share|comp|catch|success|bluechip|returning|_tb|in20|avail|continuity|stop|_r10|expl|stuff|neg|_to$|int$|sack|fumble|scr|dbrate|deep_rate|go_|conv|pyth|predict|two_pt/.test(id) && a <= 1.0001 && !/epa/.test(id)) return pc(v);
  if (/epa|cpoe|croe|proe|slope|luck|_adj$|net_epa|sos|dev_adj|halftime|ryoe|yacoe|voe|poe|conversion/.test(id) && !/punt/.test(id)) return fs(v, 3);
  return a >= 100 ? f(v, 0) : a >= 10 ? f(v, 1) : f(v, 2);
};
const teamName = id => (S.teams || []).find(t => t.team_id === id)?.school || `#${id}`;
const teamAbbr = id => { const t = (S.teams || []).find(t => t.team_id === id); return t ? (t.abbreviation || t.school.slice(0, 4)).toUpperCase() : '#'; };
const initials = s => (s || '').split(/\s+/).map(w => w[0]).join('').slice(0, 3).toUpperCase();
const panel = (title, body, sub = '') => `<section class="panel"><h2>${title}${sub ? `<small>${sub}</small>` : ''}</h2>${body}</section>`;
const err = e => `<div class="panel err">${E(e.message || e)}</div>`;

/* ------------------------------------------------------------------ boot */
async function boot() {
  try {
    S.meta = await api('/api/meta');
  } catch (e) { $('#app').innerHTML = err(e); return; }
  const m = S.meta;
  const sel = $('#season');
  sel.innerHTML = m.seasons.map(s => `<option>${s}</option>`).join('');
  S.season = +(new URLSearchParams(location.hash.split('?')[1] || '').get('season') || m.seasons[m.seasons.length - 1]);
  sel.value = S.season;
  sel.onchange = () => { S.season = +sel.value; S.cache = {}; S.teams = null; loadTeams().then(route); };
  $('#conf').onchange = e => { S.conf = e.target.value; route(); };
  const st = $('#status');
  if (m.synthetic) {
    st.classList.add('synthetic'); $('#status-text').textContent = 'SYNTHETIC DEMO DATA';
    const b = $('#banner'); b.classList.remove('hidden');
    b.innerHTML = `<b>SYNTHETIC DEMONSTRATION DATA.</b> All schools, players and results are fictional, generated to exercise the full pipeline. ` +
      `No real college football data is shown. Set <code>CFBD_API_KEY</code> and run <code>python -m cfbquant.cli ingest</code> to load licensed real data.`;
  } else { $('#status-text').textContent = 'DATA: ' + (m.sources.map(s => s.source_id.toUpperCase()).join(', ') || '—'); }
  const prod = m.models.find(x => x.status === 'production' && x.family !== 'expected_points' && x.family !== 'sim_calibrator');
  $('#status-sub').textContent = `Refreshed ${(m.sources[0]?.last_refreshed || '').replace('T', ' ').slice(0, 16)} · prod model: ${prod ? prod.model_version : 'n/a'}`;
  $('#foot-left').textContent = `${m.counts.games} games · ${m.counts.plays.toLocaleString()} plays · ${m.counts.players.toLocaleString()} players`;
  await loadTeams();
  window.addEventListener('hashchange', route);
  route();
}
async function loadTeams() {
  const t = await api(`/api/teams?season=${S.season}`);
  S.teams = t.teams; S.week = t.through_week;
  const confs = [...new Set(S.teams.map(x => x.conf))].sort();
  $('#conf').innerHTML = '<option value="">All FBS</option>' + confs.map(c => `<option ${c === S.conf ? 'selected' : ''}>${c}</option>`).join('');
}
function parseHash() {
  const [p, q] = location.hash.replace(/^#\/?/, '').split('?');
  const parts = (p || 'dashboard').split('/');
  return {r: parts[0] || 'dashboard', id: parts[1], q: new URLSearchParams(q || '')};
}
async function route() {
  const h = parseHash();
  document.querySelectorAll('#tabs a').forEach(a => { a.classList.toggle('on', a.dataset.r === h.r); a.href = `#/${a.dataset.r}`; });
  const app = $('#app');
  app.innerHTML = '<div class="loading">Computing…</div>';
  const V = {dashboard, players: playersView, teams: teamsView, matchup: matchupView, predictions: predictionsView,
             simulations: simulationsView, markets: marketsView, sgp: sgpView, recruiting: recruitingView, transfers: transfersView, audit: auditView}[h.r] || dashboard;
  try { await V(app, h); } catch (e) { console.error(e); app.innerHTML = err(e); }
}

/* ------------------------------------------------------------------ shared blocks */
function featuredGame() {
  return api(`/api/games?season=${S.season}`).then(gg => {
    let g = gg;
    const real = g.games.filter(x => x.home_id > 0 && x.away_id > 0 && S.teams.some(t => t.team_id === x.home_id) && S.teams.some(t => t.team_id === x.away_id));
    const upcoming = real.filter(x => !x.completed);
    const wk = upcoming.length ? Math.min(...upcoming.map(x => x.week)) : Math.max(...real.map(x => x.week));
    g = {games: real};
    const gr = Object.fromEntries(S.teams.map(t => [t.team_id, t.net_epa ?? 0]));
    const cand = g.games.filter(x => x.week === wk && (!S.conf || [teamConf(x.home_id), teamConf(x.away_id)].includes(S.conf)));
    cand.sort((a, b) => (gr[b.home_id] + gr[b.away_id] - Math.abs(gr[b.home_id] - gr[b.away_id])) - (gr[a.home_id] + gr[a.away_id] - Math.abs(gr[a.home_id] - gr[a.away_id])));
    return cand[0] || g.games[g.games.length - 1];
  });
}
const teamConf = id => (S.teams.find(t => t.team_id === id) || {}).conf;

function predictionBlock(fc, g) {
  const hp = fc.home_win_prob, ap = 1 - hp;
  const res = g && g.completed ? `<div class="note">Final: ${E(fc.home.abbreviation || '')} ${g.home_points} – ${g.away_points} ${E(fc.away.abbreviation || '')} · forecast uses only data before week ${fc.week} (retrospective)</div>` : '';
  return `<div class="row between wrap">
    <div class="kpi big-home" style="flex:1"><div class="l">${E(teamAbbr(fc.home_id))} win prob.</div><div class="v">${pc(hp)}</div></div>
    <div class="kpi" style="flex:1.2"><div class="l">Projected score</div>
      <div class="teamline">${E(teamAbbr(fc.home_id))} ${f(fc.home_pts.mean, 0)} – ${f(fc.away_pts.mean, 0)} ${E(teamAbbr(fc.away_id))}</div>
      <div class="sub">Total ${f(fc.total.mean, 1)} · Spread ${E(teamAbbr(fc.margin.mean >= 0 ? fc.home_id : fc.away_id))} −${f(Math.abs(fc.margin.mean), 1)}</div></div>
    <div class="kpi big-away" style="flex:1"><div class="l">${E(teamAbbr(fc.away_id))} win prob.</div><div class="v">${pc(ap)}</div></div></div>
    <div class="sub" style="margin-top:4px">${fc.n_sims.toLocaleString()} sims · seed ${fc.seed} · ${fc.calibrated ? 'Platt-calibrated (raw ' + pc(fc.home_win_prob_raw) + ')' : 'uncalibrated (no stored calibrator)'} · ${tag('predicted')}</div>${res}`;
}
function scoreDist(fc, o = {}) {
  const e = fc.total_hist.edges;
  return `<div class="legend"><span><i style="background:${HOME}"></i>${E(teamAbbr(fc.home_id))} wins (${pc(fc.home_win_prob_raw)})</span><span><i style="background:${AWAY}"></i>${E(teamAbbr(fc.away_id))} wins</span></div>` +
    C.bars(e.slice(0, -1).map(String), [{name: 'home win', color: HOME, values: fc.total_hist.home_win.map(v => v / fc.n_sims)},
      {name: 'away win', color: AWAY, values: fc.total_hist.away_win.map(v => v / fc.n_sims)}], {h: o.h || 140, fmt: v => (v * 100).toFixed(0) + '%', vline: fc.total.mean});
}
function driversBlock(fc, n = 7) {
  if (!fc.drivers) return '';
  return `<table>${fc.drivers.slice(0, n).map(d => {
    const fav = d.margin_pts >= 0 ? fc.home_id : fc.away_id;
    return `<tr><td>${d.margin_pts >= 0 ? '▲' : '▼'} ${E(d.driver.replace(/^Home /, teamAbbr(fc.home_id) + ' ').replace(/^Away /, teamAbbr(fc.away_id) + ' '))}</td><td class="n ${d.margin_pts >= 0 ? 'pos' : 'neg'}">${E(teamAbbr(fav))} +${f(Math.abs(d.margin_pts), 1)} pts</td></tr>`;
  }).join('')}</table><div class="note">Measured by re-simulating with that unit set to league average (common random numbers).</div>`;
}
function teamStatsBlock(fc) {
  const ts = fc.team_stats, rows = [['Total plays', 'plays', 1], ['Pass yards', 'pass_yds', 1], ['Rush yards', 'rush_yds', 1], ['Points', null, 1],
    ['Turnovers', 'turnovers', 1], ['Sacks taken', 'sacks', 1], ['Explosive plays', 'explosive', 1], ['Time of poss.', 'top_seconds', 0]];
  const mmss = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;
  return `<table><tr><th></th><th class="n">${E(teamAbbr(fc.home_id))}</th><th class="n">${E(teamAbbr(fc.away_id))}</th></tr>${rows.map(([l, k]) => {
    const hv = k ? ts.home[k].mean : fc.home_pts.mean, av = k ? ts.away[k].mean : fc.away_pts.mean;
    const disp = k === 'top_seconds' ? [mmss(hv), mmss(av)] : [f(hv, 1), f(av, 1)];
    return `<tr><td>${l}</td><td class="n">${disp[0]}</td><td class="n">${disp[1]}</td></tr>`;
  }).join('')}<tr><td>Total yards</td><td class="n">${f(ts.home.pass_yds.mean + ts.home.rush_yds.mean, 1)}</td><td class="n">${f(ts.away.pass_yds.mean + ts.away.rush_yds.mean, 1)}</td></tr></table>`;
}
function playerProjTable(fc, side, groups = ['QB', 'RB', 'WR', 'TE']) {
  const pl = fc.players.filter(p => p.side === side && groups.includes(p.group) && p.slot !== 'QB2');
  const main = p => p.group === 'QB' ? ['pass_yds', 'pass_td'] : p.group === 'RB' ? ['rush_yds', 'rush_td'] : ['rec_yds', 'rec_td'];
  return `<table><tr><th>Player</th><th>Pos</th><th class="n">Proj (mean)</th><th class="n">TD</th><th class="n">P10–P90</th><th class="n">P(TD)</th></tr>${pl.map(p => {
    const [y, t] = main(p), sy = p.stats[y], st = p.stats[t];
    return `<tr><td><a href="#/players/${p.player_id}">${E(p.name || p.player_id)}</a></td><td>${p.slot}</td><td class="n">${f(sy.mean, 0)} yds</td><td class="n">${f(st.mean, 1)}</td><td class="n">${f(sy.p10, 0)}–${f(sy.p90, 0)}</td><td class="n">${pc(st.p_ge1, 0)}</td></tr>`;
  }).join('')}</table>`;
}
function scenarioBlock(fc) {
  if (!fc.scenarios) return '';
  return `<table><tr><th>Scenario</th><th class="n">${E(teamAbbr(fc.home_id))} win%</th><th class="n">Proj score</th></tr>${fc.scenarios.map(s => `<tr><td>${E(s.scenario.replace('Home', teamAbbr(fc.home_id)).replace('Away', teamAbbr(fc.away_id)))}</td><td class="n">${pc(s.home_win_prob)}</td><td class="n">${f(s.home_pts, 0)} – ${f(s.away_pts, 0)}</td></tr>`).join('')}</table>`;
}
function simOutcomes(fc) {
  const t = fc.total.p50;
  return `<div class="grid" style="grid-template-columns:150px 1fr;gap:8px"><table>
    <tr><td>${E(teamAbbr(fc.home_id))} win %</td><td class="n">${pc(fc.home_win_prob_raw)}</td></tr>
    <tr><td>${E(teamAbbr(fc.away_id))} win %</td><td class="n">${pc(1 - fc.home_win_prob_raw)}</td></tr>
    <tr><td>Avg margin</td><td class="n">${fs(fc.margin.mean, 1)}</td></tr>
    <tr><td>Margin SD</td><td class="n">${f(fc.margin.sd, 1)}</td></tr>
    <tr><td>Total P10–P90</td><td class="n">${f(fc.total.p10, 0)}–${f(fc.total.p90, 0)}</td></tr>
    <tr><td>OT probability</td><td class="n">${pc(fc.ot_prob)}</td></tr></table>
    <div><div class="sub">Margin distribution (home − away)</div>${C.density(fc.margin_hist.edges, [{name: 'margin', color: HOME, values: fc.margin_hist.counts}], {h: 130, vline: 0})}</div></div>`;
}

/* ------------------------------------------------------------------ dashboard */
async function dashboard(app) {
  const g = await featuredGame();
  const [fc, mu, tm, au] = await Promise.all([
    api(`/api/forecast?${qs({home: g.home_id, away: g.away_id, season: S.season, week: g.week, sims: 10000})}`),
    api(`/api/matchup?${qs({home: g.home_id, away: g.away_id, season: S.season})}`),
    api(`/api/teams/${g.home_id}?season=${S.season}`), api('/api/audit')]);
  const qb = fc.players.find(p => p.side === 'home' && p.slot === 'QB1');
  const pl = qb ? await api(`/api/players/${qb.player_id}?season=${S.season}&project=false`).catch(() => null) : null;
  app.innerHTML = `
  <div class="grid g-dash">${playerCard(pl)}
    <div class="grid" style="gap:10px">
      ${panel(`Week ${g.week} | Featured matchup`, `<div class="row between"><div class="teamline">${E(teamName(g.away_id))}</div><div class="vs">AT</div><div class="teamline">${E(teamName(g.home_id))}</div></div>
        <div class="sub">${E(g.start_date?.slice(0, 10) || '')} · ${g.temperature_f != null ? f(g.temperature_f, 0) + '°F' : 'temp n/a'} · wind ${g.wind_mph != null ? f(g.wind_mph, 0) + ' mph' : 'n/a'} · ${g.conference_game ? 'conference game' : 'non-conference'}</div>`)}
      ${panel('Game prediction', predictionBlock(fc, g) + `<div class="grid g2" style="margin-top:6px"><div><h3>Score distribution (${fc.n_sims.toLocaleString()} sims)</h3>${scoreDist(fc)}</div><div><h3>Key drivers</h3>${driversBlock(fc, 6)}</div></div>`)}
    </div>
    ${teamCard(tm)}
  </div>
  <div class="grid g-dash2" style="margin-top:10px">
    ${panel('Matchup lab: unit-by-unit', unitTable(mu, 'home_off') + unitTable(mu, 'away_off', true), `${E(teamAbbr(g.home_id))} vs ${E(teamAbbr(g.away_id))}`)}
    ${panel('Projected game statistics', teamStatsBlock(fc))}
    ${panel('Key player projections', `<div class="subtabs" id="kp-tabs"><a class="on" data-s="home">${E(teamAbbr(g.home_id))} offense</a><a data-s="away">${E(teamAbbr(g.away_id))} offense</a></div><div id="kp">${playerProjTable(fc, 'home')}</div>`, 'mean | P10–P90')}
    ${panel('Play calling & tendencies', tendencyTable(mu), 'personnel/formation require charting')}
  </div>
  <div class="grid g-dash3" style="margin-top:10px">
    ${panel('Simulation outcomes', simOutcomes(fc), `${fc.n_sims.toLocaleString()} games`)}
    ${panel('Field advantage & environment', envBlock(fc, g, tm))}
    ${panel('Impact factors & scenarios', scenarioBlock(fc))}
    ${panel('Model & forecast audit', auditMini(au), 'walk-forward, out-of-sample')}
  </div>`;
  document.querySelectorAll('#kp-tabs a').forEach(a => a.onclick = () => {
    document.querySelectorAll('#kp-tabs a').forEach(b => b.classList.toggle('on', b === a));
    $('#kp').innerHTML = playerProjTable(fc, a.dataset.s);
  });
}
function playerCard(pl) {
  if (!pl) return panel('Player intelligence profile', '<div class="sub">No player</div>');
  const p = pl.player, key = (pl.position_metrics || []).filter(m => m.status === 'ok').slice(0, 9);
  const uni = pl.universal.flatMap(c => c.metrics).filter(m => ['adj_epa_play', 'success', 'explosive'].includes(m.metric_id));
  const rows = [...uni, ...key].slice(0, 10);
  return panel('Player intelligence profile', `<div class="row">
    <div class="avatar" style="background:${p.color || '#28327a'}">${E(initials(p.first_name + ' ' + p.last_name))}</div>
    <div style="flex:1"><div class="name1">${E(p.first_name || '').toUpperCase()}</div><div class="name2">${E(p.last_name || '').toUpperCase()}</div>
      <div>${E(p.position)} #${p.jersey ?? ''} | ${E(p.school)}</div>
      <div class="sub">Class ${['', 'FR', 'SO', 'JR', 'SR', 'GR'][p.class_year] || '—'} · ${p.height_in ? Math.floor(p.height_in / 12) + "'" + p.height_in % 12 + '"' : ''} · ${p.weight_lb || ''} lbs<br>Archetype: ${E(p.archetype || '—')} ${tag('estimated')}</div></div>
    <div class="kpi"><div class="l">${E(p.position)} grade</div><div class="v pos">${f(p.grade, 1)}</div><div class="s">PCTL ${ord(p.grade_pctl)}</div></div></div>
    <div class="subtabs"><a class="on" href="#/players/${p.player_id}">Overview</a><a href="#/players/${p.player_id}?tab=universal">${pl.universal.reduce((a, c) => a + c.metrics.length, 0)} Universal</a><a href="#/players/${p.player_id}?tab=position">${E(p.position)} Module (${pl.position_metrics.length})</a><a href="#/players/${p.player_id}?tab=log">Game log</a></div>
    <div class="grid" style="grid-template-columns:1.2fr 1fr;gap:8px"><div><h3>Key ${S.season} metrics</h3>${metricTable(rows, true)}</div>
    <div>${pl.throw_chart ? throwChart(pl.throw_chart) : `<h3>Trend</h3>${C.line([{name: 'EPA', color: HOME, points: pl.trend.map(t => [t.week, t.epa])}], {h: 140, zero: true})}`}</div></div>`);
}
function metricTable(rows, compact = false) {
  return `<table><tr><th>Metric</th><th class="n">Value</th><th class="n">FBS pctl</th>${compact ? '' : '<th class="n">Conf pctl</th><th>Type</th>'}</tr>${rows.map(m => `<tr title="${E(m.formula)}">
    <td>${E(m.name)}</td><td class="n">${m.status === 'unavailable' ? '<span class="mut">N/A</span>' : fmtMetric(m.value, m.metric_id)}</td>
    <td class="n">${m.status === 'ok' ? ord(m.pctl_fbs) : '<span class="mut">' + E(m.status || '') + '</span>'}</td>${compact ? '' : `<td class="n">${m.status === 'ok' ? ord(m.pctl_conf) : ''}</td><td>${tag(m.classification)}</td>`}</tr>`).join('')}</table>`;
}
function throwChart(tc, compact = true) {
  if (!tc.available) return `<h3>Passing chart</h3><div class="sub">${E(tc.reason)}</div>`;
  const rows = [...tc.rows].reverse();
  const cells = rows.map(r => `<tr><td>${r.depth}</td><td class="n" style="background:${C.heatColor(r.comp_pct, .3, .85)}">${r.att ? pc(r.comp_pct, 0) : '—'}</td><td class="n">${pc(r.share, 0)}</td><td class="n ${cls(r.epa)}">${fs(r.epa, 2)}</td>${compact ? '' : `<td class="n">${r.td}/${r.int}</td>`}</tr>`).join('');
  return `<h3>${S.season} passing chart (air yards)</h3><table class="heat"><tr><th>Depth</th><th class="n">Cmp%</th><th class="n">Shr</th><th class="n">EPA</th>${compact ? '' : '<th class="n">TD/INT</th>'}</tr>${cells}</table><div class="note">Air yards parsed from play text for ${pc(tc.coverage, 0)} of attempts ${tag('estimated')} ${E(tc.location_note || '')}</div>`;
}
function teamCard(tm) {
  const t = tm.team, all = tm.categories.flatMap(c => c.metrics);
  const pick = ['off_epa', 'def_epa', 'off_sr', 'def_sr', 'off_expl', 'off_ppd', 'dd_d3_epa', 'rz_td_rate', 'pass_sack', 'net_epa'];
  const rows = pick.map(id => all.find(m => m.metric_id === id)).filter(Boolean);
  const off = tm.trend.filter(x => x.side === 'off'), def = tm.trend.filter(x => x.side === 'def');
  return panel(`Team intelligence: ${E(t.school)}`, `<div class="row between">
    <div class="avatar" style="background:${t.color || '#5a1530'};width:64px;height:64px;font-size:24px">${E(initials(t.school))}</div>
    <div style="flex:1"><div class="teamline">${E(t.school.toUpperCase())} ${E((t.mascot || '').toUpperCase())}</div>
      <div class="sub">${t.rec_wins ?? '—'}–${t.rec_losses ?? '—'} (${E(t.conf)}) · Off <span class="pos">${fs(t.off_epa, 3)}</span> · Def <span class="pos">${fs(t.def_epa, 3)}</span> · Net <span class="${cls(t.net_epa)}">${fs(t.net_epa, 3)}</span> EPA/play (adj.)</div></div>
    <div class="kpi"><div class="l">Team grade</div><div class="v pos">${f(t.team_grade, 1)}</div><div class="s">FBS PCTL ${ord(t.net_epa_pctl)}</div></div></div>
    <div class="subtabs"><a class="on" href="#/teams/${t.team_id}">Overview</a><a href="#/teams/${t.team_id}?tab=metrics">${all.length} Metrics</a><a href="#/teams/${t.team_id}?tab=roster">Roster</a></div>
    <div class="grid" style="grid-template-columns:1.1fr 1fr;gap:8px"><div><h3>Key team metrics (${S.season})</h3>${teamMetricTable(rows)}</div>
    <div><h3>Team EPA trend</h3><div class="legend"><span><i style="background:${OFFC}"></i>Offense</span><span><i style="background:${DEFC}"></i>Defense (allowed)</span></div>
    ${C.line([{name: 'Offense', color: OFFC, points: off.map(x => [x.week, x.epa])}, {name: 'Defense', color: DEFC, points: def.map(x => [x.week, x.epa])}], {h: 160, zero: true, xlab: w => 'W' + w})}</div></div>`);
}
function teamMetricTable(rows) {
  return `<table><tr><th>Metric</th><th class="n">Value</th><th class="n">Adj.</th><th class="n">FBS</th><th class="n">Conf</th></tr>${rows.map(m => `<tr title="${E(m.formula)}">
    <td>${E(m.name)}</td><td class="n">${m.status === 'unavailable' ? '<span class="mut">N/A</span>' : fmtMetric(m.value, m.metric_id)}</td><td class="n">${m.adj_value != null ? fmtMetric(m.adj_value, m.metric_id) : ''}</td>
    <td class="n">${m.pctl_fbs != null && m.status !== 'unavailable' ? ord(m.pctl_fbs) : ''}</td><td class="n">${m.pctl_conf != null && m.status !== 'unavailable' ? ord(m.pctl_conf) : ''}</td></tr>`).join('')}</table>`;
}
function unitTable(mu, side, noHead = false) {
  const rows = mu.units.filter(u => u.side === side);
  const off = side === 'home_off' ? mu.home : mu.away, def = side === 'home_off' ? mu.away : mu.home;
  return `<h3>${E(off)} offense vs ${E(def)} defense</h3><table><tr><th>Unit</th><th class="n">Off</th><th class="n">Def</th><th class="n">Edge</th><th class="n">z</th></tr>${rows.map(u => `<tr>
    <td>${E(u.unit)}</td><td class="n">${fmtMetric(u.off_value, u.off_metric)}</td><td class="n">${fmtMetric(u.def_value, u.def_metric)}</td>
    <td class="n ${cls(u.edge)}">${fs(u.edge, 3)}</td><td class="n ${Math.abs(u.z || 0) >= 1.64 ? cls(u.z) : 'mut'}">${f(u.z, 1)}</td></tr>`).join('')}</table>`;
}
function tendencyTable(mu) {
  return `<table><tr><th>Tendency</th><th class="n">${E((mu.home || '').slice(0, 12))}</th><th class="n">${E((mu.away || '').slice(0, 12))}</th><th class="n">FBS avg</th></tr>${mu.tendencies.map(t => `<tr>
    <td>${E(t.name)}</td><td class="n">${fmtMetric(t.home, t.metric_id)}</td><td class="n">${fmtMetric(t.away, t.metric_id)}</td><td class="n mut">${fmtMetric(t.league, t.metric_id)}</td></tr>`).join('')}</table>`;
}
function envBlock(fc, g, tm) {
  const hfa = (fc.drivers || []).find(d => d.unit === 'hfa');
  return `<table><tr><td>Home team</td><td class="n">${E(teamName(fc.home_id))}</td></tr>
    <tr><td>Neutral site</td><td class="n">${fc.neutral ? 'Yes' : 'No'}</td></tr>
    <tr><td>Weather</td><td class="n">${g && g.temperature_f != null ? f(g.temperature_f, 0) + '°F, ' + f(g.wind_mph, 0) + ' mph' : 'n/a'}</td></tr>
    <tr><td>Home-field edge (simulated)</td><td class="n pos">${hfa ? E(teamAbbr(fc.home_id)) + ' +' + f(hfa.margin_pts, 1) + ' pts' : '—'}</td></tr>
    <tr><td>Home pace (sec/play)</td><td class="n">${f(fc.params.home.pace, 1)}</td></tr><tr><td>Away pace (sec/play)</td><td class="n">${f(fc.params.away.pace, 1)}</td></tr>
    <tr><td>Expected plays (H/A)</td><td class="n">${f(fc.team_stats.home.plays.mean, 0)} / ${f(fc.team_stats.away.plays.mean, 0)}</td></tr>
    <tr><td>Travel / time zone</td><td class="n mut">requires venue coordinates</td></tr></table>
    <div class="note">Consistency checks: ${fc.consistency_checks.all_pass ? '<span class="pos">all accounting identities hold</span>' : '<span class="neg">FAILED</span>'}</div>`;
}
function auditMini(au) {
  const V = au.validation;
  const get = (mv, m, sc = 'overall') => V.find(v => v.model_version === mv && v.metric === m && v.scope === sc);
  if (!V.length) return '<div class="sub">No stored validation. Run <code>python -m cfbquant.cli backtest</code>.</div>';
  const cal = get('sim_v1_cal', 'calibration_curve');
  const r = (l, mv, m, d = 3, sq = false) => { const x = get(mv, m); return x ? `<tr><td>${l}</td><td class="n">${f(sq ? Math.sqrt(x.value) : x.value, d)}</td></tr>` : ''; };
  const cov = get('sim_v1', 'margin_80_coverage');
  return `<div class="grid" style="grid-template-columns:150px 1fr;gap:6px"><table>
    ${r('Winner log loss', 'sim_v1_cal', 'log_loss')}${r('Winner Brier', 'sim_v1_cal', 'brier')}${r('Elo log loss', 'elo', 'log_loss')}
    ${r('Margin MAE', 'sim_v1', 'margin_mae', 2)}${r('Total MAE', 'sim_v1', 'total_mae', 2)}${r('Pass yds MAE', 'sim_v1', 'mae', 1).replace('—', '')}
    ${(() => { const x = V.find(v => v.scope === 'player:pass_yds' && v.metric === 'mae'); return x ? `<tr><td>QB pass yds MAE</td><td class="n">${f(x.value, 1)}</td></tr>` : ''; })()}
    ${(() => { const x = V.find(v => v.scope === 'player:rush_yds' && v.metric === 'mae'); return x ? `<tr><td>Rush yds MAE</td><td class="n">${f(x.value, 1)}</td></tr>` : ''; })()}
    </table><div>${C.calib(cal?.detail_json?.points, {h: 160})}<div class="sub">Slope ${f(cal?.detail_json?.slope, 2)} · int ${f(cal?.detail_json?.intercept, 2)} · margin 80% coverage ${pc(cov?.value)}</div></div></div>
    <div class="note">n=${get('sim_v1_cal', 'log_loss')?.n || 0} out-of-sample games. <a href="#/audit">Full audit →</a></div>`;
}

/* ------------------------------------------------------------------ players */
async function playersView(app, h) {
  if (h.id) return playerDetail(app, +h.id, h.q.get('tab') || 'overview');
  const pos = h.q.get('pos') || 'QB', team = h.q.get('team') || '';
  const d = await api(`/api/players?${qs({season: S.season, position: pos, team, limit: 400})}`);
  const rows = d.players.filter(p => !S.conf || teamConf(p.team_id) === S.conf);
  app.innerHTML = panel(`Player directory — ${pos}`, `<div class="row wrap" style="margin-bottom:6px">
    <div class="chips">${['QB', 'RB', 'WR', 'TE', 'OT', 'OG', 'C', 'DL', 'EDGE', 'LB', 'CB', 'S', 'K', 'P'].map(p => `<a class="chip ${p === pos ? 'on' : ''}" href="#/players?pos=${p}">${p}</a>`).join('')}</div>
    <input class="search" id="ps" placeholder="Search name…" style="max-width:240px;margin:0"></div>
    <div class="scroll" style="max-height:75vh"><table id="pt"><tr><th>#</th><th>Player</th><th>Team</th><th>Class</th><th>Archetype</th><th class="n">Grade</th><th class="n">Pctl</th></tr>
    ${rows.map((p, i) => `<tr data-n="${E((p.first_name + ' ' + p.last_name).toLowerCase())}"><td class="n">${i + 1}</td><td><a href="#/players/${p.player_id}">${E(p.first_name)} ${E(p.last_name)}</a></td><td>${E(p.school)}</td><td>${['', 'FR', 'SO', 'JR', 'SR', 'GR'][p.class_year] || ''}</td><td>${E(p.archetype || '')}</td><td class="n">${f(p.grade, 1)}</td><td class="n">${ord(p.grade_pctl)}</td></tr>`).join('')}</table></div>
    <div class="note">Grade = model composite of position metrics, shrunk by sample size (estimated, not a scouting grade). OL grades require charting data and are not computed.</div>`, `${rows.length} players · through week ${d.through_week}`);
  $('#ps').oninput = e => document.querySelectorAll('#pt tr[data-n]').forEach(r => r.style.display = r.dataset.n.includes(e.target.value.toLowerCase()) ? '' : 'none');
}
async function playerDetail(app, id, tab) {
  const pl = await api(`/api/players/${id}?season=${S.season}`);
  const p = pl.player;
  const tabs = [['overview', 'Overview'], ['universal', 'Universal'], ['position', `${p.position} module`], ['log', 'Game log'], ['proj', 'Projection']];
  let body = '';
  if (tab === 'overview') {
    body = `<div class="grid g2">${playerCard(pl)}<div class="grid" style="gap:10px">
      ${panel('EPA / opportunity by week', C.line([{name: 'EPA', color: HOME, points: pl.trend.map(t => [t.week, t.epa])}, {name: 'Success', color: '#2fdc8a', points: pl.trend.map(t => [t.week, t.sr - 0.5])}], {h: 170, zero: true, xlab: w => 'W' + w}) + '<div class="legend"><span><i style="background:#7b5cff"></i>EPA/opp</span><span><i style="background:#2fdc8a"></i>Success rate − 50%</span></div>')}
      ${panel('Percentile profile', percentileBars(pl))}</div></div>`;
  } else if (tab === 'universal') {
    body = `<div class="grid g2">${pl.universal.map(c => panel(E(c.name), metricTable(c.metrics))).join('')}</div>`;
  } else if (tab === 'position') {
    body = panel(`${p.position} module`, metricTable(pl.position_metrics), `${pl.position_metrics.filter(m => m.status === 'ok').length} computed · ${pl.position_metrics.filter(m => m.status === 'unavailable').length} require charting`);
  } else if (tab === 'log') {
    const cols = p.position === 'QB' ? ['pass_cmp', 'pass_att', 'pass_yds', 'pass_td', 'pass_int', 'sacks_taken', 'rush_att', 'rush_yds', 'rush_td'] :
      ['RB', 'WR', 'TE'].includes(p.position) ? ['rush_att', 'rush_yds', 'rush_td', 'targets', 'receptions', 'rec_yds', 'rec_td'] :
      ['K'].includes(p.position) ? ['fga', 'fgm', 'xpa', 'xpm'] : ['P'].includes(p.position) ? ['punts'] : ['tackles', 'tfl', 'def_sacks', 'def_int', 'pbu', 'forced_fumbles'];
    body = panel('Game log', `<table><tr><th>Wk</th><th>Opp</th>${cols.map(c => `<th class="n">${c.replace('_', ' ')}</th>`).join('')}</tr>${pl.game_log.map(g => `<tr><td>${g.week}</td><td>${g.ha} ${E(g.opponent)}</td>${cols.map(c => `<td class="n">${f(g[c], 0)}</td>`).join('')}</tr>`).join('')}</table>`, tag('observed'));
  } else {
    const pr = pl.projection;
    body = pr && pr.player ? panel(`Projection — week ${pr.week} (${pr.home_id === p.team_id ? 'vs ' + E(teamName(pr.away_id)) : '@ ' + E(teamName(pr.home_id))})`,
      `<div class="sub">${E(pr.note)} · team win prob ${pc(pr.team_win_prob)} · ${pr.n_sims} sims ${tag('predicted')}</div>
       <table><tr><th>Stat</th><th class="n">Mean</th><th class="n">Median</th><th class="n">SD</th><th class="n">P10</th><th class="n">P25</th><th class="n">P75</th><th class="n">P90</th><th class="n">P(≥1)</th></tr>
       ${Object.entries(pr.player.stats).map(([k, d]) => `<tr><td>${k}</td><td class="n">${f(d.mean, 1)}</td><td class="n">${f(d.median, 0)}</td><td class="n">${f(d.sd, 1)}</td><td class="n">${f(d.p10, 0)}</td><td class="n">${f(d.p25, 0)}</td><td class="n">${f(d.p75, 0)}</td><td class="n">${f(d.p90, 0)}</td><td class="n">${d.p_ge1 != null ? pc(d.p_ge1, 0) : ''}</td></tr>`).join('')}</table>`) :
      panel('Projection', '<div class="sub">Player is not a projected starter in the next simulated game.</div>');
  }
  app.innerHTML = `<div class="row" style="margin-bottom:6px"><a href="#/players?pos=${p.position}">← ${p.position} directory</a><div class="subtabs" style="margin:0 0 0 12px;border:0">${tabs.map(([k, l]) => `<a class="${k === tab ? 'on' : ''}" href="#/players/${id}?tab=${k}">${l}</a>`).join('')}</div></div>${body}`;
}
function percentileBars(pl) {
  const ms = [...pl.universal.flatMap(c => c.metrics), ...pl.position_metrics].filter(m => m.status === 'ok' && m.pctl_fbs != null && m.direction !== 0);
  const top = ms.sort((a, b) => b.pctl_fbs - a.pctl_fbs);
  const sel = [...top.slice(0, 8), ...top.slice(-5)];
  return `<table>${sel.map(m => `<tr><td>${E(m.name)}</td><td class="n">${fmtMetric(m.value, m.metric_id)}</td><td style="width:40%">${C.pbar(m.pctl_fbs)}</td><td class="n">${ord(m.pctl_fbs)}</td></tr>`).join('')}</table><div class="note">Strongest and weakest qualified metrics vs. FBS players at the position (direction-adjusted).</div>`;
}

/* ------------------------------------------------------------------ teams */
async function teamsView(app, h) {
  if (h.id) return teamDetail(app, +h.id, h.q.get('tab') || 'overview');
  const ts = S.teams.filter(t => !S.conf || t.conf === S.conf);
  app.innerHTML = panel(`Team directory — ${S.season} through week ${S.week}`, `<div class="scroll" style="max-height:78vh"><table><tr><th>#</th><th>Team</th><th>Conf</th><th class="n">Record</th><th class="n">Off EPA (adj)</th><th class="n">Def EPA (adj)</th><th class="n">Net</th><th class="n">SOS</th><th class="n">Elo</th><th class="n">Talent</th><th style="width:120px">Net pctl</th></tr>
  ${ts.map((t, i) => `<tr><td class="n">${i + 1}</td><td><a href="#/teams/${t.team_id}">${E(t.school)}</a></td><td>${E(t.conf)}</td><td class="n">${t.rec_wins}–${t.rec_losses}</td><td class="n ${cls(t.off_epa)}">${fs(t.off_epa)}</td><td class="n ${cls(t.def_epa, -1)}">${fs(t.def_epa)}</td><td class="n ${cls(t.net_epa)}">${fs(t.net_epa)}</td><td class="n">${fs(t.sos)}</td><td class="n">${f(t.elo, 0)}</td><td class="n">${f(t.talent_composite, 0)}</td><td>${C.pbar(t.net_epa_pctl)}</td></tr>`).join('')}</table></div>`,
    'opponent-adjusted (ridge), garbage time excluded');
}
async function teamDetail(app, id, tab) {
  const tm = await api(`/api/teams/${id}?season=${S.season}`);
  const tabs = [['overview', 'Overview'], ['metrics', `${tm.categories.reduce((a, c) => a + c.metrics.length, 0)} Metrics`], ['splits', 'Trends & splits'], ['roster', 'Roster'], ['schedule', 'Schedule']];
  let body = '';
  const all = tm.categories.flatMap(c => c.metrics);
  if (tab === 'overview') {
    const ax = [['Pass O', 'pass_epa'], ['Rush O', 'rush_epa'], ['Explosive', 'off_expl'], ['Red zone', 'rz_epa'], ['3rd down', 'dd_d3_epa'], ['Pass D', 'pdef_epa'], ['Rush D', 'rdef_epa'], ['Pass rush', 'pr_sack'], ['Special', 'st_fg_voe'], ['Talent', 'rec_talent']];
    const val = mid => all.find(m => m.metric_id === mid)?.pctl_fbs;
    body = `<div class="grid g2">${teamCard(tm)}<div class="grid" style="gap:10px">${panel('Team radar (FBS percentiles)', C.radar(ax.map(a => a[0]), [{name: tm.team.school, color: HOME, values: ax.map(a => val(a[1]))}], {h: 250}))}
      ${panel('Category summary', `<table><tr><th>Category</th><th class="n">Computed</th><th class="n">Avg pctl</th><th style="width:35%"></th></tr>${tm.categories.map(c => {
        const ok = c.metrics.filter(m => m.status === 'ok' && m.pctl_fbs != null && m.direction !== 0);
        const av = ok.length ? ok.reduce((a, m) => a + m.pctl_fbs, 0) / ok.length : null;
        return `<tr><td>${E(c.name)}</td><td class="n">${ok.length}/${c.metrics.length}</td><td class="n">${ord(av)}</td><td>${C.pbar(av)}</td></tr>`;
      }).join('')}</table>`)}</div></div>`;
  } else if (tab === 'metrics') {
    body = `<div class="panel"><input class="search" id="ms" placeholder="Filter metrics…">
      ${tm.categories.map((c, i) => `<div class="cat ${i < 2 ? 'open' : ''}"><h3 class="cat-head">${E(c.name)} <span class="sub">(${c.metrics.length})</span></h3><div class="cat-body">
      <table><tr><th>Metric</th><th class="n">Value</th><th class="n">Opp-adj</th><th class="n">n</th><th class="n">FBS</th><th class="n">Conf</th><th style="width:90px"></th><th>Type</th><th>Formula / requirement</th></tr>
      ${c.metrics.map(m => `<tr data-n="${E(m.name.toLowerCase())}"><td>${E(m.name)}</td><td class="n">${m.status === 'unavailable' ? '<span class="mut">N/A</span>' : fmtMetric(m.value, m.metric_id)}</td><td class="n">${m.adj_value != null ? fmtMetric(m.adj_value, m.metric_id) : ''}</td><td class="n mut">${m.sample ?? ''}</td>
      <td class="n">${m.status !== 'unavailable' && m.pctl_fbs != null ? ord(m.pctl_fbs) : ''}</td><td class="n">${m.status !== 'unavailable' && m.pctl_conf != null ? ord(m.pctl_conf) : ''}</td><td>${m.status !== 'unavailable' && m.direction !== 0 ? C.pbar(m.pctl_fbs) : ''}</td>
      <td>${tag(m.classification)}</td><td class="sub" style="white-space:normal">${E(m.formula)}</td></tr>`).join('')}</table></div></div>`).join('')}</div>`;
  } else if (tab === 'splits') {
    const off = tm.trend.filter(x => x.side === 'off'), def = tm.trend.filter(x => x.side === 'def');
    const sp = ['env_home_epa', 'env_away_epa', 'env_q4_epa', 'env_late_epa', 'env_lead_epa', 'env_trail_epa', 'env_own_epa', 'env_cold_epa', 'env_wind_pass', 'dd_std_epa', 'dd_pd_epa', 'tr_l3_off', 'tr_l5_off', 'tr_l3_def', 'tr_slope', 'tr_luck', 'tr_to_luck'].map(id => all.find(m => m.metric_id === id)).filter(Boolean);
    body = `<div class="grid g2">${panel('Pass vs rush EPA by game', C.line([{name: 'Pass', color: HOME, points: off.map(x => [x.week, x.pass_epa])}, {name: 'Rush', color: '#2fdc8a', points: off.map(x => [x.week, x.rush_epa])}], {h: 190, zero: true, xlab: w => 'W' + w}) + '<div class="legend"><span><i style="background:#7b5cff"></i>Pass EPA/db</span><span><i style="background:#2fdc8a"></i>Rush EPA/carry</span></div>')}
      ${panel('Success rate by game', C.line([{name: 'Off SR', color: OFFC, points: off.map(x => [x.week, x.sr])}, {name: 'Def SR allowed', color: DEFC, points: def.map(x => [x.week, x.sr])}], {h: 190, xlab: w => 'W' + w, fmt: v => (v * 100).toFixed(0) + '%'}))}
      ${panel('Situational splits & trend diagnostics', teamMetricTable(sp))}
      ${panel('Down & distance', teamMetricTable(tm.categories.find(c => c.name === 'Down and Distance').metrics))}</div>`;
  } else if (tab === 'roster') {
    body = panel('Roster intelligence', `<div class="scroll" style="max-height:75vh"><table><tr><th>Player</th><th>Pos</th><th>Class</th><th>Stars</th><th>Archetype</th><th class="n">Grade</th><th class="n">Pctl</th></tr>${tm.roster.map(r => `<tr><td><a href="#/players/${r.player_id}">${E(r.first_name)} ${E(r.last_name)}</a></td><td>${r.position}</td><td>${['', 'FR', 'SO', 'JR', 'SR', 'GR'][r.class_year] || ''}</td><td>${r.stars ? '★'.repeat(r.stars) : ''}</td><td>${E(r.archetype || '')}</td><td class="n">${f(r.grade, 1)}</td><td class="n">${ord(r.grade_pctl)}</td></tr>`).join('')}</table></div>`);
  } else {
    body = panel('Schedule & results', `<table><tr><th>Wk</th><th>Opponent</th><th class="n">Result</th><th></th></tr>${tm.schedule.map(g => {
      const home = g.home_id === id, opp = home ? g.away_id : g.home_id, us = home ? g.home_points : g.away_points, them = home ? g.away_points : g.home_points;
      return `<tr><td>${g.week}</td><td>${home ? 'vs' : '@'} <a href="#/teams/${opp}">${E(home ? g.away : g.home)}</a></td><td class="n ${us > them ? 'pos' : 'neg'}">${g.completed ? (us > them ? 'W ' : 'L ') + us + '–' + them : '—'}</td><td><a href="#/predictions?home=${g.home_id}&away=${g.away_id}&week=${g.week}">forecast</a></td></tr>`;
    }).join('')}</table>`);
  }
  app.innerHTML = `<div class="row" style="margin-bottom:6px"><a href="#/teams">← Teams</a><div class="subtabs" style="margin:0 0 0 12px;border:0">${tabs.map(([k, l]) => `<a class="${k === tab ? 'on' : ''}" href="#/teams/${id}?tab=${k}">${l}</a>`).join('')}</div><a style="margin-left:auto" href="#/matchup?home=${id}">Open in Matchup Lab →</a></div>${body}`;
  document.querySelectorAll('.cat-head').forEach(hh => hh.onclick = () => hh.parentElement.classList.toggle('open'));
  const ms = $('#ms'); if (ms) ms.oninput = e => document.querySelectorAll('tr[data-n]').forEach(r => r.style.display = r.dataset.n.includes(e.target.value.toLowerCase()) ? '' : 'none');
}

/* ------------------------------------------------------------------ matchup lab */
function teamSelect(id, val) { return `<select id="${id}">${[...S.teams].sort((a, b) => a.school.localeCompare(b.school)).map(t => `<option value="${t.team_id}" ${t.team_id === val ? 'selected' : ''}>${E(t.school)}</option>`).join('')}</select>`; }
async function matchupView(app, h) {
  const home = +(h.q.get('home') || S.teams[0].team_id), away = +(h.q.get('away') || S.teams[1].team_id);
  const [mu, th, ta] = await Promise.all([api(`/api/matchup?${qs({home, away, season: S.season})}`), api(`/api/teams/${home}?season=${S.season}`), api(`/api/teams/${away}?season=${S.season}`)]);
  const ax = [['Pass O', 'pass_epa'], ['Rush O', 'rush_epa'], ['Explosive', 'off_expl'], ['RZ O', 'rz_epa'], ['3rd D', 'dd_d3_epa'], ['Pass D', 'pdef_epa'], ['Rush D', 'rdef_epa'], ['Pass rush', 'pr_sack'], ['Protect', 'pass_sack'], ['Special', 'st_fg_voe']];
  const pv = (tm, mid) => tm.categories.flatMap(c => c.metrics).find(m => m.metric_id === mid)?.pctl_fbs;
  const heatRows = [['Pass offense', 'pass_epa'], ['Rush offense', 'rush_epa'], ['Explosiveness', 'off_expl'], ['Protection', 'pass_sack'], ['Red zone O', 'rz_epa']];
  const heatCols = [['Pass D', 'pdef_epa'], ['Rush D', 'rdef_epa'], ['Expl. allowed', 'def_expl'], ['Pass rush', 'pr_sack'], ['RZ D', 'rz_def_epa']];
  const heat = (o, d) => `<table class="heat"><tr><th>${E(o.team.school)} O ↓ / ${E(d.team.school)} D →</th>${heatCols.map(c => `<th class="n">${c[0]}</th>`).join('')}</tr>${heatRows.map(r => `<tr><td>${r[0]} (${ord(pv(o, r[1]))})</td>${heatCols.map(c => {
    const a = pv(o, r[1]), b = pv(d, c[1]); const v = a != null && b != null ? a - b : null;
    return `<td class="n" style="background:${v == null ? '#141c3d' : v > 0 ? `rgba(47,220,138,${Math.min(.75, Math.abs(v) / 100)})` : `rgba(255,77,109,${Math.min(.75, Math.abs(v) / 100)})`}">${v == null ? '—' : fs(v, 0)}</td>`;
  }).join('')}</tr>`).join('')}</table>`;
  app.innerHTML = `<div class="panel row wrap" style="margin-bottom:10px"><b>MATCHUP LAB</b> Home ${teamSelect('mh', home)} vs Away ${teamSelect('ma', away)} <button id="mgo">Compare</button>
    <a href="#/predictions?home=${home}&away=${away}">Simulate this matchup →</a><span class="sub">Through week ${mu.through_week}, ${S.season}</span></div>
    <div class="grid g2">${panel('Unit-by-unit edges', unitTable(mu, 'home_off') + unitTable(mu, 'away_off'), 'edge with approx. z-score')}
    <div class="grid" style="gap:10px">${panel('Profile overlay (FBS percentiles)', `<div class="legend"><span><i style="background:${HOME}"></i>${E(mu.home)}</span><span><i style="background:${AWAY}"></i>${E(mu.away)}</span></div>` + C.radar(ax.map(a => a[0]), [{name: mu.home, color: HOME, values: ax.map(a => pv(th, a[1]))}, {name: mu.away, color: AWAY, values: ax.map(a => pv(ta, a[1]))}], {h: 250}))}
    ${panel('Coaching & tendencies', tendencyTable(mu))}</div></div>
    <div class="grid g2" style="margin-top:10px">${panel('Matchup heatmap: home offense vs away defense', heat(th, ta), 'percentile difference')}${panel('Matchup heatmap: away offense vs home defense', heat(ta, th), 'percentile difference')}</div>
    <div class="note">${E(mu.note)}</div>`;
  $('#mgo').onclick = () => location.hash = `#/matchup?home=${$('#mh').value}&away=${$('#ma').value}`;
}

/* ------------------------------------------------------------------ predictions */
async function predictionsView(app, h) {
  const home = +(h.q.get('home') || 0), away = +(h.q.get('away') || 0);
  const week = h.q.get('week') || (S.week + 1), sims = h.q.get('sims') || 10000, neutral = h.q.get('neutral') === '1';
  const ctrl = `<div class="panel row wrap" style="margin-bottom:10px"><b>PREDICTION ENGINE</b> Home ${teamSelect('ph', home || S.teams[0].team_id)} Away ${teamSelect('pa', away || S.teams[1].team_id)}
    <label class="sub">As-of week <input id="pw" type="number" value="${week}" style="width:55px"></label>
    <label class="sub">Sims <input id="pn" type="number" value="${sims}" step="1000" style="width:75px"></label>
    <label class="sub"><input id="pneu" type="checkbox" ${neutral ? 'checked' : ''}> neutral</label><button id="pgo">Run simulation</button>
    <span class="sub">Uses only data before the as-of week. Seeded & reproducible.</span></div>`;
  if (!home || !away) { app.innerHTML = ctrl + panel('Select a matchup', '<div class="sub">Choose teams and run.</div>'); bindPred(); return; }
  app.innerHTML = ctrl + '<div class="loading">Simulating…</div>';
  bindPred();
  const fc = await api(`/api/forecast?${qs({home, away, season: S.season, week, sims, neutral: neutral ? 'true' : 'false'})}`);
  app.innerHTML = ctrl + `<div class="grid g2">
    ${panel(`${E(teamName(away))} at ${E(teamName(home))} — week ${week}`, predictionBlock(fc, fc.game) + `<h3>Most likely final scores</h3><div class="chips">${fc.top_scores.map(s => `<span class="chip">${s.home}–${s.away} · ${pc(s.prob)}</span>`).join('')}</div>`)}
    ${panel('Score distribution', scoreDist(fc, {h: 170}) + simOutcomes(fc))}
    ${fc.market ? panel('Market comparison — Polymarket US', marketCompare(fc), fc.market.kind === 'close' ? 'closing price (pre-kickoff)' : 'live order book') : ''}
    ${panel('Key drivers', driversBlock(fc, 11))}
    ${panel('Injury & player scenarios', scenarioBlock(fc))}
    ${panel(`${E(teamName(home))} player projections`, fullProj(fc, 'home'), 'full distributions')}
    ${panel(`${E(teamName(away))} player projections`, fullProj(fc, 'away'), 'full distributions')}
    ${panel('Projected team statistics', teamStatsBlock(fc))}
    ${panel('Model inputs (as-of parameters)', paramTable(fc), 'offense produces / defense allows')}</div>
    <div class="note">Consistency: ${Object.entries(fc.consistency_checks).map(([k, v]) => `${k}: ${v ? '✔' : '✘'}`).join(' · ')}</div>`;
  bindPred();
}
function bindPred() {
  const b = $('#pgo'); if (!b) return;
  b.onclick = () => location.hash = `#/predictions?${qs({home: $('#ph').value, away: $('#pa').value, week: $('#pw').value, sims: $('#pn').value, neutral: $('#pneu').checked ? 1 : ''})}`;
}
function fullProj(fc, side) {
  const pl = fc.players.filter(p => p.side === side && p.group !== 'DEF' || (p.side === side && ['DL1', 'LB1', 'CB1', 'S1'].includes(p.slot)));
  return `<div class="scroll-sm" style="max-height:340px"><table><tr><th>Player</th><th>Stat</th><th class="n">Mean</th><th class="n">Med</th><th class="n">P10</th><th class="n">P25</th><th class="n">P75</th><th class="n">P90</th><th class="n">P(≥1)</th></tr>
  ${pl.flatMap(p => Object.entries(p.stats).filter(([k, d]) => d.mean >= 0.05).map(([k, d], i) => `<tr><td>${i === 0 ? `<a href="#/players/${p.player_id}">${E(p.name || '')}</a> <span class="mut">${p.slot}</span>` : ''}</td><td>${k}</td><td class="n">${f(d.mean, 1)}</td><td class="n">${f(d.median, 0)}</td><td class="n">${f(d.p10, 0)}</td><td class="n">${f(d.p25, 0)}</td><td class="n">${f(d.p75, 0)}</td><td class="n">${f(d.p90, 0)}</td><td class="n">${d.p_ge1 != null ? pc(d.p_ge1, 0) : ''}</td></tr>`)).join('')}</table></div>`;
}
function paramTable(fc) {
  const h = fc.params.home, a = fc.params.away;
  return `<table><tr><th>Parameter</th><th class="n">${E(teamAbbr(fc.home_id))}</th><th class="n">${E(teamAbbr(fc.away_id))}</th></tr>${Object.keys(h).map(k => `<tr><td>${k}</td><td class="n">${f(h[k], 3)}</td><td class="n">${f(a[k], 3)}</td></tr>`).join('')}</table>`;
}

/* ------------------------------------------------------------------ simulations (slate) */
async function simulationsView(app, h) {
  const g = await api(`/api/games?season=${S.season}`);
  const weeks = [...new Set(g.games.map(x => x.week))];
  const wk = +(h.q.get('week') || weeks[weeks.length - 1]);
  const rows = g.games.filter(x => x.week === wk && (!S.conf || [teamConf(x.home_id), teamConf(x.away_id)].includes(S.conf)));
  const has = rows.some(r => r.home_win_prob_sim_v1_cal != null);
  app.innerHTML = panel(`Week ${wk} slate`, `<div class="chips" style="margin-bottom:6px">${weeks.map(w => `<a class="chip ${w === wk ? 'on' : ''}" href="#/simulations?week=${w}">W${w}</a>`).join('')}</div>
    <table><tr><th>Away</th><th>Home</th><th class="n">Final</th><th class="n">Sim P(home)</th><th class="n">Elo P(home)</th><th class="n">Sim margin</th><th class="n">Sim total</th><th class="n">Result</th><th></th></tr>
    ${rows.map(r => {
      const won = r.home_points > r.away_points, p = r.home_win_prob_sim_v1_cal;
      const hit = p == null ? '' : ((p > .5) === won ? '<span class="pos">✔</span>' : '<span class="neg">✘</span>');
      return `<tr><td>${E(r.away)}</td><td>${E(r.home)}</td><td class="n">${r.completed ? r.away_points + '–' + r.home_points : '—'}</td><td class="n">${pc(p)}</td><td class="n">${pc(r.home_win_prob_elo)}</td><td class="n">${fs(r.margin_mean_sim_v1_cal, 1)}</td><td class="n">${f(r.total_mean_sim_v1_cal, 1)}</td><td class="n">${hit}</td><td><a href="#/predictions?home=${r.home_id}&away=${r.away_id}&week=${r.week}">simulate</a></td></tr>`;
    }).join('')}</table><div class="note">${has ? 'Stored walk-forward (backtest) predictions made with data available before kickoff.' : 'No stored predictions for this week (not in backtest window). Use "simulate" to run on demand.'}</div>`, `${rows.length} games`);
}

/* ------------------------------------------------------------------ markets (Polymarket US) */
function marketCompare(fc) {
  const m = fc.market, H = teamAbbr(fc.home_id);
  return `<table><tr><th></th><th class="n">Model</th><th class="n">Market</th><th class="n">Model − market</th></tr>
    <tr><td>${E(H)} win prob</td><td class="n">${pc(fc.home_win_prob)}</td><td class="n">${pc(m.p_home)}</td><td class="n ${cls(m.model_minus_market_p)}">${m.model_minus_market_p == null ? '—' : fs(100 * m.model_minus_market_p, 1) + ' pp'}</td></tr>
    <tr><td>${E(H)} margin</td><td class="n">${fs(fc.margin.mean, 1)}</td><td class="n">${fs(m.margin_home, 1)}</td><td class="n">${fs(m.model_minus_market_margin, 1)}</td></tr>
    <tr><td>Total</td><td class="n">${f(fc.total.mean, 1)}</td><td class="n">${f(m.total, 1)}</td><td class="n">${fs(m.model_minus_market_total, 1)}</td></tr></table>
    <div class="note">${E(m.source)} · ${E(m.kind)} · as of ${E((m.as_of || '').replace('T', ' ').slice(0, 16))} · quality ${E(m.quality)}. Gaps are model disagreement, not a recommendation.</div>`;
}
async function marketsView(app, h) {
  if (h.id) return marketEvent(app, h.id);
  const kind = h.q.get('kind') || 'live';
  let b, cal;
  try { [b, cal] = await Promise.all([api(`/api/markets/board?kind=${kind}`), api('/api/markets/calibration')]); }
  catch (e) { app.innerHTML = panel('Markets', `<div class="sub">${E(e.message)}</div><div class="note">Sync with <code>python -m cfbquant.cli markets live</code> (free public API, no key).</div>`); return; }
  const rows = b.events;
  const banner = `<div class="panel" style="margin-bottom:10px;border-color:#2b6b4a"><b class="pos">REAL MARKET DATA</b> <span class="sub">${E(b.note)} Last sync ${E((b.synced_at || '').replace('T', ' ').slice(0, 16))} UTC.</span>
    ${b.synthetic_model_db ? '<div class="note amb">The model database is the fictional synthetic league, so these real games are not linked to model forecasts. Load CFBD data to compare model vs market.</div>' : ''}</div>`;
  const rk = r => r && r !== '0' ? `<span class="mut">#${r}</span> ` : '';
  const status = e => e.live || e.in_play ? `<span class="neg">IN-PLAY ${E(e.period || '')} ${E(e.elapsed || '')}</span>` : e.closed || e.ended ? 'Final' : E((e.start_time || '').replace('T', ' ').slice(5, 16));
  const score = e => e.away_score != null ? `${e.away_score}–${e.home_score}` : '';
  app.innerHTML = banner + `<div class="grid" style="grid-template-columns:2.2fr 1fr;gap:10px">
    ${panel(`College football board — ${kind === 'live' ? 'live & upcoming' : 'closing prices'}`, `<div class="chips" style="margin-bottom:6px"><a class="chip ${kind === 'live' ? 'on' : ''}" href="#/markets?kind=live">Live / upcoming</a><a class="chip ${kind === 'close' ? 'on' : ''}" href="#/markets?kind=close">Closing (final)</a><input class="search" id="mks" placeholder="Filter team…" style="max-width:200px;margin:0 0 0 8px"></div>
      <div class="scroll" style="max-height:74vh"><table id="mkt"><tr><th>Kickoff / status</th><th>Away</th><th>Home</th><th class="n">Score</th><th class="n">P(away)</th><th class="n">P(home)</th><th style="width:90px"></th><th class="n">Home spread</th><th class="n">Margin SD</th><th class="n">Total</th><th class="n">Rungs</th><th>Quality</th>${b.linked_to_model ? '<th class="n">Model</th>' : ''}</tr>
      ${rows.map(e => `<tr data-n="${E(((e.away || '') + ' ' + (e.home || '')).toLowerCase())}"><td>${status(e)}</td><td><a href="#/markets/${e.event_id}">${rk(e.away_rank)}${E(e.away || '')}</a></td><td><a href="#/markets/${e.event_id}">${rk(e.home_rank)}${E(e.home || '')}</a></td><td class="n">${score(e)}</td>
        <td class="n">${pc(e.p_away_win)}</td><td class="n">${pc(e.p_home_win)}</td><td>${C.pbar(e.p_home_win == null ? null : 100 * e.p_home_win)}</td>
        <td class="n">${e.margin_home_mean == null ? '—' : (e.margin_home_mean >= 0 ? '−' : '+') + f(Math.abs(e.margin_home_mean), 1)}</td><td class="n">${f(e.margin_sd, 1)}</td><td class="n">${f(e.total_mean, 1)}</td><td class="n mut">${e.n_spread}/${e.n_total}</td><td class="sub">${E(e.quality)}</td>
        ${b.linked_to_model ? `<td class="n">${e.game_id ? `<a href="#/predictions?home=${e.home_id}&away=${e.away_id}">forecast</a>` : '—'}</td>` : ''}</tr>`).join('')}</table></div>`, `${rows.length} games ${tag('observed')}`)}
    ${panel('Market calibration (closing moneylines)', cal.n ? `<table><tr><td>Settled games</td><td class="n">${cal.n}</td></tr><tr><td>Log loss [90% CI]</td><td class="n">${f(cal.log_loss[0], 4)} <span class="mut">[${f(cal.log_loss[1], 3)}, ${f(cal.log_loss[2], 3)}]</span></td></tr>
      <tr><td>Brier</td><td class="n">${f(cal.brier[0], 4)}</td></tr><tr><td>Favorite accuracy</td><td class="n">${pc(cal.accuracy)}</td></tr>
      <tr><td>Favorites: win rate − price</td><td class="n">${fs(100 * cal.favorite_win_rate_minus_price[0], 1)} pp <span class="mut">[${fs(100 * cal.favorite_win_rate_minus_price[1], 1)}, ${fs(100 * cal.favorite_win_rate_minus_price[2], 1)}]</span></td></tr>
      <tr><td>Score-feed / settlement conflicts</td><td class="n ${cal.score_conflicts ? 'amb' : ''}">${cal.score_conflicts}</td></tr></table>
      ${C.calib(cal.calibration.points, {color: '#2fdc8a'})}<div class="sub">slope ${f(cal.calibration.slope, 2)}, intercept ${f(cal.calibration.intercept, 2)} · ${E((cal.first_game || '').slice(0, 10))} → ${E((cal.last_game || '').slice(0, 10))}</div><div class="note">${E(cal.note)}</div>` : `<div class="sub">${E(cal.note)}</div>`)}
  </div>`;
  $('#mks').oninput = e => document.querySelectorAll('#mkt tr[data-n]').forEach(r => r.style.display = r.dataset.n.includes(e.target.value.toLowerCase()) ? '' : 'none');
}
function ladderChart(points, xKey, yKey, mean, sd, kind, o = {}) {
  const w = o.w || 420, h = o.h || 190, m = {l: 34, r: 8, t: 8, b: 24};
  const pts = points.filter(p => p[yKey] != null);
  if (!pts.length) return '<div class="sub">No ladder prices</div>';
  const xs = pts.map(p => p[xKey]); const x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
  const X = x => m.l + (x - x0) / (x1 - x0) * (w - m.l - m.r), Y = v => h - m.b - v * (h - m.t - m.b);
  const Phi = z => 0.5 * (1 + Math.tanh(0.7978845608 * (z + 0.044715 * z * z * z)));
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
  [0, .25, .5, .75, 1].forEach(v => s += `<line x1="${m.l}" x2="${w - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1d2a58"/><text x="${m.l - 3}" y="${Y(v) + 3}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`);
  const step = Math.max(1, Math.round((x1 - x0) / 8));
  for (let x = Math.ceil(x0); x <= x1; x += step) s += `<text x="${X(x)}" y="${h - 7}" text-anchor="middle">${x}</text>`;
  if (mean != null && sd) {
    const curve = []; for (let i = 0; i <= 60; i++) { const x = x0 + (x1 - x0) * i / 60; const z = kind === 'spread' ? (mean + x) / sd : (mean - x) / sd; curve.push(`${X(x)},${Y(Phi(z))}`); }
    s += `<polyline fill="none" stroke="#7b5cff" stroke-width="1.6" points="${curve.join(' ')}"/>`;
  }
  pts.forEach(p => s += `<circle cx="${X(p[xKey])}" cy="${Y(p[yKey])}" r="3" fill="#2fdc8a"><title>line ${p[xKey]}: ${(p[yKey] * 100).toFixed(1)}% (bid ${p.bid ?? '—'} / ask ${p.ask ?? '—'})</title></circle>`);
  return s + '</svg>';
}
async function marketEvent(app, id) {
  const d = await api(`/api/markets/event/${id}`);
  const e = d.event, kind = d.live && (d.live.spread_ladder.length || d.live.moneyline.length) ? 'live' : 'close';
  const L = d[kind], imp = d.implied.filter(x => x.kind === kind).slice(-1)[0] || {};
  const A = e.away_abbr || e.away, Hh = e.home_abbr || e.home;
  const props = (L.props || []);
  app.innerHTML = `<div class="row" style="margin-bottom:6px"><a href="#/markets">← Markets board</a><span class="sub" style="margin-left:10px">${E(d.note)}</span></div>
  <div class="grid g2">
    ${panel(`${E(e.away)} at ${E(e.home)}`, `<div class="row between wrap">
      <div class="kpi big-away" style="flex:1"><div class="l">${E(A)} win</div><div class="v">${pc(imp.p_away_win)}</div></div>
      <div class="kpi" style="flex:1.2"><div class="l">Market-implied</div><div class="teamline">${E(Hh)} ${imp.margin_home_mean == null ? '' : (imp.margin_home_mean >= 0 ? '−' : '+') + f(Math.abs(imp.margin_home_mean), 1)}</div><div class="sub">Total ${f(imp.total_mean, 1)} · margin SD ${f(imp.margin_sd, 1)}</div></div>
      <div class="kpi big-home" style="flex:1"><div class="l">${E(Hh)} win</div><div class="v">${pc(imp.p_home_win)}</div></div></div>
      <div class="sub" style="margin-top:4px">${kind === 'close' ? 'Closing prices (last book sample at or before kickoff)' : 'Live order book'} · ${e.live ? 'LIVE ' + E(e.period || '') : e.closed ? 'Final' : 'Kickoff ' + E((e.start_time || '').replace('T', ' ').slice(0, 16)) + ' UTC'} ${e.away_score != null ? '· score ' + e.away_score + '–' + e.home_score : ''}
      ${e.score_conflict ? '<div class="note amb">Polymarket score feed disagrees with its own settlement for this game.</div>' : ''}</div>`)}
    ${panel(`${E(Hh)} win probability — last 7 days`, C.line([{name: 'P(home)', color: '#2fdc8a', points: d.history.map(x => [x.t, x.p_home])}], {h: 170, y0: 0, y1: 1, fmt: v => (v * 100).toFixed(0) + '%', xlab: t => new Date(t * 1000).toISOString().slice(5, 10), empty: d.history_error || 'No history'}))}
    ${panel('Spread ladder', ladderChart(L.spread_ladder, 'line_home', 'p_home_cover', imp.margin_home_mean, imp.margin_sd, 'spread') + `<div class="note">Dots: P(${E(Hh)} margin + line &gt; 0) at each alternate line. Curve: fitted normal (mean ${fs(imp.margin_home_mean, 1)}, SD ${f(imp.margin_sd, 1)}, fit RMSE ${f(imp.spread_fit_rmse, 3)}).</div>`, `${L.spread_ladder.length} rungs`)}
    ${panel('Total ladder', ladderChart(L.total_ladder, 'line', 'mid', imp.total_mean, imp.total_sd, 'total') + `<div class="note">Dots: P(total &gt; line). Curve: fitted normal (mean ${f(imp.total_mean, 1)}, SD ${f(imp.total_sd, 1)}).</div>`, `${L.total_ladder.length} rungs`)}
    ${props.length ? panel('Team stat props', `<table><tr><th>Team</th><th>Stat</th><th class="n">Line</th><th class="n">P(over)</th><th class="n">Bid / Ask</th></tr>${props.map(p => `<tr><td>${p.long_pm_team_id === e.home_pm_team_id ? E(Hh) : p.long_pm_team_id === e.away_pm_team_id ? E(A) : '—'}</td><td>${E(p.stat)}</td><td class="n">${f(p.line, 1)}</td><td class="n">${pc(p.mid)}</td><td class="n mut">${f(p.bid, 3)} / ${f(p.ask, 3)}</td></tr>`).join('')}</table>`) : ''}
  </div>`;
}

/* ------------------------------------------------------------------ recruiting */
async function recruitingView(app, h) {
  const team = h.q.get('team') || '';
  const r = await api(`/api/recruiting?${qs({season: S.season, team})}`);
  const pos = ['QB', 'RB', 'WR', 'TE'], colors = {QB: HOME, RB: '#2fdc8a', WR: '#33c6ff', TE: '#ffb547'};
  const curves = pos.map(p => ({name: p, color: colors[p], points: r.development_curves.filter(c => c.position === p).map(c => [c.class_year, c.adj_epa])}));
  const tal = [...r.talent].sort((a, b) => b.talent_composite - a.talent_composite);
  app.innerHTML = `<div class="panel row wrap" style="margin-bottom:10px"><b>RECRUITING & DEVELOPMENT</b> Team <select id="rt"><option value="">All</option>${[...S.teams].sort((a, b) => a.school.localeCompare(b.school)).map(t => `<option value="${t.team_id}" ${String(t.team_id) === team ? 'selected' : ''}>${E(t.school)}</option>`).join('')}</select>
    <span class="sub">Recruiting ratings are priors, updated by observed production (Bayesian). Wide intervals = little evidence.</span></div>
    <div class="grid g2">
    ${panel('Breakout candidates', `<table><tr><th>Player</th><th>Pos</th><th>Team</th><th>Cls</th><th>★</th><th class="n">n</th><th class="n">Posterior</th><th class="n">80% interval</th></tr>${r.breakouts.map(b => `<tr><td><a href="#/players/${b.player_id}">${E(b.name)}</a></td><td>${b.position}</td><td>${E(teamAbbr(b.team_id))}</td><td>${b.class_year}</td><td>${b.stars || ''}</td><td class="n">${b.sample ?? 0}</td><td class="n">${fs(b.post_mean, 3)}</td><td class="n">${fs(b.post_p10, 2)} … ${fs(b.post_p90, 2)}</td></tr>`).join('')}</table>`, 'high upside, limited usage, underclassmen')}
    ${panel('Development curves (adj. EPA/opp by class year)', `<div class="legend">${pos.map(p => `<span><i style="background:${colors[p]}"></i>${p}</span>`).join('')}</div>` + C.line(curves, {h: 200, zero: true, xlab: x => ['', 'FR', 'SO', 'JR', 'SR', 'GR'][x] || x}))}
    ${panel('Talent & continuity', `<div class="scroll-sm"><table><tr><th>Team</th><th class="n">Talent</th><th class="n">Returning prod.</th></tr>${tal.map(t => `<tr><td><a href="#/teams/${t.team_id}">${E(t.school)}</a></td><td class="n">${f(t.talent_composite, 1)}</td><td class="n">${pc(t.returning_production, 0)}</td></tr>`).join('')}</table></div>`)}
    ${panel('Player posteriors (talent prior × production)', `<div class="scroll-sm"><table><tr><th>Player</th><th>Pos</th><th>★</th><th class="n">Obs adj EPA</th><th class="n">n</th><th class="n">Prior</th><th class="n">Posterior</th><th class="n">P10</th><th class="n">P90</th></tr>${r.posteriors.map(p => `<tr><td><a href="#/players/${p.player_id}">${E(p.name)}</a></td><td>${p.position}</td><td>${p.stars || ''}</td><td class="n">${fs(p.adj_epa_play, 3)}</td><td class="n">${p.sample ?? 0}</td><td class="n">${fs(p.prior_mean, 3)}</td><td class="n">${fs(p.post_mean, 3)}</td><td class="n">${fs(p.post_p10, 2)}</td><td class="n">${fs(p.post_p90, 2)}</td></tr>`).join('')}</table></div>`)}
    ${panel('Recruiting classes', `<div class="scroll-sm"><table><tr><th>Team</th><th>Class</th><th class="n">Signees</th><th class="n">Avg ★</th><th class="n">Avg rating</th><th class="n">Blue chips</th></tr>${r.classes.slice(-120).reverse().map(c => `<tr><td>${E(teamName(c.team_id))}</td><td>${c.class_year}</td><td class="n">${c.n}</td><td class="n">${f(c.avg_stars, 2)}</td><td class="n">${f(c.avg_rating, 4)}</td><td class="n">${c.blue_chips}</td></tr>`).join('')}</table></div>`)}
    ${panel('Transfer projections', `<div class="scroll-sm"><table><tr><th>Player</th><th>Pos</th><th>From → To</th><th class="n">Prev adj EPA</th><th class="n">Projected</th><th class="n">80% interval</th></tr>${r.transfer_projections.map(t => `<tr><td><a href="#/players/${t.player_id}">${E(t.name)}</a></td><td>${t.position}</td><td>${E(teamAbbr(t.from_team_id))} → ${E(teamAbbr(t.to_team_id))}</td><td class="n">${fs(t.prev_adj_epa, 3)}</td><td class="n">${fs(t.proj_mean, 3)}</td><td class="n">${fs(t.proj_p10, 2)} … ${fs(t.proj_p90, 2)}</td></tr>`).join('') || '<tr><td colspan=6 class="mut">No transfers with prior-season data</td></tr>'}</table></div>`, 'conference-strength & regression adjusted')}
    </div>`;
  $('#rt').onchange = e => location.hash = `#/recruiting?team=${e.target.value}`;
}
async function transfersView(app) {
  const t = await api(`/api/transfers?season=${S.season}`);
  app.innerHTML = panel(`Transfer portal — ${S.season}`, `<div class="scroll" style="max-height:78vh"><table><tr><th>Player</th><th>Pos</th><th>From</th><th>To</th><th class="n">★</th><th class="n">Rating</th><th>Eligibility</th><th>Date</th></tr>${t.transfers.map(x => `<tr><td>${x.player_id ? `<a href="#/players/${x.player_id}">${E(x.name)}</a>` : E(x.name)}</td><td>${E(x.position || '')}</td><td>${E(x.from_school || '—')}</td><td>${E(x.to_school || '—')}</td><td class="n">${x.stars || ''}</td><td class="n">${f(x.rating, 4)}</td><td>${E(x.eligibility || '')}</td><td>${E(x.transfer_date || '')}</td></tr>`).join('')}</table></div>`, `${t.transfers.length} transfers ${tag('observed')}`);
}

/* ------------------------------------------------------------------ audit */
async function auditView(app) {
  const a = await api('/api/audit');
  const V = a.validation;
  if (!V.length) { app.innerHTML = panel('Model audit', '<div class="sub">No stored validation results. Run <code>python -m cfbquant.cli backtest --test …</code></div>'); return; }
  const models = ['sim_v1_cal', 'sim_v1', 'elo', 'ridge_margin', 'xgb_margin', 'polymarket_close'].filter(m => V.some(v => v.model_version === m));
  const g = (mv, m, sc = 'overall') => V.find(v => v.model_version === mv && v.metric === m && v.scope === sc);
  const cell = x => x ? `${f(x.value, 4)} <span class="mut">[${f(x.ci_low, 3)}, ${f(x.ci_high, 3)}]</span>` : '—';
  const dec = V.find(v => v.metric === 'promotion_decision');
  const seasons = [...new Set(V.filter(v => v.scope.startsWith('season:')).map(v => v.scope))];
  const cal = g('sim_v1_cal', 'calibration_curve'), calr = g('sim_v1', 'calibration_curve');
  const playerScopes = [...new Set(V.filter(v => v.scope.startsWith('player:')).map(v => v.scope))];
  app.innerHTML = `<div class="grid g2">
    ${panel('Game winner — out-of-sample', `<table><tr><th>Model</th><th>Log loss [90% CI]</th><th>Brier [90% CI]</th><th class="n">Accuracy</th><th class="n">n</th></tr>${models.map(m => `<tr><td>${m}</td><td class="n">${cell(g(m, 'log_loss'))}</td><td class="n">${cell(g(m, 'brier'))}</td><td class="n">${pc(g(m, 'accuracy')?.value)}</td><td class="n">${g(m, 'log_loss')?.n ?? ''}</td></tr>`).join('')}</table>
      ${g('sim_v1_cal', 'log_loss_minus_market') ? `<div class="note">Model vs Polymarket US closing price (games with a market): Δ log loss ${cell(g('sim_v1_cal', 'log_loss_minus_market'))}; edge ≥5pp return/contract at mid ${cell(g('sim_v1_cal', 'return_per_contract_at_mid', 'edge>=5pp'))}</div>` : '<div class="note">Market benchmark: none (requires real CFBD games that map to Polymarket US events).</div>'}
      <div class="note">Promotion gate (sim vs Elo, paired log-loss difference): ${dec ? `${f(dec.detail_json.value, 4)} [${f(dec.ci_low, 4)}, ${f(dec.ci_high, 4)}] → <b class="${dec.value ? 'pos' : 'amb'}">${dec.value ? 'PROMOTED to production' : 'NOT promoted (CI overlaps 0) — Elo remains production'}</b>` : 'n/a'}</div>`, 'walk-forward, data before kickoff only')}
    ${panel('Calibration (win probability)', `<div class="grid g2"><div><div class="sub">Simulation (walk-forward Platt)</div>${C.calib(cal?.detail_json?.points, {color: HOME})}<div class="sub">slope ${f(cal?.detail_json?.slope, 2)}, intercept ${f(cal?.detail_json?.intercept, 2)}</div></div><div><div class="sub">Simulation (raw)</div>${C.calib(calr?.detail_json?.points, {color: '#33c6ff'})}<div class="sub">slope ${f(calr?.detail_json?.slope, 2)}, intercept ${f(calr?.detail_json?.intercept, 2)}</div></div></div>`)}
    ${panel('Points & intervals', `<table><tr><th>Metric</th><th>Value [90% CI]</th></tr>${[['Margin MAE (sim)', 'sim_v1', 'margin_mae'], ['Margin MAE (Elo)', 'elo', 'margin_mae'], ['Margin MAE (ridge)', 'ridge_margin', 'margin_mae'], ['Margin MAE (XGB)', 'xgb_margin', 'margin_mae'], ['Total points MAE', 'sim_v1', 'total_mae'], ['Total bias (pred − actual)', 'sim_v1', 'total_bias'], ['Team score MAE', 'sim_v1', 'score_mae'], ['Margin 80% interval coverage', 'sim_v1', 'margin_80_coverage'], ['Total 80% interval coverage', 'sim_v1', 'total_80_coverage']].map(([l, m, k]) => `<tr><td>${l}</td><td class="n">${cell(g(m, k))}</td></tr>`).join('')}
      <tr><td>Score RMSE (margin, sim)</td><td class="n">${f(Math.sqrt(g('sim_v1', 'margin_rmse_sq')?.value), 2)}</td></tr></table>`)}
    ${panel('Player projections', `<table><tr><th>Scope</th><th>MAE (mean)</th><th>MAE (median)</th><th>80% coverage</th><th>Brier P(≥1)</th></tr>${playerScopes.map(s => `<tr><td>${s.replace('player:', '')}</td><td class="n">${cell(g('sim_v1', 'mae', s))}</td><td class="n">${f(g('sim_v1', 'mae_median', s)?.value, 2)}</td><td class="n">${pc(g('sim_v1', 'coverage_80', s)?.value)}</td><td class="n">${f(g('sim_v1', 'brier_ge1', s)?.value, 4)}</td></tr>`).join('')}</table>
      <div class="note">Evaluated on projected starters (QB1, RB1, WR1-2, TE1, K, LB1); a projected starter who did not record the stat counts as 0.</div>`)}
    ${panel('By season / segment (sim, calibrated)', `<table><tr><th>Scope</th><th class="n">Log loss</th><th class="n">Brier</th><th class="n">Acc.</th><th class="n">Elo LL</th><th class="n">n</th></tr>${[...seasons, 'conference_games', 'non_conference', 'injury_affected'].map(s => `<tr><td>${s}</td><td class="n">${f(g('sim_v1_cal', 'log_loss', s)?.value, 4)}</td><td class="n">${f(g('sim_v1_cal', 'brier', s)?.value, 4)}</td><td class="n">${pc(g('sim_v1_cal', 'accuracy', s)?.value)}</td><td class="n">${f(g('elo', 'log_loss', s)?.value, 4)}</td><td class="n">${g('sim_v1_cal', 'log_loss', s)?.n ?? ''}</td></tr>`).join('')}
      <tr><td>Favorites: win rate − predicted</td><td class="n" colspan=5>${cell(g('sim_v1_cal', 'win_rate_minus_pred', 'favorite'))}</td></tr></table>`)}
    ${panel('Model registry', `<table><tr><th>Version</th><th>Family</th><th>Status</th><th>Window</th><th>Notes</th></tr>${a.models.map(m => `<tr><td>${m.model_version}</td><td>${m.family}</td><td class="${m.status === 'production' ? 'pos' : ''}">${m.status}</td><td>${E(m.train_start || '')} → ${E(m.train_end || '')}</td><td class="sub" style="white-space:normal">${E(m.notes || '')}</td></tr>`).join('')}</table>`)}
    ${panel('Data quality', `<table>${a.data_quality.map(d => `<tr><td>${E(d.check)}</td><td class="n">${Number(d.value).toLocaleString()}</td></tr>`).join('')}</table>`)}
    ${panel('Drift diagnostics (league environment by season)', `<table><tr><th>Season</th><th class="n">EPA/play</th><th class="n">Success</th><th class="n">Yds/play</th><th class="n">Plays/team/g</th></tr>${a.drift.map(d => `<tr><td>${d.season}</td><td class="n">${fs(d.epa_play, 3)}</td><td class="n">${pc(d.sr)}</td><td class="n">${f(d.ypp, 2)}</td><td class="n">${f(d.plays_pg, 1)}</td></tr>`).join('')}</table><div class="note">${E(a.feature_importance_note)}</div>`)}
    ${panel('Recent stored forecasts vs results', `<div class="scroll-sm"><table><tr><th>Wk</th><th>Game</th><th class="n">P(home)</th><th class="n">Pred margin [P10,P90]</th><th class="n">Actual</th></tr>${a.recent.map(r => `<tr><td>${r.season}-${r.week}</td><td>${E(r.away)} @ ${E(r.home)}</td><td class="n">${pc(r.home_win_prob)}</td><td class="n">${fs(r.margin_mean, 1)} [${f(r.margin_p10, 0)}, ${f(r.margin_p90, 0)}]</td><td class="n ${(r.home_points - r.away_points >= r.margin_p10 && r.home_points - r.away_points <= r.margin_p90) ? '' : 'amb'}">${r.home_points - r.away_points > 0 ? '+' : ''}${r.home_points - r.away_points}</td></tr>`).join('')}</table></div>`)}
  </div>`;
}

boot();

/* ------------------------------------------------------------------ SGP Lab: props + same-game parlays */
const am = v => v == null || !isFinite(v) ? '—' : (v > 0 ? '+' : '') + Math.round(v);
S.slip = S.slip || [];
async function sgpView(app, h) {
  const week = h.q.get('week');
  const sl = await api(`/api/sgp/slate?${qs({season: S.season, week})}`);
  const gid = +(h.id || (sl.games.find(g => g.spread_home != null) || sl.games[0] || {}).game_id || 0);
  const spr = v => v == null ? '—' : (v > 0 ? '+' : '') + v;
  const slate = `<div class="chips" style="margin-bottom:6px">${sl.weeks.slice(0, 6).map(w => `<a class="chip ${w === sl.week ? 'on' : ''}" href="#/sgp?week=${w}">Wk ${w}</a>`).join('')}</div>
    <div class="scroll" style="max-height:78vh"><table style="table-layout:fixed;width:100%"><colgroup><col><col style="width:54px"><col style="width:44px"></colgroup><tr><th>Game</th><th class="n">Home</th><th class="n">O/U</th></tr>
    ${sl.games.map(g => `<tr class="${g.game_id === gid ? 'on' : ''}"><td style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap"><a href="#/sgp/${g.game_id}?week=${sl.week}">${E(g.away)} @ ${E(g.home)}</a><div class="sub">${E((g.start_date || '').replace('T', ' ').slice(5, 16))}${g.open_spread_home == null ? '' : ` · open ${spr(g.open_spread_home)} / ${f(g.open_total, 1)}`}</div></td>
      <td class="n">${spr(g.spread_home)}</td><td class="n">${f(g.over_under, 1)}</td></tr>`).join('')}</table></div>
    <div class="note">Lines: DraftKings via ESPN. Games without a posted line are simulated unanchored. Refresh with <code>cli lines --season ${sl.season}</code>.</div>`;
  if (!gid) { app.innerHTML = panel('SGP Lab', '<div class="sub">No upcoming games.</div>'); return; }
  app.innerHTML = `<div class="grid" style="grid-template-columns:300px 1fr 340px;gap:10px"><div>${panel(`Slate — week ${sl.week}`, slate)}</div><div id="sgp-main"><div class="loading">Simulating game (market-anchored, 6,000 sims)…</div></div><div id="sgp-side"></div></div>`;
  let c;
  try { c = await api(`/api/sgp/card/${gid}`); } catch (e) { $('#sgp-main').innerHTML = err(e); return; }
  S.sgpCard = c;
  if (S.slipGame !== gid) { S.slip = []; S.slipGame = gid; }
  const m = c.meta;
  const head = `<div class="row between"><div><b>${E(c.away)} @ ${E(c.home)}</b> <span class="sub">${E(m.as_of)} · ${m.n_sims.toLocaleString()} sims · ${tag('predicted')}</span></div>
    <div class="sub">${m.anchored ? `<span class="pos">ANCHORED</span> to DK ${E(c.home)} ${spr(m.line.spread_home)}, total ${m.line.total}` : '<span class="neg">UNANCHORED — no line posted</span>'}</div></div>
    <div class="sub" style="margin-top:4px">Sim: ${E(c.home)} ${f(c.game.home_pts, 1)} – ${f(c.game.away_pts, 1)} ${E(c.away)} · P(${E(c.home)} win) ${pc(c.game.home_win)} · margin SD ${f(c.game.margin_sd, 1)} · total SD ${f(c.game.total_sd, 1)}</div>`;
  const ticket = (t, k) => `<div class="panel" style="padding:8px;margin:6px 0">
    <div class="row between"><b>${k}-leg</b><span>P <b>${pc(t.p_joint)}</b> <span class="mut">[${pc(t.p_joint_ci90[0], 0)}–${pc(t.p_joint_ci90[1], 0)}]</span> · fair <b>${am(t.fair_american)}</b> · indep ${am(t.indep_american)} · lift <b class="pos">${f(t.lift, 2)}×</b></span></div>
    <table>${t.legs.map(l => `<tr><td>${E(l.label)}</td><td class="n">${pc(l.p, 0)}</td><td class="n mut">${am(l.fair_american)}</td></tr>`).join('')}</table></div>`;
  const tickets = Object.entries(c.tickets).map(([k, ts]) => ts.length ? ts.map(t => ticket(t, k)).join('') : `<div class="sub">${k}-leg: no ticket inside the probability band.</div>`).join('');
  const board = `<input class="search" id="pbs" placeholder="Filter player / market…" style="max-width:240px">
    <div class="scroll" style="max-height:46vh"><table id="pbt"><tr><th>Player</th><th>Team</th><th>Market</th><th class="n">Line</th><th class="n">P(over)</th><th class="n">Fair O</th><th class="n">Fair U</th><th class="n">p10</th><th class="n">Median</th><th class="n">p90</th><th></th></tr>
    ${c.props.map((p, i) => `<tr data-n="${E(((p.name || '') + ' ' + p.market + ' ' + p.team).toLowerCase())}"><td>${E(p.name || p.slot)}</td><td class="sub">${E(p.team)}</td><td>${E(p.market)}</td><td class="n"><input class="ln" data-i="${i}" value="${p.line}" style="width:52px"></td>
      <td class="n">${pc(p.p_over, 0)}</td><td class="n">${am(p.fair_over)}</td><td class="n">${am(p.fair_under)}</td><td class="n mut">${f(p.p10, 0)}</td><td class="n">${f(p.median, 0)}</td><td class="n mut">${f(p.p90, 0)}</td>
      <td><a class="chip addleg" data-i="${i}" data-op="over">+O</a><a class="chip addleg" data-i="${i}" data-op="under">+U</a></td></tr>`).join('')}</table></div>
    <div class="row" style="margin-top:6px">Game lines: ${[['ml', 'home', null, 'ML'], ['ml', 'away', null, 'ML']].map(([k, sd]) => `<a class="chip addgl" data-k="${k}" data-s="${sd}">+${E(sd === 'home' ? c.home : c.away)} ML</a>`).join('')}
      ${m.line.spread_home != null ? `<a class="chip addgl" data-k="spread" data-s="home">+${E(c.home)} ${spr(m.line.spread_home)}</a><a class="chip addgl" data-k="spread" data-s="away">+${E(c.away)} ${spr(-m.line.spread_home)}</a>` : ''}
      ${m.line.total != null ? `<a class="chip addgl" data-k="total" data-s="over">+Over ${m.line.total}</a><a class="chip addgl" data-k="total" data-s="under">+Under ${m.line.total}</a>` : ''}</div>`;
  $('#sgp-main').innerHTML = panel('Game', head) + panel('Prop board — main lines', board, 'edit a line to re-price · add legs to the slip') +
    panel('Auto tickets — most-correlated legs inside each size\'s probability band', tickets,
      'priced on held-out sims · bet only if the book pays more than fair');
  $('#pbs').oninput = e => document.querySelectorAll('#pbt tr[data-n]').forEach(r => r.style.display = r.dataset.n.includes(e.target.value.toLowerCase()) ? '' : 'none');
  document.querySelectorAll('.addleg').forEach(a => a.onclick = () => {
    const p = c.props[+a.dataset.i]; const ln = +document.querySelector(`.ln[data-i="${a.dataset.i}"]`).value;
    S.slip.push({kind: 'player', player_id: p.player_id, stat: p.stat, op: a.dataset.op, line: ln, name: p.name}); slipRender();
  });
  document.querySelectorAll('.addgl').forEach(a => a.onclick = () => {
    const k = a.dataset.k, sd = a.dataset.s;
    const leg = k === 'ml' ? {kind: 'ml', side: sd, name: sd === 'home' ? c.home : c.away}
      : k === 'spread' ? {kind: 'spread', side: sd, line: sd === 'home' ? m.line.spread_home : -m.line.spread_home, name: sd === 'home' ? c.home : c.away}
      : {kind: 'total', op: sd, line: m.line.total};
    S.slip.push(leg); slipRender();
  });
  slipRender();
  sgpValidation();
}
function slipRender() {
  const box = $('#sgp-side'); if (!box) return;
  const lab = l => l.kind === 'player' ? `${l.name} ${l.op === 'over' ? 'O' : 'U'} ${l.line} ${(S.sgpCard.props.find(p => p.stat === l.stat) || {}).market || l.stat}`
    : l.kind === 'ml' ? `${l.name} ML` : l.kind === 'spread' ? `${l.name} ${l.line > 0 ? '+' : ''}${l.line}` : `Game ${l.op} ${l.line}`;
  box.innerHTML = panel(`Bet slip (${S.slip.length} legs)`, `${S.slip.length ? `<table>${S.slip.map((l, i) => `<tr><td>${E(lab(l))}</td><td><a class="chip rmleg" data-i="${i}">×</a></td></tr>`).join('')}</table>` : '<div class="sub">Add legs from the prop board.</div>'}
    <div class="row" style="margin-top:6px"><label class="sub">Book odds <input id="bko" placeholder="+600" style="width:70px"></label> <a class="chip on" id="prc">Price</a> <a class="chip" id="clr">Clear</a></div><div id="prc-out"></div>`) + '<div id="sgp-val"></div>';
  document.querySelectorAll('.rmleg').forEach(a => a.onclick = () => { S.slip.splice(+a.dataset.i, 1); slipRender(); });
  $('#clr').onclick = () => { S.slip = []; slipRender(); };
  $('#prc').onclick = async () => {
    if (!S.slip.length) return;
    const bo = $('#bko').value.trim(); const body = {game_id: S.slipGame, legs: S.slip, book_american: bo ? +bo : null};
    $('#prc-out').innerHTML = '<div class="loading">Pricing…</div>';
    const r = await fetch('/api/sgp/price', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    if (!r.ok) { $('#prc-out').innerHTML = err(new Error(await r.text())); return; }
    const t = await r.json();
    $('#prc-out').innerHTML = `<table style="margin-top:6px"><tr><td>Joint P</td><td class="n"><b>${pc(t.p_joint)}</b> <span class="mut">[${pc(t.p_joint_ci90[0])}–${pc(t.p_joint_ci90[1])}]</span></td></tr>
      <tr><td>Fair odds</td><td class="n"><b>${am(t.fair_american)}</b></td></tr><tr><td>If independent</td><td class="n">${pc(t.p_indep)} · ${am(t.indep_american)}</td></tr>
      <tr><td>Correlation lift</td><td class="n">${f(t.lift, 2)}×</td></tr>
      ${t.book_american != null ? `<tr><td>Book implied</td><td class="n">${pc(t.book_implied)}</td></tr><tr><td>EV / unit</td><td class="n ${cls(t.ev_per_unit)}"><b>${fs(t.ev_per_unit, 3)}</b></td></tr><tr><td>¼ Kelly stake</td><td class="n">${pc(t.quarter_kelly, 2)} of bankroll</td></tr>` : ''}</table>
      ${t.legs.map(l => `<div class="sub">${E(l.label)}: ${pc(l.p, 1)}</div>`).join('')}
      <div class="note">Model price ${tag('predicted')}. Includes parameter uncertainty; excludes injury news after the line was captured.</div>`;
  };
}
async function sgpValidation() {
  let v; try { v = await api('/api/sgp/validation'); } catch (e) { return; }
  const box = $('#sgp-val'); if (!box || !v.rows.length) return;
  const t = v.rows.filter(r => r.scope.startsWith('tickets:'));
  const by = {}; t.forEach(r => { (by[r.scope] = by[r.scope] || {})[r.metric] = r; });
  box.innerHTML = panel('Walk-forward validation', `<table><tr><th>Size</th><th class="n">n</th><th class="n">Pred</th><th class="n">Hit</th><th class="n">Indep</th></tr>
    ${Object.entries(by).map(([k, r]) => `<tr><td>${E(k.split(':')[1])}-leg</td><td class="n">${r.hit_rate?.n ?? ''}</td><td class="n">${pc(r.pred_joint?.value)}</td><td class="n">${pc(r.hit_rate?.value)} <span class="mut">±${pc(1.645 * (r.hit_rate?.detail_json ? JSON.parse(r.hit_rate.detail_json).se : 0), 0)}</span></td><td class="n mut">${pc(r.pred_indep?.value)}</td></tr>`).join('')}</table>
    <div class="note">Real 2026 games, parameters estimated before each week, priced on held-out sims. Void tickets (a player leg's player did not appear) excluded.</div>`);
}
