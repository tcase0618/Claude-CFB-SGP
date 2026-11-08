"""FastAPI backend: serves precomputed analytics + on-demand simulations, and the static terminal UI.

Every response carries `meta.synthetic` so the UI can never present demo data as real.
"""
from __future__ import annotations

import json
import math
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import ROOT, get_settings
from ..db import connect, df, is_synthetic, migrate

app = FastAPI(title="CFB Quant Intelligence Platform", version="0.1.0")
_lock = threading.Lock()
_conn = None


def conn():
    global _conn
    if _conn is None:
        _conn = connect(get_settings().db_path)
        migrate(_conn)
    return _conn


def clean(o):
    """JSON-safe conversion (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def records(frame: pd.DataFrame) -> list[dict]:
    return clean(frame.to_dict(orient="records"))


def q(sql, params=()):
    with _lock:
        return df(conn(), sql, params)


def latest(season: int | None):
    c = conn()
    if season is None:
        season = c.execute("SELECT MAX(season) FROM games").fetchone()[0]
    wk = c.execute("SELECT MAX(through_week) FROM team_metrics WHERE season=?", (season,)).fetchone()[0]
    return int(season) if season else None, int(wk or 0)


# ------------------------------------------------------------------ meta
@app.get("/api/health")
def health():
    c = conn()
    n = c.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    return {"status": "ok", "games": n, "db": str(get_settings().db_path.name)}


@app.get("/api/meta")
def meta():
    c = conn()
    src = records(q("SELECT * FROM data_sources"))
    seasons = [r[0] for r in c.execute("SELECT DISTINCT season FROM games ORDER BY 1")]
    weeks = {s: c.execute("SELECT MAX(week) FROM games WHERE season=? AND completed=1", (s,)).fetchone()[0] for s in seasons}
    mv = records(q("SELECT model_version, family, created_at, train_start, train_end, status, notes FROM model_versions ORDER BY created_at DESC"))
    counts = records(q("SELECT scope, classification, COUNT(*) n FROM metric_definitions GROUP BY 1,2"))
    last_plays = c.execute("SELECT MAX(fetched_at) FROM ingest_log").fetchone()[0]
    return {"synthetic": is_synthetic(c), "sources": src, "seasons": seasons, "latest_week": weeks,
            "models": mv, "metric_counts": counts, "last_ingest": last_plays,
            "counts": {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
                       ("teams", "games", "plays", "players", "play_events", "player_games")}}


@app.get("/api/metrics/definitions")
def definitions(scope: str | None = None):
    if scope:
        return records(q("SELECT * FROM metric_definitions WHERE scope=?", (scope,)))
    return records(q("SELECT * FROM metric_definitions"))


# ------------------------------------------------------------------ teams
def _team_table(season, wk):
    t = q("""SELECT t.team_id, t.school, t.mascot, t.abbreviation, t.color, ts.conference_id AS conf, ts.talent_composite,
             ts.returning_production FROM teams t JOIN team_seasons ts ON ts.team_id=t.team_id AND ts.season=?""", (season,))
    m = q("""SELECT team_id, metric_id, value, adj_value, pctl_fbs FROM team_metrics WHERE season=? AND through_week=?
             AND metric_id IN ('off_epa','def_epa','net_epa','rec_wins','rec_losses','off_sr','def_sr','sos','ppg','papg')""",
          (season, wk))
    v = m.assign(v=m.adj_value.where(m.adj_value.notna(), m.value)).pivot(index="team_id", columns="metric_id", values="v")
    pc = m.pivot(index="team_id", columns="metric_id", values="pctl_fbs").add_suffix("_pctl")
    t = t.join(v, on="team_id").join(pc, on="team_id")
    elo = q("""SELECT team_id, rating FROM team_ratings WHERE season=? AND system='elo' AND as_of_week=
               (SELECT MAX(as_of_week) FROM team_ratings WHERE season=? AND system='elo')""", (season, season))
    t = t.merge(elo.rename(columns={"rating": "elo"}), on="team_id", how="left")
    t["team_grade"] = (t.get("net_epa_pctl", pd.Series(dtype=float))).round(1)
    return t


@app.get("/api/teams")
def teams(season: int | None = None):
    season, wk = latest(season)
    t = _team_table(season, wk).sort_values("net_epa", ascending=False)
    return {"season": season, "through_week": wk, "teams": records(t)}


@app.get("/api/teams/{team_id}")
def team(team_id: int, season: int | None = None):
    season, wk = latest(season)
    tt = _team_table(season, wk)
    if team_id not in set(tt.team_id):
        raise HTTPException(404, "team not found for season")
    info = tt[tt.team_id == team_id].iloc[0].to_dict()
    m = q("""SELECT tm.metric_id, tm.value, tm.adj_value, tm.sample, tm.pctl_fbs, tm.pctl_conf, tm.status, md.category, md.name,
             md.formula, md.direction, md.classification, md.required_data, md.opponent_adjustable
             FROM team_metrics tm JOIN metric_definitions md ON md.metric_id=tm.metric_id AND md.scope='team'
             WHERE tm.team_id=? AND tm.season=? AND tm.through_week=?""", (team_id, season, wk))
    trend = q("""SELECT p.game_id, g.week, g.start_date,
        CASE WHEN p.offense_id=? THEN 'off' ELSE 'def' END AS side, AVG(p.epa) epa, AVG(p.success) sr,
        AVG(CASE WHEN p.play_type IN ('pass','sack','scramble') THEN p.epa END) pass_epa,
        AVG(CASE WHEN p.play_type='rush' THEN p.epa END) rush_epa, COUNT(*) n,
        CASE WHEN g.home_id=? THEN g.away_id ELSE g.home_id END AS opp_id
        FROM plays p JOIN games g ON g.game_id=p.game_id
        WHERE g.season=? AND (p.offense_id=? OR p.defense_id=?) AND p.play_type IN ('pass','rush','sack','scramble')
          AND p.garbage_time=0 GROUP BY p.game_id, side ORDER BY g.week""", (team_id, team_id, season, team_id, team_id))
    sched = q("""SELECT g.game_id, g.week, g.start_date, g.home_id, g.away_id, g.home_points, g.away_points, g.completed,
                 h.school home, a.school away FROM games g JOIN teams h ON h.team_id=g.home_id JOIN teams a ON a.team_id=g.away_id
                 WHERE g.season=? AND (g.home_id=? OR g.away_id=?) ORDER BY g.week""", (season, team_id, team_id))
    roster = q("""SELECT pp.player_id, pl.first_name, pl.last_name, pp.position, pp.archetype, pp.grade, pp.grade_pctl, r.class_year,
                  rp.stars FROM player_profiles pp JOIN players pl ON pl.player_id=pp.player_id
                  LEFT JOIN rosters r ON r.player_id=pp.player_id AND r.season=pp.season
                  LEFT JOIN recruiting_profiles rp ON rp.player_id=pp.player_id
                  WHERE pp.team_id=? AND pp.season=? AND pp.through_week=? ORDER BY pp.grade DESC""", (team_id, season, wk)) \
        if _has_table("player_profiles") else pd.DataFrame()
    cats = {}
    for r in records(m):
        cats.setdefault(r["category"], []).append(r)
    from ..analytics.team_metrics import CATS
    return {"season": season, "through_week": wk, "team": clean(info), "categories": [{"name": c, "metrics": cats.get(c, [])} for c in CATS],
            "trend": records(trend), "schedule": records(sched), "roster": records(roster)}


def _has_table(name):
    return conn().execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


# ------------------------------------------------------------------ players
@app.get("/api/players")
def players(season: int | None = None, team: int | None = None, position: str | None = None, q_: str | None = Query(None, alias="q"),
            limit: int = 200):
    season, wk = latest(season)
    sql = """SELECT pp.player_id, pl.first_name, pl.last_name, pp.position, pp.team_id, t.school, pp.archetype, pp.grade, pp.grade_pctl,
             r.class_year FROM player_profiles pp JOIN players pl ON pl.player_id=pp.player_id JOIN teams t ON t.team_id=pp.team_id
             LEFT JOIN rosters r ON r.player_id=pp.player_id AND r.season=pp.season
             WHERE pp.season=? AND pp.through_week=?"""
    params: list = [season, wk]
    if team:
        sql += " AND pp.team_id=?"; params.append(team)
    if position:
        sql += " AND pp.position=?"; params.append(position)
    if q_:
        sql += " AND (pl.first_name || ' ' || pl.last_name) LIKE ?"; params.append(f"%{q_}%")
    sql += " ORDER BY pp.grade IS NULL, pp.grade DESC LIMIT ?"
    params.append(limit)
    return {"season": season, "through_week": wk, "players": records(q(sql, tuple(params)))}


@app.get("/api/players/{player_id}")
def player(player_id: int, season: int | None = None, project: bool = True):
    season, wk = latest(season)
    prof = q("""SELECT pp.*, pl.first_name, pl.last_name, pl.height_in, pl.weight_lb, t.school, t.color, r.class_year, r.jersey,
                ts.conference_id conf FROM player_profiles pp JOIN players pl ON pl.player_id=pp.player_id
                JOIN teams t ON t.team_id=pp.team_id LEFT JOIN rosters r ON r.player_id=pp.player_id AND r.season=pp.season
                LEFT JOIN team_seasons ts ON ts.team_id=pp.team_id AND ts.season=pp.season
                WHERE pp.player_id=? AND pp.season=? AND pp.through_week=?""", (player_id, season, wk))
    if prof.empty:
        raise HTTPException(404, "player not found")
    p = prof.iloc[0].to_dict()
    pos = p["position"]
    m = q("""SELECT md.metric_id, md.scope, md.category, md.name, md.positions, md.formula, md.direction, md.classification,
             md.required_data, pm.value, pm.sample, pm.pctl_fbs, pm.pctl_conf, pm.status
             FROM metric_definitions md LEFT JOIN player_metrics pm ON pm.metric_id=md.metric_id AND pm.player_id=? AND pm.season=?
             AND pm.through_week=? WHERE md.scope IN ('player_universal','player_position')""", (player_id, season, wk))
    m = m[m.positions.apply(lambda s: s == "ALL" or pos in str(s).split(","))]
    m["status"] = np.where(m.classification == "unavailable_without_charting", "unavailable", m.status.fillna("no_data"))
    uni = {}
    for r in records(m[m.scope == "player_universal"]):
        uni.setdefault(r["category"], []).append(r)
    posm = records(m[m.scope == "player_position"])
    log = q("""SELECT pg.*, g.week, g.home_id, g.away_id, g.home_points, g.away_points,
               CASE WHEN g.home_id=pg.team_id THEN a.school ELSE h.school END AS opponent,
               CASE WHEN g.home_id=pg.team_id THEN 'vs' ELSE '@' END AS ha
               FROM player_games pg JOIN games g ON g.game_id=pg.game_id JOIN teams h ON h.team_id=g.home_id JOIN teams a ON a.team_id=g.away_id
               WHERE pg.player_id=? AND g.season=? ORDER BY g.week""", (player_id, season))
    trend = q("""SELECT g.week, AVG(p.epa) epa, AVG(p.success) sr, COUNT(*) n FROM play_events e JOIN plays p ON p.play_id=e.play_id
                 JOIN games g ON g.game_id=p.game_id WHERE e.player_id=? AND g.season=? AND p.garbage_time=0
                 AND ((e.role='passer' AND p.play_type IN ('pass','sack')) OR (e.role='rusher' AND p.play_type IN ('rush','scramble'))
                 OR (e.role='target' AND p.play_type='pass')) GROUP BY g.week ORDER BY g.week""", (player_id, season))
    chart = None
    if pos == "QB":
        tc = q("""SELECT p.air_yards, p.completion, p.epa, p.interception, p.touchdown, p.pass_location FROM play_events e
                  JOIN plays p ON p.play_id=e.play_id WHERE e.player_id=? AND e.role='passer' AND p.play_type='pass' AND p.season=?""",
               (player_id, season))
        chart = _throw_chart(tc)
    proj = None
    if project:
        proj = _next_game_projection(player_id, int(p["team_id"]), season)
    return {"season": season, "through_week": wk, "player": clean(p), "universal": [{"name": k, "metrics": v} for k, v in sorted(uni.items())],
            "position_metrics": posm, "game_log": records(log), "trend": records(trend), "throw_chart": chart, "projection": proj}


def _throw_chart(tc: pd.DataFrame) -> dict:
    if tc.empty:
        return {"available": False, "reason": "no attempts"}
    cov = float(tc.air_yards.notna().mean())
    if tc.air_yards.notna().sum() < 20:
        return {"available": False, "reason": f"Air yards parsed for only {tc.air_yards.notna().sum()} attempts (source text lacks catch/throw spots)"}
    tc = tc[tc.air_yards.notna()]
    buckets = [("Behind LOS", -99, -1), ("0-9", 0, 9), ("10-19", 10, 19), ("20-29", 20, 29), ("30+", 30, 99)]
    rows = []
    for nm, lo, hi in buckets:
        s = tc[tc.air_yards.between(lo, hi)]
        rows.append({"depth": nm, "att": int(len(s)), "share": float(len(s) / len(tc)),
                     "comp_pct": float(s.completion.mean()) if len(s) else None, "epa": float(s.epa.mean()) if len(s) else None,
                     "td": int(s.touchdown.sum()), "int": int(s.interception.sum())})
    return {"available": True, "rows": rows, "coverage": cov, "location_available": bool(tc.pass_location.notna().any()),
            "location_note": None if tc.pass_location.notna().any() else "Pass location (L/M/R) not charted in source"}


def _next_game_projection(player_id, team_id, season):
    c = conn()
    g = c.execute("""SELECT game_id, week, home_id, away_id, neutral_site, completed FROM games WHERE season=? AND (home_id=? OR away_id=?)
                     AND completed=0 ORDER BY week LIMIT 1""", (season, team_id, team_id)).fetchone()
    note = "next scheduled game"
    if g is None:   # season complete: project the last game played (retrospective, as-of before that week)
        g = c.execute("""SELECT game_id, week, home_id, away_id, neutral_site, completed FROM games WHERE season=? AND (home_id=? OR away_id=?)
                         ORDER BY week DESC LIMIT 1""", (season, team_id, team_id)).fetchone()
        note = "season complete — retrospective projection of final game using data before that week"
    if g is None:
        return None
    fc = _forecast(int(g["home_id"]), int(g["away_id"]), season, int(g["week"]), 4000, 7, bool(g["neutral_site"]), False, False)
    pl = next((x for x in fc["players"] if x["player_id"] == player_id), None)
    return {"game_id": g["game_id"], "week": g["week"], "home_id": g["home_id"], "away_id": g["away_id"], "note": note,
            "team_win_prob": fc["home_win_prob"] if team_id == g["home_id"] else fc["away_win_prob"],
            "player": pl, "n_sims": fc["n_sims"]}


# ------------------------------------------------------------------ games / forecasts
@app.get("/api/games")
def games(season: int | None = None, week: int | None = None):
    season, _ = latest(season)
    sql = """SELECT g.*, h.school home, a.school away, h.abbreviation home_abbr, a.abbreviation away_abbr FROM games g
             JOIN teams h ON h.team_id=g.home_id JOIN teams a ON a.team_id=g.away_id WHERE g.season=?"""
    params = [season]
    if week:
        sql += " AND g.week=?"; params.append(week)
    gm = q(sql + " ORDER BY g.week, g.game_id", tuple(params))
    pr = q("""SELECT game_id, model_version, home_win_prob, margin_mean, total_mean FROM model_predictions WHERE kind='backtest'
              AND model_version IN ('sim_v1_cal','elo')""")
    if not pr.empty:
        pv = pr.pivot_table(index="game_id", columns="model_version", values=["home_win_prob", "margin_mean", "total_mean"])
        pv.columns = [f"{a}_{b}" for a, b in pv.columns]
        gm = gm.join(pv, on="game_id")
    return {"season": season, "games": records(gm)}


@lru_cache(maxsize=128)
def _forecast(home, away, season, week, sims, seed, neutral, drivers=True, scenarios=True):
    from ..models.forecast import forecast_game
    with _lock:
        return clean(forecast_game(conn(), home, away, season, week, n_sims=sims, seed=seed, neutral=neutral,
                                   drivers=drivers, scenarios=scenarios))


@app.get("/api/forecast")
def forecast(home: int, away: int, season: int | None = None, week: int | None = None, sims: int = 10000, seed: int = 20241109,
             neutral: bool = False):
    season, wk = latest(season)
    if week is None:
        week = wk + 1
    sims = int(min(max(sims, 500), 50000))
    fc = dict(_forecast(home, away, season, week, sims, seed, neutral))
    names = q("SELECT team_id, school, abbreviation, color FROM teams WHERE team_id IN (?,?)", (home, away)).set_index("team_id")
    fc["home"] = clean(names.loc[home].to_dict()) if home in names.index else None
    fc["away"] = clean(names.loc[away].to_dict()) if away in names.index else None
    g = conn().execute("SELECT * FROM games WHERE season=? AND week=? AND home_id=? AND away_id=?", (season, week, home, away)).fetchone()
    fc["game"] = dict(g) if g else None
    return fc


# ------------------------------------------------------------------ matchup lab
UNIT_PAIRS = [
    ("Passing game", "pass_epa", "pdef_epa", "EPA/db"), ("Rushing game", "rush_epa", "rdef_epa", "EPA/carry"),
    ("Pass protection vs rush", "pass_sack", "pr_sack", "sack rate"), ("Explosiveness", "off_expl", "def_expl", "rate"),
    ("Success rate", "off_sr", "def_sr", "rate"), ("Run blocking vs front", "rush_stuff", "rdef_stuff", "stuff rate"),
    ("Receiving vs coverage", "rec_epa", "pdef_epa", "EPA/target"), ("Red zone", "rz_epa", "rz_def_epa", "EPA/play"),
    ("Third down", "dd_d3_epa", "dd_d3_def", "EPA / SR"), ("Turnovers", "off_to", "def_to", "rate"),
]


@app.get("/api/matchup")
def matchup(home: int, away: int, season: int | None = None):
    season, wk = latest(season)
    m = q("""SELECT team_id, metric_id, value, adj_value, sample, pctl_fbs FROM team_metrics WHERE season=? AND through_week=?
             AND team_id IN (?,?)""", (season, wk, home, away))
    lg = q("SELECT metric_id, AVG(COALESCE(adj_value, value)) lg, AVG(value) lg_raw FROM team_metrics WHERE season=? AND through_week=? GROUP BY 1",
           (season, wk)).set_index("metric_id")
    d = {(r.team_id, r.metric_id): r for r in m.itertuples()}

    def val(t, mid):
        r = d.get((t, mid))
        if r is None:
            return None, None, None
        v = r.adj_value if r.adj_value is not None and not pd.isna(r.adj_value) else r.value
        return v, r.sample, r.pctl_fbs
    from ..analytics.team_metrics import ALL_TEAM_METRICS
    dirs = {x.mid: x.direction for x in ALL_TEAM_METRICS}
    rows = []
    for off_t, def_t, side in ((home, away, "home_off"), (away, home, "away_off")):
        for label, om, dm, unit in UNIT_PAIRS:
            ov, on, op = val(off_t, om)
            dv, dn, dp = val(def_t, dm)
            if ov is None or dv is None:
                continue
            L = lg.lg.get(om, np.nan)
            Ld = lg.lg.get(dm, np.nan)
            # offence edge in offence-favourable units: offence above league + defence allowing above league
            od, dd = dirs.get(om, 1), dirs.get(dm, 1)
            edge = od * (ov - L) - dd * (dv - Ld)
            se = None
            if on and dn:
                sd_unit = 1.3 if "epa" in om else 0.5
                se = sd_unit * math.sqrt(1 / max(on, 1) + 1 / max(dn, 1))
            rows.append({"side": side, "unit": label, "metric_unit": unit, "off_metric": om, "def_metric": dm,
                         "off_value": ov, "def_value": dv, "off_pctl": op, "def_pctl": dp, "league_off": L,
                         "league_def": Ld, "edge": edge, "edge_se": se,
                         "z": (edge / se) if se else None})
    tend_ids = ["sch_pass_rate", "cc_proe", "cc_pace", "cc_plays_pg", "cc_go_rate", "cc_predict", "st_fg_voe", "st_hidden",
                "st_punt_net", "pass_adot", "sch_neutral_pass", "cc_halftime"]
    tend = []
    for mid in tend_ids:
        hv, _, hp = val(home, mid)
        av, _, ap = val(away, mid)
        tend.append({"metric_id": mid, "home": hv, "away": av, "home_pctl": hp, "away_pctl": ap, "league": lg.lg.get(mid)})
    names = q("SELECT team_id, school FROM teams WHERE team_id IN (?,?)", (home, away)).set_index("team_id").school.to_dict()
    defs = {x.mid: x.name for x in ALL_TEAM_METRICS}
    for t in tend:
        t["name"] = defs.get(t["metric_id"], t["metric_id"])
    return {"season": season, "through_week": wk, "home": names.get(home), "away": names.get(away),
            "units": clean(rows), "tendencies": clean(tend),
            "note": "Edge = offence deviation from league + defence-allowed deviation from league, opponent-adjusted where available; "
                    "SE from play counts (approximate). Positive favours the offence."}


# ------------------------------------------------------------------ recruiting / transfers
@app.get("/api/recruiting")
def recruiting(season: int | None = None, team: int | None = None):
    from ..analytics import talent
    season, wk = latest(season)
    c = conn()
    classes = q("""SELECT committed_team_id team_id, class_year, COUNT(*) n, AVG(stars) avg_stars, AVG(rating) avg_rating,
                   SUM(stars>=4) blue_chips FROM recruiting_profiles WHERE committed_team_id IS NOT NULL
                   GROUP BY 1,2 ORDER BY 2""")
    talent_t = q("""SELECT ts.team_id, t.school, ts.season, ts.talent_composite, ts.returning_production FROM team_seasons ts
                   JOIN teams t ON t.team_id=ts.team_id WHERE ts.season=?""", (season,))
    post = talent.posteriors(c, season)
    br = talent.breakout_candidates(post, team)
    curves = talent.development_curves(c)
    tp = talent.transfer_projections(c, season)
    names = q("SELECT player_id, first_name || ' ' || last_name AS name FROM players").set_index("player_id").name
    if not br.empty:
        br = br.assign(name=br.player_id.map(names))
    if team:
        classes = classes[classes.team_id == team]
        post = post[post.team_id == team] if not post.empty else post
        if not tp.empty:
            tp = tp[(tp.to_team_id == team) | (tp.from_team_id == team)]
    if not post.empty:
        post = post.assign(name=post.player_id.map(names))
    keep = ["player_id", "name", "position", "team_id", "class_year", "stars", "rating", "sample", "adj_epa_play",
            "prior_mean", "prior_sd", "post_mean", "post_p10", "post_p90", "usage"]
    return {"season": season, "classes": records(classes), "talent": records(talent_t),
            "breakouts": records(br[[c_ for c_ in keep if c_ in br]]) if not br.empty else [],
            "posteriors": records(post[[c_ for c_ in keep if c_ in post]].sort_values("post_mean", ascending=False).head(150)) if not post.empty else [],
            "development_curves": records(curves), "transfer_projections": records(tp)}


@app.get("/api/transfers")
def transfers(season: int | None = None):
    season, _ = latest(season)
    t = q("""SELECT th.*, f.school from_school, d.school to_school FROM transfer_history th
             LEFT JOIN teams f ON f.team_id=th.from_team_id LEFT JOIN teams d ON d.team_id=th.to_team_id WHERE th.season=?
             ORDER BY th.rating DESC""", (season,))
    return {"season": season, "transfers": records(t)}


# ------------------------------------------------------------------ audit
@app.get("/api/audit")
def audit():
    v = q("SELECT * FROM model_validation ORDER BY model_version, scope, metric")
    for col in ("detail_json",):
        v[col] = v[col].apply(lambda s: json.loads(s) if isinstance(s, str) else None)
    mv = q("SELECT * FROM model_versions ORDER BY created_at DESC")
    recent = q("""SELECT mp.game_id, g.season, g.week, h.school home, a.school away, g.home_points, g.away_points, mp.home_win_prob,
                  mp.margin_mean, mp.total_mean, mp.margin_p10, mp.margin_p90 FROM model_predictions mp
                  JOIN games g ON g.game_id=mp.game_id JOIN teams h ON h.team_id=g.home_id JOIN teams a ON a.team_id=g.away_id
                  WHERE mp.kind='backtest' AND mp.model_version='sim_v1_cal' ORDER BY g.season DESC, g.week DESC LIMIT 60""")
    c = conn()
    dq = []
    for nm, sql in [("plays without EPA (scrimmage)", "SELECT COUNT(*) FROM plays WHERE play_type IN ('pass','rush','sack','scramble') AND epa IS NULL AND period<=4"),
                    ("plays with player attribution", "SELECT COUNT(DISTINCT play_id) FROM play_events"),
                    ("games completed", "SELECT COUNT(*) FROM games WHERE completed=1"),
                    ("games missing scores", "SELECT COUNT(*) FROM games WHERE completed=1 AND (home_points IS NULL OR away_points IS NULL)"),
                    ("plays with air yards", "SELECT COUNT(*) FROM plays WHERE air_yards IS NOT NULL"),
                    ("charting rows (formations)", "SELECT COUNT(*) FROM offensive_formations"),
                    ("participation rows", "SELECT COUNT(*) FROM player_participation")]:
        dq.append({"check": nm, "value": c.execute(sql).fetchone()[0]})
    drift = q("""SELECT season, AVG(epa) epa_play, AVG(success) sr, AVG(yards_gained) ypp, COUNT(*)*1.0/COUNT(DISTINCT game_id)/2 plays_pg
                 FROM plays WHERE play_type IN ('pass','rush','sack','scramble') AND garbage_time=0 GROUP BY season""")
    return {"validation": records(v), "models": records(mv), "recent": records(recent), "data_quality": dq,
            "drift": records(drift), "feature_importance": None,
            "feature_importance_note": "Driver attribution for each forecast is computed by unit neutralisation (see /api/forecast drivers)."}


# ------------------------------------------------------------------ markets (Polymarket US, real public data)
_mconn = None


def mconn():
    global _mconn
    from ..ingest.polymarket import markets_db_path, open_markets_db
    if _mconn is None:
        if not markets_db_path().exists():
            raise HTTPException(404, "No market data yet. Run: python -m cfbquant.cli markets live")
        _mconn = open_markets_db()
    return _mconn


def mq(sql, params=()):
    with _lock:
        return df(mconn(), sql, params)


MARKET_NOTE = ("Real Polymarket US market data (public gateway, no API key). Prices are order-book mids; "
               "implied spreads/totals are normal fits to the alternate-line ladders. Teams listed away-first per Polymarket.")


@app.get("/api/markets/board")
def markets_board(kind: str = "live", days: int = 8):
    ev = mq("""SELECT e.*, ta.name away, ta.display_abbr away_abbr, ta.ranking away_rank, ta.color away_color, ta.conference away_conf,
               th.name home, th.display_abbr home_abbr, th.ranking home_rank, th.color home_color, th.conference home_conf
               FROM pm_events e LEFT JOIN pm_teams ta ON ta.pm_team_id=e.away_pm_team_id LEFT JOIN pm_teams th ON th.pm_team_id=e.home_pm_team_id""")
    imp = mq("SELECT * FROM pm_implied WHERE kind=?", (kind,))
    if imp.empty:
        return {"kind": kind, "events": [], "note": MARKET_NOTE, "synced_at": None, "linked_to_model": False}
    imp = imp.sort_values("ts").groupby("event_id").tail(1)
    d = ev.merge(imp, on="event_id", how="inner")
    if kind == "live":
        d = d[(d.closed == 0) & (d.ended == 0)]
    d = d.sort_values("start_time")
    linked = False
    c = conn()
    if not is_synthetic(c):
        from ..ingest.polymarket import map_events_to_games
        mp = map_events_to_games(mconn(), c)
        if not mp.empty:
            d = d.merge(mp, on="event_id", how="left")
            linked = True
    last = mconn().execute("SELECT MAX(ts) FROM pm_sync_log WHERE what=?", (kind,)).fetchone()[0]
    return {"kind": kind, "events": records(d), "note": MARKET_NOTE, "synced_at": last, "linked_to_model": linked,
            "synthetic_model_db": is_synthetic(c)}


@app.get("/api/markets/event/{event_id}")
def market_event(event_id: str, history: bool = True):
    from ..ingest.polymarket import MONEYLINE, PROP_TYPES, SPREAD, TOTAL, PolymarketUSClient, mid_from_history_point
    ev = mq("""SELECT e.*, ta.name away, th.name home, ta.display_abbr away_abbr, th.display_abbr home_abbr FROM pm_events e
               LEFT JOIN pm_teams ta ON ta.pm_team_id=e.away_pm_team_id LEFT JOIN pm_teams th ON th.pm_team_id=e.home_pm_team_id
               WHERE e.event_id=?""", (event_id,))
    if ev.empty:
        raise HTTPException(404, "event not found")
    e = ev.iloc[0].to_dict()
    mk = mq("SELECT * FROM pm_markets WHERE event_id=?", (event_id,))
    qt = mq("""SELECT q.* FROM pm_quotes q JOIN pm_markets m USING(market_slug) WHERE m.event_id=?""", (event_id,))
    imp = mq("SELECT * FROM pm_implied WHERE event_id=? ORDER BY ts", (event_id,))
    out = {"event": clean(e), "implied": records(imp), "note": MARKET_NOTE}
    if not qt.empty:
        qt = qt.sort_values("ts").groupby(["market_slug", "kind"]).tail(1)
    for kind in ("live", "close"):
        q = qt[qt.kind == kind] if not qt.empty else qt
        m = mk.merge(q, on="market_slug", how="inner") if not q.empty else mk.iloc[0:0]
        home = e["home_pm_team_id"]
        sp = m[m.sports_market_type == SPREAD].copy()
        if len(sp):   # orient every rung to P(home margin > -L_home)
            sign = np.where(sp.long_pm_team_id == home, 1.0, -1.0)
            sp["p_home_cover"] = np.where(sign > 0, sp["mid"], 1 - sp["mid"])
            sp["line_home"] = sp.line * sign
        tt = m[m.sports_market_type == TOTAL]
        pr = m[m.sports_market_type.isin(PROP_TYPES)].assign(stat=lambda d: d.sports_market_type.map(PROP_TYPES))
        out[kind] = {"moneyline": records(m[m.sports_market_type == MONEYLINE]),
                     "spread_ladder": records(sp.sort_values("line_home")[["market_slug", "line_home", "p_home_cover", "bid", "ask", "mid"]]) if len(sp) else [],
                     "total_ladder": records(tt.sort_values("line")[["market_slug", "line", "mid", "bid", "ask"]]) if len(tt) else [],
                     "props": records(pr[["market_slug", "stat", "line", "long_pm_team_id", "mid", "bid", "ask"]]) if len(pr) else []}
    out["history"] = []
    ml = mk[mk.sports_market_type == MONEYLINE]
    if history and len(ml):
        try:
            client = PolymarketUSClient()
            h = client.history_fixed(ml.iloc[0].market_slug, "INTERVAL_1W")
            long_is_home = ml.iloc[0].long_pm_team_id == e["home_pm_team_id"]
            for pt in h:
                mm = mid_from_history_point(pt)
                if mm is not None:
                    out["history"].append({"t": pt["timestamp"], "p_home": mm if long_is_home else 1 - mm})
        except Exception as ex:   # network optional for drill-down
            out["history_error"] = str(ex)
    return clean(out)


@app.get("/api/markets/calibration")
def market_calibration():
    """How well calibrated were Polymarket US closing moneylines? Outcome = settlement (authoritative for the market)."""
    from ..models.backtest import _calibration_curve, boot_ci, log_loss
    d = mq("""SELECT i.event_id, i.p_home_win, i.margin_home_mean, i.total_mean, i.quality, e.title, e.start_time,
              e.home_pm_team_id, e.away_pm_team_id, e.score_conflict, m.settlement, m.long_pm_team_id
              FROM pm_implied i JOIN pm_events e USING(event_id)
              JOIN pm_markets m ON m.event_id=i.event_id AND m.sports_market_type='football_team_full_game_winner'
              WHERE i.kind='close' AND m.settlement IN (0,1) AND i.p_home_win IS NOT NULL""")
    if d.empty:
        return {"n": 0, "note": "No settled closing prices synced yet. Run: python -m cfbquant.cli markets close"}
    d["home_win"] = np.where(d.long_pm_team_id == d.home_pm_team_id, d.settlement, 1 - d.settlement)
    y, p = d.home_win.values, d.p_home_win.values
    fav = np.where(p >= .5, p, 1 - p)
    fav_won = np.where(p >= .5, y, 1 - y)
    ll, ll_lo, ll_hi = boot_ci(log_loss(p, y))
    br, br_lo, br_hi = boot_ci((p - y) ** 2)
    fw, fw_lo, fw_hi = boot_ci(fav_won - fav)
    return clean({"n": int(len(d)), "log_loss": [ll, ll_lo, ll_hi], "brier": [br, br_lo, br_hi],
            "favorite_win_rate_minus_price": [fw, fw_lo, fw_hi], "accuracy": float(((p > .5) == (y == 1)).mean()),
            "calibration": _calibration_curve(p, y, bins=10), "score_conflicts": int(d.score_conflict.fillna(0).sum()),
            "first_game": d.start_time.min(), "last_game": d.start_time.max(),
            "note": "Outcomes from Polymarket settlement. Games where Polymarket's own score feed disagrees with its settlement are counted in score_conflicts."})


# ------------------------------------------------------------------ props / same-game parlays
from pydantic import BaseModel  # noqa: E402


class PriceRequest(BaseModel):
    game_id: int
    legs: list[dict]
    book_american: float | None = None
    sims: int | None = None


@app.get("/api/sgp/slate")
def sgp_slate(season: int | None = None, week: int | None = None):
    """Upcoming games with their DraftKings line (anchor) and opening number, for the SGP Lab slate."""
    season, _ = latest(season)
    if week is None:
        r = conn().execute("SELECT MIN(week) FROM games WHERE season=? AND completed=0", (season,)).fetchone()
        week = r[0]
    g = q("""SELECT g.game_id, g.week, g.start_date, g.home_id, g.away_id, th.school AS home, ta.school AS away,
               b.spread_home, b.over_under, b.fetched_at AS line_fetched_at,
               (SELECT open_spread_home FROM line_snapshots s WHERE s.game_id=g.game_id ORDER BY fetched_at LIMIT 1) AS open_spread_home,
               (SELECT open_total FROM line_snapshots s WHERE s.game_id=g.game_id ORDER BY fetched_at LIMIT 1) AS open_total
        FROM games g JOIN teams th ON th.team_id=g.home_id JOIN teams ta ON ta.team_id=g.away_id
        LEFT JOIN betting_lines b ON b.game_id=g.game_id AND b.provider='DraftKings'
        WHERE g.season=? AND g.week=? AND g.completed=0 ORDER BY (b.spread_home IS NULL), g.start_date""", (season, week))
    weeks = [r[0] for r in conn().execute("SELECT DISTINCT week FROM games WHERE season=? AND completed=0 ORDER BY 1", (season,))]
    return {"season": season, "week": week, "weeks": weeks, "games": records(g)}


@app.get("/api/sgp/card/{game_id}")
def sgp_card(game_id: int, sims: int = 6000):
    from ..models.sgp_service import game_card
    sims = int(min(max(sims, 1000), 20000))
    try:
        with _lock:
            return clean(game_card(conn(), game_id, n_sims=sims))
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/api/sgp/price")
def sgp_price(req: PriceRequest):
    from ..models.sgp_service import DEFAULT_SIMS, price_legs
    if not req.legs:
        raise HTTPException(400, "no legs")
    try:
        with _lock:
            return clean(price_legs(conn(), req.game_id, req.legs, req.book_american, n_sims=req.sims or DEFAULT_SIMS))
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/sgp/validation")
def sgp_validation():
    v = q("SELECT scope, metric, value, ci_low, ci_high, n, detail_json FROM model_validation WHERE model_version='sgp_v1' ORDER BY scope, metric")
    return {"rows": records(v)}


@app.get("/api/sgp/ledger")
def sgp_ledger(limit: int = 200):
    t = q("""SELECT t.*, th.school AS home, ta.school AS away FROM sgp_tickets t JOIN games g ON g.game_id=t.game_id
             JOIN teams th ON th.team_id=g.home_id JOIN teams ta ON ta.team_id=g.away_id ORDER BY ticket_id DESC LIMIT ?""", (limit,))
    summ = q("""SELECT n_legs, COUNT(*) n, SUM(status='won') won, SUM(status='lost') lost, SUM(status='void') void,
                SUM(status='open') open, AVG(CASE WHEN status IN ('won','lost') THEN p_joint END) pred,
                AVG(CASE WHEN status IN ('won','lost') THEN result END) actual FROM sgp_tickets GROUP BY n_legs""")
    return {"tickets": records(t), "summary": records(summ)}


# ------------------------------------------------------------------ static UI
WEB = ROOT / "web"
if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


@app.get("/")
def index():
    return FileResponse(str(WEB / "index.html"))
