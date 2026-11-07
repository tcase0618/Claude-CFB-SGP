"""Estimate simulator parameters for every team *as of* a forecast date.

Temporal validity: only plays with (season < s) or (season == s and week < w)
are used — never the game being predicted or anything after it.

Per stat, a weighted ridge model on game-level aggregates
    rate_g = mu + off_i + def_j + h * home
is solved with a non-zero prior mean (empirical-Bayes talent prior):
    (X'WX + lam I) b = X'Wy + lam b0,   b0 = slope * talent_z
where `slope` is the cross-sectional relationship between first-pass
estimates and recruiting talent. Prior-season games enter with reduced weight
scaled by returning production; current-season games decay by 0.95/week.
Estimates enter the engine as offsets from the observed league mean applied to
the engine's calibrated league baseline (logit scale for probabilities).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..sim.params import League, TeamParams, logit, sigmoid, OFF_SLOTS, DEF_SLOTS, SLOT
from .plays_frame import load_plays

BASE = TeamParams()
LG = League()

TARGET_POS = {"RB1": "RB", "RB2": "RB", "WR1": "WR", "WR2": "WR", "WR3": "WR", "WR4": "WR", "TE1": "TE", "TE2": "TE"}
SLOT_POS = {"QB1": ["QB"], "QB2": ["QB"], "RB1": ["RB"], "RB2": ["RB"], "WR1": ["WR"], "WR2": ["WR"], "WR3": ["WR"],
            "WR4": ["WR"], "TE1": ["TE"], "TE2": ["TE"], "K": ["K"], "P": ["P"], "DL1": ["EDGE", "DL"],
            "DL2": ["DL", "EDGE"], "DL3": ["DL", "EDGE"], "DL4": ["EDGE", "DL"], "LB1": ["LB"], "LB2": ["LB"],
            "LB3": ["LB"], "CB1": ["CB"], "CB2": ["CB"], "S1": ["S"], "S2": ["S"], "NB": ["CB", "S"]}


def ridge_prior(g: pd.DataFrame, lam: float, b0_off: dict | None = None, b0_def: dict | None = None):
    teams = sorted(set(g["off"]).union(g["def"]))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if len(g) == 0:
        return {}, {}, np.nan
    X = np.zeros((len(g), 2 * n + 1))
    r = np.arange(len(g))
    X[r, g["off"].map(ix).values] = 1
    X[r, n + g["def"].map(ix).values] = 1
    X[:, -1] = g["home"].values
    w = g["w"].values.astype(float)
    mu = np.average(g["value"].values, weights=w)
    y = g["value"].values - mu
    pen = np.full(2 * n + 1, lam)
    pen[-1] = lam * 0.01
    b0 = np.zeros(2 * n + 1)
    for t, v in (b0_off or {}).items():
        if t in ix:
            b0[ix[t]] = v
    for t, v in (b0_def or {}).items():
        if t in ix:
            b0[n + ix[t]] = v
    A = X.T @ (X * w[:, None]) + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ (y * w) + pen * b0)
    return {t: beta[ix[t]] for t in teams}, {t: beta[n + ix[t]] for t in teams}, mu


@dataclass
class Estimates:
    season: int
    before_week: int
    params: dict          # team_id -> TeamParams (full-health, unit level)
    rates: pd.DataFrame   # team-level estimated rates (for audit/UI)
    slots: dict           # team_id -> {slot: player_id}
    player_info: pd.DataFrame
    qb_value: dict        # player_id -> shrunk adj EPA/db
    league_obs: dict
    n_plays: int


STATS = {
    # name: (family mask, value column, binary?, sides)
    "sack": ("dropback", "sack", True),
    "int": ("att", "interception", True),
    "comp": ("att", "comp_n", True),
    "scramble": ("dropback", "scramble", True),
    "rexp": ("carry_d", "r10", True),
    "rcore": ("carry_core", "yards_gained", False),
    "yac": ("cmp_air", "yac", False),
    "fumble": ("touch", "fumble_lost", True),
    "passrate": ("neutral", "dropback", True),
    "adot": ("att_air", "air_yards", False),
}
LAM = {"sack": 6, "int": 10, "comp": 5, "scramble": 8, "rexp": 6, "rcore": 5, "yac": 6, "fumble": 12,
       "passrate": 3, "adot": 4}


def _prepare(p: pd.DataFrame) -> pd.DataFrame:
    p = p[p.is_off & p.clean].copy()
    p["r10"] = (p.yards_gained >= 10).astype(float)
    p["carry_d"] = p.carry
    p["carry_core"] = p.carry & (p.yards_gained < 10)
    p["touch"] = p.carry | (p.att & (p.completion == 1))
    p["cmp_air"] = p.att & (p.completion == 1) & p.air_yards.notna()
    p["att_air"] = p.att & p.air_yards.notna()
    p["neutral"] = (p.period <= 3) & (p.margin.abs() <= 14) & (p.down <= 2)
    # depth-neutral completion: completion - P(complete | air) + league completion
    p["comp_n"] = p.completion.astype(float)
    a = p[p.att & p.air_yards.notna()]
    if len(a) > 500:
        from sklearn.linear_model import LogisticRegression
        X = np.column_stack([a.air_yards.clip(-5, 60), a.air_yards.clip(-5, 60) ** 2 / 100]).astype(float)
        lr = LogisticRegression(max_iter=300).fit(X, a.completion.astype(int))
        xc = lr.predict_proba(X)[:, 1]
        p.loc[a.index, "comp_n"] = a.completion - xc + a.completion.mean()
    return p


def estimate(conn, season: int, before_week: int, injuries_week: int | None = None) -> Estimates:
    return _estimate_cached(str(conn_path(conn)), int(season), int(before_week),
                            None if injuries_week is None else int(injuries_week))


def conn_path(conn):
    return conn.execute("PRAGMA database_list").fetchone()[2]


_CONN_CACHE = {}


@lru_cache(maxsize=64)
def _estimate_cached(db: str, season: int, before_week: int, injuries_week):
    from ..db import connect
    conn = _CONN_CACHE.setdefault(db, connect(db))
    return _estimate(conn, season, before_week, injuries_week)


def clear_cache():
    _estimate_cached.cache_clear()


def _estimate(conn, season: int, before_week: int, injuries_week) -> Estimates:
    season, before_week = int(season), int(before_week)
    injuries_week = None if injuries_week is None else int(injuries_week)
    raw = load_plays(conn, season, before_week=before_week, include_prior_seasons=1)
    teams = pd.read_sql_query("SELECT team_id, talent_composite, returning_production FROM team_seasons WHERE season=?",
                              conn, params=(season,)).set_index("team_id")
    tz = (teams.talent_composite - teams.talent_composite.mean()) / (teams.talent_composite.std() or 1)
    tz = tz.fillna(0)
    rp = teams.returning_production.fillna(0.6)
    p = _prepare(raw) if not raw.empty else raw
    league_obs = {}
    rates = pd.DataFrame(index=teams.index)
    if not p.empty:
        cur = p.season == season
        weeks_ago = np.where(cur, (before_week - p.week).clip(lower=1), np.nan)
        # prior-season games: weight by offence team's returning production (roster turnover)
        prior_w = 0.45 * (0.4 + 0.6 * p.offense_id.map(rp).fillna(0.6))
        p["w"] = np.where(cur, 0.95 ** (weeks_ago - 1), prior_w * 0.95 ** 6)
        for st, (fam, col, binary) in STATS.items():
            sub = p[p[fam].astype(bool)]
            sub = sub[sub[col].notna()]
            if sub.empty:
                continue
            g = sub.assign(v=sub[col].astype(float) * sub.w, home=np.where(sub.neutral_site == 1, 0, np.where(sub.off_home, 1, -1)))
            gl = g.groupby(["game_id", "offense_id", "defense_id", "home"]).agg(s=("v", "sum"), w=("w", "sum"), n=("v", "size")).reset_index()
            gl["value"] = gl.s / gl.w
            gl["w"] = gl.n * (gl.w / gl.n)
            gl = gl.rename(columns={"offense_id": "off", "defense_id": "def"})
            lam = LAM[st] * (np.average(gl.n) if len(gl) else 1)
            o1, d1, mu = ridge_prior(gl, lam)
            # empirical-Bayes talent prior (slope from first pass)
            so = _slope(o1, tz)
            sd = _slope(d1, tz) if st not in ("scramble", "passrate", "adot") else 0.0
            o2, d2, mu = ridge_prior(gl, lam, {t: so * tz.get(t, 0) for t in o1}, {t: sd * tz.get(t, 0) for t in d1})
            league_obs[st] = mu
            rates[f"{st}_off"] = pd.Series({t: mu + v for t, v in o2.items()})
            rates[f"{st}_def"] = pd.Series({t: mu + v for t, v in d2.items()})
        # pace: offensive plays per game (current + prior weighted), shrunk
        ppg = p.groupby(["offense_id", "game_id"]).agg(n=("play_id", "size"), w=("w", "first")).reset_index()
        tp = ppg.groupby("offense_id").apply(lambda d: np.average(d.n, weights=d.w), include_groups=False)
        nw = ppg.groupby("offense_id").w.sum()
        lg_ppg = np.average(ppg.n, weights=ppg.w)
        shr = nw / (nw + 3)
        rates["plays_pg"] = lg_ppg + shr * (tp - lg_ppg)
        league_obs["plays_pg"] = lg_ppg
        # direct pace: game-clock seconds elapsed after a designed run (clock keeps running)
        q = raw[raw.period <= 3].sort_values(["game_id", "play_number"])
        if q.clock_seconds.notna().mean() > 0.5:
            nxt = q.groupby(["game_id", "period"])
            dt = q.clock_seconds - nxt.clock_seconds.shift(-1)
            same = nxt.offense_id.shift(-1) == q.offense_id
            ok = same & q.rush & (q.touchdown == 0) & (q.fumble_lost == 0) & dt.between(8, 60) & (q.margin.abs() <= 14)
            d = pd.DataFrame({"t": q.offense_id[ok], "dt": dt[ok]})
            if len(d) > 200:
                lgdt = d.dt.mean()
                g2 = d.groupby("t").dt.agg(["sum", "size"])
                rates["pace_dt"] = (g2["sum"] + 30 * lgdt) / (g2["size"] + 30)
                league_obs["pace_dt"] = lgdt
    full = load_plays(conn, season, before_week=before_week, include_prior_seasons=1) if raw is None else raw
    st_rates = _special_teams(full, season)
    for c, s in st_rates.items():
        rates[c] = s
    params = {}
    for t in teams.index:
        params[t] = _to_params(rates.loc[t] if t in rates.index else pd.Series(dtype=float), league_obs)
    slots, info, qbv, usage = _usage(conn, season, before_week, injuries_week, raw)
    for t, tp_ in list(params.items()):
        if t in usage:
            params[t] = tp_.with_(**usage[t])
    level = calibrate_level(conn, params, season, before_week)
    if level["shift"] != 0.0:
        params = {t: shift_level(tp_, level["shift"]) for t, tp_ in params.items()}
    league_obs["level"] = level
    return Estimates(season, before_week, params, rates, slots, info, qbv, league_obs, int(len(p)))


def shift_level(tp: TeamParams, l: float) -> TeamParams:
    """League-wide offensive efficiency shift (applied to every offence, so relative strengths are unchanged)."""
    return tp.with_(comp_off=float(sigmoid(logit(tp.comp_off) + 0.35 * l)), rcore_off=float(tp.rcore_off + 0.5 * l),
                    rexp_off=float(sigmoid(logit(tp.rexp_off) + 0.25 * l)), yac_off=float(tp.yac_off + 0.5 * l),
                    sack_off=float(sigmoid(logit(tp.sack_off) - 0.2 * l)), int_off=float(sigmoid(logit(tp.int_off) - 0.2 * l)))


def calibrate_level(conn, params: dict, season: int, before_week: int, n_games: int = 80, sims: int = 60) -> dict:
    """Choose a league-level shift so simulated scoring on recent real matchups matches observed scoring.

    Uses only games before the forecast week (current season weighted 1, prior season 0.35). Points are close
    to linear in the shift, so two simulation passes + linear interpolation suffice."""
    from ..sim.engine import simulate
    from ..sim.params import stack_params
    g = pd.read_sql_query("""SELECT season, week, home_id, away_id, home_points, away_points, neutral_site FROM games
        WHERE completed=1 AND ((season=? AND week<?) OR season=?) ORDER BY season DESC, week DESC LIMIT ?""",
                          conn, params=(season, before_week, season - 1, n_games))
    g = g[g.home_id.isin(params.keys()) & g.away_id.isin(params.keys())]
    if len(g) < 20:
        return {"shift": 0.0, "target": None, "n": int(len(g))}
    w = np.where(g.season == season, 1.0, 0.35)
    target = float(np.average((g.home_points + g.away_points) / 2, weights=w))
    neu = np.repeat(g.neutral_site.values.astype(bool), sims)

    def run(l):
        pairs = [(shift_level(params[h], l), shift_level(params[a], l)) for h, a in zip(g.home_id, g.away_id)]
        P = {k: np.repeat(v, sims, axis=0) for k, v in stack_params(pairs).items()}
        r = simulate(P, LG, 4242 + season + before_week, neutral=neu, track_players=False)
        per_game = ((r.home_pts + r.away_pts) / 2).reshape(len(g), sims).mean(axis=1)
        return float(np.average(per_game, weights=w))
    p0, p1 = run(0.0), run(0.6)
    slope = (p1 - p0) / 0.6
    shift = float(np.clip((target - p0) / slope, -1.5, 1.5)) if slope > 0.5 else 0.0
    return {"shift": shift, "target": target, "sim_before": p0, "n": int(len(g))}


def _slope(est: dict, tz: pd.Series) -> float:
    s = pd.Series(est)
    z = tz.reindex(s.index).fillna(0)
    if len(s) < 8 or z.std() == 0:
        return 0.0
    return float(np.cov(z, s)[0, 1] / np.var(z, ddof=1))


def _off(v, mu, base, binary=True, lo=None, hi=None):
    if v is None or pd.isna(v) or mu is None or pd.isna(mu):
        return base
    if binary:
        out = float(sigmoid(logit(base) + logit(v) - logit(mu)))
    else:
        out = float(base + v - mu)
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _to_params(r: pd.Series, lo: dict) -> TeamParams:
    g = lambda k: r.get(k) if k in r.index else None
    kw = dict(
        sack_off=_off(g("sack_off"), lo.get("sack"), BASE.sack_off), sack_def=_off(g("sack_def"), lo.get("sack"), BASE.sack_def),
        int_off=_off(g("int_off"), lo.get("int"), BASE.int_off), int_def=_off(g("int_def"), lo.get("int"), BASE.int_def),
        comp_off=_off(g("comp_off"), lo.get("comp"), BASE.comp_off), comp_def=_off(g("comp_def"), lo.get("comp"), BASE.comp_def),
        scramble=_off(g("scramble_off"), lo.get("scramble"), BASE.scramble),
        rexp_off=_off(g("rexp_off"), lo.get("rexp"), BASE.rexp_off), rexp_def=_off(g("rexp_def"), lo.get("rexp"), BASE.rexp_def),
        rcore_off=_off(g("rcore_off"), lo.get("rcore"), BASE.rcore_off, False, 1.0, 6.0),
        rcore_def=_off(g("rcore_def"), lo.get("rcore"), BASE.rcore_def, False, 1.0, 6.0),
        yac_off=_off(g("yac_off"), lo.get("yac"), BASE.yac_off, False, 2.0, 10.0),
        yac_def=_off(g("yac_def"), lo.get("yac"), BASE.yac_def, False, 2.0, 10.0),
        fumble_off=_off(g("fumble_off"), lo.get("fumble"), BASE.fumble_off),
        fumble_def=_off(g("fumble_def"), lo.get("fumble"), BASE.fumble_def),
        pass_rate=_off(g("passrate_off"), lo.get("passrate"), BASE.pass_rate),
        adot=_off(g("adot_off"), lo.get("adot"), BASE.adot, False, 4.0, 14.0),
    )
    pdt = g("pace_dt")
    ppg = g("plays_pg")
    if pdt is not None and not pd.isna(pdt) and lo.get("pace_dt"):
        kw["pace"] = float(np.clip(BASE.pace * pdt / lo["pace_dt"], 20, 42))
    elif ppg is not None and not pd.isna(ppg) and lo.get("plays_pg"):
        kw["pace"] = float(np.clip(BASE.pace * lo["plays_pg"] / ppg, 22, 40))
    for k in ("fg_skill", "punt_net", "go_aggr"):
        v = g(k)
        if v is not None and not pd.isna(v):
            kw[k] = float(v)
    return TeamParams(**kw)


def _special_teams(p: pd.DataFrame, season: int) -> dict:
    out = {}
    if p is None or p.empty:
        return out
    fg = p[(p.play_type == "field_goal") & p.fg_distance.notna()]
    if len(fg) > 30:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression().fit(fg[["fg_distance"]], fg.fg_made.astype(int))
        pe = lr.predict_proba(fg[["fg_distance"]])[:, 1]
        res = (fg.fg_made - pe).groupby(fg.offense_id).sum()
        info = pd.Series(pe * (1 - pe), index=fg.index).groupby(fg.offense_id).sum()
        out["fg_skill"] = (res / (info + 4)).clip(-1.5, 1.5)
    pu = p[p.play_type == "punt"]
    if len(pu):
        g = pu.groupby("offense_id").yards_gained.agg(["sum", "size"])
        lg = pu.yards_gained.mean()
        out["punt_net"] = (g["sum"] + 15 * lg) / (g["size"] + 15) - lg + BASE.punt_net
    f4 = p[(p.down == 4) & (p.distance <= 3) & p.yards_to_goal.between(30, 60) &
           p.play_type.isin(["pass", "rush", "sack", "scramble", "punt", "field_goal"])]
    if len(f4) > 20:
        go = f4.play_type.isin(["pass", "rush", "sack", "scramble"]).astype(float)
        lgr = go.mean()
        g = go.groupby(f4.offense_id).agg(["sum", "size"])
        sh = (g["sum"] + 8 * lgr) / (g["size"] + 8)
        out["go_aggr"] = (logit(sh) - logit(lgr)) / 1.5
    return out


# ------------------------------------------------------------------- usage & slot assignment
def _usage(conn, season, before_week, injuries_week, plays):
    ros = pd.read_sql_query("""SELECT r.player_id, r.team_id, r.position, r.class_year, p.first_name, p.last_name
        FROM rosters r JOIN players p ON p.player_id=r.player_id WHERE r.season=?""", conn, params=(season,))
    inj_week = before_week if injuries_week is None else injuries_week
    out_ids = set(pd.read_sql_query("SELECT player_id FROM injuries WHERE season=? AND week=? AND status='OUT'",
                                    conn, params=(season, inj_week)).player_id)
    dc = pd.read_sql_query("SELECT player_id, MIN(depth) AS depth FROM depth_charts WHERE season=? GROUP BY 1",
                           conn, params=(season,)).set_index("player_id").depth
    pg = pd.read_sql_query("""SELECT pg.*, g.season, g.week FROM player_games pg JOIN games g ON g.game_id=pg.game_id
        WHERE (g.season=? AND g.week<?) OR g.season=?""", conn, params=(season, before_week, season - 1))
    cur = pg[pg.season == season]
    recent_games = cur.sort_values("week").groupby("team_id").game_id.unique().apply(lambda a: list(a[-6:]))
    use_pg = pg.copy()
    use_pg["w"] = np.where(use_pg.season == season, 1.0, 0.35)
    if not cur.empty:
        last = set(g for lst in recent_games for g in lst)
        use_pg.loc[(use_pg.season == season) & ~use_pg.game_id.isin(last), "w"] = 0.5
    cols = ["pass_att", "sacks_taken", "rush_att", "targets", "receptions", "rec_yds", "tackles", "def_sacks", "def_int",
            "pbu", "fga", "xpa", "punts", "rush_yds"]
    wsum = use_pg[cols].multiply(use_pg.w, axis=0).groupby(use_pg.player_id).sum()
    ros = ros.join(wsum, on="player_id").fillna({c: 0 for c in cols})
    ros["depth"] = ros.player_id.map(dc).fillna(9)
    ros["out"] = ros.player_id.isin(out_ids)
    # QB value: shrunk EPA per dropback (prior: replacement level)
    qbv = {}
    if plays is not None and not plays.empty:
        ev = pd.read_sql_query("""SELECT e.play_id, e.player_id FROM play_events e JOIN plays p ON p.play_id=e.play_id
            WHERE e.role='passer' AND p.season BETWEEN ? AND ? AND (p.season<? OR p.week<?)""", conn,
                               params=(season - 1, season, season, before_week))
        db = plays[plays.dropback & plays.clean][["play_id", "epa", "season"]]
        q = ev.merge(db, on="play_id")
        q["w"] = np.where(q.season == season, 1.0, 0.5)
        agg = q.assign(we=q.epa * q.w).groupby("player_id").agg(s=("we", "sum"), n=("w", "sum"))
        lg = db.epa.mean()
        repl = lg - 0.12
        qbv = ((agg.s + 80 * repl) / (agg.n + 80)).to_dict()
    slots, usage = {}, {}
    for t, d in ros.groupby("team_id"):
        healthy = d[~d.out]
        used: set = set()
        sm = {}
        score_key = {"QB": "pass_att", "RB": "rush_att", "WR": "targets", "TE": "targets", "K": "xpa", "P": "punts",
                     "DL": "tackles", "EDGE": "tackles", "LB": "tackles", "CB": "tackles", "S": "tackles"}
        for s in OFF_SLOTS + DEF_SLOTS:
            pos = SLOT_POS[s]
            cand = healthy[healthy.position.isin(pos) & ~healthy.player_id.isin(used)]
            if cand.empty:
                cand = d[d.position.isin(pos) & ~d.player_id.isin(used)]
            if cand.empty:
                sm[s] = None
                continue
            k = score_key.get(cand.position.iloc[0], "tackles")
            cand = cand.assign(_s=cand[k] - 0.01 * cand.depth, _pp=cand.position.map({p_: i for i, p_ in enumerate(pos)}))
            best = cand.sort_values(["_pp", "_s"], ascending=[True, False]) if s in ("DL1", "DL2", "DL3", "DL4") else \
                cand.sort_values("_s", ascending=False)
            pid = int(best.player_id.iloc[0])
            sm[s] = pid
            used.add(pid)
        slots[t] = sm
        dd = d.set_index("player_id")
        g = lambda s, c: float(dd.loc[sm[s], c]) if sm.get(s) is not None and sm[s] in dd.index else 0.0
        tslots = ["RB1", "RB2", "WR1", "WR2", "WR3", "WR4", "TE1", "TE2"]
        tg = np.array([g(s, "targets") for s in tslots])
        base_t = np.array(BASE.target_share)
        tot = tg.sum()
        tshare = (tg + 25 * base_t * max(tot, 1) / 100) if tot > 0 else base_t.copy()
        rslots = ["QB1", "RB1", "RB2", "WR1"]
        ra = np.array([g(s, "rush_att") for s in rslots])
        base_r = np.array(BASE.rush_share)
        rshare = (ra + 20 * base_r * max(ra.sum(), 1) / 100) if ra.sum() > 0 else base_r.copy()
        rec = np.array([g(s, "receptions") for s in tslots])
        catch = np.where(tg > 0, (rec + 6 * 0.64) / (tg + 6), 0.64)
        team_c = rec.sum() / tg.sum() if tg.sum() > 0 else 0.64
        catch_adj = tuple(np.clip(logit(catch) - logit(team_c), -0.6, 0.6) * 0.6)
        ypc = np.array([(g(s, "rush_yds") + 30 * 4.5) / (g(s, "rush_att") + 30) for s in rslots])
        rush_adj = (0.0, 0.0, float(np.clip(0.5 * (ypc[2] - ypc[1]), -1, 1)), float(np.clip(0.5 * (ypc[3] - ypc[1]), -1, 1)))
        dsl = DEF_SLOTS
        tk = np.array([g(s, "tackles") for s in dsl])
        sk = np.array([g(s, "def_sacks") for s in dsl])
        it = np.array([g(s, "def_int") + 0.5 * g(s, "pbu") for s in dsl])

        def blend(obs, base, k):
            base = np.array(base)
            return tuple((obs + k * base) / (obs.sum() + k)) if obs.sum() > 0 else tuple(base)
        usage[t] = dict(target_share=tuple(tshare), rush_share=tuple(rshare), catch_adj=catch_adj, rush_adj=rush_adj,
                        tackle_run=blend(tk, BASE.tackle_run, 30), tackle_pass=blend(tk, BASE.tackle_pass, 30),
                        sack_share=blend(sk, BASE.sack_share, 4), int_share=blend(it, BASE.int_share, 4))
    info = ros.set_index("player_id")[["team_id", "position", "first_name", "last_name", "class_year", "out"]]
    return slots, info, qbv, usage


# ------------------------------------------------------------------- scenario helpers
EPA_TO_PASS = dict(comp=1.4, intr=-1.2, sack=-0.8, yac=2.5)


def apply_qb_change(tp: TeamParams, delta_epa_db: float) -> TeamParams:
    """Shift a unit's passing parameters by a change in QB value (EPA / dropback). Estimated mapping."""
    d = float(delta_epa_db)
    return tp.with_(comp_off=float(sigmoid(logit(tp.comp_off) + EPA_TO_PASS["comp"] * d)),
                    int_off=float(sigmoid(logit(tp.int_off) + EPA_TO_PASS["intr"] * d)),
                    sack_off=float(sigmoid(logit(tp.sack_off) + EPA_TO_PASS["sack"] * d)),
                    yac_off=float(max(2.0, tp.yac_off + EPA_TO_PASS["yac"] * d)))
