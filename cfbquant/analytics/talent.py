"""College-specific talent & development intelligence (Phase 5).

Recruiting ratings are treated as *uncertain priors on future production*, not
as current ability:

  prior mean  mu0 = a_pos + b_pos * recruit_rating + c_pos * class_year   (OLS on historical player-seasons)
  prior var   s0^2 = residual variance of that regression (wide)
  likelihood  observed adj EPA/opp with variance s_e^2 / n
  posterior   precision-weighted combination -> mean and 80% interval

A five-star freshman with no snaps therefore gets the prior with a wide
interval; an experienced starter's posterior is dominated by observed play.

Transfers: prior = previous-team posterior, shifted by the conference-strength
difference (destination - origin) * 0.5 and regressed 25% toward the
destination position mean (scheme / usage uncertainty).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKILL = ("QB", "RB", "WR", "TE")


def _player_seasons(conn) -> pd.DataFrame:
    q = """SELECT pm.player_id, pm.season, pm.metric_id, pm.value, pm.sample FROM player_metrics pm
           JOIN (SELECT player_id, season, MAX(through_week) w FROM player_metrics GROUP BY 1,2) m
             ON m.player_id=pm.player_id AND m.season=pm.season AND m.w=pm.through_week
           WHERE pm.metric_id IN ('adj_epa_play','epa_se','reliability','target_share','rush_share','dropback_share')"""
    pm = pd.read_sql_query(q, conn)
    if pm.empty:
        return pm
    w = pm.pivot_table(index=["player_id", "season"], columns="metric_id", values="value").reset_index()
    n = pm[pm.metric_id == "adj_epa_play"][["player_id", "season", "sample"]]
    w = w.merge(n, on=["player_id", "season"], how="left")
    ros = pd.read_sql_query("SELECT player_id, season, team_id, position, class_year FROM rosters", conn)
    rec = pd.read_sql_query("SELECT player_id, stars, rating FROM recruiting_profiles WHERE player_id IS NOT NULL", conn).drop_duplicates("player_id")
    return ros.merge(w, on=["player_id", "season"], how="left").merge(rec, on="player_id", how="left")


def posteriors(conn, season: int) -> pd.DataFrame:
    ps = _player_seasons(conn)
    if ps.empty:
        return ps
    out = []
    for pos in SKILL:
        d = ps[ps.position == pos].copy()
        hist = d[(d["sample"] >= 40) & d.rating.notna() & d.adj_epa_play.notna() & (d.season <= season)]
        if len(hist) >= 15:
            X = np.column_stack([np.ones(len(hist)), hist.rating, hist.class_year.fillna(2)])
            beta, *_ = np.linalg.lstsq(X, hist.adj_epa_play, rcond=None)
            resid = hist.adj_epa_play - X @ beta
            s_noise = float(np.nanmedian(hist.epa_se ** 2 * hist["sample"])) if hist.epa_se.notna().any() else 1.2
            s0 = max(float(resid.var()) - float(np.mean(s_noise / hist["sample"])), 0.002)
        else:
            beta, s0, s_noise = np.array([float(d.adj_epa_play.mean() or 0), 0, 0]), 0.02, 1.2
        cur = d[d.season == season].copy()
        cur["prior_mean"] = beta[0] + beta[1] * cur.rating.fillna(0.80) + beta[2] * cur.class_year.fillna(2)
        cur["prior_sd"] = np.sqrt(s0)
        n = cur["sample"].fillna(0)
        obs_var = s_noise / n.replace(0, np.nan)
        prec = 1 / s0 + np.where(n > 0, 1 / obs_var, 0)
        cur["post_mean"] = (cur.prior_mean / s0 + np.where(n > 0, cur.adj_epa_play.fillna(0) / obs_var, 0)) / prec
        cur["post_sd"] = np.sqrt(1 / prec)
        cur["post_p10"] = cur.post_mean - 1.2816 * cur.post_sd
        cur["post_p90"] = cur.post_mean + 1.2816 * cur.post_sd
        cur["usage"] = cur[["target_share", "rush_share", "dropback_share"]].max(axis=1)
        out.append(cur)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def breakout_candidates(post: pd.DataFrame, team_id: int | None = None, k: int = 15) -> pd.DataFrame:
    if post.empty:
        return post
    d = post[(post.class_year <= 3) & ((post.usage.fillna(0) < 0.18) | (post["sample"].fillna(0) < 80))].copy()
    if team_id is not None:
        d = d[d.team_id == team_id]
    pos_med = post.groupby("position").post_mean.median()
    d["upside"] = d.post_p90 - d.position.map(pos_med)
    return d.sort_values("upside", ascending=False).head(k)


def development_curves(conn) -> pd.DataFrame:
    ps = _player_seasons(conn)
    if ps.empty:
        return ps
    q = ps[(ps["sample"] >= 30) & ps.position.isin(SKILL)]
    return q.groupby(["position", "class_year"]).agg(adj_epa=("adj_epa_play", "mean"), n=("player_id", "size")).reset_index()


def transfer_projections(conn, season: int) -> pd.DataFrame:
    tr = pd.read_sql_query("SELECT * FROM transfer_history WHERE season=? AND player_id IS NOT NULL", conn, params=(season,))
    if tr.empty:
        return tr
    prev = posteriors(conn, season - 1)
    if prev.empty:
        return tr
    conf = pd.read_sql_query("SELECT team_id, season, conference_id FROM team_seasons", conn)
    tm = pd.read_sql_query("""SELECT team_id, season, adj_value FROM team_metrics WHERE metric_id='net_epa'
                              UNION ALL SELECT team_id, season, value FROM team_metrics WHERE metric_id='net_epa' AND adj_value IS NULL""", conn)
    tm = tm.groupby(["team_id", "season"]).adj_value.mean().reset_index()
    cs = tm.merge(conf, on=["team_id", "season"]).groupby(["conference_id", "season"]).adj_value.mean()
    pm = prev.set_index("player_id")
    pos_mean = prev.groupby("position").post_mean.mean()
    rows = []
    for r in tr.itertuples():
        if r.player_id not in pm.index:
            continue
        p = pm.loc[r.player_id]
        c_from = conf[(conf.team_id == r.from_team_id) & (conf.season == season - 1)].conference_id
        c_to = conf[(conf.team_id == r.to_team_id) & (conf.season == season)].conference_id
        d_conf = 0.0
        if len(c_from) and len(c_to):
            d_conf = float(cs.get((c_from.iloc[0], season - 1), 0) - cs.get((c_to.iloc[0], season - 1), 0))
        mean = 0.75 * (p.post_mean + 0.5 * d_conf) + 0.25 * pos_mean.get(p.position, 0)
        sd = float(np.sqrt(p.post_sd ** 2 + 0.03 ** 2))
        rows.append(dict(player_id=r.player_id, name=r.name, position=p.position, from_team_id=r.from_team_id,
                         to_team_id=r.to_team_id, prev_adj_epa=p.adj_epa_play, prev_sample=p["sample"],
                         proj_mean=mean, proj_p10=mean - 1.2816 * sd, proj_p90=mean + 1.2816 * sd, conf_shift=d_conf))
    return pd.DataFrame(rows)
