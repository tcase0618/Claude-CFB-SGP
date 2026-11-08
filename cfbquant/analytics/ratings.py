"""Benchmark team ratings: margin-aware Elo with talent-informed preseason priors.

Elo is a deliberately simple baseline the simulation model must beat out of
sample. Ratings are computed chronologically, so `elo_before(game)` only uses
results strictly before kickoff.
  expected = 1 / (1 + 10^(-(Rh - Ra + HFA)/400))
  K_eff    = K * ln(|margin| + 1) * 2.2 / (0.001 * |elo diff winner| + 2.2)
Preseason: R = 1500 + 0.6 * (R_prev - 1500) + talent_coef * talent_z.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K, HFA, CARRY, TALENT = 28.0, 55.0, 0.6, 45.0
FCS_START = 1150.0   # FCS teams enter far below the FBS mean (they are only observed vs FBS opponents)


def run_elo(conn) -> pd.DataFrame:
    g = pd.read_sql_query("""SELECT game_id, season, week, start_date, home_id, away_id, home_points, away_points,
                             neutral_site, completed FROM games ORDER BY start_date, game_id""", conn)
    ts = pd.read_sql_query("SELECT team_id, season, talent_composite, division FROM team_seasons", conn)
    fcs = {(r.team_id, r.season) for r in ts.itertuples() if r.division == "fcs"}
    ts["tz"] = ts.groupby("season").talent_composite.transform(lambda s: (s - s.mean()) / (s.std() or 1)).fillna(0)
    tz = {(r.team_id, r.season): r.tz for r in ts.itertuples()}
    R: dict[int, float] = {}
    season_seen: dict[int, int] = {}
    rows = []
    for r in g.itertuples():
        for t in (r.home_id, r.away_id):
            if season_seen.get(t) != r.season:
                base = FCS_START if (t, r.season) in fcs else 1500.0
                prev = R.get(t, base)
                R[t] = base + CARRY * (prev - base) + TALENT * tz.get((t, r.season), 0.0)
                season_seen[t] = r.season
        h = 0.0 if r.neutral_site else HFA
        diff = R[r.home_id] - R[r.away_id] + h
        p = 1 / (1 + 10 ** (-diff / 400))
        rows.append((r.game_id, R[r.home_id], R[r.away_id], p, diff / 25.0))
        if r.completed and r.home_points is not None:
            m = r.home_points - r.away_points
            res = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
            wd = diff if m > 0 else -diff
            mult = np.log(abs(m) + 1) * 2.2 / (0.001 * wd + 2.2)
            delta = K * mult * (res - p)
            R[r.home_id] += delta
            R[r.away_id] -= delta
    return pd.DataFrame(rows, columns=["game_id", "elo_home", "elo_away", "elo_p_home", "elo_margin"])


def current_elo(conn, season: int, before_week: int) -> dict:
    e = run_elo(conn)
    g = pd.read_sql_query("SELECT game_id, season, week, home_id, away_id FROM games", conn).merge(e, on="game_id")
    g = g[(g.season < season) | ((g.season == season) & (g.week < before_week))]
    out = {}
    for r in g.itertuples():
        out[r.home_id] = r.elo_home
        out[r.away_id] = r.elo_away
    return out


def write_ratings(conn, season: int) -> None:
    e = run_elo(conn)
    g = pd.read_sql_query("SELECT game_id, season, week, home_id, away_id FROM games WHERE season=?", conn, params=(season,))
    g = g.merge(e, on="game_id")
    rows = []
    for r in g.itertuples():
        rows.append((r.home_id, season, int(r.week), "elo", float(r.elo_home), None, None))
        rows.append((r.away_id, season, int(r.week), "elo", float(r.elo_away), None, None))
    conn.executemany("INSERT OR REPLACE INTO team_ratings VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
