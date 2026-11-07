"""Load play-level data with derived analytic columns (shared by team & player metrics)."""
from __future__ import annotations

import numpy as np
import pandas as pd

OFF_TYPES = ("pass", "rush", "sack", "scramble")


def load_plays(conn, season: int, through_week: int | None = None, *, before_week: int | None = None,
               include_prior_seasons: int = 0) -> pd.DataFrame:
    """Plays for `season` up to and including `through_week` (or strictly before `before_week`)."""
    clauses = [f"p.season BETWEEN {int(season) - include_prior_seasons} AND {int(season)}"]
    if through_week is not None:
        clauses.append(f"(p.season < {int(season)} OR p.week <= {int(through_week)})")
    if before_week is not None:
        clauses.append(f"(p.season < {int(season)} OR p.week < {int(before_week)})")
    q = f"""SELECT p.*, g.neutral_site, g.temperature_f, g.wind_mph, g.home_points, g.away_points, g.away_id
            FROM plays p JOIN games g ON g.game_id = p.game_id WHERE {' AND '.join(clauses)}"""
    p = pd.read_sql_query(q, conn)
    return derive(p)


def derive(p: pd.DataFrame) -> pd.DataFrame:
    if p.empty:
        return p
    t = p.play_type
    p["is_off"] = t.isin(OFF_TYPES)
    p["dropback"] = t.isin(("pass", "sack", "scramble"))
    p["att"] = t.eq("pass")
    p["rush"] = t.eq("rush")
    p["carry"] = t.isin(("rush", "scramble"))
    p["margin"] = (p.offense_score - p.defense_score)
    p["off_home"] = (p.offense_id == p.home_id) & (p.neutral_site == 0)
    p["off_away"] = (p.offense_id != p.home_id) & (p.neutral_site == 0)
    y = p.yards_gained.fillna(0)
    p["explosive"] = ((p.carry & (y >= 12)) | (p.att & (p.completion == 1) & (y >= 16))).astype(float)
    p["negative"] = (p.is_off & (y < 0)).astype(float)
    p["stuff"] = (p.carry & (y <= 0)).astype(float)
    p["turnover"] = ((p.interception == 1) | (p.fumble_lost == 1)).astype(float)
    p["yac"] = np.where(p.att & (p.completion == 1) & p.air_yards.notna(), y - p.air_yards, np.nan)
    p["deep"] = np.where(p.att & p.air_yards.notna(), (p.air_yards >= 20).astype(float), np.nan)
    p["clean"] = p.garbage_time.fillna(0) == 0
    return p
