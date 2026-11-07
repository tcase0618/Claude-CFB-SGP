"""Polymarket US integration: parsing, verified contract semantics, ladder fits, mapping, conflicts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cfbquant.ingest import polymarket as pm

FX = json.loads((Path(__file__).parent / "fixtures" / "pm_event_unc_tcu.json").read_text())
UNC, TCU = 1082, 1145
# Settlements verified against the live API for this game (UNC 15, TCU 10 -> UNC margin +5).
SETTLED = {"aec-cfb-ncar-tcu-2026-08-29": 1, "asc-cfb-ncar-tcu-2026-08-29-neg-1pt5": 1,
           "asc-cfb-ncar-tcu-2026-08-29-neg-7pt5": 0, "asc-cfb-ncar-tcu-2026-08-29-neg-17pt5": 0,
           "asc-cfb-ncar-tcu-2026-08-29-pos-4pt5": 1, "asc-cfb-ncar-tcu-2026-08-29-pos-21pt5": 1,
           "tsc-cfb-ncar-tcu-2026-08-29-total-27pt5": 0, "tsc-cfb-ncar-tcu-2026-08-29-total-51pt5": 0}


def test_parse_event_real_payload():
    ev, teams, mkts = pm.parse_event(FX)
    assert ev["away_pm_team_id"] == UNC and ev["home_pm_team_id"] == TCU      # away-first ordering
    assert (ev["away_score"], ev["home_score"]) == (15, 10)
    types = {m["sports_market_type"] for m in mkts}
    assert {pm.MONEYLINE, pm.SPREAD, pm.TOTAL} <= types
    assert {t["pm_team_id"] for t in teams} == {UNC, TCU}


def yes_rule(m, margin_by_team, total):
    """The documented semantics, as an executable rule."""
    if m["sports_market_type"] == pm.MONEYLINE:
        return int(margin_by_team[m["long_pm_team_id"]] > 0)
    if m["sports_market_type"] == pm.SPREAD:
        return int(margin_by_team[m["long_pm_team_id"]] + m["line"] > 0)
    if m["sports_market_type"] == pm.TOTAL:
        return int(total > m["line"])


def test_semantics_match_settlements():
    _, _, mkts = pm.parse_event(FX)
    margin = {UNC: 5, TCU: -5}
    by = {m["market_slug"]: m for m in mkts}
    for slug, settled in SETTLED.items():
        assert yes_rule(by[slug], margin, 25) == settled, slug


def test_outcome_prices_are_bid_ask():
    assert pm.bidask_from_outcome_prices('["0.9850","0.9900"]') == (0.985, 0.99)
    assert pm.bidask_from_outcome_prices('["0.70","0.60"]') == (None, None)      # crossed -> rejected
    assert pm.bidask_from_outcome_prices(None) == (None, None)


def test_history_mid():
    assert pm.mid_from_history_point({"longPrice": 0.265, "shortPrice": 0.74}) == pytest.approx((0.265 + 0.26) / 2)
    assert pm.mid_from_history_point({"longPrice": 0.30, "shortPrice": 0.60}) is None   # bid 0.40 > ask 0.30: crossed


def test_ladder_fit_recovers_distribution():
    rng = np.random.default_rng(0)
    lines = np.arange(-20.5, 21, 1.0)
    mu, sd = 6.5, 15.0
    p = norm.cdf((mu + lines) / sd) + rng.normal(0, 0.005, len(lines))
    f = pm.fit_normal_ladder(lines, p, "spread")
    assert abs(f["mean"] - mu) < 0.5 and abs(f["sd"] - sd) < 1.0
    tl = np.arange(40.5, 70, 1.0)
    pt = 1 - norm.cdf((tl - 55.0) / 14.0)
    ft = pm.fit_normal_ladder(tl, pt, "total")
    assert abs(ft["mean"] - 55) < 0.3 and abs(ft["sd"] - 14) < 0.5


def test_implied_orients_away_long_rungs_to_home():
    ev = {"event_id": "x", "away_pm_team_id": 1, "home_pm_team_id": 2}
    mu_home, sd = 7.0, 14.0
    rows = [dict(sports_market_type=pm.MONEYLINE, line=None, long_pm_team_id=1, mid=1 - norm.cdf(mu_home / sd), bid=None, ask=None)]
    for L in np.arange(-15.5, 16, 2.0):   # rungs listed with the AWAY team long
        rows.append(dict(sports_market_type=pm.SPREAD, line=L, long_pm_team_id=1, mid=norm.cdf((-mu_home + L) / sd), bid=None, ask=None))
    imp = pm.implied_from_markets(ev, pd.DataFrame(rows))
    assert imp["p_home_win"] == pytest.approx(norm.cdf(mu_home / sd))
    assert abs(imp["margin_home_mean"] - mu_home) < 0.3 and abs(imp["margin_sd"] - sd) < 0.5


def test_team_name_normalisation():
    assert pm.norm_name("Miami (FL)") == pm.norm_name("Miami")
    assert pm.norm_name("NC State") == pm.norm_name("North Carolina State")
    assert pm.norm_name("Hawai'i") == pm.norm_name("Hawaii")


def test_mapping_uses_model_home_away(tmp_path):
    from cfbquant.db import open_db
    mc = pm.open_markets_db(tmp_path / "m.db")
    ev, teams, mkts = pm.parse_event(FX)
    pm._store(mc, ev, teams, [m for m in mkts if m["sports_market_type"] == pm.MONEYLINE])
    with open_db(tmp_path / "g.db") as gc:
        gc.execute("INSERT INTO data_sources VALUES ('cfbd','t',0,NULL,NULL)")
        gc.execute("INSERT INTO teams (team_id, school, source) VALUES (153,'North Carolina','cfbd'),(2628,'TCU','cfbd')")
        # model says UNC is HOME (opposite of Polymarket's listing order)
        gc.execute("""INSERT INTO games (game_id, season, week, start_date, home_id, away_id, source, completed)
                      VALUES (9,2026,1,'2026-08-29T16:00:00Z',153,2628,'cfbd',1)""")
        gc.commit()
        mp = pm.map_events_to_games(mc, gc)
        assert len(mp) == 1 and mp.iloc[0].game_id == 9 and not mp.iloc[0].pm_home_is_home
        pm.insert_df(mc, "pm_implied", pd.DataFrame([dict(event_id=ev["event_id"], kind="close", ts="t", p_away_win=0.7,
                                                          p_home_win=0.3, margin_home_mean=-6.0, total_mean=48.0)]))
        m = pm.market_for_games(mc, gc)
        assert m.iloc[0].mkt_p_home == pytest.approx(0.7) and m.iloc[0].mkt_margin_home == pytest.approx(6.0)


def test_market_never_attached_to_synthetic(conn):
    from cfbquant.models.backtest import attach_market
    out = attach_market(conn, pd.DataFrame({"game_id": [1, 2]}))
    assert out.mkt_p_home.isna().all()
