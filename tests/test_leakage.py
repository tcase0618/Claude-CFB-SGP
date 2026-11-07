"""Temporal validity: forecasts for week w must not depend on anything from week >= w."""
import numpy as np
import pandas as pd

from cfbquant.analytics.ratings import run_elo
from cfbquant.analytics.team_params import _estimate


def _params_vec(e):
    return np.array([[getattr(e.params[t], k) for k in ("comp_off", "comp_def", "rcore_off", "rcore_def", "sack_off", "pace")]
                     for t in sorted(e.params)])


def test_estimates_ignore_future_plays(db_copy):
    base = _params_vec(_estimate(db_copy, 2025, 5, None))
    # corrupt everything from week 6 onward (plays, results, box scores)
    db_copy.execute("UPDATE plays SET yards_gained = yards_gained + 30, epa = 5, success = 1 WHERE season=2025 AND week>=5")
    db_copy.execute("UPDATE games SET home_points = home_points + 50 WHERE season=2025 AND week>=5")
    db_copy.execute("UPDATE player_games SET targets = targets + 20 WHERE game_id IN (SELECT game_id FROM games WHERE season=2025 AND week>=5)")
    db_copy.commit()
    after = _params_vec(_estimate(db_copy, 2025, 5, None))
    assert np.allclose(base, after)
    # sanity: the corruption *does* change estimates that are allowed to see it
    later = _params_vec(_estimate(db_copy, 2025, 7, None))
    assert not np.allclose(base, later)


def test_elo_pregame_independent_of_own_result(db_copy):
    e1 = run_elo(db_copy).set_index("game_id")
    gid = db_copy.execute("SELECT game_id FROM games WHERE season=2025 AND week=5 LIMIT 1").fetchone()[0]
    db_copy.execute("UPDATE games SET home_points=home_points+60 WHERE game_id=?", (gid,))
    db_copy.commit()
    e2 = run_elo(db_copy).set_index("game_id")
    assert e1.loc[gid, "elo_p_home"] == e2.loc[gid, "elo_p_home"]


def test_backtest_features_asof(db_copy):
    from cfbquant.models.backtest import _asof_features
    f1 = _asof_features(db_copy, 2025, 4)
    db_copy.execute("UPDATE plays SET epa = 9 WHERE season=2025 AND week>=4")
    db_copy.commit()
    f2 = _asof_features(db_copy, 2025, 4)
    assert f1["epa"][0] == f2["epa"][0]
