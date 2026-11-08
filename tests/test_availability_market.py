import shutil

import numpy as np
import pandas as pd
import pytest

from cfbquant.models.market_sgp import _ladder_prob, market_leg_prob, price_market
from cfbquant.models.sgp import GameSims, Leg


@pytest.fixture()
def wconn(small_db, tmp_path):
    """Writable copy of the synthetic DB (availability tests insert rows)."""
    from cfbquant.db import connect, migrate
    p = tmp_path / "w.db"
    shutil.copy(small_db, p)
    c = connect(p)
    migrate(c)
    yield c
    c.close()


def _starter(conn, season, week, slot="QB1"):
    from cfbquant.analytics.team_params import estimate
    est = estimate(conn, season, week)
    team = next(t for t, s in est.slots.items() if s.get(slot) is not None)
    return est, team, int(est.slots[team][slot])


def test_game_log_flags_missed_last(wconn):
    from cfbquant.ingest.availability import game_log_flags
    season, week = 2025, 8
    _, team, qb = _starter(wconn, season, week)
    last = wconn.execute("""SELECT MAX(g.week) FROM player_games pg JOIN games g ON g.game_id=pg.game_id
        WHERE g.season=? AND g.week<? AND pg.team_id=?""", (season, week, team)).fetchone()[0]
    wconn.execute("""DELETE FROM player_games WHERE player_id=? AND game_id IN
        (SELECT game_id FROM games WHERE season=? AND week=?)""", (qb, season, last))
    fl = game_log_flags(wconn, season, week, [team])
    mine = fl[fl.player_id == qb]
    assert "missed_last" in set(mine.flag)
    assert f"week {last}" in mine[mine.flag == "missed_last"].evidence.iloc[0]


def test_import_report_marks_out_and_replaces_starter(wconn, tmp_path):
    from cfbquant.analytics.team_params import clear_cache, estimate
    from cfbquant.ingest.availability import import_report
    season, week = 2025, 8
    est, team, qb = _starter(wconn, season, week)
    school = wconn.execute("SELECT school FROM teams WHERE team_id=?", (team,)).fetchone()[0]
    fn, ln = wconn.execute("SELECT first_name, last_name FROM players WHERE player_id=?", (qb,)).fetchone()
    csv = tmp_path / "report.csv"
    pd.DataFrame([{"Team": school, "Player": f"{fn} {ln} Jr.", "Status": "out"},
                  {"Team": school, "Player": "Nobody Atall", "Status": "Q"}]).to_csv(csv, index=False)
    r = import_report(wconn, season, week, str(csv), log=lambda *a: None)
    assert r["imported"] == 1 and len(r["unmatched"]) == 1          # unmatched names are reported, never guessed
    st = wconn.execute("SELECT status FROM availability WHERE player_id=? AND season=? AND week=?",
                       (qb, season, week)).fetchone()[0]
    assert st == "OUT"
    clear_cache()
    est2 = estimate(wconn, season, week)
    s2 = est2.slots[team]
    assert qb not in s2.values()                                      # OUT never fills any slot, not even as backup
    assert s2.get("QB1") is not None


# ----------------------------------------------------------------------------- Polymarket pricing
def _book():
    """A market book oriented to the model home team: home -3.5 at 50%, total 50.5 at 50%, home TT 26.5."""
    sp = pd.DataFrame({"L_home": [-6.5, -3.5, -0.5, 2.5], "p": [0.38, 0.50, 0.61, 0.70], "width": [0.02] * 4})
    tt = pd.DataFrame({"line": [44.5, 47.5, 50.5, 53.5, 56.5], "p": [0.75, 0.63, 0.50, 0.37, 0.26], "width": [0.02] * 5})
    th = pd.DataFrame({"line": [25.5, 27.5], "p": [0.56, 0.46], "width": [0.03, 0.03]})
    return {"ml": (0.62, 0.01), "spread": sp, "total": tt, "team_total": {"home": th, "away": th.iloc[0:0]}, "ts": "t"}


def test_market_leg_orientation_and_ladder():
    b = _book()
    assert market_leg_prob(Leg("ml", side="away"), b)[0] == pytest.approx(0.38)
    assert market_leg_prob(Leg("spread", side="home", line=-3.5), b) == (pytest.approx(0.50), "exact")
    # away +3.5 covers  <=>  home fails to cover -3.5
    assert market_leg_prob(Leg("spread", side="away", line=3.5), b)[0] == pytest.approx(0.50)
    assert market_leg_prob(Leg("spread", side="away", line=6.5), b)[0] == pytest.approx(0.62)
    assert market_leg_prob(Leg("total", op="under", line=47.5), b)[0] == pytest.approx(0.37)
    p, src = market_leg_prob(Leg("total", op="over", line=49.0), b)        # between rungs -> normal fit
    assert src == "ladder_fit" and 0.50 < p < 0.63
    assert _ladder_prob(b["total"], 80.5, "total") == (None, None)          # no wild extrapolation
    assert market_leg_prob(Leg("team_total", side="away", op="over", line=20.5), b) == (None, None)
    assert market_leg_prob(Leg("player", op="over", line=1, player_id=1, stat="rec_yds"), b) == (None, None)


def test_price_market_uses_market_marginals_and_sim_dependence():
    rng = np.random.default_rng(1)
    n = 20000
    env = rng.normal(0, 6, n)
    h = np.maximum(0, np.round(28 + env + rng.normal(0, 7, n)))
    a = np.maximum(0, np.round(23 + env + rng.normal(0, 7, n)))
    s = GameSims(h, a, None, ({}, {}), calib={"_corr_shrink": 1.0})
    legs = [Leg("team_total", side="home", op="over", line=25.5), Leg("total", op="over", line=50.5)]
    b = _book()
    out = price_market(s, legs, b, book_american=+250)
    assert [l["p"] for l in out["legs"]] == pytest.approx([0.56, 0.50])    # the market's marginals, not the sim's
    assert out["n_market_legs"] == 2 and out["classification"] == "calculated"
    assert out["lift_raw"] > 1.3                                           # shared environment -> positive dependence
    assert out["p_joint"] <= 0.50 + 1e-9
    assert out["p_joint"] == pytest.approx(0.56 * 0.50 * out["lift_raw"], rel=0.02)
    assert out["ev_per_unit"] == pytest.approx(out["p_joint"] * 3.5 - 1)
    # shrinkage toward independence
    half = price_market(s, legs, b, alpha=0.0)
    assert half["p_joint"] == pytest.approx(0.28)
    # no market: falls back to the model marginal for every leg
    none = price_market(s, legs, None)
    assert none["n_market_legs"] == 0
    assert none["legs"][1]["p"] == pytest.approx(float((h + a > 50.5).mean()))
