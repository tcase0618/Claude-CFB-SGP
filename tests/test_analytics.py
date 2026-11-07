import numpy as np
import pandas as pd

from cfbquant.analytics import player_metrics as pm
from cfbquant.analytics import team_metrics as tm
from cfbquant.analytics.ep_model import EPModel
from cfbquant.analytics.team_metrics import ridge_adjust


def test_registry_sizes():
    from collections import Counter
    team = tm.ALL_TEAM_METRICS
    assert len(team) >= 200
    cats = Counter(m.cat for m in team)
    assert len(cats) == 20 and min(cats.values()) >= 10
    assert len({m.mid for m in team}) == len(team), "duplicate team metric ids"
    uni = [d for d in pm.DEFS if d.scope == "player_universal"]
    assert len(uni) >= 80
    assert len({d.mid for d in pm.DEFS}) == len(pm.DEFS), "duplicate player metric ids"
    for pos in ("QB", "RB", "WR", "TE", "DL", "EDGE", "LB", "CB", "S"):
        assert sum(1 for d in pm.DEFS if d.scope == "player_position" and pos in d.positions.split(",")) >= 12, pos


def test_ep_monotone(conn):
    p = pd.read_sql_query("SELECT yards_to_goal, down, distance, ep_before FROM plays WHERE ep_before IS NOT NULL AND down=1 AND distance=10", conn)
    g = p.groupby(pd.cut(p.yards_to_goal, [10, 30, 50, 70, 90])).ep_before.mean().values
    assert all(np.diff(g) < 0), g


def test_success_definition(conn):
    p = pd.read_sql_query("""SELECT down, distance, yards_gained, success, touchdown, interception, fumble_lost FROM plays
                             WHERE success IS NOT NULL AND play_type IN ('rush','pass') LIMIT 5000""", conn)
    need = np.select([p.down == 1, p.down == 2], [0.5, 0.7], 1.0) * p.distance
    exp = ((p.yards_gained >= need) | (p.touchdown == 1)) & ~((p.interception == 1) | (p.fumble_lost == 1))
    assert (exp.astype(int) == p.success).mean() > 0.995


def test_unavailable_metrics_never_filled(conn):
    bad = conn.execute("""SELECT COUNT(*) FROM team_metrics tm JOIN metric_definitions md ON md.metric_id=tm.metric_id
        WHERE md.classification='unavailable_without_charting' AND tm.value IS NOT NULL""").fetchone()[0]
    assert bad == 0
    bad = conn.execute("""SELECT COUNT(*) FROM player_metrics pm JOIN metric_definitions md ON md.metric_id=pm.metric_id
        WHERE md.classification='unavailable_without_charting'""").fetchone()[0]
    assert bad == 0


def test_percentiles_bounded(conn):
    lo, hi = conn.execute("SELECT MIN(pctl_fbs), MAX(pctl_fbs) FROM team_metrics").fetchone()
    assert 0 <= lo and hi <= 100
    lo, hi = conn.execute("SELECT MIN(pctl_fbs), MAX(pctl_fbs) FROM player_metrics").fetchone()
    assert 0 <= lo and hi <= 100


def test_ridge_adjust_recovers_structure():
    rng = np.random.default_rng(0)
    teams = list(range(20))
    off = rng.normal(0, 0.1, 20)
    de = rng.normal(0, 0.1, 20)
    rows = []
    for _ in range(400):
        i, j = rng.choice(20, 2, replace=False)
        rows.append(dict(off=i, **{"def": j}, home=1, value=off[i] + de[j] + 0.02 + rng.normal(0, 0.05), w=60))
    ao, ad, mu = ridge_adjust(pd.DataFrame(rows), lam=1.0)
    est_o = np.array([ao[t] for t in teams])
    est_d = np.array([ad[t] for t in teams])
    assert np.corrcoef(est_o, off)[0, 1] > 0.9 and np.corrcoef(est_d, de)[0, 1] > 0.9


def test_opponent_adjustment_rewards_tough_schedule(conn):
    """Adjusted offense should differ from raw in the direction of opponent quality."""
    m = pd.read_sql_query("""SELECT team_id, value, adj_value FROM team_metrics WHERE season=2025 AND metric_id='off_epa'
                             AND through_week=(SELECT MAX(through_week) FROM team_metrics WHERE season=2025)""", conn)
    sos = pd.read_sql_query("""SELECT team_id, value FROM team_metrics WHERE season=2025 AND metric_id='sos'
                               AND through_week=(SELECT MAX(through_week) FROM team_metrics WHERE season=2025)""", conn)
    d = m.merge(sos, on="team_id", suffixes=("", "_sos")).dropna()
    # sos = mean opp net EPA: tougher opponents (higher net => better defences) => adj - raw tends to be positive
    assert np.corrcoef(d.adj_value - d.value, d.value_sos)[0, 1] > 0.2


def test_player_grades_and_archetypes(conn):
    g = pd.read_sql_query("SELECT * FROM player_profiles WHERE season=2025", conn)
    assert g.grade.dropna().between(0, 100).all()
    assert g[g.position == "QB"].archetype.notna().sum() >= 4


def test_param_recovery(conn):
    from cfbquant.analytics.team_params import _estimate
    tr = pd.read_sql_query("SELECT * FROM synthetic_truth WHERE season=2025", conn).set_index("team_id")
    e = _estimate(conn, 2025, 8, None)
    for k in ("comp_off", "rcore_off", "pass_rate"):
        est = [getattr(e.params[t], k) for t in tr.index]
        assert np.corrcoef(est, tr[k])[0, 1] > 0.5, k
