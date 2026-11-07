import numpy as np

from cfbquant.sim.engine import simulate
from cfbquant.sim.params import STAT, TSTAT, League, TeamParams, broadcast_params, stack_params


def run(n=3000, seed=1, h=None, a=None, **kw):
    return simulate(broadcast_params(h or TeamParams(), a or TeamParams(), n), League(), seed, **kw)


def test_accounting_identities():
    r = run()
    ps, ts = r.player_stats, r.team_stats
    assert (ps[..., STAT["pass_cmp"]] <= ps[..., STAT["pass_att"]]).all()
    assert (ps[..., STAT["receptions"]] <= ps[..., STAT["targets"]]).all()
    assert (ps[..., STAT["pass_yds"]].sum(2) == ps[..., STAT["rec_yds"]].sum(2)).all()
    assert (ps[..., STAT["pass_td"]].sum(2) == ps[..., STAT["rec_td"]].sum(2)).all()
    assert (ps[..., STAT["pass_cmp"]].sum(2) == ps[..., STAT["receptions"]].sum(2)).all()
    assert (ps[..., STAT["rush_yds"]].sum(2) == ts[..., TSTAT["rush_yds"]]).all()
    assert (ps[..., STAT["fgm"]] <= ps[..., STAT["fga"]]).all()


def test_scores_consistent_with_scoring_events():
    r = run(n=2000, seed=5)
    ts = r.team_stats
    # every point is a TD(6)+PAT, FG(3), safety(2), or 2-pt; lower bound check from TDs and FGs
    pts = np.stack([r.home_pts, r.away_pts], 1)
    assert (pts >= 6 * ts[..., TSTAT["td"]] + 3 * ts[..., TSTAT["fgm"]]).all()
    assert (pts <= 8 * ts[..., TSTAT["td"]] + 3 * ts[..., TSTAT["fgm"]] + 2 * 3 + 7 * 2 + 30).all()


def test_reproducible_with_seed():
    a, b = run(seed=42), run(seed=42)
    assert (a.home_pts == b.home_pts).all() and (a.player_stats == b.player_stats).all()
    c = run(seed=43)
    assert not (a.home_pts == c.home_pts).all()


def test_no_ties_and_overtime_used():
    r = run(n=5000)
    assert (r.home_pts != r.away_pts).all()
    assert 0.01 < r.ot.mean() < 0.15


def test_calibrated_environment():
    r = run(n=6000, seed=9)
    ppt = (r.home_pts + r.away_pts).mean() / 2
    plays = r.team_stats[..., TSTAT["plays"]].mean()
    hfa = (r.home_pts - r.away_pts).mean()
    assert 24 < ppt < 32, ppt
    assert 62 < plays < 80, plays
    assert 1.0 < hfa < 4.5, hfa
    rn = run(n=6000, seed=9, neutral=np.ones(6000, bool))
    assert abs((rn.home_pts - rn.away_pts).mean()) < 1.0


def test_better_team_wins_more():
    strong = TeamParams().with_(comp_off=0.70, rcore_off=4.0, comp_def=0.58, rcore_def=2.7)
    r = run(n=4000, h=strong)
    assert (r.home_pts > r.away_pts).mean() > 0.75


def test_per_lane_parameters():
    strong = TeamParams().with_(comp_off=0.72, rcore_off=4.2)
    P = stack_params([(strong, TeamParams())] * 500 + [(TeamParams(), strong)] * 500)
    r = simulate(P, League(), 3)
    m = r.home_pts - r.away_pts
    assert m[:500].mean() > m[500:].mean() + 5
