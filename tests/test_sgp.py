import numpy as np
import pytest

from cfbquant.ingest.espn import parse_core_odds
from cfbquant.models.props import Uncertainty, anchored_simulate, line_targets, materialize, perturb
from cfbquant.models.sgp import (GameSims, Leg, american_to_decimal, apply_factor, build_tickets, calibration_factors,
                                 candidate_legs, price, prob_to_american)
from cfbquant.sim.params import N_SLOTS, N_STATS, SLOT, STAT, League, TeamParams, broadcast_params, stack_params


def _fake_sims(n=4000, seed=0):
    """Hand-built lanes with a known correlation: QB yards drive WR yards and home points."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    ps = np.zeros((n, 2, N_SLOTS, N_STATS), np.int32)
    qb = np.maximum(0, 230 + 60 * z + rng.normal(0, 20, n)).astype(int)
    wr = np.maximum(0, 0.35 * qb + rng.normal(0, 18, n)).astype(int)
    ps[:, 0, SLOT["QB1"], STAT["pass_yds"]] = qb
    ps[:, 0, SLOT["WR1"], STAT["rec_yds"]] = wr
    ps[:, 0, SLOT["WR1"], STAT["receptions"]] = np.maximum(0, (wr / 13).round()).astype(int)
    ps[:, 1, SLOT["RB1"], STAT["rush_yds"]] = np.maximum(0, 80 + rng.normal(0, 30, n)).astype(int)   # independent
    h = np.maximum(0, 28 + 7 * z + rng.normal(0, 7, n)).round()
    a = np.maximum(0, 24 + rng.normal(0, 9, n)).round()
    slots = ({"QB1": 1, "WR1": 2}, {"RB1": 3})
    return GameSims(h, a, ps, slots, ("Home U", "Away St"), {1: "Q B", 2: "W R", 3: "R B"})


def test_odds_helpers():
    assert prob_to_american(0.5) == pytest.approx(-100)
    assert prob_to_american(0.2) == pytest.approx(400)
    assert american_to_decimal(400) == pytest.approx(5.0)
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_leg_semantics_and_joint_price():
    s = _fake_sims()
    qb = Leg("player", op="over", line=229.5, player_id=1, stat="pass_yds")
    wr = Leg("player", op="over", line=79.5, player_id=2, stat="rec_yds")
    rb = Leg("player", op="over", line=79.5, player_id=3, stat="rush_yds")
    ml = Leg("ml", side="home")
    sp = Leg("spread", side="away", line=3.5)
    assert np.array_equal(s.hits(ml), s.h > s.a)
    assert np.array_equal(s.hits(sp), (s.a - s.h) + 3.5 > 0)
    corr = price(s, [qb, wr])
    assert corr["p_joint"] <= min(l["p"] for l in corr["legs"]) + 1e-12
    assert corr["lift"] > 1.25                     # built-in positive correlation is recovered
    ind = price(s, [qb, rb])
    assert ind["lift"] == pytest.approx(1.0, abs=0.08)
    ev = price(s, [qb, wr], book_american=+250)
    assert ev["ev_per_unit"] == pytest.approx(ev["p_joint"] * 3.5 - 1)
    assert ev["kelly"] >= 0


def test_ticket_builder_respects_band_and_market_rules():
    s = _fake_sims()
    legs = candidate_legs(s, {"spread_home": -4.5, "total": 51.5})
    assert all(not (l.kind == "player" and l.stat == "rec_yds" and l.line < 19.5) for l in legs)
    for size in (3, 5):
        ts = build_tickets(s, legs, size, top=2)
        assert ts, size
        for t in ts:
            assert t["n_legs"] == size
            keys = [(l["kind"], l.get("player_id"), l.get("stat"), l.get("side")) for l in t["legs"]]
            pids = [k[1] for k in keys if k[0] == "player"]
            assert len(set(pids)) == len(pids)                 # one leg per player
            assert sum(l["kind"] in ("ml", "spread") for l in t["legs"]) <= 1
            assert t["n_sims"] == s.n - s.n // 2         # priced on the held-out half


def test_apply_factor_keeps_counts_integer_and_scales_mean():
    rng = np.random.default_rng(1)
    x = rng.poisson(5, 20000).astype(float)
    y = apply_factor(x, 0.9, True, rng)
    assert np.all(y == np.round(y)) and np.all(y <= x)
    assert y.mean() == pytest.approx(0.9 * x.mean(), rel=0.02)
    z = apply_factor(x, 1.1, True, rng)
    assert z.mean() == pytest.approx(1.1 * x.mean(), rel=0.02)
    assert np.allclose(apply_factor(x, 0.8, False, rng), 0.8 * x)
    f = calibration_factors({"pass_yds": [180.0 * 400, 200.0 * 400, 400], "rec_yds": [0, 0, 0]})
    assert 0.9 < f["pass_yds"] < 0.95 and "rec_yds" not in f       # shrunk toward 1


def test_calibrated_stat_is_reused_across_legs():
    s = _fake_sims()
    s.calib = {"pass_yds": 0.9, "receptions": 0.8}
    a, b = s.stat(1, "pass_yds"), s.stat(1, "pass_yds")
    assert np.array_equal(a, b)
    r1, r2 = s.stat(2, "receptions"), s.stat(2, "receptions")
    assert np.array_equal(r1, r2)              # one thinning draw per (player, stat)
    assert np.allclose(s.stat(1, "pass_yds", raw=True) * 0.9, a)


def test_perturb_keeps_shares_normalised():
    P = materialize(stack_params([(TeamParams(), TeamParams())] * 50))
    perturb(P, np.random.default_rng(0), Uncertainty(0.3, 0.07, 30))
    assert np.allclose(P["target_share"].sum(axis=2), 1)
    assert P["comp_off"].std() > 0 and (P["comp_off"] > 0).all() and (P["comp_off"] < 1).all()


def test_anchor_hits_line_targets():
    n, G = 500, 2
    P = {k: np.repeat(v, n, axis=0) for k, v in stack_params([(TeamParams(), TeamParams())] * G).items()}
    gl = np.repeat(np.arange(G), n)
    tg = np.array([line_targets(-10.5, 58.5), line_targets(3.0, 41.0)])
    res, sh = anchored_simulate(P, gl, tg, League(), 7, neutral=np.zeros(n * G, bool), unc=Uncertainty(0.2, 0.05, 30),
                                n_anchor=500, track_players=False)
    hp = res.home_pts.reshape(G, n).mean(1)
    ap = res.away_pts.reshape(G, n).mean(1)
    assert np.abs(hp - tg[:, 0]).max() < 1.6 and np.abs(ap - tg[:, 1]).max() < 1.6
    assert sh[0, 0] > sh[0, 1] and sh[1, 1] > sh[1, 0]      # favourite's offence shifted up


def test_line_targets_convention():
    h, a = line_targets(-7.0, 50.0)
    assert (h, a) == (28.5, 21.5)


def test_parse_core_odds():
    item = {"provider": {"name": "DraftKings"}, "spread": -6.5, "overUnder": 48.5, "overOdds": -110.0, "underOdds": -110.0,
            "homeTeamOdds": {"moneyLine": -218, "spreadOdds": -105.0,
                             "open": {"pointSpread": {"american": "-7.5"}, "moneyLine": {"american": "-305"}}},
            "awayTeamOdds": {"moneyLine": 180, "spreadOdds": -115.0,
                             "open": {"pointSpread": {"american": "+7.5"}, "moneyLine": {"american": "+245"}}},
            "open": {"total": {"american": "47.5"}}}
    r = parse_core_odds(item, 1, "t")
    assert r["spread_home"] == -6.5 and r["open_spread_home"] == -7.5 and r["open_total"] == 47.5
    assert r["home_moneyline"] == -218 and r["open_away_moneyline"] == 245 and r["over_odds"] == -110


def test_engine_benches_starters_in_blowouts():
    from cfbquant.sim.engine import simulate
    strong = TeamParams(comp_off=0.75, rcore_off=4.6, rexp_off=0.18, yac_off=7.5)
    weak = TeamParams(comp_def=0.75, rcore_def=4.6, rexp_def=0.18, yac_def=7.5, comp_off=0.52, rcore_off=2.4)
    res = simulate(broadcast_params(strong, weak, 600), League(), 3)
    ps = res.player_stats
    assert ps[:, 0, SLOT["QB2"], STAT["pass_att"]].sum() > 0       # backup sees garbage-time snaps
    assert (ps[..., STAT["pass_yds"]].sum(2) == ps[..., STAT["rec_yds"]].sum(2)).all()   # identities still hold
