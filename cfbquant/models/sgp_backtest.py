"""Walk-forward validation of the prop / same-game-parlay engine.

For every completed game in the test weeks: estimate parameters strictly before the week, anchor each game to its
DraftKings line as captured by ESPN, simulate with parameter uncertainty, then score against what happened:

  * player props   PIT (probability-integral transform) and 80% interval coverage, *conditional on the player
                   appearing* (books void props on players who do not play, so non-appearances are excluded)
  * game lines     margin / total interval coverage
  * tickets        the builder's 3/5/8-leg tickets, priced on held-out lanes; predicted joint probability vs
                   realised hit rate, and the same for the independence price (product of leg marginals). If the
                   simulated correlation is real, the joint price must predict hits better than the independent one.

Lines come from ESPN's pickcenter, fetched after the games finished. For most games that is the closing number, but
ESPN does not timestamp it, so treat the anchor as "closing or near-closing".
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..analytics.team_params import _estimate
from ..sim.params import League, stack_params
from .props import Uncertainty, anchored_simulate, line_targets
from .sgp import CALIB_STATS, COMPOSITE, GameSims, build_card, calibration_factors

EVAL_PROPS = {"QB1": ["pass_yds", "pass_cmp", "pass_td", "rush_yds"], "RB1": ["rush_yds", "rush_rec_yds", "td"],
              "WR1": ["rec_yds", "receptions", "td"], "WR2": ["rec_yds", "receptions"], "TE1": ["rec_yds", "receptions"]}


def _actual_stats(conn, game_ids) -> pd.DataFrame:
    q = f"SELECT * FROM player_games WHERE game_id IN ({','.join(str(int(g)) for g in game_ids)})"
    pg = pd.read_sql_query(q, conn)
    for k, parts in COMPOSITE.items():
        pg[k] = sum(pg[p].fillna(0) for p in parts)
    return pg.set_index(["game_id", "player_id"])


def _leg_actual(leg: dict, g, act: pd.DataFrame):
    """Realised outcome of one leg: True/False, or None when void (player did not appear)."""
    h, a = float(g.home_points), float(g.away_points)
    sgn = 1 if leg.get("side") == "home" else -1
    k = leg["kind"]
    if k == "ml":
        return sgn * (h - a) > 0
    if k == "spread":
        return sgn * (h - a) + leg["line"] > 0
    v = None
    if k == "total":
        v = h + a
    elif k == "team_total":
        v = h if leg["side"] == "home" else a
    else:
        key = (int(g.game_id), int(leg["player_id"]))
        if key not in act.index:
            return None
        v = float(np.nan_to_num(act.loc[key, leg["stat"]]))
    return v > leg["line"] if leg["op"] == "over" else v < leg["line"]


def run_sgp_backtest(conn, season: int, weeks: list[int], n_sims: int = 2000, seed: int = 21,
                     configs: dict[str, Uncertainty] | None = None, build: str = "anchored_unc", log=print) -> dict:
    configs = configs or {"anchored_unc": Uncertainty()}
    acc = {st: [0.0, 0.0, 0] for st in CALIB_STATS}      # walk-forward prop calibration (earlier weeks only)
    factors_by_week = {}
    league = League()
    names = pd.read_sql_query("SELECT team_id, school FROM teams", conn).set_index("team_id").school.to_dict()
    games = pd.read_sql_query("""SELECT g.*, b.spread_home, b.over_under FROM games g
        JOIN betting_lines b ON b.game_id=g.game_id AND b.provider='DraftKings'
        WHERE g.season=? AND g.completed=1 AND b.spread_home IS NOT NULL AND b.over_under IS NOT NULL
        ORDER BY g.week, g.game_id""", conn, params=(season,))
    prop_rows, game_rows, ticket_rows = [], [], []
    t0 = time.time()
    for w in weeks:
        gw = games[games.week == w]
        if gw.empty:
            continue
        est = _estimate(conn, season, w, None)
        gw = gw[gw.home_id.isin(est.params) & gw.away_id.isin(est.params)].reset_index(drop=True)
        G = len(gw)
        base = stack_params([(est.params[h], est.params[a]) for h, a in zip(gw.home_id, gw.away_id)])
        P = {k: np.repeat(v, n_sims, axis=0) for k, v in base.items()}
        gl = np.repeat(np.arange(G), n_sims)
        neu = np.repeat(gw.neutral_site.values.astype(bool), n_sims)
        tg = np.array([line_targets(s, o) for s, o in zip(gw.spread_home, gw.over_under)])
        act = _actual_stats(conn, gw.game_id)
        info = est.player_info
        factors = calibration_factors(acc)
        factors_by_week[w] = factors
        new_acc = []
        for cname, unc in configs.items():
            res, _ = anchored_simulate(P, gl, tg, league, seed + season * 100 + w, neutral=neu, unc=unc)
            H = res.home_pts.reshape(G, n_sims)
            A = res.away_pts.reshape(G, n_sims)
            PS = res.player_stats.reshape(G, n_sims, *res.player_stats.shape[1:])
            for i, g in enumerate(gw.itertuples()):
                m, t = H[i] - A[i], H[i] + A[i]
                am, at = g.home_points - g.away_points, g.home_points + g.away_points
                game_rows.append(dict(config=cname, game_id=g.game_id, week=w, line_margin=-g.spread_home,
                                      line_total=g.over_under, sim_margin=m.mean(), sim_total=t.mean(),
                                      margin_sd=m.std(), total_sd=t.std(), margin=am, total=at,
                                      pit_margin=_pit(m, am), pit_total=_pit(t, at)))
                slots = (est.slots[g.home_id], est.slots[g.away_id])
                pn = {int(p): f"{info.loc[p].first_name} {info.loc[p].last_name}" for s in slots for p in s.values()
                      if p is not None and p in info.index}
                tn = (names.get(g.home_id, "Home"), names.get(g.away_id, "Away"))
                sims = GameSims(H[i], A[i], PS[i], slots, tn, pn, calib=factors, seed=int(g.game_id) % 100000)
                for side, team in ((0, g.home_id), (1, g.away_id)):
                    for slot, stats in EVAL_PROPS.items():
                        pid = slots[side].get(slot)
                        if pid is None:
                            continue
                        key = (int(g.game_id), int(pid))
                        played = key in act.index
                        if played and cname == build:
                            for st in CALIB_STATS:
                                xs = float(sims.base_stat(pid, st, raw=True).mean())
                                if xs > 0.05:          # skip stats the slot doesn't produce (QB receptions etc.)
                                    new_acc.append((st, float(np.nan_to_num(act.loc[key, st])), xs))
                        for st, raw in [(x, True) for x in stats] + [(x, False) for x in stats]:
                            x = sims.stat(pid, st, raw=raw)
                            y = float(np.nan_to_num(act.loc[key, st])) if played else np.nan
                            prop_rows.append(dict(config=cname + ("_raw" if raw else "_cal"), game_id=g.game_id, week=w, player_id=int(pid), slot=slot,
                                                  stat=st, played=played, mean=x.mean(), median=np.median(x),
                                                  p10=np.quantile(x, 0.1), p90=np.quantile(x, 0.9), actual=y,
                                                  pit=_pit(x, y) if played else np.nan,
                                                  p_over_median=(x > np.floor(np.median(x)) + 0.5).mean(),
                                                  hit_over_median=float(y > np.floor(np.median(x)) + 0.5) if played else np.nan))
                if cname != build:
                    continue
                card = build_card(sims, {"spread_home": g.spread_home, "total": g.over_under}, seed=int(g.game_id) % 1000)
                for size, tickets in card["tickets"].items():
                    for rank, tk in enumerate(tickets):
                        outs = [_leg_actual(l, g, act) for l in tk["legs"]]
                        void = any(o is None for o in outs)
                        ticket_rows.append(dict(game_id=g.game_id, week=w, size=int(size), rank=rank, p_joint=tk["p_joint"],
                                                p_indep=tk["p_indep"], lift=tk["lift"], search_lift=tk["search_lift"],
                                                void=void, hit=np.nan if void else float(all(outs)),
                                                legs_hit=np.nan if void else float(np.mean(outs)),
                                                leg_p=[l["p"] for l in tk["legs"]], leg_hit=outs,
                                                labels=" | ".join(l["label"] for l in tk["legs"])))
        for st, ya, xs in new_acc:                # only now: this week's results inform next week's factors
            acc[st][0] += ya
            acc[st][1] += xs
            acc[st][2] += 1
        log(f"  sgp backtest {season} wk{w:>2}: {G} games ({time.time() - t0:.0f}s) factors={ {k: round(v, 3) for k, v in factors.items()} }")
    return {"props": pd.DataFrame(prop_rows), "games": pd.DataFrame(game_rows), "tickets": pd.DataFrame(ticket_rows),
            "factors_by_week": factors_by_week, "final_factors": calibration_factors(acc), "acc": acc}


def _pit(x: np.ndarray, y: float) -> float:
    """Randomisation-free mid-PIT for discrete outcomes."""
    return float((x < y).mean() + 0.5 * (x == y).mean())


def summarize_sgp(bt: dict) -> dict:
    out = {}
    pr = bt["props"]
    if len(pr):
        p = pr[pr.played]
        p = p.assign(in80=(p.actual >= p.p10) & (p.actual <= p.p90))
        out["props"] = (p.groupby(["config", "stat"])
                        .agg(n=("actual", "size"), coverage80=("in80", "mean"), pit_mean=("pit", "mean"),
                             pit_sd=("pit", "std"), mae=("actual", lambda s: np.nan), bias=("actual", "mean"),
                             sim_mean=("mean", "mean"))
                        .assign(bias=lambda d: d.bias - d.sim_mean).drop(columns=["mae", "sim_mean"]).round(3))
        out["appearance_rate"] = pr.drop_duplicates(["config", "game_id", "player_id"]).groupby(["config", "slot"]).played.mean().round(3)
        # over-the-median legs: calibration of a main-line prop
        out["main_line"] = p.groupby("config").apply(lambda d: pd.Series({
            "n": len(d), "pred_over": d.p_over_median.mean(), "actual_over": d.hit_over_median.mean(),
            "brier": float(((d.p_over_median - d.hit_over_median) ** 2).mean())}), include_groups=False).round(4)
    g = bt["games"]
    if len(g):
        out["games"] = g.groupby("config").apply(lambda d: pd.Series({
            "n": len(d), "margin_cov80": ((d.pit_margin > 0.1) & (d.pit_margin < 0.9)).mean(),
            "total_cov80": ((d.pit_total > 0.1) & (d.pit_total < 0.9)).mean(),
            "margin_sd": d.margin_sd.mean(), "total_sd": d.total_sd.mean(),
            "anchor_err_margin": (d.sim_margin - d.line_margin).abs().mean(),
            "anchor_err_total": (d.sim_total - d.line_total).abs().mean()}), include_groups=False).round(3)
    t = bt["tickets"]
    if len(t):
        v = t[~t.void]
        rows = []
        for size, d in v.groupby("size"):
            y = d.hit.values
            ll = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1))))
            se = float(np.sqrt(np.sum(d.p_joint * (1 - d.p_joint))) / len(d))
            rows.append(dict(size=size, n=len(d), void_rate=float(t[t["size"] == size].void.mean()),
                             pred_joint=d.p_joint.mean(), pred_indep=d.p_indep.mean(), actual=y.mean(),
                             actual_minus_joint=y.mean() - d.p_joint.mean(), se=se,
                             logloss_joint=ll(d.p_joint.values), logloss_indep=ll(d.p_indep.values),
                             mean_lift=d.lift.mean(), leg_hit_rate=d.legs_hit.mean(),
                             leg_pred=float(np.mean([np.mean(x) for x in d.leg_p]))))
        out["tickets"] = pd.DataFrame(rows).set_index("size").round(4)
    return out


def persist_sgp(conn, bt: dict, season: int, weeks: list[int]) -> None:
    """Validation rows -> model_validation (model_version sgp_v1); final factors -> the live prop calibrator."""
    import json
    from ..db import utcnow
    from .sgp_service import save_prop_calibrator
    summ = summarize_sgp(bt)
    now = utcnow()
    conn.execute("DELETE FROM model_validation WHERE model_version='sgp_v1'")
    rows = []
    for size, r in summ["tickets"].iterrows():
        sc = f"tickets:{size}"
        rows += [(sc, "hit_rate", r.actual, r.n, json.dumps({"se": r.se})), (sc, "pred_joint", r.pred_joint, r.n, None),
                 (sc, "pred_indep", r.pred_indep, r.n, None), (sc, "logloss_joint", r.logloss_joint, r.n, None),
                 (sc, "logloss_indep", r.logloss_indep, r.n, None), (sc, "void_rate", r.void_rate, r.n, None)]
    for (cfg, st), r in summ["props"].iterrows():
        rows += [(f"prop:{cfg}:{st}", "coverage80", r.coverage80, r.n, None),
                 (f"prop:{cfg}:{st}", "pit_mean", r.pit_mean, r.n, None), (f"prop:{cfg}:{st}", "bias", r.bias, r.n, None)]
    conn.executemany("""INSERT INTO model_validation (model_version, created_at, scope, metric, value, ci_low, ci_high, n,
        detail_json) VALUES ('sgp_v1', ?, ?, ?, ?, NULL, NULL, ?, ?)""",
                     [(now, a, b, float(c), int(d), e) for a, b, c, d, e in rows])
    n_cal = int(sum(v[2] for v in bt["acc"].values()))
    factors = dict(bt["final_factors"])
    factors["_corr_shrink"] = fit_corr_shrink(bt["tickets"])     # live pricing uses alpha fitted on every test week
    save_prop_calibrator(conn, factors, (f"{season} wk{weeks[0]}", f"{season} wk{weeks[-1]}"), n_cal,
                         "Per-stat volume factors (actual / simulated) for projected starters who appeared; shrunk toward 1")
    conn.commit()


def fit_corr_shrink(t: pd.DataFrame) -> float:
    """alpha minimising ticket log loss of p = p_indep * lift_raw**alpha (pooled over sizes), grid 0..1.2."""
    v = t[~t.void]
    if len(v) < 50:
        return 1.0
    y = v.hit.values
    lr = np.log(np.clip(v.p_joint.values, 1e-6, 1) / np.clip(v.p_indep.values, 1e-6, 1))
    best = (np.inf, 1.0)
    for a in np.arange(0.0, 1.21, 0.02):
        p = np.clip(v.p_indep.values * np.exp(a * lr), 1e-4, 1 - 1e-4)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        best = min(best, (ll, float(a)))
    return round(best[1], 2)


def split_report(bt: dict, fit_weeks: list[int], test_weeks: list[int]) -> dict:
    """Out-of-sample view: correlation shrinkage fitted on fit_weeks, scored on test_weeks."""
    t = bt["tickets"]
    alpha = fit_corr_shrink(t[t.week.isin(fit_weeks)])
    v = t[t.week.isin(test_weeks) & ~t.void].copy()
    lr = np.log(np.clip(v.p_joint, 1e-6, 1) / np.clip(v.p_indep, 1e-6, 1))
    v["p_shrunk"] = np.minimum(v.p_indep * np.exp(alpha * lr), v.leg_p.map(min))
    ll = lambda p, y: float(-np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))))
    rows = []
    for size, d in v.groupby("size"):
        y = d.hit.values
        rows.append(dict(size=size, n=len(d), actual=y.mean(), se=float(np.sqrt(d.p_shrunk.mul(1 - d.p_shrunk).sum()) / len(d)),
                         pred_raw=d.p_joint.mean(), pred_shrunk=d.p_shrunk.mean(), pred_indep=d.p_indep.mean(),
                         ll_raw=ll(d.p_joint.values, y), ll_shrunk=ll(d.p_shrunk.values, y), ll_indep=ll(d.p_indep.values, y),
                         leg_pred=float(np.mean([np.mean(x) for x in d.leg_p])), leg_hit=d.legs_hit.mean()))
    legs = []
    for r in t[t.week.isin(test_weeks) & ~t.void].itertuples():
        for lp, lh, lab in zip(r.leg_p, r.leg_hit, r.labels.split(" | ")):
            kind = "team" if (" pts" in lab or lab.startswith("Game ") or lab.endswith(" ML") or
                              lab.split()[-1].lstrip("+-").replace(".", "").isdigit()) else \
                ("player_over" if " Over " in lab else "player_under")
            legs.append(dict(kind=kind, p=lp, hit=float(lh)))
    L = pd.DataFrame(legs)
    L["band"] = pd.cut(L.p, [0, 0.55, 0.65, 0.75, 0.85, 1])
    p = bt["props"]
    p = p[p.played & p.week.isin(test_weeks) & p.config.str.endswith("_cal")]
    tails = p.groupby("stat").pit.apply(lambda x: pd.Series({"n": len(x), "below_p05": (x < .05).mean(), "above_p95": (x > .95).mean(),
                                                              "cov80": ((x > .1) & (x < .9)).mean(), "pit_mean": x.mean()})).unstack()
    g = bt["games"]
    g = g[g.week.isin(test_weeks)]
    return {"alpha": alpha, "tickets": pd.DataFrame(rows).set_index("size").round(4),
            "legs": L.groupby(["kind", "band"], observed=True).agg(n=("hit", "size"), pred=("p", "mean"), hit=("hit", "mean")).round(3),
            "props": tails.round(3),
            "games": pd.Series({"n": len(g), "margin_cov80": ((g.pit_margin > .1) & (g.pit_margin < .9)).mean(),
                                "total_cov80": ((g.pit_total > .1) & (g.pit_total < .9)).mean(),
                                "sim_total_sd": g.total_sd.mean(), "real_total_resid_sd": (g.total - g.line_total).std()}).round(3)}
