"""Game + player forecasts, all derived from ONE simulation run so they cannot contradict.

forecast_game() -> dict with win probabilities (raw and calibrated), score /
margin / total distributions, team stat distributions, player stat
distributions for every projected starter, scenario impacts and key drivers.
Key drivers are measured by *neutralisation*: re-simulate with one unit
matchup set to league average (common random numbers) and report the change
in expected margin. That keeps driver attribution inside the same model.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from ..db import utcnow
from ..sim.engine import simulate
from ..sim.params import (DEF_SLOTS, OFF_SLOTS, SLOT, SLOTS, STAT, STATS, TSTAT, League, TeamParams,
                          broadcast_params, logit, sigmoid)
from ..analytics.team_params import BASE, apply_qb_change, estimate

MODEL_VERSION = "sim_v1"
POS_STATS = {
    "QB": ["pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "sacks_taken", "rush_att", "rush_yds", "rush_td"],
    "RB": ["rush_att", "rush_yds", "rush_td", "targets", "receptions", "rec_yds", "rec_td"],
    "WR": ["targets", "receptions", "rec_yds", "rec_td"], "TE": ["targets", "receptions", "rec_yds", "rec_td"],
    "K": ["fga", "fgm", "xpa", "xpm"], "P": ["punts", "punt_yds"],
    "DEF": ["tackles", "tfl", "def_sacks", "def_int", "pbu", "forced_fumbles"],
}
SLOT_GROUP = {s: ("QB" if s.startswith("QB") else "RB" if s.startswith("RB") else "WR" if s.startswith("WR")
                  else "TE" if s.startswith("TE") else s if s in ("K", "P") else "DEF") for s in SLOTS}


def load_calibrator(conn) -> dict | None:
    r = conn.execute("""SELECT params_json FROM model_versions WHERE family='sim_calibrator'
                        ORDER BY CASE status WHEN 'production' THEN 0 ELSE 1 END, created_at DESC LIMIT 1""").fetchone()
    return json.loads(r[0]) if r else None


def calibrate_prob(p: float | np.ndarray, cal: dict | None):
    if not cal:
        return p
    return sigmoid(cal["a"] * logit(np.clip(p, 1e-4, 1 - 1e-4)) + cal["b"])


def _neutral(tp: TeamParams, unit: str) -> TeamParams:
    b = BASE
    if unit == "pass_off":
        return tp.with_(comp_off=b.comp_off, int_off=b.int_off, sack_off=b.sack_off, yac_off=b.yac_off, adot=b.adot)
    if unit == "rush_off":
        return tp.with_(rcore_off=b.rcore_off, rexp_off=b.rexp_off, fumble_off=b.fumble_off)
    if unit == "pass_def":
        return tp.with_(comp_def=b.comp_def, int_def=b.int_def, sack_def=b.sack_def, yac_def=b.yac_def)
    if unit == "rush_def":
        return tp.with_(rcore_def=b.rcore_def, rexp_def=b.rexp_def, fumble_def=b.fumble_def)
    if unit == "special":
        return tp.with_(fg_skill=0.0, punt_net=b.punt_net)
    return tp


def apply_scenario(est, team: int, tp: TeamParams, slots: dict, scenario: dict | None):
    """scenario: {"out": [player_ids], "qb_sd": float (team-specific), }"""
    if not scenario:
        return tp, slots
    slots = dict(slots)
    outs = set(scenario.get("out", []))
    qbv = est.qb_value
    sd = float(np.std(list(qbv.values()))) if qbv else 0.1
    for pid in outs:
        for s, v in list(slots.items()):
            if v != pid:
                continue
            if s == "QB1":
                backup = slots.get("QB2")
                d = qbv.get(backup, min(qbv.values()) if qbv else -0.1) - qbv.get(pid, 0.0)
                tp = apply_qb_change(tp, d)
                slots["QB1"], slots["QB2"] = backup, None
            elif s in ("RB1", "RB2", "WR1", "WR2", "WR3", "WR4", "TE1", "TE2"):
                ts = list(tp.target_share)
                tslots = ["RB1", "RB2", "WR1", "WR2", "WR3", "WR4", "TE1", "TE2"]
                ts[tslots.index(s)] = 0.0
                kw = {"target_share": tuple(ts)}
                rs_slots = ["QB1", "RB1", "RB2", "WR1"]
                if s in rs_slots:
                    rs = list(tp.rush_share)
                    rs[rs_slots.index(s)] = 0.0
                    kw["rush_share"] = tuple(rs)
                # modest efficiency loss when a top-2 target / lead back is replaced
                tp = tp.with_(**kw)
                if s in ("WR1", "TE1", "RB1"):
                    tp = apply_qb_change(tp, -0.02)
                slots[s] = None
            else:
                slots[s] = None
    if scenario.get("qb_sd"):
        tp = apply_qb_change(tp, scenario["qb_sd"] * sd)
    return tp, slots


def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    return np.quantile(x, qs)


def summarize(res, league: League, cal: dict | None = None) -> dict:
    h, a = res.home_pts.astype(float), res.away_pts.astype(float)
    m, t = h - a, h + a
    p_raw = float((m > 0).mean())
    ts = res.team_stats
    team = {}
    for side, nm in ((0, "home"), (1, "away")):
        team[nm] = {k: {"mean": float(ts[:, side, i].mean()), "sd": float(ts[:, side, i].std()),
                        "p10": float(np.quantile(ts[:, side, i], 0.1)), "p90": float(np.quantile(ts[:, side, i], 0.9))}
                    for k, i in TSTAT.items()}
    mq = _q(m)
    tq = _q(t)
    bins_t = np.arange(0, 132, 4)
    bins_m = np.arange(-56, 60, 4)
    return {
        "n_sims": int(len(h)),
        "home_win_prob_raw": p_raw, "home_win_prob": float(calibrate_prob(p_raw, cal)), "calibrated": bool(cal),
        "away_win_prob": 1 - float(calibrate_prob(p_raw, cal)),
        "ot_prob": float(res.ot.mean()),
        "home_pts": {"mean": float(h.mean()), "median": float(np.median(h)), "sd": float(h.std())},
        "away_pts": {"mean": float(a.mean()), "median": float(np.median(a)), "sd": float(a.std())},
        "margin": {"mean": float(m.mean()), "sd": float(m.std()), "p10": mq[0], "p25": mq[1], "p50": mq[2], "p75": mq[3], "p90": mq[4]},
        "total": {"mean": float(t.mean()), "sd": float(t.std()), "p10": tq[0], "p25": tq[1], "p50": tq[2], "p75": tq[3], "p90": tq[4]},
        "total_hist": {"edges": bins_t.tolist(), "home_win": np.histogram(t[m > 0], bins_t)[0].tolist(),
                       "away_win": np.histogram(t[m < 0], bins_t)[0].tolist()},
        "margin_hist": {"edges": bins_m.tolist(), "counts": np.histogram(m, bins_m)[0].tolist()},
        "top_scores": _top_scores(h, a),
        "team_stats": team,
    }


def _top_scores(h, a, k=8):
    s = pd.Series(list(zip(h.astype(int), a.astype(int)))).value_counts().head(k)
    return [{"home": int(x[0]), "away": int(x[1]), "prob": float(c / len(h))} for x, c in s.items()]


def player_projections(res, slots_by_side: list[dict], info: pd.DataFrame) -> list[dict]:
    out = []
    ps = res.player_stats
    for side in (0, 1):
        for s, pid in slots_by_side[side].items():
            if pid is None:
                continue
            grp = SLOT_GROUP[s]
            row = {"player_id": int(pid), "side": "home" if side == 0 else "away", "slot": s, "group": grp,
                   "name": None, "position": None, "stats": {}}
            if pid in info.index:
                r = info.loc[pid]
                row["name"] = f"{r.first_name} {r.last_name}"
                row["position"] = r.position
            for st in POS_STATS[grp]:
                x = ps[:, side, SLOT[s], STAT[st]].astype(float)
                q = _q(x)
                d = {"mean": float(x.mean()), "median": float(q[2]), "sd": float(x.std()),
                     "p10": float(q[0]), "p25": float(q[1]), "p75": float(q[3]), "p90": float(q[4])}
                if st.endswith("_td") or st in ("pass_int", "def_sacks", "def_int", "receptions", "fgm", "tackles"):
                    vals, cnt = np.unique(x.astype(int), return_counts=True)
                    d["dist"] = {int(v): float(c / len(x)) for v, c in zip(vals, cnt) if c / len(x) >= 0.002}
                    d["p_ge1"] = float((x >= 1).mean())
                row["stats"][st] = d
            out.append(row)
    return out


def forecast_game(conn, home: int, away: int, season: int, week: int, *, n_sims: int = 10000, seed: int = 1,
                  neutral: bool = False, scenario: dict | None = None, drivers: bool = True,
                  scenarios: bool = True) -> dict:
    est = estimate(conn, season, week)
    league = League()
    cal = load_calibrator(conn)
    hp, ap = est.params[home], est.params[away]
    hs, as_ = est.slots[home], est.slots[away]
    sc = scenario or {}
    hp, hs = apply_scenario(est, home, hp, hs, sc.get(str(home)) or sc.get(home))
    ap, as_ = apply_scenario(est, away, ap, as_, sc.get(str(away)) or sc.get(away))
    neu = np.full(n_sims, bool(neutral))
    res = simulate(broadcast_params(hp, ap, n_sims), league, seed, neutral=neu)
    out = summarize(res, league, cal)
    out.update({"home_id": home, "away_id": away, "season": season, "week": week, "neutral": neutral,
                "model_version": MODEL_VERSION, "seed": seed, "as_of": f"{season} before week {week}",
                "generated_at": utcnow(), "params": {"home": _pdict(hp), "away": _pdict(ap)}})
    out["players"] = player_projections(res, [hs, as_], est.player_info)
    out["consistency_checks"] = consistency_checks(res)
    n_small = max(2000, n_sims // 4)
    base_small = None
    if drivers or scenarios:
        base_small = simulate(broadcast_params(hp, ap, n_small), league, seed + 7, neutral=np.full(n_small, neutral),
                              track_players=False)
        base_m = float((base_small.home_pts - base_small.away_pts).mean())
    if drivers:
        drv = []
        for label, side, unit in [("Home pass offense", 0, "pass_off"), ("Home rush offense", 0, "rush_off"),
                                  ("Home pass defense", 0, "pass_def"), ("Home rush defense", 0, "rush_def"),
                                  ("Away pass offense", 1, "pass_off"), ("Away rush offense", 1, "rush_off"),
                                  ("Away pass defense", 1, "pass_def"), ("Away rush defense", 1, "rush_def"),
                                  ("Home special teams", 0, "special"), ("Away special teams", 1, "special")]:
            h2 = _neutral(hp, unit) if side == 0 else hp
            a2 = _neutral(ap, unit) if side == 1 else ap
            r2 = simulate(broadcast_params(h2, a2, n_small), league, seed + 7, neutral=np.full(n_small, neutral),
                          track_players=False)
            dm = base_m - float((r2.home_pts - r2.away_pts).mean())
            drv.append({"driver": label, "unit": unit, "side": "home" if side == 0 else "away", "margin_pts": dm})
        if not neutral:
            r2 = simulate(broadcast_params(hp, ap, n_small), league, seed + 7, neutral=np.full(n_small, True),
                          track_players=False)
            drv.append({"driver": "Home-field advantage", "unit": "hfa", "side": "home",
                        "margin_pts": base_m - float((r2.home_pts - r2.away_pts).mean())})
        out["drivers"] = sorted(drv, key=lambda d: -abs(d["margin_pts"]))
    if scenarios:
        # all scenario rows (incl. base) share common random numbers so differences are model, not noise
        rows = [{"scenario": "Base case (current)",
                 "home_win_prob": float(calibrate_prob(float((base_small.home_pts > base_small.away_pts).mean()), cal)),
                 "home_pts": float(base_small.home_pts.mean()), "away_pts": float(base_small.away_pts.mean())}]
        for label, sc2 in [("Home QB +1 SD", {home: {"qb_sd": 1.0}}), ("Away QB +1 SD", {away: {"qb_sd": 1.0}}),
                           ("Home QB -1 SD", {home: {"qb_sd": -1.0}}), ("Away QB -1 SD", {away: {"qb_sd": -1.0}}),
                           ("Home WR1 out", {home: {"out": [hs.get("WR1")]}}),
                           ("Away WR1 out", {away: {"out": [as_.get("WR1")]}}),
                           ("Home QB1 out", {home: {"out": [hs.get("QB1")]}}),
                           ("Away QB1 out", {away: {"out": [as_.get("QB1")]}})]:
            h2, _ = apply_scenario(est, home, hp, hs, sc2.get(home))
            a2, _ = apply_scenario(est, away, ap, as_, sc2.get(away))
            r2 = simulate(broadcast_params(h2, a2, n_small), league, seed + 7, neutral=np.full(n_small, neutral),
                          track_players=False)
            p = float(calibrate_prob(float((r2.home_pts > r2.away_pts).mean()), cal))
            rows.append({"scenario": label, "home_win_prob": p, "home_pts": float(r2.home_pts.mean()),
                         "away_pts": float(r2.away_pts.mean())})
        out["scenarios"] = rows
    return out


def consistency_checks(res) -> dict:
    ps, ts = res.player_stats, res.team_stats
    checks = {
        "completions_le_attempts": bool((ps[..., STAT["pass_cmp"]] <= ps[..., STAT["pass_att"]]).all()),
        "pass_yds_eq_rec_yds": bool((ps[..., STAT["pass_yds"]].sum(2) == ps[..., STAT["rec_yds"]].sum(2)).all()),
        "pass_td_eq_rec_td": bool((ps[..., STAT["pass_td"]].sum(2) == ps[..., STAT["rec_td"]].sum(2)).all()),
        "team_rush_eq_players": bool((ps[..., STAT["rush_yds"]].sum(2) == ts[..., TSTAT["rush_yds"]]).all()),
        "team_pass_eq_players": bool((ps[..., STAT["pass_yds"]].sum(2) == ts[..., TSTAT["pass_yds"]]).all()),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def _pdict(tp: TeamParams) -> dict:
    d = {k: getattr(tp, k) for k in ("pass_rate", "pace", "comp_off", "comp_def", "sack_off", "sack_def", "int_off",
                                     "int_def", "rcore_off", "rcore_def", "rexp_off", "rexp_def", "yac_off", "yac_def",
                                     "adot", "fg_skill", "punt_net", "go_aggr")}
    return {k: round(float(v), 4) for k, v in d.items()}


def store_forecast(conn, fc: dict, game_id: int | None, kind: str = "live") -> int:
    cur = conn.execute("""INSERT INTO model_predictions (game_id, model_version, created_at, as_of, kind, home_win_prob,
        home_pts_mean, away_pts_mean, margin_mean, total_mean, margin_sd, total_sd, margin_p10, margin_p90, total_p10,
        total_p90, ot_prob, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       (game_id or -1, fc["model_version"], utcnow(), fc["as_of"], kind, fc["home_win_prob"],
                        fc["home_pts"]["mean"], fc["away_pts"]["mean"], fc["margin"]["mean"], fc["total"]["mean"],
                        fc["margin"]["sd"], fc["total"]["sd"], fc["margin"]["p10"], fc["margin"]["p90"],
                        fc["total"]["p10"], fc["total"]["p90"], fc["ot_prob"],
                        json.dumps({k: fc[k] for k in ("top_scores", "drivers", "scenarios") if k in fc})))
    pid = cur.lastrowid
    rows = []
    for pl in fc["players"]:
        for st, d in pl["stats"].items():
            rows.append((pid, pl["player_id"], st, d["mean"], d["median"], d["sd"], d["p10"], d["p25"], d["p75"], d["p90"]))
    conn.executemany("INSERT OR REPLACE INTO player_predictions VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.execute("""INSERT INTO simulation_runs (game_id, home_id, away_id, model_version, n_sims, seed, created_at, scenario,
        summary_json) VALUES (?,?,?,?,?,?,?,?,?)""", (game_id, fc["home_id"], fc["away_id"], fc["model_version"],
                                                      fc["n_sims"], fc["seed"], utcnow(), None,
                                                      json.dumps({"margin": fc["margin"], "total": fc["total"],
                                                                  "home_win_prob": fc["home_win_prob"]})))
    conn.commit()
    return pid
