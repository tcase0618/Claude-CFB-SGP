"""Chronological walk-forward backtest, calibration, and model promotion.

For every game in the test window:
  1. build parameters from data strictly before that week (team_params.estimate
     filters on season/week; tested in tests/test_leakage.py);
  2. simulate the whole week at once (lanes = games x sims);
  3. store full predicted distributions (game + player) in model_predictions /
     player_predictions with kind='backtest';
  4. score against actual results.

Benchmarks, each also walk-forward:
  elo        - margin-aware Elo (ratings.run_elo, pre-game values)
  ridge      - OLS margin on [opponent-adjusted net EPA diff, home], trained on prior weeks
  xgb        - XGBoost margin regressor on as-of features (if xgboost installed and >= 150 training games)
  sim_cal    - simulation win prob passed through a Platt calibrator fitted on prior weeks only

Promotion gate: the simulation model is marked `production` only if its
log-loss improvement over Elo has a bootstrap 90% CI entirely below zero;
otherwise Elo stays production and the simulator stays `candidate`.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from ..db import insert_df, utcnow
from ..analytics.plays_frame import load_plays
from ..analytics.ratings import run_elo
from ..analytics.team_metrics import ridge_adjust
from ..analytics.team_params import _estimate
from ..sim.engine import simulate
from ..sim.params import SLOT, STAT, League, stack_params
from .forecast import MODEL_VERSION

PLAYER_EVAL = {"QB1": ["pass_yds", "pass_td", "pass_int", "rush_yds"], "RB1": ["rush_yds", "rush_td", "rec_yds"],
               "WR1": ["rec_yds", "receptions", "rec_td"], "WR2": ["rec_yds", "receptions"], "TE1": ["rec_yds", "receptions"],
               "K": ["fgm"], "LB1": ["tackles"]}


def _asof_features(conn, season, week):
    """Opponent-adjusted net EPA/play and success as of `week` (strictly before)."""
    p = load_plays(conn, season, before_week=week, include_prior_seasons=1)
    if p.empty:
        return {}
    p = p[p.is_off & p.clean].copy()
    p["w"] = np.where(p.season == season, 1.0, 0.35)
    from ..analytics import team_metrics as _tm
    _tm.FCS_TEAMS = {r[0] for r in conn.execute("SELECT team_id FROM team_seasons WHERE season=? AND division='fcs'", (season,))}
    feats = {}
    for col in ("epa", "success"):
        gl = p.assign(v=p[col].astype(float) * p.w).groupby(["game_id", "offense_id", "defense_id", "off_home", "neutral_site"]) \
            .agg(s=("v", "sum"), w=("w", "sum")).reset_index()
        gl["value"] = gl.s / gl.w
        gl["home"] = np.where(gl.neutral_site == 1, 0, np.where(gl.off_home, 1, -1))
        ao, ad, mu = ridge_adjust(gl.rename(columns={"offense_id": "off", "defense_id": "def"}), lam=6.0)
        feats[col] = (ao, ad, mu)
    return feats


def run_backtest(conn, test_seasons: list[int], n_sims: int = 400, seed: int = 11, feature_seasons: list[int] | None = None,
                 log=print) -> dict:
    t0 = time.time()
    league = League()
    games = pd.read_sql_query("SELECT * FROM games WHERE completed=1 ORDER BY season, week, game_id", conn)
    elo = run_elo(conn).set_index("game_id")
    seasons = sorted(games.season.unique())
    feature_seasons = [int(x) for x in (feature_seasons or seasons)]
    test_seasons = [int(x) for x in test_seasons]
    inj = pd.read_sql_query("SELECT player_id, season, week FROM injuries WHERE status='OUT'", conn)
    rows, prow = [], []
    for s in map(int, feature_seasons):
        for w in map(int, sorted(games[games.season == s].week.unique())):
            gw = games[(games.season == s) & (games.week == w)]
            feats = _asof_features(conn, s, w)
            for g in gw.itertuples():
                r = dict(game_id=g.game_id, season=s, week=w, home_id=g.home_id, away_id=g.away_id,
                         neutral=g.neutral_site, conf_game=g.conference_game, home_pts=g.home_points,
                         away_pts=g.away_points, elo_p=elo.loc[g.game_id, "elo_p_home"],
                         elo_margin=elo.loc[g.game_id, "elo_margin"], elo_diff=elo.loc[g.game_id, "elo_home"] - elo.loc[g.game_id, "elo_away"])
                if feats:
                    ao, ad, mu = feats["epa"]
                    so, sd_, smu = feats["success"]
                    for side, t in (("h", g.home_id), ("a", g.away_id)):
                        r[f"off_{side}"] = ao.get(t, mu) - mu
                        r[f"def_{side}"] = ad.get(t, mu) - mu
                        r[f"sr_off_{side}"] = so.get(t, smu) - smu
                        r[f"sr_def_{side}"] = sd_.get(t, smu) - smu
                    r["epa_diff"] = (r["off_h"] - r["def_h"]) - (r["off_a"] - r["def_a"])
                rows.append(r)
            if s not in test_seasons:
                continue
            est = _estimate(conn, s, w, None)
            out_now = set(inj[(inj.season == s) & (inj.week == w)].player_id)
            pairs = [(est.params[g.home_id], est.params[g.away_id]) for g in gw.itertuples()]
            base = stack_params(pairs)
            P = {k: np.repeat(v, n_sims, axis=0) for k, v in base.items()}
            neu = np.repeat(gw.neutral_site.values.astype(bool), n_sims)
            res = simulate(P, league, seed + s * 100 + w, neutral=neu)
            G = len(gw)
            hp = res.home_pts.reshape(G, n_sims).astype(float)
            ap = res.away_pts.reshape(G, n_sims).astype(float)
            ps = res.player_stats.reshape(G, n_sims, 2, *res.player_stats.shape[2:])
            for i, g in enumerate(gw.itertuples()):
                m, t = hp[i] - ap[i], hp[i] + ap[i]
                rr = next(x for x in reversed(rows) if x["game_id"] == g.game_id)
                starters = [est.slots[g.home_id].get(k) for k in ("QB1", "RB1", "WR1", "WR2", "TE1")] + \
                           [est.slots[g.away_id].get(k) for k in ("QB1", "RB1", "WR1", "WR2", "TE1")]
                rr.update(sim_p=float((m > 0).mean()), sim_margin=float(m.mean()), sim_margin_sd=float(m.std()),
                          sim_total=float(t.mean()), sim_home=float(hp[i].mean()), sim_away=float(ap[i].mean()),
                          m_p10=float(np.quantile(m, 0.1)), m_p90=float(np.quantile(m, 0.9)),
                          t_p10=float(np.quantile(t, 0.1)), t_p90=float(np.quantile(t, 0.9)),
                          ot_p=float(res.ot.reshape(G, n_sims)[i].mean()),
                          injury_game=int(any(pid in out_now for pid in starters if pid)) ,
                          injured_starters=len(out_now & set(pd.read_sql_query(
                              "SELECT player_id FROM depth_charts WHERE season=? AND depth=1 AND team_id IN (?,?)",
                              conn, params=(s, g.home_id, g.away_id)).player_id)))
                for side, team in ((0, g.home_id), (1, g.away_id)):
                    for slot, stats in PLAYER_EVAL.items():
                        pid = est.slots[team].get(slot)
                        if pid is None:
                            continue
                        for st in stats:
                            x = ps[i, :, side, SLOT[slot], STAT[st]].astype(float)
                            prow.append(dict(game_id=g.game_id, season=s, week=w, player_id=pid, slot=slot, stat=st,
                                             mean=x.mean(), median=np.median(x), sd=x.std(), p10=np.quantile(x, 0.1),
                                             p25=np.quantile(x, 0.25), p75=np.quantile(x, 0.75), p90=np.quantile(x, 0.9),
                                             p_ge1=(x >= 1).mean()))
            log(f"  backtest {s} wk{w:>2}: {G} games  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    pp = pd.DataFrame(prow)
    df = attach_market(conn, df)
    df["margin"] = df.home_pts - df.away_pts
    df["total"] = df.home_pts + df.away_pts
    df["home_win"] = (df.margin > 0).astype(float)
    df = _walk_forward_baselines(df)
    test = df[df.season.isin(test_seasons) & df.sim_p.notna()].copy()
    # player actuals
    if not pp.empty:
        act = pd.read_sql_query("SELECT * FROM player_games", conn)
        act = act.melt(id_vars=["player_id", "game_id"], value_vars=sorted(set(pp.stat)), var_name="stat", value_name="actual")
        pp = pp.merge(act, on=["player_id", "game_id", "stat"], how="left")
        pp["actual"] = pp.actual.fillna(0.0)   # projected starter recorded no stat (or did not play)
    return {"games": df, "test": test, "players": pp, "elapsed_s": time.time() - t0}


def attach_market(conn, df: pd.DataFrame) -> pd.DataFrame:
    """Join Polymarket US closing prices (pre-kickoff) when the active DB holds real games that map to market events.

    Never attached to synthetic data (fictional teams cannot map to real markets)."""
    from ..db import is_synthetic
    for c in ("mkt_p_home", "mkt_margin_home", "mkt_total", "dk_p_home", "dk_margin_home", "dk_total"):
        df[c] = np.nan
    if is_synthetic(conn):
        return df
    # Sportsbook lines (ESPN pickcenter, usually DraftKings): no-vig moneyline probability, -spread, total
    bl = pd.read_sql_query("SELECT * FROM betting_lines", conn)
    if not bl.empty:
        bl = bl.sort_values("provider").groupby("game_id").head(1)

        def implied(ml):
            ml = pd.to_numeric(ml, errors="coerce")
            return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
        ph, pa = implied(bl.home_moneyline), implied(bl.away_moneyline)
        novig = ph / (ph + pa)
        from scipy.stats import norm as _n
        from_spread = _n.cdf(-pd.to_numeric(bl.spread_home, errors="coerce") / 15.5)
        bl["dk_p_home"] = np.where(np.isfinite(novig), novig, from_spread)
        bl["dk_margin_home"] = -pd.to_numeric(bl.spread_home, errors="coerce")
        bl["dk_total"] = pd.to_numeric(bl.over_under, errors="coerce")
        df = df.drop(columns=["dk_p_home", "dk_margin_home", "dk_total"]).merge(
            bl[["game_id", "dk_p_home", "dk_margin_home", "dk_total"]], on="game_id", how="left")
    try:
        from ..ingest.polymarket import market_for_games, markets_db_path, open_markets_db
        if not markets_db_path().exists():
            return df
        m = market_for_games(open_markets_db(), conn, kind="close")
        if m.empty:
            return df
        df = df.drop(columns=["mkt_p_home", "mkt_margin_home", "mkt_total"]).merge(
            m[["game_id", "mkt_p_home", "mkt_margin_home", "mkt_total"]], on="game_id", how="left")
    except Exception as e:  # market data is an optional benchmark; never break the backtest
        print(f"  market benchmark unavailable: {e}")
    return df


def _walk_forward_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    df["ridge_margin"] = np.nan
    df["ridge_p"] = np.nan
    df["xgb_margin"] = np.nan
    df["xgb_p"] = np.nan
    df["sim_cal_p"] = np.nan
    try:
        import xgboost as xgb  # optional
    except Exception:          # pragma: no cover
        xgb = None
    feat = ["elo_diff", "epa_diff", "off_h", "def_h", "off_a", "def_a", "sr_off_h", "sr_def_h", "sr_off_a", "sr_def_a"]
    have = [c for c in feat if c in df]
    df["home_ind"] = 1 - df.neutral
    keys = df[["season", "week"]].drop_duplicates().itertuples(index=False)
    for s, w in keys:
        tr = df[(df.season < s) | ((df.season == s) & (df.week < w))]
        te_idx = df.index[(df.season == s) & (df.week == w)]
        trf = tr.dropna(subset=["epa_diff"]) if "epa_diff" in tr else tr.iloc[0:0]
        if len(trf) >= 60:
            lr = LinearRegression().fit(trf[["epa_diff", "home_ind"]], trf.margin)
            resid_sd = float(np.std(trf.margin - lr.predict(trf[["epa_diff", "home_ind"]]))) or 15.0
            te = df.loc[te_idx].dropna(subset=["epa_diff"])
            if len(te):
                mh = lr.predict(te[["epa_diff", "home_ind"]])
                df.loc[te.index, "ridge_margin"] = mh
                df.loc[te.index, "ridge_p"] = norm.cdf(mh / resid_sd)
            if xgb is not None and len(trf) >= 150:
                model = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbosity=0)
                X = trf[have + ["home_ind"]].astype(float)
                model.fit(X, trf.margin)
                rs = float(np.std(trf.margin - model.predict(X))) * 1.15 or 15.0  # in-sample sd inflated
                te = df.loc[te_idx].dropna(subset=["epa_diff"])
                if len(te):
                    mx = model.predict(te[have + ["home_ind"]].astype(float))
                    df.loc[te.index, "xgb_margin"] = mx
                    df.loc[te.index, "xgb_p"] = norm.cdf(mx / rs)
        if "sim_p" in df:
            trs = tr.dropna(subset=["sim_p"])
            te = df.loc[te_idx].dropna(subset=["sim_p"]) if "sim_p" in df else None
            if te is not None and len(te):
                if len(trs) >= 80:
                    X = np.log(np.clip(trs.sim_p, 1e-3, 1 - 1e-3) / (1 - np.clip(trs.sim_p, 1e-3, 1 - 1e-3)))
                    c = LogisticRegression(C=10.0).fit(X.values.reshape(-1, 1), trs.home_win)
                    Xt = np.log(np.clip(te.sim_p, 1e-3, 1 - 1e-3) / (1 - np.clip(te.sim_p, 1e-3, 1 - 1e-3)))
                    df.loc[te.index, "sim_cal_p"] = c.predict_proba(Xt.values.reshape(-1, 1))[:, 1]
                else:
                    df.loc[te.index, "sim_cal_p"] = te.sim_p
    return df


# ------------------------------------------------------------------ scoring
def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_ci(x: np.ndarray, n=1000, seed=0, alpha=0.10):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def evaluate(bt: dict) -> tuple[list[dict], dict]:
    t = bt["test"].copy()
    y = t.home_win.values
    out = []

    def add(model, scope, metric, vals, detail=None):
        m, lo, hi = boot_ci(vals)
        out.append(dict(model_version=model, scope=scope, metric=metric, value=m, ci_low=lo, ci_high=hi,
                        n=int(np.sum(~np.isnan(np.asarray(vals, float)))), detail_json=json.dumps(detail) if detail else None))
    probs = {"sim_v1": t.sim_p, "sim_v1_cal": t.sim_cal_p, "elo": t.elo_p, "ridge_margin": t.ridge_p, "xgb_margin": t.xgb_p}
    margins = {"sim_v1": t.sim_margin, "elo": t.elo_margin, "ridge_margin": t.ridge_margin, "xgb_margin": t.xgb_margin}
    scopes = {"overall": np.ones(len(t), bool)}
    for s in sorted(t.season.unique()):
        scopes[f"season:{s}"] = (t.season == s).values
    scopes["conference_games"] = (t.conf_game == 1).values
    scopes["non_conference"] = (t.conf_game == 0).values
    scopes["injury_affected"] = (t.injured_starters > 0).values
    for name, p in probs.items():
        p = np.asarray(p, float)
        for sc, msk in scopes.items():
            ok = msk & ~np.isnan(p)
            if ok.sum() == 0:
                continue
            add(name, sc, "log_loss", log_loss(p[ok], y[ok]))
            add(name, sc, "brier", (p[ok] - y[ok]) ** 2)
            add(name, sc, "accuracy", ((p[ok] > 0.5) == (y[ok] == 1)).astype(float))
        # favourite / underdog perspective (calibration by side)
        ok = ~np.isnan(p)
        fav = np.where(p >= 0.5, p, 1 - p)
        fav_won = np.where(p >= 0.5, y, 1 - y)
        add(name, "favorite", "win_rate_minus_pred", (fav_won - fav)[ok])
    for name, m in margins.items():
        m = np.asarray(m, float)
        for sc, msk in scopes.items():
            ok = msk & ~np.isnan(m)
            if ok.sum():
                add(name, sc, "margin_mae", np.abs(m[ok] - t.margin.values[ok]))
                add(name, sc, "margin_rmse_sq", (m[ok] - t.margin.values[ok]) ** 2)
    add("sim_v1", "overall", "total_mae", np.abs(t.sim_total - t.total))
    add("sim_v1", "overall", "total_bias", (t.sim_total - t.total).values)
    add("sim_v1", "overall", "score_mae", np.concatenate([np.abs(t.sim_home - t.home_pts), np.abs(t.sim_away - t.away_pts)]))
    add("sim_v1", "overall", "margin_80_coverage", ((t.margin >= t.m_p10) & (t.margin <= t.m_p90)).astype(float).values)
    add("sim_v1", "overall", "total_80_coverage", ((t.total >= t.t_p10) & (t.total <= t.t_p90)).astype(float).values)
    # paired comparison vs Elo
    d = log_loss(t.sim_p.values, y) - log_loss(t.elo_p.values, y)
    add("sim_v1", "overall", "log_loss_minus_elo", d)
    dc = log_loss(t.sim_cal_p.values, y) - log_loss(t.elo_p.values, y)
    add("sim_v1_cal", "overall", "log_loss_minus_elo", dc)
    # market benchmark (Polymarket US closing prices) on the subset of games with a market
    if "mkt_p_home" in t and t.mkt_p_home.notna().sum() >= 20:
        mk = t[t.mkt_p_home.notna()]
        ym = mk.home_win.values
        add("polymarket_close", "overall", "log_loss", log_loss(mk.mkt_p_home.values, ym))
        add("polymarket_close", "overall", "brier", (mk.mkt_p_home.values - ym) ** 2)
        add("sim_v1_cal", "market_subset", "log_loss", log_loss(mk.sim_cal_p.values, ym))
        add("sim_v1_cal", "overall", "log_loss_minus_market", log_loss(mk.sim_cal_p.values, ym) - log_loss(mk.mkt_p_home.values, ym))
        if mk.mkt_margin_home.notna().sum() >= 20:
            q = mk[mk.mkt_margin_home.notna()]
            add("polymarket_close", "overall", "margin_mae", np.abs(q.mkt_margin_home - q.margin).values)
            add("sim_v1", "market_subset", "margin_mae", np.abs(q.sim_margin - q.margin).values)
        if mk.mkt_total.notna().sum() >= 20:
            q = mk[mk.mkt_total.notna()]
            add("polymarket_close", "overall", "total_mae", np.abs(q.mkt_total - q.total).values)
            add("sim_v1", "market_subset", "total_mae", np.abs(q.sim_total - q.total).values)
        # value test: buy the side the model prefers when |model - market| >= 5pp, at the market mid (no fees)
        e = mk.sim_cal_p - mk.mkt_p_home
        sel = e.abs() >= 0.05
        if sel.sum() >= 10:
            side_home = e[sel] > 0
            price = np.where(side_home, mk.mkt_p_home[sel], 1 - mk.mkt_p_home[sel])
            won = np.where(side_home, ym[sel.values], 1 - ym[sel.values])
            add("sim_v1_cal", "edge>=5pp", "return_per_contract_at_mid", (won - price) / price)
    if "dk_p_home" in t and t.dk_p_home.notna().sum() >= 20:
        mk = t[t.dk_p_home.notna()]
        ym = mk.home_win.values
        add("sportsbook_line", "overall", "log_loss", log_loss(mk.dk_p_home.values, ym))
        add("sportsbook_line", "overall", "brier", (mk.dk_p_home.values - ym) ** 2)
        add("sportsbook_line", "overall", "accuracy", ((mk.dk_p_home.values > .5) == (ym == 1)).astype(float))
        add("sim_v1_cal", "sportsbook_subset", "log_loss", log_loss(mk.sim_cal_p.values, ym))
        add("elo", "sportsbook_subset", "log_loss", log_loss(mk.elo_p.values, ym))
        add("sim_v1_cal", "overall", "log_loss_minus_sportsbook", log_loss(mk.sim_cal_p.values, ym) - log_loss(mk.dk_p_home.values, ym))
        q = mk[mk.dk_margin_home.notna()]
        if len(q) >= 20:
            add("sportsbook_line", "overall", "margin_mae", np.abs(q.dk_margin_home - q.margin).values)
            add("sim_v1", "sportsbook_subset", "margin_mae", np.abs(q.sim_margin - q.margin).values)
            ats = np.sign(q.sim_margin - q.dk_margin_home) == np.sign(q.margin - q.dk_margin_home)
            push = (q.margin == q.dk_margin_home)
            add("sim_v1", "vs_spread", "ats_win_rate_ex_push", ats[~push].astype(float).values)
        q = mk[mk.dk_total.notna()]
        if len(q) >= 20:
            add("sportsbook_line", "overall", "total_mae", np.abs(q.dk_total - q.total).values)
            add("sim_v1", "sportsbook_subset", "total_mae", np.abs(q.sim_total - q.total).values)
    # calibration curve
    cal = _calibration_curve(t.sim_cal_p.values, y)
    cal_raw = _calibration_curve(t.sim_p.values, y)
    out.append(dict(model_version="sim_v1_cal", scope="overall", metric="calibration_curve", value=cal["slope"],
                    ci_low=None, ci_high=None, n=len(t), detail_json=json.dumps(cal)))
    out.append(dict(model_version="sim_v1", scope="overall", metric="calibration_curve", value=cal_raw["slope"],
                    ci_low=None, ci_high=None, n=len(t), detail_json=json.dumps(cal_raw)))
    # player metrics
    pp = bt["players"]
    if not pp.empty:
        for st in ("pass_yds", "rush_yds", "rec_yds", "receptions", "tackles"):
            q = pp[pp.stat == st]
            if len(q):
                add("sim_v1", f"player:{st}", "mae", np.abs(q["mean"] - q.actual).values)
                add("sim_v1", f"player:{st}", "mae_median", np.abs(q["median"] - q.actual).values)
                add("sim_v1", f"player:{st}", "coverage_80", ((q.actual >= q.p10) & (q.actual <= q.p90)).astype(float).values)
        for st in ("pass_td", "rec_td", "rush_td", "pass_int"):
            q = pp[pp.stat == st]
            if len(q):
                yb = (q.actual >= 1).astype(float).values
                add("sim_v1", f"player:{st}", "brier_ge1", (q.p_ge1.values - yb) ** 2)
                c = _calibration_curve(q.p_ge1.values, yb, bins=8)
                out.append(dict(model_version="sim_v1", scope=f"player:{st}", metric="calibration_curve",
                                value=c["slope"], ci_low=None, ci_high=None, n=len(q), detail_json=json.dumps(c)))
    # promotion decision
    row = next(r for r in out if r["model_version"] == "sim_v1_cal" and r["metric"] == "log_loss_minus_elo")
    decision = {"candidate": "sim_v1", "benchmark": "elo", "metric": "log_loss_minus_elo (walk-forward calibrated)",
                "value": row["value"], "ci90": [row["ci_low"], row["ci_high"]],
                "promote": bool(row["ci_high"] is not None and row["ci_high"] < 0)}
    return out, decision


def _calibration_curve(p, y, bins=10):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    pts = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 5:
            pts.append({"pred": float(p[m].mean()), "obs": float(y[m].mean()), "n": int(m.sum())})
    if len(p) > 20 and len(np.unique(y)) > 1:
        lp = np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
        lr = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y)
        slope, icpt = float(lr.coef_[0][0]), float(lr.intercept_[0])
    else:
        slope = icpt = float("nan")
    return {"points": pts, "slope": slope, "intercept": icpt}


def persist(conn, bt: dict, metrics: list[dict], decision: dict) -> None:
    now = utcnow()
    t = bt["test"]
    span = f"{int(t.season.min())}-wk{int(t[t.season == t.season.min()].week.min())}", f"{int(t.season.max())}-wk{int(t[t.season == t.season.max()].week.max())}"
    conn.execute("DELETE FROM model_validation")
    conn.execute("DELETE FROM player_predictions WHERE prediction_id IN (SELECT prediction_id FROM model_predictions WHERE kind='backtest')")
    conn.execute("DELETE FROM model_predictions WHERE kind='backtest'")
    for mv, fam, status, notes in [
        (MODEL_VERSION, "sim_engine", "production" if decision["promote"] else "candidate",
         "Play-by-play Monte Carlo with opponent-adjusted, talent-prior parameters"),
        ("elo", "elo", "candidate" if decision["promote"] else "production", "Margin-aware Elo benchmark"),
        ("ridge_margin", "ridge_margin", "benchmark", "OLS margin on adj. net EPA diff + HFA"),
        ("xgb_margin", "xgb_margin", "benchmark", "XGBoost margin regressor on as-of features")]:
        conn.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?,?)",
                     (mv, fam, now, span[0], span[1], json.dumps({"backtest_sims": int(bt.get("n_sims", 0))}), status, notes))
    # production calibrator fitted on ALL backtest results (used for live forecasts only)
    X = np.log(np.clip(t.sim_p, 1e-3, 1 - 1e-3) / (1 - np.clip(t.sim_p, 1e-3, 1 - 1e-3))).values.reshape(-1, 1)
    c = LogisticRegression(C=10.0).fit(X, t.home_win)
    conn.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?,?)",
                 ("sim_calibrator_v1", "sim_calibrator", now, span[0], span[1],
                  json.dumps({"a": float(c.coef_[0][0]), "b": float(c.intercept_[0]), "n": int(len(t))}), "production",
                  "Platt scaling of simulated win probability, fit on walk-forward backtest"))
    insert_df(conn, "model_validation", pd.DataFrame([{**m, "created_at": now} for m in metrics]), replace=False)
    conn.execute("INSERT INTO model_validation (model_version, created_at, scope, metric, value, ci_low, ci_high, n, detail_json) VALUES (?,?,?,?,?,?,?,?,?)",
                 (MODEL_VERSION, now, "overall", "promotion_decision", float(decision["promote"]), decision["ci90"][0],
                  decision["ci90"][1], len(t), json.dumps(decision)))
    # stored predictions (all models, full distributions for sim)
    pp = bt["players"]
    for r in t.itertuples():
        for mv, p, mm in (("sim_v1", r.sim_p, r.sim_margin), ("sim_v1_cal", r.sim_cal_p, r.sim_margin),
                          ("elo", r.elo_p, r.elo_margin), ("ridge_margin", r.ridge_p, r.ridge_margin),
                          ("xgb_margin", r.xgb_p, r.xgb_margin)):
            if p is None or pd.isna(p):
                continue
            is_sim = mv.startswith("sim")
            cur = conn.execute("""INSERT INTO model_predictions (game_id, model_version, created_at, as_of, kind, home_win_prob,
                home_pts_mean, away_pts_mean, margin_mean, total_mean, margin_sd, total_sd, margin_p10, margin_p90,
                total_p10, total_p90, ot_prob, payload_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                               (int(r.game_id), mv, now, f"{r.season} before week {r.week}", "backtest", float(p),
                                r.sim_home if is_sim else None, r.sim_away if is_sim else None,
                                None if pd.isna(mm) else float(mm), r.sim_total if is_sim else None,
                                r.sim_margin_sd if is_sim else None, None, r.m_p10 if is_sim else None,
                                r.m_p90 if is_sim else None, r.t_p10 if is_sim else None, r.t_p90 if is_sim else None,
                                r.ot_p if is_sim else None, None))
            if mv == "sim_v1" and not pp.empty:
                q = pp[pp.game_id == r.game_id]
                conn.executemany("INSERT OR REPLACE INTO player_predictions VALUES (?,?,?,?,?,?,?,?,?,?)",
                                 [(cur.lastrowid, int(x.player_id), x.stat, x.mean, x.median, x.sd, x.p10, x.p25, x.p75, x.p90)
                                  for x in q.itertuples()])
    conn.commit()
