"""Market-anchored same-game-parlay pricing with Polymarket US.

What Polymarket can and cannot do for an SGP
--------------------------------------------
Polymarket US lists CFB moneylines, spread and total ladders, and team-points ladders, all with tradeable books at
kickoff. It lists **no player props and no parlays**, and its team stat props (yards, TDs, field goals) have empty
books. So it cannot tell us what a sportsbook charges for an SGP. What it can do:

  1. **Sharp, no-vig marginals for every team leg.** A Polymarket mid is an exchange price with no bookmaker margin
     built in. Where it exists it replaces the simulator's marginal for the ML / spread / total / team-total legs.
  2. **A market-anchored fair price.** The joint is computed by quantile mapping (a copula): each team leg's
     threshold in the simulation is moved so the leg hits at exactly the market probability, then the joint is
     counted lane by lane. The simulator supplies only the *dependence*; the market supplies the marginals. The
     simulated correlation is shrunk toward independence with the same walk-forward alpha as the model price.
  3. **A real-money test of the correlation.** On 2026 games we can check whether "market marginals × simulated
     dependence" predicts team-leg combos better than "market marginals × independence" (`backtest_market_combos`).

Player legs keep the model marginal (there is no market for them), so the market-anchored price is still a
*fair* price: compare it with the book's SGP price and bet only when the book pays more.

Orientation: every quantity is converted to the model's home team via `map_events_to_games.pm_home_is_home`.
Polymarket rung semantics: ML YES = long team wins; spread YES <=> margin(long) + line > 0; totals and team
totals YES = over.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..ingest.polymarket import MONEYLINE, SPREAD, TEAM_POINTS, TOTAL, fit_normal_ladder, map_events_to_games
from .sgp import GameSims, Leg, american_to_decimal, prob_to_american

EXACT_MAX_WIDTH = 0.12      # a rung's own mid is used only when its book is at most 12c wide
FIT_MAX_WIDTH = 0.30        # rungs wider than this carry no information (same cut as the implied-line fit)
TEAM_LEGS = ("ml", "spread", "total", "team_total")


# ----------------------------------------------------------------------------- loading
def market_book(mconn, gconn, game_id: int | None = None, kind: str = "live", game_ids=None) -> dict[int, dict]:
    """Latest Polymarket quotes for model games, oriented to the model's home team.

    Returns {game_id: {"event_id", "ts", "ml": (p_home, width) | None, "spread": DataFrame[L_home, p, width],
    "total": DataFrame[line, p, width], "team_total": {"home": DataFrame, "away": DataFrame}}}.
    spread rows are P(margin_home + L_home > 0); total / team-total rows are P(points > line).
    """
    mp = map_events_to_games(mconn, gconn)
    if game_id is not None:
        mp = mp[mp.game_id == int(game_id)]
    if game_ids is not None:
        mp = mp[mp.game_id.isin([int(g) for g in game_ids])]
    if mp.empty:
        return {}
    ev_ids = ",".join(f"'{e}'" for e in mp.event_id)
    q = pd.read_sql_query(f"""SELECT q.market_slug, q.ts, q.bid, q.ask, q.mid, m.event_id, m.sports_market_type AS typ,
        m.line, m.long_pm_team_id, e.home_pm_team_id, e.away_pm_team_id FROM pm_quotes q
        JOIN pm_markets m USING(market_slug) JOIN pm_events e ON e.event_id=m.event_id
        WHERE q.kind=? AND m.event_id IN ({ev_ids}) AND m.sports_market_type IN (?,?,?,?)""",
                          mconn, params=(kind, MONEYLINE, SPREAD, TOTAL, TEAM_POINTS))
    if q.empty:
        return {}
    q = q.sort_values("ts").groupby("market_slug").tail(1)
    q["width"] = (q.ask - q.bid).abs()
    q = q[q.mid.notna() & (q.width <= FIT_MAX_WIDTH)]
    out = {}
    for r in mp.itertuples():
        g = q[q.event_id == r.event_id]
        if g.empty:
            continue
        # Polymarket's "home" (2nd listed) team id, and the pm team id of the model's home team
        pm_home = g.home_pm_team_id.iloc[0]
        pm_away = g.away_pm_team_id.iloc[0]
        model_home_pm = pm_home if r.pm_home_is_home else pm_away
        book = {"event_id": r.event_id, "ts": g.ts.max(), "ml": None}
        ml = g[g.typ == MONEYLINE]
        if len(ml):
            m = ml.iloc[0]
            p = float(m.mid) if m.long_pm_team_id == model_home_pm else 1 - float(m.mid)
            book["ml"] = (p, float(m.width))
        sp = g[(g.typ == SPREAD) & g.line.notna()]
        is_h = (sp.long_pm_team_id == model_home_pm).values
        book["spread"] = pd.DataFrame({"L_home": np.where(is_h, sp.line, -sp.line),
                                       "p": np.where(is_h, sp.mid, 1 - sp.mid), "width": sp.width.values})
        tt = g[(g.typ == TOTAL) & g.line.notna()]
        book["total"] = pd.DataFrame({"line": tt.line.values, "p": tt.mid.values, "width": tt.width.values})
        book["team_total"] = {}
        for side, pid in (("home", model_home_pm), ("away", pm_away if r.pm_home_is_home else pm_home)):
            t = g[(g.typ == TEAM_POINTS) & (g.long_pm_team_id == pid) & g.line.notna()]
            book["team_total"][side] = pd.DataFrame({"line": t.line.values, "p": t.mid.values, "width": t.width.values})
        out[int(r.game_id)] = book
    return out


# ----------------------------------------------------------------------------- leg probabilities
def _ladder_prob(df: pd.DataFrame, x: float, kind: str) -> tuple[float | None, str | None]:
    """P(YES) at line x: the rung's own mid when quoted tight, else the normal ladder fit through all rungs."""
    if df is None or df.empty:
        return None, None
    col = "L_home" if kind == "spread" else "line"
    ex = df[np.isclose(df[col], x) & (df.width <= EXACT_MAX_WIDTH)]
    if len(ex):
        return float(ex.p.iloc[0]), "exact"
    f = fit_normal_ladder(df[col].values, df.p.values, "spread" if kind == "spread" else "total")
    if f["mean"] is None:
        return None, None
    # do not extrapolate far outside the quoted rungs
    lo, hi = df[col].min(), df[col].max()
    if x < lo - 3 or x > hi + 3:
        return None, None
    p = norm.cdf((f["mean"] + x) / f["sd"]) if kind == "spread" else norm.cdf((f["mean"] - x) / f["sd"])
    return float(p), "ladder_fit"


def market_leg_prob(leg: Leg, book: dict | None) -> tuple[float | None, str | None]:
    """Polymarket probability that `leg` hits, or (None, None) if the market does not price it."""
    if not book or leg.kind not in TEAM_LEGS:
        return None, None
    if leg.kind == "ml":
        if book.get("ml") is None or book["ml"][1] > EXACT_MAX_WIDTH:
            # a wide ML book: fall back to the spread ladder at 0 (P(margin_home > 0))
            p, src = _ladder_prob(book.get("spread"), 0.0, "spread")
            if p is None:
                return None, None
            return (p if leg.side == "home" else 1 - p), src
        p = book["ml"][0]
        return (p if leg.side == "home" else 1 - p), "exact"
    if leg.kind == "spread":
        # home covers l  <=> margin_home + l > 0 ;  away covers l <=> margin_home - l < 0 = 1 - P(margin_home + (-l) > 0)
        if leg.side == "home":
            return _ladder_prob(book.get("spread"), leg.line, "spread")
        p, src = _ladder_prob(book.get("spread"), -leg.line, "spread")
        return (None, None) if p is None else (1 - p, src)
    df = book.get("total") if leg.kind == "total" else book.get("team_total", {}).get(leg.side)
    p, src = _ladder_prob(df, leg.line, "total")
    if p is None:
        return None, None
    return (p if leg.op == "over" else 1 - p), src


# ----------------------------------------------------------------------------- pricing
def _matched_hits(sims: GameSims, leg: Leg, p: float, rng: np.random.Generator) -> np.ndarray:
    """Hit indicator with the leg's threshold moved so it hits on exactly a share `p` of lanes.

    Integer scores are jittered by U(-0.5, 0.5) (a continuity correction, drawn once per quantity so legs on the
    same quantity stay consistent). This is a quantile-mapping copula: the market sets the marginal, the simulated
    lanes keep their rank order, so the dependence between legs is the simulator's."""
    key = ("_jit", leg.kind, leg.side if leg.kind != "total" else None)
    if leg.kind in ("ml", "spread"):
        key = ("_jit", "margin")
    if key not in sims._cache:
        base = (sims.h - sims.a) if key[1] == "margin" else (sims.h + sims.a if leg.kind == "total"
                                                               else (sims.h if leg.side == "home" else sims.a))
        sims._cache[key] = base + rng.uniform(-0.5, 0.5, sims.n)
    v = sims._cache[key]
    if key[1] == "margin" and leg.side == "away":
        v = -v
    under = leg.kind in ("total", "team_total") and leg.op == "under"
    if under:
        return v < np.quantile(v, p)
    return v > np.quantile(v, 1 - p)


def price_market(sims: GameSims, legs: list[Leg], book: dict | None, book_american: float | None = None,
                 alpha: float | None = None, lanes: np.ndarray | None = None, seed: int = 7) -> dict:
    """Market-anchored fair price: Polymarket marginals for team legs, model marginals for player legs, joint by
    quantile mapping onto the simulated lanes, correlation shrunk by alpha (default: the calibrator's)."""
    rng = np.random.default_rng(seed)
    a = sims.corr_shrink if alpha is None else alpha
    cols, info = [], []
    for l in legs:
        pm, src = market_leg_prob(l, book)
        if pm is not None:
            pm = float(np.clip(pm, 0.01, 0.99))
            h = _matched_hits(sims, l, pm, rng)
        else:
            h = sims.hits(l)
        cols.append(h)
        info.append((pm, src, float(sims.hits(l).mean())))
    X = np.stack(cols, axis=1)
    if lanes is not None:
        X = X[lanes]
    n = X.shape[0]
    # marginals: the market's where it prices the leg (exact by construction, up to ties), else the simulated share
    marg = np.array([pm if pm is not None else float(X[:, i].mean()) for i, (pm, _, _) in enumerate(info)])
    pj_raw = float(X.all(axis=1).mean())
    pi_sim = float(np.prod(X.mean(axis=0)))
    pi = float(np.prod(marg))
    lift_raw = pj_raw / pi_sim if pi_sim > 0 else 1.0
    lift = lift_raw ** a if lift_raw > 0 else 0.0
    pj = float(min(pi * lift, marg.min())) if lift > 0 else 0.0
    se = math.sqrt(max(pj * (1 - pj), 1e-9) / n)
    n_mkt = sum(pm is not None for pm, _, _ in info)
    out = {"legs": [dict(l.to_dict(), p=float(m), p_model=pmod, p_market=pm, market_source=src,
                         fair_american=prob_to_american(float(m)))
                    for l, m, (pm, src, pmod) in zip(legs, marg, info)],
           "n_legs": len(legs), "n_market_legs": n_mkt, "p_joint": pj,
           "p_joint_ci90": [max(0.0, pj - 1.645 * se), min(1.0, pj + 1.645 * se)],
           "p_indep": pi, "lift": pj / pi if pi > 0 else None, "lift_raw": lift_raw, "corr_shrink": a,
           "fair_american": prob_to_american(pj), "indep_american": prob_to_american(pi),
           "market_ts": book.get("ts") if book else None, "source": "Polymarket US mids + simulated dependence",
           "classification": "calculated" if n_mkt == len(legs) else "predicted"}
    if book_american is not None:
        dec = american_to_decimal(book_american)
        ev = pj * dec - 1
        out.update(book_american=float(book_american), book_implied=1 / dec, ev_per_unit=ev,
                   kelly=max(0.0, ev / (dec - 1)), quarter_kelly=max(0.0, ev / (dec - 1)) / 4)
    return out


def market_lines(book: dict | None) -> dict | None:
    """Summary for display: no-vig ML, the 50/50 spread and total, team-total centres."""
    if not book:
        return None
    out = {"ts": book.get("ts"), "p_home_ml": book["ml"][0] if book.get("ml") else None,
           "ml_width": book["ml"][1] if book.get("ml") else None}
    for k, df, kind in (("spread_home", book.get("spread"), "spread"), ("total", book.get("total"), "total"),
                        ("team_total_home", book.get("team_total", {}).get("home"), "total"),
                        ("team_total_away", book.get("team_total", {}).get("away"), "total")):
        if df is None or df.empty:
            out[k] = None
            continue
        col = "L_home" if kind == "spread" else "line"
        f = fit_normal_ladder(df[col].values, df.p.values, kind)
        if f["mean"] is None:
            out[k] = None
        else:
            # spread: report the home handicap at a coin flip (-mean), like a sportsbook line
            out[k] = round(-f["mean"] if kind == "spread" else f["mean"], 2)
        out[k + "_rungs"] = int(len(df))
    return out


# ----------------------------------------------------------------------------- validation on real games
def _fair_rung(df: pd.DataFrame, col: str) -> float | None:
    """The quoted rung closest to a coin flip (tight books only)."""
    d = df[df.width <= EXACT_MAX_WIDTH]
    if d.empty:
        return None
    return float(d.iloc[(d.p - 0.5).abs().argsort()].iloc[0][col])


def combo_legs(book: dict) -> list[list[Leg]]:
    """Two-leg team combos a book would offer as an SGP: side x total, team total x opposing side, team total x
    game total. Lines are the market's own coin-flip rungs so every leg is priced exactly, not from a fit."""
    out = []
    sl = _fair_rung(book["spread"], "L_home") if len(book.get("spread", [])) else None
    tl = _fair_rung(book["total"], "line") if len(book.get("total", [])) else None
    sides = []
    if sl is not None:
        sides += [Leg("spread", side="home", line=sl), Leg("spread", side="away", line=-sl)]
    if book.get("ml") and book["ml"][1] <= EXACT_MAX_WIDTH:
        sides += [Leg("ml", side="home"), Leg("ml", side="away")]
    totals = [Leg("total", op=o, line=tl) for o in ("over", "under")] if tl is not None else []
    tts = []
    for side in ("home", "away"):
        df = book.get("team_total", {}).get(side)
        if df is not None and len(df):
            ln = _fair_rung(df, "line")
            if ln is not None:
                tts += [Leg("team_total", side=side, op=o, line=ln) for o in ("over", "under")]
    for s in sides:
        for t in totals:
            out.append([s, t])
    for tt in tts:
        for s in sides:
            if s.kind == "spread":
                out.append([tt, s])
        for t in totals:
            out.append([tt, t])
    return out


def _actual(leg: Leg, h: float, a: float) -> bool:
    sgn = 1 if leg.side == "home" else -1
    if leg.kind == "ml":
        return sgn * (h - a) > 0
    if leg.kind == "spread":
        return sgn * (h - a) + leg.line > 0
    v = h + a if leg.kind == "total" else (h if leg.side == "home" else a)
    return v > leg.line if leg.op == "over" else v < leg.line


def backtest_market_combos(conn, mconn, season: int, weeks: list[int], n_sims: int = 2000, seed: int = 5,
                           alpha: float = 0.8, log=print) -> pd.DataFrame:
    """Score 2-leg team combos on completed games with a Polymarket closing book.

    Three prices per combo: market independence (product of Polymarket mids), market x simulated dependence
    (`price_market`), and the pure model (DraftKings-anchored simulation, same alpha). The simulation is
    estimated strictly before each week; Polymarket closing mids are the last book sample before kickoff."""
    from ..analytics.team_params import _estimate
    from ..sim.params import League, stack_params
    from .props import Uncertainty, anchored_simulate, line_targets
    games = pd.read_sql_query("""SELECT g.*, b.spread_home, b.over_under FROM games g
        JOIN betting_lines b ON b.game_id=g.game_id AND b.provider='DraftKings'
        WHERE g.season=? AND g.completed=1 AND b.spread_home IS NOT NULL AND b.over_under IS NOT NULL
        ORDER BY g.week, g.game_id""", conn, params=(season,))
    books = market_book(mconn, conn, kind="close", game_ids=games.game_id.tolist())
    games = games[games.game_id.isin(books)]
    rows = []
    league = League()
    for w in weeks:
        gw = games[games.week == w]
        if gw.empty:
            continue
        est = _estimate(conn, season, w, None)
        gw = gw[gw.home_id.isin(est.params) & gw.away_id.isin(est.params)].reset_index(drop=True)
        G = len(gw)
        if not G:
            continue
        base = stack_params([(est.params[h], est.params[a]) for h, a in zip(gw.home_id, gw.away_id)])
        P = {k: np.repeat(v, n_sims, axis=0) for k, v in base.items()}
        gl = np.repeat(np.arange(G), n_sims)
        neu = np.repeat(gw.neutral_site.values.astype(bool), n_sims)
        tg = np.array([line_targets(s, o) for s, o in zip(gw.spread_home, gw.over_under)])
        res, _ = anchored_simulate(P, gl, tg, league, seed + season * 100 + w, neutral=neu, unc=Uncertainty(),
                                   track_players=False)
        H = res.home_pts.reshape(G, n_sims)
        A = res.away_pts.reshape(G, n_sims)
        for i, g in enumerate(gw.itertuples()):
            book = books[int(g.game_id)]
            sims = GameSims(H[i], A[i], None, ({}, {}), calib={"_corr_shrink": alpha}, seed=int(g.game_id) % 9973)
            for legs in combo_legs(book):
                pm = price_market(sims, legs, book, alpha=alpha)
                if pm["n_market_legs"] < len(legs):
                    continue
                X = np.stack([sims.hits(l) for l in legs], axis=1)
                marg = X.mean(axis=0)
                pj_raw, pi_mod = float(X.all(axis=1).mean()), float(np.prod(marg))
                p_model = min(pi_mod * (pj_raw / pi_mod) ** alpha, marg.min()) if pi_mod > 0 and pj_raw > 0 else pj_raw
                hit = float(all(_actual(l, float(g.home_points), float(g.away_points)) for l in legs))
                rows.append(dict(game_id=int(g.game_id), week=w, combo=" + ".join(_kind(l) for l in legs),
                                 labels=" | ".join(l.label() for l in legs), hit=hit, p_mkt_indep=pm["p_indep"],
                                 p_mkt_copula=pm["p_joint"], p_model=p_model, lift_raw=pm["lift_raw"],
                                 leg_mkt=[x["p_market"] for x in pm["legs"]], leg_model=[x["p_model"] for x in pm["legs"]]))
        log(f"  market combos {season} wk{w:>2}: {G} games, {len(rows)} combos so far")
    return pd.DataFrame(rows)


def _kind(l: Leg) -> str:
    if l.kind in ("ml", "spread"):
        return "side"
    return l.kind


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summarize_market_combos(df: pd.DataFrame) -> dict:
    """Log loss and calibration of the three prices, overall and by combo family. Combos within a game are
    dependent (4 orientations of one pair), so SEs use game-clustered resampling."""
    if df.empty:
        return {}
    out = {}
    for fam, d in [("all", df)] + list(df.groupby("combo")):
        y = d.hit.values
        r = {"n": int(len(d)), "games": int(d.game_id.nunique()), "hit_rate": float(y.mean())}
        for c in ("p_mkt_indep", "p_mkt_copula", "p_model"):
            r[c] = {"mean_p": float(d[c].mean()), "log_loss": _ll(d[c], y)}
        # game-clustered bootstrap of the log-loss gain of the copula over market independence
        gids = d.game_id.unique()
        rng = np.random.default_rng(0)
        li = -(y * np.log(np.clip(d.p_mkt_indep, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - d.p_mkt_indep, 1e-4, 1)))
        lc = -(y * np.log(np.clip(d.p_mkt_copula, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - d.p_mkt_copula, 1e-4, 1)))
        gain = pd.Series((li - lc).values, index=d.game_id.values).groupby(level=0).sum()
        boots = [gain.loc[rng.choice(gids, len(gids))].sum() / len(d) for _ in range(500)]
        r["copula_gain_vs_indep"] = {"mean": float(gain.sum() / len(d)), "se": float(np.std(boots))}
        out[fam] = r
    return out
