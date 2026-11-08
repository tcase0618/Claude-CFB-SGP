"""Live prop / same-game-parlay service: market-anchored simulation of upcoming games -> prop board + 3/5/8 cards.

    simulate_game(conn, game_id)        -> GameSims (cached), anchored to the current DraftKings line from ESPN
    prop_board(sims)                    -> every listed player market at the main line (fair odds both ways)
    game_card(conn, game_id)            -> board + tickets for 3/5/8 legs
    price_legs(conn, game_id, legs, ..) -> joint price of user-specified legs (e.g. a book SGP) incl. EV vs book odds
    week_cards(conn, season, week)      -> cards for every upcoming game of a week, written to data/cards/ and the
                                           sgp_tickets paper-trading ledger

Everything here is classification `predicted`. The book's SGP prices are not available from free data, so tickets
carry a fair price; a ticket is only a bet if the book pays more than that.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from ..analytics.team_params import estimate
from ..config import DATA_DIR
from ..db import utcnow
from ..sim.params import League, stack_params
from .props import Uncertainty, anchored_simulate, line_targets
from .sgp import (SLOT_MARKETS, STAT_LABEL, GameSims, Leg, _line_for_prob, build_card, candidate_legs, price, split_lanes,
                  prob_to_american)

MODEL_VERSION = "sgp_v1"
DEFAULT_SIMS = 6000


def load_prop_calibrator(conn) -> dict:
    r = conn.execute("""SELECT params_json FROM model_versions WHERE family='prop_calibrator'
                        ORDER BY CASE status WHEN 'production' THEN 0 ELSE 1 END, created_at DESC LIMIT 1""").fetchone()
    return json.loads(r[0]).get("factors", {}) if r else {}


def save_prop_calibrator(conn, factors: dict, span: tuple[str, str], n: int, note: str) -> None:
    conn.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?,?)",
                 ("prop_calibrator_v1", "prop_calibrator", utcnow(), span[0], span[1],
                  json.dumps({"factors": factors, "n": n}), "production", note))
    conn.commit()


def _game(conn, game_id: int) -> dict:
    g = pd.read_sql_query("""SELECT g.*, b.spread_home, b.over_under, b.fetched_at AS line_fetched_at FROM games g
        LEFT JOIN betting_lines b ON b.game_id=g.game_id AND b.provider='DraftKings' WHERE g.game_id=?""",
                          conn, params=(int(game_id),))
    if g.empty:
        raise KeyError(f"game {game_id} not found")
    return g.iloc[0].to_dict()


def _names(conn) -> dict:
    return pd.read_sql_query("SELECT team_id, school FROM teams", conn).set_index("team_id").school.to_dict()


EXCLUDE_FLAGS = {"missed_last", "missed_2of3"}     # 62-64% appearance walk-forward -> keep off auto tickets
WARN_FLAGS = {"left_early"}                       # 81% appearance -> warn only


def _availability(conn, season, week, g, slots, names_by_pid, outs) -> dict:
    """Availability block for a card: who was removed (official report / ESPN / manual) and doubts on starters."""
    from ..ingest.availability import availability_for
    removed = pd.read_sql_query("""SELECT a.player_id, p.first_name || ' ' || p.last_name AS name, a.team_id, a.status,
        a.reason, a.evidence, a.source FROM availability a LEFT JOIN players p ON p.player_id=a.player_id
        WHERE a.season=? AND a.week=? AND a.team_id IN (?,?) AND a.status IN ('OUT','DOUBTFUL')""",
                                conn, params=(season, week, int(g["home_id"]), int(g["away_id"])))
    pids = [int(p) for s in slots for k, p in s.items() if p is not None and k[:2] in ("QB", "RB", "WR", "TE")]
    av = availability_for(conn, season, week, pids)
    starters, exclude = [], set()
    for side, s in enumerate(slots):
        for k, p in s.items():
            if p is None or k[:2] not in ("QB", "RB", "WR", "TE"):
                continue
            items = av.get(int(p), [])
            flags = [i["reason"] for i in items if i["status"] == "FLAG"]
            status = next((i["status"] for i in items if i["status"] in ("QUESTIONABLE", "PROBABLE")), None)
            level = "exclude" if (set(flags) & EXCLUDE_FLAGS or status == "QUESTIONABLE") else \
                ("warn" if set(flags) & WARN_FLAGS else "ok")
            if level == "exclude":
                exclude.add(int(p))
            starters.append({"player_id": int(p), "name": names_by_pid.get(int(p)), "side": "home" if side == 0 else "away",
                             "slot": k, "status": status, "flags": flags, "evidence": [i["evidence"] for i in items],
                             "level": level})
    return {"removed": removed.to_dict("records"), "manual_out": sorted(outs), "starters": starters,
            "exclude": sorted(exclude),
            "note": "No open CFB injury feed: removals come from imported official reports, ESPN roster removal "
                    "(only when the player also missed his team's last game) and manual OUT marks. Verify before betting."}


def simulate_games(conn, games: list[dict], n_sims: int = DEFAULT_SIMS, seed: int = 31, chunk: int = 8,
                   unc: Uncertainty | None = None, outs: set | frozenset = frozenset()) -> list[tuple[dict, GameSims, dict]]:
    """Anchored, uncertainty-aware simulation for a list of games from the same season/week (chunked for memory).

    `outs`: extra player ids to treat as OUT (manual marks); their slot passes to the next man and a QB change moves
    the team's passing efficiency (forecast.apply_scenario)."""
    from .forecast import apply_scenario
    if not games:
        return []
    season, week = int(games[0]["season"]), int(games[0]["week"])
    est = estimate(conn, season, week)
    calib = load_prop_calibrator(conn)
    names = _names(conn)
    info = est.player_info
    league = League()
    out = []
    games = [g for g in games if g["home_id"] in est.params and g["away_id"] in est.params]
    for c0 in range(0, len(games), chunk):
        gs = games[c0:c0 + chunk]
        G = len(gs)
        pairs, slot_pairs = [], []
        for g in gs:
            hp, hs = est.params[g["home_id"]], est.slots[g["home_id"]]
            ap, as_ = est.params[g["away_id"]], est.slots[g["away_id"]]
            if outs:
                hp, hs = apply_scenario(est, g["home_id"], hp, hs, {"out": [p for p in outs if p in hs.values()]})
                ap, as_ = apply_scenario(est, g["away_id"], ap, as_, {"out": [p for p in outs if p in as_.values()]})
            pairs.append((hp, ap))
            slot_pairs.append((hs, as_))
        base = stack_params(pairs)
        P = {k: np.repeat(v, n_sims, axis=0) for k, v in base.items()}
        gl = np.repeat(np.arange(G), n_sims)
        neu = np.repeat(np.array([bool(g["neutral_site"]) for g in gs]), n_sims)
        tg = np.array([line_targets(g["spread_home"], g["over_under"])
                       if _ok(g.get("spread_home")) and _ok(g.get("over_under")) else (np.nan, np.nan) for g in gs])
        res, shifts = anchored_simulate(P, gl, tg, league, seed + int(gs[0]["game_id"]) % 9973, neutral=neu,
                                        unc=unc or Uncertainty())
        H = res.home_pts.reshape(G, n_sims)
        A = res.away_pts.reshape(G, n_sims)
        PS = res.player_stats.reshape(G, n_sims, *res.player_stats.shape[1:])
        for i, g in enumerate(gs):
            slots = slot_pairs[i]
            pn = {int(p): f"{info.loc[p].first_name} {info.loc[p].last_name}" for s in slots for p in s.values()
                  if p is not None and p in info.index}
            avail = _availability(conn, season, week, g, slots, pn, set(outs))
            sims = GameSims(H[i], A[i], PS[i], slots, (names.get(g["home_id"], "Home"), names.get(g["away_id"], "Away")),
                            pn, calib=calib, seed=int(g["game_id"]) % 100000)
            meta = {"anchored": bool(not np.isnan(tg[i]).any()), "anchor_shift": shifts[i].round(3).tolist(),
                    "line": {"spread_home": _f(g.get("spread_home")), "total": _f(g.get("over_under")),
                             "source": "DraftKings via ESPN", "fetched_at": g.get("line_fetched_at")},
                    "calibration": calib, "as_of": f"{season} before week {week}", "n_sims": n_sims,
                    "model_version": MODEL_VERSION, "classification": "predicted", "availability": avail}
            out.append((g, sims, meta))
    return out


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def _f(v):
    return float(v) if _ok(v) else None


@lru_cache(maxsize=32)
def _cached(db_key: str, game_id: int, n_sims: int, line_key: str, outs: frozenset, avail_key: str):
    from ..db import connect
    conn = connect(db_key)
    g = _game(conn, game_id)
    return simulate_games(conn, [g], n_sims, outs=outs)[0]


def simulate_game(conn, game_id: int, n_sims: int = DEFAULT_SIMS, outs=()):
    from ..analytics.team_params import conn_path
    g = _game(conn, game_id)
    # availability changes (snapshot / import) invalidate the cached simulation
    ak = conn.execute("SELECT COUNT(*), MAX(fetched_at) FROM availability WHERE season=? AND week=?",
                      (int(g["season"]), int(g["week"]))).fetchone()
    return _cached(conn_path(conn), int(game_id), int(n_sims), f"{g.get('spread_home')}|{g.get('over_under')}",
                   frozenset(int(o) for o in outs), f"{ak[0]}|{ak[1]}")


def prop_board(sims: GameSims) -> list[dict]:
    """Main-line (median) projection for every listed market of every projected starter."""
    rows = []
    for pid, (side, slot) in sims.where.items():
        for st in SLOT_MARKETS.get(slot, []):
            v = sims.stat(pid, st)
            if v.mean() < 0.1:
                continue
            line = 0.5 if st == "td" else _line_for_prob(v, 0.5)
            po = float((v > line).mean())
            rows.append({"player_id": pid, "name": sims.player_names.get(pid), "team": sims.team_names[side],
                         "side": "home" if side == 0 else "away", "slot": slot, "stat": st,
                         "market": STAT_LABEL.get(st, st), "mean": float(v.mean()), "median": float(np.median(v)),
                         "p10": float(np.quantile(v, 0.1)), "p90": float(np.quantile(v, 0.9)), "line": line,
                         "p_over": po, "fair_over": prob_to_american(po), "fair_under": prob_to_american(1 - po)})
    return rows


def polymarket_book(conn, game_id: int) -> dict | None:
    """Latest Polymarket US book for the game (live quotes, else the stored closing book); None if not listed."""
    from ..ingest.polymarket import open_markets_db
    from .market_sgp import market_book
    try:
        mc = open_markets_db()
    except Exception:
        return None
    for kind in ("live", "close"):
        b = market_book(mc, conn, game_id=game_id, kind=kind).get(int(game_id))
        if b:
            return b | {"kind": kind}
    return None


def _market_summary(sims: GameSims, book: dict | None, legs: list[Leg]) -> list[dict]:
    """Team legs where Polymarket prices the same bet: model vs market probability."""
    from .market_sgp import market_leg_prob
    rows = []
    for l in legs:
        pm, src = market_leg_prob(l, book)
        if pm is None:
            continue
        pmod = float(sims.hits(l).mean())
        rows.append(dict(l.to_dict(), p_model=pmod, p_market=pm, market_source=src, edge_model_vs_market=pmod - pm))
    return rows


def _attach_market(card: dict, sims: GameSims, book: dict | None) -> None:
    """Add the Polymarket-anchored price to every ticket (team legs at market marginals, simulated dependence)."""
    from .market_sgp import price_market
    for tickets in card["tickets"].values():
        for t in tickets:
            lanes = split_lanes(sims.n, t["holdout_seed"])[1] if "holdout_seed" in t else None
            mp = price_market(sims, [Leg.from_dict(d) for d in t["legs"]], book, lanes=lanes)
            t["market"] = {k: mp[k] for k in ("p_joint", "fair_american", "n_market_legs", "p_indep", "lift")}
            # the price to beat: the more conservative of model and market-anchored
            t["fair_american_conservative"] = prob_to_american(min(t["p_joint"], mp["p_joint"]))


def game_card(conn, game_id: int, n_sims: int = DEFAULT_SIMS, sizes=(3, 5, 8), top: int = 2, outs=()) -> dict:
    from .market_sgp import market_lines
    g, sims, meta = simulate_game(conn, game_id, n_sims, outs)
    market = {"spread_home": meta["line"]["spread_home"], "total": meta["line"]["total"]}
    card = build_card(sims, market if meta["anchored"] else None, sizes=sizes, top=top, seed=int(game_id) % 1000,
                      exclude=set(meta["availability"]["exclude"]))
    book = polymarket_book(conn, game_id)
    _attach_market(card, sims, book)
    team_legs = [l for l in candidate_legs(sims, market if meta["anchored"] else None) if l.kind != "player"]
    h, a = sims.h, sims.a
    return {"game_id": int(game_id), "home": sims.team_names[0], "away": sims.team_names[1], "meta": meta,
            "game": {"home_pts": float(h.mean()), "away_pts": float(a.mean()), "home_win": float((h > a).mean()),
                     "margin_sd": float((h - a).std()), "total_sd": float((h + a).std())},
            "polymarket": {"listed": book is not None, "kind": book.get("kind") if book else None,
                           "lines": market_lines(book), "team_legs": _market_summary(sims, book, team_legs)},
            "props": prop_board(sims), **card}


def price_legs(conn, game_id: int, legs: list[dict], book_american: float | None = None,
               n_sims: int = DEFAULT_SIMS, outs=()) -> dict:
    g, sims, meta = simulate_game(conn, game_id, n_sims, outs)
    L = [Leg.from_dict(d) for d in legs]
    for l in L:
        if l.kind == "player" and int(l.player_id) not in sims.where:
            raise ValueError(f"player {l.player_id} is not a projected starter in this game")
    out = price(sims, L, book_american=book_american)
    out["meta"] = meta
    from .market_sgp import price_market
    book = polymarket_book(conn, game_id)
    mk = price_market(sims, L, book, book_american=book_american)
    mk["listed"] = book is not None
    mk["kind"] = book.get("kind") if book else None
    out["market"] = mk
    if book_american is not None:
        # fire only when the book beats BOTH prices: the pure model and the market-anchored one
        evs = [out["ev_per_unit"]] + ([mk["ev_per_unit"]] if mk["n_market_legs"] else [])
        out["ev_conservative"] = min(evs)
        out["verdict"] = "bet" if min(evs) > 0 else "pass"
    return out


def week_cards(conn, season: int, week: int, n_sims: int = DEFAULT_SIMS, log=print) -> dict:
    gs = pd.read_sql_query("""SELECT g.*, b.spread_home, b.over_under, b.fetched_at AS line_fetched_at FROM games g
        LEFT JOIN betting_lines b ON b.game_id=g.game_id AND b.provider='DraftKings'
        WHERE g.season=? AND g.week=? AND g.completed=0 ORDER BY g.start_date""", conn, params=(season, week))
    games = gs.to_dict("records")
    cards = []
    now = utcnow()
    for g, sims, meta in simulate_games(conn, games, n_sims):
        market = {"spread_home": meta["line"]["spread_home"], "total": meta["line"]["total"]}
        card = build_card(sims, market if meta["anchored"] else None, seed=int(g["game_id"]) % 1000,
                          exclude=set(meta["availability"]["exclude"]))
        _attach_market(card, sims, polymarket_book(conn, int(g["game_id"])))
        c = {"game_id": int(g["game_id"]), "start": g["start_date"], "home": sims.team_names[0],
             "away": sims.team_names[1], "meta": meta, **card}
        cards.append(c)
        log(f"  {c['away']} @ {c['home']}: " + ", ".join(f"{k}-leg x{len(v)}" for k, v in card["tickets"].items())
            + ("" if meta["anchored"] else "  [NO LINE: unanchored, not logged]"))
        # ledger: only market-anchored games, and a re-run supersedes that game's still-open tickets
        conn.execute("DELETE FROM sgp_tickets WHERE game_id=? AND status='open'", (int(g["game_id"]),))
        if not meta["anchored"]:
            continue
        for size, tickets in card["tickets"].items():
            for t in tickets:
                conn.execute("""INSERT INTO sgp_tickets (game_id, created_at, model_version, n_legs, legs_json, p_joint,
                    p_indep, fair_american, status) VALUES (?,?,?,?,?,?,?,?, 'open')""",
                             (int(g["game_id"]), now, MODEL_VERSION, int(size), json.dumps(t["legs"]), t["p_joint"],
                              t["p_indep"], t["fair_american"]))
    conn.commit()
    out = DATA_DIR / "cards"
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{season}_wk{week:02d}.json"
    fp.write_text(json.dumps({"season": season, "week": week, "generated_at": now, "cards": cards}, default=float))
    return {"path": str(fp), "games": len(cards)}


def settle_tickets(conn, log=print) -> dict:
    """Grade open ledger tickets whose games are final (void if a player leg's player did not appear)."""
    from .sgp_backtest import _actual_stats, _leg_actual
    t = pd.read_sql_query("SELECT * FROM sgp_tickets WHERE status='open'", conn)
    if t.empty:
        return {"settled": 0}
    g = pd.read_sql_query(f"SELECT * FROM games WHERE completed=1 AND game_id IN ({','.join(map(str, t.game_id.unique()))})", conn)
    if g.empty:
        return {"settled": 0}
    act = _actual_stats(conn, g.game_id)
    gi = g.set_index("game_id")
    n = 0
    for r in t[t.game_id.isin(gi.index)].itertuples():
        legs = json.loads(r.legs_json)
        game = gi.loc[r.game_id]
        game = game.copy()
        game["game_id"] = r.game_id
        outs = [_leg_actual(l, game, act) for l in legs]
        status = "void" if any(o is None for o in outs) else ("won" if all(outs) else "lost")
        conn.execute("UPDATE sgp_tickets SET status=?, result=?, settled_at=? WHERE ticket_id=?",
                     (status, None if status == "void" else float(status == "won"), utcnow(), r.ticket_id))
        n += 1
    conn.commit()
    log(f"settled {n} tickets")
    return {"settled": n}
