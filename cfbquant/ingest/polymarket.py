"""Polymarket US public market data (free, no API key) — https://docs.polymarket.us

Public gateway: https://gateway.polymarket.us  (no auth; documented limit 20 req/s/IP, enforced
aggressively in practice, so this client defaults to ~3 req/s with exponential backoff on 429).

What we use (CFB league slug `cfb`, sport `football`):
  GET /v2/leagues/cfb                       -> activeSeriesId
  GET /v1/events?seriesId=..&closed=..      -> games with teams, scores, state, and ALL markets
                                               (each market carries `outcomePrices` = [best bid, best ask])
  GET /v1/markets/{slug}/bbo                -> best bid/offer, last, volume, open interest
  GET /v1/price-history?symbol=..           -> book-derived history (longPrice = ask-based YES,
                                               shortPrice = 1 - bid); custom windows give ~1-min points
  GET /v1/markets/{slug}/settlement         -> 1 (YES) / 0 (NO)

Contract semantics — VERIFIED against settlements (UNC 15, TCU 10, 2026-08-29), not inferred from titles:
  moneyline  (football_team_full_game_winner):  YES <=> the `long: true` side's team wins
  spread     (football_team_full_game_spread):  YES <=> margin(long team) + line > 0
             (the human `title` contradicts this for positive lines — never parse titles)
  total      (football_team_full_game_total):   YES <=> total points > line  (long side = "Over")
  team props (e.g. football_team_total_pass_yards): YES <=> that team's stat > line

Home/away: Polymarket lists CFB teams away-first (league `ordering: "away"`) but has no explicit flag.
When a market event maps to a game in our database, home/away comes from our game record instead.

This module only reads public market data. It never places orders and needs no credentials.
Market data is stored in its own database (CFBQ_MARKETS_DB, default data/polymarket.db) so real market
prices are never written into the synthetic demo database.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

from ..db import connect, insert_df, utcnow

GATEWAY = "https://gateway.polymarket.us"
SOURCE = "polymarket_us"
MONEYLINE, SPREAD, TOTAL = "football_team_full_game_winner", "football_team_full_game_spread", "football_team_full_game_total"
PROP_TYPES = {"football_team_total_pass_yards": "pass_yds", "football_team_total_rush_yards": "rush_yds",
              "football_team_total_scrimmage_yards": "scrimmage_yds", "football_team_total_touchdowns": "td",
              "football_team_total_pass_touchdowns": "pass_td", "football_team_total_rush_touchdowns": "rush_td",
              "football_team_total_receptions": "receptions", "football_team_points_full_game_total": "team_points",
              "football_team_total_field_goals_made": "fgm"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS pm_teams (
    pm_team_id INTEGER PRIMARY KEY, name TEXT, alias TEXT, abbreviation TEXT, display_abbr TEXT,
    conference TEXT, ranking TEXT, record TEXT, color TEXT, sportsdataio_id TEXT, sportradar_id TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS pm_events (
    event_id TEXT PRIMARY KEY, slug TEXT UNIQUE, title TEXT, series_slug TEXT, start_time TEXT,
    away_pm_team_id INTEGER, home_pm_team_id INTEGER,     -- per Polymarket away-first ordering (unverified per game)
    score TEXT, away_score INTEGER, home_score INTEGER, period TEXT, elapsed TEXT,
    live INTEGER, ended INTEGER, closed INTEGER, sportradar_game_id TEXT, n_markets INTEGER, updated_at TEXT,
    settled_winner_pm_team_id INTEGER,                     -- from moneyline settlement (authoritative for the market)
    score_conflict INTEGER);                               -- 1 if Polymarket's score feed disagrees with its settlement
CREATE INDEX IF NOT EXISTS ix_pm_events_start ON pm_events(start_time);
CREATE TABLE IF NOT EXISTS pm_markets (
    market_slug TEXT PRIMARY KEY, event_id TEXT, sports_market_type TEXT, market_type TEXT, line REAL,
    long_pm_team_id INTEGER, long_desc TEXT, active INTEGER, closed INTEGER, settlement REAL);
CREATE INDEX IF NOT EXISTS ix_pm_markets_event ON pm_markets(event_id, sports_market_type);
CREATE TABLE IF NOT EXISTS pm_quotes (
    market_slug TEXT NOT NULL, kind TEXT NOT NULL,        -- live | close
    ts TEXT NOT NULL, bid REAL, ask REAL, mid REAL, last REAL, volume REAL, open_interest REAL,
    PRIMARY KEY (market_slug, kind, ts));
CREATE TABLE IF NOT EXISTS pm_implied (
    event_id TEXT NOT NULL, kind TEXT NOT NULL, ts TEXT NOT NULL,
    p_away_win REAL, p_home_win REAL, ml_spread REAL,
    margin_home_mean REAL, margin_sd REAL, n_spread INTEGER, spread_fit_rmse REAL,
    total_mean REAL, total_sd REAL, n_total INTEGER, total_fit_rmse REAL,
    max_ml_width REAL, quality TEXT, in_play INTEGER DEFAULT 0, PRIMARY KEY (event_id, kind, ts));
CREATE TABLE IF NOT EXISTS pm_sync_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, what TEXT, n INTEGER, note TEXT);
"""


def markets_db_path() -> Path:
    import os
    from ..config import DATA_DIR
    return Path(os.environ.get("CFBQ_MARKETS_DB", DATA_DIR / "polymarket.db"))


def open_markets_db(path: Path | None = None):
    c = connect(path or markets_db_path())
    c.executescript(SCHEMA)
    for tbl, col, typ in (("pm_implied", "in_play", "INTEGER DEFAULT 0"), ("pm_events", "settled_winner_pm_team_id", "INTEGER"),
                          ("pm_events", "score_conflict", "INTEGER")):
        if col not in [r[1] for r in c.execute(f"PRAGMA table_info({tbl})")]:
            c.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
    c.commit()
    return c


MAX_QUOTE_WIDTH = 0.30   # wider books (e.g. bid 0 / ask 1 after a game) carry no price information


# ----------------------------------------------------------------------------- client
class PolymarketUSClient:
    """Read-only client for the public gateway. Caches immutable responses (closed events/settled history)."""

    def __init__(self, base: str = GATEWAY, min_interval: float = 0.34, cache_dir: Path | None = None,
                 session: requests.Session | None = None, max_retries: int = 6):
        from ..config import DATA_DIR
        self.base = base.rstrip("/")
        self.min_interval = min_interval
        self.cache = Path(cache_dir or DATA_DIR / "cache" / "polymarket")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.s = session or requests.Session()
        self.max_retries = max_retries
        self._last = 0.0
        self.calls = 0
        self.base_interval = min_interval
        self._ok_streak = 0
        self.n429 = 0

    def get(self, path: str, params: dict | None = None, cache: bool = False):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        key = hashlib.sha1(f"{path}?{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        fp = self.cache / f"{key}.json"
        if cache and fp.exists():
            return json.loads(fp.read_text())
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            r = self.s.get(self.base + path, params=params, timeout=30, headers={"Accept": "application/json"})
            self._last = time.time()
            self.calls += 1
            if r.status_code == 429:      # adaptive (AIMD) throttle: slow down multiplicatively on 429
                self.n429 += 1
                self.min_interval = min(2.0, self.min_interval * 1.5)
                self._ok_streak = 0
                time.sleep(min(30, 2.0 * 2 ** attempt))
                continue
            self._ok_streak += 1
            if self._ok_streak >= 10 and self.min_interval > self.base_interval:
                self.min_interval = max(self.base_interval, self.min_interval * 0.7)
                self._ok_streak = 0
            if r.status_code == 404:
                return None
            if r.status_code >= 500:      # transient gateway errors: retry, then skip this resource
                if attempt < 2:
                    time.sleep(2 + 2 * attempt)
                    continue
                return None
            r.raise_for_status()
            data = r.json()
            if cache:
                fp.write_text(json.dumps(data))
            return data
        raise RuntimeError(f"Polymarket US rate limit persisted for {path}")

    # -- endpoints
    def cfb_series_id(self) -> int:
        return int(self.get("/v2/leagues/cfb")["league"]["activeSeriesId"])

    def events(self, series_id: int, *, closed: bool | None = None, start_min: str | None = None,
               start_max: str | None = None, page: int = 50, max_pages: int = 40) -> list[dict]:
        out = []
        for i in range(max_pages):
            d = self.get("/v1/events", {"seriesId": series_id, "limit": page, "offset": i * page,
                                        "closed": None if closed is None else str(closed).lower(),
                                        "startDateMin": start_min, "startDateMax": start_max},
                         cache=bool(closed))
            ev = (d or {}).get("events", [])
            out += ev
            if len(ev) < page:
                break
        return out

    def event(self, slug: str, cache: bool = False) -> dict | None:
        d = self.get("/v1/events", {"slug": slug, "limit": 1}, cache=cache)
        ev = (d or {}).get("events", [])
        return ev[0] if ev else None

    def bbo(self, slug: str) -> dict | None:
        d = self.get(f"/v1/markets/{slug}/bbo")
        return (d or {}).get("marketData")

    def settlement(self, slug: str):
        d = self.get(f"/v1/markets/{slug}/settlement", cache=True)
        return None if d is None else d.get("settlement")

    def history(self, slug: str, start_ts: int, end_ts: int, cache: bool = True) -> list[dict]:
        d = self.get("/v1/price-history", {"symbol": slug, "timestamp.startTimestamp": int(start_ts),
                                           "timestamp.endTimestamp": int(end_ts), "fidelity": 1}, cache=cache)
        return (d or {}).get("history", [])

    def history_fixed(self, slug: str, interval: str = "INTERVAL_ALL") -> list[dict]:
        fid = {"INTERVAL_1H": 1, "INTERVAL_6H": 1, "INTERVAL_LIVE": 1, "INTERVAL_1D": 5}.get(interval, 180)
        d = self.get("/v1/price-history", {"symbol": slug, "fixedInterval": interval, "fidelity": fid})
        return (d or {}).get("history", [])


# ----------------------------------------------------------------------------- parsing (pure)
def _f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def parse_team(t: dict) -> dict:
    prov = {p.get("provider"): p.get("providerId") for p in t.get("providerIds", []) or []}
    return dict(pm_team_id=int(t["id"]), name=t.get("name"), alias=t.get("alias"), abbreviation=t.get("abbreviation"),
                display_abbr=t.get("displayAbbreviation"), conference=t.get("conference"), ranking=t.get("ranking"),
                record=t.get("record"), color=t.get("colorPrimary"),
                sportsdataio_id=prov.get("PROVIDER_SPORTSDATAIO"), sportradar_id=prov.get("PROVIDER_SPORTRADAR"),
                updated_at=utcnow())


def parse_event(e: dict) -> tuple[dict, list[dict], list[dict]]:
    """-> (event row, team rows, market rows). Score string follows the listed team order."""
    teams = [parse_team(t) for t in e.get("teams", []) or [] if t.get("id") is not None]
    away = teams[0]["pm_team_id"] if len(teams) > 0 else None
    home = teams[1]["pm_team_id"] if len(teams) > 1 else None
    # Per-team totals from periodScores (keyed by team id) — the bare `score` string is not reliably ordered.
    a_s = h_s = None
    state = e.get("eventState") or (e.get("metadata") or {}).get("gameState") or {}
    tot: dict[str, int] = {}
    for per in state.get("periodScores", []) or []:
        for sc in per.get("scores", []) or []:
            tot[str(sc.get("competitorId"))] = tot.get(str(sc.get("competitorId")), 0) + int(sc.get("score") or 0)
    if away is not None and str(away) in tot and str(home) in tot:
        a_s, h_s = tot[str(away)], tot[str(home)]
    else:
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", str(e.get("score") or ""))
        if m:
            a_s, h_s = int(m.group(1)), int(m.group(2))
    ev = dict(event_id=str(e["id"]), slug=e["slug"], title=e.get("title"), series_slug=e.get("seriesSlug"),
              start_time=e.get("startTime") or e.get("startDate"), away_pm_team_id=away, home_pm_team_id=home,
              score=e.get("score"), away_score=a_s, home_score=h_s, period=e.get("period"), elapsed=e.get("elapsed"),
              live=int(bool(e.get("live"))), ended=int(bool(e.get("ended"))), closed=int(bool(e.get("closed"))),
              sportradar_game_id=e.get("sportradarGameId"), n_markets=len(e.get("markets", []) or []),
              updated_at=utcnow(), settled_winner_pm_team_id=None, score_conflict=None)
    mkts = []
    for mk in e.get("markets", []) or []:
        smt = mk.get("sportsMarketType")
        if smt not in (MONEYLINE, SPREAD, TOTAL) and smt not in PROP_TYPES:
            continue
        sides = mk.get("marketSides", []) or []
        long_side = next((s for s in sides if s.get("long")), sides[0] if sides else {})
        mkts.append(dict(market_slug=mk["slug"], event_id=ev["event_id"], sports_market_type=smt,
                         market_type=mk.get("marketType"), line=_f(mk.get("line")),
                         long_pm_team_id=long_side.get("teamId"), long_desc=long_side.get("description"),
                         active=int(bool(mk.get("active"))), closed=int(bool(mk.get("closed"))), settlement=None,
                         _bidask=mk.get("outcomePrices")))
    return ev, teams, mkts


def bidask_from_outcome_prices(op) -> tuple[float | None, float | None]:
    """`outcomePrices` on the gateway carries [best bid, best ask] (verified against /bbo)."""
    try:
        v = json.loads(op) if isinstance(op, str) else op
        b, a = _f(v[0]), _f(v[1])
    except Exception:
        return None, None
    if b is None or a is None or b > a + 1e-9 or (a - b) > MAX_QUOTE_WIDTH:
        return None, None
    return b, a


def mid_from_history_point(pt: dict) -> float | None:
    """history: longPrice ~ best ask (YES display), shortPrice ~ 1 - best bid  =>  mid = (ask + bid)/2."""
    lp, sp = _f(pt.get("longPrice")), _f(pt.get("shortPrice"))
    if lp is None or sp is None:
        return None
    bid, ask = 1 - sp, lp
    if bid > ask + 1e-9 or (ask - bid) > MAX_QUOTE_WIDTH:   # crossed, stale or empty book
        return None
    return (bid + ask) / 2


# ----------------------------------------------------------------------------- ladder fitting (pure)
def fit_normal_ladder(lines, probs, kind: str, min_p=0.03, max_p=0.97) -> dict:
    """Fit the market's implied distribution from a ladder of binary prices.

    spread: P(YES at line L) = P(margin_long > -L) = Phi((mu + L)/sigma)  -> probit(p) = mu/s + L/s
    total : P(YES at line L) = P(total > L)         = Phi((mu - L)/sigma)  -> probit(p) = mu/s - L/s
    Weighted least squares on the probit scale over informative rungs (min_p < p < max_p).
    """
    L = np.asarray(lines, float)
    p = np.asarray(probs, float)
    ok = np.isfinite(L) & np.isfinite(p) & (p > min_p) & (p < max_p)
    L, p = L[ok], p[ok]
    if len(L) < 2 or np.ptp(L) == 0:
        return {"mean": None, "sd": None, "n": int(len(L)), "rmse": None}
    z = norm.ppf(p)
    w = p * (1 - p)
    x = L if kind == "spread" else -L
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
    a, b = beta
    if b <= 0:
        return {"mean": None, "sd": None, "n": int(len(L)), "rmse": None}
    sd = 1 / b
    mean = a * sd
    rmse = float(np.sqrt(np.average((norm.cdf(a + b * x) - p) ** 2, weights=w)))
    return {"mean": float(mean), "sd": float(sd), "n": int(len(L)), "rmse": rmse}


def implied_from_markets(ev: dict, mk: pd.DataFrame, price_col: str = "mid", in_play: bool = False) -> dict:
    """Event-level implied quantities from per-market mids. Margins expressed for Polymarket's 'home' (2nd listed)."""
    out = dict(event_id=ev["event_id"], p_away_win=None, p_home_win=None, ml_spread=None, margin_home_mean=None,
               margin_sd=None, n_spread=0, spread_fit_rmse=None, total_mean=None, total_sd=None, n_total=0,
               total_fit_rmse=None, max_ml_width=None, quality="no_data")
    away, home = ev.get("away_pm_team_id"), ev.get("home_pm_team_id")
    ml = mk[mk.sports_market_type == MONEYLINE].dropna(subset=[price_col])
    if len(ml):
        r = ml.iloc[0]
        p_long = float(r[price_col])
        p_home = p_long if r.long_pm_team_id == home else 1 - p_long
        out.update(p_home_win=p_home, p_away_win=1 - p_home, ml_spread=float(15.5 * norm.ppf(np.clip(p_home, 1e-3, 1 - 1e-3))))
        if "ask" in ml and pd.notna(r.get("ask")) and pd.notna(r.get("bid")):
            out["max_ml_width"] = float(r["ask"] - r["bid"])
    out["in_play"] = int(bool(in_play))
    sp = mk[(mk.sports_market_type == SPREAD)].dropna(subset=[price_col, "line"])
    if in_play:      # in-game prices are conditional on the score: no pre-game distribution to fit
        sp = sp.iloc[0:0]
    if len(sp):
        # express every rung as P(margin_home > -L_home)
        sign = np.where(sp.long_pm_team_id == home, 1.0, -1.0)
        p_home = np.where(sign > 0, sp[price_col], 1 - sp[price_col])
        L_home = sp.line.values * sign          # long=away with line L  <=>  home YES-equivalent at line -L
        f = fit_normal_ladder(L_home, p_home, "spread")
        out.update(margin_home_mean=f["mean"], margin_sd=f["sd"], n_spread=f["n"], spread_fit_rmse=f["rmse"])
    tt = mk[(mk.sports_market_type == TOTAL)].dropna(subset=[price_col, "line"])
    if in_play:
        tt = tt.iloc[0:0]
    if len(tt):
        f = fit_normal_ladder(tt.line.values, tt[price_col].values, "total")
        out.update(total_mean=f["mean"], total_sd=f["sd"], n_total=f["n"], total_fit_rmse=f["rmse"])
    q = []
    if out["p_home_win"] is not None:
        q.append("ml")
    if out["margin_home_mean"] is not None:
        q.append("spread")
    if out["total_mean"] is not None:
        q.append("total")
    if out["max_ml_width"] is not None and out["max_ml_width"] > 0.10:
        q.append("wide")
    if in_play:
        q.append("in_play")
    out["quality"] = "+".join(q) or "no_data"
    return out


# ----------------------------------------------------------------------------- sync
def _store(conn, ev, teams, mkts):
    insert_df(conn, "pm_events", pd.DataFrame([ev]))
    if teams:
        insert_df(conn, "pm_teams", pd.DataFrame(teams).drop_duplicates("pm_team_id"))
    if mkts:
        insert_df(conn, "pm_markets", pd.DataFrame(mkts).drop(columns=["_bidask"]))


def sync_live(conn, client: PolymarketUSClient, days_ahead: int = 8, log=print) -> int:
    """Upcoming + in-progress games: one event call per game gives bid/ask for every market."""
    sid = client.cfb_series_id()
    now = datetime.now(timezone.utc)
    evs = client.events(sid, closed=False, start_min=(now - pd.Timedelta(hours=6)).isoformat().replace("+00:00", "Z"),
                        start_max=(now + pd.Timedelta(days=days_ahead)).isoformat().replace("+00:00", "Z"))
    ts = utcnow()
    n = 0
    for e in evs:
        if e.get("ended") or e.get("closed"):
            continue
        full = client.event(e["slug"]) if len(e.get("markets", [])) <= 1 else e
        if not full:
            continue
        ev, teams, mkts = parse_event(full)
        _store(conn, ev, teams, mkts)
        rows = []
        for m in mkts:
            b, a = bidask_from_outcome_prices(m["_bidask"])
            if b is None:
                continue
            rows.append(dict(market_slug=m["market_slug"], kind="live", ts=ts, bid=b, ask=a, mid=(a + b) / 2,
                             last=None, volume=None, open_interest=None))
        insert_df(conn, "pm_quotes", pd.DataFrame(rows))
        mk = pd.DataFrame(mkts).merge(pd.DataFrame(rows), on="market_slug", how="left") if rows else pd.DataFrame(mkts)
        if rows:
            imp = implied_from_markets(ev, mk, in_play=bool(ev["live"]))
            imp.update(kind="live", ts=ts)
            insert_df(conn, "pm_implied", pd.DataFrame([imp]))
        n += 1
        conn.commit()
    conn.execute("INSERT INTO pm_sync_log (ts, what, n, note) VALUES (?,?,?,?)", (ts, "live", n, f"calls={client.calls}"))
    conn.commit()
    log(f"Polymarket live sync: {n} games ({client.calls} API calls)")
    return n


def _closing_mid(client, slug, start_ts, window_s=4 * 3600):
    hist = client.history(slug, start_ts - window_s, start_ts, cache=True)
    for pt in reversed(hist):
        if pt.get("timestamp", 0) <= start_ts:
            m = mid_from_history_point(pt)
            if m is not None:
                return m, int(pt["timestamp"]), 1 - _f(pt.get("shortPrice")), _f(pt.get("longPrice"))
    return None, None, None, None


def closing_snapshot(client: PolymarketUSClient, e: dict, max_spread_rungs: int = 5, max_total_rungs: int = 6,
                     props: bool = False) -> tuple[dict, list[dict], dict] | None:
    """Pre-kickoff ('closing') prices: last book sample at or before kickoff.

    To respect rate limits only informative rungs are fetched: moneyline first, then spread rungs
    centred on the moneyline-implied spread, then a coarse-to-fine pass over the total ladder.
    """
    ev, teams, mkts = parse_event(e)
    st = _ts(ev["start_time"])
    mk = pd.DataFrame(mkts)
    if mk.empty:
        return None
    quotes = []

    def grab(row):
        mid, t, bid, ask = _closing_mid(client, row.market_slug, st)
        if mid is not None:
            quotes.append(dict(market_slug=row.market_slug, kind="close", ts=datetime.fromtimestamp(t, timezone.utc).isoformat(),
                               bid=bid, ask=ask, mid=mid, last=None, volume=None, open_interest=None))
        return mid
    ml = mk[mk.sports_market_type == MONEYLINE]
    p_long = grab(ml.iloc[0]) if len(ml) else None
    # moneyline -> approximate spread for the long team; centre spread rungs there
    sp = mk[mk.sports_market_type == SPREAD].dropna(subset=["line"]).copy()
    if len(sp):
        centre = 0.0
        if p_long is not None and len(ml):
            mu_long = 15.5 * norm.ppf(np.clip(p_long, 0.01, 0.99))
            # rung L (for long team X) is a coin flip when margin_X ~ -L; convert per rung's long team
            sp["_target"] = np.where(sp.long_pm_team_id == ml.iloc[0].long_pm_team_id, -mu_long, mu_long)
            sp["_d"] = (sp.line - sp._target).abs()
        else:
            sp["_d"] = (sp.line - centre).abs()
        for row in sp.sort_values("_d").head(max_spread_rungs).itertuples():
            grab(row)
    tt = mk[mk.sports_market_type == TOTAL].dropna(subset=["line"]).sort_values("line")
    if len(tt):
        coarse = tt.iloc[:: max(1, len(tt) // 4)]
        got = {r.line: grab(r) for r in coarse.itertuples()}
        known = sorted((l, p) for l, p in got.items() if p is not None)
        cross = next((l for (l, p), (l2, p2) in zip(known, known[1:]) if p >= 0.5 >= p2), None)
        if cross is not None:
            rest = tt[~tt.line.isin(got)].assign(_d=lambda d: (d.line - cross).abs()).sort_values("_d")
            for row in rest.head(max(0, max_total_rungs - len(got))).itertuples():
                grab(row)
    if props:
        for row in mk[mk.sports_market_type.isin(PROP_TYPES)].itertuples():
            grab(row)
    q = pd.DataFrame(quotes)
    merged = mk.merge(q, on="market_slug", how="left") if len(q) else mk.assign(mid=np.nan)
    imp = implied_from_markets(ev, merged)
    imp.update(kind="close", ts=ev["start_time"])
    return ev | {"_teams": teams, "_mkts": mkts}, quotes, imp


FBS_CONFERENCES = {"Southeastern", "Big Ten", "Big 12", "Atlantic Coast", "Pac-12", "American Athletic", "American",
                   "Mountain West", "Sun Belt", "Sun Belt - East", "Sun Belt - West", "Mid-American", "Conference USA", "FBS Independents", "Independent"}


def is_fbs_event(e: dict) -> bool:
    confs = [t.get("conference") for t in e.get("teams", []) or []]
    return len(confs) == 2 and all(c in FBS_CONFERENCES for c in confs)


def sync_closing(conn, client: PolymarketUSClient, start_min: str | None = None, start_max: str | None = None,
                 limit: int | None = None, settle: bool = True, fbs_only: bool = True, log=print) -> int:
    """Closed games: closing prices + implied lines + settlements. Immutable responses are cached on disk."""
    sid = client.cfb_series_id()
    evs = client.events(sid, closed=True, start_min=start_min, start_max=start_max)
    done = {r[0] for r in conn.execute("SELECT event_id FROM pm_implied WHERE kind='close'")}
    evs = [e for e in evs if str(e["id"]) not in done and (not fbs_only or is_fbs_event(e))]
    if limit:
        evs = evs[:limit]
    n = 0
    for e in evs:
        full = client.event(e["slug"], cache=True) or e
        snap = closing_snapshot(client, full)
        if snap is None:
            continue
        ev, quotes, imp = snap
        teams, mkts = ev.pop("_teams"), ev.pop("_mkts")
        if settle:
            for m in mkts:
                if m["sports_market_type"] == MONEYLINE:
                    m["settlement"] = _f(client.settlement(m["market_slug"]))
                    if m["settlement"] in (0.0, 1.0) and m["long_pm_team_id"] is not None:
                        other = ev["home_pm_team_id"] if m["long_pm_team_id"] == ev["away_pm_team_id"] else ev["away_pm_team_id"]
                        ev["settled_winner_pm_team_id"] = m["long_pm_team_id"] if m["settlement"] == 1.0 else other
                        if ev["away_score"] is not None and ev["home_score"] is not None and ev["away_score"] != ev["home_score"]:
                            fw = ev["away_pm_team_id"] if ev["away_score"] > ev["home_score"] else ev["home_pm_team_id"]
                            ev["score_conflict"] = int(fw != ev["settled_winner_pm_team_id"])
        _store(conn, ev, teams, mkts)
        insert_df(conn, "pm_quotes", pd.DataFrame(quotes))
        insert_df(conn, "pm_implied", pd.DataFrame([imp]))
        conn.commit()
        n += 1
        if n % 10 == 0:
            log(f"  closing sync: {n}/{len(evs)} games ({client.calls} calls)")
    conn.execute("INSERT INTO pm_sync_log (ts, what, n, note) VALUES (?,?,?,?)", (utcnow(), "close", n, f"calls={client.calls}"))
    conn.commit()
    log(f"Polymarket closing sync: {n} games ({client.calls} API calls)")
    return n


TEAM_POINTS = "football_team_points_full_game_total"


def backfill_team_totals(conn, client: PolymarketUSClient, rungs: int = 2, log=print) -> int:
    """Closing (pre-kickoff) prices for team-points totals: the `rungs` lines nearest each team's implied total
    (from the game's closing spread/total fit). Team totals are the one prop family with a live book (~80% of
    pre-game quotes within 10c); the stat props (yards, TDs, FGs) have empty books and are skipped."""
    imp = pd.read_sql_query("""SELECT i.event_id, i.margin_home_mean, i.total_mean, e.start_time, e.home_pm_team_id,
        e.away_pm_team_id FROM pm_implied i JOIN pm_events e USING(event_id) WHERE i.kind='close'
        AND i.margin_home_mean IS NOT NULL AND i.total_mean IS NOT NULL""", conn)
    mk = pd.read_sql_query("SELECT market_slug, event_id, line, long_pm_team_id, long_desc FROM pm_markets WHERE sports_market_type=?",
                           conn, params=(TEAM_POINTS,))
    have = {r[0] for r in conn.execute("""SELECT DISTINCT m.event_id FROM pm_quotes q JOIN pm_markets m USING(market_slug)
        WHERE q.kind='close' AND m.sports_market_type=?""", (TEAM_POINTS,))}
    n = 0
    for r in imp[~imp.event_id.isin(have)].itertuples():
        st = _ts(r.start_time)
        target = {r.home_pm_team_id: (r.total_mean + r.margin_home_mean) / 2, r.away_pm_team_id: (r.total_mean - r.margin_home_mean) / 2}
        quotes = []
        for team, tgt in target.items():
            m = mk[(mk.event_id == r.event_id) & (mk.long_pm_team_id == team)].dropna(subset=["line"])
            for row in m.assign(_d=(m.line - tgt).abs()).sort_values("_d").head(rungs).itertuples():
                mid, t, bid, ask = _closing_mid(client, row.market_slug, st)
                if mid is not None:
                    quotes.append(dict(market_slug=row.market_slug, kind="close", ts=datetime.fromtimestamp(t, timezone.utc).isoformat(),
                                       bid=bid, ask=ask, mid=mid, last=None, volume=None, open_interest=None))
        if quotes:
            insert_df(conn, "pm_quotes", pd.DataFrame(quotes))
            conn.commit()
        n += 1
        if n % 25 == 0:
            log(f"  team totals: {n} events ({client.calls} calls)")
    log(f"team-total backfill: {n} events")
    return n


def recompute_implied(conn, kind: str = "close") -> int:
    """Re-derive event-level implied numbers from stored quotes (e.g. after a filter or fitting change)."""
    q = pd.read_sql_query("""SELECT q.*, m.event_id, m.sports_market_type, m.line, m.long_pm_team_id FROM pm_quotes q
                             JOIN pm_markets m USING(market_slug) WHERE q.kind=?""", conn, params=(kind,))
    if q.empty:
        return 0
    q = q.sort_values("ts").groupby("market_slug").tail(1)
    ok = (q.ask - q.bid).abs() <= MAX_QUOTE_WIDTH
    q.loc[~ok, "mid"] = np.nan
    ev = pd.read_sql_query("SELECT * FROM pm_events", conn).set_index("event_id")
    old = pd.read_sql_query("SELECT event_id, ts, in_play FROM pm_implied WHERE kind=?", conn, params=(kind,))
    old = old.sort_values("ts").groupby("event_id").tail(1).set_index("event_id")
    rows = []
    for eid, g in q.groupby("event_id"):
        if eid not in ev.index:
            continue
        e = ev.loc[eid].to_dict() | {"event_id": eid}
        inp = bool(old.in_play.get(eid, 0) or 0) if kind == "live" else False
        imp = implied_from_markets(e, g, in_play=inp)
        imp.update(kind=kind, ts=old.ts.get(eid, e["start_time"]) if kind == "live" else e["start_time"])
        rows.append(imp)
    insert_df(conn, "pm_implied", pd.DataFrame(rows))
    conn.commit()
    return len(rows)


# ----------------------------------------------------------------------------- mapping to model games
_ALIASES = {"miami (fl)": "miami", "miami fl": "miami", "ole miss": "mississippi", "pitt": "pittsburgh",
            "uconn": "connecticut", "usc": "southern california", "lsu": "louisiana state", "ucf": "central florida",
            "smu": "southern methodist", "byu": "brigham young", "tcu": "texas christian", "utsa": "ut san antonio",
            "fiu": "florida international", "fau": "florida atlantic", "umass": "massachusetts",
            "app state": "appalachian state", "nc state": "north carolina state", "southern miss": "southern mississippi",
            "hawaii": "hawai'i", "sam houston": "sam houston state", "ul monroe": "louisiana monroe",
            "louisiana": "louisiana lafayette"}


def norm_name(s: str | None) -> str:
    s = (s or "").lower().replace("&", "and").replace("é", "e")
    s = re.sub(r"[^a-z0-9' ()]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _ALIASES.get(s, s)
    return s.replace("state university", "state").replace(" university", "").replace("'", "")


def map_events_to_games(mconn, gconn, tolerance_hours: int = 36) -> pd.DataFrame:
    """Match Polymarket events to model games by unordered team pair + kickoff within tolerance.

    Returns event_id, game_id, home_id, away_id, and `pm_home_is_home` (whether Polymarket's 2nd-listed team
    is the model's home team) so probabilities can be oriented from our authoritative home/away record."""
    ev = pd.read_sql_query("""SELECT e.event_id, e.start_time, ta.name a_name, th.name h_name FROM pm_events e
        LEFT JOIN pm_teams ta ON ta.pm_team_id=e.away_pm_team_id LEFT JOIN pm_teams th ON th.pm_team_id=e.home_pm_team_id""", mconn)
    g = pd.read_sql_query("""SELECT g.game_id, g.start_date, g.home_id, g.away_id, h.school h_school, a.school a_school
        FROM games g JOIN teams h ON h.team_id=g.home_id JOIN teams a ON a.team_id=g.away_id""", gconn)
    if ev.empty or g.empty:
        return pd.DataFrame(columns=["event_id", "game_id", "home_id", "away_id", "pm_home_is_home"])
    g["k"] = [frozenset((norm_name(a), norm_name(b))) for a, b in zip(g.h_school, g.a_school)]
    g["t"] = pd.to_datetime(g.start_date, utc=True, errors="coerce")
    ev["k"] = [frozenset((norm_name(a), norm_name(b))) for a, b in zip(ev.h_name, ev.a_name)]
    ev["t"] = pd.to_datetime(ev.start_time, utc=True, errors="coerce")
    m = ev.merge(g, on="k", suffixes=("_pm", "_g"))
    m = m[(m.t_pm - m.t_g).abs() <= pd.Timedelta(hours=tolerance_hours)]
    m["pm_home_is_home"] = [norm_name(h) == norm_name(s) for h, s in zip(m.h_name, m.h_school)]
    return m[["event_id", "game_id", "home_id", "away_id", "pm_home_is_home"]].drop_duplicates("event_id")


def market_for_games(mconn, gconn, kind: str = "close") -> pd.DataFrame:
    """Implied market numbers oriented to the model's home team, one row per mapped game."""
    mp = map_events_to_games(mconn, gconn)
    if mp.empty:
        return pd.DataFrame()
    imp = pd.read_sql_query("SELECT * FROM pm_implied WHERE kind=?", mconn, params=(kind,))
    imp = imp.sort_values("ts").groupby("event_id").tail(1)
    d = mp.merge(imp, on="event_id")
    flip = ~d.pm_home_is_home
    d["mkt_p_home"] = np.where(flip, d.p_away_win, d.p_home_win)
    d["mkt_margin_home"] = np.where(flip, -d.margin_home_mean, d.margin_home_mean)
    d["mkt_total"] = d.total_mean
    return d[["game_id", "event_id", "mkt_p_home", "mkt_margin_home", "margin_sd", "mkt_total", "total_sd", "quality", "ts"]]
