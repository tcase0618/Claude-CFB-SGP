"""Command-line pipeline. No LLM involvement at any stage.

  python -m cfbquant.cli synthetic              # build fictional demo DB
  python -m cfbquant.cli ingest --seasons 2023,2024 [--with-plays]   # real CFBD data (needs CFBD_API_KEY)
  python -m cfbquant.cli espn --seasons 2024,2025,2026  # real data, NO KEY (ESPN public feeds) -> data/cfb.db
  python -m cfbquant.cli recruiting             # recruiting/talent/portal from CFBD (free key) into the ESPN DB
  python -m cfbquant.cli analyze --season 2025  # EP/EPA, team & player metrics, ratings, registry
  python -m cfbquant.cli backtest --test 2024,2025 --sims 400
  python -m cfbquant.cli forecast --home ID --away ID --season S --week W
  python -m cfbquant.cli markets live           # Polymarket US live board (free public API, no key)
  python -m cfbquant.cli markets close --since 2026-08-20   # closing prices + settlements (cached)
  python -m cfbquant.cli markets calibration    # how calibrated were closing prices?
  python -m cfbquant.cli serve                  # API + terminal UI on :8000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .config import get_settings
from .db import insert_df, open_db


def _seasons(s):
    return [int(x) for x in s.split(",") if x]


def write_registry(conn):
    from .analytics import player_metrics, team_metrics
    conn.execute("DELETE FROM metric_definitions")
    insert_df(conn, "metric_definitions", pd.DataFrame(team_metrics.registry_rows() + player_metrics.registry_rows()))
    conn.commit()


def analyze(conn, seasons, log=print):
    from .analytics.ep_model import enrich_plays
    from .analytics.player_metrics import write_player_metrics
    from .analytics.ratings import write_ratings
    from .analytics.team_metrics import write_team_metrics
    t = time.time()
    log(f"EP model: {enrich_plays(conn)}")
    write_registry(conn)
    for s in seasons:
        n = write_team_metrics(conn, s)
        write_player_metrics(conn, s)
        write_ratings(conn, s)
        log(f"season {s}: {n} team-metric rows, player metrics, ratings ({time.time() - t:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cfbquant")
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic"); s.add_argument("--seasons", default="2023,2024,2025"); s.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("ingest"); s.add_argument("--seasons", required=True); s.add_argument("--with-plays", action="store_true")
    s = sub.add_parser("espn"); s.add_argument("--seasons", required=True); s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("recruiting"); s.add_argument("--classes", default="2019,2020,2021,2022,2023,2024,2025,2026")
    s.add_argument("--seasons", default="2024,2025,2026")
    s = sub.add_parser("analyze"); s.add_argument("--seasons", default=None)
    s = sub.add_parser("backtest"); s.add_argument("--test", required=True); s.add_argument("--sims", type=int, default=None)
    s = sub.add_parser("forecast"); s.add_argument("--home", type=int, required=True); s.add_argument("--away", type=int, required=True)
    s.add_argument("--season", type=int, required=True); s.add_argument("--week", type=int, required=True)
    s.add_argument("--sims", type=int, default=None); s.add_argument("--seed", type=int, default=None)
    s.add_argument("--store", action="store_true")
    s = sub.add_parser("markets"); s.add_argument("action", choices=["live", "close", "calibration"])
    s.add_argument("--since", default=None); s.add_argument("--until", default=None); s.add_argument("--limit", type=int, default=None)
    s.add_argument("--days", type=int, default=8)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000); s.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    st = get_settings()
    db = Path(a.db) if a.db else st.db_path

    if a.cmd == "synthetic":
        from .ingest.synthetic import build_synthetic
        if db.exists():
            db.unlink()
        with open_db(db) as conn:
            build_synthetic(conn, tuple(_seasons(a.seasons)), a.seed)
            analyze(conn, _seasons(a.seasons))
        return
    if a.cmd == "markets":
        from .ingest import polymarket as pm
        mc = pm.open_markets_db()
        cl = pm.PolymarketUSClient()
        if a.action == "live":
            pm.sync_live(mc, cl, days_ahead=a.days)
        elif a.action == "close":
            pm.sync_closing(mc, cl, start_min=a.since, start_max=a.until, limit=a.limit)
        else:
            import os
            os.environ.setdefault("CFBQ_DB", str(db))
            from .api.app import market_calibration
            print(json.dumps({k: v for k, v in market_calibration().items() if k != "calibration"}, indent=1, default=str))
        return
    if a.cmd == "espn":
        from .ingest.espn import ESPNClient, ingest_espn
        with open_db(db) as conn:
            ingest_espn(conn, ESPNClient(), _seasons(a.seasons), workers=a.workers)
        return
    if a.cmd == "recruiting":
        from .ingest.cfbd import CFBDClient
        from .ingest.recruiting import load_recruiting
        with open_db(db) as conn:
            print(json.dumps(load_recruiting(conn, CFBDClient(st), _seasons(a.classes), _seasons(a.seasons)), indent=1))
        return
    if a.cmd == "ingest":
        from .ingest.cfbd import CFBDClient, ingest_seasons
        with open_db(db) as conn:
            ingest_seasons(conn, CFBDClient(st), _seasons(a.seasons), with_plays=a.with_plays)
        return
    with open_db(db) as conn:
        if a.cmd == "analyze":
            seasons = _seasons(a.seasons) if a.seasons else [r[0] for r in conn.execute("SELECT DISTINCT season FROM games ORDER BY 1")]
            analyze(conn, seasons)
        elif a.cmd == "backtest":
            from .models.backtest import evaluate, persist, run_backtest
            n = a.sims or st.backtest_sims
            bt = run_backtest(conn, _seasons(a.test), n_sims=n)
            bt["n_sims"] = n
            metrics, decision = evaluate(bt)
            persist(conn, bt, metrics, decision)
            show = pd.DataFrame(metrics)
            show = show[(show.scope == "overall") & ~show.metric.isin(["calibration_curve"])]
            print(show[["model_version", "metric", "value", "ci_low", "ci_high", "n"]].to_string(index=False))
            print("promotion:", json.dumps(decision))
        elif a.cmd == "forecast":
            from .models.forecast import forecast_game, store_forecast
            fc = forecast_game(conn, a.home, a.away, a.season, a.week, n_sims=a.sims or st.default_sims, seed=a.seed or st.seed)
            if a.store:
                store_forecast(conn, fc, None)
            print(json.dumps({k: fc[k] for k in ("home_win_prob", "home_pts", "away_pts", "margin", "total", "ot_prob",
                                                  "consistency_checks")}, indent=1, default=float))
        elif a.cmd == "serve":
            import uvicorn
            import os
            os.environ["CFBQ_DB"] = str(db)
            uvicorn.run("cfbquant.api.app:app", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
