# CFB Quant Intelligence Platform

Institutional-style college football research and prediction terminal: player intelligence, 200+ metric team
intelligence, opponent-adjusted ratings, a play-by-play Monte Carlo simulator that produces game **and** player
forecasts from one consistent simulation, walk-forward validation, and a dense terminal UI.

**Runs without any LLM.** Data processing, statistics, inference, simulation and the frontend are deterministic code.

> The bundled demo uses a **fictional synthetic league** (fictional schools/players), clearly flagged in the UI and
> stored in a separate database. Real data requires a CollegeFootballData API key (`CFBD_API_KEY`).

## Quick start (demo, no credentials)

```bash
pip install -r requirements.txt
python -m cfbquant.cli synthetic                 # build data/synthetic.db + all analytics (~3 min)
python -m cfbquant.cli backtest --test 2024,2025 --sims 400   # walk-forward validation (~6 min)
python -m cfbquant.cli serve                     # http://127.0.0.1:8000
```

## Real data — no key needed (ESPN public feeds)

```bash
python -m cfbquant.cli --db data/cfb.db espn --seasons 2024,2025,2026   # ~15 min first run, cached afterwards
export CFBQ_DB=data/cfb.db
python -m cfbquant.cli analyze
python -m cfbquant.cli markets live && python -m cfbquant.cli markets close --since 2026-08-20T00:00:00Z
python -m cfbquant.cli backtest --test 2026
python -m cfbquant.cli serve
```

See `docs/DATA_SOURCES.md` (what ESPN provides, text-parsed charting fields, terms caveat).

## Real data (CollegeFootballData, optional — adds recruiting/talent/portal)

```bash
export CFBD_API_KEY=...                          # https://collegefootballdata.com/key
export CFBQ_DB=data/cfb.db                       # real data lives in its own file
python -m cfbquant.cli ingest --seasons 2022,2023,2024 --with-plays
python -m cfbquant.cli analyze
python -m cfbquant.cli backtest --test 2023,2024
python -m cfbquant.cli serve
```

Responses are cached on disk (`data/cache/cfbd`), requests are rate-limited, and re-runs are incremental.
The real-data path is implemented and unit-tested against CFBD-shaped fixtures but has **not** been run against the
live API in this environment (no key was available) — see `IMPLEMENTATION_STATUS.md`.

## Player props & same-game parlays (the focus)

```bash
export CFBQ_DB=data/cfb.db
python -m cfbquant.cli lines --season 2026                 # snapshot opening + current DraftKings lines (run daily)
python -m cfbquant.cli sgp board --game <espn_game_id>     # prop board + best 3/5/8-leg tickets for one game
python -m cfbquant.cli sgp cards --season 2026 --week 11   # whole slate -> data/cards/ + paper-trading ledger
python -m cfbquant.cli sgp price --game <id> --legs slip.json --odds +600   # EV of a book SGP
python scripts/validate_sgp.py --season 2026 --weeks 3-10  # walk-forward validation (~16 min)
```

The simulation is anchored to the market's spread and total, adds parameter uncertainty and garbage-time
substitution, and prices any combination of legs from one consistent joint simulation. The **SGP Lab** tab has the
board, a bet slip with EV vs the book price, and auto tickets. Method, validation and limits: `docs/SGP.md`.

## Market data (Polymarket US — free, no key)

```bash
python -m cfbquant.cli markets live               # live/upcoming CFB board: win prob, implied spread & total
python -m cfbquant.cli markets close --since 2026-08-20T00:00:00Z   # closing prices + settlements (cached)
```

Real market prices are stored separately (`data/polymarket.db`), shown in the **Markets** tab, and used as a
closing-line benchmark when the model DB holds real CFBD games. See `docs/MARKETS.md`.

## Other commands

```bash
python -m cfbquant.cli forecast --home 1003 --away 1001 --season 2025 --week 12 --sims 10000
python -m pytest -q                              # 55 tests
python scripts/gen_metrics_doc.py                # regenerate docs/METRICS.md from the registry
```

## Docs

* `docs/ARCHITECTURE.md` — data model, pipeline, models, simulator, validation, API
* `docs/METRICS.md` — every metric: formula, direction, classification (generated)
* `docs/DATA_SOURCES.md` — ESPN (keyless), Polymarket US, CFBD; what each provides and its caveats
* `docs/SGP.md` — props / same-game parlays: anchoring, uncertainty, ticket builder, validation, limits
* `docs/MARKETS.md` — Polymarket US integration: endpoints, verified contract semantics, data-quality findings
* `IMPLEMENTATION_STATUS.md` — what is done, what is not, verification results, blockers
