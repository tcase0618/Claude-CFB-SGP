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

## Real data (CollegeFootballData)

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

## Other commands

```bash
python -m cfbquant.cli forecast --home 1003 --away 1001 --season 2025 --week 12 --sims 10000
python -m pytest -q                              # 28 tests
python scripts/gen_metrics_doc.py                # regenerate docs/METRICS.md from the registry
```

## Docs

* `docs/ARCHITECTURE.md` — data model, pipeline, models, simulator, validation, API
* `docs/METRICS.md` — every metric: formula, direction, classification (generated)
* `IMPLEMENTATION_STATUS.md` — what is done, what is not, verification results, blockers
