# Architecture

## Phase 0 audit result
The repository was empty at start (no commits, no files). There was no existing NFL/NBA/NHL infrastructure to reuse, so
the platform was built from scratch. Environment constraints that shaped decisions: no `CFBD_API_KEY` (API returns 401),
4 CPUs / 15 GB RAM, Python 3.13, Node 22, no AutoGluon installed.

## Layout

```
cfbquant/
  config.py               env-driven settings (CFBQ_DB, CFBD_API_KEY, sims, seeds)
  db/schema.sql           relational schema (all Phase-1 entities; SQLite dialect, Postgres-portable)
  db/__init__.py          connection, migrations, bulk insert, numpy->SQL adapters
  ingest/cfbd.py          CFBD client (auth, rate limit, disk cache) + pure normalisers
  ingest/synthetic.py     fictional demo league (ground truth known)
  analytics/plays_frame.py   play loader + derived columns
  analytics/ep_model.py      EP model, EPA, success, garbage time
  analytics/team_metrics.py  241-metric team registry + computation + ridge opponent adjustment
  analytics/player_metrics.py 365-metric player registry (100 universal) + grades + archetypes
  analytics/team_params.py   as-of simulator parameter estimation (empirical-Bayes ridge, talent prior, level calibration)
  analytics/ratings.py       Elo benchmark
  analytics/talent.py        recruiting priors, Bayesian posteriors, breakouts, transfers, development curves
  sim/params.py, sim/engine.py  vectorised play-by-play simulator
  models/forecast.py         game + player forecasts, drivers, scenarios, calibration
  models/backtest.py         walk-forward backtest, benchmarks, scoring, promotion gate
  api/app.py                 FastAPI
  cli.py                     pipeline entry point
web/                         terminal UI (vanilla JS + SVG, no build step)
tests/                       pytest suite
```

**Why these choices.** SQLite + typed schema keeps the system runnable on constrained hardware with zero services; the
schema avoids SQLite-only features so it can move to Postgres. Analytical work is vectorised pandas/numpy over
play-level frames (Parquet export is a straightforward addition when data volume requires it). The frontend has no
build chain so it deploys anywhere FastAPI runs.

## Data integrity rules (enforced in code)
* Real and synthetic data live in **different database files**; `data_sources.is_synthetic` drives a permanent UI banner.
* Every row carries `source`. Every metric carries a classification. Metrics needing charting/tracking/participation
  are registered but never computed from fabricated inputs (`tests/test_analytics.py::test_unavailable_metrics_never_filled`).
* CFBD does not provide air yards, pass location, formations, coverages or snap participation → those columns stay NULL
  and dependent metrics show "N/A — requires charting". The synthetic league generates air yards (as part of its
  simulated play-by-play) so those code paths are exercised; they will show as unavailable on real CFBD data.

## Expected points
Multinomial logistic regression of the next score in the half (7 outcomes) on yards-to-goal (cubic), down, log
distance, goal-to-go and interactions. EPA handles scoring, defensive scores, safeties, changes of possession and end of
half. Success uses the 50/70/100% rule; garbage time uses Connelly's thresholds and is excluded from efficiency metrics.

## Opponent adjustment
For each adjustable metric, game-level values are fit with weighted ridge regression
`y = mu + off_i + def_j + h·home` (ridge = Gaussian prior → small samples shrink to league average). Used for team metrics,
player opponent-adjusted EPA (subtracting the opponent unit's adjusted EPA allowed on that play family), backtest
features, and SOS. All as-of computations filter plays strictly before the forecast week.

## Simulator (`sim/engine.py`)
Each lane is a game; all lanes advance one snap per iteration with numpy masks (10,000 sims ≈ 2–3 s on 4 CPUs).
Models kickoffs/touchbacks/returns, down & distance, run/pass/scramble/sack selection by situation, air yards (gamma) +
YAC, depth-dependent completion and interception, fumbles, penalties, punts, distance-dependent FGs, PAT/2-pt chart,
4th-down policy (team aggressiveness), clock with team pace, hurry-up/clock-kill, kneels, timeouts, halftime and the
2021+ NCAA OT format. Player box scores are accumulated from the same events that move the score, so completions ≤
attempts, team passing = Σ receiving, passing TD ↔ receiving TD, team rushing = Σ rushers hold by construction
(tested on every forecast: `consistency_checks`).

Parameters combine offence-produces and defence-allows rates with log5 (probabilities) or additive (continuous) rules
around the league mean; home field is a symmetric efficiency shift (~2.5 pts at defaults).

## Parameter estimation (`analytics/team_params.py`)
For every stat (sack, INT, depth-neutral completion, scramble, explosive-run, core-run yards, YAC, fumbles, neutral pass
rate, aDOT): weighted ridge on game aggregates, two passes — the second shrinks toward an **empirical-Bayes talent prior**
(slope of first-pass estimates on recruiting talent). Prior-season games enter at reduced weight scaled by returning
production; current-season games decay 5%/week. Pace is measured from game-clock run-off after designed runs; special
teams from FG make-over-expected, net punting and 4th-down go rate. Usage shares (targets, carries, tackles, sacks, INT)
come from recent box scores of **healthy** players (injury table for that week), blended with positional priors.
Finally a **league-level calibration** shifts every offence equally so simulated scoring on recent real matchups matches
observed scoring (fixes the data-environment level without changing relative strengths).

## Forecasts (`models/forecast.py`)
One simulation run produces win probability (raw + Platt-calibrated using a calibrator fit on the walk-forward backtest),
score/margin/total distributions, team stat distributions, and per-player distributions (mean, median, SD, P10–P90, count
distributions, P(≥1)). Drivers are measured by **neutralising** one unit to league average and re-simulating with common
random numbers. Scenarios: QB ±1 SD, QB1 out (backup's shrunk EPA/db mapped to pass parameters), WR1 out.

## Validation (`models/backtest.py`)
Walk-forward by week: features, parameters, benchmarks and the sim calibrator are all fit only on prior weeks. Benchmarks:
Elo (margin-aware, talent-informed preseason), OLS margin on adjusted net EPA, XGBoost margin model. Metrics with bootstrap
90% CIs: log loss, Brier, accuracy, margin/total/score MAE, RMSE, interval coverage, calibration curves, player stat MAE
and 80% coverage, TD/INT P(≥1) Brier + calibration, splits by season / conference / injury-affected / favourites.
**Promotion gate:** the simulator becomes `production` only if the paired log-loss improvement over Elo has a 90% CI
entirely below zero. Leakage is tested in `tests/test_leakage.py` by corrupting future data and asserting estimates
do not change.

## ML stack notes
XGBoost is used as a benchmark (it currently underperforms the simpler models out of sample on the demo data, which is
reported, not hidden). AutoGluon is not installed in this environment; the backtest's feature frame (`run_backtest()["games"]`)
is the intended input for an AutoGluon benchmark on a suitably provisioned machine.

## API
`/api/meta, /api/teams, /api/teams/{id}, /api/players, /api/players/{id}, /api/games, /api/forecast, /api/matchup,
/api/recruiting, /api/transfers, /api/audit, /api/metrics/definitions, /api/health`. Every view shows data freshness,
the production model, and the synthetic flag.
