# Implementation Status

_Last updated: 2026-11-07. All numbers below come from runs in this repository, not estimates._

## Verdict
A working end-to-end platform: ingestion (CFBD + synthetic), relational store, EP/EPA, 241 team metrics, 365 player
metrics, opponent adjustment, Bayesian talent priors, a consistent play-by-play simulator, game + player forecasts,
walk-forward validation with benchmarks and a promotion gate, a FastAPI backend and a terminal UI.
**It has only been exercised on the fictional synthetic league.** The CFBD path is implemented and fixture-tested but has
never touched the live API here because no key was available. Treat every real-data claim as unverified until
`cli ingest` has been run with a key.

## Blocked / external dependencies
| Item | Status | What unblocks it |
|---|---|---|
| Live CFBD data | **Blocked** — no `CFBD_API_KEY` in environment (API returned 401) | Export key, run `cli ingest` |
| Charting/tracking metrics (PBWR, pressure, coverage, separation, routes, formations) | Registered, flagged unavailable, never faked | Licensed charting feed → populate `offensive_formations`, `defensive_formations`, `play_charting`, `player_participation`, then implement the computations |
| AutoGluon benchmark | Not installed | Train on a provisioned machine using the backtest feature frame |
| Weather / portal endpoints | Implemented with graceful fallback | May require a higher CFBD tier |

## Phase checklist
| Phase | Status | Notes |
|---|---|---|
| 0 Audit & plan | Done | Repo was empty; see `docs/ARCHITECTURE.md` |
| 1 Data infrastructure | Done (real path unverified live) | All listed entities in `schema.sql`; cache, rate limit, incremental; historical affiliations per season |
| 2 Universal player intelligence | Done | 100 universal metrics in categories A–H, with formula/direction/classification/percentiles; on/off, snaps, athletic testing flagged unavailable |
| 3 Position libraries | Done (data-limited) | QB 39, RB 20, WR 28, TE 30, OL 21×3 (charting-only + 2 proxies), DL/EDGE 15, LB 12, CB/S 15, K/P/LS/KR/PR. QB archetypes implemented. 147 player metrics require charting and are labelled so |
| 4 Team intelligence | Done | 241 metrics, 20 categories (≥10 each); 55 charting-only; opponent-adjusted where flagged |
| 5 Talent & development | Done (first version) | Recruiting priors → Bayesian posteriors, breakout candidates, development curves, transfer projections, returning production. Coaching/scheme-transition effects **not** modelled (no coaching data in synthetic league; CFBD `/coaches` not yet ingested) |
| 6 Opponent adjustment | Done | Weighted ridge with priors; temporally valid; SOS; conference strength. FCS opponents: FBS-only ingestion currently drops FBS-vs-FCS games — **gap** |
| 7 Player prediction | Done | From the game simulation: availability (injury table), usage shares, efficiency adjustments, full distributions |
| 8 Game prediction | Done | Calibrated win prob, score/margin/total distributions, OT prob, team stat distributions |
| 9 Simulation | Done | 2021+ OT rules, seeded, 10k sims ≈ 2–3 s; `simulation_runs` stored on request. Per-play `simulation_events` table exists but is not populated by default (size) |
| 10 ML stack | Partial | Ridge / empirical-Bayes, logistic calibration, XGBoost benchmark, Monte Carlo. No hyperparameter search or AutoGluon; model registry is the `model_versions` table |
| 11 Frontend | Done | 9 workspaces (Dashboard, Player, Team, Matchup Lab, Predictions, Simulations, Recruiting, Transfers, Audit) |
| 12 Visualisations | Mostly done | EPA/SR trends, pass vs rush, air-yard depth chart, radars, percentile bars, matchup heatmaps, score/margin distributions, calibration curves. Run-direction diagrams and coverage matrices need charting data. Filters: season, conference, team, position, week; down/distance/score-state splits are shown as metrics rather than interactive chart filters |
| 13 Validation | Done | Walk-forward, benchmarks, bootstrap CIs, splits, promotion gate, leakage tests |
| 14 Performance | Done for demo scale | Cached estimates, batched week simulation, indexes. Background job scheduling not implemented (use cron + CLI) |
| 16 Definition of done | 15/18 met on synthetic data | Not met: (2) real data ingested, (4/5) profiles from *real* data — blocked on API key; partially (11) opponent adjustment lacks FCS games |

## Verification results

### Tests
`python -m pytest -q` → **28 passed** (engine accounting identities, reproducibility, OT, calibration envelope,
per-lane params; EP monotonicity; success definition; registry sizes/uniqueness; unavailable metrics never filled;
percentile bounds; ridge recovery; opponent-adjustment direction; synthetic ground-truth parameter recovery; **leakage**
(future-data corruption leaves as-of estimates unchanged, Elo pre-game independent of own result, backtest features
as-of); forecast determinism and consistency; QB scenario ordering; all API endpoints; CFBD normalisers incl.
snake_case tolerance; Bayesian shrinkage).

### Parameter recovery (synthetic ground truth, 2025)
Correlation of estimated vs true unit parameters at season end: completion off/def 0.85/0.85, core-run off/def
0.83/0.84, pass rate 0.81, sacks 0.61/0.63, INT 0.45. By week 4: 0.64–0.78 for efficiency rates.

### Walk-forward backtest (576 games, synthetic 2024–2025, 400 sims/game, data strictly before kickoff)
| Model | Log loss [90% CI] | Brier | Accuracy |
|---|---|---|---|
| Simulator (raw) | 0.596 [0.575, 0.617] | 0.205 | 69.6% |
| Simulator (walk-forward calibrated) | 0.596 [0.567, 0.627] | 0.205 | 68.8% |
| Elo benchmark | 0.604 [0.574, 0.637] | 0.207 | 66.8% |
| OLS adj-EPA margin | 0.619 | 0.215 | 64.9% |
| XGBoost margin | 0.632 | 0.218 | 66.7% |

* Paired log-loss vs Elo: −0.008 [−0.025, +0.008] → **not promoted**; Elo stays `production`. The simulator is better on
  point estimate but the evidence is not yet decisive at n=576.
* Margin MAE 12.09 (Elo 12.10); total MAE 11.27; total bias −0.17 [−1.14, +0.78]; team score MAE 8.50.
* 80% interval coverage: margin 80.2%, total 79.7% (well calibrated).
* Calibration slope (calibrated sim) 0.92, intercept 0.04.
* Player projections (projected starters): pass yds MAE 61.7 (80% coverage 73.6%), rush yds 29.0 (70.9%), rec yds 24.8
  (77.7%), receptions 1.96 (81.5%), tackles 2.86 (77.3%). **Player intervals are too narrow** — parameter uncertainty is
  not yet propagated into the simulation.
* A fix made during this work: totals were initially biased −4.5 pts because engine scoring level was fixed; league-level
  calibration (`calibrate_level`) removed the bias and improved total MAE 11.86 → 11.27.

Caveat: these numbers measure the pipeline on data generated by a closely related model; they are a correctness and
calibration check, **not** evidence of real-world accuracy. Expect worse numbers on real data.

## Known limitations & recommended next steps (priority order)
1. **Run real ingestion** with a CFBD key; review normaliser mappings against live payloads (play types, stat types, drive results).
2. Ingest FCS opponents (division='fcs') so opponent adjustment covers FBS-vs-FCS games.
3. Propagate parameter uncertainty (draw team/player parameters from their posterior per simulation lane) to fix player-interval under-coverage.
4. Coaching data + scheme-transition effects; conference championship / bowl handling (neutral sites are supported).
5. Hyperparameter search over ridge λ / recency decay using the backtest; AutoGluon benchmark offline.
6. Charting feed integration for the 202 metrics currently flagged unavailable.
7. Persist per-play simulation events for drill-down; scheduled refresh jobs.
