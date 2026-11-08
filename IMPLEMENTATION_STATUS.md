# Implementation Status

_Last updated: 2026-11-08. All numbers below come from runs in this repository, not estimates._

## Props & same-game parlays (current focus) — `docs/SGP.md`
The simulator is now used where it has an edge: **joint distributions for player props and SGPs**, anchored to the
DraftKings spread/total so it never fights the line on the mean. Walk-forward on real 2026 games (weeks 3–10, 496
games, 2,196 tickets; correlation shrinkage fitted on weeks 3–6 and scored on 7–10):

| Size | Hit rate (OOS) | Model | Independence | Log loss model / indep |
|---|---|---|---|---|
| 3-leg | 22.9% ± 2.1 | 24.2% | 14.1% | 0.535 / 0.566 |
| 5-leg | 18.4% ± 2.1 | 18.0% | 11.4% | 0.477 / 0.498 |
| 8-leg | 13.1% ± 2.1 | 11.8% | 7.8% | 0.388 / 0.411 |

* Correlated tickets hit ~1.6× their independence price; the model prices that within sampling error. Team legs are
  calibrated within ~2 pp; alt-line player overs are still ~8 pp overconfident (residual low tail).
* What made the difference, in order: one-leg-per-player and realistic lines (removed tautological "lift"),
  garbage-time substitution, recency-weighted starters (QB1 appearance 79% → 91%), fitted 4th-down policy and pace,
  an early-exit mixture for starters, a shared game-environment shock, and walk-forward volume calibration.
* Not available from free data: book prop/SGP prices (ESPN has no CFB props), injuries. Output is a fair price;
  `cli sgp price --odds` gives EV against a book quote. Opening + current lines are now snapshotted (`cli lines`).
* Full-game accuracy after the engine changes (2026, 681 games): simulator log loss 0.523 (was 0.521), margin MAE 14.9,
  total MAE 12.7 — unchanged within noise; the sportsbook line (0.435) remains the bar for sides.

## Real-data results (ESPN, keyless)
Real data now comes from ESPN's public feeds, so no key is needed: 2,534 completed games (2024 through 2026 week 10),
452k plays, 860k player-role events, rosters, and DraftKings lines. Final scores reconstruct from play-level points within
2 points in 98.3% of games. The 2024 team rankings come out plausibly: Notre Dame, Ohio State, Texas, Penn State and Oregon top net EPA.

**Walk-forward backtest, 2026 (681 games, data strictly before kickoff, 2026 never used for tuning).**
The "+ talent" column adds CFBD recruiting/talent/portal (free key, 19 API calls).

| Model | Log loss, ESPN only | Log loss, + talent | Margin MAE (+ talent) |
|---|---|---|---|
| **DraftKings line** (ESPN pickcenter) | **0.435** | **0.435** | **11.7** |
| Polymarket US closing (486-game subset) | 0.506 | 0.506 | 11.7 |
| XGBoost margin | 0.501 | 0.496 | 13.7 |
| Elo (talent-informed preseason) | 0.515 | 0.498 | 15.3 |
| OLS adjusted-EPA margin | 0.502 | 0.502 | 13.8 |
| Simulator (raw; k=1.75 → k=1.25 re-tuned on 2025) | 0.543 | 0.521 | 14.5 |
| Simulator (walk-forward Platt) | 0.554 | 0.538 | 14.5 |

* **The talent prior is the single biggest improvement so far:** simulator −0.022, Elo −0.017, XGB −0.005 log loss out of sample.
* **The simulator is still the weakest in-house game model.** It trails XGB/Elo by ~0.025 and the sportsbook line by ~0.09.
  Against the spread it shows no edge.
* The walk-forward Platt calibrator *hurts* in a single-season test because it's fit on only a few weeks. It should be fit on
  the prior season instead (next-steps item 4).
* Diagnosed cause of the simulator gap: compressed team strengths (actual-vs-simulated margin slope 1.54 at k=1). The
  dispersion factor k is tuned on 2025 only (`scripts/tune_strength.py`).
* **Markets are far ahead.** Polymarket closing prices are well calibrated (512 FBS games: log loss 0.499, slope 1.02).

**Priority next steps (by expected gain):**
1. Anchor the simulator to the best validated margin model: per-matchup level shifts so the simulated mean margin equals
   the ridge/XGB margin. This keeps the coherent distributions and player projections and fixes headline accuracy.
2. Add market priors as features (or blend toward the line). Beating the closing line is the real bar; without that,
   the value is in distributions, player props and the research terminal, not picking winners.
3. Tune ridge λ and recency decay on 2025 and validate on 2026.
4. Fit the sim calibrator on the full prior season rather than the expanding in-season window.


## Verdict
A working end-to-end platform: ingestion (CFBD + synthetic), relational store, EP/EPA, 241 team metrics, 365 player
metrics, opponent adjustment, Bayesian talent priors, a consistent play-by-play simulator, game + player forecasts,
walk-forward validation with benchmarks and a promotion gate, a FastAPI backend and a terminal UI.
**It has only been exercised on the fictional synthetic league.** The CFBD path is implemented and fixture-tested but has
never touched the live API here because no key was available. Treat every real-data claim as unverified until
`cli ingest` has been run with a key.

## Polymarket US market data (added)
Live and working against the real public API (no key): live board, closing-price backfill, ladder-implied spreads/totals,
settlement-based calibration, model-vs-market comparison hooks, and a Markets tab. 9 dedicated tests. Contract semantics were
verified against real settlements. Details and data-quality findings (incl. a Polymarket score-feed vs settlement conflict)
are in `docs/MARKETS.md`.

**Real-data benchmark (2026 FBS, 512 games):** Polymarket US closing moneylines had a log loss of 0.499 [0.465, 0.530],
Brier 0.167 and calibration slope 1.02. The ladder-implied spread MAE was 11.4 and total MAE 12.5. That's the target for the
model once CFBD data is loaded. Not yet done: model-vs-market evaluation on real games, which needs CFBD data in the model DB, plus
streaming via the Markets WebSocket and team-prop benchmarking against player projections.

## Blocked / external dependencies
| Item | Status | What unblocks it |
|---|---|---|
| Live CFBD data | Not needed — ESPN keyless loader covers core data | CFBD key only adds recruiting/talent/portal |
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
3. ~~Propagate parameter uncertainty~~ — done for props/SGP (per-lane team/pace/usage draws, early-exit mixture); next: model the target/rush tail (WR5+/RB3+) explicitly instead of via the volume calibrator, and capture book SGP prices to measure realised EV.
4. Coaching data + scheme-transition effects; conference championship / bowl handling (neutral sites are supported).
5. Hyperparameter search over ridge λ / recency decay using the backtest; AutoGluon benchmark offline.
6. Charting feed integration for the 202 metrics currently flagged unavailable.
7. Persist per-play simulation events for drill-down; scheduled refresh jobs.
