# Player props and same-game parlays

The simulator's comparative advantage is not the side or the total — the sportsbook closing line beats every model we
have on those (2026 out-of-sample log loss 0.435 vs 0.50+). It is the **joint distribution**: one simulation produces
the score, game script and every projected starter's box score together, so the correlation between legs of a
same-game parlay is measured, not assumed. This module turns that into prop prices and 3/5/8-leg tickets.

## Pipeline

```
estimate (as of the week) ─► stack params ─► perturb per lane ─► anchor to DK line ─► simulate ─► calibrate volumes
                                              (team strength,     (per-game offence     (players,    (walk-forward,
                                               pace, usage)        level shifts)         game)        per stat)
                                                                                           │
                         prop board (main line, fair odds) ◄───────────────────────────────┤
                         ticket builder (3/5/8 legs)       ◄───────────────────────────────┤
                         price any user slip (+EV vs book)  ◄──────────────────────────────┘
```

| Step | Code | What it does |
|---|---|---|
| Market anchor | `models/props.py: anchored_simulate` | Shifts each offence's efficiency (same mapping as league level calibration) so simulated mean home/away points equal the line-implied team totals: home = (total − spread)/2. Secant iterations on a 400-lane subsample with an opposite-sign probe (own-offence slope plus the negative cross effect), converging to the sampling-noise floor (~0.5 pt). |
| Parameter uncertainty | `models/props.py: perturb` | Each lane draws its own offence and defence level offsets (SD 0.25 level units), pace multiplier (log-SD 0.07) and target/rush shares (Dirichlet, concentration 30). Without this every lane is the "average" team and player intervals are too narrow. |
| Garbage time | `sim/engine.py: BENCH_MAP` | Starters sit in blowouts (Q2 ≥45, Q3 ≥35, Q4 ≥24 point margins): QB1→QB2, RB1→RB2, WR1/2→WR3/4, TE1→TE2. Without it, "team wins big + QB over" is overstated. |
| Starter selection | `analytics/team_params.py: _usage` | Recency-weighted usage (QB decay 0.35 per game, others 0.6). ESPN has no injury feed, so who played last week is the best availability signal; raised projected-QB1 appearance from 79% to 89%. |
| 4th-down policy | `sim/engine.py` | Go-for-it log-odds fitted on 24,111 real 2024–25 FBS 4th downs (the old rule went for it 2–4× too often in FG range, which starved the sim of field goals). |
| Pace | `analytics/team_params.py: calibrate_pace` | League pace multiplier so simulated scrimmage snaps per team-game match recent real games (was 72 vs 66). |
| Early-exit mixture | `models/sgp.py: EXIT_P` | Per lane, a projected starter's game is cut short (injury, benching, role collapse) with probability QB 8%, RB 10%, WR 3%, TE 0%; he keeps a uniform 0–50% share of his stats, all stats together. Fitted on weeks 3–6 only, where 15% of appearing QB/RB yardage outcomes fell below the simulated 5th percentile. |
| Game environment | `models/props.py: perturb` (`env_sd`) | A shared offence shock per lane (SD 0.45 level units) plus a shared pace component: weather/officiating/tempo move both teams together, widening the total without touching the margin (simulated total SD was 13.3 vs 15.6 real around the closing total). |
| Correlation shrinkage | `models/sgp.py: price` (`corr_shrink`) | p = p_indep × lift^α. α fitted by ticket log loss; 0.8 on weeks 3–6 and on all weeks. Stored with the prop calibrator. |
| Prop calibrator | `models/sgp.py: calibration_factors, apply_factor` | Per-stat volume factor actual/simulated for projected starters who appeared, from **earlier weeks only**, shrunk toward 1. Yards are scaled; counts are binomially thinned / Poisson-topped so they stay integers. One draw per (player, stat), so every leg on a ticket sees the same lanes. |
| Pricing | `models/sgp.py: price` | Joint hit share across lanes, independence price, lift, fair American odds, 90% sampling CI; with book odds: EV per unit and Kelly. |
| Ticket builder | `models/sgp.py: build_tickets` | Beam search for the most positively correlated legs inside each size's leg-probability band (3: 50–85%, 5: 58–88%, 8: 66–92%). One leg per player (same-player stacks are blocked or heavily haircut by books, and their lift is mostly tautology), at most one side leg (ML or spread), realistic minimum lines. Searched on half the lanes, priced on the other half to remove winner's-curse bias. |

## Why 3 / 5 / 8

Each leg a book adds carries its margin. At a standard −110 two-way price the hold compounds:
2 legs ≈ 9%, 3 ≈ 13%, 5 ≈ 21%, 8 ≈ 31% if priced independently (books then add correlation adjustments on SGPs).
A long parlay is only +EV if every leg's edge, plus the correlation the book fails to price, outruns that compounding.

* **3-leg — the core.** Highest hit rate, the fastest to validate (a ~25% event is measurable in a few hundred
  tickets), and a validated ~1.6× correlation lift on a 3-leg is large relative to the hold.
* **5-leg — the secondary.** Alt lines (≈65%) keep the hit rate near 20%. Most of the value is still correlation.
* **8-leg — the lottery ticket.** Only with alt-line legs (≈70–80% each). Hit rates around 10–15% need hundreds of
  tickets before the realised rate means anything; size it as a small fraction of the 3-leg stake. Judge the model on
  leg and 3-leg calibration, not on 8-leg P&L.

## Validation (walk-forward, real 2026 games)

`scripts/validate_sgp.py --season 2026 --weeks 3-10 --sims 2000` — parameters estimated strictly before each week,
anchored to the DraftKings line captured by ESPN (closing or near-closing; ESPN does not timestamp it), calibration
factors learned only from earlier weeks, tickets priced on held-out lanes. Results are written to
`model_validation` (`model_version = 'sgp_v1'`) and shown in the SGP Lab.

**Out-of-sample tickets** (α = 0.8 fitted on weeks 3–6, scored on weeks 7–10; 233 games; void tickets excluded):

| Size | Tickets | Hit rate (±1 SE) | Model, shrunk | Model, raw | Independence | Log loss: model / independence |
|---|---|---|---|---|---|---|
| 3-leg | 415 | **22.9%** ± 2.1 | 24.2% | 27.8% | 14.1% | 0.535 / 0.566 |
| 5-leg | 326 | **18.4%** ± 2.1 | 18.0% | 20.2% | 11.4% | 0.477 / 0.498 |
| 8-leg | 245 | **13.1%** ± 2.1 | 11.8% | 13.1% | 7.8% | 0.388 / 0.411 |

Real tickets hit **~1.6× the independence price at every size**, and the joint model beats independence on log loss
at every size. That gap is the correlation a same-game parlay is built on; whether it is *money* depends on how much
of it the book already charges for (unknown from free data — compare `fair_american` with the book's SGP price).

**Legs** (weeks 7–10): team legs (sides, totals, team totals) are calibrated within ~2 pp at every probability level.
Player overs at main lines are calibrated (63% predicted / 64% hit); alt-line overs remain overconfident
(78% predicted / 69% hit in the 75–85% band) — the residual low tail. Treat 75%+ player legs as ~70%.

**Player props** (projected starters who appeared, calibrated, weeks 7–10):

| Stat | n | Below sim p5 | Above sim p95 | 80% interval coverage | Mean PIT |
|---|---|---|---|---|---|
| Pass yds | 414 | 7.7% | 4.6% | 77.8% | 0.47 |
| Completions | 414 | 7.2% | 6.0% | 73.4% | 0.46 |
| Rush yds | 848 | 9.7% | 7.9% | 70.5% | 0.48 |
| Rush+rec yds | 434 | 7.6% | 7.4% | 74.4% | 0.46 |
| Rec yds | 1,260 | 5.9% | 7.5% | 76.8% | 0.50 |
| Receptions | 1,260 | 3.9% | 4.0% | 83.7% | 0.50 |

Before the early-exit mixture and environment shock, 15% of QB/RB yardage outcomes fell below the simulated 5th
percentile and 3-leg tickets were over-predicted by 4 pp (all weeks: 28.8% predicted, 25.1% hit).

**Game level:** margin 80% coverage 79–80%, total 72–75% (simulated total SD 14.5 vs 16.0 real — still slightly
narrow). Anchor error vs the line: 0.65 pt margin, 0.55 pt total (sampling noise). Projected starters appeared:
QB1 91%, RB1 94%, WR1 96%, WR2 93%, TE1 80%; tickets with a non-appearing player are void (9% of 3-leg, 25% of
5-leg, 33% of 8-leg).

**Walk-forward volume factors** (final, applied live): pass yds 0.96, completions 0.98, pass TD 0.90, rush yds 0.91,
rush TD 0.98, receptions 0.91, rec yds 0.91, rec TD 0.87 — the simulator still gives its eight target slots and four
rushers slightly more volume than real offences, which spread it to deeper reserves.

## Using it

```bash
export CFBQ_DB=data/cfb.db
python -m cfbquant.cli lines --season 2026                    # append opening + current DK lines (run daily)
python -m cfbquant.cli sgp board --game 401856887             # prop board + 3/5/8 tickets for one game
python -m cfbquant.cli sgp cards --season 2026 --week 11      # every game; writes data/cards/ + the ticket ledger
python -m cfbquant.cli sgp price --game 401856887 --legs slip.json --odds +600   # price a book SGP
python -m cfbquant.cli sgp settle                             # grade ledger tickets once games are final
python -m cfbquant.cli availability snapshot --season 2026 --week 11   # roster removal + game-log flags
python -m cfbquant.cli availability import --season 2026 --week 11 --file report.csv   # official report
python -m cfbquant.cli markets live                           # Polymarket books for upcoming games (team legs)
python -m cfbquant.cli markets close                          # closing books + team-total ladders for finals
python scripts/validate_market_sgp.py --season 2026 --weeks 1-10      # team-combo correlation vs Polymarket
python -m cfbquant.cli serve                                  # SGP Lab tab: board, slip, auto tickets
```

A leg (JSON): `{"kind": "player", "player_id": 4870906, "stat": "pass_yds", "op": "over", "line": 249.5}`;
`{"kind": "spread", "side": "home", "line": -6.5}`; `{"kind": "total", "op": "over", "line": 52.5}`;
`{"kind": "team_total", "side": "away", "op": "under", "line": 20.5}`; `{"kind": "ml", "side": "away"}`.
Player stats: `pass_yds pass_cmp pass_td rush_yds rush_td receptions rec_yds rec_td td (anytime) rush_rec_yds pass_rush_yds`.

## Availability (the injury gap)

There is no open CFB injury feed: ESPN's injuries endpoints are empty for college, core depth charts return "not
supported", and the conference availability reports (Big Ten, SEC, ACC, ...) sit behind bot walls or robots.txt
rules that disallow AI agents, so they are not scraped. `cfbquant/ingest/availability.py` combines what is available:

| Signal | Source | Effect | Walk-forward evidence (2026 wk 3–10, projected starters) |
|---|---|---|---|
| Official report | `cli availability import --file report.csv` (columns team, player, status) | OUT/DOUBTFUL removed before simulation; QUESTIONABLE flagged | — (authoritative) |
| Roster removal | ESPN current roster: player with real 2026 production is gone **and** missed his team's last game | OUT | removal alone is noise (112 of 159 "missing" players played the prior week) |
| Missed last game | game log | kept in the sim, **kept off auto tickets**, shown for a manual check | appeared 62% vs 92% unflagged |
| Missed 2 of last 3 | game log | same | appeared 64% |
| Low volume last game | game log (< 40% of his average) | warning only | appeared 81% |

`cli availability snapshot --season 2026 --week N` refreshes the ESPN signals (and adds ESPN roster players missing
from the DB, so an OUT starter always has a backup to promote). In the SGP Lab, **mark OUT** on any starter re-runs
the simulation with the next man up (QB changes also move the team's passing efficiency) and re-builds every ticket.
An OUT player can no longer be used as anyone's backup.

Example (week 11): the card's Michigan QB1 is Tommy Carr. The data support it: Carr started weeks 8–10, and Bryce
Underwood is off the current ESPN roster and last appeared in week 8, so he is removed automatically. Mark Carr OUT
and the sim promotes the QB2 and the tickets change. The anchor still uses the posted line: if a starter is ruled
out late, re-run `cli lines` once the book moves.

## Polymarket (the book-price gap)

Polymarket US has **no player props and no parlays**, and its team stat props (yards, TDs, FGs) have empty books,
so it cannot show what a sportsbook charges for an SGP. It does list moneylines, spread and total ladders and
team-points ladders with tradeable books up to kickoff. `cfbquant/models/market_sgp.py` uses them in three ways:

1. **No-vig marginals for team legs.** A rung's own mid is used when its book is ≤12c wide; otherwise a normal fit
   through the whole ladder (no extrapolation beyond 3 points past the quoted rungs).
2. **Market-anchored fair price** (`price_market`). Each team leg's threshold in the simulation is moved so that leg
   hits at exactly the Polymarket probability (quantile mapping, a copula), then the joint is counted lane by lane.
   The market sets the marginals and the simulation supplies only the dependence (shrunk with the same α). Player
   legs keep the model marginal. Tickets show both prices plus a **price to beat** (the more conservative of the
   two). The slip returns a **BET / PASS** verdict: bet only if the book's odds give positive EV under both prices.
3. **A real-money test of the correlation** (`scripts/validate_market_sgp.py`, results in `model_validation` as
   `sgp_market_v1`).

**Validation:** 486 completed 2026 games (weeks 1–10), Polymarket closing mids (last book sample before kickoff),
every orientation of each 2-leg team combo at the market's own coin-flip rungs, simulation estimated before each
week. Log loss (lower is better):

| Combo | n | Market independence | Market × sim dependence | Model only | Gain vs independence (±1 SE, game-clustered) |
|---|---|---|---|---|---|
| Team total + opposing side | 3,748 | 0.5550 | **0.5140** | 0.5153 | +0.041 ± 0.003 |
| Team total + game total | 3,748 | 0.5569 | **0.5198** | 0.5217 | +0.037 ± 0.003 |
| Side + total | 3,888 | 0.5305 | **0.5303** | 0.5326 | +0.000 ± 0.001 |

For the positively correlated pairs (simulated lift > 1.3, n = 3,624) the real hit rate was **36.6%**. The
market-anchored price said 34.0% and market independence said 24.6%. The negatively correlated pairs hit 12.5%
(priced at 14.1%, independence 24.6%). So the simulator's team-level dependence is real and sized correctly. The
best α for team-only pairs is ≈1.0 (0.95 when fitted on weeks 1–5 and scored on 6–10), so the live α = 0.8 is
slightly conservative on these legs. Side × total is close to independent in CFB, and both the model and the market
treat it that way.

What this does **not** prove: the player-leg correlation is validated only against realised outcomes (the ticket
tables above), not against a market. The edge on a ticket still depends on what the book charges, so enter its
SGP odds in the slip.

## Limits — read before betting

* **No book prop or SGP prices.** Neither ESPN (the core `propBets` endpoint 404s for CFB) nor Polymarket lists CFB
  player props or parlays. Team legs now carry a no-vig Polymarket price, but a ticket is still a bet only when the
  book pays more than the *price to beat*. Enter the book's odds in the slip and fire only on a BET verdict.
* **Availability.** Signals, not a feed (see above). Game-log-flagged players are kept off auto tickets. A late
  scratch still voids the leg at most books (and the whole SGP at some). Import the official report when it posts.
* **Usage tail.** Only eight target slots and four rushers per team are simulated; the calibrator absorbs the volume
  that real offences spread to WR5+/RB3+, but that is a correction, not a model of it.
* **Line timing.** Validation anchors to a near-closing line. Live cards anchor to whatever `cli lines` last captured;
  re-run it close to kickoff.
* ESPN data is for personal, non-commercial research (see `DATA_SOURCES.md`).
