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
| Prop calibrator | `models/sgp.py: calibration_factors, apply_factor` | Per-stat volume factor actual/simulated for projected starters who appeared, from **earlier weeks only**, shrunk toward 1. Yards are scaled; counts are binomially thinned / Poisson-topped so they stay integers. One draw per (player, stat), so every leg on a ticket sees the same lanes. |
| Pricing | `models/sgp.py: price` | Joint hit share across lanes, independence price, lift, fair American odds, 90% sampling CI; with book odds: EV per unit and Kelly. |
| Ticket builder | `models/sgp.py: build_tickets` | Beam search for the most positively correlated legs inside each size's leg-probability band (3: 50–85%, 5: 58–88%, 8: 66–92%). One leg per player (same-player stacks are blocked or heavily haircut by books, and their lift is mostly tautology), at most one side leg (ML or spread), realistic minimum lines. Searched on half the lanes, priced on the other half to remove winner's-curse bias. |

## Why 3 / 5 / 8

Each leg a book adds carries its margin. At a standard −110 two-way price the hold compounds:
2 legs ≈ 9%, 3 ≈ 13%, 5 ≈ 21%, 8 ≈ 31% if priced independently (books then add correlation adjustments on SGPs).
A long parlay is only +EV if every leg's edge, plus the correlation the book fails to price, outruns that compounding.

* **3-leg — the core.** Highest hit rate, the fastest to validate (a ~30% event is measurable in a few hundred
  tickets), and correlation lift of 2× on a 3-leg is large relative to the hold.
* **5-leg — the secondary.** Alt lines (≈65%) keep the hit rate near 20%. Most of the value is still correlation.
* **8-leg — the lottery ticket.** Only with alt-line legs (≈70–80% each). Hit rates around 10–15% need hundreds of
  tickets before the realised rate means anything; size it as a small fraction of the 3-leg stake. Judge the model on
  leg and 3-leg calibration, not on 8-leg P&L.

## Validation (walk-forward, real 2026 games)

`scripts/validate_sgp.py --season 2026 --weeks 3-10 --sims 2000` — parameters estimated strictly before each week,
anchored to the DraftKings line captured by ESPN (closing or near-closing; ESPN does not timestamp it), calibration
factors learned only from earlier weeks, tickets priced on held-out lanes. Results are written to
`model_validation` (`model_version = 'sgp_v1'`) and shown in the SGP Lab.

RESULTS_PLACEHOLDER

## Using it

```bash
export CFBQ_DB=data/cfb.db
python -m cfbquant.cli lines --season 2026                    # append opening + current DK lines (run daily)
python -m cfbquant.cli sgp board --game 401856887             # prop board + 3/5/8 tickets for one game
python -m cfbquant.cli sgp cards --season 2026 --week 11      # every game; writes data/cards/ + the ticket ledger
python -m cfbquant.cli sgp price --game 401856887 --legs slip.json --odds +600   # price a book SGP
python -m cfbquant.cli sgp settle                             # grade ledger tickets once games are final
python -m cfbquant.cli serve                                  # SGP Lab tab: board, slip, auto tickets
```

A leg (JSON): `{"kind": "player", "player_id": 4870906, "stat": "pass_yds", "op": "over", "line": 249.5}`;
`{"kind": "spread", "side": "home", "line": -6.5}`; `{"kind": "total", "op": "over", "line": 52.5}`;
`{"kind": "team_total", "side": "away", "op": "under", "line": 20.5}`; `{"kind": "ml", "side": "away"}`.
Player stats: `pass_yds pass_cmp pass_td rush_yds rush_td receptions rec_yds rec_td td (anytime) rush_rec_yds pass_rush_yds`.

## Limits — read before betting

* **No book prop or SGP prices.** ESPN's free feed has no CFB player props (the core `propBets` endpoint 404s), so the
  output is a *fair price*. A ticket is a bet only when the book pays more than `fair_american`; enter the book's
  price in the slip to get EV and a Kelly stake.
* **Availability.** No injury feed: a projected starter who is ruled out late voids the leg at most books (and the
  whole SGP at some). Check the depth chart before betting.
* **Usage tail.** Only eight target slots and four rushers per team are simulated; the calibrator absorbs the volume
  that real offences spread to WR5+/RB3+, but that is a correction, not a model of it.
* **Line timing.** Validation anchors to a near-closing line. Live cards anchor to whatever `cli lines` last captured;
  re-run it close to kickoff.
* ESPN data is for personal, non-commercial research (see `DATA_SOURCES.md`).
