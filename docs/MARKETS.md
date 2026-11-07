# Polymarket US market data

Free, keyless public market data from **Polymarket US** (CFTC-regulated exchange), used as an external benchmark
and as a live market board. Read-only: the platform never places orders and needs no credentials.

* Docs: https://docs.polymarket.us (index: https://docs.polymarket.us/llms.txt)
* Public gateway: `https://gateway.polymarket.us` (no key). Trading API `api.polymarket.us` is **not** used.
* CFB: league slug `cfb`, `GET /v2/leagues/cfb` → `activeSeriesId` (2026 season = series 225, slug `cfb-2026`).

## Commands

```bash
python -m cfbquant.cli markets live                    # upcoming + in-play board (1 request per game)
python -m cfbquant.cli markets close --since 2026-08-20T00:00:00Z   # closing prices + settlements (cached)
python -m cfbquant.cli markets calibration             # how calibrated were closing moneylines?
```

Data goes to `data/polymarket.db` (`CFBQ_MARKETS_DB`), **separate from model databases**, so real market prices are
never written into the synthetic demo DB. Immutable responses (closed events, pre-kickoff history, settlements) are cached
in `data/cache/polymarket/`, so a season backfill only runs once.

## Endpoints used

| Purpose | Endpoint |
|---|---|
| Games + every market with live `[bid, ask]` (`outcomePrices`) | `GET /v1/events?seriesId=…&closed=…&startDateMin/Max=…` |
| Best bid/offer, last, volume, open interest | `GET /v1/markets/{slug}/bbo` |
| Book-derived history (`longPrice`≈ask, `shortPrice`≈1−bid) | `GET /v1/price-history?symbol=…` (custom window → ~1-min points) |
| Settlement (1 = YES, 0 = NO) | `GET /v1/markets/{slug}/settlement` |

## Contract semantics (verified against settlements, not titles)

Verified on UNC 15 – TCU 10 (2026-08-29) across the moneyline, 7 spread rungs and 7 total rungs:

| Market | `sportsMarketType` | YES pays when |
|---|---|---|
| Moneyline | `football_team_full_game_winner` | the `long: true` side's team wins |
| Spread (alternate-line ladder, ~40 rungs) | `football_team_full_game_spread` | margin(long team) + `line` > 0 |
| Total (ladder, ~20 rungs) | `football_team_full_game_total` | total points > `line` |
| Team props | `football_team_total_pass_yards`, `…_rush_yards`, `…_touchdowns`, … | that team's stat > `line` |

The market `title` contradicts the YES side for positive spread lines (e.g. "TCU wins by over 4.5" on an instrument
whose YES is "UNC +4.5"). Titles are never parsed.

## Derived quantities

* **Prices:** order-book mid `(bid + ask)/2`. Quotes wider than 30¢ (e.g. the bid 0 / ask 1 left after a game) are discarded.
* **Closing price:** last book sample at or before kickoff from a 4-hour pre-kickoff history window.
* **Implied margin/total distribution:** weighted probit regression on the ladder,
  `P(margin_home > −L) = Φ((μ + L)/σ)` and `P(total > L) = Φ((μ − L)/σ)`, using rungs with 3% < p < 97%.
  This gives the market's mean margin, mean total, **and their spreads (σ)**.
  On the 2026 data, fits average ~1.2 pp RMSE, and the implied margin SD is ~14–16 points, the same scale as the simulator's.
* **In-play:** prices are conditional on the score, so ladders are not fitted in-play; only win probability is shown, labelled in-play.
* **Rate limits:** documented 20 req/s per IP, but enforcement is stricter in practice. The client starts at ~3 req/s with
  AIMD backoff. A closing snapshot fetches only informative rungs: the moneyline, 5 spread rungs centred on the
  moneyline-implied spread, and 6 total rungs found coarse-to-fine. That's ~15 calls per game.

## Data-quality findings

* **Polymarket's score feed can contradict its own settlement.** On Maine vs Rhode Island (2026-10-31) the feed reports
  Maine 13–7, but the Maine moneyline settled 0 and traded to $0.01. Therefore:
  * model evaluation always uses CFBD final scores, never Polymarket scores;
  * market outcomes use settlement;
  * every disagreement is stored as `pm_events.score_conflict` and counted in `/api/markets/calibration`.
* The bare `score` string is not reliably ordered. Per-team `periodScores` (keyed by team id) are used when present.
* There's no home/away flag. CFB teams are listed away-first (`ordering: "away"`). For games mapped to the model DB,
  orientation comes from the model's game record (`pm_home_is_home`), never from Polymarket order.
* Some FCS teams have no conference. The closing backfill defaults to FBS-vs-FBS games (`fbs_only=True`).

## Results: 2026 FBS closing prices (real data, 512 games, Aug 29 – Nov 7)

| Measure | Value |
|---|---|
| Moneyline log loss [90% CI] | 0.499 [0.465, 0.530] |
| Brier | 0.167 |
| Favorite accuracy | 76.5% |
| Calibration slope / intercept (logit) | 1.02 / +0.15 |
| Favorites' win rate − price | +2.1 pp [−0.7, +5.1] |
| Ladder-implied spread: MAE / bias | 11.4 pts / +1.2 |
| Actual margin residual SD vs implied SD | 14.4 vs 13.6 → implied 80% interval covers 77% |
| Ladder-implied total: MAE / bias | 12.5 pts / +0.7 |
| Score-feed / settlement conflicts (FBS) | 0 |

Takeaways:
* Closing moneylines are well calibrated (slope ≈ 1).
* The implied margin distribution is slightly too narrow (77% vs 80% coverage).
* "Home" here means Polymarket's second-listed team: neutral sites aren't identified, so don't read the +0.15
  intercept as home-field mispricing.
* **This is the bar a real-data model has to clear:** log loss ≈ 0.50, spread MAE ≈ 11.4. The synthetic-league backtest
  numbers in `IMPLEMENTATION_STATUS.md` are not comparable.

## How it plugs into the platform

* **Mapping:** Polymarket events are matched to model games by unordered normalised team names, within 36 h of kickoff (`map_events_to_games`).
* **Forecasts:** `forecast_game()` adds a `market` block (model − market win prob / margin / total) when the active DB is real and the game maps.
* **Backtest:** a `polymarket_close` benchmark is added (log loss, Brier, margin MAE, total MAE on the market subset),
  plus a paired model-minus-market log loss. A "value" diagnostic records return per contract at mid when the model
  disagrees by ≥5 pp, ignoring fees. It's a diagnostic, not a strategy claim.
* **Never on synthetic data:** fictional teams can't map to real markets (`tests/test_polymarket.py::test_market_never_attached_to_synthetic`).
* **UI:** the Markets tab shows the board, calibration of closing prices, and an event drill-down (spread/total ladders with
  fitted curves, 7-day win-probability history, team props). The Predictions view shows market vs model when linked.
