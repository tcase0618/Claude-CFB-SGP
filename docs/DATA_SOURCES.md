# Data sources

| Source | Key? | What we take | Status |
|---|---|---|---|
| **ESPN public JSON** (site + core APIs) | **No** | schedules, scores, venues, neutral sites, weather, rankings, every play with down/distance/field position, athlete IDs by role on each play, box scores, season rosters (position, size, class), sportsbook lines (DraftKings via pickcenter) | **Primary real-data source** (`cli espn`) |
| **Polymarket US** public gateway | **No** | live and closing prediction-market prices, spread and total ladders, settlements | Market benchmark (`cli markets`) — see `MARKETS.md` |
| CollegeFootballData API | Free key | same core data plus recruiting ratings, talent composite, transfer portal | Optional (`cli ingest`); not required |
| sportsdataverse / cfbfastR release files | No | historical play-by-play parquet | Not reachable from this environment (GitHub release downloads blocked by the session's network scope); usable elsewhere |
| 247Sports / On3 / Sports-Reference | — | recruiting, historical | **Not used**: their terms prohibit scraping |

## ESPN (keyless)

```bash
python -m cfbquant.cli --db data/cfb.db espn --seasons 2024,2025,2026
CFBQ_DB=data/cfb.db python -m cfbquant.cli analyze
CFBQ_DB=data/cfb.db python -m cfbquant.cli backtest --test 2025,2026
CFBQ_DB=data/cfb.db python -m cfbquant.cli serve
```

* **Cost:** ~3 requests per game (plays, summary, sometimes a second plays page), plus athlete lookups for involved players.
  A full season takes ~3–5 min; every completed-game response is cached in `data/cache/espn/`.
* **Coverage:** FBS games plus every FCS team that plays an FBS opponent. FCS teams get an unpenalised division effect in
  opponent adjustment, and a lower Elo prior, because they are only seen against FBS opponents.
* **Season selection:** the scoreboard ignores `year=` and silently returns the current season; `dates=YYYY` is correct.
  Each event's `season.year` is checked against the request.
* **Scoring:** play points come from ESPN's per-play `scoreValue` plus play semantics (the scoring team comes from the
  play's `team` ref, the PAT is parsed from text, defaulting to a made kick). ESPN's running score fields can be mis-sequenced
  around penalties, so they aren't used. Model evaluation always uses the game's final score.
* **Parsed from play text** (classification `estimated`): air yards from "caught at / thrown to" spots relative to the
  line of scrimmage (bounded −15…75), pass depth and location ("short/deep left/middle/right"), Shotgun/No-Huddle tags, and
  breakups. Only part of ESPN's feed uses this charting-style text (~55–60% of passes), so these metrics have partial
  coverage and say so.
* **Not available from ESPN:** recruiting ratings and talent composite (the talent prior falls back to zero), transfer
  portal, injuries (availability modelling then relies on who actually played), snap counts.
* **Terms:** ESPN publishes no developer terms for these feeds, and its site terms restrict automated and commercial reuse.
  This loader is for personal, non-commercial research: it rate-limits, caches each game once, and attributes ESPN.
  Use licensed data for anything commercial.
