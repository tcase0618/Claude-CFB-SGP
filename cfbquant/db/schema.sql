-- CFB Quant Intelligence Platform — relational schema (SQLite dialect, portable to Postgres).
-- Every source-derived row carries `source` so observed / calculated / synthetic data is traceable.
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_sources (
    source_id     TEXT PRIMARY KEY,           -- 'cfbd', 'synthetic', 'manual'
    description   TEXT NOT NULL,
    is_synthetic  INTEGER NOT NULL DEFAULT 0,
    license_note  TEXT,
    last_refreshed TEXT
);

CREATE TABLE IF NOT EXISTS conferences (
    conference_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    division      TEXT                          -- 'fbs' | 'fcs'
);

CREATE TABLE IF NOT EXISTS teams (
    team_id       INTEGER PRIMARY KEY,
    school        TEXT NOT NULL,
    mascot        TEXT,
    abbreviation  TEXT,
    color         TEXT,
    alt_color     TEXT,
    venue_id      INTEGER,
    source        TEXT NOT NULL REFERENCES data_sources(source_id)
);

-- Conference affiliation is stored per season so realignment is preserved historically.
CREATE TABLE IF NOT EXISTS team_seasons (
    team_id       INTEGER NOT NULL REFERENCES teams(team_id),
    season        INTEGER NOT NULL,
    conference_id TEXT REFERENCES conferences(conference_id),
    division      TEXT NOT NULL DEFAULT 'fbs',
    talent_composite REAL,
    returning_production REAL,              -- share of prior-season production (0..1)
    head_coach_id INTEGER,
    PRIMARY KEY (team_id, season)
);

CREATE TABLE IF NOT EXISTS venues (
    venue_id   INTEGER PRIMARY KEY,
    name       TEXT, city TEXT, state TEXT,
    capacity   INTEGER, grass INTEGER, dome INTEGER,
    elevation_m REAL, latitude REAL, longitude REAL, timezone TEXT
);

CREATE TABLE IF NOT EXISTS games (
    game_id      INTEGER PRIMARY KEY,
    season       INTEGER NOT NULL,
    week         INTEGER NOT NULL,
    season_type  TEXT NOT NULL DEFAULT 'regular',
    start_date   TEXT NOT NULL,                -- ISO-8601 kickoff (UTC)
    neutral_site INTEGER NOT NULL DEFAULT 0,
    conference_game INTEGER NOT NULL DEFAULT 0,
    venue_id     INTEGER,
    home_id      INTEGER NOT NULL REFERENCES teams(team_id),
    away_id      INTEGER NOT NULL REFERENCES teams(team_id),
    home_points  INTEGER,
    away_points  INTEGER,
    completed    INTEGER NOT NULL DEFAULT 0,
    temperature_f REAL, wind_mph REAL, weather_desc TEXT,
    source       TEXT NOT NULL REFERENCES data_sources(source_id)
);
CREATE INDEX IF NOT EXISTS ix_games_season_week ON games(season, week);
CREATE INDEX IF NOT EXISTS ix_games_home ON games(home_id, season);
CREATE INDEX IF NOT EXISTS ix_games_away ON games(away_id, season);

CREATE TABLE IF NOT EXISTS drives (
    drive_id     INTEGER PRIMARY KEY,
    game_id      INTEGER NOT NULL REFERENCES games(game_id),
    offense_id   INTEGER NOT NULL,
    defense_id   INTEGER NOT NULL,
    drive_number INTEGER,
    start_yards_to_goal INTEGER,
    end_yards_to_goal   INTEGER,
    plays        INTEGER,
    yards        INTEGER,
    result       TEXT,                       -- TD, FG, PUNT, TURNOVER, DOWNS, MISSED_FG, END_HALF, SAFETY
    points       INTEGER,
    start_period INTEGER
);
CREATE INDEX IF NOT EXISTS ix_drives_game ON drives(game_id);

CREATE TABLE IF NOT EXISTS plays (
    play_id       INTEGER PRIMARY KEY,
    game_id       INTEGER NOT NULL REFERENCES games(game_id),
    drive_id      INTEGER,
    season        INTEGER NOT NULL,
    week          INTEGER NOT NULL,
    play_number   INTEGER,
    offense_id    INTEGER NOT NULL,
    defense_id    INTEGER NOT NULL,
    home_id       INTEGER NOT NULL,
    period        INTEGER NOT NULL,
    clock_seconds INTEGER,                    -- seconds remaining in period
    offense_score INTEGER, defense_score INTEGER,
    down          INTEGER, distance INTEGER, yards_to_goal INTEGER,
    play_type     TEXT NOT NULL,              -- normalized: pass, rush, sack, punt, field_goal, kickoff, pat, two_point, penalty, kneel, spike, other
    yards_gained  INTEGER,
    air_yards     REAL,                       -- NULL when not charted (CFBD does not provide)
    pass_location TEXT,                       -- left/middle/right; NULL when not charted
    completion    INTEGER, interception INTEGER, sack INTEGER, fumble_lost INTEGER,
    touchdown     INTEGER, first_down INTEGER, scramble INTEGER,
    points        INTEGER DEFAULT 0,          -- points scored by offense on play (negative for def score)
    penalty_yards INTEGER,
    fg_distance   INTEGER, fg_made INTEGER,
    ep_before     REAL, ep_after REAL, epa REAL,
    success       INTEGER,
    garbage_time  INTEGER DEFAULT 0,
    play_text     TEXT,
    source        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_plays_game ON plays(game_id);
CREATE INDEX IF NOT EXISTS ix_plays_off ON plays(offense_id, season, week);
CREATE INDEX IF NOT EXISTS ix_plays_def ON plays(defense_id, season, week);
CREATE INDEX IF NOT EXISTS ix_plays_type ON plays(season, play_type);

-- Player attribution per play (passer, rusher, target, receiver, tackler, sacker, interceptor, pbu, forced_fumble, kicker, punter, returner)
CREATE TABLE IF NOT EXISTS play_events (
    play_id   INTEGER NOT NULL REFERENCES plays(play_id),
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    team_id   INTEGER NOT NULL,
    role      TEXT NOT NULL,
    value     REAL,
    PRIMARY KEY (play_id, player_id, role)
);
CREATE INDEX IF NOT EXISTS ix_play_events_player ON play_events(player_id);

CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    first_name  TEXT, last_name TEXT,
    position    TEXT,                          -- QB RB WR TE OT OG C DL EDGE LB CB S K P LS
    height_in   INTEGER, weight_lb INTEGER,
    birth_year  INTEGER,
    hometown    TEXT,
    source      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rosters (
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    team_id   INTEGER NOT NULL REFERENCES teams(team_id),
    season    INTEGER NOT NULL,
    jersey    INTEGER,
    position  TEXT,
    class_year INTEGER,                        -- 1=FR .. 5=GR
    PRIMARY KEY (player_id, season, team_id)
);
CREATE INDEX IF NOT EXISTS ix_rosters_team ON rosters(team_id, season);

CREATE TABLE IF NOT EXISTS depth_charts (
    team_id INTEGER NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
    position TEXT NOT NULL, depth INTEGER NOT NULL, player_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (team_id, season, week, position, depth)
);

CREATE TABLE IF NOT EXISTS player_seasons (
    player_id INTEGER NOT NULL, season INTEGER NOT NULL, team_id INTEGER NOT NULL,
    games INTEGER, stats_json TEXT NOT NULL,    -- aggregated counting stats
    PRIMARY KEY (player_id, season, team_id)
);

CREATE TABLE IF NOT EXISTS player_games (
    player_id INTEGER NOT NULL, game_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
    pass_att INTEGER DEFAULT 0, pass_cmp INTEGER DEFAULT 0, pass_yds INTEGER DEFAULT 0,
    pass_td INTEGER DEFAULT 0, pass_int INTEGER DEFAULT 0, sacks_taken INTEGER DEFAULT 0,
    rush_att INTEGER DEFAULT 0, rush_yds INTEGER DEFAULT 0, rush_td INTEGER DEFAULT 0,
    targets INTEGER DEFAULT 0, receptions INTEGER DEFAULT 0, rec_yds INTEGER DEFAULT 0, rec_td INTEGER DEFAULT 0,
    tackles REAL DEFAULT 0, tfl REAL DEFAULT 0, def_sacks REAL DEFAULT 0, def_int INTEGER DEFAULT 0,
    pbu INTEGER DEFAULT 0, forced_fumbles INTEGER DEFAULT 0,
    fga INTEGER DEFAULT 0, fgm INTEGER DEFAULT 0, xpa INTEGER DEFAULT 0, xpm INTEGER DEFAULT 0,
    punts INTEGER DEFAULT 0, punt_yds INTEGER DEFAULT 0,
    source TEXT NOT NULL,
    PRIMARY KEY (player_id, game_id)
);
CREATE INDEX IF NOT EXISTS ix_player_games_game ON player_games(game_id);

CREATE TABLE IF NOT EXISTS player_participation (
    player_id INTEGER NOT NULL, game_id INTEGER NOT NULL,
    snaps INTEGER, snap_share REAL, routes INTEGER, pass_block_snaps INTEGER,
    source TEXT NOT NULL,
    PRIMARY KEY (player_id, game_id)
);

CREATE TABLE IF NOT EXISTS coaches (
    coach_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT
);
CREATE TABLE IF NOT EXISTS team_personnel (
    team_id INTEGER, season INTEGER, coach_id INTEGER, role TEXT,
    PRIMARY KEY (team_id, season, coach_id, role)
);

-- Charting tables: populated only from licensed charting sources. Empty => metrics flagged unavailable.
CREATE TABLE IF NOT EXISTS offensive_formations (
    play_id INTEGER PRIMARY KEY, personnel TEXT, formation TEXT, shotgun INTEGER,
    motion INTEGER, play_action INTEGER, rpo INTEGER, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS defensive_formations (
    play_id INTEGER PRIMARY KEY, personnel TEXT, box_count INTEGER, coverage TEXT,
    man_zone TEXT, blitz INTEGER, pressure INTEGER, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS play_charting (
    play_id INTEGER NOT NULL, player_id INTEGER NOT NULL, metric TEXT NOT NULL, value REAL,
    source TEXT NOT NULL, PRIMARY KEY (play_id, player_id, metric)
);

CREATE TABLE IF NOT EXISTS injuries (
    player_id INTEGER NOT NULL, team_id INTEGER NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
    status TEXT NOT NULL,                       -- OUT, DOUBTFUL, QUESTIONABLE, PROBABLE
    reported_at TEXT, source TEXT NOT NULL,
    PRIMARY KEY (player_id, season, week)
);

CREATE TABLE IF NOT EXISTS recruiting_profiles (
    player_id INTEGER,                          -- NULL if recruit never matched to a roster player
    recruit_id INTEGER PRIMARY KEY,
    name TEXT, class_year INTEGER NOT NULL, position TEXT,
    stars INTEGER, rating REAL, national_rank INTEGER,
    committed_team_id INTEGER, source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_recruit_team ON recruiting_profiles(committed_team_id, class_year);

CREATE TABLE IF NOT EXISTS transfer_history (
    transfer_id INTEGER PRIMARY KEY,
    player_id INTEGER, name TEXT, position TEXT, season INTEGER NOT NULL,
    from_team_id INTEGER, to_team_id INTEGER,
    stars INTEGER, rating REAL, eligibility TEXT, transfer_date TEXT, source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_definitions (
    metric_id     TEXT PRIMARY KEY,
    scope         TEXT NOT NULL,                -- player_universal | player_position | team
    category      TEXT NOT NULL,
    name          TEXT NOT NULL,
    positions     TEXT,                          -- comma list or ALL
    formula       TEXT NOT NULL,
    units         TEXT, denominator TEXT,
    direction     INTEGER NOT NULL,              -- +1 higher is better, -1 lower is better, 0 neutral
    classification TEXT NOT NULL,                -- observed | calculated | estimated | unavailable_without_charting
    required_data TEXT,
    min_sample    INTEGER,
    opponent_adjustable INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS player_metrics (
    player_id INTEGER NOT NULL, season INTEGER NOT NULL, through_week INTEGER NOT NULL,
    metric_id TEXT NOT NULL, value REAL, sample INTEGER,
    pctl_fbs REAL, pctl_conf REAL, adj_value REAL, status TEXT NOT NULL,
    PRIMARY KEY (player_id, season, through_week, metric_id)
);
CREATE TABLE IF NOT EXISTS team_metrics (
    team_id INTEGER NOT NULL, season INTEGER NOT NULL, through_week INTEGER NOT NULL,
    metric_id TEXT NOT NULL, value REAL, sample INTEGER,
    pctl_fbs REAL, pctl_conf REAL, adj_value REAL, status TEXT NOT NULL,
    PRIMARY KEY (team_id, season, through_week, metric_id)
);
CREATE INDEX IF NOT EXISTS ix_team_metrics_lookup ON team_metrics(season, through_week, metric_id);

CREATE TABLE IF NOT EXISTS opponent_adjustments (
    team_id INTEGER NOT NULL, season INTEGER NOT NULL, as_of_week INTEGER NOT NULL,
    stat TEXT NOT NULL, side TEXT NOT NULL,      -- off | def
    raw REAL, adjusted REAL, sample INTEGER,
    PRIMARY KEY (team_id, season, as_of_week, stat, side)
);

CREATE TABLE IF NOT EXISTS team_ratings (
    team_id INTEGER NOT NULL, season INTEGER NOT NULL, as_of_week INTEGER NOT NULL,
    system TEXT NOT NULL,                        -- elo | adj_epa | composite
    rating REAL, off_rating REAL, def_rating REAL,
    PRIMARY KEY (team_id, season, as_of_week, system)
);

CREATE TABLE IF NOT EXISTS model_features (
    game_id INTEGER NOT NULL, model_version TEXT NOT NULL, snapshot_at TEXT NOT NULL,
    features_json TEXT NOT NULL,
    PRIMARY KEY (game_id, model_version)
);

CREATE TABLE IF NOT EXISTS model_versions (
    model_version TEXT PRIMARY KEY,
    family        TEXT NOT NULL,                 -- sim_engine | elo | ridge_margin | xgb_margin
    created_at    TEXT NOT NULL,
    train_start   TEXT, train_end TEXT,
    params_json   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'candidate',   -- candidate | production | retired
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    as_of        TEXT NOT NULL,                  -- information cutoff (strictly before kickoff for backtests)
    kind         TEXT NOT NULL,                  -- backtest | live
    home_win_prob REAL, home_pts_mean REAL, away_pts_mean REAL,
    margin_mean REAL, total_mean REAL, margin_sd REAL, total_sd REAL,
    margin_p10 REAL, margin_p90 REAL, total_p10 REAL, total_p90 REAL,
    ot_prob REAL, payload_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_preds_game ON model_predictions(game_id, model_version);

CREATE TABLE IF NOT EXISTS player_predictions (
    prediction_id INTEGER NOT NULL REFERENCES model_predictions(prediction_id),
    player_id INTEGER NOT NULL, stat TEXT NOT NULL,
    mean REAL, median REAL, sd REAL, p10 REAL, p25 REAL, p75 REAL, p90 REAL,
    PRIMARY KEY (prediction_id, player_id, stat)
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      INTEGER, home_id INTEGER, away_id INTEGER,
    model_version TEXT NOT NULL, n_sims INTEGER NOT NULL, seed INTEGER NOT NULL,
    created_at   TEXT NOT NULL, scenario TEXT, summary_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS simulation_events (
    run_id INTEGER NOT NULL, sim_index INTEGER NOT NULL, seq INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    PRIMARY KEY (run_id, sim_index, seq)
);

CREATE TABLE IF NOT EXISTS model_validation (
    validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL, created_at TEXT NOT NULL,
    scope TEXT NOT NULL,                         -- overall | season:2024 | conf:X | fav | dog
    metric TEXT NOT NULL, value REAL, ci_low REAL, ci_high REAL, n INTEGER,
    detail_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_validation ON model_validation(model_version, scope);

CREATE TABLE IF NOT EXISTS ingest_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, endpoint TEXT, params TEXT,
    rows INTEGER, fetched_at TEXT, status TEXT, message TEXT
);

-- Sportsbook lines (e.g. ESPN pickcenter: DraftKings). Spread is from the HOME team's perspective (negative = home favoured).
CREATE TABLE IF NOT EXISTS betting_lines (
    game_id INTEGER NOT NULL, provider TEXT NOT NULL, spread_home REAL, over_under REAL,
    home_moneyline REAL, away_moneyline REAL, fetched_at TEXT, source TEXT NOT NULL,
    PRIMARY KEY (game_id, provider)
);

-- Every line observation, append-only, so line movement and closing-line value can be measured later.
-- ESPN core odds carry the opening and current DraftKings numbers; each `cli lines` run appends one row per game.
CREATE TABLE IF NOT EXISTS line_snapshots (
    game_id INTEGER NOT NULL, provider TEXT NOT NULL, fetched_at TEXT NOT NULL,
    spread_home REAL, over_under REAL, home_moneyline REAL, away_moneyline REAL,
    home_spread_odds REAL, away_spread_odds REAL, over_odds REAL, under_odds REAL,
    open_spread_home REAL, open_total REAL, open_home_moneyline REAL, open_away_moneyline REAL,
    source TEXT NOT NULL,
    PRIMARY KEY (game_id, provider, fetched_at)
);

-- Tickets the SGP builder produced (paper-trading ledger). Book odds are entered by the user when known.
CREATE TABLE IF NOT EXISTS sgp_tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER, created_at TEXT, model_version TEXT,
    n_legs INTEGER, legs_json TEXT, p_joint REAL, p_indep REAL, fair_american REAL,
    book_american REAL, stake REAL, status TEXT DEFAULT 'open', result REAL, settled_at TEXT, note TEXT
);
