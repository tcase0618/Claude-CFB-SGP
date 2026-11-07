"""CollegeFootballData (CFBD) API ingestion — https://api.collegefootballdata.com/

* Auth: Bearer token from the CFBD_API_KEY environment variable (never hardcoded,
  never sent to the frontend).
* Politeness: minimum interval between requests (CFBD_MIN_INTERVAL_S) and an
  on-disk JSON cache keyed by endpoint+params, so stable historical data is
  fetched once. Re-running ingestion is incremental.
* Normalisation functions are pure (dict -> row) and unit-tested against
  fixtures shaped like CFBD v2 responses (camelCase), tolerating snake_case.
* Conventions: player rushing in `player_games` is derived from play-level
  attribution and EXCLUDES sacks (NFL convention, matches the simulator);
  CFBD/NCAA box scores include sacks in QB rushing. Defensive box-score
  stats (tackles, TFL, PD) come from /games/players because /plays/stats
  does not carry every tackle.
* Fields CFBD does not provide (air yards, pass location, formations,
  coverages, participation) stay NULL and the metrics that depend on them are
  reported as unavailable.

Endpoints and field names follow the public CFBD v2 docs; verify against the
current docs when upgrading (CFBD occasionally renames fields or tiers access).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..db import insert_df, utcnow

SOURCE = "cfbd"


def _g(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class CFBDClient:
    def __init__(self, settings, session: requests.Session | None = None):
        if not settings.cfbd_api_key:
            raise RuntimeError("CFBD_API_KEY is not set. Get a key at https://collegefootballdata.com/key and export it.")
        self.base = settings.cfbd_base_url.rstrip("/")
        self.key = settings.cfbd_api_key
        self.min_interval = settings.cfbd_min_interval_s
        self.cache = Path(settings.cache_dir) / "cfbd"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.s = session or requests.Session()
        self._last = 0.0

    def get(self, endpoint: str, use_cache: bool = True, **params):
        params = {k: v for k, v in params.items() if v is not None}
        key = hashlib.sha1(f"{endpoint}?{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if use_cache and path.exists():
            return json.loads(path.read_text())
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            r = self.s.get(f"{self.base}{endpoint}", params=params,
                           headers={"Authorization": f"Bearer {self.key}", "Accept": "application/json"}, timeout=60)
            self._last = time.time()
            if r.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            path.write_text(json.dumps(data))
            return data
        raise RuntimeError(f"CFBD rate limited on {endpoint}")


# ----------------------------------------------------------------------------- normalisers
PLAY_TYPE_MAP = {
    "pass reception": ("pass", dict(completion=1)), "pass completion": ("pass", dict(completion=1)),
    "pass incompletion": ("pass", {}), "passing touchdown": ("pass", dict(completion=1, touchdown=1)),
    "pass interception": ("pass", dict(interception=1)), "interception": ("pass", dict(interception=1)),
    "pass interception return": ("pass", dict(interception=1)),
    "interception return touchdown": ("pass", dict(interception=1, def_td=1)),
    "sack": ("sack", dict(sack=1)), "rush": ("rush", {}), "rushing touchdown": ("rush", dict(touchdown=1)),
    "fumble recovery (opponent)": ("rush", dict(fumble_lost=1)), "fumble recovery (own)": ("rush", {}),
    "fumble return touchdown": ("rush", dict(fumble_lost=1, def_td=1)),
    "punt": ("punt", {}), "blocked punt": ("punt", {}), "punt return touchdown": ("punt", dict(def_td=1)),
    "blocked punt touchdown": ("punt", dict(def_td=1)),
    "field goal good": ("field_goal", dict(fg_made=1)), "field goal missed": ("field_goal", dict(fg_made=0)),
    "blocked field goal": ("field_goal", dict(fg_made=0)), "missed field goal return": ("field_goal", dict(fg_made=0)),
    "missed field goal return touchdown": ("field_goal", dict(fg_made=0, def_td=1)),
    "blocked field goal touchdown": ("field_goal", dict(fg_made=0, def_td=1)),
    "kickoff": ("kickoff", {}), "kickoff return (offense)": ("kickoff", {}), "kickoff return touchdown": ("kickoff", {}),
    "extra point good": ("pat", dict(pat_ok=1)), "extra point missed": ("pat", {}),
    "two point pass": ("two_point", {}), "two point rush": ("two_point", {}), "2pt conversion": ("two_point", {}),
    "penalty": ("penalty", {}), "safety": ("other", dict(safety=1)), "timeout": ("other", {}),
    "end period": ("other", {}), "end of half": ("other", {}), "end of game": ("other", {}),
    "end of regulation": ("other", {}), "uncategorized": ("other", {}),
}


def normalize_play(d: dict, team_ids: dict[str, int], season: int, week: int) -> dict | None:
    raw_type = str(_g(d, "playType", "play_type", default="")).strip().lower()
    ptype, flags = PLAY_TYPE_MAP.get(raw_type, ("other", {}))
    off, de = _g(d, "offense"), _g(d, "defense")
    home = _g(d, "home")
    if off not in team_ids or de not in team_ids:
        return None
    clock = _g(d, "clock", default={}) or {}
    secs = None
    if isinstance(clock, dict) and clock:
        secs = int(clock.get("minutes", 0) or 0) * 60 + int(clock.get("seconds", 0) or 0)
    text = str(_g(d, "playText", "play_text", default="") or "")
    ytg = _g(d, "yardsToGoal", "yards_to_goal")
    yg = _g(d, "yardsGained", "yards_gained", default=0)
    td = int(flags.get("touchdown", 0) or (ptype in ("pass", "rush") and "touchdown" in raw_type and not flags.get("def_td")))
    points = 0
    if flags.get("def_td"):
        points = -6
    elif td:
        points = 6
    elif ptype == "field_goal" and flags.get("fg_made"):
        points = 3
    elif flags.get("safety"):
        points = -2
    elif ptype == "pat" and flags.get("pat_ok"):
        points = 1
    fgd = None
    if ptype == "field_goal":
        m = re.search(r"(\d+)\s*(?:yd|yard)", text, re.I)
        fgd = int(m.group(1)) if m else (int(ytg) + 17 if ytg is not None else None)
    down = _g(d, "down")
    dist = _g(d, "distance")
    first = int(ptype in ("pass", "rush") and dist is not None and yg is not None and yg >= dist and not td
                and not flags.get("interception") and not flags.get("fumble_lost"))
    return dict(
        play_id=int(_g(d, "id")), game_id=int(_g(d, "gameId", "game_id")), drive_id=_g(d, "driveId", "drive_id"),
        season=season, week=week, play_number=_g(d, "playNumber", "play_number"),
        offense_id=team_ids[off], defense_id=team_ids[de], home_id=team_ids.get(home, team_ids[off]),
        period=int(_g(d, "period", default=1)), clock_seconds=secs,
        offense_score=_g(d, "offenseScore", "offense_score"), defense_score=_g(d, "defenseScore", "defense_score"),
        down=down if down not in (0,) else None, distance=dist, yards_to_goal=ytg, play_type=ptype, yards_gained=yg,
        air_yards=None, pass_location=None, completion=int(flags.get("completion", 0)),
        interception=int(flags.get("interception", 0)), sack=int(flags.get("sack", 0)),
        fumble_lost=int(flags.get("fumble_lost", 0)), touchdown=td, first_down=first,
        scramble=int(ptype == "rush" and "scramble" in text.lower()), points=points, penalty_yards=None,
        fg_distance=fgd, fg_made=flags.get("fg_made") if ptype == "field_goal" else None,
        ep_before=None, ep_after=None, epa=None, success=None, garbage_time=0, play_text=text[:500], source=SOURCE)


STAT_ROLE = {"completion": "passer", "incompletion": "passer", "interception thrown": "passer", "sack taken": "passer",
             "reception": "target", "target": "target", "rush": "rusher", "sack": "sacker", "interception": "interceptor",
             "pass breakup": "pbu", "fumble forced": "forced_fumble", "field goal attempt": "kicker",
             "field goal made": "kicker", "field goal missed": "kicker", "punt": "punter", "tackle": "tackler"}


def normalize_play_stat(d: dict, team_ids: dict[str, int]) -> dict | None:
    role = STAT_ROLE.get(str(_g(d, "statType", "stat_type", default="")).lower())
    team = _g(d, "team")
    aid = _g(d, "athleteId", "athlete_id")
    if role is None or team not in team_ids or aid in (None, "", 0):
        return None
    try:
        aid = int(aid)
    except (TypeError, ValueError):
        return None
    return dict(play_id=int(_g(d, "playId", "play_id")), player_id=aid, team_id=team_ids[team], role=role,
                value=float(_g(d, "stat", default=0) or 0), _name=_g(d, "athleteName", "athlete_name"))


def normalize_game(d: dict, team_ids: dict[str, int]) -> dict | None:
    h, a = _g(d, "homeId", "home_id"), _g(d, "awayId", "away_id")
    if h not in team_ids.values() or a not in team_ids.values():
        return None
    return dict(game_id=int(_g(d, "id")), season=int(_g(d, "season")), week=int(_g(d, "week")),
                season_type=_g(d, "seasonType", "season_type", default="regular"),
                start_date=_g(d, "startDate", "start_date"), neutral_site=int(bool(_g(d, "neutralSite", "neutral_site", default=False))),
                conference_game=int(bool(_g(d, "conferenceGame", "conference_game", default=False))),
                venue_id=_g(d, "venueId", "venue_id"), home_id=int(h), away_id=int(a),
                home_points=_g(d, "homePoints", "home_points"), away_points=_g(d, "awayPoints", "away_points"),
                completed=int(bool(_g(d, "completed", default=False))), temperature_f=None, wind_mph=None,
                weather_desc=None, source=SOURCE)


DRIVE_RESULT = {"INT": "TURNOVER", "FUMBLE": "TURNOVER", "INT TD": "TURNOVER", "FUMBLE TD": "TURNOVER",
                "MISSED FG": "MISSED_FG", "END OF HALF": "END_HALF", "END OF GAME": "END_HALF", "SF": "SAFETY",
                "FG GOOD": "FG", "PASSING TD": "TD", "RUSHING TD": "TD"}

POS_MAP = {"OL": "OG", "G": "OG", "T": "OT", "DT": "DL", "NT": "DL", "DE": "EDGE", "OLB": "LB", "ILB": "LB", "MLB": "LB",
           "DB": "CB", "FS": "S", "SS": "S", "PK": "K", "ATH": "WR", "FB": "RB"}


def normalize_roster(d: dict, team_ids: dict[str, int], season: int):
    team = _g(d, "team")
    if team not in team_ids or _g(d, "id") is None:
        return None, None
    pos = str(_g(d, "position", default="") or "").upper()
    pos = POS_MAP.get(pos, pos) or None
    pid = int(_g(d, "id"))
    player = dict(player_id=pid, first_name=_g(d, "firstName", "first_name"), last_name=_g(d, "lastName", "last_name"),
                  position=pos, height_in=_g(d, "height"), weight_lb=_g(d, "weight"), birth_year=None,
                  hometown=_g(d, "homeCity", "home_city"), source=SOURCE)
    ros = dict(player_id=pid, team_id=team_ids[team], season=season, jersey=_g(d, "jersey"), position=pos,
               class_year=_g(d, "year"))
    return player, ros


def derive_player_games(plays: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Offensive/kicking box scores from play-level attribution (simulator conventions)."""
    if plays.empty or events.empty:
        return pd.DataFrame()
    e = events.merge(plays[["play_id", "game_id", "play_type", "yards_gained", "completion", "touchdown", "interception",
                            "sack", "yards_to_goal", "fg_made"]], on="play_id")
    y = e.yards_gained.fillna(0).clip(upper=e.yards_to_goal.fillna(99))
    f = pd.DataFrame({"player_id": e.player_id, "game_id": e.game_id, "team_id": e.team_id})
    pas = (e.role == "passer") & (e.play_type == "pass")
    f["pass_att"] = pas.astype(int)
    f["pass_cmp"] = (pas & (e.completion == 1)).astype(int)
    f["pass_yds"] = np.where(pas & (e.completion == 1), y, 0)
    f["pass_td"] = (pas & (e.touchdown == 1)).astype(int)
    f["pass_int"] = (pas & (e.interception == 1)).astype(int)
    f["sacks_taken"] = ((e.role == "passer") & (e.play_type == "sack")).astype(int)
    ru = (e.role == "rusher") & (e.play_type == "rush")
    f["rush_att"] = ru.astype(int)
    f["rush_yds"] = np.where(ru, y, 0)
    f["rush_td"] = (ru & (e.touchdown == 1)).astype(int)
    tg = (e.role == "target") & (e.play_type == "pass")
    f["targets"] = tg.astype(int)
    f["receptions"] = (tg & (e.completion == 1)).astype(int)
    f["rec_yds"] = np.where(tg & (e.completion == 1), y, 0)
    f["rec_td"] = (tg & (e.touchdown == 1) & (e.completion == 1)).astype(int)
    f["def_sacks"] = (e.role == "sacker").astype(float)
    f["def_int"] = (e.role == "interceptor").astype(int)
    f["pbu"] = (e.role == "pbu").astype(int)
    f["forced_fumbles"] = (e.role == "forced_fumble").astype(int)
    k = (e.role == "kicker") & (e.play_type == "field_goal")
    f["fga"] = k.astype(int)
    f["fgm"] = (k & (e.fg_made == 1)).astype(int)
    f["punts"] = ((e.role == "punter") & (e.play_type == "punt")).astype(int)
    out = f.groupby(["player_id", "game_id", "team_id"], as_index=False).sum()
    out["source"] = SOURCE
    return out


def parse_box_defense(game: dict) -> list[dict]:
    """Defensive stats (TOT, TFL, SACKS, PD, QB HUR) from /games/players nested box scores."""
    rows = []
    gid = _g(game, "id")
    for team in _g(game, "teams", default=[]) or []:
        for cat in _g(team, "categories", default=[]) or []:
            if str(cat.get("name", "")).lower() != "defensive":
                continue
            for typ in cat.get("types", []):
                nm = str(typ.get("name", "")).upper()
                col = {"TOT": "tackles", "TFL": "tfl", "PD": "pbu"}.get(nm)
                if not col:
                    continue
                for ath in typ.get("athletes", []):
                    try:
                        rows.append(dict(game_id=int(gid), player_id=int(ath["id"]), stat=col, value=float(ath.get("stat") or 0)))
                    except (TypeError, ValueError, KeyError):
                        continue
    return rows


# ----------------------------------------------------------------------------- orchestration
def ingest_seasons(conn, client: CFBDClient, seasons: list[int], with_plays: bool = True, log=print) -> None:
    conn.execute("INSERT OR REPLACE INTO data_sources VALUES (?,?,?,?,?)",
                 (SOURCE, "CollegeFootballData.com API", 0, "Per CFBD terms of service; attribute CFBD", utcnow()))
    confs = client.get("/conferences")
    insert_df(conn, "conferences", pd.DataFrame([dict(conference_id=_g(c, "abbreviation", "name"), name=_g(c, "name"),
                                                      division=_g(c, "classification", default="fbs")) for c in confs]))
    venues = client.get("/venues")
    insert_df(conn, "venues", pd.DataFrame([dict(
        venue_id=_g(v, "id"), name=_g(v, "name"), city=_g(v, "city"), state=_g(v, "state"), capacity=_g(v, "capacity"),
        grass=int(bool(_g(v, "grass", default=False))), dome=int(bool(_g(v, "dome", default=False))),
        elevation_m=_g(v, "elevation"), latitude=_g(v, "latitude"), longitude=_g(v, "longitude"),
        timezone=_g(v, "timezone")) for v in venues]))
    for season in seasons:
        teams = client.get("/teams/fbs", year=season)
        conf_ids = {c["name"]: _g(c, "abbreviation", "name") for c in confs}
        insert_df(conn, "teams", pd.DataFrame([dict(team_id=t["id"], school=t["school"], mascot=_g(t, "mascot"),
                                                    abbreviation=_g(t, "abbreviation"), color=_g(t, "color"),
                                                    alt_color=_g(t, "alternateColor", "alt_color"),
                                                    venue_id=_g(_g(t, "location", default={}) or {}, "id", "venueId"),
                                                    source=SOURCE) for t in teams]))
        team_ids = {t["school"]: int(t["id"]) for t in teams}
        talent = {x["team"]: float(x["talent"]) for x in client.get("/talent", year=season) if x.get("team") in team_ids}
        ret = {_g(x, "team"): _g(x, "percentPPA", "percent_ppa") for x in client.get("/player/returning", year=season)}
        insert_df(conn, "team_seasons", pd.DataFrame([dict(team_id=int(t["id"]), season=season,
                                                           conference_id=conf_ids.get(t.get("conference"), t.get("conference")),
                                                           division="fbs", talent_composite=talent.get(t["school"]),
                                                           returning_production=ret.get(t["school"]), head_coach_id=None)
                                                      for t in teams]))
        games = [normalize_game(g, team_ids) for g in client.get("/games", year=season, seasonType="regular")]
        games += [normalize_game(g, team_ids) for g in client.get("/games", year=season, seasonType="postseason")]
        gdf = pd.DataFrame([g for g in games if g])
        try:
            wx = {int(_g(w, "id", "gameId")): w for w in client.get("/games/weather", year=season)}
            gdf["temperature_f"] = gdf.game_id.map(lambda i: _g(wx.get(i, {}), "temperature"))
            gdf["wind_mph"] = gdf.game_id.map(lambda i: _g(wx.get(i, {}), "windSpeed", "wind_speed"))
            gdf["weather_desc"] = gdf.game_id.map(lambda i: _g(wx.get(i, {}), "weatherCondition", "weather_condition"))
        except requests.HTTPError as e:   # weather may require a higher CFBD tier
            log(f"  weather unavailable ({e}); continuing without")
        insert_df(conn, "games", gdf)
        players, rosters = [], []
        for r in client.get("/roster", year=season):
            p, ro = normalize_roster(r, team_ids, season)
            if p:
                players.append(p)
                rosters.append(ro)
        insert_df(conn, "players", pd.DataFrame(players).drop_duplicates("player_id"))
        insert_df(conn, "rosters", pd.DataFrame(rosters).drop_duplicates(["player_id", "season", "team_id"]))
        recs = client.get("/recruiting/players", year=season, classification="HighSchool")
        insert_df(conn, "recruiting_profiles", pd.DataFrame([dict(
            player_id=_g(r, "athleteId", "athlete_id"), recruit_id=int(_g(r, "id")), name=_g(r, "name"), class_year=season,
            position=_g(r, "position"), stars=_g(r, "stars"), rating=_g(r, "rating"), national_rank=_g(r, "ranking"),
            committed_team_id=team_ids.get(_g(r, "committedTo", "committed_to")), source=SOURCE) for r in recs]))
        try:
            portal = client.get("/player/portal", year=season)
            insert_df(conn, "transfer_history", pd.DataFrame([dict(
                transfer_id=int(season * 100000 + i), player_id=None,
                name=f"{_g(t, 'firstName', default='')} {_g(t, 'lastName', default='')}".strip(), position=_g(t, "position"),
                season=season, from_team_id=team_ids.get(_g(t, "origin")), to_team_id=team_ids.get(_g(t, "destination")),
                stars=_g(t, "stars"), rating=_g(t, "rating"), eligibility=_g(t, "eligibility"),
                transfer_date=_g(t, "transferDate", "transfer_date"), source=SOURCE) for i, t in enumerate(portal)]))
        except requests.HTTPError as e:
            log(f"  portal unavailable ({e})")
        conn.commit()
        if not with_plays:
            log(f"{season}: {len(gdf)} games (no plays)")
            continue
        drives = client.get("/drives", year=season)
        insert_df(conn, "drives", pd.DataFrame([dict(
            drive_id=int(_g(d, "id")), game_id=int(_g(d, "gameId", "game_id")),
            offense_id=team_ids.get(_g(d, "offense")), defense_id=team_ids.get(_g(d, "defense")),
            drive_number=_g(d, "driveNumber"), start_yards_to_goal=_g(d, "startYardsToGoal"),
            end_yards_to_goal=_g(d, "endYardsToGoal"), plays=_g(d, "plays"), yards=_g(d, "yards"),
            result=DRIVE_RESULT.get(str(_g(d, "driveResult", default="")).upper(), str(_g(d, "driveResult", default="")).upper()), points=None, start_period=_g(d, "startPeriod"))
            for d in drives if team_ids.get(_g(d, "offense")) and team_ids.get(_g(d, "defense"))]))
        weeks = sorted(gdf.week.unique())
        for wk in weeks:
            st = "postseason" if wk > 15 else "regular"
            plays = [normalize_play(p, team_ids, season, int(wk)) for p in client.get("/plays", year=season, week=int(wk), seasonType=st)]
            pdf = pd.DataFrame([p for p in plays if p])
            pdf = pdf[pdf.game_id.isin(gdf.game_id)]
            insert_df(conn, "plays", pdf)
            stats = [normalize_play_stat(s, team_ids) for s in client.get("/plays/stats", year=season, week=int(wk))]
            sdf = pd.DataFrame([s for s in stats if s])
            if not sdf.empty:
                sdf = sdf[sdf.play_id.isin(pdf.play_id)]
                known = set(r[0] for r in conn.execute("SELECT player_id FROM players"))
                newp = sdf[~sdf.player_id.isin(known)].drop_duplicates("player_id")
                if len(newp):
                    nm = newp._name.fillna("").str.split(" ", n=1)
                    insert_df(conn, "players", pd.DataFrame(dict(player_id=newp.player_id, first_name=nm.str[0],
                                                                  last_name=nm.str[1], position=None, height_in=None,
                                                                  weight_lb=None, birth_year=None, hometown=None, source=SOURCE)))
                ev = sdf.drop(columns=["_name"]).drop_duplicates(["play_id", "player_id", "role"])
                insert_df(conn, "play_events", ev)
                pg = derive_player_games(pdf, ev)
                box = []
                for g in client.get("/games/players", year=season, week=int(wk), seasonType=st):
                    box += parse_box_defense(g)
                if box:
                    b = pd.DataFrame(box).pivot_table(index=["game_id", "player_id"], columns="stat", values="value", aggfunc="sum").reset_index()
                    pg = pg.merge(b, on=["game_id", "player_id"], how="outer", suffixes=("", "_box"))
                    for c in ("tackles", "tfl"):
                        if c in b:
                            pg[c] = pg[c].fillna(0)
                    pg = pg[pg.team_id.notna()]
                    pg = pg[[c for c in pg.columns if not c.endswith("_box")]]
                insert_df(conn, "player_games", pg)
            conn.execute("INSERT INTO ingest_log (source, endpoint, params, rows, fetched_at, status, message) VALUES (?,?,?,?,?,?,?)",
                         (SOURCE, "/plays", json.dumps({"year": season, "week": int(wk)}), len(pdf), utcnow(), "ok", None))
            conn.commit()
            log(f"{season} wk{wk}: {len(pdf)} plays")
        # drive points from plays
        conn.execute("""UPDATE drives SET points = (SELECT COALESCE(SUM(CASE WHEN p.points>0 THEN p.points ELSE 0 END),0)
                        FROM plays p WHERE p.drive_id = drives.drive_id) WHERE game_id IN (SELECT game_id FROM games WHERE season=?)""", (season,))
        conn.commit()
