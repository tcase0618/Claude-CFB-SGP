"""Keyless ingestion from ESPN's public JSON endpoints (the same feeds espn.com uses).

No API key. Endpoints (undocumented, public):
  site API  https://site.api.espn.com/apis/site/v2/sports/football/college-football
    /scoreboard?groups=80&year=Y&seasontype=2|3&week=W   games, scores, venue, neutral, weather, rankings
    /summary?event=ID                                      box scores, sportsbook lines (pickcenter), win prob
  standings https://site.api.espn.com/apis/v2/sports/football/college-football/standings?group=80&season=Y
  core API  https://sports.core.api.espn.com/v2/sports/football/leagues/college-football
    /events/ID/competitions/ID/plays?limit=500            every play with down/distance/field position and
                                                          athlete IDs by role (passer, receiver, rusher, tackler...)
    /seasons/Y/athletes/ID                                position, height, weight, class for that season

Terms: ESPN publishes no developer terms for these feeds and its site Terms of Use restrict automated/commercial
reuse. This loader is intended for personal, non-commercial research: it rate-limits, caches every immutable
response (completed games) so each is fetched once, and attributes ESPN as the source. For a commercial product,
license data instead.

Text parsing (classification `estimated`, flagged in docs): ESPN play text carries charting detail that the
structured fields do not — catch/throw spot ("caught at WVU34", "thrown to WVU16") gives air yards relative to the
line of scrimmage; "short|deep left|middle|right" gives pass depth/location; "Shotgun"/"No Huddle" formation tags;
"broken up by" pass breakups. Parsed values are bounded and left NULL when absent or implausible.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..db import insert_df, utcnow
from .cfbd import PLAY_TYPE_MAP, POS_MAP, derive_player_games

SOURCE = "espn"
SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
STANDINGS = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"

EXTRA_TYPES = {"punt return": ("punt", {}), "punt return touchdown": ("punt", dict(def_td=1)),
               "two-point conversion": ("two_point", {}), "two point conversion": ("two_point", {}),
               "defensive 2pt conversion": ("two_point", {}), "pass interception return": ("pass", dict(interception=1)),
               "kickoff return touchdown": ("kickoff", {}), "end of regulation": ("other", {}),
               "blocked punt": ("punt", {}), "blocked field goal": ("field_goal", dict(fg_made=0)),
               "extra point good": ("pat", dict(pat_ok=1)), "extra point missed": ("pat", {}),
               "coin toss": ("other", {}), "fumble return touchdown": ("rush", dict(fumble_lost=1, def_td=1))}
ROLE = {"passer": "passer", "receiver": "target", "rusher": "rusher", "tackler": "tackler", "assistedBy": "tackler",
        "sackedBy": "sacker", "passDefender": "pbu", "forcedBy": "forced_fumble", "kicker": "kicker",
        "punter": "punter", "returner": "returner"}
ESPN_POS = {**POS_MAP, "OL": "OG", "G": "OG", "OT": "OT", "C": "C", "DE": "EDGE", "DT": "DL", "NT": "DL", "DL": "DL",
            "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB", "CB": "CB", "DB": "CB", "S": "S", "FS": "S", "SS": "S",
            "PK": "K", "K": "K", "P": "P", "LS": "LS", "ATH": "WR", "FB": "RB", "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE"}


class ESPNClient:
    def __init__(self, min_interval: float = 0.08, cache_dir: Path | None = None, session: requests.Session | None = None):
        from ..config import DATA_DIR
        self.min_interval = min_interval
        self.cache = Path(cache_dir or DATA_DIR / "cache" / "espn")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": "cfbquant-research/0.1 (personal research; cached, rate-limited)"})
        self._lock = threading.Lock()
        self._last = 0.0
        self.calls = 0

    def get(self, url: str, params: dict | None = None, cache: bool = False):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        key = hashlib.sha1(f"{url}?{json.dumps(params, sort_keys=True)}".encode()).hexdigest()
        fp = self.cache / f"{key}.json"
        if cache and fp.exists():
            return json.loads(fp.read_text())
        for attempt in range(5):
            with self._lock:
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
                self.calls += 1
            try:
                r = self.s.get(url, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            if cache:
                fp.write_text(json.dumps(data))
            return data
        return None

    # endpoints
    def scoreboard(self, year, week, seasontype=2, cache=False):
        # NB: the site API ignores `year=`; `dates=YYYY` selects the season (verified)
        return self.get(f"{SITE}/scoreboard", {"groups": 80, "dates": year, "week": week, "seasontype": seasontype, "limit": 400},
                        cache=cache)

    def summary(self, event_id, cache=True):
        return self.get(f"{SITE}/summary", {"event": event_id}, cache=cache)

    def plays(self, event_id, cache=True) -> list[dict]:
        out, page = [], 1
        while True:
            d = self.get(f"{CORE}/events/{event_id}/competitions/{event_id}/plays", {"limit": 500, "page": page}, cache=cache)
            if not d:
                break
            out += d.get("items", [])
            if page >= int(d.get("pageCount", 1) or 1):
                break
            page += 1
        return out

    def standings(self, season):
        return self.get(STANDINGS, {"group": 80, "season": season}, cache=True)

    def roster(self, team_id):
        return self.get(f"{SITE}/teams/{team_id}/roster")

    def athlete(self, season, athlete_id):
        return self.get(f"{CORE}/seasons/{season}/athletes/{athlete_id}", cache=True)


# ----------------------------------------------------------------------------- pure normalisers
def _ref_id(ref) -> str | None:
    if isinstance(ref, dict):
        ref = ref.get("$ref", "")
    m = re.search(r"/(teams|athletes|drives)/(\d+)", ref or "")
    return m.group(2) if m else None


def parse_scoreboard_event(e: dict, season: int, week: int, season_type: str) -> tuple[dict, list[dict]]:
    c = e["competitions"][0]
    comp = {t["homeAway"]: t for t in c["competitors"]}
    h, a = comp["home"], comp["away"]
    st = c.get("status", {}).get("type", {})
    w = e.get("weather") or {}
    teams = []
    for t in (h, a):
        tm = t["team"]
        teams.append(dict(team_id=int(tm["id"]), school=tm.get("location") or tm.get("displayName"), mascot=tm.get("name"),
                          abbreviation=tm.get("abbreviation"), color=("#" + tm["color"]) if tm.get("color") else None,
                          alt_color=("#" + tm["alternateColor"]) if tm.get("alternateColor") else None,
                          venue_id=None, source=SOURCE, _conf=tm.get("conferenceId"),
                          _rank=(t.get("curatedRank") or {}).get("current")))
    done = bool(st.get("completed"))
    g = dict(game_id=int(e["id"]), season=season, week=week, season_type=season_type, start_date=e["date"],
             neutral_site=int(bool(c.get("neutralSite"))), conference_game=int(bool(c.get("conferenceCompetition"))),
             venue_id=int(c["venue"]["id"]) if c.get("venue", {}).get("id") else None,
             home_id=int(h["team"]["id"]), away_id=int(a["team"]["id"]),
             home_points=int(h["score"]) if done and h.get("score") not in (None, "") else None,
             away_points=int(a["score"]) if done and a.get("score") not in (None, "") else None,
             completed=int(done), temperature_f=w.get("temperature"), wind_mph=None,
             weather_desc=w.get("conditionId"), source=SOURCE)
    return g, teams


SPOT = re.compile(r"(?:caught at|thrown to|intercepted at)\s+(?:the\s+)?([A-Z][A-Z&]{1,5})\s?(\d{1,2})\b")
LOC = re.compile(r"pass(?:\s+(?:complete|incomplete|intercepted))?\s+(short|deep)\s+(left|middle|right)", re.I)


def spot_to_ytg(abbr: str, yard: int, off_abbr: str, def_abbr: str) -> int | None:
    if yard == 50:
        return 50
    if abbr == off_abbr:
        return 100 - yard
    if abbr == def_abbr:
        return yard
    return None


def parse_text(text: str, los_ytg, off_abbr: str | None, def_abbr: str | None) -> dict:
    """Charting detail parsed from ESPN play text (bounded; NULL when absent/implausible)."""
    out = {"air_yards": None, "pass_location": None, "pass_depth": None, "shotgun": None, "no_huddle": None, "pbu_text": False}
    if not text:
        return out
    low = text.lower()
    out["shotgun"] = 1 if "shotgun" in low else None
    out["no_huddle"] = 1 if "no huddle" in low else None
    out["pbu_text"] = "broken up by" in low
    m = LOC.search(text)
    if m:
        out["pass_depth"], out["pass_location"] = m.group(1).lower(), m.group(2).lower()
    m = SPOT.search(text)
    if m and los_ytg is not None and off_abbr and def_abbr:
        y = spot_to_ytg(m.group(1), int(m.group(2)), off_abbr.upper(), def_abbr.upper())
        if y is not None:
            air = int(los_ytg) - y
            if -15 <= air <= 75:
                out["air_yards"] = float(air)
    return out


def pat_points(text: str) -> int:
    """PAT after a TD, from play text. Default +1 (kick; ~97% success) when the text is silent."""
    t = (text or "").lower()
    if "two-point" in t or "2pt" in t or "two point" in t or "conversion" in t:
        return 0 if any(w in t for w in ("failed", "no good", "unsuccessful", "intercepted")) else 2
    if any(w in t for w in ("kick failed", "kick missed", "kick blocked", "pat failed", "pat missed", "pat blocked", "no good")):
        return 0
    return 1


def scoring(p: dict, tt: str, ptype: str, flags: dict, off: int, de: int) -> tuple[int, int, bool]:
    """(points to offence, points to defence, offensive TD?) from scoreValue + play semantics."""
    if not p.get("scoringPlay"):
        return 0, 0, False
    sv = int(p.get("scoreValue") or 0)
    scorer = _ref_id(p.get("team") or {})
    scorer = int(scorer) if scorer and scorer.isdigit() else None
    if tt == "safety" or ("safety" in tt):
        return (2, 0, False) if scorer == off else (0, 2, False)
    if ptype == "field_goal" or sv == 3:
        return 3, 0, False
    off_scored = (scorer == off) if scorer in (off, de) else not flags.get("def_td")
    if flags.get("def_td") or flags.get("interception") or flags.get("fumble_lost") or ptype in ("punt", "kickoff"):
        off_scored = scorer == off if scorer in (off, de) and ptype == "kickoff" else False
    if sv >= 6:
        tot = 6 + pat_points(p.get("text"))
        return (tot, 0, ptype in ("pass", "rush")) if off_scored else (0, tot, False)
    return (sv, 0, False) if off_scored else (0, sv, False)


def normalize_plays(items: list[dict], game: dict, abbr: dict[int, str]) -> tuple[list[dict], list[dict], list[dict]]:
    """ESPN core plays -> (plays rows, play_events rows, formation rows)."""
    plays, events, forms = [], [], []
    home, away = game["home_id"], game["away_id"]
    items = sorted(items, key=lambda p: int(p.get("sequenceNumber") or 0))
    run = {home: 0, away: 0}     # running score rebuilt from per-play scoreValue (ESPN score fields can be mis-sequenced)
    n = 0
    for p in items:
        tt = str(p.get("type", {}).get("text", "")).strip().lower()
        ptype, flags = EXTRA_TYPES.get(tt) or PLAY_TYPE_MAP.get(tt, ("other", {}))
        st = p.get("start") or {}
        off = _ref_id((st.get("team") or {})) or _ref_id(p.get("team") or {})
        try:
            off = int(off)
        except (TypeError, ValueError):
            off = None
        if off not in (home, away):
            continue
        de = away if off == home else home
        off_pre, def_pre = run[off], run[de]
        pts_off, pts_def, off_td = scoring(p, tt, ptype, flags, off, de)
        run[off] += pts_off
        run[de] += pts_def
        n += 1
        text = p.get("text") or ""
        if " sacked" in f" {text.lower()}" and ptype in ("rush", "pass", "other"):   # strip-sacks are typed as fumble recoveries
            ptype, flags = "sack", {**flags, "sack": 1}
        ytg = st.get("yardsToEndzone")
        down = st.get("down") or None
        dist = st.get("distance")
        yg = p.get("statYardage")
        td = int(off_td and ptype in ("pass", "rush"))
        comp = int(bool(flags.get("completion")) or (ptype == "pass" and "pass complete" in text.lower())
                   or (ptype == "pass" and td and not flags.get("interception")))
        tp = parse_text(text, ytg, abbr.get(off), abbr.get(de))
        if ptype not in ("pass", "sack"):
            tp.update(air_yards=None, pass_location=None)
        fgd = None
        if ptype == "field_goal":
            m = re.search(r"(\d+)\s*(?:yd|yard)", text, re.I)
            fgd = int(m.group(1)) if m else (int(ytg) + 17 if ytg is not None else None)
        pid = int(p["id"])
        first = int(ptype in ("pass", "rush") and not td and (("1st down" in text.lower()) or
                    (dist is not None and yg is not None and yg >= dist)) and not flags.get("interception")
                    and not flags.get("fumble_lost"))
        plays.append(dict(
            play_id=pid, game_id=game["game_id"], drive_id=int(_ref_id(p.get("drive")) or 0) or None,
            season=game["season"], week=game["week"], play_number=n, offense_id=off, defense_id=de, home_id=home,
            period=int((p.get("period") or {}).get("number") or 1),
            clock_seconds=int((p.get("clock") or {}).get("value") or 0),
            offense_score=off_pre, defense_score=def_pre, down=down if down in (1, 2, 3, 4) else None,
            distance=dist, yards_to_goal=ytg, play_type=ptype, yards_gained=yg,
            air_yards=tp["air_yards"], pass_location=tp["pass_location"],
            completion=comp, interception=int(bool(flags.get("interception"))), sack=int(bool(flags.get("sack"))),
            fumble_lost=int(bool(flags.get("fumble_lost"))), touchdown=td, first_down=first, scramble=0,
            points=int(pts_off - pts_def), penalty_yards=None, fg_distance=fgd,
            fg_made=(1 if (flags.get("fg_made") or pts_off == 3) else 0) if ptype == "field_goal" else None,
            ep_before=None, ep_after=None, epa=None, success=None, garbage_time=0, play_text=text[:500], source=SOURCE))
        if tp.get("shotgun") or tp.get("no_huddle"):
            forms.append(dict(play_id=pid, personnel=None, formation="no_huddle_shotgun" if tp.get("no_huddle") else "shotgun",
                              shotgun=tp.get("shotgun"), motion=None, play_action=None, rpo=None, source=SOURCE + "_text"))
        team_of = {off: off, de: de}
        for part in p.get("participants", []) or []:
            r = part.get("type") or ""
            role = ROLE.get(r) or ("interceptor" if "intercept" in r.lower() else None)
            aid = _ref_id(part.get("athlete") or {})
            if not role or not aid:
                continue
            tid = off if role in ("passer", "target", "rusher", "kicker", "punter") else de
            if role == "returner":
                tid = de
            events.append(dict(play_id=pid, player_id=int(aid), team_id=tid, role=role, value=float(yg or 0)))
        if tp.get("pbu_text") and not any(e["play_id"] == pid and e["role"] == "pbu" for e in events[-6:]):
            pass   # no athlete id in text; PBU without an id stays unattributed
    return plays, events, forms


def parse_box(summary: dict) -> tuple[list[dict], list[dict]]:
    """Box-score rows (per player per game) and athlete identity rows from /summary."""
    rows, ath = {}, {}
    if not summary:
        return [], []
    gid = int(summary.get("header", {}).get("id") or 0)
    for team in summary.get("boxscore", {}).get("players", []) or []:
        tid = int(team["team"]["id"])
        for cat in team.get("statistics", []) or []:
            labels = cat.get("labels", [])
            name = cat.get("name")
            for a in cat.get("athletes", []) or []:
                at = a.get("athlete") or {}
                if not at.get("id"):
                    continue
                pid = int(at["id"])
                ath[pid] = dict(player_id=pid, first_name=at.get("firstName"), last_name=at.get("lastName"),
                                team_id=tid)
                v = dict(zip(labels, a.get("stats", [])))
                r = rows.setdefault(pid, dict(player_id=pid, game_id=gid, team_id=tid))

                def num(x):
                    try:
                        return float(str(x).replace("--", "0"))
                    except ValueError:
                        return 0.0
                if name == "defensive":
                    r["tackles"] = num(v.get("TOT"))
                    r["tfl"] = num(v.get("TFL"))
                    r["def_sacks"] = num(v.get("SACKS"))
                    r["pbu"] = num(v.get("PD"))
                elif name == "interceptions":
                    r["def_int"] = num(v.get("INT"))
                elif name == "kicking":
                    fg = str(v.get("FG", "0/0")).split("/")
                    xp = str(v.get("XP", "0/0")).split("/")
                    if len(fg) == 2:
                        r["fgm"], r["fga"] = num(fg[0]), num(fg[1])
                    if len(xp) == 2:
                        r["xpm"], r["xpa"] = num(xp[0]), num(xp[1])
                elif name == "punting":
                    r["punts"] = num(v.get("NO"))
                    r["punt_yds"] = num(v.get("YDS"))
    return list(rows.values()), list(ath.values())


def parse_lines(summary: dict, game_id: int) -> list[dict]:
    out = []
    for x in (summary or {}).get("pickcenter", []) or []:
        prov = (x.get("provider") or {}).get("name")
        if not prov:
            continue
        out.append(dict(game_id=game_id, provider=prov, spread_home=x.get("spread"), over_under=x.get("overUnder"),
                        home_moneyline=(x.get("homeTeamOdds") or {}).get("moneyLine"),
                        away_moneyline=(x.get("awayTeamOdds") or {}).get("moneyLine"), fetched_at=utcnow(), source=SOURCE))
    return out


def _num(x):
    try:
        return float(str(x).replace("+", "")) if x not in (None, "", "OFF", "EVEN") else (100.0 if x == "EVEN" else None)
    except (TypeError, ValueError):
        return None


def parse_core_odds(item: dict, game_id: int, fetched_at: str) -> dict:
    """One provider entry from the core-API odds collection -> a line_snapshots row (current + opening numbers)."""
    h, a = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}
    ho, ao, op = h.get("open") or {}, a.get("open") or {}, item.get("open") or {}
    g = lambda d, *ks: _num(_dig(d, *ks))
    return dict(game_id=game_id, provider=(item.get("provider") or {}).get("name"), fetched_at=fetched_at,
                spread_home=_num(item.get("spread")), over_under=_num(item.get("overUnder")),
                home_moneyline=_num(h.get("moneyLine")), away_moneyline=_num(a.get("moneyLine")),
                home_spread_odds=_num(h.get("spreadOdds")), away_spread_odds=_num(a.get("spreadOdds")),
                over_odds=_num(item.get("overOdds")), under_odds=_num(item.get("underOdds")),
                open_spread_home=g(ho, "pointSpread", "american"), open_total=g(op, "total", "american"),
                open_home_moneyline=g(ho, "moneyLine", "american"), open_away_moneyline=g(ao, "moneyLine", "american"),
                source=SOURCE)


def _dig(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def snapshot_lines(conn, client: "ESPNClient", season: int, week: int | None = None, log=print) -> int:
    """Append current + opening lines for upcoming games (all not-yet-completed games of `week`, or of the season)
    to line_snapshots, and refresh betting_lines with the current number (which the SGP anchor reads)."""
    q = "SELECT game_id FROM games WHERE season=? AND completed=0" + (" AND week=?" if week is not None else "")
    ids = [r[0] for r in conn.execute(q, (season,) + ((week,) if week is not None else ()))]
    now = utcnow()
    rows = []
    for gid in ids:
        d = client.get(f"{CORE}/events/{gid}/competitions/{gid}/odds")
        for it in (d or {}).get("items", []) or []:
            r = parse_core_odds(it, gid, now)
            if r["provider"] and (r["spread_home"] is not None or r["over_under"] is not None):
                rows.append(r)
    if rows:
        df = pd.DataFrame(rows)
        insert_df(conn, "line_snapshots", df)
        cur = df[["game_id", "provider", "spread_home", "over_under", "home_moneyline", "away_moneyline"]].assign(
            fetched_at=now, source=SOURCE)
        insert_df(conn, "betting_lines", cur)
        conn.commit()
    log(f"lines: {len(ids)} upcoming games, {len(rows)} provider lines snapshotted at {now}")
    return len(rows)


# ----------------------------------------------------------------------------- orchestration
def _conferences(client, season) -> tuple[list[dict], dict[int, str]]:
    d = client.standings(season) or {}
    confs, team_conf = [], {}
    for c in d.get("children", []) or []:
        cid = c.get("abbreviation") or c.get("name")
        confs.append(dict(conference_id=cid, name=c.get("name"), division="fbs"))
        groups = [c] + (c.get("children") or [])
        for g in groups:
            for e in (g.get("standings") or {}).get("entries", []) or []:
                team_conf[int(e["team"]["id"])] = cid
    return confs, team_conf


def ingest_espn(conn, client: ESPNClient, seasons: list[int], workers: int = 8, max_week: int = 17, log=print) -> None:
    conn.execute("INSERT OR REPLACE INTO data_sources VALUES (?,?,?,?,?)",
                 (SOURCE, "ESPN public JSON feeds (site + core APIs), keyless; personal research use", 0,
                  "ESPN Terms of Use apply; cached, rate-limited, attributed; not licensed for commercial redistribution",
                  utcnow()))
    for season in seasons:
        t0 = time.time()
        confs, team_conf = _conferences(client, season)
        insert_df(conn, "conferences", pd.DataFrame(confs + [dict(conference_id="FCS", name="FCS / other", division="fcs")]))
        games, teams = [], {}
        for stype, weeks in ((2, range(1, max_week + 1)), (3, range(1, 2))):
            for wk in weeks:
                sb = client.scoreboard(season, wk, stype)
                evs = (sb or {}).get("events", []) or []
                if not evs and stype == 2 and wk > 1:
                    continue
                for e in evs:
                    if int((e.get("season") or {}).get("year") or season) != season:   # guard against silent season fallback
                        continue
                    g, ts = parse_scoreboard_event(e, season, wk if stype == 2 else 99, "regular" if stype == 2 else "postseason")
                    if g["home_id"] <= 0 or g["away_id"] <= 0:      # TBD placeholders (unset bowl/playoff matchups)
                        continue
                    games.append(g)
                    for t in ts:
                        teams[t["team_id"]] = t
        if not games:
            log(f"{season}: no games")
            continue
        tdf = pd.DataFrame(teams.values())
        insert_df(conn, "teams", tdf.drop(columns=["_conf", "_rank"]))
        insert_df(conn, "team_seasons", pd.DataFrame([dict(
            team_id=t, season=season, conference_id=team_conf.get(t, "FCS"), division="fbs" if t in team_conf else "fcs",
            talent_composite=None, returning_production=None, head_coach_id=None) for t in teams]))
        gdf = pd.DataFrame(games).drop_duplicates("game_id")
        insert_df(conn, "games", gdf)
        conn.commit()
        abbr = {int(t["team_id"]): (t["abbreviation"] or "").upper() for t in teams.values()}
        have = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM plays WHERE season=?", (season,))}
        todo = [g for g in gdf.to_dict("records") if g["completed"] and g["game_id"] not in have]
        log(f"{season}: {len(gdf)} games on schedule, {len(todo)} completed games to fetch")

        def fetch(g):
            return g, client.plays(g["game_id"]), client.summary(g["game_id"])
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fetch, g) for g in todo]
            for fu in as_completed(futs):
                g, items, summ = fu.result()
                plays, events, forms = normalize_plays(items or [], g, abbr)
                box, ath = parse_box(summ)
                if not plays:
                    continue
                pdf = pd.DataFrame(plays)
                ids = {e["player_id"] for e in events} | {b["player_id"] for b in box}
                known = {r[0] for r in conn.execute(f"SELECT player_id FROM players WHERE player_id IN ({','.join(map(str, ids)) or 0})")}
                names = {a["player_id"]: a for a in ath}
                newp = [dict(player_id=i, first_name=names.get(i, {}).get("first_name"), last_name=names.get(i, {}).get("last_name"),
                             position=None, height_in=None, weight_lb=None, birth_year=None, hometown=None, source=SOURCE)
                        for i in ids - known]
                insert_df(conn, "players", pd.DataFrame(newp), replace=False)
                for a in ath:   # fill names learned from box scores
                    conn.execute("UPDATE players SET first_name=COALESCE(first_name,?), last_name=COALESCE(last_name,?) WHERE player_id=?",
                                 (a["first_name"], a["last_name"], a["player_id"]))
                insert_df(conn, "plays", pdf)
                ev = pd.DataFrame(events).drop_duplicates(["play_id", "player_id", "role"]) if events else pd.DataFrame()
                insert_df(conn, "play_events", ev)
                insert_df(conn, "offensive_formations", pd.DataFrame(forms))
                pg = derive_player_games(pdf, ev) if not ev.empty else pd.DataFrame()
                bx = pd.DataFrame(box)
                if not bx.empty:
                    bx = bx.assign(game_id=g["game_id"])
                    cols = [c for c in ("tackles", "tfl", "def_sacks", "pbu", "def_int", "fga", "fgm", "xpa", "xpm", "punts", "punt_yds") if c in bx]
                    if pg.empty:
                        pg = bx[["player_id", "game_id", "team_id"] + cols].assign(source=SOURCE)
                    else:
                        pg = pg.drop(columns=[c for c in cols if c in pg]).merge(bx[["player_id", "game_id", "team_id"] + cols],
                                                                              on=["player_id", "game_id", "team_id"], how="outer")
                        pg["source"] = SOURCE
                    pg = pg.fillna({c: 0 for c in pg.columns if c not in ("source",)})
                insert_df(conn, "player_games", pg)
                insert_df(conn, "betting_lines", pd.DataFrame(parse_lines(summ, g["game_id"])))
                for r in (summ or {}).get("gameInfo", {}).get("weather", {}) and [summ["gameInfo"]["weather"]] or []:
                    conn.execute("UPDATE games SET temperature_f=COALESCE(temperature_f, ?) WHERE game_id=?", (r.get("temperature"), g["game_id"]))
                done += 1
                if done % 50 == 0:
                    conn.commit()
                    log(f"  {season}: {done}/{len(todo)} games ({client.calls} calls, {time.time() - t0:.0f}s)")
        conn.commit()
        _derive_drives(conn, season)
        _rosters(conn, client, season, workers, log)
        _returning(conn, season)
        conn.commit()
        log(f"{season}: done in {time.time() - t0:.0f}s ({client.calls} calls total)")


def _derive_drives(conn, season):
    from .synthetic import _drives_from_plays
    p = pd.read_sql_query("SELECT * FROM plays WHERE season=? AND drive_id IS NOT NULL", conn, params=(season,))
    if p.empty:
        return
    d = _drives_from_plays(p)
    d["drive_number"] = d.groupby("game_id").cumcount() + 1
    insert_df(conn, "drives", d)


def _rosters(conn, client, season, workers, log):
    """Season rosters from appearances; position/size/class from the season athlete endpoint (cached)."""
    app = pd.read_sql_query("""SELECT pg.player_id, pg.team_id, COUNT(*) n FROM player_games pg JOIN games g ON g.game_id=pg.game_id
                               WHERE g.season=? GROUP BY 1,2""", conn, params=(season,))
    if app.empty:
        return
    app = app.sort_values("n").groupby("player_id").tail(1)
    have = {r[0] for r in conn.execute("SELECT player_id FROM rosters WHERE season=?", (season,))}
    info = {}
    # 1) current team rosters (one call per team) cover this season's players cheaply
    latest = conn.execute("SELECT MAX(season) FROM games").fetchone()[0]
    if season == latest:
        for tid in app.team_id.unique():
            d = client.roster(int(tid)) or {}
            for grp in d.get("athletes", []) or []:
                for a in grp.get("items", []) or []:
                    if a.get("id"):
                        info[int(a["id"])] = a
    # 2) season athlete lookups only for players with real involvement (others get usage-inferred positions)
    inv = pd.read_sql_query("""SELECT pg.player_id, SUM(pg.pass_att + pg.rush_att + pg.targets + pg.tackles + pg.fga + pg.punts) inv
        FROM player_games pg JOIN games g ON g.game_id=pg.game_id WHERE g.season=? GROUP BY 1""", conn, params=(season,))
    involved = set(inv[inv.inv >= 8].player_id.astype(int))
    known_pos = {r[0] for r in conn.execute("SELECT player_id FROM players WHERE position IS NOT NULL")}
    todo = [int(x) for x in app.player_id if int(x) not in have and int(x) not in info and int(x) in involved
            and int(x) not in known_pos]

    def fa(pid):
        return pid, client.athlete(season, pid)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(fa, p) for p in todo])):
            pid, d = fu.result()
            if d:
                info[pid] = d
            if (i + 1) % 1000 == 0:
                log(f"  {season}: athletes {i + 1}/{len(todo)} ({client.calls} calls)")
    rows = []
    for r in app.itertuples():
        d = info.get(int(r.player_id)) or {}
        pos = ESPN_POS.get(str((d.get("position") or {}).get("abbreviation") or "").upper())
        yrs = (d.get("experience") or {}).get("years")
        if d:
            conn.execute("""UPDATE players SET position=COALESCE(?, position), height_in=COALESCE(?, height_in),
                            weight_lb=COALESCE(?, weight_lb), first_name=COALESCE(first_name, ?), last_name=COALESCE(last_name, ?)
                            WHERE player_id=?""", (pos, d.get("height"), d.get("weight"), d.get("firstName"), d.get("lastName"), int(r.player_id)))
        rows.append(dict(player_id=int(r.player_id), team_id=int(r.team_id), season=season,
                         jersey=int(d["jersey"]) if str(d.get("jersey", "")).isdigit() else None,
                         position=pos, class_year=int(min(max(yrs, 1), 5)) if isinstance(yrs, (int, float)) and yrs else None))
    insert_df(conn, "rosters", pd.DataFrame(rows))
    # infer missing positions from usage (labelled by leaving players.position NULL -> roster position inferred)
    conn.execute("""UPDATE rosters SET position = (
        SELECT CASE WHEN SUM(pg.pass_att) > 5 THEN 'QB' WHEN SUM(pg.rush_att) > SUM(pg.targets) THEN 'RB'
                    WHEN SUM(pg.targets) > 0 THEN 'WR' WHEN SUM(pg.def_sacks) > 1 THEN 'EDGE'
                    WHEN SUM(pg.pbu) + SUM(pg.def_int) > 2 THEN 'CB' WHEN SUM(pg.tackles) > 0 THEN 'LB'
                    WHEN SUM(pg.fga) > 0 THEN 'K' WHEN SUM(pg.punts) > 0 THEN 'P' END
        FROM player_games pg JOIN games g ON g.game_id=pg.game_id WHERE pg.player_id=rosters.player_id AND g.season=rosters.season)
        WHERE season=? AND position IS NULL""", (season,))


def _returning(conn, season):
    """Share of prior-season offensive yards + tackle value returning to the same team."""
    prev = pd.read_sql_query("""SELECT pg.team_id, pg.player_id, SUM(pg.pass_yds + pg.rush_yds + pg.rec_yds) + 15*SUM(pg.tackles) prod
        FROM player_games pg JOIN games g ON g.game_id=pg.game_id WHERE g.season=? GROUP BY 1,2""", conn, params=(season - 1,))
    if prev.empty:
        return
    cur = pd.read_sql_query("SELECT player_id, team_id FROM rosters WHERE season=?", conn, params=(season,))
    prev = prev.merge(cur.assign(back=1), on=["player_id", "team_id"], how="left").fillna({"back": 0})
    rp = prev.assign(r=prev["prod"] * prev.back).groupby("team_id").agg(r=("r", "sum"), t=("prod", "sum"))
    for tid, row in rp.iterrows():
        if row.t > 0:
            conn.execute("UPDATE team_seasons SET returning_production=? WHERE team_id=? AND season=?", (float(row.r / row.t), int(tid), season))
