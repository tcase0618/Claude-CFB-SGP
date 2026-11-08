"""Player availability for upcoming games — the best free signal set, since no CFB injury feed is open.

Sources, strongest first:
  1. Official conference availability reports (Big Ten / SEC / ACC / Big 12 / MWC / MAC publish them). Their sites
     block automated access (Incapsula bot walls; the SEC's robots.txt disallows AI agents), so they are *imported*:
     paste a report into a CSV (team, player, status) and run `cli availability import`. OUT/DOUBTFUL removes the
     player from the simulation; QUESTIONABLE keeps him but flags every ticket he is on.
  2. ESPN roster removal: a player with 2026 production who is no longer on his team's current ESPN roster AND
     missed his team's most recent game (ESPN CFB rosters are incomplete, so removal alone is not evidence).
  3. Game-log flags (walk-forward on 2026 weeks 3-10, projected starters, appearance rate):
        missed his team's last game      62%  (vs 92% unflagged)   -> 'missed_last'
        missed 2 of the team's last 3    64%                      -> 'missed_2of3'
        last game volume < 40% of norm   81%                      -> 'left_early'
     Flags do not remove anyone (recency-weighted starter selection already uses the game log); the ticket builder
     skips flagged players by default and the card lists them for a manual check.

Writes `availability` (all signals, with evidence) and mirrors OUT/DOUBTFUL into `injuries`, which the parameter
estimator already reads when choosing starters.
"""
from __future__ import annotations

import csv
import re

import numpy as np
import pandas as pd

from ..db import insert_df, utcnow

REMOVE = {"OUT", "DOUBTFUL"}
STATUS_ALIASES = {"o": "OUT", "out": "OUT", "d": "DOUBTFUL", "doubtful": "DOUBTFUL", "q": "QUESTIONABLE",
                  "questionable": "QUESTIONABLE", "p": "PROBABLE", "probable": "PROBABLE", "available": "AVAILABLE",
                  "game-time decision": "QUESTIONABLE", "gtd": "QUESTIONABLE", "out for season": "OUT"}
SKILL_VOL = {"QB": "pass_att", "RB": "rush_att", "WR": "targets", "TE": "targets"}


def _pname(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def _store(conn, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    keys = df[["season", "week", "source"]].drop_duplicates()
    for k in keys.itertuples():
        conn.execute("DELETE FROM availability WHERE season=? AND week=? AND source=?", (int(k.season), int(k.week), k.source))
    insert_df(conn, "availability", df)
    rem = df[df.status.isin(REMOVE) & df.player_id.notna()]
    if len(rem):
        insert_df(conn, "injuries", pd.DataFrame({"player_id": rem.player_id.astype(int), "team_id": rem.team_id.astype(int),
                                                  "season": rem.season, "week": rem.week, "status": "OUT",
                                                  "reported_at": rem.fetched_at, "source": rem.source}))
    conn.commit()
    try:                                           # starter choice is cached per (db, season, week)
        from ..analytics.team_params import clear_cache
        clear_cache()
    except Exception:
        pass


def _add_missing_roster(conn, d: dict, team: int, season: int, known: set) -> int:
    """Players on the current ESPN roster but absent from our DB (they have no recorded stats yet). Without them a
    team can look like it has no healthy backup QB, so an OUT starter could not be replaced."""
    from .cfbd import POS_MAP
    rows_p, rows_r = [], []
    for g in d.get("athletes", []):
        for a in g.get("items", []):
            pid = str(a.get("id", ""))
            if not pid.isdigit() or (int(pid), team) in known:
                continue
            pos = ((a.get("position") or {}).get("abbreviation") or "").upper()
            pos = POS_MAP.get(pos, pos) or None
            rows_p.append(dict(player_id=int(pid), first_name=a.get("firstName"), last_name=a.get("lastName"),
                               position=pos, source="espn"))
            rows_r.append(dict(player_id=int(pid), team_id=team, season=season, position=pos))
            known.add((int(pid), team))
    if rows_p:
        insert_df(conn, "players", pd.DataFrame(rows_p), replace=False)
        insert_df(conn, "rosters", pd.DataFrame(rows_r), replace=False)
    return len(rows_r)


def game_log_flags(conn, season: int, week: int, teams: list[int] | None = None) -> pd.DataFrame:
    """Walk-forward flags for skill players with a real role, from games strictly before `week`."""
    pg = pd.read_sql_query("""SELECT pg.player_id, pg.team_id, g.week, pg.pass_att, pg.rush_att, pg.targets
        FROM player_games pg JOIN games g ON g.game_id=pg.game_id WHERE g.season=? AND g.week<?""", conn, params=(season, week))
    pos = pd.read_sql_query("SELECT player_id, position FROM rosters WHERE season=?", conn, params=(season,)) \
        .drop_duplicates("player_id").set_index("player_id").position
    if teams is not None:
        pg = pg[pg.team_id.isin(teams)]
    out = []
    tw = pg[["team_id", "week"]].drop_duplicates()
    for (pid, team), d in pg.groupby(["player_id", "team_id"]):
        p = pos.get(pid)
        col = SKILL_VOL.get(p)
        if col is None:
            continue
        v = d.sort_values("week")[col].fillna(0).values
        if v.max() < {"QB": 12, "RB": 8, "WR": 4, "TE": 3}[p]:
            continue                               # never had a starter-level role
        weeks = sorted(tw[tw.team_id == team].week)
        mine = set(d.week)
        last_team, last3 = weeks[-1], set(weeks[-3:])
        f = {}
        if last_team not in mine:
            f["missed_last"] = f"did not appear in week {last_team}"
        if len(last3 - mine) >= 2:
            f["missed_2of3"] = f"appeared in {len(last3 & mine)} of the team's last 3 games"
        if last_team in mine and len(v) > 2 and v[:-1].mean() > 0 and v[-1] < 0.4 * v[:-1].mean():
            f["left_early"] = f"{col.replace('_', ' ')} {int(v[-1])} in week {last_team} vs {v[:-1].mean():.1f} avg"
        for k, why in f.items():
            out.append(dict(player_id=int(pid), team_id=int(team), position=p, flag=k, evidence=why))
    return pd.DataFrame(out)


def snapshot_availability(conn, client, season: int, week: int, log=print) -> dict:
    """ESPN roster removal + game-log flags for every team playing in `week`."""
    teams = [int(t) for (t,) in conn.execute("""SELECT home_id FROM games WHERE season=? AND week=? AND completed=0
        UNION SELECT away_id FROM games WHERE season=? AND week=? AND completed=0""", (season, week, season, week))]
    now = utcnow()
    rows = []
    # 1) roster removal: produced this season, no longer on the current ESPN roster
    from .espn import SITE
    prod = pd.read_sql_query("""SELECT pg.player_id, pg.team_id, MAX(g.week) AS last_week, SUM(pg.pass_att) pa,
        SUM(pg.rush_att) ra, SUM(pg.targets) tg FROM player_games pg JOIN games g ON g.game_id=pg.game_id
        WHERE g.season=? GROUP BY 1, 2""", conn, params=(season,))
    prod = prod[(prod.pa >= 20) | (prod.ra >= 15) | (prod.tg >= 10)]
    team_last_week = {int(t): int(w) for t, w in conn.execute(
        "SELECT team_id, MAX(g.week) FROM player_games pg JOIN games g ON g.game_id=pg.game_id WHERE g.season=? GROUP BY 1",
        (season,))}
    n_rost = n_added = 0
    known = {(int(p), int(t)) for p, t in conn.execute("SELECT player_id, team_id FROM rosters WHERE season=?", (season,))}
    for t in teams:
        d = client.get(f"{SITE}/teams/{t}/roster")
        if not d or not d.get("athletes"):
            continue
        n_rost += 1
        on = {int(a["id"]) for g in d["athletes"] for a in g.get("items", []) if str(a.get("id", "")).isdigit()}
        n_added += _add_missing_roster(conn, d, t, season, known)
        inj = {int(a["id"]): a.get("injuries") for g in d["athletes"] for a in g.get("items", []) if a.get("injuries")}
        team_last = team_last_week.get(t)
        for r in prod[prod.team_id == t].itertuples():
            # ESPN's CFB rosters are incomplete (112 of 159 "missing" producers played the previous week), so a
            # missing roster entry only counts when the player also missed his team's most recent game
            if int(r.player_id) not in on and team_last is not None and int(r.last_week) < team_last:
                rows.append(dict(season=season, week=week, team_id=t, player_id=int(r.player_id), status="OUT",
                                 reason="off current ESPN roster and missed last game",
                                 evidence=f"last appeared week {int(r.last_week)}; team last played week {team_last}",
                                 source="espn_roster", fetched_at=now))
        for pid, items in inj.items():           # populated for NFL; empty for CFB today, kept in case ESPN adds it
            st = STATUS_ALIASES.get(str((items[0] or {}).get("status", "")).lower(), "QUESTIONABLE")
            rows.append(dict(season=season, week=week, team_id=t, player_id=pid, status=st, reason="ESPN injury listing",
                             evidence=str(items[0])[:200], source="espn_injury", fetched_at=now))
    removed = {r["player_id"] for r in rows if r["status"] in REMOVE}
    # 2) game-log flags
    fl = game_log_flags(conn, season, week, teams)
    for r in fl.itertuples():
        if r.player_id in removed:
            continue
        rows.append(dict(season=season, week=week, team_id=r.team_id, player_id=r.player_id, status="FLAG",
                         reason=r.flag, evidence=r.evidence, source="game_log", fetched_at=now))
    for src in ("espn_roster", "espn_injury", "game_log"):
        conn.execute("DELETE FROM availability WHERE season=? AND week=? AND source=?", (season, week, src))
    conn.execute("DELETE FROM injuries WHERE season=? AND week=? AND source='espn_roster'", (season, week))
    _store(conn, rows)
    out = {"teams": len(teams), "rosters_fetched": n_rost, "roster_players_added": n_added,
           "removed_not_on_roster": len(removed),
           "flags": int(len(fl)), "fetched_at": now}
    log(f"availability wk{week}: {out}")
    return out


def import_report(conn, season: int, week: int, path: str, source: str = "official_report", log=print) -> dict:
    """CSV with columns team, player, status (header row required; extra columns ignored). Team can be the school
    name or ESPN team id. Names are matched within the team's roster; unmatched rows are reported, never guessed."""
    from .polymarket import norm_name
    teams = pd.read_sql_query("SELECT team_id, school, abbreviation FROM teams", conn)
    tmap = {}
    for r in teams.itertuples():
        for k in (r.school, r.abbreviation, str(r.team_id)):
            if k:
                tmap[norm_name(str(k))] = int(r.team_id)
    ros = pd.read_sql_query("""SELECT r.player_id, r.team_id, p.first_name, p.last_name FROM rosters r
        JOIN players p ON p.player_id=r.player_id WHERE r.season=?""", conn, params=(season,))
    ros["k"] = (ros.first_name.fillna("") + ros.last_name.fillna("")).map(_pname)
    now = utcnow()
    rows, unmatched = [], []
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            rec = {k.strip().lower(): (v or "").strip() for k, v in rec.items() if k}
            t = tmap.get(norm_name(rec.get("team", "")))
            st = STATUS_ALIASES.get(rec.get("status", "").lower(), rec.get("status", "").upper() or "OUT")
            cand = ros[(ros.team_id == t) & (ros.k == _pname(rec.get("player", "")))] if t else ros.iloc[0:0]
            if len(cand) != 1:
                unmatched.append(rec)
                continue
            rows.append(dict(season=season, week=week, team_id=t, player_id=int(cand.player_id.iloc[0]), status=st,
                             reason=rec.get("note") or "official availability report", evidence=rec.get("status"),
                             source=source, fetched_at=now))
    conn.execute("DELETE FROM injuries WHERE season=? AND week=? AND source=?", (season, week, source))
    _store(conn, rows)
    log(f"imported {len(rows)} rows, {len(unmatched)} unmatched")
    return {"imported": len(rows), "unmatched": unmatched}


def availability_for(conn, season: int, week: int, player_ids: list[int]) -> dict[int, list[dict]]:
    if not player_ids:
        return {}
    q = f"""SELECT player_id, status, reason, evidence, source FROM availability WHERE season=? AND week=?
            AND player_id IN ({','.join(str(int(p)) for p in player_ids)})"""
    out: dict[int, list[dict]] = {}
    for r in conn.execute(q, (season, week)).fetchall():
        out.setdefault(int(r[0]), []).append({"status": r[1], "reason": r[2], "evidence": r[3], "source": r[4]})
    return out
