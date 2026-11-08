"""Recruiting, talent composite and transfer portal from CFBD, merged into an ESPN-built database.

Uses the free CFBD key (CFBD_API_KEY env var; never stored in the repo). ~2 calls per class/season, all cached on
disk, so the free tier's 1,000 calls/month is not a constraint.

Matching:
  recruits  -> players: CFBD `athleteId` is ESPN's athlete ID (verified: e.g. Arch Manning 4870906); fallback on
               normalised name + committed team appearing on that player's roster.
  teams     -> normalised school names (shared normaliser with the Polymarket mapper).
  portal    -> players: normalised first+last name AND (origin team on the prior-season roster OR destination team
               on the transfer-season roster). Ambiguous name matches are left unmatched rather than guessed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..db import insert_df, utcnow
from .polymarket import norm_name

SOURCE = "cfbd"


def _pname(s: str | None) -> str:
    s = (s or "").lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def team_map(conn) -> dict[str, int]:
    t = pd.read_sql_query("SELECT team_id, school, mascot, abbreviation FROM teams", conn)
    m = {}
    for r in t.itertuples():
        for k in (r.school, f"{r.school} {r.mascot or ''}".strip()):
            m.setdefault(norm_name(k), int(r.team_id))
    return m


def load_recruiting(conn, client, classes: list[int], seasons: list[int], log=print) -> dict:
    conn.execute("INSERT OR REPLACE INTO data_sources VALUES (?,?,?,?,?)",
                 (SOURCE, "CollegeFootballData.com API (free key): recruiting, talent, transfer portal", 0,
                  "Per CFBD terms of service; attribute CFBD", utcnow()))
    tm = team_map(conn)
    tid = lambda name: tm.get(norm_name(name))
    players = pd.read_sql_query("SELECT player_id, first_name, last_name FROM players", conn)
    players["k"] = (players.first_name.fillna("") + players.last_name.fillna("")).map(_pname)
    known = set(players.player_id.astype(int))
    ros = pd.read_sql_query("SELECT player_id, team_id, season FROM rosters", conn)
    stats = {"recruits": 0, "recruits_matched": 0, "talent_teams": 0, "portal": 0, "portal_matched": 0, "unmatched_teams": set()}

    # ---- recruiting classes
    rows = []
    for yr in classes:
        recs = client.get("/recruiting/players", year=yr, classification="HighSchool") or []
        for r in recs:
            pid = r.get("athleteId")
            pid = int(pid) if pid and str(pid).isdigit() and int(pid) in known else None
            team = tid(r.get("committedTo")) if r.get("committedTo") else None
            if pid is None and team is not None:   # fallback: name + committed team on a roster
                k = _pname(r.get("name"))
                cand = players[players.k == k]
                if len(cand):
                    cand = cand[cand.player_id.isin(ros[ros.team_id == team].player_id)]
                    if len(cand) == 1:
                        pid = int(cand.player_id.iloc[0])
            if r.get("committedTo") and team is None:
                stats["unmatched_teams"].add(r["committedTo"])
            rows.append(dict(player_id=pid, recruit_id=int(r["id"]), name=r.get("name"), class_year=yr,
                             position=r.get("position"), stars=r.get("stars"), rating=r.get("rating"),
                             national_rank=r.get("ranking"), committed_team_id=team, source=SOURCE))
        log(f"  class {yr}: {len(recs)} recruits")
    rec = pd.DataFrame(rows)
    if not rec.empty:
        # one profile per player (a player can appear twice, e.g. re-classified); keep the highest rating
        matched = rec[rec.player_id.notna()].sort_values("rating", ascending=False).drop_duplicates("player_id")
        rec = pd.concat([matched, rec[rec.player_id.isna()]])
        conn.execute("DELETE FROM recruiting_profiles WHERE source=?", (SOURCE,))
        insert_df(conn, "recruiting_profiles", rec)
        stats["recruits"] = len(rec)
        stats["recruits_matched"] = int(rec.player_id.notna().sum())

    # ---- talent composite
    for s in seasons:
        for t in client.get("/talent", year=s) or []:
            team = tid(t.get("team") or t.get("school"))
            if team is None:
                continue
            n = conn.execute("UPDATE team_seasons SET talent_composite=? WHERE team_id=? AND season=?",
                             (float(t["talent"]), team, s)).rowcount
            stats["talent_teams"] += n

    # ---- transfer portal
    trows = []
    for s in seasons:
        portal = client.get("/player/portal", year=s) or []
        prev_ros = ros[ros.season == s - 1]
        cur_ros = ros[ros.season == s]
        for i, t in enumerate(portal):
            o, d = tid(t.get("origin")), tid(t.get("destination"))
            k = _pname(f"{t.get('firstName', '')}{t.get('lastName', '')}")
            cand = players[players.k == k]
            pid = None
            if len(cand):
                ok = cand[cand.player_id.isin(prev_ros[prev_ros.team_id == o].player_id) |
                          cand.player_id.isin(cur_ros[cur_ros.team_id == d].player_id)] if (o or d) else cand.iloc[0:0]
                if len(ok) == 1:
                    pid = int(ok.player_id.iloc[0])
            trows.append(dict(transfer_id=int(s * 100000 + i), player_id=pid,
                              name=f"{t.get('firstName', '')} {t.get('lastName', '')}".strip(), position=t.get("position"),
                              season=s, from_team_id=o, to_team_id=d, stars=t.get("stars"), rating=t.get("rating"),
                              eligibility=t.get("eligibility"), transfer_date=t.get("transferDate"), source=SOURCE))
        log(f"  portal {s}: {len(portal)} entries")
    if trows:
        tr = pd.DataFrame(trows)
        conn.execute("DELETE FROM transfer_history WHERE source=?", (SOURCE,))
        insert_df(conn, "transfer_history", tr)
        stats["portal"] = len(tr)
        stats["portal_matched"] = int(tr.player_id.notna().sum())
    conn.commit()
    stats["unmatched_teams"] = sorted(stats["unmatched_teams"])[:25]
    return stats
