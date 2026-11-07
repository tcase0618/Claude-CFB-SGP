"""Synthetic demonstration league.

PURPOSE: exercise and validate every pipeline stage end-to-end when licensed
real data (CFBD_API_KEY) is not available. All entities are FICTIONAL
(fictional schools, players, conferences). Every row is tagged
source='synthetic' and lives in its own database file, so it can never be
mistaken for, or mixed with, real data.

Generative story (ground truth is known, which lets us test whether the
analytics recover it):
  recruiting stars -> latent player skill -> development by class year
  -> depth chart (with random in-season injuries) -> team unit parameters
  -> play-by-play simulation (cfbquant.sim.engine) -> plays, play_events,
  drives, box scores -> season rollover (graduation, transfers, new class).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..db import insert_df, utcnow
from ..sim.engine import LOG_COLS, PT_NAMES, simulate
from ..sim.params import (DEF_SLOTS, OFF_SLOTS, SLOT, League, TeamParams, logit,
                          sigmoid, stack_params)

SOURCE = "synthetic"

CONFERENCES = [("ATL", "Atlantic Ridge Conference"), ("PRA", "Prairie Athletic League"),
               ("SUN", "Sun Coast Conference"), ("MTN", "Mountain Frontier Conference")]
PLACES = ["Northgate", "Cedar Valley", "Iron Bluff", "Lakeshore", "Red Mesa", "Granite Falls", "Harbor City",
          "Pinecrest", "Silver Lake", "Bayview", "Stonebridge", "Westmoor", "Eastfield", "Clearwater",
          "Ridgeway", "Fox Hollow", "Oak Ridge", "Summit", "Blue River", "Copper Canyon", "Maple Grove",
          "Thunder Bay", "Coral Point", "High Plains", "Riverside", "Ashford", "Kingsport", "Brookhaven",
          "Elk Creek", "Sandstone", "Willow Bend", "Highland", "Grand Prairie", "Crescent", "Falcon Ridge",
          "Glenwood", "Marshall Flats", "Pioneer", "Shadow Pines", "Wolf Creek", "Desert Springs",
          "Timberline", "Bramble", "Hawthorne", "Ironwood", "Seaside", "Sterling", "Valley Forge"]
SUFFIX = ["State", "University", "Tech", "A&M", "College", "State"]
MASCOTS = ["Hawks", "Wolves", "Rams", "Titans", "Comets", "Bison", "Storm", "Knights", "Raptors", "Miners",
           "Mustangs", "Pirates", "Owls", "Cougars", "Falcons", "Vipers", "Stallions", "Grizzlies"]
FIRST = ["Jalen", "Marcus", "Tyler", "Caleb", "Isaiah", "Brandon", "Devin", "Jordan", "Cole", "Mason", "Elijah",
         "Trey", "Darius", "Logan", "Andre", "Nate", "Xavier", "Malik", "Owen", "Reed", "Carter", "Bryce", "Jace",
         "Kendall", "Micah", "Tate", "Quinn", "Rashad", "Dillon", "Gavin", "Hunter", "Jaylen", "Keon", "Luke",
         "Miles", "Noah", "Parker", "Roman", "Silas", "Zane"]
LAST = ["Hollins", "Brooks", "Carver", "Dawson", "Ellison", "Fowler", "Garrison", "Hayes", "Ingram", "Jennings",
        "Kirkland", "Lawson", "Maddox", "Norwood", "Oakley", "Prescott", "Quarles", "Ramsey", "Sutton", "Talley",
        "Underwood", "Vance", "Whitfield", "Yates", "Ashby", "Barlow", "Cobb", "Draper", "Easton", "Fenwick",
        "Griggs", "Holloway", "Irby", "Jessup", "Kimbrough", "Langford", "Merritt", "Nixon", "Ogden", "Pruitt"]

# roster template: position -> count. Slots are filled from the depth chart.
ROSTER = {"QB": 3, "RB": 3, "WR": 5, "TE": 2, "OT": 3, "OG": 3, "C": 2, "DL": 4, "EDGE": 3, "LB": 4,
          "CB": 4, "S": 3, "K": 1, "P": 1}
SLOT_POS = {"QB1": ("QB", 0), "QB2": ("QB", 1), "RB1": ("RB", 0), "RB2": ("RB", 1), "WR1": ("WR", 0),
            "WR2": ("WR", 1), "WR3": ("WR", 2), "WR4": ("WR", 3), "TE1": ("TE", 0), "TE2": ("TE", 1),
            "K": ("K", 0), "P": ("P", 0), "DL1": ("EDGE", 0), "DL2": ("DL", 0), "DL3": ("DL", 1),
            "DL4": ("EDGE", 1), "LB1": ("LB", 0), "LB2": ("LB", 1), "LB3": ("LB", 2), "CB1": ("CB", 0),
            "CB2": ("CB", 1), "S1": ("S", 0), "S2": ("S", 1), "NB": ("CB", 2)}
HEIGHT = {"QB": 74, "RB": 70, "WR": 72, "TE": 76, "OT": 78, "OG": 76, "C": 75, "DL": 75, "EDGE": 76, "LB": 73,
          "CB": 71, "S": 72, "K": 71, "P": 73}
WEIGHT = {"QB": 215, "RB": 210, "WR": 195, "TE": 245, "OT": 310, "OG": 315, "C": 300, "DL": 300, "EDGE": 255,
          "LB": 235, "CB": 190, "S": 205, "K": 190, "P": 200}


@dataclass
class Player:
    pid: int
    first: str
    last: str
    pos: str
    team: int
    class_year: int
    stars: int
    rating: float
    base: float
    dev: float
    entry: int
    height: int
    weight: int
    out_until: int = -1

    def skill(self) -> float:
        # development curve: rapid early growth, flattening by year 4-5
        yrs = min(self.class_year - 1, 4)
        return self.base + self.dev * (1 - np.exp(-0.7 * yrs)) * 1.6


@dataclass
class Team:
    tid: int
    school: str
    mascot: str
    abbr: str
    conf: str
    prestige: float        # drives recruiting quality
    pace: float
    pass_rate: float
    go_aggr: float
    adot_style: float
    players: list = field(default_factory=list)


class SyntheticLeague:
    def __init__(self, seed: int = 7, n_per_conf: int = 12):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.league = League()
        self.next_pid = 100000
        self.next_recruit = 1
        self.next_transfer = 1
        self.teams: list[Team] = []
        tid = 1000
        places = list(PLACES)
        self.rng.shuffle(places)
        k = 0
        for cid, _ in CONFERENCES:
            for _ in range(n_per_conf):
                place = places[k]; k += 1
                suffix = SUFFIX[int(self.rng.integers(len(SUFFIX)))]
                school = f"{place} {suffix}" if suffix not in ("University", "College") else f"{place} {suffix}"
                t = Team(tid, school, MASCOTS[int(self.rng.integers(len(MASCOTS)))],
                         "".join(w[0] for w in place.split())[:2].upper() + suffix[0],
                         cid, float(self.rng.normal(0, 1)), float(self.rng.normal(31.0, 2.4)),
                         float(np.clip(self.rng.normal(0.50, 0.045), 0.36, 0.64)),
                         float(self.rng.normal(0, 0.45)), float(self.rng.normal(0, 1.0)))
                self.teams.append(t)
                tid += 1
        self.team_by_id = {t.tid: t for t in self.teams}
        self.players: dict[int, Player] = {}
        self.recruits: list[dict] = []
        self.transfers: list[dict] = []

    # ------------------------------------------------------------------ roster
    def _new_player(self, team: Team, pos: str, class_year: int, season_entry: int) -> Player:
        rng = self.rng
        p_star = sigmoid(np.array([-1.5, 0.3, 1.0]) - 0.9 * team.prestige)  # P(stars<=2,3,4)
        u = rng.random()
        stars = 2 if u < p_star[0] else 3 if u < p_star[1] else 4 if u < p_star[2] else 5
        rating = float(np.clip(0.70 + 0.055 * stars + rng.normal(0, 0.018), 0.70, 1.0))
        base = 0.55 * (stars - 3.2) + rng.normal(0, 0.55) - 0.6
        dev = float(np.clip(rng.normal(0.55, 0.25), 0.05, 1.2))
        p = Player(self.next_pid, FIRST[int(rng.integers(len(FIRST)))], LAST[int(rng.integers(len(LAST)))], pos,
                   team.tid, class_year, stars, rating, float(base), dev, season_entry,
                   int(HEIGHT[pos] + rng.integers(-2, 3)), int(WEIGHT[pos] + rng.integers(-15, 16)))
        self.next_pid += 1
        self.players[p.pid] = p
        team.players.append(p.pid)
        self.recruits.append(dict(recruit_id=self.next_recruit, player_id=p.pid, name=f"{p.first} {p.last}",
                                  class_year=season_entry, position=pos, stars=stars, rating=rating,
                                  national_rank=None, committed_team_id=team.tid, source=SOURCE))
        self.next_recruit += 1
        return p

    def init_rosters(self, season: int):
        for t in self.teams:
            for pos, n in ROSTER.items():
                for i in range(n + 1):   # one extra for depth
                    cy = int(self.rng.integers(1, 6))
                    self._new_player(t, pos, cy, season - cy + 1)

    def depth(self, team: Team, week: int) -> dict[str, list[int]]:
        by_pos: dict[str, list[Player]] = {}
        for pid in team.players:
            p = self.players[pid]
            if p.out_until >= week:
                continue
            by_pos.setdefault(p.pos, []).append(p)
        return {pos: [p.pid for p in sorted(lst, key=lambda x: -x.skill())] for pos, lst in by_pos.items()}

    def slot_players(self, team: Team, week: int) -> list[int]:
        d = self.depth(team, week)
        out = []
        for s in OFF_SLOTS + DEF_SLOTS:
            pos, i = SLOT_POS[s]
            lst = d.get(pos, [])
            if i < len(lst):
                out.append(lst[i])
            else:   # fall back to any healthy body at a related position
                fb = d.get({"EDGE": "DL", "DL": "EDGE", "S": "CB", "CB": "S"}.get(pos, pos), [])
                out.append(fb[min(i, len(fb) - 1)] if fb else (lst[-1] if lst else team.players[0]))
        return out

    # ----------------------------------------------------------- team params
    def team_params(self, team: Team, slots: list[int]) -> TeamParams:
        L = self.league
        sk = {s: self.players[pid].skill() for s, pid in zip(OFF_SLOTS + DEF_SLOTS, slots)}
        d = self.depth(team, 10_000)
        ol = np.mean([self.players[p].skill() for pos in ("OT", "OG", "C") for p in d.get(pos, [])[:2]] or [0])
        qb = sk["QB1"]
        rec = np.mean([sk["WR1"], sk["WR2"], sk["WR3"], sk["TE1"]])
        rb = sk["RB1"]
        dl = np.mean([sk["DL1"], sk["DL2"], sk["DL3"], sk["DL4"]])
        lb = np.mean([sk["LB1"], sk["LB2"], sk["LB3"]])
        db = np.mean([sk["CB1"], sk["CB2"], sk["S1"], sk["S2"], sk["NB"]])
        tw = np.array([sk[s] for s in ["RB1", "RB2", "WR1", "WR2", "WR3", "WR4", "TE1", "TE2"]])
        tbase = np.array([0.08, 0.03, 0.22, 0.19, 0.15, 0.06, 0.17, 0.05])
        tshare = tbase * np.exp(0.45 * tw)
        rbase = np.array([0.10 + 0.12 * max(qb, 0) * (team.pass_rate < 0.5), 0.55, 0.22, 0.04])
        rw = np.array([sk["QB1"], sk["RB1"], sk["RB2"], sk["WR1"]])
        rshare = rbase * np.exp(0.35 * rw)
        dsk = np.array([sk[s] for s in DEF_SLOTS])
        tr = np.array(TeamParams().tackle_run) * np.exp(0.25 * dsk)
        tp = np.array(TeamParams().tackle_pass) * np.exp(0.25 * dsk)
        ss = np.array(TeamParams().sack_share) * np.exp(0.6 * dsk)
        ish = np.array(TeamParams().int_share) * np.exp(0.5 * dsk)
        return TeamParams(
            pass_rate=team.pass_rate, pace=team.pace, go_aggr=team.go_aggr,
            sack_off=float(sigmoid(logit(L.sack) - 0.28 * ol - 0.10 * qb)),
            scramble=float(np.clip(L.scramble + 0.01 * qb, 0.01, 0.1)),
            int_off=float(sigmoid(logit(L.intr) - 0.22 * qb)),
            comp_off=float(sigmoid(logit(L.comp) + 0.20 * qb + 0.08 * rec + 0.04 * ol)),
            adot=float(L.adot + 0.9 * team.adot_style),
            yac_off=float(L.yac + 0.55 * rec),
            rcore_off=float(L.rcore + 0.30 * rb + 0.32 * ol),
            rexp_off=float(sigmoid(logit(L.rexp) + 0.18 * rb + 0.08 * ol)),
            fumble_off=L.fumble,
            sack_def=float(sigmoid(logit(L.sack) + 0.30 * dl)),
            int_def=float(sigmoid(logit(L.intr) + 0.20 * db)),
            comp_def=float(sigmoid(logit(L.comp) - 0.20 * db - 0.05 * lb)),
            yac_def=float(L.yac - 0.35 * lb - 0.25 * db),
            rcore_def=float(L.rcore - 0.30 * dl - 0.22 * lb),
            rexp_def=float(sigmoid(logit(L.rexp) - 0.15 * lb - 0.12 * db)),
            fumble_def=L.fumble,
            fg_skill=float(0.5 * sk["K"]), punt_net=float(39 + 2.0 * sk["P"]),
            target_share=tuple(tshare), rush_share=tuple(rshare),
            catch_adj=tuple(0.12 * (tw - tw.mean())),
            rush_adj=(0.0, 0.0, float(0.3 * (sk["RB2"] - sk["RB1"])), float(0.3 * (sk["WR1"] - sk["RB1"]))),
            tackle_run=tuple(tr), tackle_pass=tuple(tp), sack_share=tuple(ss), int_share=tuple(ish))

    # -------------------------------------------------------------- schedule
    def schedule(self, season: int) -> list[tuple[int, int, int]]:
        """Returns (week, home_id, away_id). 4 non-conference + 8 conference games."""
        rng = self.rng
        games = []
        by_conf = {c: [t.tid for t in self.teams if t.conf == c] for c, _ in CONFERENCES}
        all_ids = [t.tid for t in self.teams]
        for w in range(1, 5):
            ids = list(all_ids)
            for _ in range(50):
                rng.shuffle(ids)
                pairs = [(ids[i], ids[i + 1]) for i in range(0, len(ids), 2)]
                if all(self.team_by_id[a].conf != self.team_by_id[b].conf for a, b in pairs):
                    break
            for a, b in pairs:
                games.append((w, a, b) if rng.random() < 0.5 else (w, b, a))
        for c, ids in by_conf.items():
            ids = list(ids)
            rng.shuffle(ids)
            n = len(ids)
            rounds = []
            arr = ids[:]
            for r in range(n - 1):
                rounds.append([(arr[i], arr[n - 1 - i]) for i in range(n // 2)])
                arr = [arr[0]] + [arr[-1]] + arr[1:-1]
            for k, rnd in enumerate(rounds[:8]):
                for i, (a, b) in enumerate(rnd):
                    games.append((5 + k, a, b) if (k + i + season) % 2 == 0 else (5 + k, b, a))
        return games

    # --------------------------------------------------------------- season
    def play_season(self, conn, season: int, game_id_start: int) -> int:
        rng = self.rng
        sched = self.schedule(season)
        gid = game_id_start
        season_start = datetime(season, 8, 31, 19, 30, tzinfo=timezone.utc)
        rows_games, rows_inj = [], []
        for week in sorted({w for w, _, _ in sched}):
            wk = [(h, a) for w, h, a in sched if w == week]
            # in-season injuries (announced before kickoff)
            for t in self.teams:
                for pid in t.players:
                    p = self.players[pid]
                    if p.out_until < week and rng.random() < 0.012:
                        p.out_until = week + int(rng.integers(0, 4))
                        for wk_out in range(week, p.out_until + 1):
                            rows_inj.append(dict(player_id=pid, team_id=t.tid, season=season, week=wk_out,
                                                 status="OUT", reported_at=None, source=SOURCE))
            pairs, slotmap, meta = [], [], []
            for h, a in wk:
                th, ta = self.team_by_id[h], self.team_by_id[a]
                sh, sa = self.slot_players(th, week), self.slot_players(ta, week)
                pairs.append((self.team_params(th, sh), self.team_params(ta, sa)))
                slotmap.append((sh, sa))
                meta.append((gid, h, a))
                gid += 1
            P = stack_params(pairs)
            res = simulate(P, self.league, seed=int(rng.integers(1 << 31)), record=True, track_players=False)
            kickoff = season_start + timedelta(days=7 * (week - 1))
            for lane, (g, h, a) in enumerate(meta):
                rows_games.append(dict(
                    game_id=g, season=season, week=week, season_type="regular",
                    start_date=kickoff.isoformat(), neutral_site=0,
                    conference_game=int(self.team_by_id[h].conf == self.team_by_id[a].conf),
                    venue_id=h, home_id=h, away_id=a, home_points=int(res.home_pts[lane]),
                    away_points=int(res.away_pts[lane]), completed=1,
                    temperature_f=float(np.round(rng.normal(75 - 2.2 * (week - 1), 8), 0)),
                    wind_mph=float(np.round(abs(rng.normal(6, 4)), 0)),
                    weather_desc="Clear", source=SOURCE))
            insert_df(conn, "games", pd.DataFrame(rows_games))
            rows_games = []
            self._write_plays(conn, season, week, meta, slotmap, res.log)
        insert_df(conn, "injuries", pd.DataFrame(rows_inj).drop_duplicates(["player_id", "season", "week"]) if rows_inj else None)
        return gid

    def _write_plays(self, conn, season, week, meta, slotmap, log):
        lg = pd.DataFrame({k: log[k] for k in LOG_COLS})
        lg = lg.sort_values(["lane", "step"]).reset_index(drop=True)
        plays, events, pgame = [], [], {}
        base_pid = None
        for lane, grp in lg.groupby("lane", sort=True):
            g, h, a = meta[lane]
            team_of = (h, a)
            slots = slotmap[lane]
            n = 0
            for r in grp.itertuples(index=False):
                off_side = int(r.offense)
                off_id, def_id = team_of[off_side], team_of[1 - off_side]
                ptype = PT_NAMES[int(r.ptype)]
                pid = g * 1000 + n
                n += 1
                is_ko = ptype == "kickoff"
                td_points = 6 if r.touchdown else 0
                pts = (td_points if r.touchdown else int(r.points)) - int(r.def_points)
                if ptype == "field_goal" and r.fg_made:
                    pts = 3
                plays.append(dict(
                    play_id=pid, game_id=g, drive_id=g * 100 + int(r.drive), season=season, week=week, play_number=n,
                    offense_id=off_id, defense_id=def_id, home_id=h, period=int(r.period),
                    clock_seconds=int(max(r.clock, 0)) if r.period <= 4 else None,
                    offense_score=int(r.off_score), defense_score=int(r.def_score),
                    down=None if is_ko else int(r.down), distance=None if is_ko else int(r.distance),
                    yards_to_goal=65 if is_ko else int(r.ytg),
                    play_type=ptype, yards_gained=int(r.yards),
                    air_yards=None if np.isnan(r.air) else float(r.air), pass_location=None,
                    completion=int(r.completion), interception=int(r.interception), sack=int(r.sack),
                    fumble_lost=int(r.fumble_lost), touchdown=int(r.touchdown), first_down=int(r.first_down),
                    scramble=int(r.scramble), points=pts, penalty_yards=int(r.penalty_yards),
                    fg_distance=int(r.fg_dist) if r.fg_dist else None,
                    fg_made=int(r.fg_made) if ptype == "field_goal" else None,
                    ep_before=None, ep_after=None, epa=None, success=None, garbage_time=0,
                    play_text=None, source=SOURCE))
                if r.pat_result >= 0:
                    two = r.pat_result >= 20
                    ok = int(r.pat_result % 10)
                    plays.append(dict(
                        play_id=g * 1000 + n, game_id=g, drive_id=g * 100 + int(r.drive), season=season, week=week,
                        play_number=n + 1, offense_id=off_id, defense_id=def_id, home_id=h, period=int(r.period),
                        clock_seconds=int(max(r.clock, 0)) if r.period <= 4 else None,
                        offense_score=int(r.off_score), defense_score=int(r.def_score), down=None, distance=None,
                        yards_to_goal=3, play_type="two_point" if two else "pat", yards_gained=0, air_yards=None,
                        pass_location=None, completion=0, interception=0, sack=0, fumble_lost=0, touchdown=0,
                        first_down=0, scramble=0, points=(2 if two else 1) * ok, penalty_yards=0, fg_distance=None,
                        fg_made=None, ep_before=None, ep_after=None, epa=None, success=None, garbage_time=0,
                        play_text=None, source=SOURCE))
                    if not two:
                        k_id = slots[off_side][SLOT["K"]]
                        events.append((g * 1000 + n, k_id, off_id, "kicker", float(ok)))
                        self._acc(pgame, k_id, g, off_id, xpa=1, xpm=ok)
                    n += 1
                # player attribution (none for conversion attempts / kickoffs)
                if ptype in ("two_point", "kickoff"):
                    continue
                so, sd = slots[off_side], slots[1 - off_side]
                y = int(r.yards)
                cred = min(y, int(r.ytg)) if not is_ko else y
                if r.passer >= 0:
                    qb = so[int(r.passer)]
                    events.append((pid, qb, off_id, "passer", float(cred)))
                    if ptype == "pass":
                        self._acc(pgame, qb, g, off_id, pass_att=1, pass_cmp=int(r.completion),
                                  pass_yds=cred if r.completion else 0, pass_td=int(r.touchdown and r.completion),
                                  pass_int=int(r.interception))
                    elif ptype == "sack":
                        self._acc(pgame, qb, g, off_id, sacks_taken=1)
                if r.target >= 0 and ptype == "pass":
                    tg = so[int(r.target)]
                    events.append((pid, tg, off_id, "target", float(cred if r.completion else 0)))
                    self._acc(pgame, tg, g, off_id, targets=1, receptions=int(r.completion),
                              rec_yds=cred if r.completion else 0, rec_td=int(r.touchdown and r.completion))
                if r.rusher >= 0 and ptype in ("rush", "scramble", "kneel"):
                    rb = so[int(r.rusher)]
                    events.append((pid, rb, off_id, "rusher", float(cred)))
                    if ptype != "kneel":
                        self._acc(pgame, rb, g, off_id, rush_att=1, rush_yds=cred, rush_td=int(r.touchdown))
                for col, role, kw in (("tackler", "tackler", "tackles"), ("sacker", "sacker", "def_sacks"),
                                      ("interceptor", "interceptor", "def_int"), ("pbu", "pbu", "pbu"),
                                      ("forced_fumble", "forced_fumble", "forced_fumbles")):
                    v = int(getattr(r, col))
                    if v >= 0:
                        dp = sd[v]
                        events.append((pid, dp, def_id, role, float(y)))
                        extra = {kw: 1}
                        if role == "tackler" and y < 0 and ptype in ("rush", "scramble"):
                            extra["tfl"] = 1
                        if role == "sacker":
                            extra.update(tackles=1, tfl=1)
                        self._acc(pgame, dp, g, def_id, **extra)
                if ptype == "field_goal":
                    k_id = so[SLOT["K"]]
                    events.append((pid, k_id, off_id, "kicker", float(r.fg_dist)))
                    self._acc(pgame, k_id, g, off_id, fga=1, fgm=int(r.fg_made))
                if ptype == "punt":
                    p_id = so[SLOT["P"]]
                    events.append((pid, p_id, off_id, "punter", float(y)))
                    self._acc(pgame, p_id, g, off_id, punts=1)
        pdf = pd.DataFrame(plays)
        insert_df(conn, "plays", pdf)
        ev = pd.DataFrame(events, columns=["play_id", "player_id", "team_id", "role", "value"]).drop_duplicates(
            ["play_id", "player_id", "role"])
        insert_df(conn, "play_events", ev)
        pg = pd.DataFrame([dict(player_id=k[0], game_id=k[1], team_id=v.pop("_team"), source=SOURCE, **v)
                           for k, v in pgame.items()])
        insert_df(conn, "player_games", pg)
        insert_df(conn, "drives", _drives_from_plays(pdf))

    @staticmethod
    def _acc(store, pid, gid, team, **kw):
        d = store.setdefault((pid, gid), {"_team": team})
        for k, v in kw.items():
            d[k] = d.get(k, 0) + int(v)

    # -------------------------------------------------------------- rollover
    def rollover(self, new_season: int):
        rng = self.rng
        for t in self.teams:
            keep = []
            for pid in t.players:
                p = self.players[pid]
                p.out_until = -1
                leave = (p.class_year >= 5) or (p.class_year == 4 and rng.random() < 0.75) or \
                        (p.class_year == 3 and p.skill() > 1.6 and rng.random() < 0.5)   # early draft entry
                if leave:
                    continue
                # transfer portal: low-usage players with upside, or random
                if rng.random() < 0.06:
                    dest = self.teams[int(rng.integers(len(self.teams)))]
                    if dest.tid != t.tid:
                        self.transfers.append(dict(
                            transfer_id=self.next_transfer, player_id=pid, name=f"{p.first} {p.last}",
                            position=p.pos, season=new_season, from_team_id=t.tid, to_team_id=dest.tid,
                            stars=p.stars, rating=p.rating, eligibility="Immediate",
                            transfer_date=f"{new_season}-01-10", source=SOURCE))
                        self.next_transfer += 1
                        p.team = dest.tid
                        p.class_year += 1
                        dest.players.append(pid)
                        continue
                p.class_year += 1
                keep.append(pid)
            t.players = keep
        # refill rosters with a new recruiting class
        for t in self.teams:
            t.prestige = 0.85 * t.prestige + 0.15 * float(rng.normal(0, 1))
            have: dict[str, int] = {}
            for pid in t.players:
                have[self.players[pid].pos] = have.get(self.players[pid].pos, 0) + 1
            for pos, n in ROSTER.items():
                for _ in range(max(0, n + 1 - have.get(pos, 0))):
                    self._new_player(t, pos, 1, new_season)

    # ------------------------------------------------------------- persist
    def write_static(self, conn, season: int):
        insert_df(conn, "conferences", pd.DataFrame([dict(conference_id=c, name=n, division="fbs")
                                                     for c, n in CONFERENCES]))
        insert_df(conn, "teams", pd.DataFrame([dict(team_id=t.tid, school=t.school, mascot=t.mascot,
                                                    abbreviation=t.abbr, color=None, alt_color=None,
                                                    venue_id=t.tid, source=SOURCE) for t in self.teams]))
        insert_df(conn, "venues", pd.DataFrame([dict(
            venue_id=t.tid, name=f"{t.school} Stadium", city=t.school.split()[0], state=None,
            capacity=int(35000 + 30000 * sigmoid(t.prestige) + self.rng.integers(0, 15000)), grass=1, dome=0,
            elevation_m=float(max(0, self.rng.normal(250, 300))), latitude=None, longitude=None, timezone=None)
            for t in self.teams]))

    def write_season_rosters(self, conn, season: int):
        rows, players, ts = [], [], []
        for t in self.teams:
            ratings = []
            for pid in t.players:
                p = self.players[pid]
                rows.append(dict(player_id=pid, team_id=t.tid, season=season, jersey=int(pid % 99) + 1,
                                 position=p.pos, class_year=p.class_year))
                players.append(dict(player_id=pid, first_name=p.first, last_name=p.last, position=p.pos,
                                    height_in=p.height, weight_lb=p.weight, birth_year=season - 17 - p.class_year,
                                    hometown=None, source=SOURCE))
                ratings.append(p.rating)
            ts.append(dict(team_id=t.tid, season=season, conference_id=t.conf, division="fbs",
                           talent_composite=float(np.round(1000 * np.mean(sorted(ratings)[-25:]) - 0, 1)),
                           returning_production=None, head_coach_id=None))
        insert_df(conn, "players", pd.DataFrame(players))
        insert_df(conn, "rosters", pd.DataFrame(rows))
        insert_df(conn, "team_seasons", pd.DataFrame(ts))
        # depth chart snapshot (preseason)
        dc = []
        for t in self.teams:
            for pos, lst in self.depth(t, 0).items():
                for i, pid in enumerate(lst[:4]):
                    dc.append(dict(team_id=t.tid, season=season, week=0, position=pos, depth=i + 1,
                                   player_id=pid, source=SOURCE))
        insert_df(conn, "depth_charts", pd.DataFrame(dc))

    def truth(self, season: int) -> pd.DataFrame:
        """Ground-truth team parameters at full health (for recovery tests)."""
        rows = []
        for t in self.teams:
            tp = self.team_params(t, self.slot_players(t, 0))
            rows.append(dict(team_id=t.tid, season=season, **{k: getattr(tp, k) for k in
                             ("comp_off", "comp_def", "rcore_off", "rcore_def", "sack_off", "sack_def",
                              "int_off", "int_def", "pace", "pass_rate")}))
        return pd.DataFrame(rows)


def _drives_from_plays(pdf: pd.DataFrame) -> pd.DataFrame:
    sc = pdf[~pdf.play_type.isin(["kickoff", "pat", "two_point"])]
    out = []
    for (gid, did), d in sc.groupby(["game_id", "drive_id"], sort=False):
        first, last = d.iloc[0], d.iloc[-1]
        pts = int(pdf[(pdf.drive_id == did)].points.clip(lower=0).sum())
        if last.touchdown:
            res = "TD"
        elif last.play_type == "field_goal":
            res = "FG" if last.fg_made else "MISSED_FG"
        elif last.play_type == "punt":
            res = "PUNT"
        elif last.interception or last.fumble_lost:
            res = "TURNOVER"
        elif last.points < 0:
            res = "SAFETY"
        elif last.down == 4:
            res = "DOWNS"
        else:
            res = "END_HALF"
        out.append(dict(drive_id=int(did), game_id=int(gid), offense_id=int(first.offense_id),
                        defense_id=int(first.defense_id), drive_number=int(did % 100),
                        start_yards_to_goal=int(first.yards_to_goal), end_yards_to_goal=int(last.yards_to_goal),
                        plays=int(len(d)), yards=int(first.yards_to_goal - last.yards_to_goal + last.yards_gained)
                        if res == "TD" else int(first.yards_to_goal - last.yards_to_goal),
                        result=res, points=pts if res in ("TD", "FG") else 0, start_period=int(first.period)))
    return pd.DataFrame(out)


def build_synthetic(conn, seasons=(2023, 2024, 2025), seed: int = 7) -> SyntheticLeague:
    conn.execute("INSERT OR REPLACE INTO data_sources VALUES (?,?,?,?,?)",
                 (SOURCE, "Fictional demonstration league generated by cfbquant.ingest.synthetic "
                          "(all teams/players fictional; NOT real college football data)", 1,
                  "Generated; no license restrictions", utcnow()))
    lg = SyntheticLeague(seed=seed)
    lg.init_rosters(seasons[0])
    lg.write_static(conn, seasons[0])
    gid = int(seasons[0]) * 10000
    truths = []
    for i, s in enumerate(seasons):
        if i > 0:
            lg.rollover(s)
        lg.write_season_rosters(conn, s)
        truths.append(lg.truth(s))
        gid = lg.play_season(conn, s, s * 10000)
        conn.commit()
    insert_df(conn, "recruiting_profiles", pd.DataFrame(lg.recruits))
    if lg.transfers:
        insert_df(conn, "transfer_history", pd.DataFrame(lg.transfers))
    # returning production (share of prior-season offensive yards returning)
    _returning_production(conn, seasons)
    pd.concat(truths).to_sql("synthetic_truth", conn, if_exists="replace", index=False)
    conn.commit()
    return lg


def _returning_production(conn, seasons):
    for s in seasons[1:]:
        q = """
        SELECT pg.team_id, pg.player_id, SUM(pg.pass_yds+pg.rush_yds+pg.rec_yds) AS yds,
               SUM(pg.tackles) AS tkl
        FROM player_games pg JOIN games g ON g.game_id=pg.game_id
        WHERE g.season=? GROUP BY pg.team_id, pg.player_id"""
        prev = pd.read_sql_query(q, conn, params=(s - 1,))
        ros = pd.read_sql_query("SELECT player_id, team_id FROM rosters WHERE season=?", conn, params=(s,))
        prev["returns"] = prev.apply(lambda r: ((ros.player_id == r.player_id) & (ros.team_id == r.team_id)).any(), axis=1)
        prev["prod"] = prev.yds + 15 * prev.tkl
        rp = prev.groupby("team_id").apply(lambda d: d.loc[d.returns, "prod"].sum() / max(d["prod"].sum(), 1),
                                           include_groups=False)
        for tid, v in rp.items():
            conn.execute("UPDATE team_seasons SET returning_production=? WHERE team_id=? AND season=?",
                         (float(v), int(tid), s))
