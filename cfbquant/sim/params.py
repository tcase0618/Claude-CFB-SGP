"""Simulation parameters.

Every parameter maps to a directly observable play-level rate so it can be
estimated from data (see cfbquant.analytics.team_params). Offense parameters
are what a unit produces; defense parameters are what a unit *allows*.
Matchups combine them with an odds-ratio (log5) rule for probabilities and an
additive rule for continuous quantities, both centred on the league average.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np

# ---- player slot layout (per team side) -------------------------------------
OFF_SLOTS = ["QB1", "QB2", "RB1", "RB2", "WR1", "WR2", "WR3", "WR4", "TE1", "TE2", "K", "P"]
DEF_SLOTS = ["DL1", "DL2", "DL3", "DL4", "LB1", "LB2", "LB3", "CB1", "CB2", "S1", "S2", "NB"]
SLOTS = OFF_SLOTS + DEF_SLOTS
SLOT = {s: i for i, s in enumerate(SLOTS)}
N_SLOTS = len(SLOTS)

TARGET_SLOTS = [SLOT[s] for s in ["RB1", "RB2", "WR1", "WR2", "WR3", "WR4", "TE1", "TE2"]]
RUSH_SLOTS = [SLOT[s] for s in ["QB1", "RB1", "RB2", "WR1"]]
DEF_IDX = [SLOT[s] for s in DEF_SLOTS]

# ---- player stat layout -----------------------------------------------------
STATS = ["pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "sacks_taken",
         "rush_att", "rush_yds", "rush_td", "targets", "receptions", "rec_yds", "rec_td",
         "tackles", "tfl", "def_sacks", "def_int", "pbu", "forced_fumbles",
         "fga", "fgm", "xpa", "xpm", "punts", "punt_yds"]
STAT = {s: i for i, s in enumerate(STATS)}
N_STATS = len(STATS)

TEAM_STATS = ["plays", "dropbacks", "pass_att", "pass_cmp", "pass_yds", "rush_att", "rush_yds",
              "sacks", "ints", "fumbles_lost", "turnovers", "first_downs", "explosive",
              "drives", "td", "fga", "fgm", "punts", "top_seconds", "pass_td", "rush_td",
              "fourth_att", "fourth_conv", "rz_trips", "rz_td"]
TSTAT = {s: i for i, s in enumerate(TEAM_STATS)}


@dataclass(frozen=True)
class League:
    """League-average environment (FBS-like defaults; re-estimated from data)."""
    pass_rate: float = 0.50       # dropbacks / (dropbacks + designed rushes), neutral
    sack: float = 0.065           # sacks / dropbacks
    scramble: float = 0.045       # scrambles / dropbacks
    intr: float = 0.024           # interceptions / attempts
    comp: float = 0.64            # completions / attempts at league-average depth
    adot: float = 8.5             # air yards / attempt
    yac: float = 6.0              # yards after catch / completion
    rcore: float = 3.3            # mean of rushes gaining < 10 yards
    rexp: float = 0.10            # P(rush >= 10 yards)
    fumble: float = 0.008         # fumbles lost / touch
    pace: float = 31.0            # seconds run off per clock-running play
    punt_net: float = 39.0
    fg_skill: float = 0.0         # logit offset on FG make probability
    pen_rate: float = 0.055       # accepted penalties / scrimmage play
    two_pt: float = 0.45
    xp: float = 0.97
    hfa: float = 0.35             # home-field scale (0.35 ~= 2.5 points)


@dataclass(frozen=True)
class TeamParams:
    """Unit-level parameters for one team (offense produces, defense allows)."""
    # offense
    pass_rate: float = 0.50
    sack_off: float = 0.065
    scramble: float = 0.045
    int_off: float = 0.024
    comp_off: float = 0.64
    adot: float = 8.5
    yac_off: float = 6.0
    rcore_off: float = 3.3
    rexp_off: float = 0.10
    fumble_off: float = 0.008
    pace: float = 31.0
    go_aggr: float = 0.0
    # defense (allowed)
    sack_def: float = 0.065
    int_def: float = 0.024
    comp_def: float = 0.64
    yac_def: float = 6.0
    rcore_def: float = 3.3
    rexp_def: float = 0.10
    fumble_def: float = 0.008
    # special teams
    fg_skill: float = 0.0
    punt_net: float = 39.0
    # usage (shares are normalised inside the engine)
    target_share: tuple = (0.07, 0.03, 0.24, 0.20, 0.15, 0.06, 0.18, 0.07)
    rush_share: tuple = (0.18, 0.55, 0.22, 0.05)      # QB1(designed), RB1, RB2, WR1
    catch_adj: tuple = (0.0,) * 8                     # logit adj on completion per target slot
    rush_adj: tuple = (0.0,) * 4                      # yards shift on rcore per rusher slot
    tackle_run: tuple = (0.07, 0.07, 0.06, 0.06, 0.16, 0.14, 0.10, 0.07, 0.07, 0.09, 0.07, 0.04)
    tackle_pass: tuple = (0.02, 0.02, 0.02, 0.02, 0.10, 0.09, 0.07, 0.15, 0.15, 0.13, 0.12, 0.11)
    sack_share: tuple = (0.24, 0.18, 0.12, 0.22, 0.10, 0.06, 0.04, 0.0, 0.0, 0.02, 0.01, 0.01)
    int_share: tuple = (0.0, 0.0, 0.0, 0.0, 0.07, 0.06, 0.03, 0.23, 0.22, 0.19, 0.12, 0.08)

    def with_(self, **kw) -> "TeamParams":
        return replace(self, **kw)


SCALAR_FIELDS = [f.name for f in fields(TeamParams) if not isinstance(f.default, tuple)]
VECTOR_FIELDS = [f.name for f in fields(TeamParams) if isinstance(f.default, tuple)]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def stack_params(pairs: list[tuple[TeamParams, TeamParams]]) -> dict[str, np.ndarray]:
    """Stack (home, away) parameter pairs into lane arrays: scalars (L,2), vectors (L,2,k)."""
    out: dict[str, np.ndarray] = {}
    for name in SCALAR_FIELDS:
        out[name] = np.array([[getattr(h, name), getattr(a, name)] for h, a in pairs], dtype=np.float64)
    for name in VECTOR_FIELDS:
        arr = np.array([[getattr(h, name), getattr(a, name)] for h, a in pairs], dtype=np.float64)
        if name.endswith("share") or name.startswith("tackle"):
            s = arr.sum(axis=2, keepdims=True)
            arr = np.where(s > 0, arr / np.where(s > 0, s, 1), 1.0 / arr.shape[2])
        out[name] = arr
    return out


def broadcast_params(home: TeamParams, away: TeamParams, n: int) -> dict[str, np.ndarray]:
    """Same matchup across n lanes, as read-only broadcast views (no copy)."""
    base = stack_params([(home, away)])
    return {k: np.broadcast_to(v, (n,) + v.shape[1:]) for k, v in base.items()}
