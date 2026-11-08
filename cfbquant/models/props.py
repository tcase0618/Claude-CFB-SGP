"""Market-anchored, uncertainty-aware simulation for player props and same-game parlays.

Two changes to the base simulator, both validated walk-forward (scripts/validate_sgp.py):

1. **Market anchor.** The full-game backtest shows the sportsbook closing line beats every model on mean margin and
   total (log loss 0.435 vs 0.50+). So for prop/SGP work the simulator does not try to out-guess the line: each
   offence gets a per-game efficiency shift (the same `shift_level` mapping used for league calibration) so that the
   simulated mean home/away points match the line-implied team totals. The simulator then contributes what the line
   does not: the *joint* distribution of game script, team stats and every player's box score.

2. **Parameter uncertainty.** Point-estimated parameters make every lane the "average" version of each team, which
   under-disperses player outcomes (80% intervals covered ~47-68% in the base backtest). Each lane draws its own
   offence/defence efficiency offset, pace multiplier and usage shares (Dirichlet around the estimated shares), so
   the simulated distribution reflects both game-day variance and what we do not know about the teams.

All arrays are lane-major, as produced by sim.params.stack_params: scalars (L, 2), vectors (L, 2, k).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from ..sim.engine import simulate
from ..sim.params import League, logit, sigmoid

# offence-side mapping of one "level" unit (matches team_params.shift_level)
_OFF_LOGIT = {"comp_off": 0.35, "rexp_off": 0.25, "sack_off": -0.2, "int_off": -0.2}
_OFF_ADD = {"rcore_off": 0.5, "yac_off": 0.5}
# defence-allowed mapping: +1 = a defence that allows more
_DEF_LOGIT = {"comp_def": 0.35, "rexp_def": 0.25, "sack_def": -0.2, "int_def": -0.2}
_DEF_ADD = {"rcore_def": 0.5, "yac_def": 0.5}


@dataclass(frozen=True)
class Uncertainty:
    team_sd: float = float(os.environ.get("CFBQ_UNC_TEAM", "0.25"))     # level units, per unit per lane
    env_sd: float = float(os.environ.get("CFBQ_UNC_ENV", "0.45"))       # shared game-environment shock (both offences)
    pace_sd: float = float(os.environ.get("CFBQ_UNC_PACE", "0.07"))     # log-scale pace multiplier
    usage_conc: float = float(os.environ.get("CFBQ_UNC_USAGE", "30"))   # Dirichlet concentration (lower = more spread)

    @classmethod
    def none(cls) -> "Uncertainty":
        return cls(0.0, 0.0, 0.0, 0.0)


def materialize(P: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Writable copies (broadcast views from broadcast_params are read-only)."""
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in P.items()}


def _shift(P, side: int, l: np.ndarray, logit_map: dict, add_map: dict) -> None:
    for f, c in logit_map.items():
        P[f][:, side] = sigmoid(logit(P[f][:, side]) + c * l)
    for f, c in add_map.items():
        P[f][:, side] = P[f][:, side] + c * l


def shift_offense(P, side: int, l: np.ndarray) -> None:
    _shift(P, side, np.asarray(l, float), _OFF_LOGIT, _OFF_ADD)


def shift_defense(P, side: int, l: np.ndarray) -> None:
    _shift(P, side, np.asarray(l, float), _DEF_LOGIT, _DEF_ADD)


def perturb(P: dict[str, np.ndarray], rng: np.random.Generator, unc: Uncertainty) -> None:
    """In-place per-lane draws of team strength, pace and usage around the point estimates."""
    L = P["pace"].shape[0]
    if unc.team_sd > 0:
        for side in (0, 1):
            shift_offense(P, side, rng.normal(0, unc.team_sd, L))
            shift_defense(P, side, rng.normal(0, unc.team_sd, L))
    if unc.env_sd > 0:
        # weather, officiating, tempo: moves both offences together, so it widens the total but not the margin
        # (real residual SD around the closing total is ~15.6 vs ~13.3 simulated without it)
        env = rng.normal(0, unc.env_sd, L)
        for side in (0, 1):
            shift_offense(P, side, env)
    if unc.pace_sd > 0:
        common = rng.normal(0, unc.pace_sd * 0.7, (L, 1))
        P["pace"] *= np.exp(common + rng.normal(0, unc.pace_sd * 0.7, P["pace"].shape))
    if unc.usage_conc > 0:
        for k in ("target_share", "rush_share"):
            a = np.maximum(P[k] * unc.usage_conc, 1e-3)
            g = rng.gamma(a)
            P[k] = g / g.sum(axis=2, keepdims=True)


def line_targets(spread_home: float, total: float) -> tuple[float, float]:
    """ESPN/DraftKings convention: spread_home < 0 means the home team is favoured."""
    m = -float(spread_home)
    return (total + m) / 2.0, (total - m) / 2.0


def anchored_simulate(P: dict[str, np.ndarray], game_of_lane: np.ndarray, targets: np.ndarray, league: League,
                      seed: int, *, neutral: np.ndarray, unc: Uncertainty = Uncertainty(), n_anchor: int | None = None,
                      track_players: bool = True, iters: int = 3):
    """Simulate with each game's mean home/away points pinned to `targets` (G, 2); NaN rows are left unanchored.

    Each offence's level shift is solved per game by secant steps on a smaller anchoring run that uses the same
    perturbation draws (so the anchor holds for the perturbed distribution). Returns (SimResult, shifts (G, 2)).
    """
    G = targets.shape[0]
    P = materialize(P)
    perturb(P, np.random.default_rng(seed + 991), unc)
    # anchoring subsample: the first n_anchor lanes of each game
    lanes_by_game = [np.flatnonzero(game_of_lane == g) for g in range(G)]
    na = n_anchor or min(400, min(len(x) for x in lanes_by_game))
    sub = np.concatenate([x[:na] for x in lanes_by_game])
    gsub = game_of_lane[sub]
    has = ~np.isnan(targets).any(axis=1)
    shifts = np.zeros((G, 2))

    def run(sh):
        Q = {k: v[sub].copy() for k, v in P.items()}
        for side in (0, 1):
            shift_offense(Q, side, sh[gsub, side])
        r = simulate(Q, league, seed + 17, neutral=neutral[sub], track_players=False)
        pts = np.stack([r.home_pts, r.away_pts], axis=1).astype(float)
        return np.stack([np.bincount(gsub, pts[:, s], G) / np.bincount(gsub, minlength=G) for s in (0, 1)], axis=1)

    if has.any():
        base = run(shifts)
        # opposite-sign probe: own-offence slope plus the (negative) cross effect both enlarge |dpts|, so the pooled
        # slope errs high and the iteration converges monotonically instead of oscillating
        probe = run(shifts + np.array([0.5, -0.5]))
        slope = np.abs(probe - base) / 0.5
        pooled = float(np.clip(np.median(slope[has]), 4.0, 14.0))
        cur = base
        for _ in range(iters):
            step = np.where(has[:, None], (targets - cur) / pooled, 0.0)
            shifts = np.clip(shifts + step, -3.0, 3.0)
            cur = run(shifts)
            if os.environ.get("CFBQ_DEBUG_ANCHOR"):
                print("anchor iter resid", np.nanmean(np.abs(cur - targets)[has], axis=0), "slope", round(pooled, 2))
        anchored_simulate.last_residual = np.where(has[:, None], cur - targets, np.nan)
    for side in (0, 1):
        shift_offense(P, side, shifts[game_of_lane, side])
    res = simulate(P, league, seed, neutral=neutral, track_players=track_players)
    return res, shifts
