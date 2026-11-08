"""Same-game parlay pricing and ticket construction from one consistent simulation.

A leg is a predicate on a simulated game: every lane either hits it or not, so a parlay's hit probability is the
share of lanes hitting *all* legs. That gives correlation for free and exactly: QB passing yards, his WR1's
receiving yards, the team total and the game total all come from the same simulated plays.

Key quantities per ticket (all classification `predicted`):
  p_joint   share of lanes where every leg hits (the price we believe)
  p_indep   product of the legs' marginal probabilities (what a no-correlation book would charge)
  lift      p_joint / p_indep: how much the legs move together. Books apply a correlation haircut to SGPs; when it
            is smaller than the true lift the ticket is under-priced. We cannot see the book's haircut from free
            data, so the output is a *fair price*: bet only when the book offers more than `fair_american`.

Selection bias: picking the highest-lift tickets out of thousands of combinations on the same lanes overstates the
lift (winner's curse). The builder searches on one half of the lanes and reports prices from the other half.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from ..sim.params import SLOT, STAT

# composite player stats (books list these as one market)
COMPOSITE = {"td": ("rush_td", "rec_td"), "rush_rec_yds": ("rush_yds", "rec_yds"),
             "pass_rush_yds": ("pass_yds", "rush_yds")}
STAT_LABEL = {"pass_yds": "Pass Yds", "pass_td": "Pass TD", "pass_cmp": "Completions", "pass_att": "Pass Att",
              "pass_int": "INT", "rush_yds": "Rush Yds", "rush_att": "Rush Att", "rush_td": "Rush TD",
              "receptions": "Receptions", "rec_yds": "Rec Yds", "rec_td": "Rec TD", "td": "Anytime TD",
              "rush_rec_yds": "Rush+Rec Yds", "pass_rush_yds": "Pass+Rush Yds", "tackles": "Tackles",
              "fgm": "FG Made"}

# candidate player markets by slot (what books actually list for CFB)
SLOT_MARKETS = {
    "QB1": ["pass_yds", "pass_td", "pass_cmp", "rush_yds", "pass_rush_yds"],
    "RB1": ["rush_yds", "rush_rec_yds", "td", "receptions"],
    "RB2": ["rush_yds", "td"],
    "WR1": ["rec_yds", "receptions", "td"],
    "WR2": ["rec_yds", "receptions", "td"],
    "WR3": ["rec_yds", "receptions"],
    "TE1": ["rec_yds", "receptions", "td"],
}


# ----------------------------------------------------------------------------- odds helpers
def prob_to_american(p: float) -> float | None:
    if not 0 < p < 1:
        return None
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def american_to_decimal(a: float) -> float:
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def american_to_prob(a: float) -> float:
    return 1 / american_to_decimal(a)


def fmt_american(a: float | None) -> str:
    if a is None or not math.isfinite(a):
        return "—"
    return f"+{a:.0f}" if a > 0 else f"{a:.0f}"


# ----------------------------------------------------------------------------- volume calibration
COUNT_STATS = {"pass_cmp", "pass_att", "pass_td", "pass_int", "rush_att", "rush_td", "targets", "receptions", "rec_td"}
CALIB_STATS = ["pass_yds", "pass_cmp", "pass_td", "rush_yds", "rush_td", "receptions", "rec_yds", "rec_td"]


def apply_factor(x: np.ndarray, f: float, count: bool, rng: np.random.Generator) -> np.ndarray:
    """Rescale a simulated stat by f while keeping its type: yards scale, counts are binomially thinned (f < 1)
    or topped up with Poisson extras (f > 1), so a count stays an integer and P(X >= k) moves smoothly."""
    if not count:
        return x * f
    xi = np.maximum(x, 0).astype(np.int64)
    if f < 1:
        return rng.binomial(xi, f).astype(float)
    return (xi + rng.poisson(xi * (f - 1))).astype(float)


def calibration_factors(acc: dict, k: float = 150.0, lo: float = 0.75, hi: float = 1.25) -> dict:
    """acc: stat -> [sum_actual, sum_sim_mean, n] over projected starters who appeared, from *earlier* games only.
    The raw ratio is shrunk toward 1 with pseudo-count k."""
    out = {}
    for st, (sa, ss, n) in acc.items():
        if ss <= 0 or n == 0:
            continue
        raw = sa / ss
        out[st] = float(np.clip(1 + (raw - 1) * n / (n + k), lo, hi))
    return out


# ----------------------------------------------------------------------------- legs
@dataclass(frozen=True)
class Leg:
    """kind: ml | spread | total | team_total | player.  side: home|away.  op: over|under.

    spread: `line` is the side's handicap (-6.5 = side must win by 7+).  player legs carry player_id + stat; the
    player's (side, slot) is resolved from the simulation's slot map."""
    kind: str
    side: str | None = None
    op: str | None = None
    line: float | None = None
    player_id: int | None = None
    stat: str | None = None
    name: str | None = None
    book_american: float | None = None

    def key(self) -> tuple:
        """Two legs with the same key are the same market (can't both be on a ticket)."""
        if self.kind == "player":
            return ("player", self.player_id, self.stat)
        if self.kind in ("ml", "spread"):
            return ("side",)               # ML + spread on one ticket is near-redundant; books block most pairs
        if self.kind == "team_total":
            return ("team_total", self.side)
        return (self.kind,)

    def label(self) -> str:
        if self.kind == "ml":
            return f"{self.name or self.side} ML"
        if self.kind == "spread":
            return f"{self.name or self.side} {self.line:+g}"
        if self.kind == "total":
            return f"Game {self.op.title()} {self.line:g}"
        if self.kind == "team_total":
            return f"{self.name or self.side} {self.op.title()} {self.line:g} pts"
        return f"{self.name or self.player_id} {self.op.title()} {self.line:g} {STAT_LABEL.get(self.stat, self.stat)}"

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("kind", "side", "op", "line", "player_id", "stat", "name", "book_american")}
        d["label"] = self.label()
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Leg":
        return cls(**{k: d.get(k) for k in ("kind", "side", "op", "line", "player_id", "stat", "name", "book_american")})


class GameSims:
    """The lanes of one simulated game, with lookups for leg evaluation."""

    def __init__(self, home_pts, away_pts, player_stats, slots: tuple[dict, dict], team_names=("home", "away"),
                 player_names: dict | None = None, calib: dict | None = None, seed: int = 0):
        self.h = np.asarray(home_pts, float)
        self.a = np.asarray(away_pts, float)
        self.ps = player_stats                     # (n, 2, N_SLOTS, N_STATS)
        self.n = len(self.h)
        self.slots = slots
        self.team_names = team_names
        self.player_names = player_names or {}
        self.calib = calib or {}                   # base stat -> walk-forward volume factor (see calibrate_stats)
        self._rng = np.random.default_rng(seed)
        self._cache: dict = {}
        self.where = {}                            # player_id -> (side, slot)
        for side in (0, 1):
            for s, pid in slots[side].items():
                if pid is not None and pid not in self.where:
                    self.where[int(pid)] = (side, s)

    def base_stat(self, pid: int, st: str, raw: bool = False) -> np.ndarray:
        side, slot = self.where[int(pid)]
        x = self.ps[:, side, SLOT[slot], STAT[st]].astype(float)
        f = self.calib.get(st, 1.0)
        if raw or f == 1.0:
            return x
        key = (int(pid), st)
        if key not in self._cache:                 # one draw per (player, stat) so every leg sees the same lanes
            self._cache[key] = apply_factor(x, f, st in COUNT_STATS, self._rng)
        return self._cache[key]

    def stat(self, pid: int, stat: str, raw: bool = False) -> np.ndarray:
        return sum(self.base_stat(pid, p, raw) for p in COMPOSITE.get(stat, (stat,)))

    def values(self, leg: Leg) -> np.ndarray:
        """Continuous quantity the leg is about (side-oriented)."""
        sgn = 1.0 if leg.side == "home" else -1.0
        if leg.kind in ("ml", "spread"):
            return sgn * (self.h - self.a)
        if leg.kind == "total":
            return self.h + self.a
        if leg.kind == "team_total":
            return self.h if leg.side == "home" else self.a
        return self.stat(leg.player_id, leg.stat)

    def hits(self, leg: Leg) -> np.ndarray:
        v = self.values(leg)
        if leg.kind == "ml":
            return v > 0
        if leg.kind == "spread":
            return v + leg.line > 0
        return v > leg.line if leg.op == "over" else v < leg.line


# ----------------------------------------------------------------------------- pricing
def price(sims: GameSims, legs: list[Leg], book_american: float | None = None, lanes: np.ndarray | None = None) -> dict:
    X = np.stack([sims.hits(l) for l in legs], axis=1)
    if lanes is not None:
        X = X[lanes]
    n = X.shape[0]
    marg = X.mean(axis=0)
    pj = float(X.all(axis=1).mean())
    pi = float(np.prod(marg))
    se = math.sqrt(max(pj * (1 - pj), 1e-9) / n)
    out = {"legs": [dict(l.to_dict(), p=float(m), fair_american=prob_to_american(float(m))) for l, m in zip(legs, marg)],
           "n_legs": len(legs), "p_joint": pj, "p_joint_ci90": [max(0.0, pj - 1.645 * se), min(1.0, pj + 1.645 * se)],
           "p_indep": pi, "lift": pj / pi if pi > 0 else None, "fair_american": prob_to_american(pj),
           "fair_decimal": 1 / pj if pj > 0 else None, "indep_american": prob_to_american(pi), "n_sims": int(n)}
    if book_american is not None:
        dec = american_to_decimal(book_american)
        ev = pj * dec - 1
        out.update(book_american=float(book_american), book_implied=1 / dec, ev_per_unit=ev,
                   kelly=max(0.0, ev / (dec - 1)), quarter_kelly=max(0.0, ev / (dec - 1)) / 4)
    return out


def _round_half(x: float) -> float:
    return math.floor(x) + 0.5


def _line_for_prob(v: np.ndarray, p_over: float) -> float:
    """x.5 line whose P(over) is closest to p_over."""
    q = float(np.quantile(v, 1 - p_over))
    cands = {_round_half(q - 1), _round_half(q), _round_half(q + 1)}
    cands = {c for c in cands if c > 0}
    return min(cands, key=lambda c: abs((v > c).mean() - p_over)) if cands else 0.5


def candidate_legs(sims: GameSims, market: dict | None = None, ladder=(0.5, 0.65, 0.78), min_mean: dict | None = None) -> list[Leg]:
    """Generate the leg menu: game lines at the market number, and player/team-total legs on an alt-line ladder.

    `ladder` holds target over-probabilities: 0.5 ~ the main prop line a book would post (the median), higher values
    are alt lines. Unders are added at the main line only. Players with negligible volume are skipped."""
    min_mean = min_mean or {"pass_yds": 120, "rush_yds": 25, "rec_yds": 20, "receptions": 2.0, "pass_cmp": 10,
                            "rush_rec_yds": 30, "pass_rush_yds": 150, "pass_td": 0.6, "td": 0.15}
    legs: list[Leg] = []
    hn, an = sims.team_names
    nm = {"home": hn, "away": an}
    for side in ("home", "away"):
        legs.append(Leg("ml", side=side, name=nm[side]))
    if market and market.get("spread_home") is not None:
        sh = float(market["spread_home"])
        legs += [Leg("spread", side="home", line=sh, name=hn), Leg("spread", side="away", line=-sh, name=an)]
    if market and market.get("total") is not None:
        t = float(market["total"])
        legs += [Leg("total", op="over", line=t), Leg("total", op="under", line=t)]
    else:
        v = sims.h + sims.a
        t = _line_for_prob(v, 0.5)
        legs += [Leg("total", op="over", line=t), Leg("total", op="under", line=t)]
    for side in ("home", "away"):
        v = sims.h if side == "home" else sims.a
        for p in ladder:
            legs.append(Leg("team_total", side=side, op="over", line=_line_for_prob(v, p), name=nm[side]))
        legs.append(Leg("team_total", side=side, op="under", line=_line_for_prob(v, 0.5), name=nm[side]))
    for pid, (side, slot) in sims.where.items():
        for st in SLOT_MARKETS.get(slot, []):
            v = sims.stat(pid, st)
            if v.mean() < min_mean.get(st, 0):
                continue
            name = sims.player_names.get(pid) or f"{nm['home' if side == 0 else 'away']} {slot}"
            if st in ("td", "pass_td"):            # count markets: books list fixed thresholds
                for thr in ((0.5,) if st == "td" else (0.5, 1.5, 2.5)):
                    if 0.15 <= (v > thr).mean() <= 0.92:
                        legs.append(Leg("player", op="over", line=thr, player_id=pid, stat=st, name=name))
                continue
            seen = set()
            floor = MIN_LINE.get(st, 0.5)
            for p in ladder:
                ln = _line_for_prob(v, p)
                if ln < floor:                     # no "over 0.5 yards" style lines: books don't list them
                    continue
                if ln not in seen:
                    seen.add(ln)
                    legs.append(Leg("player", op="over", line=ln, player_id=pid, stat=st, name=name))
            mid = _line_for_prob(v, 0.5)
            if mid >= floor:
                legs.append(Leg("player", op="under", line=mid, player_id=pid, stat=st, name=name))
    return legs


# smallest line a book realistically lists for each market (alt "X+" ladders start around here)
MIN_LINE = {"pass_yds": 149.5, "pass_cmp": 11.5, "rush_yds": 19.5, "rec_yds": 19.5, "receptions": 1.5,
            "rush_rec_yds": 24.5, "pass_rush_yds": 174.5}

# leg-probability band and per-player cap per ticket size: short tickets can carry main lines, long tickets need
# alt lines or the hit rate collapses (0.5^8 = 0.4%)
TIERS = {3: (0.50, 0.85), 5: (0.58, 0.88), 8: (0.66, 0.92)}


def build_tickets(sims: GameSims, legs: list[Leg], size: int, *, top: int = 2, beam: int = 40, seed: int = 0,
                  band: tuple[float, float] | None = None, min_joint: float | None = None) -> list[dict]:
    """Beam search for the most positively correlated `size`-leg ticket inside the tier's leg-probability band.

    Objective: log lift (joint / independent), i.e. the part of the price a correlation-naive book gets wrong.
    Searched on half A of the lanes, priced on half B."""
    lo, hi = band or TIERS.get(size, (0.5, 0.9))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sims.n)
    A, B = perm[: sims.n // 2], perm[sims.n // 2:]
    X = np.stack([sims.hits(l) for l in legs], axis=1)
    pa = X[A].mean(axis=0)
    ok = np.flatnonzero((pa >= lo) & (pa <= hi))
    if len(ok) < size:
        return []
    XA = X[A][:, ok]
    keys = [legs[i].key() for i in ok]
    pid_of = [legs[i].player_id for i in ok]
    logm = np.log(XA.mean(axis=0))
    min_joint = min_joint if min_joint is not None else {3: 0.12, 5: 0.06, 8: 0.025}.get(size, 0.01)

    def ok_add(state, j):
        if keys[j] in {keys[i] for i in state}:
            return False
        if pid_of[j] is not None and any(pid_of[i] == pid_of[j] for i in state):
            return False                           # one leg per player: same-player stacks are blocked or
        return True                                # heavily haircut by books, and their "lift" is mostly tautology

    # each beam state: (tuple of local idx, joint hit mask on A)
    states = [((), np.ones(len(A), bool))]
    for _ in range(size):
        nxt = []
        for st, mask in states:
            for j in range(len(ok)):
                if st and j <= max(st) and False:
                    continue
                if j in st or not ok_add(st, j):
                    continue
                m2 = mask & XA[:, j]
                pj = m2.mean()
                if pj <= 0:
                    continue
                st2 = tuple(sorted(st + (j,)))
                score = math.log(pj) - sum(logm[i] for i in st2)
                nxt.append((score, st2, m2, pj))
        # dedupe identical sets, keep best by score
        uniq = {}
        for sc, st2, m2, pj in nxt:
            if st2 not in uniq or sc > uniq[st2][0]:
                uniq[st2] = (sc, st2, m2, pj)
        cand = sorted(uniq.values(), key=lambda x: -x[0])
        floor = min_joint ** (len(cand[0][1]) / size) if cand else 0      # keep partial tickets on track
        cand = [c for c in cand if c[3] >= floor] or cand
        states = [(st2, m2) for _, st2, m2, _ in cand[:beam]]
        if not states:
            return []
    finals = []
    for st, mask in states:
        if len(st) == size and mask.mean() >= min_joint:
            finals.append(st)
    out, used = [], []
    for st in finals:
        s = set(st)
        if any(len(s & u) / size > 0.5 for u in used):
            continue                               # keep returned tickets meaningfully different
        used.append(s)
        tl = [legs[ok[i]] for i in st]
        t = price(sims, tl, lanes=B)
        t["search_lift"] = float(np.exp(math.log(max(mask_mean(XA, st), 1e-12)) - sum(logm[i] for i in st)))
        out.append(t)
        if len(out) >= top:
            break
    return out


def mask_mean(XA, st):
    return float(XA[:, list(st)].all(axis=1).mean())


def build_card(sims: GameSims, market: dict | None = None, sizes=(3, 5, 8), top: int = 2, seed: int = 0) -> dict:
    legs = candidate_legs(sims, market)
    card = {"n_sims": sims.n, "n_candidate_legs": len(legs), "tickets": {}}
    for k in sizes:
        card["tickets"][str(k)] = build_tickets(sims, legs, k, top=top, seed=seed + k)
    return card
