"""Vectorised play-by-play college football simulator.

Design
------
* Each *lane* is an independent game. All lanes advance in lockstep, one snap
  per iteration, with numpy masks selecting the lanes each branch applies to.
  For a forecast every lane shares the same parameters (10k sims ~1s); for
  synthetic league generation each lane is a different game.
* Player box scores are accumulated *from the same play events* that move the
  score, so football accounting identities hold by construction:
  completions <= attempts, team passing yards == sum of receiving yards,
  each passing TD has exactly one receiving TD, team rushing == sum of rushers.
* Seeded `numpy.random.Generator` => fully reproducible.

Modelled: kickoffs, touchbacks, returns, down & distance, run/pass/scramble/sack,
air yards + YAC, completions, interceptions (with returns / pick-sixes),
fumbles, penalties (simplified), punts, field goals (distance-dependent),
PAT/two-point decisions, 4th-down decision policy, clock with pace, hurry-up,
clock-killing, kneel-downs, timeouts (trailing defence late), halftime, and
the 2021+ NCAA overtime format (25-yard line possessions, mandatory 2-pt from
OT2, alternating 2-pt attempts from OT3).

Not modelled (documented limitation): formation/personnel, play-level defensive
coverage, onside kicks, fake punts/FGs, blocked kicks, kick-return player stats.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import (DEF_IDX, N_SLOTS, N_STATS, RUSH_SLOTS, SLOT, STAT, TARGET_SLOTS,
                     TEAM_STATS, TSTAT, League, logit, sigmoid)

KICK, SCRIM, DONE = 0, 1, 2
PT = {"pass": 0, "rush": 1, "sack": 2, "scramble": 3, "punt": 4, "field_goal": 5,
      "kickoff": 6, "kneel": 7, "penalty": 8, "two_point": 9}
PT_NAMES = {v: k for k, v in PT.items()}

# Log columns recorded per snap when record=True (synthetic data generation)
LOG_COLS = ["lane", "step", "offense", "period", "clock", "down", "distance", "ytg",
            "off_score", "def_score", "ptype", "yards", "air", "completion", "interception",
            "sack", "fumble_lost", "touchdown", "first_down", "points", "def_points",
            "penalty_yards", "fg_dist", "fg_made", "passer", "rusher", "target", "tackler",
            "sacker", "interceptor", "pbu", "forced_fumble", "kicker", "punter", "drive",
            "pat_result", "scramble"]


@dataclass
class SimResult:
    home_pts: np.ndarray
    away_pts: np.ndarray
    ot: np.ndarray
    team_stats: np.ndarray          # (L, 2, n_team_stats)
    player_stats: np.ndarray | None  # (L, 2, N_SLOTS, N_STATS)
    log: dict | None


def _cat(rng, cum: np.ndarray) -> np.ndarray:
    """Vectorised categorical draw given cumulative probabilities (m, k)."""
    u = rng.random(cum.shape[0])[:, None]
    return np.minimum((u > cum).sum(axis=1), cum.shape[1] - 1)


def simulate(P: dict[str, np.ndarray], league: League, seed: int, *,
             neutral: np.ndarray | None = None, record: bool = False,
             track_players: bool = True, max_steps: int = 420) -> SimResult:
    """Run len(lanes) games. P comes from params.stack_params / broadcast_params."""
    rng = np.random.default_rng(seed)
    L = P["pace"].shape[0]
    ar = np.arange(L)
    neutral = np.zeros(L, bool) if neutral is None else np.asarray(neutral, bool)
    # home-field: +0.5 unit to home offence, -0.5 to away offence (0 when neutral)
    hfa_side = np.where(neutral[:, None], 0.0, np.array([[0.5, -0.5]])) * league.hfa

    # cumulative usage tables
    cum = {k: np.cumsum(P[k], axis=2) for k in
           ("target_share", "rush_share", "tackle_run", "tackle_pass", "sack_share", "int_share")}

    # --- state ---
    first_recv = (rng.random(L) < 0.5).astype(np.int64)
    poss = 1 - first_recv                     # in KICK phase, poss == kicking side
    phase = np.full(L, KICK)
    period = np.ones(L, np.int64)
    clock = np.full(L, 900.0)
    down = np.ones(L, np.int64)
    dist = np.full(L, 10, np.int64)
    ytg = np.full(L, 75, np.int64)
    score = np.zeros((L, 2), np.int64)
    tos = np.full((L, 2), 3, np.int64)
    drive_no = np.zeros(L, np.int64)
    in_rz = np.zeros(L, bool)
    # overtime
    ot = np.zeros(L, bool)
    ot_round = np.zeros(L, np.int64)
    ot_first = np.zeros(L, np.int64)
    ot_count = np.zeros(L, np.int64)
    two_try = np.zeros(L, bool)

    tstats = np.zeros((L, 2, len(TEAM_STATS)), np.int64)
    pstats = np.zeros((L, 2, N_SLOTS, N_STATS), np.int32) if track_players else None
    logs: list[dict] = [] if record else None

    def padd(lanes, side, slot, stat, val=1):
        if pstats is None or len(lanes) == 0:
            return
        np.add.at(pstats, (lanes, side, slot, STAT[stat]), val)

    def tadd(lanes, side, stat, val=1):
        if len(lanes):
            np.add.at(tstats, (lanes, side, TSTAT[stat]), val)

    def ot_setup(lanes):
        r = ot_round[lanes]
        shoot = r >= 3
        two_try[lanes] = shoot
        ytg[lanes] = np.where(shoot, 3, 25)
        down[lanes] = 1
        dist[lanes] = np.where(shoot, 3, 10)
        phase[lanes] = SCRIM
        period[lanes] = 4 + r

    def ot_advance(lanes):
        """Called when an OT possession ends."""
        if len(lanes) == 0:
            return
        first = ot_count[lanes] == 0
        l1 = lanes[first]
        ot_count[l1] = 1
        poss[l1] = 1 - ot_first[l1]
        ot_setup(l1)
        l2 = lanes[~first]
        if len(l2):
            diff = score[l2, 0] != score[l2, 1]
            phase[l2[diff]] = DONE
            cont = l2[~diff]
            ot_round[cont] += 1
            ot_count[cont] = 0
            poss[cont] = ot_first[cont]
            ot_setup(cont)

    def possession_change(lanes, new_ytg):
        """Ball goes to the other team at new_ytg (from their perspective)."""
        if len(lanes) == 0:
            return
        o = lanes[ot[lanes]]
        ot_advance(o)
        n = lanes[~ot[lanes]]
        ny = new_ytg[~ot[lanes]]
        poss[n] = 1 - poss[n]
        in_rz[n] = False
        ytg[n] = np.clip(ny, 1, 99)
        down[n] = 1
        dist[n] = np.minimum(10, ytg[n])
        phase[n] = SCRIM
        drive_no[n] += 1
        tadd(n, poss[n], "drives")

    def score_then_kick(lanes, scorer_side):
        """After a non-OT score the scoring team kicks off."""
        o = lanes[ot[lanes]]
        ot_advance(o)
        n = lanes[~ot[lanes]]
        poss[n] = scorer_side[~ot[lanes]]
        phase[n] = KICK

    for step in range(max_steps):
        live = phase != DONE
        if not live.any():
            break
        # ======================= KICKOFFS =======================
        K = np.flatnonzero(phase == KICK)
        if len(K):
            kicker = poss[K]
            tb = rng.random(len(K)) < 0.47
            ret_line = np.clip(np.round(rng.normal(25, 9, len(K))), 1, 99).astype(np.int64)
            ret_td = rng.random(len(K)) < 0.005
            new_y = np.where(tb, 75, 100 - ret_line)
            # receiving team possession
            poss[K] = 1 - kicker
            ytg[K] = np.where(ret_td, 0, new_y)
            down[K] = 1
            dist[K] = 10
            phase[K] = SCRIM
            clock[K] -= np.where(tb, 0, 6)
            drive_no[K] += 1
            tadd(K, poss[K], "drives")
            if record:
                logs.append(_mk_log(K, step, kicker, period, clock, 0, 0, ytg, score, PT["kickoff"],
                                    yards=np.where(tb, 0, ret_line), drive=drive_no))
            # return TD
            rt = K[ret_td]
            if len(rt):
                score[rt, poss[rt]] += 7
                score_then_kick(rt, poss[rt].copy())

        # ======================= SCRIMMAGE =======================
        S = np.flatnonzero(phase == SCRIM)
        S = S[~np.isin(S, K)] if len(K) else S
        m = len(S)
        if m == 0:
            continue
        off = poss[S]
        de = 1 - off
        q = period[S]
        clk = clock[S]
        diff = score[S, off] - score[S, de]          # offence lead
        late = (q == 4) & (clk < 300) & ~ot[S]
        y0 = ytg[S].copy()
        clock_pre = clock.copy()
        score_pre = score.copy()
        drive_pre = drive_no.copy()
        d0 = down[S].copy()
        di0 = dist[S].copy()

        # ---- play selection ----
        ptype = np.full(m, -1)
        kneel = (~ot[S]) & (q == 4) & (diff > 0) & (clk < 38 * (5 - d0)) & (tos[S, de] == 0) | \
                ((~ot[S]) & (q == 4) & (diff > 0) & (clk < 25 * (5 - d0)))
        ptype[kneel] = PT["kneel"]
        fourth = (d0 == 4) & ~kneel & ~two_try[S]
        if fourth.any():
            f = np.flatnonzero(fourth)
            fg_dist = y0[f] + 17
            in_range = fg_dist <= 55 + 2 * P["fg_skill"][S[f], off[f]]
            gs = (P["go_aggr"][S[f], off[f]] + 1.6 - 0.55 * di0[f] + 0.025 * (60 - y0[f]))
            trailing_late = late[f] & (diff[f] < 0)
            gs = gs + np.where(trailing_late & (diff[f] < -3), 6.0, 0.0)
            gs = gs + np.where(late[f] & (diff[f] < 0) & (diff[f] >= -3) & in_range, -4.0, 0.0)
            in_ot = ot[S[f]]
            # OT: no punts; second possession trailing by >3 must go
            ot_must_go = in_ot & ((~in_range) | ((ot_count[S[f]] == 1) & (diff[f] < -3)))
            go = (rng.random(len(f)) < sigmoid(1.5 * gs)) | ot_must_go
            go = go & ~(in_ot & in_range & (di0[f] > 3) & ~ot_must_go)
            kick_fg = ~go & in_range
            punt = ~go & ~in_range
            ptype[f[kick_fg]] = PT["field_goal"]
            ptype[f[punt]] = PT["punt"]
            tadd(S[f[go]], off[f[go]], "fourth_att")
        open_ = ptype == -1
        if open_.any():
            o = np.flatnonzero(open_)
            lp = logit(P["pass_rate"][S[o], off[o]])
            lp = lp + np.select([d0[o] == 1, (d0[o] >= 2) & (di0[o] >= 7), (d0[o] >= 2) & (di0[o] <= 2)],
                                [-0.15, 0.85, -0.9], 0.0)
            second_half = q[o] >= 3
            lp = lp + np.where(second_half & (diff[o] <= -14), 0.9, 0.0) + np.where(second_half & (diff[o] >= 14), -0.8, 0.0)
            lp = lp + np.where(late[o] & (diff[o] < 0), 1.2, 0.0) + np.where(late[o] & (diff[o] > 0), -1.2, 0.0)
            lp = lp + np.where(y0[o] <= 3, -0.6, 0.0)
            is_pass = rng.random(len(o)) < sigmoid(lp)
            ptype[o[~is_pass]] = PT["rush"]
            po = o[is_pass]
            # dropback: sack / scramble / attempt
            s_p = sigmoid(logit(P["sack_off"][S[po], off[po]]) + logit(P["sack_def"][S[po], de[po]])
                          - logit(league.sack) - 0.25 * hfa_side[S[po], off[po]])
            u = rng.random(len(po))
            sc_p = P["scramble"][S[po], off[po]]
            ptype[po] = np.where(u < s_p, PT["sack"], np.where(u < s_p + sc_p, PT["scramble"], PT["pass"]))
        # penalties replace a small share of non-special snaps
        pen = (rng.random(m) < league.pen_rate) & np.isin(ptype, [PT["pass"], PT["rush"]]) & ~two_try[S]
        ptype[pen] = PT["penalty"]

        # ---- outcome arrays ----
        yards = np.zeros(m, np.int64)
        air = np.full(m, np.nan)
        comp = np.zeros(m, bool)
        intc = np.zeros(m, bool)
        fum = np.zeros(m, bool)
        sackm = ptype == PT["sack"]
        tclock = np.zeros(m)
        stop_clock = np.zeros(m, bool)
        passer = np.full(m, -1); rusher = np.full(m, -1); target = np.full(m, -1)
        tackler = np.full(m, -1); sacker = np.full(m, -1); intcp = np.full(m, -1)
        pbu = np.full(m, -1); ffum = np.full(m, -1)
        pen_y = np.zeros(m, np.int64)
        fgd = np.zeros(m, np.int64); fgm = np.zeros(m, bool)
        punt_net = np.zeros(m, np.int64)
        int_ret_y = np.zeros(m, np.int64)

        pace = P["pace"][S, off]
        pace = np.where(late & (diff < 0), pace * 0.55, np.where(late & (diff > 0), np.minimum(pace * 1.25, 40), pace))

        # -- rushes (designed) and scrambles --
        r = np.flatnonzero((ptype == PT["rush"]) | (ptype == PT["scramble"]))
        if len(r):
            designed = ptype[r] == PT["rush"]
            slot_idx = np.where(designed, _cat(rng, cum["rush_share"][S[r], off[r]]), 0)
            rs = np.array(RUSH_SLOTS)[slot_idx]
            rs = np.where(designed, rs, SLOT["QB1"])
            rusher[r] = rs
            rc = (P["rcore_off"][S[r], off[r]] + P["rcore_def"][S[r], de[r]] - league.rcore
                  + 0.35 * hfa_side[S[r], off[r]]
                  + np.where(designed, P["rush_adj"][S[r], off[r], slot_idx], 1.2))
            rx = sigmoid(logit(P["rexp_off"][S[r], off[r]]) + logit(P["rexp_def"][S[r], de[r]]) - logit(league.rexp)
                         + 0.1 * hfa_side[S[r], off[r]])
            short_yard = y0[r] <= 3
            core = np.minimum(np.round(rng.normal(rc, 3.3)), 9)
            expl = 10 + np.round(rng.exponential(10.0, len(r)))
            yy = np.where(rng.random(len(r)) < rx * np.where(short_yard, 0.2, 1.0), expl, core)
            yards[r] = yy.astype(np.int64)
            fum[r] = rng.random(len(r)) < sigmoid(logit(P["fumble_off"][S[r], off[r]]) + logit(P["fumble_def"][S[r], de[r]]) - logit(league.fumble))
            tclock[r] = pace[r] * rng.uniform(0.85, 1.15, len(r))
        # -- passes --
        pp = np.flatnonzero(ptype == PT["pass"])
        if len(pp):
            passer[pp] = SLOT["QB1"]
            ti = _cat(rng, cum["target_share"][S[pp], off[pp]])
            target[pp] = np.array(TARGET_SLOTS)[ti]
            adot = P["adot"][S[pp], off[pp]]
            a = -3.0 + rng.gamma(1.6, np.maximum(adot + 3.0, 1.0) / 1.6)
            a = np.minimum(np.round(a), y0[pp])           # cannot throw past the end line
            air[pp] = a
            lc = (logit(P["comp_off"][S[pp], off[pp]]) + logit(P["comp_def"][S[pp], de[pp]]) - logit(league.comp)
                  - 0.055 * (a - league.adot) + P["catch_adj"][S[pp], off[pp], ti] + 0.25 * hfa_side[S[pp], off[pp]])
            li = (logit(P["int_off"][S[pp], off[pp]]) + logit(P["int_def"][S[pp], de[pp]]) - logit(league.intr)
                  + 0.02 * (a - league.adot) - 0.2 * hfa_side[S[pp], off[pp]])
            u = rng.random(len(pp))
            pi = sigmoid(li)
            intc[pp] = u < pi
            comp[pp] = (~intc[pp]) & (rng.random(len(pp)) < sigmoid(lc))
            yac_mu = np.maximum(P["yac_off"][S[pp], off[pp]] + P["yac_def"][S[pp], de[pp]] - league.yac, 0.5)
            yac = np.round(rng.gamma(1.15, yac_mu / 1.15) * np.where(a > 15, 0.6, 1.0))
            yards[pp] = np.where(comp[pp], a + yac, 0).astype(np.int64)
            fum[pp] = comp[pp] & (rng.random(len(pp)) < 0.007)
            oob = rng.random(len(pp)) < 0.25
            tclock[pp] = np.where(comp[pp] & ~oob, pace[pp] * rng.uniform(0.85, 1.15, len(pp)), 6.0)
            stop_clock[pp] = ~comp[pp] | oob
            pbu_m = (~comp[pp]) & (~intc[pp]) & (rng.random(len(pp)) < 0.35)
            pl = pp[pbu_m]
            if len(pl):
                pbu[pl] = np.array(DEF_IDX)[_cat(rng, cum["int_share"][S[pl], de[pl]])]
            il = pp[intc[pp]]
            if len(il):
                intcp[il] = np.array(DEF_IDX)[_cat(rng, cum["int_share"][S[il], de[il]])]
                int_ret_y[il] = np.round(rng.gamma(1.0, 9.0, len(il))).astype(np.int64)
        # -- sacks --
        sk = np.flatnonzero(sackm)
        if len(sk):
            passer[sk] = SLOT["QB1"]
            yards[sk] = -np.round(3 + rng.gamma(2.0, 2.5, len(sk))).astype(np.int64)
            sacker[sk] = np.array(DEF_IDX)[_cat(rng, cum["sack_share"][S[sk], de[sk]])]
            fum[sk] = rng.random(len(sk)) < 0.08
            tclock[sk] = pace[sk] * rng.uniform(0.85, 1.15, len(sk))
        # -- kneels --
        kn = np.flatnonzero(ptype == PT["kneel"])
        if len(kn):
            yards[kn] = -1
            rusher[kn] = SLOT["QB1"]
            tclock[kn] = 40.0
        # -- penalties --
        pn = np.flatnonzero(ptype == PT["penalty"])
        if len(pn):
            on_off = rng.random(len(pn)) < 0.5
            big = rng.random(len(pn)) < 0.25
            pen_y[pn] = np.where(on_off, -np.where(big, 10, 5), np.where(big, 15, 5))
            tclock[pn] = 4.0
            stop_clock[pn] = True
        # -- field goals --
        fg = np.flatnonzero(ptype == PT["field_goal"])
        if len(fg):
            fgd[fg] = y0[fg] + 17
            fgm[fg] = rng.random(len(fg)) < sigmoid(5.6 - 0.10 * fgd[fg] + P["fg_skill"][S[fg], off[fg]])
            tclock[fg] = 5.0
        # -- punts --
        pt = np.flatnonzero(ptype == PT["punt"])
        if len(pt):
            punt_net[pt] = np.round(rng.normal(P["punt_net"][S[pt], off[pt]], 9)).astype(np.int64)
            yards[pt] = punt_net[pt]
            tclock[pt] = 7.0

        # ---- defensive tackle attribution on ball-carrier plays ----
        if pstats is not None or record:
            run_tk = np.flatnonzero(((ptype == PT["rush"]) | (ptype == PT["scramble"])) & (yards < y0))
            if len(run_tk):
                tackler[run_tk] = np.array(DEF_IDX)[_cat(rng, cum["tackle_run"][S[run_tk], de[run_tk]])]
            pass_tk = np.flatnonzero(comp & (yards < y0))
            if len(pass_tk):
                tackler[pass_tk] = np.array(DEF_IDX)[_cat(rng, cum["tackle_pass"][S[pass_tk], de[pass_tk]])]
            fl = np.flatnonzero(fum)
            if len(fl):
                ffum[fl] = np.where(tackler[fl] >= 0, tackler[fl],
                                    np.where(sacker[fl] >= 0, sacker[fl], np.array(DEF_IDX)[_cat(rng, cum["tackle_run"][S[fl], de[fl]])]))

        # ---- timeouts by trailing defence late ----
        to_use = late & (diff > 0) & ~stop_clock & (tos[S, de] > 0) & (clk < 150) & np.isin(ptype, [PT["rush"], PT["pass"], PT["scramble"], PT["sack"]])
        if to_use.any():
            tu = np.flatnonzero(to_use)
            tos[S[tu], de[tu]] -= 1
            tclock[tu] = np.minimum(tclock[tu], 6.0)

        # ======================= APPLY RESULTS =======================
        new_y = y0 - yards
        # two-point tries (OT round >= 3) resolve in a single snap
        tt = two_try[S]
        td = (new_y <= 0) & np.isin(ptype, [PT["rush"], PT["pass"], PT["scramble"]]) & ~intc & ~fum
        td = td & ((ptype != PT["pass"]) | comp)
        safety = (new_y >= 100) & np.isin(ptype, [PT["rush"], PT["sack"], PT["scramble"], PT["kneel"]]) & ~fum
        first = (yards >= di0) & ~td & ~intc & ~fum & np.isin(ptype, [PT["rush"], PT["pass"], PT["scramble"]])
        # penalty: spot & down logic
        pen_first = (ptype == PT["penalty"]) & (pen_y >= 15)

        # team counting stats
        sc = S
        is_dropback = np.isin(ptype, [PT["pass"], PT["sack"], PT["scramble"]])
        is_play = np.isin(ptype, [PT["pass"], PT["sack"], PT["scramble"], PT["rush"]])
        tadd(sc[is_play & ~tt], off[is_play & ~tt], "plays")
        tadd(sc[is_dropback & ~tt], off[is_dropback & ~tt], "dropbacks")
        pm = (ptype == PT["pass"]) & ~tt
        tadd(sc[pm], off[pm], "pass_att")
        cm = comp & ~tt
        tadd(sc[cm], off[cm], "pass_cmp")
        tadd(sc[cm], off[cm], "pass_yds", np.minimum(yards[cm], y0[cm]))
        rm = np.isin(ptype, [PT["rush"], PT["scramble"]]) & ~tt
        tadd(sc[rm], off[rm], "rush_att")
        tadd(sc[rm], off[rm], "rush_yds", np.minimum(yards[rm], y0[rm]))
        tadd(sc[sackm], off[sackm], "sacks")
        tadd(sc[intc], off[intc], "ints")
        tadd(sc[fum], off[fum], "fumbles_lost")
        tadd(sc[intc | fum], off[intc | fum], "turnovers")
        tadd(sc[first | pen_first], off[first | pen_first], "first_downs")
        ex = ((rm & (yards >= 12)) | (cm & (yards >= 16)))
        tadd(sc[ex], off[ex], "explosive")
        tadd(sc, off, "top_seconds", np.round(tclock).astype(np.int64))
        rz_new = (y0 <= 20) & ~in_rz[S] & is_play & ~ot[S]
        in_rz[S[rz_new]] = True
        tadd(sc[rz_new], off[rz_new], "rz_trips")

        # player counting stats (all from the same events)
        if pstats is not None:
            nt = ~tt
            gy = np.minimum(yards, y0)  # yards credited (capped at goal line)
            m_ = pm
            padd(sc[m_], off[m_], passer[m_], "pass_att")
            padd(sc[m_], off[m_], target[m_], "targets")
            m_ = cm
            padd(sc[m_], off[m_], passer[m_], "pass_cmp"); padd(sc[m_], off[m_], passer[m_], "pass_yds", gy[m_])
            padd(sc[m_], off[m_], target[m_], "receptions"); padd(sc[m_], off[m_], target[m_], "rec_yds", gy[m_])
            m_ = cm & td
            padd(sc[m_], off[m_], passer[m_], "pass_td"); padd(sc[m_], off[m_], target[m_], "rec_td")
            m_ = intc & nt
            padd(sc[m_], off[m_], passer[m_], "pass_int")
            padd(sc[m_], de[m_], intcp[m_], "def_int")
            m_ = sackm & nt
            padd(sc[m_], off[m_], passer[m_], "sacks_taken")
            padd(sc[m_], de[m_], sacker[m_], "def_sacks"); padd(sc[m_], de[m_], sacker[m_], "tackles"); padd(sc[m_], de[m_], sacker[m_], "tfl")
            m_ = rm
            padd(sc[m_], off[m_], rusher[m_], "rush_att"); padd(sc[m_], off[m_], rusher[m_], "rush_yds", gy[m_])
            m_ = rm & td
            padd(sc[m_], off[m_], rusher[m_], "rush_td")
            m_ = (tackler >= 0) & nt
            padd(sc[m_], de[m_], tackler[m_], "tackles")
            m_ = (tackler >= 0) & (yards < 0) & rm
            padd(sc[m_], de[m_], tackler[m_], "tfl")
            m_ = (pbu >= 0)
            padd(sc[m_], de[m_], pbu[m_], "pbu")
            m_ = (ffum >= 0) & fum
            padd(sc[m_], de[m_], ffum[m_], "forced_fumbles")
            m_ = fgd > 0
            padd(sc[m_], off[m_], np.full(m_.sum(), SLOT["K"]), "fga")
            m_ = fgm
            padd(sc[m_], off[m_], np.full(m_.sum(), SLOT["K"]), "fgm")
            m_ = ptype == PT["punt"]
            padd(sc[m_], off[m_], np.full(m_.sum(), SLOT["P"]), "punts")
            padd(sc[m_], off[m_], np.full(m_.sum(), SLOT["P"]), "punt_yds", np.minimum(punt_net[m_] + 0, y0[m_]))
        tadd(sc[ptype == PT["field_goal"]], off[ptype == PT["field_goal"]], "fga")
        tadd(sc[fgm], off[fgm], "fgm")
        tadd(sc[ptype == PT["punt"]], off[ptype == PT["punt"]], "punts")
        rtd = td & ~tt & rm
        ptd = td & ~tt & cm
        tadd(sc[rtd], off[rtd], "rush_td"); tadd(sc[ptd], off[ptd], "pass_td")
        conv4 = (d0 == 4) & (first | td) & is_play
        tadd(sc[conv4], off[conv4], "fourth_conv")

        # ---- clock ----
        not_ot = ~ot[S]
        clock[S] = np.where(not_ot, clk - tclock, clk)

        points_off = np.zeros(m, np.int64)
        points_def = np.zeros(m, np.int64)
        pat_res = np.full(m, -1)

        # (1) two-point-try lanes (OT shootout): success => +2, possession over
        if tt.any():
            ti_ = np.flatnonzero(tt)
            ok = td[ti_]
            score[S[ti_[ok]], off[ti_[ok]]] += 2
            points_off[ti_[ok]] = 2
            ot_advance(S[ti_])

        # (2) touchdowns -> PAT
        tdl = np.flatnonzero(td & ~tt)
        if len(tdl):
            lanes = S[tdl]
            o_ = off[tdl]
            score[lanes, o_] += 6
            tadd(lanes, o_, "td")
            tadd(lanes[in_rz[lanes]], o_[in_rz[lanes]], "rz_td")
            margin_after = score[lanes, o_] - score[lanes, 1 - o_]
            go2 = (ot[lanes] & (ot_round[lanes] >= 2)) | ((period[lanes] == 4) & (clock[lanes] < 600) &
                                                            np.isin(margin_after, [-2, -5, -9, -12, 1, 5]))
            succ2 = rng.random(len(lanes)) < league.two_pt
            xp_ok = rng.random(len(lanes)) < league.xp
            pts = np.where(go2, np.where(succ2, 2, 0), np.where(xp_ok, 1, 0))
            score[lanes, o_] += pts
            points_off[tdl] = 6 + pts
            pat_res[tdl] = np.where(go2, 20 + succ2, 10 + xp_ok)
            if pstats is not None:
                kick = ~go2
                padd(lanes[kick], o_[kick], np.full(kick.sum(), SLOT["K"]), "xpa")
                padd(lanes[kick & xp_ok], o_[kick & xp_ok], np.full((kick & xp_ok).sum(), SLOT["K"]), "xpm")
            score_then_kick(lanes, o_.copy())

        # (3) field goals
        fgl = np.flatnonzero(ptype == PT["field_goal"])
        if len(fgl):
            made = fgm[fgl]
            lm = S[fgl[made]]
            score[lm, off[fgl[made]]] += 3
            points_off[fgl[made]] = 3
            score_then_kick(lm, off[fgl[made]].copy())
            miss = fgl[~made]
            possession_change(S[miss], np.clip(100 - (y0[miss] + 7), 1, 80))

        # (4) punts
        pl = np.flatnonzero(ptype == PT["punt"])
        if len(pl):
            land = y0[pl] - punt_net[pl]
            recv_y = np.where(land <= 0, 80, 100 - land)
            possession_change(S[pl], recv_y)

        # (5) turnovers: interceptions (with return / pick-six) and fumbles
        il = np.flatnonzero(intc & ~tt)
        if len(il):
            catch_spot = y0[il] - np.nan_to_num(air[il]).astype(np.int64)   # offence perspective
            tb = catch_spot <= 0
            recv_y = np.where(tb, 75, 100 - catch_spot - int_ret_y[il])
            six = (~tb) & (recv_y <= 0) & ~ot[S[il]]
            if six.any():
                sl = S[il[six]]
                dside = de[il[six]]
                score[sl, dside] += 7
                points_def[il[six]] = 7
                score_then_kick(sl, dside.copy())
            rest = il[~six]
            possession_change(S[rest], np.where(recv_y[~six] <= 0, 1, recv_y[~six]))
        fl = np.flatnonzero(fum & ~tt & ~td)
        if len(fl):
            spot = np.clip(y0[fl] - np.maximum(yards[fl], -20), 1, 99)
            possession_change(S[fl], 100 - spot)

        # (6) safeties
        sf = np.flatnonzero(safety & ~tt)
        if len(sf):
            lanes = S[sf]
            score[lanes, de[sf]] += 2
            points_def[sf] = 2
            # conceding team free-kicks
            poss[lanes] = off[sf]
            phase[lanes] = KICK
            ytg[lanes] = 80

        # (7) normal down/distance progression
        normal = np.isin(ptype, [PT["rush"], PT["pass"], PT["scramble"], PT["sack"], PT["kneel"]]) & \
            ~td & ~intc & ~fum & ~safety & ~tt
        nl = np.flatnonzero(normal)
        if len(nl):
            lanes = S[nl]
            ny = new_y[nl]
            fd = first[nl]
            ytg[lanes] = ny
            down[lanes] = np.where(fd, 1, down[lanes] + 1)
            dist[lanes] = np.where(fd, np.minimum(10, ny), dist[lanes] - yards[nl])
            # turnover on downs
            tod = (down[lanes] > 4)
            if tod.any():
                possession_change(lanes[tod], 100 - ny[tod])
        # penalties
        pnl = np.flatnonzero(ptype == PT["penalty"])
        if len(pnl):
            lanes = S[pnl]
            ny = np.clip(y0[pnl] - pen_y[pnl], 1, 99)
            # half-the-distance near goal line
            ny = np.where(ny < 1, np.maximum(1, y0[pnl] // 2), ny)
            ytg[lanes] = ny
            fd = pen_first[pnl]
            down[lanes] = np.where(fd, 1, down[lanes])
            dist[lanes] = np.where(fd, np.minimum(10, ny), np.maximum(1, dist[lanes] - pen_y[pnl]))
            dist[lanes] = np.minimum(dist[lanes], ny)

        # ---- logging ----
        if record:
            logs.append(_mk_log(S, step, off, period, clock_pre, d0, di0, None, score_pre,
                                np.where(tt, PT["two_point"], ptype), yards=yards, air=air,
                                comp=comp, intc=intc, sack=sackm, fum=fum, td=td & ~tt, first=first | pen_first,
                                points=points_off, def_points=points_def, pen_y=pen_y, fgd=fgd, fgm=fgm,
                                passer=passer, rusher=rusher, target=target, tackler=tackler, sacker=sacker,
                                intcp=intcp, pbu=pbu, ffum=ffum, drive=drive_pre, y0=y0, pat=pat_res,
                                pre_score=None))

        # ======================= PERIOD / GAME END =======================
        exp = np.flatnonzero((phase != DONE) & (clock <= 0) & ~ot)
        if len(exp):
            reg_end = exp[period[exp] == 4]
            q_end = exp[period[exp] < 4]
            period[q_end] += 1
            clock[q_end] = 900.0
            half = q_end[period[q_end] == 3]
            if len(half):
                # second-half kickoff: team that received first now kicks
                poss[half] = first_recv[half]
                phase[half] = KICK
                tos[half] = 3
                in_rz[half] = False
            if len(reg_end):
                tied = score[reg_end, 0] == score[reg_end, 1]
                phase[reg_end[~tied]] = DONE
                ot_l = reg_end[tied]
                if len(ot_l):
                    ot[ot_l] = True
                    ot_round[ot_l] = 1
                    ot_count[ot_l] = 0
                    ot_first[ot_l] = (rng.random(len(ot_l)) < 0.5).astype(np.int64)
                    poss[ot_l] = ot_first[ot_l]
                    ot_setup(ot_l)
        # drive resets for red-zone flag when possession changed
        newposs = (phase == KICK)
        in_rz[newposs] = False
        # safety valve: OT cannot exceed 9 rounds (2-pt shootout terminates quickly in practice)
        long_ot = np.flatnonzero(ot & (ot_round > 9) & (phase != DONE))
        if len(long_ot):
            win = (rng.random(len(long_ot)) < 0.5).astype(np.int64)
            score[long_ot, win] += 2
            phase[long_ot] = DONE

    still = phase != DONE
    if still.any():   # pathological lanes (should not occur): close them deterministically
        phase[still] = DONE
    log = None
    if record:
        log = {k: np.concatenate([d[k] for d in logs]) for k in LOG_COLS}
    return SimResult(score[:, 0].copy(), score[:, 1].copy(), ot.copy(), tstats, pstats, log)


def _mk_log(lanes, step, off, period, clock, d0, di0, ytg_now, score, ptype, *, yards, drive,
            air=None, comp=None, intc=None, sack=None, fum=None, td=None, first=None, points=None,
            def_points=None, pen_y=None, fgd=None, fgm=None, passer=None, rusher=None, target=None,
            tackler=None, sacker=None, intcp=None, pbu=None, ffum=None, y0=None, pat=None, pre_score=None):
    n = len(lanes)
    z = np.zeros(n, np.int64)
    neg = np.full(n, -1)
    off = np.asarray(off)
    sc = score[lanes]
    y_ = y0 if y0 is not None else ytg_now[lanes]
    pt = np.broadcast_to(np.asarray(ptype), (n,)).copy()
    out = {
        "lane": lanes.copy(), "step": np.full(n, step), "offense": off.copy(),
        "period": period[lanes].copy(), "clock": clock[lanes].copy(),
        "down": np.asarray(d0).copy() if np.ndim(d0) else np.full(n, d0),
        "distance": np.asarray(di0).copy() if np.ndim(di0) else np.full(n, di0),
        "ytg": np.asarray(y_).copy(),
        "off_score": sc[np.arange(n), off], "def_score": sc[np.arange(n), 1 - off],
        "ptype": pt, "yards": np.asarray(yards).copy(),
        "air": air.copy() if air is not None else np.full(n, np.nan),
        "completion": comp if comp is not None else z, "interception": intc if intc is not None else z,
        "sack": sack if sack is not None else z, "fumble_lost": fum if fum is not None else z,
        "touchdown": td if td is not None else z, "first_down": first if first is not None else z,
        "points": points if points is not None else z, "def_points": def_points if def_points is not None else z,
        "penalty_yards": pen_y if pen_y is not None else z, "fg_dist": fgd if fgd is not None else z,
        "fg_made": fgm if fgm is not None else z,
        "passer": passer if passer is not None else neg, "rusher": rusher if rusher is not None else neg,
        "target": target if target is not None else neg, "tackler": tackler if tackler is not None else neg,
        "sacker": sacker if sacker is not None else neg, "interceptor": intcp if intcp is not None else neg,
        "pbu": pbu if pbu is not None else neg, "forced_fumble": ffum if ffum is not None else neg,
        "kicker": np.where(pt == PT["field_goal"], SLOT["K"], -1), "punter": np.where(pt == PT["punt"], SLOT["P"], -1),
        "drive": drive[lanes].copy(), "pat_result": pat if pat is not None else neg,
        "scramble": (pt == PT["scramble"]),
    }
    return {k: np.asarray(v) for k, v in out.items()}
