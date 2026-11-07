"""Expected Points model + play enrichment (EPA, success, garbage time).

EP is the expected value of the *next score in the same half* from the
offence's perspective, estimated with a multinomial logistic regression over
seven outcomes {TD, FG, Safety, none, -Safety, -FG, -TD} (nflfastR-style).
EP = sum_k P(k | state) * value_k.  State = down, distance, yards-to-goal.

EPA(play) = EP(after) - EP(before), with scoring / possession-change handling:
  offence TD -> 7 - EPb ; defensive TD -> -7 - EPb ; FG -> 3 - EPb ;
  safety -> -2 - EPb ; change of possession -> -EP(opp next state) - EPb ;
  end of half -> -EPb.

Success (CFBD / Football Outsiders definition): gain >= 50% of distance on
1st down, >= 70% on 2nd, 100% on 3rd/4th.
Garbage time (Connelly): score margin > 38 in Q2, > 28 in Q3, > 22 in Q4.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..db import utcnow

OUTCOME_VALUES = np.array([7, 3, 2, 0, -2, -3, -7], dtype=float)
SCRIMMAGE = ("pass", "rush", "sack", "scramble", "kneel", "penalty", "punt", "field_goal")
OFFENSIVE_PLAYS = ("pass", "rush", "sack", "scramble")


def _features(ytg, down, dist) -> np.ndarray:
    y = np.asarray(ytg, float) / 100.0
    d = np.asarray(down, float)
    dist = np.asarray(dist, float)
    gtg = (dist >= np.asarray(ytg, float)).astype(float)
    cols = [y, y ** 2, y ** 3, np.log1p(dist), (d == 2), (d == 3), (d == 4), gtg,
            y * (d == 3), y * (d == 4), np.log1p(dist) * (d >= 3)]
    return np.column_stack([np.asarray(c, float) for c in cols])


class EPModel:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=2000, C=10.0)

    def fit(self, ytg, down, dist, outcome_idx) -> "EPModel":
        self.clf.fit(_features(ytg, down, dist), outcome_idx)
        return self

    def ep(self, ytg, down, dist) -> np.ndarray:
        X = _features(ytg, down, dist)
        proba = self.clf.predict_proba(X)
        vals = OUTCOME_VALUES[self.clf.classes_]
        return proba @ vals

    def to_json(self) -> str:
        return json.dumps({"coef": self.clf.coef_.tolist(), "intercept": self.clf.intercept_.tolist(),
                           "classes": self.clf.classes_.tolist()})


def _half(period: pd.Series) -> pd.Series:
    return np.where(period <= 2, 1, np.where(period <= 4, 2, 3))


def label_next_score(plays: pd.DataFrame) -> pd.Series:
    """Index into OUTCOME_VALUES of the next score in the half, offence perspective."""
    p = plays
    scoring = (p.points != 0) & ~p.play_type.isin(["pat", "two_point"])
    scorer = np.where(p.points > 0, p.offense_id, p.defense_id)
    absval = p.points.abs().map({6: 7, 7: 7, 8: 7, 3: 3, 2: 2}).fillna(0)
    ev_team = pd.Series(np.where(scoring, scorer, np.nan), index=p.index)
    ev_val = pd.Series(np.where(scoring, absval, np.nan), index=p.index)
    key = [p.game_id, pd.Series(_half(p.period), index=p.index)]
    nt = ev_team.groupby(key).bfill()
    nv = ev_val.groupby(key).bfill()
    signed = np.where(nt.isna(), 0.0, np.where(nt == p.offense_id, nv, -nv))
    idx = pd.Series(signed).map({7: 0, 3: 1, 2: 2, 0: 3, -2: 4, -3: 5, -7: 6}).fillna(3).astype(int)
    return pd.Series(idx.values, index=p.index)


def enrich_plays(conn, seasons: list[int] | None = None) -> dict:
    """Fit EP on all regulation scrimmage plays, then write ep/epa/success/garbage back."""
    where = "" if not seasons else f"WHERE season IN ({','.join(str(int(s)) for s in seasons)})"
    p = pd.read_sql_query(f"""SELECT play_id, game_id, season, week, play_number, offense_id, defense_id, period,
        offense_score, defense_score, down, distance, yards_to_goal, play_type, yards_gained, points, touchdown,
        interception, fumble_lost, fg_made FROM plays {where} ORDER BY game_id, play_number, play_id""", conn)
    if p.empty:
        return {"plays": 0}
    p["next_idx"] = label_next_score(p)
    scr = p[p.play_type.isin(SCRIMMAGE) & p.down.notna() & (p.period <= 4)].copy()
    fit = scr[scr.down.between(1, 4) & scr.yards_to_goal.between(1, 99)]
    model = EPModel().fit(fit.yards_to_goal, fit.down, fit.distance.clip(lower=1), fit.next_idx)
    scr["ep_before"] = model.ep(scr.yards_to_goal.clip(1, 99), scr.down.clip(1, 4), scr.distance.clip(lower=1))

    # EP after: next scrimmage play in same game & half
    scr["half"] = _half(scr.period)
    g = scr.groupby(["game_id", "half"], sort=False)
    nxt_ep = g.ep_before.shift(-1)
    nxt_off = g.offense_id.shift(-1)
    same = nxt_off == scr.offense_id
    after = np.where(nxt_ep.isna(), 0.0, np.where(same, nxt_ep, -nxt_ep))
    pts = scr.points.values
    off_td = (scr.touchdown == 1) & (pts > 0)
    after = np.where(off_td, 7.0, after)
    after = np.where((scr.play_type == "field_goal") & (scr.fg_made == 1), 3.0, after)
    after = np.where(pts <= -6, -7.0, after)          # defensive TD
    after = np.where(pts == -2, -2.0, after)          # safety
    scr["ep_after"] = after
    scr["epa"] = scr.ep_after - scr.ep_before

    off = scr.play_type.isin(OFFENSIVE_PLAYS)
    need = np.select([scr.down == 1, scr.down == 2], [0.5, 0.7], 1.0) * scr.distance
    gained = scr.yards_gained.where(~(scr.interception.astype(bool) | scr.fumble_lost.astype(bool)), -99)
    scr["success"] = np.where(off, ((gained >= need) | (scr.touchdown == 1)).astype(int), None)
    margin = (scr.offense_score - scr.defense_score).abs()
    scr["garbage_time"] = (((scr.period == 2) & (margin > 38)) | ((scr.period == 3) & (margin > 28)) |
                           ((scr.period == 4) & (margin > 22))).astype(int)
    rows = list(zip(scr.ep_before.round(4), scr.ep_after.round(4), scr.epa.round(4),
                    [None if v is None else int(v) for v in scr.success], scr.garbage_time.astype(int),
                    scr.play_id.astype(int)))
    rows = [tuple(None if (isinstance(x, float) and x != x) else (x.item() if hasattr(x, "item") else x) for x in r)
            for r in rows]
    conn.executemany("UPDATE plays SET ep_before=?, ep_after=?, epa=?, success=?, garbage_time=? WHERE play_id=?", rows)
    conn.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?,?)",
                 ("ep_v1", "expected_points", utcnow(), str(p.season.min()), str(p.season.max()),
                  model.to_json(), "production", "Multinomial logit next-score EP model"))
    conn.commit()
    return {"plays": len(scr), "mean_epa_offense": float(scr.loc[off, "epa"].mean())}
