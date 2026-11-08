"""Tune the strength-dispersion factor k on one season (walk-forward), for out-of-sample use on a later season.

Estimates are computed once per week (data strictly before the week); each candidate k is simulated on the same
estimates with common random numbers. Selection metric: log loss of raw simulated win probability.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from cfbquant.db import open_db
import cfbquant.analytics.team_params as T
from cfbquant.sim.engine import simulate
from cfbquant.sim.params import League, stack_params

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/cfb.db"); ap.add_argument("--season", type=int, default=2025)
ap.add_argument("--ks", default="1.0,1.25,1.5,1.75,2.0"); ap.add_argument("--sims", type=int, default=200)
a = ap.parse_args()
ks = [float(x) for x in a.ks.split(",")]
rows = []
t0 = time.time()
with open_db(a.db) as conn:
    games = pd.read_sql_query("SELECT * FROM games WHERE season=? AND completed=1 ORDER BY week", conn, params=(a.season,))
    for w in sorted(games.week.unique()):
        gw = games[games.week == w]
        T.STRENGTH_K = 1.0
        est = T._estimate(conn, a.season, int(w), None)
        base = {t: p for t, p in est.params.items()}
        lvl = est.league_obs.get("level", {}).get("shift", 0.0)
        for k in ks:
            # undo level shift, amplify, re-apply the same level shift (scoring level is not what we tune)
            pairs = [(T.shift_level(T.amplify(T.shift_level(base[h], -lvl), k), lvl), T.shift_level(T.amplify(T.shift_level(base[aw], -lvl), k), lvl))
                     for h, aw in zip(gw.home_id, gw.away_id)]
            P = {kk: np.repeat(v, a.sims, axis=0) for kk, v in stack_params(pairs).items()}
            r = simulate(P, League(), 1000 + int(w), neutral=np.repeat(gw.neutral_site.values.astype(bool), a.sims), track_players=False)
            m = (r.home_pts - r.away_pts).reshape(len(gw), a.sims)
            for i, g in enumerate(gw.itertuples()):
                rows.append(dict(k=k, week=w, game_id=g.game_id, p=(m[i] > 0).mean(), mm=m[i].mean(), y=g.home_points - g.away_points))
        print(f"week {w}: {len(gw)} games ({time.time() - t0:.0f}s)", flush=True)
d = pd.DataFrame(rows)
d["win"] = (d.y > 0).astype(float)
pc = d.p.clip(0.01, 0.99)
d["ll"] = -(d.win * np.log(pc) + (1 - d.win) * np.log(1 - pc))
res = d.groupby("k").apply(lambda q: pd.Series({"log_loss": q.ll.mean(), "margin_mae": (q.mm - q.y).abs().mean(),
                                                "slope": np.polyfit(q.mm, q.y, 1)[0]}), include_groups=False)
print(res.round(4).to_string())
best = float(res.log_loss.idxmin())
print(json.dumps({"season": a.season, "best_k": best}))
