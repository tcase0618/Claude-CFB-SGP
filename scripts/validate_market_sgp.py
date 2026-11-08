"""Real-money test of the simulated correlation on 2-leg team SGPs, using Polymarket US closing prices.

    python scripts/validate_market_sgp.py --season 2026 --weeks 1-10 --sims 2000

For every completed game with a Polymarket closing book: side x total, team total x opposing side, team total x
game total, every orientation, at the market's own coin-flip rungs. Compares three prices on log loss:
market independence, market marginals x simulated dependence (the live "Polymarket-anchored" price) and the pure
DraftKings-anchored model. Writes model_validation rows (model_version 'sgp_market_v1').
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfbquant.db import connect, utcnow  # noqa: E402
from cfbquant.ingest.polymarket import open_markets_db  # noqa: E402
from cfbquant.models.market_sgp import backtest_market_combos, summarize_market_combos  # noqa: E402


def _weeks(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--weeks", default="1-10")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--db", default=os.environ.get("CFBQ_DB", "data/cfb.db"))
    a = ap.parse_args()
    conn = connect(a.db)
    df = backtest_market_combos(conn, open_markets_db(), a.season, _weeks(a.weeks), n_sims=a.sims, alpha=a.alpha)
    summ = summarize_market_combos(df)
    print(json.dumps(summ, indent=1))
    out = Path("data") / "market_sgp_combos.csv"
    df.to_csv(out, index=False)
    conn.execute("DELETE FROM model_validation WHERE model_version='sgp_market_v1'")
    now = utcnow()
    for fam, r in summ.items():
        for price in ("p_mkt_indep", "p_mkt_copula", "p_model"):
            conn.execute("""INSERT INTO model_validation (model_version, scope, metric, value, n, detail_json, created_at)
                VALUES (?,?,?,?,?,?,?)""", ("sgp_market_v1", f"combo:{fam}", f"log_loss:{price}", r[price]["log_loss"],
                                            r["n"], json.dumps({"mean_p": r[price]["mean_p"], "hit_rate": r["hit_rate"],
                                                                "games": r["games"]}), now))
        g = r["copula_gain_vs_indep"]
        conn.execute("""INSERT INTO model_validation (model_version, scope, metric, value, ci_low, ci_high, n, created_at)
            VALUES (?,?,?,?,?,?,?,?)""", ("sgp_market_v1", f"combo:{fam}", "copula_gain_vs_indep", g["mean"],
                                          g["mean"] - 1.645 * g["se"], g["mean"] + 1.645 * g["se"], r["n"], now))
    conn.commit()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
