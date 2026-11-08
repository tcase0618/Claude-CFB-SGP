"""Walk-forward validation of the prop / SGP engine on one season (see cfbquant/models/sgp_backtest.py).

    CFBQ_DB=data/cfb.db python scripts/validate_sgp.py --season 2026 --weeks 3-10 --sims 2000
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from cfbquant.config import get_settings
from cfbquant.db import connect, migrate
from cfbquant.models.sgp_backtest import persist_sgp, run_sgp_backtest, summarize_sgp

ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int, default=2026)
ap.add_argument("--weeks", default="3-10")
ap.add_argument("--sims", type=int, default=2000)
ap.add_argument("--out", default="data/sgp_backtest.pkl")
a = ap.parse_args()
lo, _, hi = a.weeks.partition("-")
weeks = list(range(int(lo), int(hi or lo) + 1))
conn = connect(get_settings().db_path)
migrate(conn)
bt = run_sgp_backtest(conn, a.season, weeks, n_sims=a.sims)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pickle.dump(bt, open(a.out, "wb"))
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
for k, v in summarize_sgp(bt).items():
    print(f"\n== {k}\n{v}")

persist_sgp(conn, bt, a.season, weeks)
print("\nfinal prop calibrator:", json.dumps({k: round(v, 3) for k, v in bt["final_factors"].items()}))
