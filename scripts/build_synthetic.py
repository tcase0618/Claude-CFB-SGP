"""Build the fictional demonstration database (data/synthetic.db)."""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfbquant.db import open_db
from cfbquant.ingest.synthetic import build_synthetic

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/synthetic.db")
ap.add_argument("--seasons", default="2023,2024,2025")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
p = Path(a.db)
if p.exists():
    p.unlink()
t = time.time()
with open_db(p) as conn:
    build_synthetic(conn, tuple(int(s) for s in a.seasons.split(",")), a.seed)
    for tb in ("games", "plays", "play_events", "player_games", "drives", "players", "recruiting_profiles", "transfer_history", "injuries"):
        print(tb, conn.execute(f"select count(*) from {tb}").fetchone()[0])
print(f"built in {time.time()-t:.1f}s")
