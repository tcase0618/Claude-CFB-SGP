"""Slot scan for CFB: build 3/5/8 tickets for the given ESPN game ids and append them to ledger/paper_ledger.jsonl.

Rules (ops, Nov 2026): one ticket per size per game, a leg may appear on at most 2 tickets per game,
player legs priced >= 0.75 are treated as 0.70 (Claude-CFB-SGP#1), games that already kicked off are skipped.
Usage: CFBQ_DB=data/cfb.db .venv/bin/python scripts/slot_cards.py "2026-11-14 12:00ET" 401... 401... [--workers 3]
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfbquant.db import open_db
from cfbquant.models import sgp_service as svc

HC = lambda p: 0.70 if p >= 0.75 else p

def card(gid):
    with open_db(os.environ.get("CFBQ_DB", "data/cfb.db")) as conn:
        start = conn.execute("select start_date from games where game_id=?", (gid,)).fetchone()[0]
        if datetime.fromisoformat(start.replace("Z", "+00:00")) <= datetime.now(timezone.utc):
            return gid, None, "already started"
        try:
            c = svc.game_card(conn, gid)
        except Exception as e:  # noqa: BLE001 - one bad game must not kill the slot
            return gid, None, f"error: {e}"
        return gid, {"home": c["home"], "away": c["away"], "tickets": c["tickets"], "anchored": c["meta"].get("anchored"),
                     "poly": c["polymarket"]["team_legs"]}, None

def pick(c):
    used, out = {}, []
    for size in ("3", "5", "8"):
        for t in c["tickets"].get(size, c["tickets"].get(int(size), [])):
            labels = [l["label"] for l in t["legs"]]
            if all(used.get(x, 0) < 2 for x in labels):
                for x in labels: used[x] = used.get(x, 0) + 1
                out.append((size, t)); break
    return out

def main():
    slot, ids = sys.argv[1], [int(x) for x in sys.argv[2:] if x.isdigit()]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    now = datetime.now(timezone.utc).isoformat(); rows = []
    with ProcessPoolExecutor(workers) as ex:
        for gid, c, why in ex.map(card, ids):
            if c is None:
                print(f"{gid}: SKIP ({why})"); rows.append({"type": "ticket", "slot": slot, "game_id": gid, "logged_at": now, "status": "PASS", "note": why}); continue
            m = f"{c['away']}@{c['home']}"
            for size, t in pick(c):
                legs = [{"leg": l["label"], "p": round(float(l["p"]), 4), "kind": l.get("kind", "player")} for l in t["legs"]]
                adj = float(t["p_joint"])
                for l in legs:
                    if l["kind"] == "player": adj *= HC(l["p"]) / l["p"]
                rows.append({"type": "ticket", "slot": slot, "game_id": gid, "matchup": m, "name": f"{size}-leg", "logged_at": now,
                             "legs": legs, "joint_sim": round(float(t["p_joint"]), 4), "joint_adjusted": round(adj, 4), "lift": round(float(t["lift"]), 2),
                             "fair_decimal": round(1 / adj, 2), "anchored": c["anchored"], "status": "OPEN",
                             "note": "slot_cards.py; exposure<=2; 75%+ player legs haircut; no combined book price"})
                print(f"{m:34} {size}-leg p={adj:.3f} fair {1/adj:5.2f} | " + " ; ".join(f"{l['leg']} ({l['p']:.2f})" for l in legs))
    os.makedirs("ledger", exist_ok=True)
    with open("ledger/paper_ledger.jsonl", "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    print(f"logged {sum(1 for r in rows if r['status']=='OPEN')} tickets, {sum(1 for r in rows if r['status']=='PASS')} skips")

if __name__ == "__main__":
    main()
