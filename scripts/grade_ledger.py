"""Grade ledger/paper_ledger.jsonl CFB SGP tickets against ESPN final box scores; append settlement rows."""
import json, re, sys, unicodedata, urllib.request
from datetime import datetime, timezone
def norm(s):
    s = re.sub(r"[^a-z ]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()).split()
    return " ".join(w for w in s if w not in {"jr", "sr", "ii", "iii", "iv", "v"})
STAT = {"pass yds": ("passing", "passingYards"), "pass td": ("passing", "passingTouchdowns"),
        "completions": ("passing", "completions/passingAttempts"), "rush yds": ("rushing", "rushingYards"),
        "rec yds": ("receiving", "receivingYards"), "receptions": ("receiving", "receptions")}
def box(eid):
    j = json.load(urllib.request.urlopen(f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={eid}", timeout=30))
    c = j["header"]["competitions"][0]; final = c["status"]["type"]["completed"]
    teams = {t["team"]["location"].lower(): int(t["score"]) for t in c["competitors"]}
    home = [t["team"]["location"].lower() for t in c["competitors"] if t["homeAway"] == "home"][0]
    pl = {}
    for t in j["boxscore"].get("players", []):
        for g in t["statistics"]:
            for a in g["athletes"]:
                d = pl.setdefault(norm(a["athlete"]["displayName"]), {}); v = dict(zip(g["keys"], a["stats"]))
                for k, (grp, key) in STAT.items():
                    if g["name"] == grp and key in v:
                        try: d[k] = int(str(v[key]).split("/")[0])
                        except ValueError: pass
    return teams, home, pl, final
def val(leg, teams, pl):
    s = leg.lower()
    m = re.match(r"(.+?) (-?\+?[\d.]+)$", s)
    if s.endswith(" ml"):
        t = s[:-3]; o = [x for x in teams if x != t][0]; return teams[t] - teams[o], 0, "over"
    if s.startswith("game "):
        m = re.match(r"game (over|under) ([\d.]+)", s); return sum(teams.values()), float(m[2]), m[1]
    m2 = re.match(r"(.+?) (over|under) ([\d.]+) pts$", s)
    if m2 and m2[1] in teams: return teams[m2[1]], float(m2[3]), m2[2]
    if m and m[1] in teams:   # spread, e.g. "pittsburgh -3.5"
        t = m[1]; o = [x for x in teams if x != t][0]; return teams[t] - teams[o] + float(m[2]), 0, "over"
    m3 = re.match(r"(.+?) (over|under) ([\d.]+) (.+)$", s)
    name, side, line, stat = norm(m3[1]), m3[2], float(m3[3]), m3[4]
    if name not in pl: return None, line, side
    d = pl[name]
    if stat == "rush+rec yds": v = d.get("rush yds", 0) + d.get("rec yds", 0)
    elif stat == "pass+rush yds": v = d.get("pass yds", 0) + d.get("rush yds", 0)
    else: v = d.get(stat, 0)
    return v, line, side
def main(led):
    rows = [json.loads(l) for l in open(led)]; done = {r["ref_row"] for r in rows if r.get("type") == "settlement"}
    cache, out = {}, []
    for i, r in enumerate(rows):
        if r.get("type") != "ticket" or i in done or not r.get("legs"): continue
        gid = r["game_id"]; cache.setdefault(gid, box(gid)); teams, home, pl, final = cache[gid]
        if not final: continue
        gs = []
        for l in r["legs"]:
            v, line, side = val(l["leg"], teams, pl)
            g = "VOID" if v is None else ("WON" if (v > line if side == "over" else v < line) else "LOST")
            gs.append((l["leg"], v, g))
        res = "LOST" if any(g == "LOST" for *_, g in gs) else ("WON" if all(g == "WON" for *_, g in gs) else "VOID_AFFECTED")
        out.append({"type": "settlement", "ref_row": i, "settled_at": datetime.now(timezone.utc).isoformat(), "source": "ESPN final box score",
                    "result": res, "legs": [f"{a}: {b} {c}" for a, b, c in gs]})
    open(led, "a").write("".join(json.dumps(x) + "\n" for x in out))
    return rows, out
if __name__ == "__main__":
    rows, out = main(sys.argv[1] if len(sys.argv) > 1 else "ledger/paper_ledger.jsonl")
    for o in out: print(rows[o["ref_row"]]["matchup"], rows[o["ref_row"]]["name"], o["result"], "|", " ; ".join(o["legs"]))
