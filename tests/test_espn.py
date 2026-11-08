"""ESPN keyless ingestion: normalisation of a real cached game + text parsing."""
import json
from pathlib import Path

import pandas as pd

from cfbquant.ingest.espn import normalize_plays, parse_box, parse_lines, parse_text, pat_points, spot_to_ytg
from cfbquant.ingest.cfbd import derive_player_games

FX = json.loads((Path(__file__).parent / "fixtures" / "espn_game.json").read_text())
G = FX["game"]
ABBR = {int(k): v for k, v in FX["abbr"].items()}
P, E, F = normalize_plays(FX["plays"], G, ABBR)
BY = {p["play_text"][:40]: p for p in P}


def test_play_types_and_scoring():
    types = {p["play_type"] for p in P}
    assert {"rush", "pass", "sack", "field_goal", "punt", "kickoff", "penalty"} <= types
    fg = next(p for p in P if p["play_type"] == "field_goal")
    assert fg["points"] == 3 and fg["fg_made"] == 1 and fg["fg_distance"] == 41
    tds = [p for p in P if p["touchdown"] == 1]
    assert tds and all(p["points"] == 7 for p in tds)          # TD + kicked PAT
    pick6 = next(p for p in P if "intercepted" in (p["play_text"] or "") and p["points"] != 0)
    assert pick6["points"] == -7 and pick6["touchdown"] == 0 and pick6["interception"] == 1
    td_pass = next(p for p in tds if p["play_type"] == "pass")
    assert td_pass["completion"] == 1


def test_offense_defense_and_field_position():
    for p in P:
        assert {p["offense_id"], p["defense_id"]} == {G["home_id"], G["away_id"]}
        if p["play_type"] in ("rush", "pass", "sack"):
            assert 1 <= p["yards_to_goal"] <= 99 and p["down"] in (1, 2, 3, 4)


def test_participants_to_roles_and_box_identities():
    roles = {e["role"] for e in E}
    assert {"passer", "target", "rusher", "sacker", "kicker", "punter"} <= roles
    pg = derive_player_games(pd.DataFrame(P), pd.DataFrame(E))
    assert (pg.pass_cmp <= pg.pass_att).all()
    assert pg.pass_yds.sum() == pg.rec_yds.sum()


def test_box_and_lines():
    rows, ath = parse_box(FX["summary"])
    assert rows and ath and any(r.get("tackles", 0) > 0 for r in rows)
    ln = parse_lines(FX["summary"], G["game_id"])
    assert ln and ln[0]["provider"] and ln[0]["spread_home"] is not None


def test_text_parsing_charting_style():
    t = "(14:03) No Huddle-Shotgun #4 T.Castellanos pass complete short middle to #3 C.Eakin caught at WVU34, for 22 yards to the WVU27"
    r = parse_text(t, 49, "TTU", "WVU")       # LOS at WVU 49 (49 yards to go for TTU)
    assert r["air_yards"] == 15 and r["pass_location"] == "middle" and r["pass_depth"] == "short"
    assert r["shotgun"] == 1 and r["no_huddle"] == 1
    t2 = "(07:51) Shotgun #4 T.Castellanos pass incomplete deep right to #13 C.Campbell thrown to WVU16 broken up by #3 M.Sieg"
    r2 = parse_text(t2, 49, "TTU", "WVU")
    assert r2["air_yards"] == 33 and r2["pass_location"] == "right" and r2["pbu_text"]
    t3 = "#4 X pass complete short left to #3 Y caught at TTU30, for 5 yards"   # catch behind LOS in own territory
    assert parse_text(t3, 65, "TTU", "WVU")["air_yards"] == -5
    assert parse_text("Keelon Russell pass complete to Daniel Hill for 7 yds", 60, "ALA", "ECU")["air_yards"] is None


def test_spots_and_pat():
    assert spot_to_ytg("TTU", 30, "TTU", "WVU") == 70 and spot_to_ytg("WVU", 30, "TTU", "WVU") == 30
    assert spot_to_ytg("XYZ", 30, "TTU", "WVU") is None
    assert pat_points("run for 3 yds for a TD (Conor Talty KICK)") == 1
    assert pat_points("pass for a TD (kick failed)") == 0
    assert pat_points("run for a TD (two-point conversion failed)") == 0
