import json
from pathlib import Path

import pandas as pd

from cfbquant.ingest.cfbd import (derive_player_games, normalize_game, normalize_play, normalize_play_stat,
                                  normalize_roster, parse_box_defense)

FX = json.loads((Path(__file__).parent / "fixtures" / "cfbd_sample.json").read_text())
TEAMS = {"Alpha State": 1, "Beta Tech": 2}


def test_normalize_game():
    g = normalize_game(FX["games"][0], TEAMS)
    assert g["home_id"] == 1 and g["away_id"] == 2 and g["completed"] == 1 and g["source"] == "cfbd"
    assert normalize_game({**FX["games"][0], "awayId": 999}, TEAMS) is None


def test_normalize_plays_types_and_points():
    rows = [normalize_play(p, TEAMS, 2024, 1) for p in FX["plays"]]
    by = {r["play_id"]: r for r in rows}
    assert by[1]["play_type"] == "pass" and by[1]["completion"] == 1 and by[1]["clock_seconds"] == 14 * 60 + 30
    assert by[2]["play_type"] == "rush" and by[2]["touchdown"] == 1 and by[2]["points"] == 6
    assert by[3]["play_type"] == "field_goal" and by[3]["fg_made"] == 1 and by[3]["fg_distance"] == 42 and by[3]["points"] == 3
    assert by[4]["play_type"] == "pass" and by[4]["interception"] == 1 and by[4]["points"] == -6
    assert by[5]["play_type"] == "sack" and by[5]["sack"] == 1
    assert all(r["air_yards"] is None for r in rows), "CFBD has no air yards; must stay NULL"


def test_snake_case_tolerated():
    p = dict(FX["plays"][0])
    snake = {"id": p["id"], "game_id": p["gameId"], "offense": p["offense"], "defense": p["defense"], "home": p["home"],
             "period": 1, "yards_to_goal": 75, "down": 1, "distance": 10, "yards_gained": 8, "play_type": "Rush"}
    r = normalize_play(snake, TEAMS, 2024, 1)
    assert r["play_type"] == "rush" and r["yards_to_goal"] == 75


def test_play_stats_and_derived_box_scores():
    plays = pd.DataFrame([normalize_play(p, TEAMS, 2024, 1) for p in FX["plays"]])
    ev = pd.DataFrame([s for s in (normalize_play_stat(x, TEAMS) for x in FX["play_stats"]) if s]).drop(columns=["_name"])
    assert set(ev.role) >= {"passer", "target", "rusher", "interceptor", "sacker"}
    pg = derive_player_games(plays, ev).set_index("player_id")
    assert pg.loc[100, "pass_att"] == 2 and pg.loc[100, "pass_cmp"] == 1 and pg.loc[100, "pass_int"] == 1
    assert pg.loc[100, "pass_yds"] == pg.loc[200, "rec_yds"] == 12
    assert pg.loc[100, "sacks_taken"] == 1 and pg.loc[300, "rush_td"] == 1
    assert pg.loc[900, "def_int"] == 1


def test_roster_and_box_defense():
    p, r = normalize_roster(FX["roster"][0], TEAMS, 2024)
    assert p["position"] == "EDGE" and r["team_id"] == 1
    rows = parse_box_defense(FX["box"][0])
    assert {"tackles", "tfl", "pbu"} <= {x["stat"] for x in rows}
