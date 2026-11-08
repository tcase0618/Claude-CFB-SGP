"""CFBD recruiting/talent/portal merge into an ESPN-built DB (offline, fake client)."""
import pandas as pd

from cfbquant.db import open_db
from cfbquant.ingest.recruiting import _pname, load_recruiting


class FakeCFBD:
    def __init__(self):
        self.calls = []

    def get(self, endpoint, use_cache=True, **params):
        self.calls.append((endpoint, params))
        if endpoint == "/recruiting/players":
            return [{"id": "1", "athleteId": "4870906", "name": "Arch Manning", "committedTo": "Texas", "position": "QB",
                     "stars": 5, "rating": 0.9995, "ranking": 1},
                    {"id": "2", "athleteId": None, "name": "Joe Smith Jr.", "committedTo": "Ohio State", "position": "WR",
                     "stars": 4, "rating": 0.93, "ranking": 80},
                    {"id": "3", "athleteId": "999", "name": "Never Played", "committedTo": "Texas", "stars": 3, "rating": 0.85}]
        if endpoint == "/talent":
            return [{"team": "Texas", "talent": 985.2}, {"team": "Ohio State", "talent": 964.3}]
        if endpoint == "/player/portal":
            return [{"firstName": "Joe", "lastName": "Smith", "position": "WR", "origin": "Ohio State", "destination": "Texas",
                     "rating": 0.9, "stars": 4, "eligibility": "Immediate", "transferDate": "2026-01-05"},
                    {"firstName": "Ghost", "lastName": "Player", "origin": "Harvard", "destination": "Texas"}]
        return []


def test_merge(tmp_path):
    with open_db(tmp_path / "r.db") as c:
        c.execute("INSERT INTO data_sources VALUES ('espn','t',0,NULL,NULL)")
        c.execute("INSERT INTO conferences VALUES ('X','X','fbs')")
        c.execute("INSERT INTO teams (team_id, school, mascot, source) VALUES (251,'Texas','Longhorns','espn'),(194,'Ohio State','Buckeyes','espn')")
        for t in (251, 194):
            c.execute("INSERT INTO team_seasons (team_id, season, conference_id, division) VALUES (?,2026,'X','fbs')", (t,))
        c.execute("INSERT INTO players (player_id, first_name, last_name, source) VALUES (4870906,'Arch','Manning','espn'),(555,'Joe','Smith','espn')")
        c.execute("INSERT INTO rosters (player_id, team_id, season) VALUES (4870906,251,2026),(555,194,2025),(555,251,2026)")
        c.commit()
        st = load_recruiting(c, FakeCFBD(), [2023], [2026], log=lambda *a: None)
        rp = pd.read_sql_query("SELECT * FROM recruiting_profiles", c)
        assert rp.set_index("recruit_id").loc[1, "player_id"] == 4870906            # athleteId == ESPN id
        assert rp.set_index("recruit_id").loc[2, "player_id"] == 555                # name + committed-team fallback
        assert pd.isna(rp.set_index("recruit_id").loc[3, "player_id"])              # unknown athlete stays unmatched
        assert c.execute("SELECT talent_composite FROM team_seasons WHERE team_id=251").fetchone()[0] == 985.2
        tr = pd.read_sql_query("SELECT * FROM transfer_history ORDER BY transfer_id", c)
        assert tr.player_id.iloc[0] == 555 and tr.from_team_id.iloc[0] == 194 and tr.to_team_id.iloc[0] == 251
        assert pd.isna(tr.player_id.iloc[1]) and pd.isna(tr.from_team_id.iloc[1])   # non-feed school: unmatched, not guessed
        assert st["recruits_matched"] == 2 and st["portal_matched"] == 1


def test_name_normalisation():
    assert _pname("Joe Smith Jr.") == _pname("joe smith") == "joesmith"
    assert _pname("Michael Hawkins Jr") == "michaelhawkins"
