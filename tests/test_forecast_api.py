import os

import pytest


def test_forecast_consistency_and_determinism(conn):
    from cfbquant.models.forecast import forecast_game
    g = conn.execute("SELECT home_id, away_id, week FROM games WHERE season=2025 AND week=6 LIMIT 1").fetchone()
    a = forecast_game(conn, g[0], g[1], 2025, g[2], n_sims=1500, seed=5, drivers=False, scenarios=False)
    b = forecast_game(conn, g[0], g[1], 2025, g[2], n_sims=1500, seed=5, drivers=False, scenarios=False)
    assert a["consistency_checks"]["all_pass"]
    assert a["home_win_prob_raw"] == b["home_win_prob_raw"]
    assert 0 < a["home_win_prob"] < 1
    qb = next(p for p in a["players"] if p["slot"] == "QB1" and p["side"] == "home")
    s = qb["stats"]["pass_yds"]
    assert s["p10"] <= s["median"] <= s["p90"]
    team_pass = a["team_stats"]["home"]["pass_yds"]["mean"]
    rec = sum(p["stats"].get("rec_yds", {}).get("mean", 0) for p in a["players"] if p["side"] == "home")
    assert rec <= team_pass + 1e-6


def test_qb_out_scenario_hurts(conn):
    from cfbquant.models.forecast import forecast_game
    g = conn.execute("SELECT home_id, away_id, week FROM games WHERE season=2025 AND week=7 LIMIT 1").fetchone()
    fc = forecast_game(conn, g[0], g[1], 2025, g[2], n_sims=1500, seed=2, drivers=True, scenarios=True)
    sc = {s["scenario"]: s["home_win_prob"] for s in fc["scenarios"]}
    assert sc["Home QB -1 SD"] < sc["Base case (current)"] < sc["Home QB +1 SD"]
    assert any(d["unit"] == "hfa" for d in fc["drivers"])


@pytest.fixture()
def client(small_db, monkeypatch):
    monkeypatch.setenv("CFBQ_DB", str(small_db))
    import cfbquant.api.app as appmod
    appmod._conn = None
    appmod._forecast.cache_clear()
    from fastapi.testclient import TestClient
    return TestClient(appmod.app)


def test_api_endpoints(client):
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert meta["synthetic"] is True
    teams = client.get("/api/teams").json()["teams"]
    tid = teams[0]["team_id"]
    t = client.get(f"/api/teams/{tid}").json()
    assert len(t["categories"]) == 20 and sum(len(c["metrics"]) for c in t["categories"]) >= 200
    pl = client.get("/api/players?position=QB").json()["players"]
    p = client.get(f"/api/players/{pl[0]['player_id']}?project=false").json()
    assert sum(len(c["metrics"]) for c in p["universal"]) >= 80
    assert any(m["status"] == "unavailable" for m in p["position_metrics"])
    fc = client.get(f"/api/forecast?home={teams[0]['team_id']}&away={teams[1]['team_id']}&sims=800&week=5").json()
    assert fc["consistency_checks"]["all_pass"] and fc["n_sims"] == 800
    mu = client.get(f"/api/matchup?home={teams[0]['team_id']}&away={teams[1]['team_id']}").json()
    assert len(mu["units"]) >= 10
    for path in ("/api/recruiting", "/api/transfers", "/api/audit", "/api/games", "/api/metrics/definitions"):
        assert client.get(path).status_code == 200, path
    assert client.get("/").status_code == 200
