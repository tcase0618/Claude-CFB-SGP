"""200+ metric team intelligence library.

Each metric is declared once with its formula, side, direction and
classification, then computed from play/drive/game/roster data. Metrics that
genuinely require licensed charting or tracking data are *registered* (so the
UI can list them) but are classified `unavailable_without_charting` and are
never filled with fabricated values.

Opponent adjustment: for metrics flagged `adj`, game-level values are fit with
a weighted ridge model  y_g = mu + off_i + def_j + h*home  (ridge = Gaussian
prior shrinking small samples to the league mean), giving adjusted offence
and defence values on the same scale as the raw metric.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .plays_frame import load_plays

# --------------------------------------------------------------------------- filters
F = {
    "all": lambda p: p.is_off,
    "db": lambda p: p.dropback, "att": lambda p: p.att, "rush": lambda p: p.rush, "carry": lambda p: p.carry,
    "d1": lambda p: p.down == 1, "d2": lambda p: p.down == 2, "d3": lambda p: p.down == 3, "d4": lambda p: p.down == 4,
    "early": lambda p: p.down <= 2,
    "pd": lambda p: ((p.down == 2) & (p.distance >= 8)) | ((p.down >= 3) & (p.distance >= 5)),
    "std": lambda p: ~(((p.down == 2) & (p.distance >= 8)) | ((p.down >= 3) & (p.distance >= 5))),
    "d3s": lambda p: (p.down == 3) & (p.distance <= 3), "d3m": lambda p: (p.down == 3) & p.distance.between(4, 6),
    "d3l": lambda p: (p.down == 3) & (p.distance >= 7), "d4s": lambda p: (p.down == 4) & (p.distance <= 2),
    "sy": lambda p: (p.down >= 3) & (p.distance <= 2),
    "rz": lambda p: p.yards_to_goal <= 20, "gtg": lambda p: p.distance >= p.yards_to_goal,
    "gl": lambda p: p.yards_to_goal <= 5, "own": lambda p: p.yards_to_goal >= 80,
    "neutral": lambda p: (p.period <= 3) & (p.margin.abs() <= 14),
    "q1": lambda p: p.period == 1, "q4": lambda p: p.period == 4,
    "h1": lambda p: p.period <= 2, "h2": lambda p: p.period.isin([3, 4]),
    "lead": lambda p: p.margin > 0, "trail": lambda p: p.margin < 0, "close": lambda p: p.margin.abs() <= 8,
    "late": lambda p: (p.period == 4) & (p.margin.abs() <= 8),
    "home": lambda p: p.off_home, "away": lambda p: p.off_away,
    "cold": lambda p: p.temperature_f < 45, "wind": lambda p: p.wind_mph >= 15,
    "deep": lambda p: p.att & (p.air_yards >= 20), "short": lambda p: p.att & (p.air_yards < 10),
    "inter": lambda p: p.att & p.air_yards.between(10, 19),
}


def mask(p: pd.DataFrame, expr: str) -> pd.Series:
    m = pd.Series(True, index=p.index)
    for part in expr.split("&"):
        m &= F[part.strip()](p).fillna(False).astype(bool)
    return m


# --------------------------------------------------------------------------- stats
def _mean(col):
    return lambda d: d[col].astype(float)
STAT = {
    "epa": _mean("epa"), "sr": _mean("success"), "ypp": _mean("yards_gained"), "expl": _mean("explosive"),
    "neg": _mean("negative"), "stuff": _mean("stuff"), "fd": _mean("first_down"), "comp": _mean("completion"),
    "sack": _mean("sack"), "int": _mean("interception"), "to": _mean("turnover"), "td": _mean("touchdown"),
    "yac": _mean("yac"), "air": _mean("air_yards"), "deeprate": _mean("deep"), "scr": _mean("scramble"),
    "dbrate": lambda d: d["dropback"].astype(float),
    "r10": lambda d: (d["yards_gained"] >= 10).astype(float),
    "rcore": lambda d: d["yards_gained"].where(d["yards_gained"] < 10).astype(float),
    "fumble": _mean("fumble_lost"),
}
NEEDS_AIR = {"yac", "air", "deeprate"}


@dataclass(frozen=True)
class TM:
    mid: str
    cat: str
    name: str
    side: str = "off"            # off | def | team
    filt: str = "all"
    stat: str = "epa"
    direction: int = 1
    adj: bool = False
    formula: str = ""
    classification: str = "calculated"
    clean: bool = True           # exclude garbage time
    required: str = "plays"


CATS = ["Overall Offensive Efficiency", "Passing Offense", "Rushing Offense", "Offensive Line", "Receiving Corps",
        "Offensive Scheme", "Overall Defensive Efficiency", "Pass Defense", "Rush Defense", "Pass Rush",
        "Coverage Schemes", "Down and Distance", "Red Zone and Scoring", "Special Teams", "Coaching and Play Calling",
        "Personnel and Roster Strength", "Recruiting and Talent", "Game Environment", "Trends and Regression",
        "Matchup Intelligence"]
C = {i + 1: c for i, c in enumerate(CATS)}

PLAY = []  # play-level specs


def _p(mid, cat, name, side, filt, stat, direction, adj=False, formula="", **kw):
    PLAY.append(TM(mid, C[cat], name, side, filt, stat, direction, adj, formula or f"{stat} over {filt} ({side})", **kw))


# 1 overall offense
_p("off_epa", 1, "EPA / Play (Offense)", "off", "all", "epa", 1, True, "mean(EPA) over non-garbage offensive plays")
_p("off_sr", 1, "Success Rate (Offense)", "off", "all", "sr", 1, True, "share of plays meeting 50/70/100% distance rule")
_p("off_ypp", 1, "Yards / Play", "off", "all", "ypp", 1, True, "mean yards gained per offensive play")
_p("off_expl", 1, "Explosive Play Rate", "off", "all", "expl", 1, True, "rush>=12 or completion>=16 yards / plays")
_p("off_fd", 1, "First Down Rate", "off", "all", "fd", 1, False, "first downs / plays")
_p("off_neg", 1, "Negative Play Rate", "off", "all", "neg", -1, False, "plays losing yardage / plays")
_p("off_to", 1, "Turnover Rate / Play", "off", "all", "to", -1, False, "(INT + fumbles lost) / plays")
# 2 passing offense
_p("pass_epa", 2, "Passing EPA / Dropback", "off", "db", "epa", 1, True, "mean EPA on dropbacks (incl. sacks, scrambles)")
_p("pass_sr", 2, "Passing Success Rate", "off", "db", "sr", 1, True)
_p("pass_nypd", 2, "Net Yards / Dropback", "off", "db", "ypp", 1, True, "yards incl. sack losses / dropbacks")
_p("pass_comp", 2, "Completion %", "off", "att", "comp", 1, True, "completions / attempts")
_p("pass_expl", 2, "Explosive Completion Rate", "off", "att", "expl", 1, False, "completions >=16 yds / attempts")
_p("pass_int", 2, "Interception Rate", "off", "att", "int", -1, True, "INT / attempts")
_p("pass_sack", 2, "Sack Rate Allowed", "off", "db", "sack", -1, True, "sacks / dropbacks")
_p("pass_deep_epa", 2, "Deep Pass EPA (20+ air)", "off", "deep", "epa", 1, False, "EPA on attempts with air yards >= 20", required="air_yards")
_p("pass_adot", 2, "Average Depth of Target", "off", "att", "air", 0, False, "mean air yards / attempt", required="air_yards")
_p("pass_scramble", 2, "Scramble Rate", "off", "db", "scr", 0, False, "scrambles / dropbacks")
# 3 rushing offense
_p("rush_epa", 3, "Rush EPA / Carry", "off", "rush", "epa", 1, True, "mean EPA on designed rushes")
_p("rush_sr", 3, "Rush Success Rate", "off", "rush", "sr", 1, True)
_p("rush_ypc", 3, "Yards / Carry", "off", "rush", "ypp", 1, True)
_p("rush_expl", 3, "Explosive Run Rate", "off", "carry", "expl", 1, True, "carries >= 12 yards / carries")
_p("rush_stuff", 3, "Stuff Rate", "off", "carry", "stuff", -1, True, "carries <= 0 yards / carries")
_p("rush_r10", 3, "10+ Yard Run Rate", "off", "carry", "r10", 1, False)
_p("rush_core", 3, "Core Run Yards (<10)", "off", "carry", "rcore", 1, False, "mean yards on carries gaining < 10 (efficiency excluding breakaways)")
_p("rush_gl_sr", 3, "Goal-Line Rush Success", "off", "rush&gl", "sr", 1, False, "success on designed runs inside the 5")
_p("rush_sy_sr", 3, "Short-Yardage Rush Success", "off", "rush&sy", "sr", 1, False, "3rd/4th & <=2 rush success")
_p("rush_fumble", 3, "Fumble-Lost Rate (Carries)", "off", "carry", "fumble", -1, False)
# 4 offensive line (play-derived proxies are labelled estimated)
_p("ol_sack", 4, "Sack Rate Allowed (OL proxy)", "off", "db", "sack", -1, True, "sacks / dropbacks — shared OL/QB responsibility", classification="estimated")
_p("ol_stuff", 4, "Run Stuff Rate Allowed (OL proxy)", "off", "rush", "stuff", -1, True, classification="estimated")
_p("ol_core", 4, "Core Rush Yards (OL proxy)", "off", "rush", "rcore", 1, True, "line-yards proxy: mean of runs < 10 yds", classification="estimated")
_p("ol_sy", 4, "Power Success (3rd/4th & <=2 runs)", "off", "rush&sy", "sr", 1, False, classification="estimated")
_p("ol_neg", 4, "Negative Play Rate Allowed", "off", "all", "neg", -1, False, classification="estimated")
# 5 receiving corps
_p("rec_comp", 5, "Catch Rate (Team Targets)", "off", "att", "comp", 1, False)
_p("rec_yac", 5, "YAC / Completion", "off", "att", "yac", 1, False, required="air_yards")
_p("rec_expl", 5, "Explosive Reception Rate", "off", "att", "expl", 1, False)
_p("rec_epa", 5, "Receiving EPA / Target", "off", "att", "epa", 1, True)
_p("rec_short_comp", 5, "Short Target Catch Rate (<10 air)", "off", "short", "comp", 1, False, required="air_yards")
_p("rec_inter_comp", 5, "Intermediate Catch Rate (10-19)", "off", "inter", "comp", 1, False, required="air_yards")
_p("rec_deep_comp", 5, "Deep Catch Rate (20+)", "off", "deep", "comp", 1, False, required="air_yards")
# 6 offensive scheme (play-derived)
_p("sch_pass_rate", 6, "Overall Dropback Rate", "off", "all", "dbrate", 0, False, "dropbacks / plays")
_p("sch_neutral_pass", 6, "Neutral-Situation Pass Rate", "off", "neutral", "dbrate", 0, False, "dropbacks / plays in Q1-3 with margin <= 14")
_p("sch_early_pass", 6, "Early-Down Pass Rate", "off", "early&neutral", "dbrate", 0, False)
_p("sch_deep_rate", 6, "Deep Shot Rate", "off", "att", "deeprate", 0, False, required="air_yards")
_p("sch_early_epa", 6, "Early-Down EPA", "off", "early", "epa", 1, False)
# 7 overall defense
_p("def_epa", 7, "EPA / Play Allowed", "def", "all", "epa", -1, True)
_p("def_sr", 7, "Success Rate Allowed", "def", "all", "sr", -1, True)
_p("def_ypp", 7, "Yards / Play Allowed", "def", "all", "ypp", -1, True)
_p("def_expl", 7, "Explosive Plays Allowed Rate", "def", "all", "expl", -1, True)
_p("def_to", 7, "Turnover Creation Rate", "def", "all", "to", 1, False)
_p("def_neg", 7, "Negative Play Generation", "def", "all", "neg", 1, False)
_p("def_fd", 7, "First Down Rate Allowed", "def", "all", "fd", -1, False)
# 8 pass defense
_p("pdef_epa", 8, "Passing EPA Allowed / Dropback", "def", "db", "epa", -1, True)
_p("pdef_sr", 8, "Passing Success Allowed", "def", "db", "sr", -1, True)
_p("pdef_comp", 8, "Completion % Allowed", "def", "att", "comp", -1, True)
_p("pdef_nypd", 8, "Net Yards / Dropback Allowed", "def", "db", "ypp", -1, True)
_p("pdef_expl", 8, "Explosive Passes Allowed", "def", "att", "expl", -1, False)
_p("pdef_int", 8, "Interception Rate", "def", "att", "int", 1, True)
_p("pdef_yac", 8, "YAC Allowed", "def", "att", "yac", -1, False, required="air_yards")
_p("pdef_deep_epa", 8, "Deep Pass EPA Allowed", "def", "deep", "epa", -1, False, required="air_yards")
# 9 rush defense
_p("rdef_epa", 9, "Rush EPA Allowed", "def", "rush", "epa", -1, True)
_p("rdef_sr", 9, "Rush Success Allowed", "def", "rush", "sr", -1, True)
_p("rdef_ypc", 9, "Yards / Carry Allowed", "def", "rush", "ypp", -1, True)
_p("rdef_stuff", 9, "Stuff Rate", "def", "carry", "stuff", 1, True)
_p("rdef_expl", 9, "Explosive Runs Allowed", "def", "carry", "expl", -1, True)
_p("rdef_core", 9, "Core Rush Yards Allowed", "def", "carry", "rcore", -1, False)
_p("rdef_sy", 9, "Short-Yardage Stop Rate", "def", "rush&sy", "sr", -1, False, "success allowed on 3rd/4th & <=2 runs (lower=better)")
_p("rdef_fumble", 9, "Forced Fumble-Lost Rate", "def", "carry", "fumble", 1, False)
# 10 pass rush (play-derived)
_p("pr_sack", 10, "Sack Rate", "def", "db", "sack", 1, True)
_p("pr_sack_pd", 10, "Passing-Down Sack Rate", "def", "db&pd", "sack", 1, False)
_p("pr_sack_std", 10, "Standard-Down Sack Rate", "def", "db&std", "sack", 1, False)
_p("pr_neg", 10, "Negative Dropback Rate Forced", "def", "db", "neg", 1, False)
# 12 down and distance (offense)
for d, nm in ((1, "First"), (2, "Second"), (3, "Third"), (4, "Fourth")):
    _p(f"dd_d{d}_epa", 12, f"{nm}-Down EPA", "off", f"d{d}", "epa", 1)
_p("dd_d3s_sr", 12, "3rd & Short Conversion (SR)", "off", "d3s", "sr", 1)
_p("dd_d3m_sr", 12, "3rd & Medium Conversion (SR)", "off", "d3m", "sr", 1)
_p("dd_d3l_sr", 12, "3rd & Long Conversion (SR)", "off", "d3l", "sr", 1)
_p("dd_d4s_sr", 12, "4th & Short Conversion (SR)", "off", "d4s", "sr", 1)
_p("dd_early_sr", 12, "Early-Down Success", "off", "early", "sr", 1)
_p("dd_std_epa", 12, "Standard-Down EPA", "off", "std", "epa", 1)
_p("dd_pd_epa", 12, "Passing-Down EPA", "off", "pd", "epa", 1)
_p("dd_d3_def", 12, "Third-Down Success Allowed", "def", "d3", "sr", -1)
# 13 red zone
_p("rz_epa", 13, "Red-Zone EPA / Play", "off", "rz", "epa", 1)
_p("rz_sr", 13, "Red-Zone Success Rate", "off", "rz", "sr", 1)
_p("rz_pass_epa", 13, "Red-Zone Passing EPA", "off", "rz&db", "epa", 1)
_p("rz_rush_epa", 13, "Red-Zone Rushing EPA", "off", "rz&rush", "epa", 1)
_p("rz_gtg_td", 13, "Goal-to-Go TD Rate / Play", "off", "gtg", "td", 1)
_p("rz_def_epa", 13, "Red-Zone EPA Allowed", "def", "rz", "epa", -1)
_p("rz_def_gl", 13, "Goal-Line Stop Rate", "def", "gl", "sr", -1, False, "success allowed inside the 5 (lower=better)")
_p("rz_to", 13, "Red-Zone Turnover Rate", "off", "rz", "to", -1)
# 18 environment splits
_p("env_home_epa", 18, "Home EPA / Play", "off", "home", "epa", 1)
_p("env_away_epa", 18, "Away EPA / Play", "off", "away", "epa", 1)
_p("env_cold_epa", 18, "Cold (<45F) EPA / Play", "off", "cold", "epa", 1, required="weather")
_p("env_wind_pass", 18, "Windy (>=15mph) Pass EPA", "off", "wind&db", "epa", 1, required="weather")
_p("env_q4_epa", 18, "4th-Quarter EPA", "off", "q4", "epa", 1)
_p("env_late_epa", 18, "Late & Close EPA", "off", "late", "epa", 1)
_p("env_trail_epa", 18, "Trailing EPA", "off", "trail", "epa", 1)
_p("env_lead_epa", 18, "Leading EPA", "off", "lead", "epa", 1)
_p("env_own_epa", 18, "Backed-Up (own 20) EPA", "off", "own", "epa", 1)

# --------------------------------------------------------------------------- custom (non play-filter) metrics
CUSTOM: list[TM] = []


def _c(mid, cat, name, side, direction, formula, classification="calculated", required="drives", adj=False):
    CUSTOM.append(TM(mid, C[cat], name, side, "", "", direction, adj, formula, classification, True, required))


_c("off_ppd", 1, "Points / Drive", "off", 1, "offensive points / offensive drives", adj=True)
_c("off_score_rate", 1, "Scoring Drive Rate", "off", 1, "drives ending TD or FG / drives")
_c("off_3out", 1, "Three-and-Out Rate", "off", -1, "drives of <=3 plays ending in punt / drives")
_c("off_consistency", 1, "Offensive Consistency (game EPA SD)", "off", -1, "SD of game-level EPA/play", required="plays")
_c("def_ppd", 7, "Points / Drive Allowed", "def", -1, "points allowed / opponent drives", adj=True)
_c("def_stop_rate", 7, "Defensive Stop Rate", "def", 1, "opponent drives without points / drives")
_c("def_3out", 7, "Three-and-Outs Forced", "def", 1, "opponent drives <=3 plays ending in punt / drives")
_c("def_consistency", 7, "Defensive Consistency (game EPA SD)", "def", -1, "SD of game-level EPA/play allowed", required="plays")
_c("rz_trips_pg", 13, "Red-Zone Trips / Game", "off", 1, "drives reaching opp 20 / games", required="plays")
_c("rz_td_rate", 13, "Red-Zone TD Rate", "off", 1, "TD drives / drives reaching opp 20", required="plays")
_c("rz_def_td_rate", 13, "Red-Zone TD Rate Allowed", "def", -1, "opp TD drives / opp drives reaching 20", required="plays")
_c("pts_per_opp", 13, "Points / Scoring Opportunity", "off", 1, "points on drives reaching opp 40 / such drives", required="plays")
_c("def_pts_per_opp", 13, "Points / Opportunity Allowed", "def", -1, "opp points on drives reaching our 40 / such drives", required="plays")
_c("st_fg_pct", 14, "Field Goal %", "team", 1, "FG made / attempted", required="plays")
_c("st_fg_voe", 14, "FG Points Over Expected / Game", "team", 1, "sum(3*made - 3*P(make|distance)) / games; P from league logistic fit", required="plays")
_c("st_fg_long", 14, "FG Attempts 40+ Make %", "team", 1, "made / attempts with distance >= 40", required="plays")
_c("st_xp_pct", 14, "Extra Point %", "team", 1, "PAT made / attempted", required="plays")
_c("st_punt_net", 14, "Net Punt Yards", "team", 1, "mean net yards per punt (from line of scrimmage to next spot)", required="plays")
_c("st_ko_tb", 14, "Kickoff Touchback Rate", "team", 0, "touchbacks / kickoffs", required="plays")
_c("st_ko_cover", 14, "Kick Return Yards Allowed", "team", -1, "mean return yards on non-touchback kickoffs", required="plays")
_c("st_kr", 14, "Kick Return Yards", "team", 1, "mean return yards on returned kickoffs received", required="plays")
_c("st_start_fp", 14, "Avg Starting Field Position", "team", -1, "mean yards-to-goal at drive start (lower=better)")
_c("st_opp_start_fp", 14, "Opp Avg Starting Field Position", "team", 1, "mean opp yards-to-goal at drive start (higher=better)")
_c("st_hidden", 14, "Hidden Yards / Game", "team", 1, "(opp start ytg - own start ytg) * drives/game / 2")
_c("cc_go_rate", 15, "4th-Down Go Rate", "team", 0, "4th-down plays run as scrimmage / 4th downs", required="plays")
_c("cc_go_conv", 15, "4th-Down Conversion Rate", "team", 1, "4th-down conversions / attempts", required="plays")
_c("cc_go_short", 15, "4th & <=2 Go Rate (opp 50-35)", "team", 0, "aggressiveness in the decision band", required="plays")
_c("cc_proe", 15, "Pass Rate Over Expected", "team", 0, "mean(dropback - P(dropback|down,dist,ytg,margin,period)) on non-garbage plays", required="plays")
_c("cc_pace", 15, "Neutral Pace (sec / play)", "team", 0, "mean game-clock seconds between snaps in neutral situations", required="plays")
_c("cc_plays_pg", 15, "Offensive Plays / Game", "team", 0, "offensive plays / games", required="plays")
_c("cc_predict", 15, "Play-Call Predictability", "team", -1, "1 - mean binary entropy of run/pass by down-distance bucket", required="plays")
_c("cc_halftime", 15, "Halftime Adjustment (2H-1H EPA)", "team", 1, "EPA/play 2nd half - 1st half (offense+defense net)", "estimated", required="plays")
_c("cc_late_go", 15, "Late-Game 4th-Down Go Rate", "team", 0, "4th-down go rate in Q4 within 8 points", required="plays")
_c("cc_two_pt", 15, "Two-Point Attempt Rate", "team", 0, "2-pt tries / TDs", required="plays")
_c("ro_qb", 16, "QB Rating (QB1 adj. EPA/db)", "team", 1, "QB1 opponent-adjusted EPA per dropback (shrunk)", required="play_events")
_c("ro_rb", 16, "Rushing Rating (lead RB EPA/carry)", "team", 1, "lead rusher EPA/carry, shrunk to league", required="play_events")
_c("ro_rec", 16, "Receiving Rating (EPA/target)", "team", 1, "target-weighted EPA/target of top-4 receivers", required="play_events")
_c("ro_ol", 16, "OL Rating (estimated)", "team", 1, "z(-sack rate allowed) + z(-stuff rate) — play-derived proxy", "estimated", required="plays")
_c("ro_front", 16, "Defensive Front Rating", "team", 1, "z(sack rate) + z(stuff rate) + z(-rush EPA allowed)", "estimated", required="plays")
_c("ro_secondary", 16, "Secondary Rating", "team", 1, "z(-pass EPA allowed) + z(INT rate)", "estimated", required="plays")
_c("ro_avail", 16, "Starter Availability", "team", 1, "1 - starter-games missed / starter-games", required="injuries")
_c("ro_continuity", 16, "Roster Continuity", "team", 1, "returning production share", required="team_seasons")
_c("ro_depth", 16, "Depth Quality", "team", 1, "mean recruiting rating of 2nd-string players", required="recruiting")
_c("ro_experience", 16, "Starter Experience (class yr)", "team", 1, "mean class year of depth-chart starters", required="rosters")
_c("rec_talent", 17, "Talent Composite", "team", 1, "mean recruiting rating of top-25 rostered players x1000", required="team_seasons")
_c("rec_stars", 17, "Roster Avg Stars", "team", 1, "mean recruiting stars of roster", required="recruiting")
_c("rec_bluechip", 17, "Blue-Chip Ratio", "team", 1, "4/5-star share of roster", required="recruiting")
_c("rec_class4", 17, "4-Year Class Rating", "team", 1, "mean recruit rating of the last four signing classes", required="recruiting")
_c("rec_tr_in", 17, "Transfer Additions", "team", 1, "incoming transfers this season", required="transfers")
_c("rec_tr_in_rating", 17, "Transfer Additions Avg Rating", "team", 1, "mean rating of incoming transfers", required="transfers")
_c("rec_tr_out", 17, "Transfer Departures", "team", -1, "outgoing transfers this season", required="transfers")
_c("rec_returning", 17, "Returning Production", "team", 1, "share of prior-season production returning", required="team_seasons")
_c("rec_dev_adj", 17, "Development-Adjusted Talent", "team", 1, "adj. net EPA/play minus talent-predicted net EPA (OLS on season)", "estimated", required="plays")
_c("rec_conversion", 17, "Talent Conversion Efficiency", "team", 1, "adj. net EPA percentile - talent percentile", "estimated", required="plays")
_c("env_rest", 18, "Avg Rest Days", "team", 1, "mean days since previous game", required="games")
_c("env_elev", 18, "Home Elevation (m)", "team", 0, "venue elevation", required="venues")
_c("tr_l3_off", 19, "Last 3 Games EPA (Off)", "off", 1, "EPA/play over last 3 games", required="plays")
_c("tr_l5_off", 19, "Last 5 Games EPA (Off)", "off", 1, "EPA/play over last 5 games", required="plays")
_c("tr_l3_def", 19, "Last 3 Games EPA (Def)", "def", -1, "EPA/play allowed over last 3 games", required="plays")
_c("tr_l5_def", 19, "Last 5 Games EPA (Def)", "def", -1, "EPA/play allowed over last 5 games", required="plays")
_c("tr_l3_sr", 19, "Rolling Success Rate (L3)", "off", 1, "success rate over last 3 games", required="plays")
_c("tr_l3_expl", 19, "Rolling Explosive Rate (L3)", "off", 1, "explosive rate over last 3 games", required="plays")
_c("tr_slope", 19, "Net EPA Trend (slope / game)", "team", 1, "OLS slope of game net EPA/play on game index", required="plays")
_c("tr_to_luck", 19, "Turnover Margin vs Expected / Game", "team", 0, "(takeaways - giveaways) - expected from league rates x opportunities", required="plays")
_c("tr_pyth", 19, "Pythagorean Win %", "team", 1, "PF^2.37 / (PF^2.37 + PA^2.37)", required="games")
_c("tr_luck", 19, "Actual - Expected Wins", "team", 0, "wins - games x pythagorean win%", required="games")
_c("tr_var", 19, "Game Margin SD", "team", 0, "SD of game scoring margin", required="games")
_c("tr_shrink", 19, "Regression-to-Mean Weight", "team", 0, "n/(n+k): weight on observed EPA vs prior (k=350 plays)", required="plays")
_c("rec_wins", 19, "Wins", "team", 1, "games won", required="games")
_c("rec_losses", 19, "Losses", "team", -1, "games lost", required="games")
_c("ppg", 1, "Points / Game", "off", 1, "points scored / games", required="games")
_c("papg", 7, "Points Allowed / Game", "def", -1, "points allowed / games", required="games")
_c("net_epa", 19, "Net EPA / Play (adj.)", "team", 1, "adj. offensive EPA/play - adj. defensive EPA/play allowed", required="plays")
_c("sos", 19, "Strength of Schedule", "team", 1, "mean opponent adj. net EPA/play", required="plays")

# --------------------------------------------------------------------------- unavailable (charting) metrics
UNAVAIL: list[TM] = []


def _u(mid, cat, name, direction, required):
    UNAVAIL.append(TM(mid, C[cat], name, "team", "", "", direction, False,
                      f"requires {required}", "unavailable_without_charting", True, required))


for mid, cat, name, d, req in [
    ("ol_pbwr", 4, "Pass-Block Win Rate", 1, "charting: pass-block win/loss per rep"),
    ("ol_rbwr", 4, "Run-Block Win Rate", 1, "charting: run-block win/loss per rep"),
    ("ol_int_press", 4, "Interior Pressure Allowed", -1, "charting: pressure source"),
    ("ol_edge_press", 4, "Edge Pressure Allowed", -1, "charting: pressure source"),
    ("ol_blitz_pickup", 4, "Blitz Pickup Rate", 1, "charting: blitz + pressure"),
    ("ol_stunt_pickup", 4, "Stunt Pickup Rate", 1, "charting: stunts"),
    ("ol_continuity", 4, "OL Starting-Five Continuity", 1, "participation / snap data"),
    ("ol_pressure_rate", 4, "Pressure Rate Allowed", -1, "charting: pressures"),
    ("rec_yprr", 5, "Yards / Route Run", 1, "participation: routes"),
    ("rec_tprr", 5, "Targets / Route Run", 1, "participation: routes"),
    ("rec_drop", 5, "Drop Rate", -1, "charting: drops"),
    ("rec_sep", 5, "Average Separation", 1, "tracking data"),
    ("rec_manzone", 5, "Man vs Zone Receiving EPA", 1, "charting: coverage type"),
    ("rec_yacoe", 5, "YAC Over Expected", 1, "tracking data"),
    ("sch_personnel", 6, "Personnel Usage (11/12/10)", 0, "charting: personnel"),
    ("sch_formation", 6, "Formation Usage", 0, "charting: formation"),
    ("sch_shotgun", 6, "Shotgun Frequency", 0, "charting: QB alignment"),
    ("sch_under_center", 6, "Under-Center Frequency", 0, "charting: QB alignment"),
    ("sch_pistol", 6, "Pistol Frequency", 0, "charting: QB alignment"),
    ("sch_motion", 6, "Motion Frequency", 0, "charting: pre-snap motion"),
    ("sch_pa", 6, "Play-Action Frequency", 0, "charting: play-action"),
    ("sch_rpo", 6, "RPO Usage", 0, "charting: RPO"),
    ("sch_proe_early", 6, "Formation-Adjusted Efficiency", 1, "charting: formation"),
    ("pdef_cov_success", 8, "Coverage Success Rate", 1, "charting: coverage assignments"),
    ("pdef_pbu", 8, "Pass Breakup Rate", 1, "charting/box: PBU per target (available via player events when present)"),
    ("rdef_ybc", 9, "Yards Before Contact Allowed", -1, "tracking/charting: first contact"),
    ("rdef_mtr", 9, "Missed Tackle Rate", -1, "charting: missed tackles"),
    ("rdef_gap", 9, "Gap Integrity", 1, "charting: gap assignments"),
    ("rdef_concept", 9, "Run Concept Defense (zone/gap)", 1, "charting: run concept"),
    ("pr_pressure", 10, "Pressure Rate", 1, "charting: pressures"),
    ("pr_quick", 10, "Quick Pressure Rate (<2.5s)", 1, "tracking: time to pressure"),
    ("pr_prwr", 10, "Pass-Rush Win Rate", 1, "charting: pass-rush wins"),
    ("pr_ttp", 10, "Time to Pressure", -1, "tracking"),
    ("pr_four_man", 10, "4-Man Pressure Rate", 1, "charting: rushers + pressure"),
    ("pr_blitz", 10, "Blitz Pressure Rate", 1, "charting: blitz + pressure"),
    ("pr_interior", 10, "Interior Pressure", 1, "charting: pressure source"),
    ("pr_edge", 10, "Edge Pressure", 1, "charting: pressure source"),
    ("cov_man", 11, "Man Coverage Rate", 0, "charting: coverage"),
    ("cov_zone", 11, "Zone Coverage Rate", 0, "charting: coverage"),
    ("cov_c0", 11, "Cover 0 Frequency", 0, "charting: coverage shell"),
    ("cov_c1", 11, "Cover 1 Frequency", 0, "charting: coverage shell"),
    ("cov_c2", 11, "Cover 2 Frequency", 0, "charting: coverage shell"),
    ("cov_c3", 11, "Cover 3 Frequency", 0, "charting: coverage shell"),
    ("cov_c4", 11, "Cover 4 Frequency", 0, "charting: coverage shell"),
    ("cov_disguise", 11, "Coverage Disguise Rate", 1, "charting: pre/post-snap shell"),
    ("cov_scheme_epa", 11, "Scheme-Adjusted EPA", -1, "charting: coverage"),
    ("cov_predict", 11, "Coverage Predictability", -1, "charting: coverage"),
    ("cc_timeouts", 15, "Timeout Management Value", 1, "charting: timeout context"),
    ("cc_blitz_tend", 15, "Blitz Tendency by Situation", 0, "charting: blitz"),
    ("cc_form_tend", 15, "Formation Tendencies", 0, "charting: formation"),
    ("cc_opp_scheme", 15, "Opponent-Specific Scheme Change", 0, "charting: formation/coverage"),
    ("env_travel", 18, "Travel Distance Effect", 0, "venue coordinates"),
    ("env_tz", 18, "Time-Zone Adjustment", 0, "venue time zones"),
    ("env_surface", 18, "Field Surface Split", 0, "venue surface + split sample"),
    ("tr_fum_rec", 19, "Fumble Recovery Rate", 0, "all fumbles incl. recovered (play-by-play fumble events)"),
]:
    _u(mid, cat, name, d, req)

# matchup intelligence (game-specific; computed by cfbquant.models.matchup)
MATCHUP = [TM(m, C[20], n, "matchup", "", "", 1, False, f, "calculated", True, "plays") for m, n, f in [
    ("mu_pass", "Pass Offense vs Pass Defense", "adj pass EPA/db (off) + adj pass EPA/db allowed (def) - league"),
    ("mu_rush", "Rush Offense vs Rush Defense", "adj rush EPA/carry + adj rush EPA allowed - league"),
    ("mu_protect", "Pass Rush vs Protection", "log5(sack rate allowed, sack rate generated)"),
    ("mu_expl", "Explosive Offense vs Prevention", "adj explosive rate + allowed - league"),
    ("mu_rz", "Red-Zone Offense vs Defense", "RZ EPA + RZ EPA allowed - league"),
    ("mu_st", "Special Teams Edge", "FG POE + hidden yards (points)"),
    ("mu_qb_press", "QB Pressure Response vs Pass Rush", "QB sack-avoidance x opponent sack rate"),
    ("mu_rec_cov", "Receiving Corps vs Secondary", "EPA/target vs EPA/att allowed"),
    ("mu_avail", "Player Availability Effect", "simulated points change from injured starters"),
    ("mu_xepa", "Matchup Expected EPA / Play", "weighted sum of unit edges by play mix"),
    ("mu_d3", "Third-Down Offense vs Defense", "3rd-down SR vs SR allowed"),
    ("mu_ol_dl", "OL vs DL (run)", "stuff rate allowed vs generated (log5)"),
]]

ALL_TEAM_METRICS = PLAY + CUSTOM + UNAVAIL + MATCHUP


def registry_rows() -> list[dict]:
    return [dict(metric_id=m.mid, scope="team", category=m.cat, name=m.name, positions="TEAM", formula=m.formula,
                 units=None, denominator=m.filt or None, direction=m.direction, classification=m.classification,
                 required_data=m.required, min_sample=None, opponent_adjustable=int(m.adj))
            for m in ALL_TEAM_METRICS]


# --------------------------------------------------------------------------- opponent adjustment
FCS_TEAMS: set = set()   # set per computation from team_seasons.division


def ridge_adjust(g: pd.DataFrame, lam: float = 4.0) -> tuple[dict, dict, float]:
    """g columns: off, def, home(+1/-1/0), value, w. Returns (off_adj, def_adj, mu).
    FCS teams (FCS_TEAMS) get an unpenalised division effect so they shrink toward the FCS level."""
    g = g.dropna(subset=["value"])
    if g.empty:
        return {}, {}, np.nan
    if FCS_TEAMS and (g["off"].isin(FCS_TEAMS).any() or g["def"].isin(FCS_TEAMS).any()):
        from .team_params import ridge_prior
        o, d, mu = ridge_prior(g.assign(w=g["w"] / g["w"].mean()), lam, fcs=FCS_TEAMS)
        return {t: mu + v for t, v in o.items()}, {t: mu + v for t, v in d.items()}, mu
    teams = sorted(set(g["off"]).union(g["def"]))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(g), 2 * n + 1))
    r = np.arange(len(g))
    X[r, g["off"].map(ix).values] = 1
    X[r, n + g["def"].map(ix).values] = 1
    X[:, -1] = g["home"].values
    w = g["w"].values.astype(float)
    w = w / w.mean()
    mu = np.average(g["value"].values, weights=w)
    y = g["value"].values - mu
    # penalty on team effects only; HFA lightly penalised
    pen = np.full(2 * n + 1, lam)
    pen[-1] = 0.1
    A = X.T @ (X * w[:, None]) + np.diag(pen)
    b = X.T @ (y * w)
    beta = np.linalg.solve(A, b)
    return ({t: mu + beta[ix[t]] for t in teams}, {t: mu + beta[n + ix[t]] for t in teams}, mu)


# --------------------------------------------------------------------------- computation
def _pctl(values: pd.Series, direction: int) -> pd.Series:
    v = values.astype(float)
    if direction == 0:
        return v.rank(pct=True) * 100
    return (v * direction).rank(pct=True) * 100


def compute_team_metrics(conn, season: int, through_week: int | None = None) -> pd.DataFrame:
    """Long frame: team_id, metric_id, value, sample, adj_value, status."""
    if through_week is None:
        through_week = int(conn.execute("SELECT MAX(week) FROM games WHERE season=? AND completed=1",
                                        (season,)).fetchone()[0] or 0)
    p = load_plays(conn, season, through_week)
    p = p[p.season == season]
    teams = pd.read_sql_query("SELECT team_id, conference_id, division FROM team_seasons WHERE season=?", conn, params=(season,))
    global FCS_TEAMS
    FCS_TEAMS = set(teams.team_id[teams.division == "fcs"])
    teams = teams[teams.division != "fcs"][["team_id", "conference_id"]]   # profiles/percentiles: FBS only
    tids = teams.team_id.tolist()
    rows: list[dict] = []
    has_air = p.air_yards.notna().any() if not p.empty else False
    has_weather = p.temperature_f.notna().any() if not p.empty else False
    games_pl = p.groupby("offense_id").game_id.nunique() if not p.empty else pd.Series(dtype=float)

    # ---- play-level specs
    adj_cache: dict[tuple, tuple] = {}
    for m in PLAY:
        if m.stat in NEEDS_AIR and not has_air or (m.required == "air_yards" and not has_air):
            for t in tids:
                rows.append(dict(team_id=t, metric_id=m.mid, value=None, sample=0, adj_value=None,
                                 status="unavailable"))
            continue
        if m.required == "weather" and not has_weather:
            for t in tids:
                rows.append(dict(team_id=t, metric_id=m.mid, value=None, sample=0, adj_value=None, status="unavailable"))
            continue
        base = p[p.is_off & (p.clean if m.clean else True)]
        sub = base[mask(base, m.filt)]
        key = "offense_id" if m.side == "off" else "defense_id"
        vals = STAT[m.stat](sub)
        agg = vals.groupby(sub[key]).agg(["mean", "count"])
        adj_off = adj_def = None
        if m.adj:
            ck = (m.filt, m.stat)
            if ck not in adj_cache:
                gg = pd.DataFrame({"off": sub.offense_id, "def": sub.defense_id, "gid": sub.game_id,
                                   "home": np.where(sub.neutral_site == 1, 0, np.where(sub.off_home, 1, -1)),
                                   "v": vals})
                gl = gg.groupby(["gid", "off", "def", "home"]).v.agg(["mean", "count"]).reset_index()
                gl = gl.rename(columns={"mean": "value", "count": "w"})
                adj_cache[ck] = ridge_adjust(gl)
            adj_off, adj_def, _ = adj_cache[ck]
        for t in tids:
            if t in agg.index and agg.loc[t, "count"] > 0:
                v, n = agg.loc[t, "mean"], int(agg.loc[t, "count"])
                a = (adj_off if m.side == "off" else adj_def).get(t) if m.adj else None
                rows.append(dict(team_id=t, metric_id=m.mid, value=float(v), sample=n,
                                 adj_value=None if a is None else float(a),
                                 status="ok" if n >= 20 else "low_sample"))
            else:
                rows.append(dict(team_id=t, metric_id=m.mid, value=None, sample=0, adj_value=None, status="no_data"))

    # ---- custom metrics
    cv = _custom(conn, season, through_week, p, teams, adj_cache)
    for m in CUSTOM:
        d = cv.get(m.mid, {})
        for t in tids:
            v = d.get(t)
            if isinstance(v, tuple):
                v, a = v
            else:
                a = None
            ok = v is not None and not (isinstance(v, float) and np.isnan(v))
            rows.append(dict(team_id=t, metric_id=m.mid, value=float(v) if ok else None, sample=None,
                             adj_value=None if a is None or (isinstance(a, float) and np.isnan(a)) else float(a),
                             status="ok" if ok else "no_data"))
    for m in UNAVAIL:
        for t in tids:
            rows.append(dict(team_id=t, metric_id=m.mid, value=None, sample=0, adj_value=None, status="unavailable"))

    out = pd.DataFrame(rows)
    out["season"] = season
    out["through_week"] = through_week
    # percentiles (direction-aware; use adjusted value when available)
    dirs = {m.mid: m.direction for m in ALL_TEAM_METRICS}
    conf = teams.set_index("team_id").conference_id
    out["conf"] = out.team_id.map(conf)
    basis = out.adj_value.where(out.adj_value.notna(), out.value)
    out["_b"] = basis
    out["pctl_fbs"] = out.groupby("metric_id")["_b"].transform(lambda s: _pctl(s, dirs.get(s.name, 1)))
    out["pctl_conf"] = out.groupby(["metric_id", "conf"])["_b"].transform(lambda s: _pctl(s, dirs.get(s.name[0] if isinstance(s.name, tuple) else s.name, 1)))
    return out.drop(columns=["conf", "_b"])


def _custom(conn, season, wk, p, teams, adj_cache) -> dict[str, dict]:
    res: dict[str, dict] = {}
    tids = teams.team_id.tolist()
    g = pd.read_sql_query("SELECT * FROM games WHERE season=? AND week<=? AND completed=1", conn, params=(season, wk))
    dr = pd.read_sql_query("""SELECT d.* FROM drives d JOIN games g ON g.game_id=d.game_id
                               WHERE g.season=? AND g.week<=?""", conn, params=(season, wk))
    # ---- game results
    long = pd.concat([
        pd.DataFrame({"team": g.home_id, "pf": g.home_points, "pa": g.away_points, "date": g.start_date, "gid": g.game_id}),
        pd.DataFrame({"team": g.away_id, "pf": g.away_points, "pa": g.home_points, "date": g.start_date, "gid": g.game_id})])
    long["win"] = (long.pf > long.pa).astype(int)
    agg = long.groupby("team").agg(gp=("pf", "size"), pf=("pf", "sum"), pa=("pa", "sum"), w=("win", "sum"),
                                   msd=("pf", lambda s: np.nan))
    margins = (long.pf - long.pa).groupby(long.team).std()
    res["ppg"] = (agg.pf / agg.gp).to_dict()
    res["papg"] = (agg.pa / agg.gp).to_dict()
    res["rec_wins"] = agg.w.to_dict()
    res["rec_losses"] = (agg.gp - agg.w).to_dict()
    pyth = agg.pf ** 2.37 / (agg.pf ** 2.37 + agg.pa ** 2.37)
    res["tr_pyth"] = pyth.to_dict()
    res["tr_luck"] = (agg.w - agg.gp * pyth).to_dict()
    res["tr_var"] = margins.to_dict()
    long = long.sort_values("date")
    long["dt"] = pd.to_datetime(long.date)
    rest = long.groupby("team").dt.diff().dt.days
    res["env_rest"] = rest.groupby(long.team).mean().to_dict()
    # ---- drives
    if not dr.empty:
        dr["scored"] = dr.result.isin(["TD", "FG"])
        dr["three_out"] = (dr.plays <= 3) & (dr.result == "PUNT")
        o = dr.groupby("offense_id")
        d = dr.groupby("defense_id")
        ppd = o.points.mean()
        # opponent-adjusted points/drive
        gl = dr.merge(g[["game_id", "home_id", "neutral_site"]], on="game_id")
        gl = gl.groupby(["game_id", "offense_id", "defense_id", "home_id", "neutral_site"]).points.agg(["mean", "count"]).reset_index()
        gl["home"] = np.where(gl.neutral_site == 1, 0, np.where(gl.offense_id == gl.home_id, 1, -1))
        ao, ad, _ = ridge_adjust(gl.rename(columns={"offense_id": "off", "defense_id": "def", "mean": "value", "count": "w"}))
        res["off_ppd"] = {t: (ppd.get(t), ao.get(t)) for t in ppd.index}
        dppd = d.points.mean()
        res["def_ppd"] = {t: (dppd.get(t), ad.get(t)) for t in dppd.index}
        res["off_score_rate"] = o.scored.mean().to_dict()
        res["off_3out"] = o.three_out.mean().to_dict()
        res["def_stop_rate"] = (1 - d.scored.mean()).to_dict()
        res["def_3out"] = d.three_out.mean().to_dict()
        sfp = o.start_yards_to_goal.mean()
        osfp = d.start_yards_to_goal.mean()
        res["st_start_fp"] = sfp.to_dict()
        res["st_opp_start_fp"] = osfp.to_dict()
        dpg = o.size() / agg.gp.reindex(o.size().index)
        res["st_hidden"] = ((osfp - sfp) * dpg / 2).to_dict()
    if p.empty:
        return res
    off = p[p.is_off]
    clean = off[off.clean]
    # ---- consistency / trends from game-level EPA
    ge_o = clean.groupby(["offense_id", "game_id"]).agg(epa=("epa", "mean"), sr=("success", "mean"),
                                                        ex=("explosive", "mean"), n=("epa", "size"),
                                                        wk=("week", "first")).reset_index()
    ge_d = clean.groupby(["defense_id", "game_id"]).agg(epa=("epa", "mean"), n=("epa", "size"),
                                                        wk=("week", "first")).reset_index()
    res["off_consistency"] = ge_o.groupby("offense_id").epa.std().to_dict()
    res["def_consistency"] = ge_d.groupby("defense_id").epa.std().to_dict()

    def lastn(ge, key, col, n):
        return ge.sort_values("wk").groupby(key).tail(n).groupby(key)[col].mean().to_dict()
    res["tr_l3_off"] = lastn(ge_o, "offense_id", "epa", 3)
    res["tr_l5_off"] = lastn(ge_o, "offense_id", "epa", 5)
    res["tr_l3_def"] = lastn(ge_d, "defense_id", "epa", 3)
    res["tr_l5_def"] = lastn(ge_d, "defense_id", "epa", 5)
    res["tr_l3_sr"] = lastn(ge_o, "offense_id", "sr", 3)
    res["tr_l3_expl"] = lastn(ge_o, "offense_id", "ex", 3)
    net = ge_o.rename(columns={"offense_id": "team"})[["team", "game_id", "epa", "wk"]].merge(
        ge_d.rename(columns={"defense_id": "team", "epa": "depa"})[["team", "game_id", "depa"]], on=["team", "game_id"])
    net["net"] = net.epa - net.depa
    res["tr_slope"] = net.groupby("team").apply(
        lambda d: float(np.polyfit(np.arange(len(d)), d.sort_values("wk").net, 1)[0]) if len(d) >= 3 else np.nan,
        include_groups=False).to_dict()
    nplays = clean.groupby("offense_id").size()
    res["tr_shrink"] = (nplays / (nplays + 350)).to_dict()
    # adjusted net EPA + SOS
    if ("all", "epa") in adj_cache:
        ao, ad, mu = adj_cache[("all", "epa")]
        netadj = {t: ao.get(t, mu) - ad.get(t, mu) for t in tids}
        res["net_epa"] = netadj
        opp = pd.concat([pd.DataFrame({"t": g.home_id, "o": g.away_id}), pd.DataFrame({"t": g.away_id, "o": g.home_id})])
        opp["on"] = opp.o.map(netadj)
        res["sos"] = opp.groupby("t").on.mean().to_dict()
        # talent relationships
        ts = pd.read_sql_query("SELECT team_id, talent_composite FROM team_seasons WHERE season=?", conn, params=(season,))
        ts["net"] = ts.team_id.map(netadj)
        ts = ts.dropna()
        if len(ts) > 5 and ts.talent_composite.std() > 0:
            b = np.polyfit(ts.talent_composite, ts.net, 1)
            res["rec_dev_adj"] = dict(zip(ts.team_id, ts.net - np.polyval(b, ts.talent_composite)))
            res["rec_conversion"] = dict(zip(ts.team_id, ts.net.rank(pct=True) * 100 - ts.talent_composite.rank(pct=True) * 100))
    # ---- red zone & opportunities (drive-level from plays)
    dv = off.groupby(["drive_id", "offense_id", "defense_id"]).agg(miny=("yards_to_goal", "min")).reset_index()
    if not dr.empty:
        dv = dv.merge(dr[["drive_id", "result", "points"]], on="drive_id", how="left")
        rz = dv[dv.miny <= 20]
        res["rz_trips_pg"] = (rz.groupby("offense_id").size() / agg.gp).dropna().to_dict()
        res["rz_td_rate"] = (rz.result == "TD").groupby(rz.offense_id).mean().to_dict()
        res["rz_def_td_rate"] = (rz.result == "TD").groupby(rz.defense_id).mean().to_dict()
        op = dv[dv.miny <= 40]
        res["pts_per_opp"] = op.groupby("offense_id").points.mean().to_dict()
        res["def_pts_per_opp"] = op.groupby("defense_id").points.mean().to_dict()
    # ---- special teams
    fg = p[p.play_type == "field_goal"].copy()
    if not fg.empty:
        from sklearn.linear_model import LogisticRegression
        fg = fg.dropna(subset=["fg_distance"])
        if fg.fg_made.nunique() > 1:
            lr = LogisticRegression().fit(fg[["fg_distance"]], fg.fg_made.astype(int))
            fg["pexp"] = lr.predict_proba(fg[["fg_distance"]])[:, 1]
        else:
            fg["pexp"] = fg.fg_made.mean()
        fg["voe"] = 3 * (fg.fg_made - fg.pexp)
        res["st_fg_pct"] = fg.groupby("offense_id").fg_made.mean().to_dict()
        res["st_fg_voe"] = (fg.groupby("offense_id").voe.sum() / agg.gp).dropna().to_dict()
        lf = fg[fg.fg_distance >= 40]
        res["st_fg_long"] = lf.groupby("offense_id").fg_made.mean().to_dict()
    pat = p[p.play_type == "pat"]
    if not pat.empty:
        res["st_xp_pct"] = (pat.points > 0).groupby(pat.offense_id).mean().to_dict()
    pu = p[p.play_type == "punt"]
    if not pu.empty:
        res["st_punt_net"] = pu.groupby("offense_id").yards_gained.mean().to_dict()
    ko = p[p.play_type == "kickoff"]
    if not ko.empty:
        tb = ko.yards_gained == 0
        res["st_ko_tb"] = tb.groupby(ko.offense_id).mean().to_dict()
        ret = ko[~tb]
        res["st_ko_cover"] = ret.groupby("offense_id").yards_gained.mean().to_dict()
        res["st_kr"] = ret.groupby("defense_id").yards_gained.mean().to_dict()
    # ---- coaching
    f4 = p[(p.down == 4) & p.play_type.isin(["pass", "rush", "sack", "scramble", "punt", "field_goal", "penalty"])
           & (p.play_type != "penalty")]
    if not f4.empty:
        go = f4.is_off
        res["cc_go_rate"] = go.groupby(f4.offense_id).mean().to_dict()
        att = f4[go]
        conv = (att.first_down == 1) | (att.touchdown == 1)
        res["cc_go_conv"] = conv.groupby(att.offense_id).mean().to_dict()
        band = f4[(f4.distance <= 2) & f4.yards_to_goal.between(35, 50)]
        res["cc_go_short"] = band.is_off.groupby(band.offense_id).mean().to_dict()
        lt = f4[(f4.period == 4) & (f4.margin.abs() <= 8)]
        res["cc_late_go"] = lt.is_off.groupby(lt.offense_id).mean().to_dict()
    if len(clean) > 200:
        from sklearn.linear_model import LogisticRegression
        X = np.column_stack([clean.down == 1, clean.down == 2, clean.down == 3, np.log1p(clean.distance.clip(lower=1)),
                             clean.yards_to_goal / 100, clean.margin.clip(-28, 28) / 14, clean.period >= 3,
                             (clean.margin.clip(-28, 28) / 14) * (clean.period == 4)]).astype(float)
        lr = LogisticRegression(max_iter=500).fit(X, clean.dropback.astype(int))
        resid = clean.dropback.astype(float) - lr.predict_proba(X)[:, 1]
        res["cc_proe"] = resid.groupby(clean.offense_id).mean().to_dict()
        # predictability: entropy of run/pass by down-distance bucket
        b = clean.assign(bk=clean.down.astype(str) + "_" + pd.cut(clean.distance, [0, 3, 6, 10, 99], labels=False).astype(str))
        pr = b.groupby(["offense_id", "bk"]).dropback.agg(["mean", "size"]).reset_index()
        pm = pr["mean"].clip(1e-3, 1 - 1e-3)
        pr["H"] = -(pm * np.log2(pm) + (1 - pm) * np.log2(1 - pm))
        res["cc_predict"] = pr.groupby("offense_id").apply(lambda d: 1 - np.average(d.H, weights=d["size"]),
                                                           include_groups=False).to_dict()
    res["cc_plays_pg"] = (off.groupby("offense_id").size() / games_pl_or(off)).to_dict()
    pc = p[p.is_off & (p.period <= 3) & (p.margin.abs() <= 14)].sort_values(["game_id", "play_number"])
    if pc.clock_seconds.notna().any():
        pc = pc.assign(dt=pc.groupby(["game_id", "period"]).clock_seconds.diff(-1))
        pc = pc[(pc.dt > 0) & (pc.dt < 60) & (pc.offense_id == pc.groupby(["game_id", "period"]).offense_id.shift(-1))]
        res["cc_pace"] = pc.groupby("offense_id").dt.mean().to_dict()
    h = clean.assign(h2=clean.period >= 3)
    ho = h.groupby(["offense_id", "h2"]).epa.mean().unstack()
    hd = h.groupby(["defense_id", "h2"]).epa.mean().unstack()
    if ho.shape[1] == 2 and hd.shape[1] == 2:
        res["cc_halftime"] = ((ho[True] - ho[False]) - (hd[True] - hd[False])).to_dict()
    tp = p[p.play_type == "two_point"].groupby("offense_id").size()
    tds = off[off.touchdown == 1].groupby("offense_id").size()
    res["cc_two_pt"] = (tp.reindex(tds.index).fillna(0) / tds).to_dict()
    # turnover luck: realised margin vs expected from league rates x opportunities
    lg_int = off[off.att].interception.mean()
    lg_fum = off.fumble_lost.mean()
    exp_give = off[off.att].groupby("offense_id").size() * lg_int + off.groupby("offense_id").size() * lg_fum
    exp_take = off[off.att].groupby("defense_id").size() * lg_int + off.groupby("defense_id").size() * lg_fum
    giv = off.groupby("offense_id").turnover.sum()
    tak = off.groupby("defense_id").turnover.sum()
    res["tr_to_luck"] = (((tak - giv) - (exp_take - exp_give)) / agg.gp).dropna().to_dict()
    # ---- roster / talent / personnel
    res.update(_roster_metrics(conn, season, wk, p, off, clean, tids))
    return res


def games_pl_or(off):
    return off.groupby("offense_id").game_id.nunique()


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() or 1)


def _roster_metrics(conn, season, wk, p, off, clean, tids) -> dict[str, dict]:
    res: dict[str, dict] = {}
    ts = pd.read_sql_query("SELECT * FROM team_seasons WHERE season=?", conn, params=(season,)).set_index("team_id")
    res["rec_talent"] = ts.talent_composite.dropna().to_dict()
    res["rec_returning"] = ts.returning_production.dropna().to_dict()
    res["ro_continuity"] = ts.returning_production.dropna().to_dict()
    rr = pd.read_sql_query("""SELECT r.team_id, r.player_id, r.class_year, rp.stars, rp.rating FROM rosters r
        LEFT JOIN recruiting_profiles rp ON rp.player_id=r.player_id WHERE r.season=?""", conn, params=(season,))
    if not rr.empty and rr.stars.notna().any():
        res["rec_stars"] = rr.groupby("team_id").stars.mean().to_dict()
        res["rec_bluechip"] = (rr.stars >= 4).groupby(rr.team_id).mean().to_dict()
    cls = pd.read_sql_query("""SELECT committed_team_id AS team_id, AVG(rating) AS r FROM recruiting_profiles
        WHERE class_year BETWEEN ? AND ? GROUP BY 1""", conn, params=(season - 3, season))
    res["rec_class4"] = dict(zip(cls.team_id, cls.r))
    tr = pd.read_sql_query("SELECT * FROM transfer_history WHERE season=?", conn, params=(season,))
    if not tr.empty:
        res["rec_tr_in"] = tr.groupby("to_team_id").size().reindex(tids).fillna(0).to_dict()
        res["rec_tr_in_rating"] = tr.groupby("to_team_id").rating.mean().to_dict()
        res["rec_tr_out"] = tr.groupby("from_team_id").size().reindex(tids).fillna(0).to_dict()
    dc = pd.read_sql_query("""SELECT dc.team_id, dc.player_id, dc.depth, r.class_year, rp.rating FROM depth_charts dc
        LEFT JOIN rosters r ON r.player_id=dc.player_id AND r.season=dc.season
        LEFT JOIN recruiting_profiles rp ON rp.player_id=dc.player_id
        WHERE dc.season=? AND dc.week=0""", conn, params=(season,))
    if not dc.empty:
        st = dc[dc.depth == 1]
        res["ro_experience"] = st.groupby("team_id").class_year.mean().to_dict()
        res["ro_depth"] = dc[dc.depth == 2].groupby("team_id").rating.mean().to_dict()
        inj = pd.read_sql_query("SELECT player_id, COUNT(*) AS wk_out FROM injuries WHERE season=? AND week<=? GROUP BY 1",
                                conn, params=(season, wk))
        st = st.merge(inj, on="player_id", how="left").fillna({"wk_out": 0})
        gp = max(int(wk), 1)
        res["ro_avail"] = (1 - st.groupby("team_id").wk_out.sum() / (st.groupby("team_id").size() * gp)).to_dict()
    elev = pd.read_sql_query("SELECT t.team_id, v.elevation_m FROM teams t JOIN venues v ON v.venue_id=t.venue_id", conn)
    res["env_elev"] = dict(zip(elev.team_id, elev.elevation_m))
    # personnel ratings from play_events
    ev = pd.read_sql_query("""SELECT e.play_id, e.player_id, e.team_id, e.role FROM play_events e
        JOIN plays p ON p.play_id=e.play_id WHERE p.season=? AND p.week<=? AND e.role IN ('passer','rusher','target')""",
                           conn, params=(season, wk))
    if not ev.empty and not clean.empty:
        pe = ev.merge(clean[["play_id", "epa", "dropback", "rush"]], on="play_id")
        qb = pe[pe.role == "passer"].groupby(["team_id", "player_id"]).epa.agg(["sum", "size"]).reset_index()
        qb = qb.sort_values("size").groupby("team_id").tail(1)
        res["ro_qb"] = dict(zip(qb.team_id, qb["sum"] / (qb["size"] + 60)))
        rb = pe[(pe.role == "rusher") & pe.rush].groupby(["team_id", "player_id"]).epa.agg(["sum", "size"]).reset_index()
        rb = rb.sort_values("size").groupby("team_id").tail(1)
        res["ro_rb"] = dict(zip(rb.team_id, rb["sum"] / (rb["size"] + 40)))
        wr = pe[pe.role == "target"].groupby(["team_id", "player_id"]).epa.agg(["sum", "size"]).reset_index()
        wr = wr.sort_values("size").groupby("team_id").tail(4).groupby("team_id")[["sum", "size"]].sum()
        res["ro_rec"] = (wr["sum"] / (wr["size"] + 60)).to_dict()
    db = clean[clean.dropback]
    ru = clean[clean.rush]
    if not db.empty:
        ol = _z(-db.groupby("offense_id").sack.mean()) + _z(-ru.groupby("offense_id").stuff.mean())
        res["ro_ol"] = ol.to_dict()
        fr = _z(db.groupby("defense_id").sack.mean()) + _z(ru.groupby("defense_id").stuff.mean()) + \
            _z(-ru.groupby("defense_id").epa.mean())
        res["ro_front"] = fr.to_dict()
        sec = _z(-db.groupby("defense_id").epa.mean()) + _z(clean[clean.att].groupby("defense_id").interception.mean())
        res["ro_secondary"] = sec.to_dict()
    return res


def write_team_metrics(conn, season: int, through_week: int | None = None) -> int:
    from ..db import insert_df
    out = compute_team_metrics(conn, season, through_week)
    conn.execute("DELETE FROM team_metrics WHERE season=? AND through_week=?", (season, int(out.through_week.iloc[0])))
    n = insert_df(conn, "team_metrics", out[["team_id", "season", "through_week", "metric_id", "value", "sample",
                                             "pctl_fbs", "pctl_conf", "adj_value", "status"]])
    conn.commit()
    return n
