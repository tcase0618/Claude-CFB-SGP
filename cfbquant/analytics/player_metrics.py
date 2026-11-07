"""Universal (80+) and position-specific player intelligence.

Every metric is declared with id, category, positions, formula, direction and
classification:
  observed    - counted directly from source box scores / play attribution
  calculated  - deterministic formula over observed play data
  estimated   - model-based (e.g. completion-probability model) or a proxy
  unavailable_without_charting - needs licensed charting/tracking; never faked

Players are evaluated on *their own* involvement plays (passer dropbacks,
rusher carries, targets, defensive events), not on team aggregates, and the
opponent-adjusted variants subtract the opposing unit's adjusted EPA allowed
on that play family, so a player on a weak team is not penalised for it.

Recruiting-only players (no snaps) get no performance metrics; their talent
prior lives in cfbquant.analytics.talent with a deliberately wide interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from .plays_frame import load_plays
from .team_metrics import ridge_adjust

ALL = "ALL"
OFF_SKILL = "QB,RB,WR,TE"
DEFN = "DL,EDGE,LB,CB,S"


@dataclass(frozen=True)
class PM:
    mid: str
    scope: str          # player_universal | player_position
    cat: str
    name: str
    positions: str
    formula: str
    direction: int = 1
    classification: str = "calculated"
    required: str = "plays+play_events"
    min_sample: int = 0


DEFS: list[PM] = []


def U(mid, cat, name, pos, formula, d=1, cl="calculated", req="plays+play_events", ms=0):
    DEFS.append(PM(mid, "player_universal", cat, name, pos, formula, d, cl, req, ms))


def P(mid, pos, name, formula, d=1, cl="calculated", req="plays+play_events", ms=0):
    DEFS.append(PM(mid, "player_position", pos, name, pos, formula, d, cl, req, ms))


# ---------------- A. identity & physical
A, B, Cc, D, E, Fc, G, H = ("A. Identity & Physical", "B. Opportunity & Usage", "C. Production",
                            "D. Advanced Efficiency", "E. Consistency & Volatility", "F. Situational",
                            "G. Competition Adjustments", "H. Player Impact")
U("age", A, "Age (est.)", ALL, "season - birth_year", 0, "observed", "players")
U("height", A, "Height (in)", ALL, "listed height", 0, "observed", "players")
U("weight", A, "Weight (lb)", ALL, "listed weight", 0, "observed", "players")
U("class_year", A, "Class Year", ALL, "1=FR .. 5=GR", 0, "observed", "rosters")
U("experience", A, "Years on FBS Rosters", ALL, "count of seasons rostered through this season", 0, "observed", "rosters")
U("stars", A, "Recruiting Stars", ALL, "composite stars at signing", 0, "observed", "recruiting")
U("recruit_rating", A, "Recruiting Rating", ALL, "composite rating at signing (0-1)", 0, "observed", "recruiting")
U("starter", A, "Preseason Starter", ALL, "1 if depth-chart #1 at position (week 0)", 0, "observed", "depth_charts")
U("depth", A, "Depth-Chart Rank", ALL, "preseason depth position", -1, "observed", "depth_charts")
U("transfer_in", A, "Transfer This Season", ALL, "1 if arrived via portal this season", 0, "observed", "transfers")
U("athletic_test", A, "Athletic Testing Composite", ALL, "requires verified combine/pro-day testing", 1,
  "unavailable_without_charting", "athletic testing")
U("archetype", A, "Archetype", ALL, "rule-based classification from position metrics (see archetype())", 0, "estimated")
# ---------------- B. usage
U("games", B, "Games Played", ALL, "games with >=1 attributed event", 1, "observed")
U("games_out", B, "Games Missed (injury)", ALL, "weeks listed OUT", -1, "observed", "injuries")
U("plays", B, "Involvement Plays", ALL, "plays with this player as passer/rusher/target or defensive event", 1)
U("touches", B, "Touches", OFF_SKILL, "carries + receptions", 1)
U("touches_pg", B, "Touches / Game", OFF_SKILL, "touches / games", 1)
U("target_share", B, "Target Share", "RB,WR,TE", "targets / team pass attempts in games played", 1)
U("rush_share", B, "Rush Share", "QB,RB,WR", "carries / team carries in games played", 1)
U("dropback_share", B, "Dropback Share", "QB", "dropbacks / team dropbacks in games played", 1)
U("rz_touches", B, "Red-Zone Opportunities", OFF_SKILL, "carries + targets + dropbacks inside opp 20", 1)
U("rz_share", B, "Red-Zone Opportunity Share", "RB,WR,TE", "RZ carries+targets / team RZ plays", 1)
U("d3_usage", B, "Third-Down Usage Share", OFF_SKILL, "share of player's opportunities on 3rd down", 0)
U("opp_quality", B, "Opponent Defense Faced (adj EPA)", OFF_SKILL, "mean adj. EPA/play allowed by opponents faced", -1)
U("snaps", B, "Snaps", ALL, "requires participation data", 1, "unavailable_without_charting", "participation")
U("snap_pct", B, "Snap %", ALL, "requires participation data", 1, "unavailable_without_charting", "participation")
U("routes", B, "Routes Run", "RB,WR,TE", "requires participation data", 1, "unavailable_without_charting", "participation")
# ---------------- C. production
for mid, nm, pos in [("pass_att", "Pass Attempts", "QB"), ("pass_cmp", "Completions", "QB"), ("pass_yds", "Passing Yards", "QB"),
                     ("pass_td", "Passing TD", "QB"), ("pass_int", "Interceptions Thrown", "QB"),
                     ("rush_att", "Carries", "QB,RB,WR"), ("rush_yds", "Rushing Yards", "QB,RB,WR"), ("rush_td", "Rushing TD", "QB,RB,WR"),
                     ("targets", "Targets", "RB,WR,TE"), ("receptions", "Receptions", "RB,WR,TE"), ("rec_yds", "Receiving Yards", "RB,WR,TE"),
                     ("rec_td", "Receiving TD", "RB,WR,TE"), ("tackles", "Tackles", DEFN), ("tfl", "Tackles for Loss", DEFN),
                     ("def_sacks", "Sacks", DEFN), ("def_int", "Interceptions", DEFN), ("pbu", "Pass Breakups", DEFN),
                     ("forced_fumbles", "Forced Fumbles", DEFN), ("fgm", "Field Goals Made", "K"), ("punts", "Punts", "P")]:
    U(mid, Cc, nm, pos, f"sum of {mid} from box scores", -1 if mid == "pass_int" else 1, "observed", "player_games")
U("scrim_yds", Cc, "Scrimmage Yards", OFF_SKILL, "rush yds + rec yds", 1, "observed", "player_games")
U("total_td", Cc, "Total Touchdowns", OFF_SKILL, "pass + rush + rec TD", 1, "observed", "player_games")
U("yds_pg", Cc, "Total Yards / Game", OFF_SKILL, "(pass + rush + rec yds) / games", 1, "observed", "player_games")
U("havoc", Cc, "Havoc Plays", DEFN, "TFL + sacks + INT + PBU + forced fumbles", 1, "observed", "player_games")
# ---------------- D. advanced
U("epa_total", D, "Total EPA", OFF_SKILL, "sum EPA over player's opportunities (dropbacks, carries, targets)", 1)
U("epa_play", D, "EPA / Opportunity", OFF_SKILL, "EPA_total / opportunities (non-garbage)", 1, ms=20)
U("adj_epa_play", D, "Opponent-Adj. EPA / Opportunity", OFF_SKILL,
  "mean(EPA - (opp adj EPA allowed on play family - league))", 1, ms=20)
U("success", D, "Success Rate", OFF_SKILL, "50/70/100 rule over opportunities", 1, ms=20)
U("explosive", D, "Explosive Play Rate", OFF_SKILL, "rush>=12 / catch>=16 yds per opportunity", 1, ms=20)
U("negative", D, "Negative Play Rate", OFF_SKILL, "opportunities losing yards", -1, ms=20)
U("first_down", D, "First-Down Rate", OFF_SKILL, "opportunities gaining a first down", 1, ms=20)
U("epa_early", D, "Early-Down EPA", OFF_SKILL, "EPA/opp on 1st-2nd down", 1, ms=15)
U("epa_pd", D, "Passing-Down EPA", OFF_SKILL, "EPA/opp on passing downs", 1, ms=10)
U("epa_pg", D, "EPA / Game", OFF_SKILL, "EPA_total / games", 1)
U("def_epa_event", D, "EPA on Defensive Events", DEFN, "-mean EPA of plays where player made a tackle/sack/INT/PBU (higher = more value)", 1, "estimated")
# ---------------- E. consistency
U("l3_epa", E, "Last 3 Games EPA / Opp", OFF_SKILL, "mean game EPA/opp, last 3 games", 1)
U("l5_epa", E, "Last 5 Games EPA / Opp", OFF_SKILL, "mean game EPA/opp, last 5 games", 1)
U("l10_epa", E, "Last 10 Games EPA / Opp", OFF_SKILL, "includes prior season when needed", 1)
U("game_sd_epa", E, "Game EPA SD", OFF_SKILL, "SD of game EPA/opp", -1)
U("game_sd_yds", E, "Game Yards SD", OFF_SKILL, "SD of game total yards", 0)
U("yds_cv", E, "Yards Coefficient of Variation", OFF_SKILL, "SD / mean of game yards", -1)
U("yds_p10", E, "Floor (P10 Game Yards)", OFF_SKILL, "10th percentile of game yards", 1)
U("yds_p90", E, "Ceiling (P90 Game Yards)", OFF_SKILL, "90th percentile of game yards", 1)
U("trend", E, "EPA Trend (slope / game)", OFF_SKILL, "OLS slope of game EPA/opp", 1)
U("reliability", E, "Sample Reliability", ALL, "n / (n + k), k=60 opportunities: weight on observed vs prior", 1)
U("epa_se", E, "EPA / Opp Standard Error", OFF_SKILL, "SD(EPA)/sqrt(n)", -1)
U("regressed_epa", E, "Regressed EPA / Opp", OFF_SKILL, "reliability * adj EPA + (1-reliability) * position mean", 1)
# ---------------- F. situational
for mid, nm, filt in [("rz_epa", "Red-Zone EPA / Opp", "rz"), ("d3_sr", "Third-Down Success", "d3"),
                      ("d4_sr", "Fourth-Down Success", "d4"), ("sy_sr", "Short-Yardage Success", "sy"),
                      ("ly_sr", "Long-Yardage (10+) Success", "ly"), ("early_sr", "Early-Down Success", "early"),
                      ("late_epa", "Late & Close EPA", "late"), ("lead_epa", "Leading EPA", "lead"),
                      ("trail_epa", "Trailing EPA", "trail"), ("home_epa", "Home EPA", "home"),
                      ("away_epa", "Away EPA", "away"), ("q4_epa", "4th-Quarter EPA", "q4")]:
    U(mid, Fc, nm, OFF_SKILL, f"metric over {filt} opportunities", 1, ms=5)
# ---------------- G. competition
U("adj_delta", G, "Opponent Adjustment (adj - raw)", OFF_SKILL, "adj EPA/opp - raw EPA/opp (positive = tougher schedule)", 0)
U("conf_strength", G, "Conference Strength", ALL, "mean adj net EPA/play of player's conference", 1)
U("team_off_quality", G, "Supporting Cast (team adj off EPA)", OFF_SKILL, "team adj offensive EPA/play excluding the player's opportunities", 0, "estimated")
U("rel_team", G, "EPA vs Teammates", OFF_SKILL, "player EPA/opp - teammates' EPA/opp on same play family", 1)
U("scheme_proe", G, "Scheme Context (team PROE)", OFF_SKILL, "team pass rate over expected", 0)
U("vs_top_def", G, "EPA vs Top-40% Defenses", OFF_SKILL, "EPA/opp vs opponents in top 40% adj defense", 1, ms=10)
U("matchup_difficulty", G, "Individual Matchup Difficulty", OFF_SKILL, "requires player-vs-defender charting", -1,
  "unavailable_without_charting", "charting: matchups")
# ---------------- H. impact
U("epa_oar", H, "EPA Over Replacement", OFF_SKILL, "sum(EPA) - n * replacement EPA/opp (20th pctl at position)", 1)
U("team_epa_share", H, "Share of Team Offensive EPA", OFF_SKILL, "player EPA_total / team EPA_total (sign-aware)", 1)
U("pts_oar_pg", H, "Points Over Replacement / Game", OFF_SKILL, "epa_oar / games", 1)
U("impact_ci", H, "Impact 80% Interval Half-Width", OFF_SKILL, "1.28 * epa_se * opportunities / games", -1)
U("onoff", H, "On/Off EPA Differential", ALL, "requires snap-level participation", 1, "unavailable_without_charting", "participation")
U("def_impact", H, "Defensive Havoc / Game", DEFN, "havoc plays / games", 1, "observed", "player_games")
U("tackle_share", H, "Team Tackle Share", DEFN, "tackles / team tackles in games played", 1)

# ---------------- position libraries
QB = "QB"
for mid, nm, f, d, cl in [
    ("qb_epa_db", "Passing EPA / Dropback", "mean EPA on dropbacks (pass, sack, scramble)", 1, "calculated"),
    ("qb_sr", "Passing Success Rate", "success on dropbacks", 1, "calculated"),
    ("qb_cpoe", "CPOE", "completion % - expected (logit on air yards, down, distance, ytg)", 1, "estimated"),
    ("qb_comp", "Completion %", "cmp / att", 1, "observed"),
    ("qb_adj_comp", "Adjusted Completion %", "requires drop/throwaway charting", 1, "unavailable_without_charting"),
    ("qb_adot", "Average Depth of Target", "mean air yards / att", 0, "calculated"),
    ("qb_air_ypa", "Air Yards / Attempt", "sum air yards on completions / att", 1, "calculated"),
    ("qb_ypa", "Yards / Attempt", "pass yds / att", 1, "observed"),
    ("qb_anya", "Adj. Net Yards / Attempt", "(yds + 20*TD - 45*INT - sack yds) / (att + sacks)", 1, "calculated"),
    ("qb_btt", "Big-Time Throw Rate", "requires charting", 1, "unavailable_without_charting"),
    ("qb_twp", "Turnover-Worthy Play Rate", "requires charting", -1, "unavailable_without_charting"),
    ("qb_int_rate", "Interception Rate", "INT / att", -1, "observed"),
    ("qb_xint", "Expected Interception Rate", "mean league INT probability given air depth", -1, "estimated"),
    ("qb_pressure", "Pressure Rate Faced", "requires charting", 0, "unavailable_without_charting"),
    ("qb_p2s", "Pressure-to-Sack Rate", "requires charting", -1, "unavailable_without_charting"),
    ("qb_sack_rate", "Sack Rate", "sacks / dropbacks", -1, "observed"),
    ("qb_ttt", "Time to Throw", "requires tracking", 0, "unavailable_without_charting"),
    ("qb_clean", "Clean-Pocket EPA", "requires charting", 1, "unavailable_without_charting"),
    ("qb_press_epa", "Under-Pressure EPA", "requires charting", 1, "unavailable_without_charting"),
    ("qb_blitz", "Blitz EPA", "requires charting", 1, "unavailable_without_charting"),
    ("qb_man", "Man Coverage EPA", "requires charting", 1, "unavailable_without_charting"),
    ("qb_zone", "Zone Coverage EPA", "requires charting", 1, "unavailable_without_charting"),
    ("qb_deep_rate", "Deep Shot Rate", "att with air >= 20 / att", 0, "calculated"),
    ("qb_deep_acc", "Deep Completion %", "cmp / att, air >= 20", 1, "calculated"),
    ("qb_inter_acc", "Intermediate Completion %", "cmp / att, air 10-19", 1, "calculated"),
    ("qb_short_acc", "Short Completion %", "cmp / att, air 0-9", 1, "calculated"),
    ("qb_behind_los", "Behind-LOS Throw Rate", "att with air < 0 / att", 0, "calculated"),
    ("qb_pa", "Play-Action EPA", "requires charting", 1, "unavailable_without_charting"),
    ("qb_rpo", "RPO EPA", "requires charting", 1, "unavailable_without_charting"),
    ("qb_scramble", "Scramble Rate", "scrambles / dropbacks", 0, "calculated"),
    ("qb_designed_run", "Designed-Run Rate", "designed QB carries / (dropbacks + designed carries)", 0, "calculated"),
    ("qb_rush_epa", "QB Rushing EPA / Carry", "mean EPA on QB carries (designed + scramble)", 1, "calculated"),
    ("qb_rz_epa", "Red-Zone Passing EPA", "EPA / dropback inside 20", 1, "calculated"),
    ("qb_d3_epa", "Third-Down EPA", "EPA / dropback on 3rd down", 1, "calculated"),
    ("qb_expl_rate", "Explosive Completion Rate", "completions >= 16 yds / att", 1, "calculated"),
    ("qb_throwaway", "Throwaway Rate", "requires charting", 0, "unavailable_without_charting"),
    ("qb_first_read", "First-Read Throw Rate", "requires charting", 0, "unavailable_without_charting"),
    ("qb_td_rate", "Passing TD Rate", "pass TD / att", 1, "observed"),
    ("qb_rush_td_rate", "QB Rushing TD Rate", "rush TD / carries", 1, "observed"),
]:
    P(mid, QB, nm, f, d, cl, ms=30)
for mid, nm, f, d, cl in [
    ("rb_ypc", "Yards / Carry", "rush yds / carries", 1, "observed"),
    ("rb_epa", "Rushing EPA / Carry", "mean EPA on carries", 1, "calculated"),
    ("rb_sr", "Rushing Success Rate", "success on carries", 1, "calculated"),
    ("rb_expl", "Explosive Run Rate", "carries >= 12 / carries", 1, "calculated"),
    ("rb_stuff", "Stuff Rate", "carries <= 0 / carries", -1, "calculated"),
    ("rb_ybc", "Yards Before Contact", "requires tracking/charting", 1, "unavailable_without_charting"),
    ("rb_yaco", "Yards After Contact", "requires charting", 1, "unavailable_without_charting"),
    ("rb_mtf", "Missed Tackles Forced", "requires charting", 1, "unavailable_without_charting"),
    ("rb_ryoe", "Rush Yards Over Expected (situational)", "yards - E[yards | down, dist, ytg, opp rush def]", 1, "estimated"),
    ("rb_gl_carries", "Goal-Line Carries", "carries inside 5", 1, "calculated"),
    ("rb_sy_conv", "Short-Yardage Conversion", "3rd/4th & <=2 carries converted", 1, "calculated"),
    ("rb_dir", "Run Direction Splits", "requires charting", 0, "unavailable_without_charting"),
    ("rb_zone_gap", "Zone vs Gap Efficiency", "requires charting", 1, "unavailable_without_charting"),
    ("rb_box", "Light/Stacked Box Efficiency", "requires charting", 1, "unavailable_without_charting"),
    ("rb_targets_pg", "Targets / Game", "targets / games", 1, "observed"),
    ("rb_rec_epa", "Receiving EPA / Target", "mean EPA on targets", 1, "calculated"),
    ("rb_yac", "YAC / Reception", "yards - air yards on catches", 1, "calculated"),
    ("rb_pass_pro", "Pass Protection Grade", "requires charting", 1, "unavailable_without_charting"),
    ("rb_fumble", "Fumble-Lost Rate", "fumbles lost / touches", -1, "calculated"),
    ("rb_core", "Core Yards (<10) / Carry", "mean of carries gaining < 10 yds", 1, "calculated"),
]:
    P(mid, "RB", nm, f, d, cl, ms=25)
for pos in ("WR", "TE"):
    pre = pos.lower()
    for mid, nm, f, d, cl in [
        ("tgt_pg", "Targets / Game", "targets / games", 1, "observed"),
        ("epa_tgt", "Receiving EPA / Target", "mean EPA on targets", 1, "calculated"),
        ("sr", "Target Success Rate", "success on targets", 1, "calculated"),
        ("catch", "Catch Rate", "receptions / targets", 1, "observed"),
        ("ypt", "Yards / Target", "rec yds / targets", 1, "observed"),
        ("ypr", "Yards / Reception", "rec yds / receptions", 1, "observed"),
        ("adot", "Average Depth of Target", "mean air yards on targets", 0, "calculated"),
        ("air_share", "Air-Yards Share", "player air yards / team air yards in games played", 1, "calculated"),
        ("deep_rate", "Deep Target Rate", "targets with air >= 20 / targets", 0, "calculated"),
        ("deep_catch", "Deep Catch Rate", "catch rate, air >= 20", 1, "calculated"),
        ("short_catch", "Short Catch Rate", "catch rate, air < 10", 1, "calculated"),
        ("croe", "Catch Rate Over Expected", "catches - sum P(catch | air, situation) per target", 1, "estimated"),
        ("yacoe", "YAC Over Expected", "YAC - league mean YAC at same air depth bucket", 1, "estimated"),
        ("yac", "YAC / Reception", "yards after catch per reception", 1, "calculated"),
        ("expl", "Explosive Reception Rate", "catches >= 16 yds / targets", 1, "calculated"),
        ("fd_rate", "First-Down Rate / Target", "first downs / targets", 1, "calculated"),
        ("rz_share", "Red-Zone Target Share", "RZ targets / team RZ att", 1, "calculated"),
        ("td_rate", "TD / Target", "rec TD / targets", 1, "observed"),
        ("yprr", "Yards / Route Run", "requires routes", 1, "unavailable_without_charting"),
        ("tprr", "Targets / Route Run", "requires routes", 1, "unavailable_without_charting"),
        ("sep", "Separation", "requires tracking", 1, "unavailable_without_charting"),
        ("manzone", "Man vs Zone Efficiency", "requires charting", 1, "unavailable_without_charting"),
        ("contested", "Contested Catch Rate", "requires charting", 1, "unavailable_without_charting"),
        ("drop", "Drop Rate", "requires charting", -1, "unavailable_without_charting"),
        ("align", "Slot / Inline / Wide Alignment", "requires charting", 0, "unavailable_without_charting"),
        ("opp_cov", "Opponent Coverage Difficulty", "mean adj pass EPA/db allowed by opponents", -1, "calculated"),
    ]:
        P(f"{pre}_{mid}", pos, nm, f, d, cl, ms=15)
    if pos == "TE":
        for mid, nm in [("te_block_run", "Run-Blocking Success"), ("te_block_pass", "Pass-Blocking Success"),
                        ("te_press_allowed", "Pressures Allowed"), ("te_inline", "Inline Snap Rate")]:
            P(mid, "TE", nm, "requires charting", 1, "unavailable_without_charting")
    if pos == "WR":
        for mid, nm in [("wr_press", "Press Coverage Success"), ("wr_route_win", "Route Win Rate")]:
            P(mid, "WR", nm, "requires charting", 1, "unavailable_without_charting")
for pos in ("OT", "OG", "C"):
    for mid, nm, d in [("pbwr", "Pass-Block Win Rate", 1), ("rbwr", "Run-Block Win Rate", 1), ("press", "Pressures Allowed", -1),
                       ("sacks", "Sacks Allowed", -1), ("hurries", "Hurries Allowed", -1), ("hits", "QB Hits Allowed", -1),
                       ("true_pass", "True Pass-Set Efficiency", 1), ("one_on_one", "One-on-One Pass Blocking", 1),
                       ("speed", "Speed-Rush Defense", 1), ("power", "Power-Rush Defense", 1), ("stunt", "Stunt Pickup", 1),
                       ("blitz", "Blitz Recognition", 1), ("concept", "Run Concept Success", 1), ("gap", "Gap Blocking", 1),
                       ("zone", "Zone Blocking", 1), ("pull", "Pulling Effectiveness", 1), ("second", "Second-Level Blocks", 1),
                       ("pen", "Penalties", -1), ("assign", "Assignment Failures", -1)]:
        P(f"{pos.lower()}_{mid}", pos, nm, "requires OL charting", d, "unavailable_without_charting", "charting: OL")
    P(f"{pos.lower()}_unit_sack", pos, "Team Sack Rate Allowed While Rostered (proxy)", "team sack rate in games played", -1, "estimated")
    P(f"{pos.lower()}_unit_stuff", pos, "Team Stuff Rate Allowed (proxy)", "team rush stuff rate in games played", -1, "estimated")
P("c_snap", "C", "Snap Execution", "requires charting", 1, "unavailable_without_charting")
for pos in ("DL", "EDGE"):
    pre = pos.lower()
    for mid, nm, f, d, cl in [
        ("sack_rate", "Sack Rate (per opp dropback)", "sacks / opponent dropbacks in games played", 1, "calculated"),
        ("tfl_rate", "TFL Rate (per opp carry)", "TFL / opponent carries in games played", 1, "calculated"),
        ("tkl_pg", "Tackles / Game", "tackles / games", 1, "observed"),
        ("run_stop", "Run-Stop Rate", "tackles on unsuccessful opp runs / opp carries", 1, "calculated"),
        ("ff", "Forced Fumbles", "forced fumbles", 1, "observed"),
        ("tkl_depth", "Tackle Depth", "mean yards gained on plays player tackled", -1, "calculated"),
        ("prwr", "Pass-Rush Win Rate", "requires charting", 1, "unavailable_without_charting"),
        ("press", "Pressure Rate", "requires charting", 1, "unavailable_without_charting"),
        ("quick", "Quick Pressure Rate", "requires tracking", 1, "unavailable_without_charting"),
        ("ttp", "Time to Pressure", "requires tracking", -1, "unavailable_without_charting"),
        ("double", "Double-Team Rate", "requires charting", 0, "unavailable_without_charting"),
        ("one_on_one", "One-on-One Win Rate", "requires charting", 1, "unavailable_without_charting"),
        ("mtr", "Missed Tackle Rate", "requires charting", -1, "unavailable_without_charting"),
        ("contain", "Edge Containment / Lane Integrity", "requires charting", 1, "unavailable_without_charting"),
        ("opp_ol", "Opponent OL Quality", "mean opp sack-rate-allowed (adj)", 0, "calculated"),
    ]:
        P(f"{pre}_{mid}", pos, nm, f, d, cl, ms=3)
for mid, nm, f, d, cl in [
    ("lb_tkl_pg", "Tackles / Game", "tackles / games", 1, "observed"),
    ("lb_run_stop", "Run-Stop Rate", "tackles on unsuccessful opp runs / opp carries", 1, "calculated"),
    ("lb_tkl_depth", "Tackle Depth", "mean yards gained on tackled plays", -1, "calculated"),
    ("lb_tfl_rate", "TFL Rate", "TFL / opp carries", 1, "calculated"),
    ("lb_blitz_sack", "Sack Rate (per opp dropback)", "sacks / opp dropbacks", 1, "calculated"),
    ("lb_cov_events", "Coverage Plays (INT+PBU) / Game", "(INT + PBU) / games", 1, "observed"),
    ("lb_cov_epa", "Coverage EPA Allowed", "requires charting", -1, "unavailable_without_charting"),
    ("lb_mtr", "Missed Tackle Rate", "requires charting", -1, "unavailable_without_charting"),
    ("lb_fit", "Run-Fit Success", "requires charting", 1, "unavailable_without_charting"),
    ("lb_gap", "Gap-Fill Success", "requires charting", 1, "unavailable_without_charting"),
    ("lb_pa_recog", "Play-Action Recognition", "requires charting", 1, "unavailable_without_charting"),
    ("lb_assign", "Assignment Errors", "requires charting", -1, "unavailable_without_charting"),
]:
    P(mid, "LB", nm, f, d, cl, ms=3)
for pos in ("CB", "S"):
    pre = pos.lower()
    for mid, nm, f, d, cl in [
        ("ball_pg", "Ball Production (INT+PBU) / Game", "(INT + PBU) / games", 1, "observed"),
        ("int", "Interceptions", "INT", 1, "observed"),
        ("pbu", "Pass Breakups", "PBU", 1, "observed"),
        ("tkl_pg", "Tackles / Game", "tackles / games", 0, "observed"),
        ("tkl_depth", "Tackle Depth (pass)", "mean yards on completions player tackled", -1, "calculated"),
        ("expl_tkl", "Explosive Plays Ended", "share of player's tackles on explosive plays (lower=better)", -1, "estimated"),
        ("run_support", "Run-Support Tackles / Game", "tackles on runs / games", 1, "calculated"),
        ("cov_snaps", "Coverage Snaps", "requires participation", 0, "unavailable_without_charting"),
        ("tgts_allowed", "Targets Allowed", "requires coverage charting", -1, "unavailable_without_charting"),
        ("cmp_allowed", "Completion % Allowed", "requires coverage charting", -1, "unavailable_without_charting"),
        ("ypcs", "Yards / Coverage Snap", "requires coverage charting", -1, "unavailable_without_charting"),
        ("cov_epa", "Coverage EPA Allowed", "requires coverage charting", -1, "unavailable_without_charting"),
        ("sep_allowed", "Separation Allowed", "requires tracking", -1, "unavailable_without_charting"),
        ("manzone", "Man / Zone Efficiency", "requires charting", 1, "unavailable_without_charting"),
        ("align", "Slot / Box / Deep Alignment", "requires charting", 0, "unavailable_without_charting"),
    ]:
        P(f"{pre}_{mid}", pos, nm, f, d, cl, ms=3)
for mid, nm, f, d, cl, pos in [
    ("k_fg_pct", "Field Goal %", "fgm / fga", 1, "observed", "K"),
    ("k_fg_poe", "FG Points Over Expected", "sum 3*(made - P(make|distance))", 1, "calculated", "K"),
    ("k_fg_40", "FG % 40+", "made / att with distance >= 40", 1, "calculated", "K"),
    ("k_long", "Long Make", "max distance made", 1, "observed", "K"),
    ("k_xp_pct", "Extra Point %", "xpm / xpa", 1, "observed", "K"),
    ("p_net", "Net Punt Average", "mean net yards", 1, "calculated", "P"),
    ("p_in20", "Inside-20 Rate", "punts leaving receiver at or beyond own 20 / punts", 1, "calculated", "P"),
    ("p_tb", "Touchback Rate", "punts into end zone / punts", -1, "calculated", "P"),
    ("p_epa", "Punt EPA / Punt", "mean EPA of punt plays", 1, "calculated", "P"),
    ("ls_snap", "Long Snapper Accuracy", "requires charting", 1, "unavailable_without_charting", "LS"),
    ("kr_avg", "Kick Return Average", "requires returner attribution", 1, "unavailable_without_charting", "KR"),
    ("pr_avg", "Punt Return Average", "requires returner attribution", 1, "unavailable_without_charting", "PR"),
]:
    P(mid, pos, nm, f, d, cl, ms=3)

DEF_BY_ID = {d.mid: d for d in DEFS}


def registry_rows() -> list[dict]:
    return [dict(metric_id=d.mid, scope=d.scope, category=d.cat, name=d.name, positions=d.positions,
                 formula=d.formula, units=None, denominator=None, direction=d.direction,
                 classification=d.classification, required_data=d.required, min_sample=d.min_sample,
                 opponent_adjustable=int(d.mid.startswith("adj_"))) for d in DEFS]


# =========================================================================== computation
def _sitmask(p, name):
    return {"rz": p.yards_to_goal <= 20, "d3": p.down == 3, "d4": p.down == 4,
            "sy": (p.down >= 3) & (p.distance <= 2), "ly": p.distance >= 10, "early": p.down <= 2,
            "late": (p.period == 4) & (p.margin.abs() <= 8), "lead": p.margin > 0, "trail": p.margin < 0,
            "home": p.off_home, "away": p.off_away, "q4": p.period == 4,
            "pd": ((p.down == 2) & (p.distance >= 8)) | ((p.down >= 3) & (p.distance >= 5))}[name].fillna(False)


def compute_player_metrics(conn, season: int, through_week: int | None = None) -> pd.DataFrame:
    if through_week is None:
        through_week = int(conn.execute("SELECT MAX(week) FROM games WHERE season=? AND completed=1", (season,)).fetchone()[0] or 0)
    allp = load_plays(conn, season, through_week, include_prior_seasons=1)
    p = allp[allp.season == season]
    ros = pd.read_sql_query("""SELECT r.player_id, r.team_id, r.position, r.class_year, pl.first_name, pl.last_name,
        pl.height_in, pl.weight_lb, pl.birth_year, ts.conference_id
        FROM rosters r JOIN players pl ON pl.player_id=r.player_id
        LEFT JOIN team_seasons ts ON ts.team_id=r.team_id AND ts.season=r.season WHERE r.season=?""", conn, params=(season,))
    ros = ros.drop_duplicates("player_id").set_index("player_id")
    W = pd.DataFrame(index=ros.index)
    W["position"] = ros.position
    W["team_id"] = ros.team_id
    W["conf"] = ros.conference_id
    W["age"] = season - ros.birth_year
    W["height"] = ros.height_in
    W["weight"] = ros.weight_lb
    W["class_year"] = ros.class_year
    exp = pd.read_sql_query("SELECT player_id, COUNT(DISTINCT season) n FROM rosters WHERE season<=? GROUP BY 1", conn, params=(season,))
    W["experience"] = exp.set_index("player_id").n
    rec = pd.read_sql_query("SELECT player_id, stars, rating FROM recruiting_profiles WHERE player_id IS NOT NULL", conn)
    rec = rec.drop_duplicates("player_id").set_index("player_id")
    W["stars"] = rec.stars
    W["recruit_rating"] = rec.rating
    dc = pd.read_sql_query("SELECT player_id, MIN(depth) d FROM depth_charts WHERE season=? AND week=0 GROUP BY 1", conn, params=(season,)).set_index("player_id").d
    W["depth"] = dc
    W["starter"] = (dc == 1).astype(float).reindex(W.index)
    tr = pd.read_sql_query("SELECT player_id FROM transfer_history WHERE season=? AND player_id IS NOT NULL", conn, params=(season,))
    W["transfer_in"] = W.index.isin(tr.player_id).astype(float)
    inj = pd.read_sql_query("SELECT player_id, COUNT(*) n FROM injuries WHERE season=? AND week<=? GROUP BY 1", conn, params=(season, through_week))
    W["games_out"] = inj.set_index("player_id").n.reindex(W.index).fillna(0)

    # ---- box scores
    pg = pd.read_sql_query("""SELECT pg.*, g.week FROM player_games pg JOIN games g ON g.game_id=pg.game_id
                              WHERE g.season=? AND g.week<=?""", conn, params=(season, through_week))
    box_cols = ["pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "sacks_taken", "rush_att", "rush_yds", "rush_td",
                "targets", "receptions", "rec_yds", "rec_td", "tackles", "tfl", "def_sacks", "def_int", "pbu",
                "forced_fumbles", "fga", "fgm", "xpa", "xpm", "punts", "punt_yds"]
    tot = pg.groupby("player_id")[box_cols].sum()
    for c in box_cols:
        W[c] = tot[c]
    W["games"] = pg.groupby("player_id").game_id.nunique()
    W["scrim_yds"] = W.rush_yds.fillna(0) + W.rec_yds.fillna(0)
    W["total_td"] = W.pass_td.fillna(0) + W.rush_td.fillna(0) + W.rec_td.fillna(0)
    W["yds_pg"] = (W.pass_yds.fillna(0) + W.scrim_yds) / W.games
    W["havoc"] = W[["tfl", "def_sacks", "def_int", "pbu", "forced_fumbles"]].fillna(0).sum(axis=1)
    W["touches"] = W.rush_att.fillna(0) + W.receptions.fillna(0)
    W["touches_pg"] = W.touches / W.games

    # ---- event-level opportunities
    ev = pd.read_sql_query("""SELECT e.play_id, e.player_id, e.team_id, e.role FROM play_events e JOIN plays p ON p.play_id=e.play_id
                              WHERE p.season BETWEEN ? AND ? AND (p.season<? OR p.week<=?)""", conn,
                           params=(season - 1, season, season, through_week))
    allp = allp.set_index("play_id")
    # opportunity roles: passer on dropback (incl. sacks); rusher on carries (scrambles count as QB dropbacks)
    ev = ev.join(allp[["season", "week", "game_id", "offense_id", "defense_id", "play_type", "epa", "success", "yards_gained",
                       "explosive", "negative", "first_down", "air_yards", "completion", "interception", "touchdown",
                       "down", "distance", "yards_to_goal", "period", "margin", "off_home", "off_away", "clean",
                       "dropback", "rush", "carry", "att", "fumble_lost", "stuff"]], on="play_id", how="inner")
    cur = ev[ev.season == season]
    # league opponent adjustment by play family
    off = p[p.is_off & p.clean]
    fam_adj = {}
    for fam, m in (("db", off.dropback), ("carry", off.carry), ("att", off.att)):
        sub = off[m]
        gl = sub.groupby(["game_id", "offense_id", "defense_id", "off_home", "neutral_site"]).epa.agg(["mean", "count"]).reset_index()
        gl["home"] = np.where(gl.neutral_site == 1, 0, np.where(gl.off_home, 1, -1))
        ao, ad, mu = ridge_adjust(gl.rename(columns={"offense_id": "off", "defense_id": "def", "mean": "value", "count": "w"}))
        fam_adj[fam] = (ad, mu)
    allfam = ridge_adjust(off.groupby(["game_id", "offense_id", "defense_id"]).epa.agg(["mean", "count"]).reset_index()
                          .assign(home=0).rename(columns={"offense_id": "off", "defense_id": "def", "mean": "value", "count": "w"}))

    def is_opp(e):
        return (((e.role == "passer") & e.dropback) | ((e.role == "rusher") & e.carry & ~((e.play_type == "scramble"))) |
                ((e.role == "target") & e.att))
    opp = cur[is_opp(cur) & cur.clean].copy()
    fam = np.where(opp.role == "passer", "db", np.where(opp.role == "rusher", "carry", "att"))
    opp["fam"] = fam
    opp["opp_allowed"] = [fam_adj[f][0].get(d, fam_adj[f][1]) - fam_adj[f][1] for f, d in zip(fam, opp.defense_id)]
    opp["adj_epa"] = opp.epa - opp.opp_allowed
    gO = opp.groupby("player_id")
    W["plays"] = cur.groupby("player_id").play_id.nunique()
    W["epa_total"] = gO.epa.sum()
    n_opp = gO.size()
    W["epa_play"] = gO.epa.mean()
    W["adj_epa_play"] = gO.adj_epa.mean()
    W["success"] = gO.success.mean()
    W["explosive"] = gO.explosive.mean()
    W["negative"] = gO.negative.mean()
    W["first_down"] = gO.first_down.mean()
    W["epa_early"] = opp[opp.down <= 2].groupby("player_id").epa.mean()
    W["epa_pd"] = opp[_sitmask(opp, "pd")].groupby("player_id").epa.mean()
    W["epa_pg"] = W.epa_total / W.games
    W["opp_quality"] = gO.opp_allowed.mean()
    W["adj_delta"] = W.adj_epa_play - W.epa_play
    for mid, filt, col in [("rz_epa", "rz", "epa"), ("d3_sr", "d3", "success"), ("d4_sr", "d4", "success"),
                           ("sy_sr", "sy", "success"), ("ly_sr", "ly", "success"), ("early_sr", "early", "success"),
                           ("late_epa", "late", "epa"), ("lead_epa", "lead", "epa"), ("trail_epa", "trail", "epa"),
                           ("home_epa", "home", "epa"), ("away_epa", "away", "epa"), ("q4_epa", "q4", "epa")]:
        W[mid] = opp[_sitmask(opp, filt)].groupby("player_id")[col].mean()
    W["d3_usage"] = opp.assign(d3=opp.down == 3).groupby("player_id").d3.mean()
    W["rz_touches"] = opp[opp.yards_to_goal <= 20].groupby("player_id").size()
    # consistency (game-level)
    ge = opp.groupby(["player_id", "game_id"]).agg(epa=("epa", "mean"), wk=("week", "first")).reset_index().sort_values("wk")
    W["l3_epa"] = ge.groupby("player_id").tail(3).groupby("player_id").epa.mean()
    W["l5_epa"] = ge.groupby("player_id").tail(5).groupby("player_id").epa.mean()
    allopp = ev[is_opp(ev) & ev.clean]
    ge10 = allopp.groupby(["player_id", "game_id"]).agg(epa=("epa", "mean"), s=("season", "first"), wk=("week", "first")).reset_index()
    ge10 = ge10.sort_values(["s", "wk"])
    W["l10_epa"] = ge10.groupby("player_id").tail(10).groupby("player_id").epa.mean()
    W["game_sd_epa"] = ge.groupby("player_id").epa.std()
    W["trend"] = ge.groupby("player_id").apply(lambda d: np.polyfit(np.arange(len(d)), d.epa, 1)[0] if len(d) >= 3 else np.nan,
                                               include_groups=False)
    pg["tot_yds"] = pg.pass_yds + pg.rush_yds + pg.rec_yds
    gy = pg.groupby("player_id").tot_yds
    W["game_sd_yds"] = gy.std()
    W["yds_cv"] = gy.std() / gy.mean().replace(0, np.nan)
    W["yds_p10"] = gy.quantile(0.1)
    W["yds_p90"] = gy.quantile(0.9)
    W["reliability"] = n_opp / (n_opp + 60)
    W["epa_se"] = gO.epa.std() / np.sqrt(n_opp)
    pos_mean = W.groupby("position").adj_epa_play.transform("mean")
    W["regressed_epa"] = W.reliability * W.adj_epa_play + (1 - W.reliability) * pos_mean
    # shares (team totals in games the player appeared)
    tg = off.groupby(["offense_id", "game_id"]).agg(t_att=("att", "sum"), t_carry=("carry", "sum"), t_db=("dropback", "sum"),
                                                    t_rz=("yards_to_goal", lambda s: (s <= 20).sum()),
                                                    t_air=("air_yards", lambda s: s.clip(lower=0).sum()),
                                                    t_epa=("epa", "sum")).reset_index()
    pgm = pg[["player_id", "game_id", "team_id"]].merge(tg, left_on=["team_id", "game_id"], right_on=["offense_id", "game_id"])
    tt = pgm.groupby("player_id")[["t_att", "t_carry", "t_db", "t_rz", "t_air", "t_epa"]].sum()
    tgt = opp[opp.role == "target"]
    W["target_share"] = tgt.groupby("player_id").size() / tt.t_att
    W["rush_share"] = W.rush_att / tt.t_carry
    W["dropback_share"] = opp[opp.role == "passer"].groupby("player_id").size() / tt.t_db
    W["rz_share"] = opp[(opp.yards_to_goal <= 20) & (opp.role != "passer")].groupby("player_id").size() / tt.t_rz
    W["team_epa_share"] = W.epa_total / tt.t_epa.abs().replace(0, np.nan)
    # competition
    ao_all, ad_all, mu_all = allfam
    teams_net = pd.Series({t: ao_all.get(t, mu_all) - ad_all.get(t, mu_all) for t in set(ao_all) | set(ad_all)})
    tconf = ros.groupby("team_id").conference_id.first()
    conf_str = teams_net.groupby(tconf.reindex(teams_net.index)).mean()
    W["conf_strength"] = W.conf.map(conf_str)
    team_off = off.groupby("offense_id").epa.agg(["sum", "size"])
    psum = gO.epa.sum()
    W["team_off_quality"] = [(team_off.loc[t, "sum"] - (psum.get(pid, 0)))
                             / max(team_off.loc[t, "size"] - n_opp.get(pid, 0), 1) if t in team_off.index else np.nan
                             for pid, t in zip(W.index, W.team_id)]
    famm = opp.groupby(["offense_id", "fam"]).epa.agg(["sum", "size"])
    pf = opp.groupby(["player_id", "offense_id", "fam"]).epa.agg(["sum", "size"]).reset_index()
    pf = pf.join(famm, on=["offense_id", "fam"], rsuffix="_t")
    pf["mate"] = (pf.sum_t - pf["sum"]) / (pf.size_t - pf["size"]).replace(0, np.nan)
    pf["rel"] = pf["sum"] / pf["size"] - pf.mate
    W["rel_team"] = pf.groupby("player_id").apply(lambda d: np.average(d.rel.fillna(0), weights=d["size"]), include_groups=False)
    if not off.empty:
        lr = LogisticRegression(max_iter=500)
        X = np.column_stack([off.down == 1, off.down == 2, np.log1p(off.distance.clip(lower=1)), off.yards_to_goal / 100,
                             off.margin.clip(-28, 28) / 14]).astype(float)
        lr.fit(X, off.dropback.astype(int))
        proe = (off.dropback - lr.predict_proba(X)[:, 1]).groupby(off.offense_id).mean()
        W["scheme_proe"] = W.team_id.map(proe)
    topdef = pd.Series(ad_all).rank(pct=True)
    topd = set(topdef[topdef <= 0.4].index)
    W["vs_top_def"] = opp[opp.defense_id.isin(topd)].groupby("player_id").epa.mean()
    # impact
    repl = W[W.reliability > 0.3].groupby("position").epa_play.quantile(0.2)
    W["epa_oar"] = W.epa_total - n_opp.reindex(W.index) * W.position.map(repl)
    W["pts_oar_pg"] = W.epa_oar / W.games
    W["impact_ci"] = 1.28 * W.epa_se * n_opp.reindex(W.index) / W.games
    W["def_impact"] = W.havoc / W.games
    deve = cur[cur.role.isin(["tackler", "sacker", "interceptor", "pbu", "forced_fumble"])]
    W["def_epa_event"] = -deve.groupby("player_id").epa.mean()
    team_tk = pg.groupby(["team_id", "game_id"]).tackles.sum().rename("tt").reset_index()
    ptk = pg.merge(team_tk, on=["team_id", "game_id"]).groupby("player_id")[["tackles", "tt"]].sum()
    W["tackle_share"] = ptk.tackles / ptk.tt.replace(0, np.nan)

    W = W.copy()
    _position_metrics(W, cur, opp, off, p, pg, fam_adj, ros)
    W = W.copy()
    # archetypes use within-season percentiles so rules transfer across data environments
    for pos, cols in {"QB": ["qb_designed_run", "qb_scramble", "qb_rush_epa", "qb_adot", "qb_deep_rate", "qb_comp",
                             "qb_int_rate", "qb_epa_db"], "RB": ["target_share", "rb_expl"],
                      "WR": ["wr_adot"], "TE": ["te_adot"]}.items():
        q = (W.position == pos) & (W.reliability.fillna(0) > 0.3)
        for c in cols:
            W.loc[q, "_pct_" + c] = W.loc[q, c].rank(pct=True)
    arche = W.apply(archetype, axis=1)
    W = W.drop(columns=[c for c in W.columns if c.startswith("_pct_")])
    # ---- long format + percentiles
    long_rows = []
    pos_of = W.position
    games = W.games.fillna(0)
    for d in DEFS:
        if d.classification == "unavailable_without_charting":
            continue
        if d.mid not in W.columns:
            continue
        s = W[d.mid]
        long_rows.append(pd.DataFrame({"player_id": W.index, "metric_id": d.mid, "value": s.values}))
    L = pd.concat(long_rows, ignore_index=True)
    L["position"] = L.player_id.map(pos_of)
    L["conf"] = L.player_id.map(W.conf)
    L["sample"] = L.player_id.map(n_opp).fillna(L.player_id.map(W.plays)).fillna(0).astype(int)
    L["games"] = L.player_id.map(games)
    ms = L.metric_id.map(lambda m: DEF_BY_ID[m].min_sample)
    L["status"] = np.where(L.value.isna(), "no_data", np.where(L["sample"] < ms, "low_sample", "ok"))
    dirs = L.metric_id.map(lambda m: DEF_BY_ID[m].direction)
    qualified = (L.status == "ok") & (L.games >= max(1, through_week // 3))
    L["_v"] = np.where(qualified, L.value * np.where(dirs == 0, 1, dirs), np.nan)
    L["pctl_fbs"] = L.groupby(["metric_id", "position"])["_v"].rank(pct=True) * 100
    L["pctl_conf"] = L.groupby(["metric_id", "position", "conf"])["_v"].rank(pct=True) * 100
    L["adj_value"] = np.where(L.metric_id == "epa_play", L.player_id.map(W.adj_epa_play), np.nan)
    L["season"] = season
    L["through_week"] = through_week
    L = L.drop(columns=["_v", "conf", "games", "position"])
    grades = position_grades(W, n_opp)
    return L, W.assign(archetype=arche, grade=grades["grade"], grade_pctl=grades["pctl"])


# --------------------------------------------------------------------------- positions
def _position_metrics(W, cur, opp, off, p, pg, fam_adj, ros):
    pas = cur[(cur.role == "passer")]
    att = pas[pas.att]
    # completion probability model (league), for CPOE / CROE / xINT — only when air yards exist
    has_air = att.air_yards.notna().mean() > 0.5 if len(att) else False
    A = off[off.att & off.air_yards.notna()]
    if has_air and len(A) > 500:
        X = lambda d: np.column_stack([d.air_yards.clip(-5, 60), d.air_yards.clip(-5, 60) ** 2 / 100,
                                       d.down.fillna(1) >= 3, np.log1p(d.distance.fillna(10).clip(lower=1)),
                                       d.yards_to_goal / 100]).astype(float)
        cm = LogisticRegression(max_iter=500).fit(X(A), A.completion.astype(int))
        im = LogisticRegression(max_iter=500).fit(X(A), A.interception.astype(int))
        attc = att[att.air_yards.notna()].copy()
        attc["xc"] = cm.predict_proba(X(attc))[:, 1]
        attc["xi"] = im.predict_proba(X(attc))[:, 1]
        W["qb_cpoe"] = (attc.completion - attc.xc).groupby(attc.player_id).mean()
        W["qb_xint"] = attc.groupby("player_id").xi.mean()
        tg = cur[(cur.role == "target") & cur.att & cur.air_yards.notna()].copy()
        tg["xc"] = cm.predict_proba(X(tg))[:, 1]
        croe = (tg.completion - tg.xc).groupby(tg.player_id).mean()
        A2 = A[A.completion == 1].assign(b=lambda d: pd.cut(d.air_yards, [-99, 0, 5, 10, 15, 20, 30, 99]),
                                         yac=lambda d: d.yards_gained - d.air_yards)
        yac_b = A2.groupby("b", observed=True).yac.mean()
        cc = tg[tg.completion == 1].assign(b=lambda d: pd.cut(d.air_yards, [-99, 0, 5, 10, 15, 20, 30, 99]),
                                           yac=lambda d: d.yards_gained - d.air_yards)
        cc["xyac"] = cc.b.map(yac_b).astype(float)
        yacoe = (cc.yac - cc.xyac).groupby(cc.player_id).mean()
        yac = cc.groupby("player_id").yac.mean()
    else:
        croe = yacoe = yac = pd.Series(dtype=float)
    db = pas[pas.dropback]
    qbr = cur[(cur.role == "rusher") & cur.carry]
    scr = qbr[qbr.play_type == "scramble"]
    dbs = db.groupby("player_id").size().add(scr.groupby("player_id").size(), fill_value=0)
    qbepa = db.groupby("player_id").epa.sum().add(scr.groupby("player_id").epa.sum(), fill_value=0)
    W["qb_epa_db"] = qbepa / dbs
    W["qb_sr"] = (db.groupby("player_id").success.sum().add(scr.groupby("player_id").success.sum(), fill_value=0)) / dbs
    W["qb_comp"] = W.pass_cmp / W.pass_att
    ag = att.groupby("player_id")
    W["qb_adot"] = ag.air_yards.mean()
    W["qb_air_ypa"] = att[att.completion == 1].groupby("player_id").air_yards.sum() / W.pass_att
    W["qb_ypa"] = W.pass_yds / W.pass_att
    sack_yds = pas[pas.play_type == "sack"].groupby("player_id").yards_gained.sum()
    W["qb_anya"] = (W.pass_yds + 20 * W.pass_td - 45 * W.pass_int + sack_yds.reindex(W.index).fillna(0)) / (W.pass_att + W.sacks_taken)
    W["qb_int_rate"] = W.pass_int / W.pass_att
    W["qb_sack_rate"] = W.sacks_taken / dbs
    for mid, lo, hi in (("qb_deep", 20, 99), ("qb_inter", 10, 19), ("qb_short", 0, 9)):
        s = att[att.air_yards.between(lo, hi)]
        W[f"{mid}_acc"] = s.groupby("player_id").completion.mean()
    W["qb_deep_rate"] = att.assign(d=att.air_yards >= 20).groupby("player_id").d.mean().mul(1.0 if has_air else np.nan)
    W["qb_behind_los"] = att.assign(d=att.air_yards < 0).groupby("player_id").d.mean().mul(1.0 if has_air else np.nan)
    W["qb_scramble"] = scr.groupby("player_id").size().reindex(W.index) / dbs
    des = qbr[qbr.play_type == "rush"].groupby("player_id").size()
    W["qb_designed_run"] = des.reindex(W.index) / (dbs + des.reindex(W.index).fillna(0))
    W["qb_rush_epa"] = qbr.groupby("player_id").epa.mean()
    W["qb_rz_epa"] = db[db.yards_to_goal <= 20].groupby("player_id").epa.mean()
    W["qb_d3_epa"] = db[db.down == 3].groupby("player_id").epa.mean()
    W["qb_expl_rate"] = ag.explosive.mean()
    W["qb_td_rate"] = W.pass_td / W.pass_att
    W["qb_rush_td_rate"] = W.rush_td / W.rush_att
    for c in [c for c in W.columns if c.startswith("qb_")]:
        W.loc[W.position != "QB", c] = np.nan
    # RB
    car = cur[(cur.role == "rusher") & cur.carry]
    cg = car.groupby("player_id")
    W["rb_ypc"] = W.rush_yds / W.rush_att
    W["rb_epa"] = cg.epa.mean()
    W["rb_sr"] = cg.success.mean()
    W["rb_expl"] = cg.explosive.mean()
    W["rb_stuff"] = cg.stuff.mean()
    W["rb_core"] = car[car.yards_gained < 10].groupby("player_id").yards_gained.mean()
    # situational expected yards (league by down/dist bucket/ytg bucket) + opponent rush allowed
    lc = off[off.carry].copy()
    key = lambda d: d.down.astype(str) + "_" + pd.cut(d.distance, [0, 2, 5, 9, 99], labels=False).astype(str) + "_" + \
        pd.cut(d.yards_to_goal, [0, 5, 20, 50, 100], labels=False).astype(str)
    exp_y = lc.groupby(key(lc)).yards_gained.mean()
    car2 = car.assign(k=key(car))
    W["rb_ryoe"] = (car2.yards_gained - car2.k.map(exp_y)).groupby(car2.player_id).mean()
    W["rb_gl_carries"] = car[car.yards_to_goal <= 5].groupby("player_id").size()
    sy = car[(car.down >= 3) & (car.distance <= 2)]
    W["rb_sy_conv"] = ((sy.first_down == 1) | (sy.touchdown == 1)).groupby(sy.player_id).mean()
    W["rb_targets_pg"] = W.targets / W.games
    tg_all = cur[(cur.role == "target") & cur.att]
    tgg = tg_all.groupby("player_id")
    W["rb_rec_epa"] = tgg.epa.mean()
    W["rb_yac"] = yac
    fum = cur[cur.role.isin(["rusher", "target"])].groupby("player_id").fumble_lost.sum()
    W["rb_fumble"] = fum / W.touches.replace(0, np.nan)
    for c in [c for c in W.columns if c.startswith("rb_")]:
        W.loc[W.position != "RB", c] = np.nan
    # WR / TE
    pgm = pg.merge(off.groupby(["offense_id", "game_id"]).agg(t_air=("air_yards", lambda s: s.clip(lower=0).sum()),
                                                              t_rz=("yards_to_goal", lambda s: 0)).reset_index(),
                   left_on=["team_id", "game_id"], right_on=["offense_id", "game_id"])
    t_air = pgm.groupby("player_id").t_air.sum()
    rz_att = off[off.att & (off.yards_to_goal <= 20)].groupby(["offense_id", "game_id"]).size().rename("n").reset_index()
    pg_rz = pg.merge(rz_att, left_on=["team_id", "game_id"], right_on=["offense_id", "game_id"]).groupby("player_id").n.sum()
    ad = fam_adj["db"]
    for pos in ("WR", "TE"):
        pre = pos.lower()
        m = W.position == pos
        vals = {
            "tgt_pg": W.targets / W.games, "epa_tgt": tgg.epa.mean(), "sr": tgg.success.mean(),
            "catch": W.receptions / W.targets, "ypt": W.rec_yds / W.targets, "ypr": W.rec_yds / W.receptions,
            "adot": tgg.air_yards.mean(), "air_share": tg_all.groupby("player_id").air_yards.apply(lambda s: s.clip(lower=0).sum()) / t_air,
            "deep_rate": tg_all.assign(d=tg_all.air_yards >= 20).groupby("player_id").d.mean().mul(1.0 if has_air else np.nan),
            "deep_catch": tg_all[tg_all.air_yards >= 20].groupby("player_id").completion.mean(),
            "short_catch": tg_all[tg_all.air_yards < 10].groupby("player_id").completion.mean(),
            "croe": croe, "yacoe": yacoe, "yac": yac, "expl": tgg.explosive.mean(), "fd_rate": tgg.first_down.mean(),
            "rz_share": tg_all[tg_all.yards_to_goal <= 20].groupby("player_id").size() / pg_rz,
            "td_rate": W.rec_td / W.targets,
            "opp_cov": tg_all.groupby("player_id").defense_id.apply(lambda s: np.mean([ad[0].get(d, ad[1]) for d in s])),
        }
        for k, v in vals.items():
            W[f"{pre}_{k}"] = pd.Series(v).reindex(W.index).where(m)
    # OL proxies
    ol_team = off.groupby("offense_id").agg(sack=("sack", lambda s: s[off.loc[s.index, "dropback"]].mean()),
                                            stuff=("stuff", lambda s: s[off.loc[s.index, "carry"]].mean()))
    for pos in ("OT", "OG", "C"):
        m = W.position == pos
        W[f"{pos.lower()}_unit_sack"] = W.team_id.map(ol_team["sack"]).where(m)
        W[f"{pos.lower()}_unit_stuff"] = W.team_id.map(ol_team["stuff"]).where(m)
    # defensive per-opportunity rates (opp dropbacks / carries in games played)
    opp_vol = off.groupby(["defense_id", "game_id"]).agg(odb=("dropback", "sum"), ocar=("carry", "sum")).reset_index()
    pv = pg.merge(opp_vol, left_on=["team_id", "game_id"], right_on=["defense_id", "game_id"]).groupby("player_id")[["odb", "ocar"]].sum()
    tk = cur[cur.role == "tackler"]
    run_stop = tk[tk.carry & (tk.success == 0)].groupby("player_id").size() / pv.ocar
    tdepth = tk.groupby("player_id").yards_gained.mean()
    tdepth_pass = tk[tk.att].groupby("player_id").yards_gained.mean()
    sack_rate_allowed = off[off.dropback].groupby("offense_id").sack.mean()
    opp_ol = pg.assign(opp=pg.game_id).merge(off[["game_id", "offense_id", "defense_id"]].drop_duplicates(),
                                              left_on=["game_id", "team_id"], right_on=["game_id", "defense_id"])
    opp_ol = opp_ol.assign(s=opp_ol.offense_id.map(sack_rate_allowed)).groupby("player_id").s.mean()
    for pos in ("DL", "EDGE"):
        pre = pos.lower()
        m = W.position == pos
        vals = {"sack_rate": W.def_sacks / pv.odb, "tfl_rate": W.tfl / pv.ocar, "tkl_pg": W.tackles / W.games,
                "run_stop": run_stop, "ff": W.forced_fumbles, "tkl_depth": tdepth, "opp_ol": opp_ol}
        for k, v in vals.items():
            W[f"{pre}_{k}"] = pd.Series(v).reindex(W.index).where(m)
    m = W.position == "LB"
    for k, v in {"lb_tkl_pg": W.tackles / W.games, "lb_run_stop": run_stop, "lb_tkl_depth": tdepth,
                 "lb_tfl_rate": W.tfl / pv.ocar, "lb_blitz_sack": W.def_sacks / pv.odb,
                 "lb_cov_events": (W.def_int.fillna(0) + W.pbu.fillna(0)) / W.games}.items():
        W[k] = pd.Series(v).reindex(W.index).where(m)
    expl_tk = tk.groupby("player_id").explosive.mean()
    run_tk = tk[tk.carry].groupby("player_id").size()
    for pos in ("CB", "S"):
        pre = pos.lower()
        m = W.position == pos
        for k, v in {"ball_pg": (W.def_int.fillna(0) + W.pbu.fillna(0)) / W.games, "int": W.def_int, "pbu": W.pbu,
                     "tkl_pg": W.tackles / W.games, "tkl_depth": tdepth_pass, "expl_tkl": expl_tk,
                     "run_support": run_tk / W.games}.items():
            W[f"{pre}_{k}"] = pd.Series(v).reindex(W.index).where(m)
    # kickers / punters
    kk = cur[(cur.role == "kicker") & (cur.play_type == "field_goal")].copy()
    allfg = p[p.play_type == "field_goal"].dropna(subset=["fg_distance"])
    if len(allfg) > 30 and allfg.fg_made.nunique() > 1:
        lr = LogisticRegression().fit(allfg[["fg_distance"]], allfg.fg_made.astype(int))
        kk = kk.join(p.set_index("play_id")[["fg_distance", "fg_made"]], on="play_id")
        kk = kk.dropna(subset=["fg_distance"])
        kk["pexp"] = lr.predict_proba(kk[["fg_distance"]])[:, 1]
        W["k_fg_poe"] = (3 * (kk.fg_made - kk.pexp)).groupby(kk.player_id).sum()
        W["k_fg_40"] = kk[kk.fg_distance >= 40].groupby("player_id").fg_made.mean()
        W["k_long"] = kk[kk.fg_made == 1].groupby("player_id").fg_distance.max()
    W["k_fg_pct"] = W.fgm / W.fga
    W["k_xp_pct"] = W.xpm / W.xpa
    pu = cur[(cur.role == "punter") & (cur.play_type == "punt")]
    land = pu.yards_to_goal - pu.yards_gained
    W["p_net"] = pu.groupby("player_id").yards_gained.mean()
    W["p_in20"] = ((land > 0) & (land <= 20)).groupby(pu.player_id).mean()
    W["p_tb"] = (land <= 0).groupby(pu.player_id).mean()
    W["p_epa"] = pu.groupby("player_id").epa.mean()
    for c, pos in (("k_", "K"), ("p_", "P")):
        for col in [x for x in W.columns if x.startswith(c)]:
            W.loc[W.position != pos, col] = np.nan


GRADE_KEYS = {
    "QB": [("adj_epa_play", 1, 3), ("qb_sr", 1, 1.5), ("qb_int_rate", -1, 1), ("qb_sack_rate", -1, 0.7), ("qb_cpoe", 1, 1)],
    "RB": [("adj_epa_play", 1, 2), ("rb_sr", 1, 1.5), ("rb_expl", 1, 1), ("rb_ryoe", 1, 1), ("touches_pg", 1, 0.7)],
    "WR": [("adj_epa_play", 1, 2), ("wr_ypt", 1, 1.5), ("target_share", 1, 1.2), ("wr_croe", 1, 0.8)],
    "TE": [("adj_epa_play", 1, 2), ("te_ypt", 1, 1.5), ("target_share", 1, 1.0)],
    "DL": [("dl_sack_rate", 1, 1.5), ("dl_tfl_rate", 1, 1.5), ("dl_run_stop", 1, 1)],
    "EDGE": [("edge_sack_rate", 1, 2), ("edge_tfl_rate", 1, 1.2), ("edge_run_stop", 1, 0.8)],
    "LB": [("lb_run_stop", 1, 1.5), ("lb_tkl_pg", 1, 1), ("lb_tfl_rate", 1, 1), ("lb_cov_events", 1, 0.8)],
    "CB": [("cb_ball_pg", 1, 2), ("cb_tkl_depth", -1, 0.7)],
    "S": [("s_ball_pg", 1, 1.5), ("s_tkl_depth", -1, 0.7), ("s_run_support", 1, 0.7)],
    "K": [("k_fg_poe", 1, 2), ("k_xp_pct", 1, 0.5)],
    "P": [("p_net", 1, 2), ("p_in20", 1, 1)],
}


def position_grades(W: pd.DataFrame, n_opp: pd.Series) -> pd.DataFrame:
    """Model grade 0-100: weighted z-composite within position, shrunk by sample reliability,
    mapped through the normal CDF. Classification: estimated (a model output, not a scout grade)."""
    out = pd.DataFrame(index=W.index, columns=["grade", "pctl"], dtype=float)
    for pos, keys in GRADE_KEYS.items():
        m = (W.position == pos) & (W.games.fillna(0) >= 1)
        if m.sum() < 3:
            continue
        sub = W[m]
        comp = pd.Series(0.0, index=sub.index)
        wsum = pd.Series(0.0, index=sub.index)
        for col, d, w in keys:
            if col not in sub or sub[col].notna().sum() < 3:
                continue
            z = (sub[col] - sub[col].mean()) / (sub[col].std() or 1) * d
            comp = comp.add(z.fillna(0) * w)
            wsum = wsum.add(z.notna() * w)
        comp = comp / wsum.replace(0, np.nan)
        rel = sub.games / (sub.games + 3) if pos not in GRADE_KEYS or pos in ("DL", "EDGE", "LB", "CB", "S", "K", "P") \
            else sub.reliability.fillna(0)
        shrunk = comp * rel
        sd = shrunk.std() or 1
        out.loc[sub.index, "grade"] = np.clip(100 * norm.cdf(shrunk / sd), 0.1, 99.9).round(1)
        out.loc[sub.index, "pctl"] = shrunk.rank(pct=True) * 100
    return out


def archetype(r) -> str | None:
    """Rule-based archetype from within-season position percentiles (estimated)."""
    pos = r.get("position")

    def q(k):
        v = r.get("_pct_" + k)
        return None if v is None or pd.isna(v) else float(v)
    if pos == "QB" and q("qb_epa_db") is not None:
        if (q("qb_designed_run") or 0) > 0.85:
            return "Designed-Run QB"
        if (q("qb_scramble") or 0) > 0.7 and (q("qb_rush_epa") or 0) > 0.7:
            return "Dual-Threat Creator"
        if (q("qb_adot") or 0) > 0.8 and (q("qb_deep_rate") or 0) > 0.65:
            return "Deep-Ball Aggressor"
        if (q("qb_adot") or 1) < 0.25 and (q("qb_comp") or 0) > 0.55:
            return "Quick-Game Distributor"
        if (q("qb_int_rate") or 1) < 0.3 and (q("qb_epa_db") or 0) > 0.6:
            return "High-Efficiency Game Manager"
        return "Pocket Passer / Distributor"
    if pos == "RB" and q("rb_expl") is not None:
        if (q("target_share") or 0) > 0.75:
            return "Receiving Back"
        if (q("rb_expl") or 0) > 0.75:
            return "Home-Run Hitter"
        return "Workhorse / Grinder"
    if pos in ("WR", "TE") and q(f"{pos.lower()}_adot") is not None:
        a = q(f"{pos.lower()}_adot")
        return "Vertical Threat" if a > 0.75 else "Underneath / YAC" if a < 0.25 else "Possession / Chain-Mover"
    g = lambda k: r.get(k) if pd.notna(r.get(k)) else None
    if pos in ("DL", "EDGE") and g("games"):
        return "Pass Rusher" if (g(f"{pos.lower()}_sack_rate") or 0) > 0.015 else "Run Defender"
    if pos == "LB" and g("games"):
        return "Coverage LB" if (g("lb_cov_events") or 0) > 0.25 else "Downhill Run Stopper"
    if pos in ("CB", "S") and g("games"):
        return "Ballhawk" if (g(f"{pos.lower()}_ball_pg") or 0) > 0.5 else "Tackler / Support"
    return None


def write_player_metrics(conn, season: int, through_week: int | None = None) -> pd.DataFrame:
    from ..db import insert_df
    L, W = compute_player_metrics(conn, season, through_week)
    wk = int(L.through_week.iloc[0])
    conn.execute("DELETE FROM player_metrics WHERE season=? AND through_week=?", (season, wk))
    insert_df(conn, "player_metrics", L[["player_id", "season", "through_week", "metric_id", "value", "sample",
                                          "pctl_fbs", "pctl_conf", "adj_value", "status"]])
    prof = W[["position", "team_id", "archetype", "grade", "grade_pctl"]].reset_index().rename(columns={"index": "player_id"})
    prof["season"] = season
    prof["through_week"] = wk
    prof.to_sql("player_profiles_tmp", conn, if_exists="replace", index=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS player_profiles (player_id INTEGER, season INTEGER, through_week INTEGER,
        position TEXT, team_id INTEGER, archetype TEXT, grade REAL, grade_pctl REAL, PRIMARY KEY (player_id, season, through_week))""")
    conn.execute("DELETE FROM player_profiles WHERE season=? AND through_week=?", (season, wk))
    conn.execute("""INSERT INTO player_profiles SELECT player_id, season, through_week, position, team_id, archetype, grade, grade_pctl
                    FROM player_profiles_tmp""")
    conn.execute("DROP TABLE player_profiles_tmp")
    conn.commit()
    return W
