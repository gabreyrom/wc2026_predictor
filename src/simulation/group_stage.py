"""
Group Stage Simulation — Exact Enumeration.

For each group of 4 teams (6 matches), enumerate all 3^6 = 729 possible
outcome combinations. Weight each combination by the joint probability of
its match outcomes under Dixon-Coles. This gives exact (no Monte Carlo noise)
distributions for group standings.

FIFA 2026 tiebreaker rules (applied in order):
    1. Points
    2. Goal difference (within group)
    3. Goals scored (within group)
    4. Points in head-to-head matches (tied teams only)
    5. Goal difference in head-to-head matches
    6. Goals scored in head-to-head matches
    7. Drawing of lots (modelled as random uniform)
"""

import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import Generator
from tqdm import tqdm

from src.model.dixon_coles import DixonColesModel, score_matrix
from tournament.wc2026_draw import HOST_TEAMS


def host_flags(team_i: str, team_j: str) -> tuple[bool, bool]:
    """
    Home-advantage flags for a WC 2026 match. A host nation (USA/Mexico/Canada)
    plays in its own country; if both teams are hosts the venue is ambiguous
    and we treat it as neutral.
    """
    hi = team_i in HOST_TEAMS
    hj = team_j in HOST_TEAMS
    if hi and hj:
        return False, False
    return hi, hj


# ── Match outcome enumeration ─────────────────────────────────────────────────

# We discretise each match into 3 outcomes (home win, draw, away win) weighted
# by the score matrix. For standings we also need goal counts, so we sample
# a representative scoreline per outcome — or better: we enumerate the top-k
# most probable scorelines.
#
# Strategy: enumerate over (home_score, away_score) pairs up to MAX_GOALS.
# This is exact but more expensive: 11^2 * 6 matches * 12 groups still tractable.

MAX_GOALS_SIM = 7  # max goals per team for enumeration (covers >99% of probability)


def calibrated_score_matrix(
    model: DixonColesModel,
    home: str,
    away: str,
    calibrator=None,
    match_importance: float = 1.0,
    max_goals: int = MAX_GOALS_SIM,
) -> np.ndarray:
    """
    Score matrix for a WC fixture, with outcome-level calibration applied.

    With calibrator=None this is the raw (host-adjusted) Dixon-Coles matrix.
    With a fitted LGBMCalibrator, the three outcome regions (home win / draw /
    away win) are rescaled so their total mass matches the calibrated
    probabilities, then renormalised:

        M'[i,j] = M[i,j] · p_cal(outcome of (i,j)) / p_DC(outcome of (i,j))

    The conditional scoreline distribution WITHIN each outcome stays DC's —
    a documented approximation (margins of re-rated teams shift only at the
    outcome level, not within it).
    """
    h_home, h_away = host_flags(home, away)
    pred = model.predict(home, away, match_importance=match_importance,
                         home_i=h_home, home_j=h_away, max_goals=max_goals)
    mat = pred["score_matrix"].copy()
    if calibrator is None:
        return mat

    from src.data.market_values import log_value_ratio
    from src.data.elo import elo_diff
    pred["log_value_ratio"] = log_value_ratio(home, away, model.default_predict_date)
    pred["elo_diff"] = elo_diff(home, away, model.default_predict_date)
    cal = calibrator.predict_proba_row(pred)

    n = mat.shape[0]
    rows, cols = np.indices((n, n))
    for mask, outcome in [(rows > cols, "home"),
                          (rows == cols, "draw"),
                          (rows < cols, "away")]:
        p_dc = float(mat[mask].sum())
        if p_dc > 1e-12:
            mat[mask] *= cal[outcome] / p_dc

    mat /= mat.sum()
    return mat


def match_score_probs(
    model: DixonColesModel,
    home: str,
    away: str,
    match_importance: float = 1.0,
    calibrator=None,
) -> dict[tuple[int, int], float]:
    """
    Return a dict mapping (home_goals, away_goals) -> probability for all
    scorelines up to MAX_GOALS_SIM, optionally outcome-calibrated.

    Used by monte_carlo.py for inverse-CDF sampling. Not used by the exact
    group enumeration (which uses match_outcome_probs instead).
    """
    mat = calibrated_score_matrix(model, home, away, calibrator,
                                  match_importance, MAX_GOALS_SIM)
    result = {}
    for i in range(MAX_GOALS_SIM + 1):
        for j in range(MAX_GOALS_SIM + 1):
            if mat[i, j] > 1e-8:
                result[(i, j)] = float(mat[i, j])
    return result


def match_outcome_probs(
    model: DixonColesModel,
    home: str,
    away: str,
    match_importance: float = 1.0,
    calibrator=None,
) -> list[tuple[str, float, float, float]]:
    """
    Collapse the score matrix into 3 outcomes for exact group enumeration.
    With a calibrator, the outcome masses are the LGBM-calibrated ones.

    Returns a list of 3 tuples:
        (outcome, probability, exp_home_goals, exp_away_goals)
    where outcome ∈ {'H', 'D', 'A'} and exp_*_goals are the expected
    goal counts *conditioned on that outcome* — used for GD/GF tiebreakers.

    This is O(MAX_GOALS_SIM²) per match, called once per match pair.
    The calling code then enumerates 3^6 = 729 outcome combinations,
    not ~50^6 ≈ 15 billion scoreline combinations.
    """
    mat = calibrated_score_matrix(model, home, away, calibrator,
                                  match_importance, MAX_GOALS_SIM)

    n = mat.shape[0]
    idx = np.arange(n, dtype=float)
    home_g = idx[:, None]   # (n, 1) broadcast
    away_g = idx[None, :]   # (1, n) broadcast

    win_mask  = home_g > away_g   # home wins
    draw_mask = home_g == away_g  # draw
    lose_mask = home_g < away_g   # away wins

    def _cond(mask):
        p = float((mat * mask).sum())
        if p < 1e-12:
            return p, 0.0, 0.0
        e_h = float((mat * mask * home_g).sum()) / p
        e_a = float((mat * mask * away_g).sum()) / p
        return p, e_h, e_a

    p_w, eh_w, ea_w = _cond(win_mask)
    p_d, eh_d, ea_d = _cond(draw_mask)
    p_l, eh_l, ea_l = _cond(lose_mask)

    return [
        ("H", p_w, eh_w, ea_w),   # home win
        ("D", p_d, eh_d, ea_d),   # draw
        ("A", p_l, eh_l, ea_l),   # away win
    ]


# ── Group standings ───────────────────────────────────────────────────────────

def compute_standings(
    teams: list[str],
    results: dict[tuple[str, str], tuple[int, int]],
) -> list[str]:
    """
    Given a dict of match results {(home, away): (home_goals, away_goals)},
    return the teams sorted by FIFA group stage tiebreaker rules.

    Returns: list of teams from 1st to 4th place.
    """
    stats: dict[str, dict] = {
        t: {"pts": 0, "gd": 0, "gf": 0, "ga": 0} for t in teams
    }

    for (home, away), (hg, ag) in results.items():
        stats[home]["gf"] += hg
        stats[home]["ga"] += ag
        stats[home]["gd"] += hg - ag
        stats[away]["gf"] += ag
        stats[away]["ga"] += hg
        stats[away]["gd"] += ag - hg

        if hg > ag:
            stats[home]["pts"] += 3
        elif hg == ag:
            stats[home]["pts"] += 1
            stats[away]["pts"] += 1
        else:
            stats[away]["pts"] += 3

    def sort_key(team):
        s = stats[team]
        return (s["pts"], s["gd"], s["gf"])

    # First pass: sort by global group stats
    ranked = sorted(teams, key=sort_key, reverse=True)

    # Head-to-head tiebreaker for teams equal on pts/gd/gf
    ranked = _apply_h2h_tiebreaker(ranked, stats, results)

    return ranked


def _apply_h2h_tiebreaker(
    ranked: list[str],
    stats: dict,
    results: dict[tuple[str, str], tuple[int, int]],
) -> list[str]:
    """Apply head-to-head tiebreaker within tied groups of teams."""
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and (
            stats[ranked[j]]["pts"] == stats[ranked[i]]["pts"]
            and stats[ranked[j]]["gd"] == stats[ranked[i]]["gd"]
            and stats[ranked[j]]["gf"] == stats[ranked[i]]["gf"]
        ):
            j += 1

        if j - i > 1:
            tied = ranked[i:j]
            # Head-to-head among tied teams
            h2h_pts  = defaultdict(int)
            h2h_gd   = defaultdict(int)
            h2h_gf   = defaultdict(int)

            for a, b in itertools.combinations(tied, 2):
                for (home, away), (hg, ag) in results.items():
                    if {home, away} == {a, b}:
                        if home == a:
                            h2h_gf[a] += hg; h2h_gd[a] += hg - ag
                            h2h_gf[b] += ag; h2h_gd[b] += ag - hg
                            if hg > ag:   h2h_pts[a] += 3
                            elif hg == ag: h2h_pts[a] += 1; h2h_pts[b] += 1
                            else:          h2h_pts[b] += 3
                        else:
                            h2h_gf[b] += hg; h2h_gd[b] += hg - ag
                            h2h_gf[a] += ag; h2h_gd[a] += ag - hg
                            if hg > ag:   h2h_pts[b] += 3
                            elif hg == ag: h2h_pts[b] += 1; h2h_pts[a] += 1
                            else:          h2h_pts[a] += 3

            tied_sorted = sorted(
                tied,
                key=lambda t: (h2h_pts[t], h2h_gd[t], h2h_gf[t]),
                reverse=True,
            )
            ranked[i:j] = tied_sorted

        i = j

    return ranked


# ── Exact group enumeration ───────────────────────────────────────────────────

def enumerate_group(
    group_name: str,
    teams: list[str],
    model: DixonColesModel,
    calibrator=None,
    fixed_results: dict | None = None,
) -> tuple[dict[str, dict[str, float]], dict]:
    """
    Exactly enumerate all 3^6 = 729 outcome combinations for a group of 4 teams.

    Each of the 6 matches has 3 outcomes (H/D/A). For each combination we use:
      - Exact points from the W/D/L outcome
      - Expected goals conditioned on outcome for GD/GF tiebreakers

    This is O(729) per group — milliseconds per group vs. hours for full
    scoreline enumeration (~50^6 ≈ 15 billion combinations).

    Returns:
        position_probs : {team: {"1st": p, "2nd": p, "3rd": p, "4th": p,
                                 "exp_pts": float, "exp_gd": float, "exp_gf": float}}
        third_place_dist: {team: {(pts, gd, gf): prob}} for 3rd-place qualification
    """
    matches = [
        (teams[i], teams[j])
        for i in range(len(teams))
        for j in range(i + 1, len(teams))
    ]  # 6 matchups

    # Precompute 3-outcome distributions for each match (fast: O(MAX_GOALS_SIM²)).
    # PLAYED matches collapse to a single certain branch carrying the ACTUAL
    # goals — the enumeration conditions on reality, and tiebreakers use real
    # scorelines instead of expected goals for those matches.
    from src.data.wc_results import lookup_group_result
    outcome_data = []
    for h, a in matches:
        actual = lookup_group_result(fixed_results or {}, h, a)
        if actual is not None:
            hg, ag = actual
            outcome = "H" if hg > ag else ("D" if hg == ag else "A")
            outcome_data.append([(outcome, 1.0, float(hg), float(ag))])
        else:
            outcome_data.append(
                match_outcome_probs(model, h, a, calibrator=calibrator)
            )

    # Result accumulators
    position_probs: dict[str, dict[str, float]] = {
        t: {"1st": 0.0, "2nd": 0.0, "3rd": 0.0, "4th": 0.0,
            "exp_pts": 0.0, "exp_gd": 0.0, "exp_gf": 0.0}
        for t in teams
    }
    third_place_dist: dict[str, dict[tuple, float]] = {t: defaultdict(float) for t in teams}

    # Enumerate 3^6 = 729 outcome combinations
    for combo in itertools.product(*outcome_data):
        # combo: tuple of (outcome_str, prob, exp_home_g, exp_away_g) per match
        joint_prob = 1.0
        for _, p, _, _ in combo:
            joint_prob *= p

        if joint_prob < 1e-15:
            continue

        # Compute team stats for this combination
        stats: dict[str, dict] = {t: {"pts": 0, "gd": 0.0, "gf": 0.0} for t in teams}
        for match_idx, (outcome, _, e_hg, e_ag) in enumerate(combo):
            home, away = matches[match_idx]
            stats[home]["gf"] += e_hg
            stats[home]["gd"] += e_hg - e_ag
            stats[away]["gf"] += e_ag
            stats[away]["gd"] += e_ag - e_hg
            if outcome == "H":
                stats[home]["pts"] += 3
            elif outcome == "D":
                stats[home]["pts"] += 1
                stats[away]["pts"] += 1
            else:
                stats[away]["pts"] += 3

        # Sort by (pts, gd, gf) — tiebreakers use expected goals
        ranking = sorted(
            teams,
            key=lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"]),
            reverse=True,
        )

        for pos, team in enumerate(ranking):
            pos_name = ["1st", "2nd", "3rd", "4th"][pos]
            position_probs[team][pos_name]  += joint_prob
            position_probs[team]["exp_pts"] += joint_prob * stats[team]["pts"]
            position_probs[team]["exp_gd"]  += joint_prob * stats[team]["gd"]
            position_probs[team]["exp_gf"]  += joint_prob * stats[team]["gf"]

            if pos == 2:  # 3rd place — record (pts, gd, gf) for cross-group ranking
                key = (
                    stats[team]["pts"],
                    round(stats[team]["gd"], 1),
                    round(stats[team]["gf"], 1),
                )
                third_place_dist[team][key] += joint_prob

    return position_probs, dict(third_place_dist)


def simulate_all_groups(
    groups: dict[str, list[str]],
    model: DixonColesModel,
    calibrator=None,
    fixed_results: dict | None = None,
) -> tuple[dict, dict]:
    """
    Run exact enumeration for all groups.

    Returns:
        all_position_probs: {group: {team: {pos: prob}}}
        all_third_dists:    {group: {team: {(pts,gd,gf): prob}}}
    """
    all_position_probs = {}
    all_third_dists = {}
    for group_name, teams in tqdm(groups.items(), desc="Groups", unit="group"):
        pos_probs, third_dist = enumerate_group(group_name, teams, model,
                                                calibrator, fixed_results)
        all_position_probs[group_name] = pos_probs
        all_third_dists[group_name] = third_dist
    return all_position_probs, all_third_dists


def qualification_probs(
    all_position_probs: dict,
) -> dict[str, float]:
    """
    P(team qualifies as 1st or 2nd) — exact, no MC needed.
    Note: 3rd-place qualification requires MC (see monte_carlo.py).
    """
    result = {}
    for group, team_probs in all_position_probs.items():
        for team, probs in team_probs.items():
            result[team] = probs["1st"] + probs["2nd"]
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.model.dixon_coles import fit
    from src.data.fetch_matches import fetch_and_process

    print("Loading data and fitting model...")
    df = fetch_and_process(force=False)
    model = fit(df, xi=0.003)

    # Test on Group A
    from tournament.wc2026_draw import GROUPS
    pos_probs, third_dist = enumerate_group("A", GROUPS["A"], model)

    print("\n=== Group A Position Probabilities ===")
    for team, probs in pos_probs.items():
        print(f"  {team:20s}  1st:{probs['1st']:.1%}  2nd:{probs['2nd']:.1%}  "
              f"3rd:{probs['3rd']:.1%}  4th:{probs['4th']:.1%}  "
              f"E[pts]:{probs['exp_pts']:.2f}")
