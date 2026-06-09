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

from src.model.dixon_coles import DixonColesModel, score_matrix


# ── Match outcome enumeration ─────────────────────────────────────────────────

# We discretise each match into 3 outcomes (home win, draw, away win) weighted
# by the score matrix. For standings we also need goal counts, so we sample
# a representative scoreline per outcome — or better: we enumerate the top-k
# most probable scorelines.
#
# Strategy: enumerate over (home_score, away_score) pairs up to MAX_GOALS.
# This is exact but more expensive: 11^2 * 6 matches * 12 groups still tractable.

MAX_GOALS_SIM = 7  # max goals per team for enumeration (covers >99% of probability)


def match_score_probs(
    model: DixonColesModel,
    home: str,
    away: str,
) -> dict[tuple[int, int], float]:
    """
    Return a dict mapping (home_goals, away_goals) -> probability for all
    scorelines up to MAX_GOALS_SIM.
    """
    mat = score_matrix(
        model.lambda_ij(home, away),
        model.lambda_ij(away, home),
        model.rho,
        max_goals=MAX_GOALS_SIM,
    )
    result = {}
    for i in range(MAX_GOALS_SIM + 1):
        for j in range(MAX_GOALS_SIM + 1):
            if mat[i, j] > 1e-8:
                result[(i, j)] = float(mat[i, j])
    return result


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
) -> dict[str, dict[str, float]]:
    """
    Exactly enumerate all possible outcomes for a group of 4 teams.

    For each team, returns a probability distribution over finishing positions:
        {team: {"1st": p, "2nd": p, "3rd": p, "4th": p,
                "pts": expected_pts, "gd": expected_gd, "gf": expected_gf,
                "3rd_pts": P(3rd with N pts) dict}}

    NOTE: For the 8 best 3rd-place teams qualification, we also record
    the distribution of (pts, gd, gf) for each team's 3rd-place outcomes.
    """
    matches = [
        (teams[i], teams[j])
        for i in range(len(teams))
        for j in range(i + 1, len(teams))
    ]  # 6 matchups

    # Precompute score distributions for each match
    score_dists = {(h, a): match_score_probs(model, h, a) for h, a in matches}

    # Enumerate all scoreline combinations
    # Each match has a set of (score, prob) pairs
    match_score_lists = [list(score_dists[(h, a)].items()) for h, a in matches]

    # Result accumulators
    position_probs: dict[str, dict[str, float]] = {
        t: {"1st": 0.0, "2nd": 0.0, "3rd": 0.0, "4th": 0.0,
            "exp_pts": 0.0, "exp_gd": 0.0, "exp_gf": 0.0}
        for t in teams
    }
    # For 3rd-place qualification: track (pts, gd, gf) distribution per team
    third_place_dist: dict[str, dict[tuple, float]] = {t: defaultdict(float) for t in teams}

    total_combinations = 1
    for scores in match_score_lists:
        total_combinations *= len(scores)

    # Iterate all combinations (product of per-match score options)
    for combo in itertools.product(*match_score_lists):
        # combo is a tuple of ((hg, ag), prob) for each match
        joint_prob = 1.0
        results: dict[tuple[str, str], tuple[int, int]] = {}

        for (match_idx, ((hg, ag), p)) in enumerate(combo):
            home, away = matches[match_idx]
            joint_prob *= p
            results[(home, away)] = (hg, ag)

        if joint_prob < 1e-12:
            continue

        ranking = compute_standings(teams, results)

        # Accumulate stats for each team
        team_stats: dict[str, dict] = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        for (home, away), (hg, ag) in results.items():
            team_stats[home]["gf"] += hg; team_stats[home]["gd"] += hg - ag
            team_stats[away]["gf"] += ag; team_stats[away]["gd"] += ag - hg
            if hg > ag:   team_stats[home]["pts"] += 3
            elif hg == ag: team_stats[home]["pts"] += 1; team_stats[away]["pts"] += 1
            else:          team_stats[away]["pts"] += 3

        for pos, team in enumerate(ranking):
            pos_name = ["1st", "2nd", "3rd", "4th"][pos]
            position_probs[team][pos_name]    += joint_prob
            position_probs[team]["exp_pts"]   += joint_prob * team_stats[team]["pts"]
            position_probs[team]["exp_gd"]    += joint_prob * team_stats[team]["gd"]
            position_probs[team]["exp_gf"]    += joint_prob * team_stats[team]["gf"]

            if pos == 2:  # 3rd place
                key = (
                    team_stats[team]["pts"],
                    team_stats[team]["gd"],
                    team_stats[team]["gf"],
                )
                third_place_dist[team][key] += joint_prob

    print(f"Group {group_name}: enumerated {total_combinations:,} combinations")
    return position_probs, dict(third_place_dist)


def simulate_all_groups(
    groups: dict[str, list[str]],
    model: DixonColesModel,
) -> tuple[dict, dict]:
    """
    Run exact enumeration for all groups.

    Returns:
        all_position_probs: {group: {team: {pos: prob}}}
        all_third_dists:    {group: {team: {(pts,gd,gf): prob}}}
    """
    all_position_probs = {}
    all_third_dists = {}
    for group_name, teams in groups.items():
        pos_probs, third_dist = enumerate_group(group_name, teams, model)
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
