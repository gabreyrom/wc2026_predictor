"""
Monte Carlo Tournament Simulation.

Used for two purposes:
    1. Cross-validation against exact group-stage probabilities
    2. Handling the 8 best 3rd-place qualification (cross-group comparison)
       which is intractable by exact enumeration (12 independent groups)

Each simulation run:
    a. Simulate all 12 groups (sample scorelines per match)
    b. Determine 1st, 2nd, and best-8 3rd-place qualifiers
    c. Simulate knockout bracket to find the winner

FIFA 2026 WC format:
    - 12 groups of 4
    - Top 2 + 8 best 3rd-place teams = 32 in Round of 32
    - Standard single-elimination bracket through to Final
"""

import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from src.model.dixon_coles import DixonColesModel, score_matrix
from src.simulation.group_stage import compute_standings, match_score_probs


# ── 3rd-place ranking criteria (FIFA rules) ──────────────────────────────────

THIRD_PLACE_TIEBREAKER = ["pts", "gd", "gf"]


def rank_third_place_teams(third_teams: list[dict]) -> list[dict]:
    """
    Given a list of 3rd-place team records (one per group), sort by:
    1. Points, 2. GD, 3. GF, then random draw.
    Return top 8.
    """
    ranked = sorted(
        third_teams,
        key=lambda t: (t["pts"], t["gd"], t["gf"]),
        reverse=True,
    )
    return ranked[:8]


# ── Single match sampler ─────────────────────────────────────────────────────

def sample_match(
    model: DixonColesModel,
    team_i: str,
    team_j: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Sample a single scoreline (goals_i, goals_j) from the Dixon-Coles model.
    Uses inverse CDF sampling on the score matrix.
    """
    lam = model.lambda_ij(team_i, team_j)
    mu  = model.lambda_ij(team_j, team_i)
    mat = score_matrix(lam, mu, model.rho, max_goals=7)

    flat = mat.flatten()
    flat /= flat.sum()

    idx = rng.choice(len(flat), p=flat)
    i, j = divmod(idx, mat.shape[1])
    return int(i), int(j)


def sample_knockout_winner(
    model: DixonColesModel,
    team_i: str,
    team_j: str,
    rng: np.random.Generator,
) -> str:
    """
    Sample a knockout match winner. If draw after 90 min, 50/50 shootout.
    """
    gi, gj = sample_match(model, team_i, team_j, rng)
    if gi > gj:
        return team_i
    elif gj > gi:
        return team_j
    else:
        return team_i if rng.random() < 0.5 else team_j


# ── Simulate a single group ───────────────────────────────────────────────────

def simulate_group(
    teams: list[str],
    model: DixonColesModel,
    rng: np.random.Generator,
) -> tuple[list[str], dict]:
    """
    Simulate one group: sample all 6 match results, return ranking and stats.

    Returns:
        ranking : list of teams [1st, 2nd, 3rd, 4th]
        stats   : {team: {pts, gd, gf}}
    """
    matches = [
        (teams[i], teams[j])
        for i in range(len(teams))
        for j in range(i + 1, len(teams))
    ]

    results: dict[tuple[str, str], tuple[int, int]] = {}
    stats: dict[str, dict] = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}

    for home, away in matches:
        hg, ag = sample_match(model, home, away, rng)
        results[(home, away)] = (hg, ag)
        stats[home]["gf"] += hg; stats[home]["gd"] += hg - ag
        stats[away]["gf"] += ag; stats[away]["gd"] += ag - hg
        if hg > ag:    stats[home]["pts"] += 3
        elif hg == ag: stats[home]["pts"] += 1; stats[away]["pts"] += 1
        else:           stats[away]["pts"] += 3

    ranking = compute_standings(teams, results)
    return ranking, stats


# ── Full tournament simulation ────────────────────────────────────────────────

def simulate_tournament(
    groups: dict[str, list[str]],
    model: DixonColesModel,
    rng: np.random.Generator,
) -> dict[str, str]:
    """
    Simulate one full World Cup tournament.

    Returns:
        round_reached : {team: last_round_reached}
            Possible values: 'Group', 'R32', 'R16', 'QF', 'SF', 'Final', 'Winner'
    """
    round_reached: dict[str, str] = {}
    for group_teams in groups.values():
        for t in group_teams:
            round_reached[t] = "Group"

    # ── Group stage ──────────────────────────────────────────────────────────
    qualifiers_by_group: dict[str, list[str]] = {}
    third_place_teams: list[dict] = []

    for group_name, teams in groups.items():
        ranking, stats = simulate_group(teams, model, rng)
        qualifiers_by_group[group_name] = ranking[:2]

        for i, team in enumerate(ranking):
            if i < 2:
                round_reached[team] = "R32"
            elif i == 2:
                third_place_teams.append({
                    "team":  team,
                    "group": group_name,
                    "pts":   stats[team]["pts"],
                    "gd":    stats[team]["gd"],
                    "gf":    stats[team]["gf"],
                })

    # ── 8 best 3rd-place teams ────────────────────────────────────────────────
    best_thirds = rank_third_place_teams(third_place_teams)
    for entry in best_thirds:
        round_reached[entry["team"]] = "R32"

    # ── Build R32 bracket ─────────────────────────────────────────────────────
    # FIFA 2026 bracket seeding: we use a simplified bracket
    # (full bracket seeding depends on which 3rd-place teams qualify)
    # For now: group winners on one side, runners-up + 3rd on the other
    r32_teams = []
    for group_name in sorted(groups.keys()):
        r32_teams.extend(qualifiers_by_group[group_name])
    for entry in best_thirds:
        r32_teams.append(entry["team"])

    # Shuffle 3rd-place into bracket positions (simplified)
    current_round_teams = r32_teams[:32]

    # ── Knockout rounds ───────────────────────────────────────────────────────
    round_names = ["R16", "QF", "SF", "Final"]
    remaining = current_round_teams[:]

    for round_name in round_names:
        next_round = []
        for k in range(0, len(remaining), 2):
            if k + 1 >= len(remaining):
                next_round.append(remaining[k])
                continue
            a, b = remaining[k], remaining[k + 1]
            winner = sample_knockout_winner(model, a, b, rng)
            loser  = b if winner == a else a
            round_reached[loser] = round_name
            next_round.append(winner)
        remaining = next_round

    if remaining:
        round_reached[remaining[0]] = "Winner"

    return round_reached


# ── Run N simulations ─────────────────────────────────────────────────────────

ROUND_ORDER = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]


def run_simulations(
    groups: dict[str, list[str]],
    model: DixonColesModel,
    n: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run N full tournament simulations.

    Returns a DataFrame with columns:
        team, Group, R32, R16, QF, SF, Final, Winner
    where each value is the probability of reaching that round.
    """
    rng = np.random.default_rng(seed)
    all_teams = [t for teams in groups.values() for t in teams]

    # Tally: round_counts[team][round] = count
    round_counts: dict[str, dict[str, int]] = {
        t: {r: 0 for r in ROUND_ORDER} for t in all_teams
    }

    print(f"Running {n:,} Monte Carlo simulations...")
    for _ in tqdm(range(n)):
        results = simulate_tournament(groups, model, rng)
        for team, last_round in results.items():
            idx = ROUND_ORDER.index(last_round)
            # A team "reached" all rounds up to and including last_round
            for r in ROUND_ORDER[:idx + 1]:
                round_counts[team][r] += 1

    rows = []
    for team in all_teams:
        row = {"team": team}
        for r in ROUND_ORDER:
            row[r] = round_counts[team][r] / n
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Winner", ascending=False).reset_index(drop=True)
    return df


# ── Cross-validation against exact enumeration ───────────────────────────────

def validate_against_exact(
    mc_results: pd.DataFrame,
    exact_probs: dict[str, float],
    tolerance: float = 0.01,
) -> None:
    """
    Compare MC group-stage qualification probabilities against exact enumeration.
    Prints warnings for teams where MC deviates by more than `tolerance`.
    """
    print("\n=== MC vs Exact cross-validation (top-2 qualification) ===")
    max_dev = 0.0
    for team, exact_p in exact_probs.items():
        mc_row = mc_results[mc_results["team"] == team]
        if mc_row.empty:
            continue
        mc_p = float(mc_row["R32"].values[0])
        dev = abs(mc_p - exact_p)
        max_dev = max(max_dev, dev)
        flag = " *** DEVIATION" if dev > tolerance else ""
        print(f"  {team:20s}  exact:{exact_p:.3f}  mc:{mc_p:.3f}  dev:{dev:.3f}{flag}")
    print(f"\nMax deviation: {max_dev:.4f}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.model.dixon_coles import fit
    from src.data.fetch_matches import fetch_and_process
    from tournament.wc2026_draw import GROUPS

    print("Loading data and fitting model...")
    df = fetch_and_process(force=False)
    model = fit(df, xi=0.003)

    results = run_simulations(GROUPS, model, n=10_000, seed=42)
    print("\n=== Top 20 teams by WC win probability ===")
    print(results[["team", "R32", "R16", "QF", "SF", "Final", "Winner"]].head(20).to_string(index=False))
