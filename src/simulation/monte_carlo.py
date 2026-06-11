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
from src.simulation.group_stage import (
    compute_standings, match_score_probs, host_flags, calibrated_score_matrix,
)


# ── Precomputed match cache ───────────────────────────────────────────────────

class MatchCache:
    """
    Precomputes and stores the flat score-matrix CDF for every ordered team
    pair before the simulation loop. Each sample_match call then costs only
    a dict lookup + np.searchsorted instead of recomputing the score matrix.

    With a calibrator, the cached 90-minute matrices are outcome-calibrated
    (LGBM probabilities drive who wins; DC drives the scorelines within each
    outcome) — so the tournament simulation and the reported calibrated match
    probabilities are consistent.

    Also caches an EXTRA-TIME distribution per pair: a DC matrix at one-third
    rates (30 min of play), used when a knockout match is level after 90'.

    Speedup: ~100–500x vs. computing score_matrix inside each MC iteration.
    """

    def __init__(
        self,
        model: DixonColesModel,
        teams: list[str],
        match_importance: float = 1.0,
        max_goals: int = 7,
        calibrator=None,
    ) -> None:
        self._n = max_goals + 1
        self._n_et = 4 + 1                          # 0–4 goals in 30 minutes
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._et_cache: dict[tuple[str, str], np.ndarray] = {}

        for home in teams:
            for away in teams:
                if home == away:
                    continue
                # 90-minute matrix (outcome-calibrated when calibrator given)
                mat = calibrated_score_matrix(
                    model, home, away, calibrator, match_importance, max_goals,
                )
                flat = mat.ravel()
                self._cache[(home, away)] = np.cumsum(flat / flat.sum())

                # Extra-time matrix: 30 min ≈ one-third of the 90' rates.
                # Raw DC (no calibration — the calibrator was trained on
                # 90-minute outcomes only).
                h_home, h_away = host_flags(home, away)
                lam = model.lambda_ij(home, away, home=h_home)
                mu  = model.lambda_ij(away, home, home=h_away)
                rho = model.rho_for_match(
                    model._match_context(home, away, match_importance)
                )
                mat_et = score_matrix(lam / 3.0, mu / 3.0, rho,
                                      max_goals=self._n_et - 1)
                flat_et = mat_et.ravel()
                self._et_cache[(home, away)] = np.cumsum(flat_et / flat_et.sum())

    def sample(self, home: str, away: str, rng: np.random.Generator) -> tuple[int, int]:
        """Return (home_goals, away_goals) sampled from the precomputed CDF."""
        cdf = self._cache[(home, away)]
        idx = int(np.searchsorted(cdf, rng.random()))
        return divmod(idx, self._n)

    def sample_extra_time(self, home: str, away: str,
                          rng: np.random.Generator) -> tuple[int, int]:
        """Sample a 30-minute extra-time scoreline."""
        cdf = self._et_cache[(home, away)]
        idx = int(np.searchsorted(cdf, rng.random()))
        return divmod(idx, self._n_et)


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


# ── Official FIFA 2026 Round-of-32 bracket ────────────────────────────────────
# Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage
#
# R32 match definitions. "1X"/"2X" = winner/runner-up of group X;
# None marks a slot filled by a qualified 3rd-place team (see R32_THIRD_SLOTS).
R32_FIXED: dict[int, tuple[str, str | None]] = {
    73: ("2A", "2B"),  74: ("1E", None),  75: ("1F", "2C"),  76: ("1C", "2F"),
    77: ("1I", None),  78: ("2E", "2I"),  79: ("1A", None),  80: ("1L", None),
    81: ("1D", None),  82: ("1G", None),  83: ("2K", "2L"),  84: ("1H", "2J"),
    85: ("1B", None),  86: ("1J", "2H"),  87: ("1K", None),  88: ("2D", "2G"),
}

# Allowed source groups for each 3rd-place slot (FIFA's allocation constraints).
R32_THIRD_SLOTS: dict[int, frozenset[str]] = {
    74: frozenset("ABCDF"),  77: frozenset("CDFGH"),
    79: frozenset("CEFHI"),  80: frozenset("EHIJK"),
    81: frozenset("BEFIJ"),  82: frozenset("AEHIJ"),
    85: frozenset("EFGIJ"),  87: frozenset("DEIJL"),
}

# R32 matches ordered so that adjacent-pair halving reproduces the official
# flow: R16 (89:W74-W77, 90:W73-W75, 93:W83-W84, 94:W81-W82, 91:W76-W78,
# 92:W79-W80, 95:W86-W88, 96:W85-W87), QF (97:89-90, 98:93-94, 99:91-92,
# 100:95-96), SF (101:97-98, 102:99-100), Final (104:101-102).
R32_MATCH_ORDER = [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87]

# Official match numbers for each knockout round, in bracket-list order
# (pair k of the remaining list plays the k-th number of its round).
ROUND_MATCH_NUMBERS = {
    32: R32_MATCH_ORDER,
    16: [89, 90, 93, 94, 91, 92, 95, 96],
    8:  [97, 98, 99, 100],
    4:  [101, 102],
    2:  [104],
}
THIRD_PLACE_MATCH = 103   # SF losers


def assign_third_place_slots(qualified_groups: set[str]) -> dict[int, str]:
    """
    Assign the 8 qualified 3rd-place groups to the 8 bracket slots, respecting
    FIFA's allowed-group constraints for each slot.

    FIFA publishes this as a 495-row lookup table (one row per C(12,8)
    combination); the underlying rule is a constrained perfect matching, which
    we solve by backtracking (most-constrained slot first). FIFA designed the
    slot lists so every combination admits at least one valid assignment.

    Returns {match_number: group_letter}.
    """
    slots = sorted(
        R32_THIRD_SLOTS,
        key=lambda m: len(R32_THIRD_SLOTS[m] & qualified_groups),
    )
    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(i: int) -> bool:
        if i == len(slots):
            return True
        m = slots[i]
        for g in sorted(R32_THIRD_SLOTS[m] & qualified_groups - used):
            assignment[m] = g
            used.add(g)
            if backtrack(i + 1):
                return True
            del assignment[m]
            used.discard(g)
        return False

    if not backtrack(0):
        raise ValueError(
            f"No valid 3rd-place assignment for groups {sorted(qualified_groups)}"
        )
    return assignment


def build_r32_bracket(
    qualifiers_by_group: dict[str, list[str]],
    best_thirds: list[dict],
) -> list[str]:
    """
    Build the 32-team R32 list in official bracket order from group results.

    Args:
        qualifiers_by_group : {group: [winner, runner_up]}
        best_thirds         : list of dicts with 'team' and 'group' keys
                              (the 8 qualified 3rd-place teams)

    Returns a list of 32 team names; adjacent pairs play each other, and
    iterated halving follows the official R16/QF/SF/Final flow.
    """
    third_by_group = {e["group"]: e["team"] for e in best_thirds}
    slot_assignment = assign_third_place_slots(set(third_by_group))

    def resolve(code: str | None, match_no: int) -> str:
        if code is None:
            return third_by_group[slot_assignment[match_no]]
        pos, group = int(code[0]), code[1]
        return qualifiers_by_group[group][pos - 1]

    bracket: list[str] = []
    for m in R32_MATCH_ORDER:
        home, away = R32_FIXED[m]
        bracket.append(resolve(home, m))
        bracket.append(resolve(away, m))
    return bracket


# ── Single match sampler ─────────────────────────────────────────────────────

def sample_match(
    model: DixonColesModel,
    team_i: str,
    team_j: str,
    rng: np.random.Generator,
    match_importance: float = 1.0,
    cache: MatchCache | None = None,
) -> tuple[int, int]:
    """
    Sample a single scoreline (goals_i, goals_j) from the Dixon-Coles model.

    If a MatchCache is provided (recommended), uses a precomputed CDF lookup —
    orders of magnitude faster than recomputing the score matrix each call.
    """
    if cache is not None:
        return cache.sample(team_i, team_j, rng)

    # Fallback: compute on the fly (slow — avoid in tight loops)
    h_i, h_j = host_flags(team_i, team_j)
    lam = model.lambda_ij(team_i, team_j, home=h_i)
    mu  = model.lambda_ij(team_j, team_i, home=h_j)
    rho = model.rho_for_match(model._match_context(team_i, team_j, match_importance))
    mat = score_matrix(lam, mu, rho, max_goals=7)
    flat = mat.ravel()
    flat = flat / flat.sum()
    idx = int(np.searchsorted(np.cumsum(flat), rng.random()))
    return divmod(idx, mat.shape[1])


def sample_knockout_winner(
    model: DixonColesModel,
    team_i: str,
    team_j: str,
    rng: np.random.Generator,
    cache: MatchCache | None = None,
) -> str:
    """
    Sample a knockout match winner:
        90 minutes → if level, 30 minutes of extra time at one-third scoring
        rates (favors the stronger team) → if still level, 50/50 penalties.

    Penalties as a coin flip is supported by the literature — shootout
    outcomes correlate only weakly with team strength.
    """
    gi, gj = sample_match(model, team_i, team_j, rng, cache=cache)
    if gi != gj:
        return team_i if gi > gj else team_j

    # Extra time — stronger team gets a real edge here
    if cache is not None:
        ei, ej = cache.sample_extra_time(team_i, team_j, rng)
        if ei != ej:
            return team_i if ei > ej else team_j

    # Penalty shootout
    return team_i if rng.random() < 0.5 else team_j


# ── Simulate a single group ───────────────────────────────────────────────────

def simulate_group(
    teams: list[str],
    model: DixonColesModel,
    rng: np.random.Generator,
    cache: MatchCache | None = None,
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
        hg, ag = sample_match(model, home, away, rng, cache=cache)
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
    cache: MatchCache | None = None,
    pairing_counts: dict | None = None,
    top2_counts: dict | None = None,
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
        ranking, stats = simulate_group(teams, model, rng, cache=cache)
        qualifiers_by_group[group_name] = ranking[:2]
        if top2_counts is not None:
            for t in ranking[:2]:
                top2_counts[t] = top2_counts.get(t, 0) + 1

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

    # ── Build R32 bracket (official FIFA 2026 structure) ─────────────────────
    remaining = build_r32_bracket(qualifiers_by_group, best_thirds)
    assert len(remaining) == 32

    # ── Knockout rounds ───────────────────────────────────────────────────────
    # 5 rounds: 32 → 16 → 8 → 4 → 2 → champion.
    # Winners of each round are promoted to the NEXT round's label;
    # losers keep the label of the round they were eliminated in.
    advance_labels = ["R16", "QF", "SF", "Final", "Winner"]
    sf_losers: list[str] = []

    for label in advance_labels:
        match_numbers = ROUND_MATCH_NUMBERS[len(remaining)]
        next_round = []
        for k in range(0, len(remaining), 2):
            a, b = remaining[k], remaining[k + 1]
            if pairing_counts is not None:
                m_no = match_numbers[k // 2]
                key = (a, b)
                pairing_counts.setdefault(m_no, {})
                pairing_counts[m_no][key] = pairing_counts[m_no].get(key, 0) + 1
            winner = sample_knockout_winner(model, a, b, rng, cache=cache)
            round_reached[winner] = label
            if label == "Final":   # SF round: record losers for 3rd-place match
                sf_losers.append(b if winner == a else a)
            next_round.append(winner)
        remaining = next_round

    if pairing_counts is not None and len(sf_losers) == 2:
        key = (sf_losers[0], sf_losers[1])
        pairing_counts.setdefault(THIRD_PLACE_MATCH, {})
        pairing_counts[THIRD_PLACE_MATCH][key] = \
            pairing_counts[THIRD_PLACE_MATCH].get(key, 0) + 1

    return round_reached


# ── Run N simulations ─────────────────────────────────────────────────────────

ROUND_ORDER = ["Group", "R32", "R16", "QF", "SF", "Final", "Winner"]


def run_simulations(
    groups: dict[str, list[str]],
    model: DixonColesModel,
    n: int = 100_000,
    seed: int = 42,
    return_pairings: bool = False,
    calibrator=None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Run N full tournament simulations.

    Returns a DataFrame with columns:
        team, Group, R32, R16, QF, SF, Final, Winner
    where each value is the probability of reaching that round.

    With return_pairings=True, also returns
        {match_no: {(team_a, team_b): probability}}
    for every knockout match slot (73–104), where probability is the share
    of simulations in which that exact pairing occurred.
    """
    rng = np.random.default_rng(seed)
    all_teams = [t for teams in groups.values() for t in teams]

    # Build match cache — precomputes score-matrix CDFs for all team pairs.
    # One-time cost (~2,256 matrices for 48 teams) replaces per-match
    # score_matrix recomputation inside the simulation loop.
    print(f"  Precomputing match distributions for {len(all_teams)} teams"
          f"{' (LGBM-calibrated)' if calibrator is not None else ''}...", flush=True)
    cache = MatchCache(model, all_teams, match_importance=1.0, max_goals=7,
                       calibrator=calibrator)

    # Tally: round_counts[team][round] = count
    round_counts: dict[str, dict[str, int]] = {
        t: {r: 0 for r in ROUND_ORDER} for t in all_teams
    }

    pairing_counts: dict = {} if return_pairings else None
    top2_counts: dict = {}

    print(f"  Running {n:,} simulations...")
    for _ in tqdm(range(n), unit="sim"):
        results = simulate_tournament(groups, model, rng, cache=cache,
                                      pairing_counts=pairing_counts,
                                      top2_counts=top2_counts)
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
        row["top2"] = top2_counts.get(team, 0) / n   # group top-2 (no 3rd route)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Winner", ascending=False).reset_index(drop=True)

    if return_pairings:
        pairing_probs = {
            m_no: {pair: cnt / n for pair, cnt in pairs.items()}
            for m_no, pairs in pairing_counts.items()
        }
        return df, pairing_probs
    return df


# ── Cross-validation against exact enumeration ───────────────────────────────

def validate_against_exact(
    mc_results: pd.DataFrame,
    exact_probs: dict[str, float],
    tolerance: float = 0.02,
) -> None:
    """
    Compare MC top-2 group qualification probabilities against the exact
    enumeration. This is a like-for-like comparison (the previous version
    wrongly compared exact top-2 vs MC R32, which includes the 3rd-place
    qualification route and always 'deviated').

    Residual deviation sources: MC sampling noise and the enumeration's
    expected-goals tiebreaker approximation vs MC's sampled scorelines.
    """
    print("\n=== MC vs Exact cross-validation (top-2 qualification) ===")
    max_dev = 0.0
    n_flagged = 0
    for team, exact_p in exact_probs.items():
        mc_row = mc_results[mc_results["team"] == team]
        if mc_row.empty or "top2" not in mc_row.columns:
            continue
        mc_p = float(mc_row["top2"].values[0])
        dev = abs(mc_p - exact_p)
        max_dev = max(max_dev, dev)
        if dev > tolerance:
            n_flagged += 1
            print(f"  {team:20s}  exact:{exact_p:.3f}  mc:{mc_p:.3f}  "
                  f"dev:{dev:.3f} *** DEVIATION")
    print(f"  {n_flagged} teams above tolerance ({tolerance:.0%}); "
          f"max deviation: {max_dev:.4f}")


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
