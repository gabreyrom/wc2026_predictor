"""
Output module — formats and prints prediction results.

Provides:
    - Win probability tables
    - Modal bracket (most likely team at each node)
    - Sensitivity analysis (Elo perturbation)
"""

import pandas as pd
import numpy as np

from src.model.dixon_coles import DixonColesModel
from tournament.wc2026_draw import GROUPS


ROUND_COLS = ["R32", "R16", "QF", "SF", "Final", "Winner"]


# ── Probability table ─────────────────────────────────────────────────────────

def print_prob_table(
    mc_results: pd.DataFrame,
    top_n: int = 32,
) -> None:
    """Print formatted probability table sorted by WC win probability."""
    cols = ["team"] + ROUND_COLS
    display = mc_results[cols].head(top_n).copy()
    for col in ROUND_COLS:
        display[col] = display[col].map(lambda x: f"{x:.1%}")

    print("\n" + "=" * 75)
    print(f"{'FIFA WORLD CUP 2026 — WIN PROBABILITIES':^75}")
    print("=" * 75)
    print(f"{'Team':<22} {'R32':>6} {'R16':>6} {'QF':>6} {'SF':>6} {'Final':>7} {'Win':>7}")
    print("-" * 75)
    for _, row in display.iterrows():
        print(
            f"{row['team']:<22} "
            f"{row['R32']:>6} {row['R16']:>6} {row['QF']:>6} "
            f"{row['SF']:>6} {row['Final']:>7} {row['Winner']:>7}"
        )
    print("=" * 75)


# ── Modal bracket ─────────────────────────────────────────────────────────────

def modal_bracket(
    groups: dict[str, list[str]],
    group_probs: dict,   # from group_stage.simulate_all_groups
    mc_results: pd.DataFrame,
) -> None:
    """Print the most likely bracket path."""
    print("\n" + "=" * 60)
    print(f"{'MODAL BRACKET — WC 2026':^60}")
    print("=" * 60)

    print("\nGroup Stage — Predicted Qualifiers:")
    print(f"{'Group':<8} {'1st':^20} {'2nd':^20} {'Best 3rd'}")
    print("-" * 60)

    for group_name, team_probs in group_probs.items():
        sorted_teams = sorted(
            team_probs.items(),
            key=lambda x: x[1]["1st"],
            reverse=True,
        )
        pred_1st = sorted_teams[0][0]
        pred_2nd = max(
            [(t, p) for t, p in team_probs.items() if t != pred_1st],
            key=lambda x: x[1]["2nd"],
        )[0]
        print(f"  {group_name:<6} {pred_1st:<20} {pred_2nd:<20}")

    # Knockout modal winner
    top_team = mc_results.iloc[0]
    print(f"\nPredicted Champion: {top_team['team']} "
          f"(P={top_team['Winner']:.1%})")
    print("=" * 60)


# ── Sensitivity analysis ──────────────────────────────────────────────────────

def sensitivity_analysis(
    model: DixonColesModel,
    groups: dict[str, list[str]],
    teams_to_perturb: list[str],
    n_mc: int = 5_000,
    elo_delta: float = 50.0,
    seed: int = 99,
) -> pd.DataFrame:
    """
    For each team in teams_to_perturb, shift its Elo by ±elo_delta and
    re-run a quick MC simulation. Reports change in WC win probability.

    This exposes model fragility: teams whose win probability moves a lot
    with small Elo changes are at the edge of a phase transition in the bracket.

    Returns a DataFrame with columns:
        team, base_win, up_win, down_win, up_delta, down_delta
    """
    from src.simulation.monte_carlo import run_simulations

    # Base run
    print("\nRunning base MC for sensitivity analysis...")
    base_mc = run_simulations(groups, model, n=n_mc, seed=seed)
    base_wins = dict(zip(base_mc["team"], base_mc["Winner"]))

    rows = []
    for team in teams_to_perturb:
        if team not in model.alpha:
            print(f"  Skipping {team} (not in model)")
            continue

        base_alpha = model.alpha[team]

        # Elo +50 -> alpha += 50/400 * log(10) ≈ +0.2876
        elo_to_alpha = elo_delta / 400 * np.log(10)

        # Perturb up
        model.alpha[team] = base_alpha + elo_to_alpha
        up_mc = run_simulations(groups, model, n=n_mc, seed=seed)
        up_win = dict(zip(up_mc["team"], up_mc["Winner"])).get(team, 0)

        # Perturb down
        model.alpha[team] = base_alpha - elo_to_alpha
        down_mc = run_simulations(groups, model, n=n_mc, seed=seed)
        down_win = dict(zip(down_mc["team"], down_mc["Winner"])).get(team, 0)

        # Restore
        model.alpha[team] = base_alpha

        rows.append({
            "team":       team,
            "base_win":   base_wins.get(team, 0),
            "up_win":     up_win,
            "down_win":   down_win,
            "up_delta":   up_win - base_wins.get(team, 0),
            "down_delta": down_win - base_wins.get(team, 0),
        })
        print(f"  {team}: base={base_wins.get(team,0):.1%}  "
              f"+{elo_delta:.0f}Elo={up_win:.1%}  -{elo_delta:.0f}Elo={down_win:.1%}")

    return pd.DataFrame(rows)


# ── Tournament advancement table ─────────────────────────────────────────────

def print_tournament_table(
    mc_results: pd.DataFrame,
    groups: dict[str, list[str]],
    sort_by: str = "Winner",
) -> None:
    """
    Print a grouped tournament advancement probability table.

    Columns: Group | Team | R32 | R16 | QF | SF | Final | Winner

    Teams are shown within their group (sorted by Winner prob within group),
    then groups are ordered by the highest Winner prob team in each group.

    Args:
        mc_results : DataFrame from run_simulations() — must have ROUND_COLS
        groups     : GROUPS dict from wc2026_draw.py
        sort_by    : column to use for global sort (default 'Winner')
    """
    # Build group membership lookup
    team_to_group: dict[str, str] = {}
    for g, teams in groups.items():
        for t in teams:
            team_to_group[t] = g

    # Attach group label
    df = mc_results.copy()
    df["Group"] = df["team"].map(team_to_group).fillna("?")

    # Sort groups by best team's Winner probability
    group_best = (
        df.groupby("Group")[sort_by].max()
        .sort_values(ascending=False)
    )
    group_order = list(group_best.index)

    W = 76
    print("\n" + "═" * W)
    print(f"{'WC 2026 — TOURNAMENT ADVANCEMENT PROBABILITIES':^{W}}")
    print("═" * W)
    print(f"  {'Grp':<4} {'Team':<22} {'R32':>6} {'R16':>6} "
          f"{'QF':>6} {'SF':>6} {'Final':>7} {'Win':>7}")
    print("  " + "─" * (W - 2))

    for g in group_order:
        group_df = (
            df[df["Group"] == g]
            .sort_values(sort_by, ascending=False)
        )
        for _, row in group_df.iterrows():
            team = str(row["team"])
            # Truncate long names to fit column
            team_display = team if len(team) <= 22 else team[:19] + "…"
            print(
                f"  {g:<4} {team_display:<22} "
                f"{row['R32']:>6.1%} {row['R16']:>6.1%} "
                f"{row['QF']:>6.1%} {row['SF']:>6.1%} "
                f"{row['Final']:>7.1%} {row['Winner']:>7.1%}"
            )
        print("  " + "·" * (W - 2))   # group separator

    print("═" * W)


# ── Save to CSV ───────────────────────────────────────────────────────────────

def save_results(mc_results: pd.DataFrame, path: str = "data/processed/wc2026_probs.csv") -> None:
    mc_results.to_csv(path, index=False)
    print(f"Saved results -> {path}")


if __name__ == "__main__":
    print("Run main.py to generate full results.")
