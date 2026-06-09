"""
WC 2026 Predictor — Main Entry Point

Pipeline:
    1. Fetch & process historical match data
    2. Compute Elo ratings
    3. Fit Dixon-Coles model
    4. Exact group-stage enumeration
    5. Monte Carlo simulation (N=100k)
    6. Exact bracket propagation
    7. Cross-validation + output
"""

import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.data.fetch_matches import fetch_and_process
from src.data.elo import compute_elo_ratings, save_ratings_to_csv
from src.model.dixon_coles import fit as fit_dixon_coles
from src.simulation.group_stage import simulate_all_groups, qualification_probs
from src.simulation.monte_carlo import run_simulations, validate_against_exact
from src.simulation.bracket import BracketPropagator, build_r32_slots_from_mc
from src.output.results import print_prob_table, modal_bracket, save_results
from tournament.wc2026_draw import GROUPS


def main(
    force_download: bool = False,
    n_mc: int = 100_000,
    xi: float = 0.003,
    seed: int = 42,
):
    print("=" * 60)
    print("   FIFA WORLD CUP 2026 PREDICTOR")
    print("=" * 60)

    # ── Step 1: Data ─────────────────────────────────────────────────────────
    print("\n[1/6] Fetching and processing match data...")
    matches = fetch_and_process(force=force_download)
    print(f"      {len(matches):,} matches loaded "
          f"({matches['date'].min().year}–{matches['date'].max().year})")

    # ── Step 2: Elo ratings ──────────────────────────────────────────────────
    print("\n[2/6] Computing Elo ratings...")
    elo_ratings = compute_elo_ratings(matches, time_weight=True, xi=xi)
    save_ratings_to_csv(elo_ratings, "data/processed/elo_ratings.csv")

    # Print top 10 WC teams by Elo
    wc_teams = set(t for teams in GROUPS.values() for t in teams)
    wc_elos = {t: elo_ratings.get(t, 1500) for t in wc_teams}
    print("\n  Top 10 WC 2026 teams by Elo:")
    for team, elo in sorted(wc_elos.items(), key=lambda x: -x[1])[:10]:
        print(f"    {team:<20s} {elo:.0f}")

    # ── Step 3: Dixon-Coles ──────────────────────────────────────────────────
    print("\n[3/6] Fitting Dixon-Coles model...")
    fit_data = matches[matches["date"] >= "2000-01-01"].copy()
    model = fit_dixon_coles(fit_data, xi=xi)
    print(f"      rho = {model.rho:.4f}  (Dixon-Coles low-score correction)")

    # ── Step 4: Exact group stage ────────────────────────────────────────────
    print("\n[4/6] Exact group stage enumeration (3^6 per group)...")
    group_pos_probs, group_third_dists = simulate_all_groups(GROUPS, model)
    exact_qual_probs = qualification_probs(group_pos_probs)

    # ── Step 5: Monte Carlo ──────────────────────────────────────────────────
    print(f"\n[5/6] Monte Carlo simulation (n={n_mc:,})...")
    mc_results = run_simulations(GROUPS, model, n=n_mc, seed=seed)
    validate_against_exact(mc_results, exact_qual_probs)

    # ── Step 6: Exact bracket propagation ────────────────────────────────────
    print("\n[6/6] Exact bracket propagation...")
    slots = build_r32_slots_from_mc(GROUPS, mc_results)
    propagator = BracketPropagator(model, slots)
    bracket_probs = propagator.reach_probs()

    # Merge exact bracket probs into results for knockout rounds
    for team, probs in bracket_probs.items():
        for round_name, p in probs.items():
            if round_name in ["QF", "SF", "Final", "Winner"]:
                mask = mc_results["team"] == team
                if mask.any() and round_name in mc_results.columns:
                    mc_results.loc[mask, round_name] = p

    mc_results = mc_results.sort_values("Winner", ascending=False).reset_index(drop=True)

    # ── Output ───────────────────────────────────────────────────────────────
    print_prob_table(mc_results, top_n=32)
    modal_bracket(GROUPS, group_pos_probs, mc_results)
    save_results(mc_results)

    print("\nDone.")
    return mc_results, model, group_pos_probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WC 2026 Predictor")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download match data even if cached")
    parser.add_argument("--n-mc", type=int, default=100_000,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        force_download=args.force_download,
        n_mc=args.n_mc,
        seed=args.seed,
    )
