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
from src.model.calibration import (
    temporal_split,
    calibration_report,
    calibration_report_with_lgbm,
    predict_matches,
)
from src.model.lgbm_calibrator import LGBMCalibrator
from src.simulation.group_stage import simulate_all_groups, qualification_probs
from src.simulation.monte_carlo import run_simulations, validate_against_exact
from src.output.results import (
    print_prob_table,
    print_tournament_table,
    modal_bracket,
    save_results,
    save_match_scorelines,
)
from src.output.match_report import print_match_report
from tournament.wc2026_draw import GROUPS



def main(
    force_download: bool = False,
    n_mc: int = 100_000,
    xi: float = 0.003,
    seed: int = 42,
    skip_calibration: bool = False,
    skip_lgbm: bool = False,
    save_calibrator: bool = True,
    match: tuple[str, str] | None = None,
    n_bootstrap: int = 500,
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

    # ── Step 3: Dixon-Coles (production model) ───────────────────────────────
    print("\n[3/6] Fitting Dixon-Coles model (production — all 2010+ data)...")
    fit_data = matches[matches["date"] >= "2010-01-01"].copy()
    model = fit_dixon_coles(fit_data, xi=xi)
    print(f"      baseline rho (zero context) = {model.rho:.4f}")

    # ── Step 3.5 & 3.7: Honest evaluation + LightGBM calibration layer ───────
    # Leakage-free design: a SECOND model is fitted on pre-2018 data only.
    # The production model above saw the val/test years, so scoring it there
    # would grade it on its own training data. The eval model never saw
    # val/test, making the report a true out-of-sample estimate — and the
    # LGBM calibrator trains on the biases of a model facing genuinely unseen
    # matches, which is the situation the production model will be in at the WC.
    calibrator: LGBMCalibrator | None = None

    if not skip_calibration:
        print("\n[3.5/6] Temporal cross-validation "
              "(fitting separate eval model on pre-2018 train only)...")
        train, val, test = temporal_split(
            fit_data,
            val_start="2018-01-01",
            test_start="2022-01-01",
        )
        eval_model = fit_dixon_coles(train, xi=xi)
        calibration_report(eval_model, val, test)

        # ── Step 3.7: LightGBM calibration layer ─────────────────────────────
        if not skip_lgbm:
            print("\n[3.7/6] Fitting LightGBM calibration layer "
                  "(on eval model's out-of-sample predictions)...")
            _, calibrator = calibration_report_with_lgbm(eval_model, val, test)

            if save_calibrator:
                calibrator.save("models/lgbm_calibrator.joblib")

    # ── Step 4: Exact group stage ────────────────────────────────────────────
    print("\n[4/6] Exact group stage enumeration (3^6 per group)...")
    group_pos_probs, group_third_dists = simulate_all_groups(GROUPS, model)
    exact_qual_probs = qualification_probs(group_pos_probs)

    # ── Step 5: Monte Carlo ──────────────────────────────────────────────────
    print(f"\n[5/6] Monte Carlo simulation (n={n_mc:,})...")
    mc_results = run_simulations(GROUPS, model, n=n_mc, seed=seed)
    validate_against_exact(mc_results, exact_qual_probs)

    # ── Step 6: Sanity checks on MC output ───────────────────────────────────
    # (The previous "exact bracket propagation" overwrite was removed: its
    #  R32 slot construction was incoherent — top-32 teams by R32 prob with
    #  non-normalised slot probabilities — and corrupted the Winner column.
    #  MC with the proper 5-round bracket is now the single source of truth.)
    print("\n[6/6] Validating MC output...")
    winner_sum = mc_results["Winner"].sum()
    print(f"      Sum of Winner probabilities: {winner_sum:.4f} (should be ~1.0)")
    for hi, lo in [("R32", "R16"), ("R16", "QF"), ("QF", "SF"),
                   ("SF", "Final"), ("Final", "Winner")]:
        violations = (mc_results[lo] > mc_results[hi] + 1e-9).sum()
        if violations:
            print(f"      WARNING: {violations} teams have P({lo}) > P({hi})")

    mc_results = mc_results.sort_values("Winner", ascending=False).reset_index(drop=True)

    # ── Output ───────────────────────────────────────────────────────────────
    print_tournament_table(mc_results, GROUPS)
    modal_bracket(GROUPS, group_pos_probs, mc_results)
    save_results(mc_results, group_pos_probs=group_pos_probs)
    save_match_scorelines(model, GROUPS, top_n=5)

    # ── Optional: single match deep dive ─────────────────────────────────────
    if match is not None:
        team_i, team_j = match
        print(f"\n{'─'*60}")
        print(f"  MATCH REPORT: {team_i} vs {team_j}")
        print_match_report(
            model, team_i, team_j,
            calibrator=calibrator,
            n_bootstrap=n_bootstrap,
            match_importance=1.0,
            top_scores=3,
            seed=seed,
        )

    print("\nDone.")
    return mc_results, model, group_pos_probs, calibrator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WC 2026 Predictor")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download match data even if cached")
    parser.add_argument("--n-mc", type=int, default=100_000,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip temporal cross-validation and LightGBM (faster run)")
    parser.add_argument("--skip-lgbm", action="store_true",
                        help="Run DC calibration report but skip LightGBM fitting")
    parser.add_argument("--no-save-calibrator", action="store_true",
                        help="Do not save the fitted LightGBM calibrator to disk")
    parser.add_argument("--match", nargs=2, metavar=("TEAM1", "TEAM2"),
                        help="Print a detailed match report (e.g. --match Brazil France)")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Bootstrap samples for match report CIs (default 500)")
    args = parser.parse_args()

    main(
        force_download=args.force_download,
        n_mc=args.n_mc,
        seed=args.seed,
        skip_calibration=args.skip_calibration,
        skip_lgbm=args.skip_lgbm,
        save_calibrator=not args.no_save_calibrator,
        match=tuple(args.match) if args.match else None,
        n_bootstrap=args.n_bootstrap,
    )
