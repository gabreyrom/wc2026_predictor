"""
Match Report — detailed probabilistic breakdown for a single fixture.

Combines:
    • Dixon-Coles raw probabilities (win / draw / loss)
    • Parametric bootstrap 90% confidence intervals on those probabilities
    • LightGBM-calibrated probabilities (if calibrator is provided)
    • Top-N most probable scorelines from the DC score matrix

Usage
─────
    from src.output.match_report import print_match_report

    print_match_report(model, "Brazil", "France",
                       calibrator=calibrator,
                       n_bootstrap=500,
                       top_scores=3)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.model.dixon_coles import DixonColesModel
    from src.model.lgbm_calibrator import LGBMCalibrator


# ── Core computation ──────────────────────────────────────────────────────────

def match_report(
    model: "DixonColesModel",
    team_i: str,
    team_j: str,
    calibrator: Optional["LGBMCalibrator"] = None,
    n_bootstrap: int = 500,
    match_importance: float = 1.0,
    top_scores: int = 3,
    ci_level: float = 0.90,
    seed: int = 42,
) -> dict:
    """
    Compute a full probabilistic match report.

    Parameters
    ----------
    model          : fitted DixonColesModel
    team_i         : home team name (as in model.alpha)
    team_j         : away team name
    calibrator     : optional fitted LGBMCalibrator for corrected probabilities
    n_bootstrap    : number of parametric bootstrap samples for CIs
    match_importance: 0=friendly … 1.0=WC knockout
    top_scores     : number of most-probable scorelines to return
    ci_level       : confidence level for the interval (default 0.90 → 5th–95th pct)
    seed           : random seed for reproducibility

    Returns
    -------
    dict with keys:
        team_i, team_j, match_importance,
        lambda_i, mu_j, rho,
        dc: {home, draw, away},
        ci: {home: (lo, hi), draw: (lo, hi), away: (lo, hi)},
        cal: {home, draw, away} or None,
        top_scorelines: list of {"score": "X-Y", "prob": float, "winner": str}
    """
    if team_i not in model.alpha:
        raise ValueError(f"'{team_i}' not found in model. Check spelling.")
    if team_j not in model.alpha:
        raise ValueError(f"'{team_j}' not found in model. Check spelling.")

    # ── DC point prediction ───────────────────────────────────────────────────
    dc_pred = model.predict(team_i, team_j, match_importance=match_importance)

    # ── Parametric bootstrap CI ───────────────────────────────────────────────
    boot = model.parametric_bootstrap(
        team_i, team_j,
        n_samples=n_bootstrap,
        match_importance=match_importance,
        seed=seed,
    )
    alpha = (1.0 - ci_level) / 2.0
    lo = np.quantile(boot, alpha, axis=0)
    hi = np.quantile(boot, 1.0 - alpha, axis=0)

    ci = {
        "home": (float(lo[0]), float(hi[0])),
        "draw": (float(lo[1]), float(hi[1])),
        "away": (float(lo[2]), float(hi[2])),
    }

    # ── Calibrated prediction ─────────────────────────────────────────────────
    cal = None
    if calibrator is not None:
        cal = calibrator.predict_proba_row(dc_pred)

    # ── Top scorelines ────────────────────────────────────────────────────────
    mat = dc_pred["score_matrix"]
    n = mat.shape[0]
    flat_idx = np.argsort(mat.ravel())[::-1][:top_scores]
    scorelines = []
    for idx in flat_idx:
        home_g, away_g = divmod(int(idx), n)
        if home_g > away_g:
            winner = team_i
        elif home_g == away_g:
            winner = "Draw"
        else:
            winner = team_j
        scorelines.append({
            "score":  f"{home_g}–{away_g}",
            "prob":   float(mat[home_g, away_g]),
            "winner": winner,
        })

    return {
        "team_i":           team_i,
        "team_j":           team_j,
        "match_importance": match_importance,
        "lambda_i":         dc_pred["lambda_i"],
        "mu_j":             dc_pred["mu_j"],
        "rho":              dc_pred["rho"],
        "dc":               {
            "home": dc_pred["home"],
            "draw": dc_pred["draw"],
            "away": dc_pred["away"],
        },
        "ci":               ci,
        "cal":              cal,
        "top_scorelines":   scorelines,
        "ci_level":         ci_level,
        "n_bootstrap":      n_bootstrap,
        "boot_samples":     boot,   # raw samples — useful for histograms
    }


# ── Formatted printer ─────────────────────────────────────────────────────────

def print_match_report(
    model: "DixonColesModel",
    team_i: str,
    team_j: str,
    calibrator: Optional["LGBMCalibrator"] = None,
    n_bootstrap: int = 500,
    match_importance: float = 1.0,
    top_scores: int = 3,
    ci_level: float = 0.90,
    seed: int = 42,
) -> dict:
    """
    Compute and print a formatted match report. Returns the report dict.

    Example output
    ──────────────
    ══════════════════════════════════════════════════════════════════
      BRAZIL vs FRANCE
    ══════════════════════════════════════════════════════════════════
      Expected goals   Brazil 1.84  —  France 1.31   (ρ = -0.42)

      Outcome probabilities           90% CI
      ──────────────────────────────────────────────────────────────
      Brazil win        54.2%      [ 48.1% –  60.3% ]
      Draw              26.1%      [ 21.4% –  30.8% ]
      France win        19.7%      [ 15.2% –  24.2% ]

      Calibrated (LightGBM)
      Brazil win        48.3%   Draw  27.1%   France win  24.6%

      Top 3 most likely scorelines
      ──────────────────────────────────────────────────────────────
        1.  1 – 0  Brazil      11.3%
        2.  1 – 1  Draw        10.8%
        3.  2 – 1  Brazil       9.2%
    ══════════════════════════════════════════════════════════════════
    """
    r = match_report(
        model, team_i, team_j,
        calibrator=calibrator,
        n_bootstrap=n_bootstrap,
        match_importance=match_importance,
        top_scores=top_scores,
        ci_level=ci_level,
        seed=seed,
    )

    ci_pct = int(r["ci_level"] * 100)
    W = 66

    print("\n" + "═" * W)
    print(f"  {team_i.upper()}  vs  {team_j.upper()}")
    importance_label = {
        1.0: "World Cup knockout",
        0.7: "Continental championship",
        0.3: "Qualifier",
        0.0: "Friendly",
    }.get(round(match_importance, 1), f"importance={match_importance:.1f}")
    print(f"  {importance_label}")
    print("═" * W)

    # ── Expected goals ────────────────────────────────────────────────────────
    print(f"\n  Expected goals   {team_i} {r['lambda_i']:.2f}  —  "
          f"{team_j} {r['mu_j']:.2f}   (ρ = {r['rho']:.3f})\n")

    # ── Outcome probabilities + CI ────────────────────────────────────────────
    print(f"  {'Outcome':<22} {'DC prob':>8}    {ci_pct}% confidence interval")
    print("  " + "─" * (W - 2))
    outcome_labels = [
        (f"{team_i} win",  r["dc"]["home"], r["ci"]["home"]),
        ("Draw",           r["dc"]["draw"], r["ci"]["draw"]),
        (f"{team_j} win",  r["dc"]["away"], r["ci"]["away"]),
    ]
    for label, prob, (lo, hi) in outcome_labels:
        print(f"  {label:<22} {prob:>7.1%}    [ {lo:>5.1%} – {hi:>5.1%} ]")

    # ── Calibrated ────────────────────────────────────────────────────────────
    if r["cal"] is not None:
        cal = r["cal"]
        print(f"\n  Calibrated (LightGBM — corrects draw bias)")
        print(f"  {team_i} win  {cal['home']:.1%}   "
              f"Draw  {cal['draw']:.1%}   "
              f"{team_j} win  {cal['away']:.1%}")

    # ── Bootstrap distribution summary ───────────────────────────────────────
    boot = r["boot_samples"]
    std_home = float(boot[:, 0].std())
    std_draw = float(boot[:, 1].std())
    std_away = float(boot[:, 2].std())
    print(f"\n  Parameter uncertainty (±1σ from {r['n_bootstrap']} bootstrap samples)")
    print(f"  {team_i} win  ±{std_home:.1%}   "
          f"Draw  ±{std_draw:.1%}   "
          f"{team_j} win  ±{std_away:.1%}")

    # ── Top scorelines ────────────────────────────────────────────────────────
    print(f"\n  Top {top_scores} most likely scorelines")
    print("  " + "─" * (W - 2))
    for rank, s in enumerate(r["top_scorelines"], 1):
        print(f"    {rank}.  {s['score']:<6}  {s['winner']:<24}  {s['prob']:.1%}")

    print("═" * W + "\n")
    return r


# ── Group fixture scan ────────────────────────────────────────────────────────

def print_group_match_reports(
    groups: dict[str, list[str]],
    model: "DixonColesModel",
    calibrator: Optional["LGBMCalibrator"] = None,
    n_bootstrap: int = 300,
    top_scores: int = 3,
) -> None:
    """
    Print concise calibrated odds for every group-stage fixture.
    Use print_match_report() for the full breakdown of a single match.
    """
    from itertools import combinations

    W = 76
    print("\n" + "═" * W)
    print(f"{'GROUP STAGE — ALL FIXTURES':^{W}}")
    print("═" * W)
    print(f"  {'Match':<36}  {'DC: H/D/A':>12}  {'LGBM: H/D/A':>12}  "
          f"{'Top score':>10}")
    print("  " + "─" * (W - 2))

    for group_name, teams in sorted(groups.items()):
        print(f"\n  Group {group_name}")
        for home, away in combinations(teams, 2):
            if home not in model.alpha or away not in model.alpha:
                continue
            dc = model.predict(home, away, match_importance=1.0)

            cal_str = "—"
            if calibrator is not None:
                cal = calibrator.predict_proba_row(dc)
                cal_str = f"{cal['home']:.0%}/{cal['draw']:.0%}/{cal['away']:.0%}"

            mat = dc["score_matrix"]
            best_idx = int(mat.argmax())
            bg, ag = divmod(best_idx, mat.shape[0])

            dc_str    = f"{dc['home']:.0%}/{dc['draw']:.0%}/{dc['away']:.0%}"
            score_str = f"{bg}–{ag} ({mat[bg, ag]:.0%})"
            matchup   = f"{home} vs {away}"

            print(f"    {matchup:<34}  {dc_str:>12}  {cal_str:>12}  {score_str:>10}")

    print("═" * W)
