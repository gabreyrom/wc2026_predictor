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
    print(f"{'Group':<8} {'1st':<20} {'2nd':<20} {'3rd (P reach R32, any route)'}")
    print("-" * 72)

    # P(reach R32) per team — includes 3rd-place qualification (from MC)
    r32_prob = dict(zip(mc_results["team"], mc_results["R32"]))

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
        pred_3rd = max(
            [(t, p) for t, p in team_probs.items() if t not in (pred_1st, pred_2nd)],
            key=lambda x: x[1]["3rd"],
        )[0]
        third_qual = r32_prob.get(pred_3rd, 0.0)
        print(f"  {group_name:<6} {pred_1st:<20} {pred_2nd:<20} "
              f"{pred_3rd} ({third_qual:.0%})")

    # Top 5 most probable champions
    top5 = mc_results.nlargest(5, "Winner")
    print("\nTop 5 possible champions:")
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {rank}. {row['team']:<20s} {row['Winner']:.1%}")
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


# ── Daily output folder ───────────────────────────────────────────────────────

def _daily_dir(results_dir: str = "results"):
    """
    Return (and create) today's output folder: results/YYYY-MM-DD/.
    All of a day's outputs live together; same-day reruns overwrite in place.
    """
    from datetime import date
    from pathlib import Path
    out = Path(results_dir) / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    return out


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


# ── Per-match top-N scorelines ────────────────────────────────────────────────

def save_match_scorelines(
    model: DixonColesModel,
    groups: dict[str, list[str]],
    top_n: int = 5,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    For every known fixture, compute the top-N most probable scorelines and
    save them to a date-stamped CSV in `results_dir`.

    Covers the 72 group-stage fixtures (the only matches whose pairings are
    known before the tournament). Knockout fixtures can be appended once the
    real R32 pairings exist.

    Output columns:
        stage, group, home_team, away_team,
        p_home, p_draw, p_away          — outcome probs (4 decimals)
        rank, score, prob               — one row per (match, rank)
    """
    from datetime import date
    from pathlib import Path
    from itertools import combinations
    from src.simulation.group_stage import host_flags

    rows = []
    for group_name in sorted(groups):
        teams = groups[group_name]
        for home, away in combinations(teams, 2):
            if home not in model.alpha or away not in model.alpha:
                continue
            h_i, h_j = host_flags(home, away)
            pred = model.predict(home, away, match_importance=1.0,
                                 home_i=h_i, home_j=h_j)
            mat = pred["score_matrix"]
            n = mat.shape[0]
            flat_idx = np.argsort(mat.ravel())[::-1][:top_n]

            for rank, idx in enumerate(flat_idx, 1):
                hg, ag = divmod(int(idx), n)
                rows.append({
                    "stage":     "group",
                    "group":     group_name,
                    "home_team": home,
                    "away_team": away,
                    "p_home":    round(pred["home"], 4),
                    "p_draw":    round(pred["draw"], 4),
                    "p_away":    round(pred["away"], 4),
                    "rank":      rank,
                    "score":     f"{hg}-{ag}",
                    "prob":      round(float(mat[hg, ag]), 4),
                })

    df = pd.DataFrame(rows)
    path = _daily_dir(results_dir) / "match_scorelines.csv"
    df.to_csv(path, index=False)
    n_matches = len(df) // top_n
    print(f"Saved top-{top_n} scorelines for {n_matches} matches -> {path}")
    return df


# ── All-104-matches probabilities with confidence intervals ──────────────────

def save_match_probabilities(
    model: DixonColesModel,
    groups: dict[str, list[str]],
    pairings: dict | None = None,
    calibrator=None,
    n_bootstrap: int = 200,
    top_pairings: int = 3,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Save per-match outcome probabilities with 90% parametric-bootstrap CIs
    for the full tournament:

      • 72 group fixtures — pairings known, p_pairing = 1
      • knockout slots 73–104 — top-N most likely pairings from the MC
        simulation, each with its occurrence probability and the outcome
        probabilities CONDITIONAL on that pairing

    Columns (probabilities rounded to 4 decimals):
        stage, match_no, group, home_team, away_team, p_pairing,
        p_home, p_draw, p_away,
        ci_home_lo, ci_home_hi, ci_draw_lo, ci_draw_hi, ci_away_lo, ci_away_hi,
        cal_home, cal_draw, cal_away   (LGBM-calibrated, if calibrator given)

    For knockout rows p_draw is the 90-minute draw probability (match then
    goes to extra time / penalties).
    """
    from datetime import date
    from pathlib import Path
    from itertools import combinations
    from src.simulation.group_stage import host_flags

    def _row(stage, match_no, group, home, away, p_pairing):
        h_i, h_j = host_flags(home, away)
        pred = model.predict(home, away, match_importance=1.0,
                             home_i=h_i, home_j=h_j)
        boot = model.parametric_bootstrap(
            home, away, n_samples=n_bootstrap,
            match_importance=1.0, home_i=h_i, home_j=h_j,
        )
        lo = np.quantile(boot, 0.05, axis=0)
        hi = np.quantile(boot, 0.95, axis=0)

        row = {
            "stage":      stage,
            "match_no":   match_no,
            "group":      group,
            "home_team":  home,
            "away_team":  away,
            "p_pairing":  round(p_pairing, 4),
            "p_home":     round(pred["home"], 4),
            "p_draw":     round(pred["draw"], 4),
            "p_away":     round(pred["away"], 4),
            "ci_home_lo": round(float(lo[0]), 4),
            "ci_home_hi": round(float(hi[0]), 4),
            "ci_draw_lo": round(float(lo[1]), 4),
            "ci_draw_hi": round(float(hi[1]), 4),
            "ci_away_lo": round(float(lo[2]), 4),
            "ci_away_hi": round(float(hi[2]), 4),
        }
        if calibrator is not None:
            from src.data.market_values import log_value_ratio
            from src.data.elo import elo_diff
            pred["log_value_ratio"] = log_value_ratio(home, away, "2026-06-11")
            pred["elo_diff"] = elo_diff(home, away, "2026-06-11")
            cal = calibrator.predict_proba_row(pred)
            row["cal_home"] = round(cal["home"], 4)
            row["cal_draw"] = round(cal["draw"], 4)
            row["cal_away"] = round(cal["away"], 4)
        return row

    rows = []

    # ── Group stage: 72 known fixtures ────────────────────────────────────────
    for group_name in sorted(groups):
        for home, away in combinations(groups[group_name], 2):
            if home not in model.alpha or away not in model.alpha:
                continue
            rows.append(_row("group", None, group_name, home, away, 1.0))

    # ── Knockout slots: top pairings from MC ──────────────────────────────────
    if pairings:
        for m_no in sorted(pairings):
            ranked = sorted(pairings[m_no].items(), key=lambda x: -x[1])
            for (home, away), p in ranked[:top_pairings]:
                if home not in model.alpha or away not in model.alpha:
                    continue
                stage = ("R32" if 73 <= m_no <= 88 else
                         "R16" if 89 <= m_no <= 96 else
                         "QF"  if 97 <= m_no <= 100 else
                         "SF"  if m_no in (101, 102) else
                         "3rd-place" if m_no == 103 else "Final")
                rows.append(_row(stage, m_no, None, home, away, p))

    df = pd.DataFrame(rows)
    path = _daily_dir(results_dir) / "match_probabilities.csv"
    df.to_csv(path, index=False)
    n_group = (df["stage"] == "group").sum()
    n_ko = df["match_no"].nunique()
    print(f"Saved match probabilities ({n_group} group fixtures + "
          f"{n_ko} knockout slots × top-{top_pairings} pairings) -> {path}")
    return df


# ── Save to CSV (versioned) ───────────────────────────────────────────────────

def save_results(
    mc_results: pd.DataFrame,
    group_pos_probs: dict | None = None,
    results_dir: str = "results",
    latest_path: str = "data/processed/wc2026_probs.csv",
) -> None:
    """
    Save prediction outputs in two places:
      1. `latest_path` — canonical "current" file, overwritten each run
      2. `results_dir/YYYY-MM-DD/*.csv` — one folder per day holding all of
         that day's outputs, giving an audit trail of how predictions evolve
         across matchdays (same-day reruns overwrite in place)

    Args:
        mc_results      : tournament advancement table from run_simulations()
        group_pos_probs : optional {group: {team: {1st/2nd/3rd/4th/...}}}
                          from simulate_all_groups()
    """
    from pathlib import Path

    day_dir = _daily_dir(results_dir)
    Path(latest_path).parent.mkdir(parents=True, exist_ok=True)

    # Tournament advancement table
    mc_results.to_csv(latest_path, index=False)
    versioned = day_dir / "tournament_probs.csv"
    mc_results.to_csv(versioned, index=False)
    print(f"Saved results -> {latest_path}")
    print(f"             -> {versioned}")

    # Group position probabilities (flattened to one row per team)
    if group_pos_probs is not None:
        rows = []
        for group, team_probs in group_pos_probs.items():
            for team, p in team_probs.items():
                rows.append({
                    "group":   group,
                    "team":    team,
                    "p_1st":   p["1st"],
                    "p_2nd":   p["2nd"],
                    "p_3rd":   p["3rd"],
                    "p_4th":   p["4th"],
                    "exp_pts": p["exp_pts"],
                    "exp_gd":  p["exp_gd"],
                    "exp_gf":  p["exp_gf"],
                })
        group_df = pd.DataFrame(rows).sort_values(
            ["group", "p_1st"], ascending=[True, False]
        )
        group_versioned = day_dir / "group_position_probs.csv"
        group_df.to_csv(group_versioned, index=False)
        print(f"             -> {group_versioned}")


if __name__ == "__main__":
    print("Run main.py to generate full results.")
