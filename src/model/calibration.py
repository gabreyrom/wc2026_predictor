"""
Temporal Cross-Validation and Calibration for the Dixon-Coles Model.

Why this matters
────────────────
Fitting on 30k+ matches and reporting training-set log-loss is meaningless —
the model has seen every result it's being scored on. We need a time-respecting
evaluation: train on the past, score on the future.

Split strategy (mirrors the article's approach)
───────────────────────────────────────────────
    Train  : pre-2018  (~23k matches) — parameter estimation
    Val    : 2018–2022 (~5k matches)  — model selection / hyperparameter tuning
    Test   : 2022+     (~3k matches)  — final, one-shot honest evaluation

Key metric: multiclass log-loss
────────────────────────────────
    L = -1/N · Σ_k [y_H·log(p_H) + y_D·log(p_D) + y_A·log(p_A)]

    where y_H/D/A is the one-hot actual outcome and p_H/D/A are model
    probabilities. Lower is better. A perfectly calibrated 33/33/33 baseline
    scores log(3) ≈ 1.099. A good model should be ~0.92–0.96.

Calibration check
─────────────────
For each of 10 probability bins, compare predicted p_draw with the fraction
of matches that actually ended in a draw. A well-calibrated model sits on
the diagonal. Systematic deviation reveals bias.
"""

import math
import numpy as np
import pandas as pd
from collections import defaultdict

from src.model.dixon_coles import DixonColesModel, fit as fit_dc


# ── Split ─────────────────────────────────────────────────────────────────────

def temporal_split(
    matches: pd.DataFrame,
    val_start: str = "2018-01-01",
    test_start: str = "2022-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split matches chronologically into train / validation / test sets.

    Args:
        matches    : full cleaned match DataFrame (sorted by date)
        val_start  : first date of the validation set
        test_start : first date of the test set

    Returns:
        (train, val, test) DataFrames
    """
    train = matches[matches["date"] <  val_start].copy()
    val   = matches[(matches["date"] >= val_start)
                    & (matches["date"] < test_start)].copy()
    test  = matches[matches["date"] >= test_start].copy()

    print(f"Temporal split:")
    print(f"  Train : {len(train):>6,} matches  "
          f"({train['date'].min().date()} – {train['date'].max().date()})")
    print(f"  Val   : {len(val):>6,} matches  "
          f"({val['date'].min().date()} – {val['date'].max().date()})")
    print(f"  Test  : {len(test):>6,} matches  "
          f"({test['date'].min().date()} – {test['date'].max().date()})")
    return train, val, test


# ── Match outcome probabilities ───────────────────────────────────────────────

def _outcome(row: pd.Series) -> str:
    """Return 'home', 'draw', or 'away' for a match row."""
    if row["home_score"] > row["away_score"]:
        return "home"
    elif row["home_score"] == row["away_score"]:
        return "draw"
    return "away"


def predict_matches(
    model: DixonColesModel,
    matches: pd.DataFrame,
    match_importance: float = 0.5,
) -> pd.DataFrame:
    """
    Run model predictions on a set of matches.

    Returns a DataFrame with columns:
        home_team, away_team, date, tournament_category,
        p_home, p_draw, p_away,   — model probabilities
        actual,                   — 'home' | 'draw' | 'away'
        log_loss                  — per-match contribution
    """
    rows = []
    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Skip teams the model hasn't seen
        if home not in model.alpha or away not in model.alpha:
            continue

        # Use tournament_category to set match_importance
        imp = {
            "World Cup":                1.0,
            "Continental Championship": 0.7,
            "World Cup Qualifier":      0.3,
            "Continental Qualifier":    0.3,
            "Friendly":                 0.0,
        }.get(row.get("tournament_category", "Friendly"), 0.0)

        pred = model.predict(home, away, match_importance=imp)
        actual = _outcome(row)

        p_actual = pred[actual]
        ll = -math.log(max(p_actual, 1e-10))

        rows.append({
            "date":                row["date"],
            "home_team":           home,
            "away_team":           away,
            "tournament_category": row.get("tournament_category", "Unknown"),
            "p_home":              pred["home"],
            "p_draw":              pred["draw"],
            "p_away":              pred["away"],
            "rho":                 pred["rho"],
            "actual":              actual,
            "log_loss":            ll,
        })

    return pd.DataFrame(rows)


# ── Log-loss scorer ───────────────────────────────────────────────────────────

def log_loss_score(pred_df: pd.DataFrame) -> float:
    """Mean log-loss over all matches in pred_df."""
    if pred_df.empty:
        return float("nan")
    return float(pred_df["log_loss"].mean())


BASELINE_LOG_LOSS = math.log(3)   # uniform 1/3 per outcome ≈ 1.0986


# ── Full calibration report ───────────────────────────────────────────────────

def calibration_report(
    model: DixonColesModel,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_bins: int = 10,
) -> dict:
    """
    Print a full calibration report comparing val and test log-loss,
    broken down by tournament type and year, plus a draw calibration table.

    Returns a dict of summary metrics.
    """
    print("\n" + "=" * 62)
    print(f"{'MODEL CALIBRATION REPORT':^62}")
    print("=" * 62)
    print(f"  Baseline (uniform 1/3 each outcome): {BASELINE_LOG_LOSS:.4f}")

    results = {}
    for split_name, split_df in [("Validation (2018–2022)", val),
                                  ("Test       (2022+    )", test)]:
        pred_df = predict_matches(model, split_df)
        ll = log_loss_score(pred_df)
        improvement = (BASELINE_LOG_LOSS - ll) / BASELINE_LOG_LOSS * 100
        print(f"\n  {split_name}:  log-loss = {ll:.4f}  "
              f"({improvement:+.1f}% vs baseline)  "
              f"n={len(pred_df):,}")
        results[split_name] = {"log_loss": ll, "n": len(pred_df), "pred_df": pred_df}

    # ── Breakdown by tournament type (validation set) ────────────────────────
    val_pred = results["Validation (2018–2022)"]["pred_df"]
    if not val_pred.empty:
        print(f"\n  {'Tournament':30s} {'Log-loss':>9}  {'N':>6}  {'Draw%actual':>11}  {'Draw%pred':>9}")
        print("  " + "-" * 68)
        for cat, grp in val_pred.groupby("tournament_category"):
            actual_draw_rate = (grp["actual"] == "draw").mean()
            pred_draw_rate   = grp["p_draw"].mean()
            print(f"  {str(cat):30s} {log_loss_score(grp):>9.4f}  "
                  f"{len(grp):>6,}  "
                  f"{actual_draw_rate:>11.1%}  "
                  f"{pred_draw_rate:>9.1%}")

    # ── Draw calibration bins ────────────────────────────────────────────────
    print(f"\n  Draw Calibration (validation set) — "
          f"predicted vs actual draw rate:")
    print(f"  {'Pred draw bin':20s} {'Pred mean':>9}  {'Actual rate':>11}  "
          f"{'N':>6}  {'Bias':>7}")
    print("  " + "-" * 58)

    if not val_pred.empty:
        val_pred = val_pred.copy()
        val_pred["draw_bin"] = pd.cut(val_pred["p_draw"], bins=n_bins, labels=False)
        for b, grp in val_pred.groupby("draw_bin"):
            pred_mean   = grp["p_draw"].mean()
            actual_rate = (grp["actual"] == "draw").mean()
            bias        = actual_rate - pred_mean
            bar = "▲" if bias > 0.03 else ("▼" if bias < -0.03 else "·")
            print(f"  {pred_mean:.2f}–{pred_mean:.2f}          "
                  f"{pred_mean:>9.3f}  {actual_rate:>11.3f}  "
                  f"{len(grp):>6,}  {bias:>+6.3f} {bar}")

    # ── Outcome accuracy ─────────────────────────────────────────────────────
    print(f"\n  Outcome accuracy (modal prediction vs actual):")
    for split_name, res in results.items():
        pred_df = res["pred_df"]
        if pred_df.empty:
            continue
        pred_df = pred_df.copy()
        pred_df["predicted"] = pred_df[["p_home", "p_draw", "p_away"]].idxmax(axis=1).str[2:]
        acc = (pred_df["predicted"] == pred_df["actual"]).mean()

        # Per-class recall
        for cls in ["home", "draw", "away"]:
            mask = pred_df["actual"] == cls
            if mask.sum() > 0:
                recall = (pred_df.loc[mask, "predicted"] == cls).mean()
                n_cls  = mask.sum()
                print(f"  {split_name[:10]}  {cls:5s}: "
                      f"recall={recall:.1%}  (n={n_cls:,})")
        print(f"  {split_name[:10]}  overall accuracy: {acc:.1%}")

    print("=" * 62)
    return results


# ── Full pipeline: split → fit → evaluate ────────────────────────────────────

def fit_and_evaluate(
    matches: pd.DataFrame,
    xi: float = 0.003,
    val_start: str = "2018-01-01",
    test_start: str = "2022-01-01",
) -> tuple[DixonColesModel, dict]:
    """
    Full pipeline:
        1. Split data chronologically
        2. Fit Dixon-Coles on training set
        3. Run calibration_report on val + test

    Returns:
        model   : fitted on training data only (honest evaluation)
        results : calibration metrics dict
    """
    train, val, test = temporal_split(matches, val_start, test_start)

    print(f"\nFitting on training data only ({len(train):,} matches)...")
    model = fit_dc(train, xi=xi)

    results = calibration_report(model, val, test)
    return model, results


# ── Xi / date-cutoff sensitivity ─────────────────────────────────────────────

def hyperparameter_search(
    matches: pd.DataFrame,
    xi_values: list[float] | None = None,
    train_starts: list[str] | None = None,
    val_start: str = "2018-01-01",
    test_start: str = "2022-01-01",
) -> pd.DataFrame:
    """
    Grid search over time-decay (xi) and training window start date.
    Scores each combination by validation log-loss.

    Returns a DataFrame sorted by val_log_loss ascending.
    """
    if xi_values is None:
        xi_values = [0.001, 0.003, 0.005, 0.010]
    if train_starts is None:
        train_starts = ["2000-01-01", "2010-01-01", "2015-01-01"]

    _, val, _ = temporal_split(matches, val_start, test_start)
    rows = []

    for train_start in train_starts:
        for xi in xi_values:
            train = matches[
                (matches["date"] >= train_start)
                & (matches["date"] < val_start)
            ].copy()
            if len(train) < 500:
                continue

            print(f"  train_start={train_start}  xi={xi:.3f}  "
                  f"n_train={len(train):,} ... ", end="", flush=True)
            model = fit_dc(train, xi=xi)
            pred_df = predict_matches(model, val)
            ll = log_loss_score(pred_df)
            print(f"val_log_loss={ll:.4f}")

            rows.append({
                "train_start": train_start,
                "xi":          xi,
                "n_train":     len(train),
                "val_log_loss": ll,
            })

    df = pd.DataFrame(rows).sort_values("val_log_loss").reset_index(drop=True)
    print("\n  Best hyperparameters:")
    print(df.head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.fetch_matches import fetch_and_process

    matches = fetch_and_process(force=False)
    model, results = fit_and_evaluate(matches, xi=0.003)
