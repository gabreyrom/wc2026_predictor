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
from src.data.market_values import log_value_ratio
from src.data.elo import elo_diff


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
    extra_feature_fns: dict | None = None,
) -> pd.DataFrame:
    """
    Run model predictions on a set of matches.

    Returns a DataFrame with columns:
        home_team, away_team, date, tournament_category,
        p_home, p_draw, p_away,       — raw DC probabilities
        lambda_i, mu_j,               — expected goals
        rho,                          — match-specific rho
        abs_alpha_diff,               — |alpha_home - alpha_away|
        match_importance,             — importance scale used
        actual,                       — 'home' | 'draw' | 'away'
        log_loss                      — per-match log-loss contribution

    The extra columns (lambda_i, mu_j, abs_alpha_diff, match_importance)
    are consumed by LGBMCalibrator.fit() / predict_proba_df().
    """
    IMPORTANCE_MAP = {
        "World Cup":                1.0,
        "Continental Championship": 0.7,
        "World Cup Qualifier":      0.3,
        "Continental Qualifier":    0.3,
        "UEFA Nations League":      0.3,
        "CONCACAF Nations League":  0.3,
        "Friendly":                 0.0,
    }

    rows = []
    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Skip teams the model hasn't seen
        if home not in model.alpha or away not in model.alpha:
            continue

        # Use tournament_category to set match_importance
        imp = IMPORTANCE_MAP.get(row.get("tournament_category", "Friendly"), 0.0)

        # Home advantage: applies when the listed home team is not on neutral ground
        is_home = not bool(row.get("neutral", True))
        pred = model.predict(home, away, match_importance=imp, home_i=is_home,
                             match_date=row["date"])
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
            "lambda_i":            pred["lambda_i"],
            "mu_j":                pred["mu_j"],
            "rho":                 pred["rho"],
            "abs_alpha_diff":      abs(
                model.alpha.get(home, 0.0) - model.alpha.get(away, 0.0)
            ),
            "match_importance":    imp,
            "log_value_ratio":     log_value_ratio(home, away, row["date"]),
            "elo_diff":            elo_diff(home, away, row["date"]),
            "actual":              actual,
            "log_loss":            ll,
            **{name: fn(home, away, row["date"])
               for name, fn in (extra_feature_fns or {}).items()},
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


# ── Out-of-fold temporal calibration (rolling origin) ────────────────────────

def oof_calibration_report(
    fit_data: pd.DataFrame,
    xi: float = 0.003,
    fold_cutoffs: list[str] | None = None,
    extra_feature_fns: dict | None = None,
) -> tuple[dict, "LGBMCalibrator"]:  # type: ignore[name-defined]
    """
    Rolling-origin out-of-fold calibration:

        fold k: fit Dixon-Coles on data < cutoff_k,
                predict matches in [cutoff_k, cutoff_{k+1})

    The calibrator trains on the UNION of all folds' out-of-sample
    predictions. Because each fold's DC model has a different vintage
    (different training size, different α/β scales), the calibrator learns a
    correction that is robust to model-vintage shift — addressing the
    transfer risk of training on one frozen eval model and applying to the
    production model.

    The final cutoff defines the held-out TEST predictor: a model fitted on
    everything before it, scored on everything after. Test rows never enter
    calibrator training.

    Returns (results dict, fitted LGBMCalibrator).
    """
    from src.model.lgbm_calibrator import LGBMCalibrator

    if fold_cutoffs is None:
        fold_cutoffs = ["2016-01-01", "2018-01-01", "2020-01-01", "2022-01-01"]

    print("\n" + "=" * 62)
    print(f"{'OUT-OF-FOLD TEMPORAL CALIBRATION':^62}")
    print("=" * 62)
    print(f"  Baseline (uniform 1/3): {BASELINE_LOG_LOSS:.4f}")

    # ── OOF folds for calibrator training ────────────────────────────────────
    oof_frames = []
    for k in range(len(fold_cutoffs) - 1):
        c0, c1 = fold_cutoffs[k], fold_cutoffs[k + 1]
        train  = fit_data[fit_data["date"] <  c0].copy()
        window = fit_data[(fit_data["date"] >= c0)
                          & (fit_data["date"] < c1)].copy()
        print(f"\n  Fold {k+1}: fit < {c0[:7]}  →  predict {c0[:7]} – {c1[:7]} "
              f"(n_train={len(train):,}, n_window={len(window):,})")
        fold_model = fit_dc(train, xi=xi)
        pf = predict_matches(fold_model, window, extra_feature_fns=extra_feature_fns)
        pf["fold"] = f"{c0[:4]}–{c1[:4]}"
        print(f"  Fold {k+1} DC out-of-sample log-loss: {log_loss_score(pf):.4f}  "
              f"(n={len(pf):,})")
        oof_frames.append(pf)

    cal_train = pd.concat(oof_frames, ignore_index=True)
    print(f"\n  Calibrator training set: {len(cal_train):,} OOF predictions "
          f"across {len(oof_frames)} model vintages")

    # ── Held-out test: fresh model fitted on everything before last cutoff ───
    test_cut = fold_cutoffs[-1]
    print(f"\n  Test predictor: fit < {test_cut[:7]}, score {test_cut[:7]}+ ...")
    test_model = fit_dc(fit_data[fit_data["date"] < test_cut].copy(), xi=xi)
    test_pred  = predict_matches(
        test_model, fit_data[fit_data["date"] >= test_cut].copy(),
        extra_feature_fns=extra_feature_fns,
    )
    dc_test_ll = log_loss_score(test_pred)
    print(f"  DC test log-loss: {dc_test_ll:.4f}  (n={len(test_pred):,})")

    # ── Fit calibrator on OOF union, evaluate once on test ───────────────────
    print(f"\n  Fitting LGBMCalibrator on OOF predictions (5-fold CV tuning)...")
    calibrator = LGBMCalibrator()
    calibrator.fit_cv(cal_train, verbose=True)

    cal_test = calibrator.predict_proba_df(test_pred)
    lgbm_test_ll = float(cal_test["cal_log_loss"].mean())

    print(f"\n  {'Model':20s}  {'Test log-loss':>13}  {'vs baseline':>11}")
    print("  " + "-" * 50)
    print(f"  {'Baseline (uniform)':20s}  {BASELINE_LOG_LOSS:>13.4f}  {'—':>11}")
    print(f"  {'Dixon-Coles':20s}  {dc_test_ll:>13.4f}  "
          f"{(BASELINE_LOG_LOSS - dc_test_ll) / BASELINE_LOG_LOSS:>+10.1%}")
    print(f"  {'DC + LightGBM':20s}  {lgbm_test_ll:>13.4f}  "
          f"{(BASELINE_LOG_LOSS - lgbm_test_ll) / BASELINE_LOG_LOSS:>+10.1%}")

    boot = paired_bootstrap_test(
        ll_base=test_pred["log_loss"].values,
        ll_alt=cal_test["cal_log_loss"].values,
    )
    sig = (boot["ci_low"] > 0) or (boot["ci_high"] < 0)
    print(f"\n  Paired bootstrap on test set (n={boot['n']:,}, 10k resamples):")
    print(f"    Δ log-loss (LGBM − DC) = {boot['mean_diff']:+.4f}  "
          f"95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]  "
          f"p ≈ {boot['p_value']:.3f}")
    if sig and boot["mean_diff"] < 0:
        print("    → LGBM is significantly BETTER than raw DC.")
    elif sig:
        print("    → LGBM is significantly WORSE than raw DC. Prefer raw DC.")
    else:
        print("    → No detectable difference; parsimony favors raw DC.")
    print("=" * 62)

    results = {
        "dc_test":   {"log_loss": dc_test_ll,   "n": len(test_pred)},
        "lgbm_test": {"log_loss": lgbm_test_ll, "n": len(test_pred)},
        "bootstrap": boot,
        "cal_train_df": cal_train,
        "test_pred_df": test_pred,
    }
    return results, calibrator


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


# ── Paired bootstrap significance test ────────────────────────────────────────

def paired_bootstrap_test(
    ll_base: np.ndarray,
    ll_alt: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test on per-match log-loss differences.

    Both models score the SAME matches, so the per-match difference
    d_k = ll_alt(k) − ll_base(k) removes between-match variance entirely.
    We resample the matches with replacement n_boot times and look at the
    distribution of mean(d): if the 95% CI excludes 0, the difference is
    statistically meaningful; if it straddles 0, it's indistinguishable
    from noise.

    Returns dict: mean_diff, ci_low, ci_high, p_value, n
    (mean_diff > 0 means the alternative model is WORSE — higher log-loss).
    """
    d = np.asarray(ll_alt, dtype=float) - np.asarray(ll_base, dtype=float)
    n = len(d)
    rng = np.random.default_rng(seed)

    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = d[rng.integers(0, n, n)].mean()

    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    # Two-sided bootstrap p-value: how often the resampled mean crosses zero
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())

    return {
        "mean_diff": float(d.mean()),
        "ci_low":    float(ci_low),
        "ci_high":   float(ci_high),
        "p_value":   float(min(p, 1.0)),
        "n":         n,
    }


# ── LightGBM calibration layer report ────────────────────────────────────────

def calibration_report_with_lgbm(
    model: DixonColesModel,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict, "LGBMCalibrator"]:  # type: ignore[name-defined]
    """
    Fit a LightGBM calibrator on validation-set DC predictions, then compare
    DC-alone vs DC+LGBM log-loss on the held-out test set.

    Training data  : val predictions (model never saw these during DC fitting)
    Evaluation data: test predictions (LightGBM never saw these)

    Returns
    -------
    results    : dict with keys 'dc_val', 'dc_test', 'lgbm_val', 'lgbm_test'
                 each mapping to {'log_loss': float, 'n': int}
    calibrator : fitted LGBMCalibrator (ready to save / use in production)
    """
    # Lazy import to avoid hard dependency at module load time
    from src.model.lgbm_calibrator import LGBMCalibrator

    print("\n" + "=" * 62)
    print(f"{'LGBM CALIBRATION COMPARISON':^62}")
    print("=" * 62)
    print(f"  Baseline (uniform 1/3): {BASELINE_LOG_LOSS:.4f}")

    # ── 1. DC predictions on both splits ────────────────────────────────────
    print("\n  Running Dixon-Coles predictions on val + test sets...")
    val_pred_df  = predict_matches(model, val)
    test_pred_df = predict_matches(model, test)

    dc_val_ll  = log_loss_score(val_pred_df)
    dc_test_ll = log_loss_score(test_pred_df)

    print(f"  DC  val  log-loss: {dc_val_ll:.4f}  (n={len(val_pred_df):,})")
    print(f"  DC  test log-loss: {dc_test_ll:.4f}  (n={len(test_pred_df):,})")

    # ── 2. Fit LGBM on val predictions (hyperparams chosen by internal CV) ───
    print(f"\n  Fitting LGBMCalibrator on {len(val_pred_df):,} val predictions "
          f"(5-fold CV tuning)...")
    calibrator = LGBMCalibrator()
    calibrator.fit_cv(val_pred_df, verbose=True)

    # ── 3. Calibrated predictions on test set ───────────────────────────────
    cal_test_df = calibrator.predict_proba_df(test_pred_df)
    lgbm_test_ll = float(cal_test_df["cal_log_loss"].mean())

    cal_val_df = calibrator.predict_proba_df(val_pred_df)
    lgbm_val_ll = float(cal_val_df["cal_log_loss"].mean())

    # ── 4. Summary table ─────────────────────────────────────────────────────
    print(f"\n  {'Model':20s}  {'Val log-loss':>12}  {'Test log-loss':>13}  {'Δ test':>8}")
    print("  " + "-" * 58)
    print(f"  {'Baseline (uniform)':20s}  "
          f"{BASELINE_LOG_LOSS:>12.4f}  {BASELINE_LOG_LOSS:>13.4f}  {'—':>8}")
    print(f"  {'Dixon-Coles':20s}  "
          f"{dc_val_ll:>12.4f}  {dc_test_ll:>13.4f}  {'—':>8}")
    lgbm_delta = lgbm_test_ll - dc_test_ll
    better = "▲ better" if lgbm_delta < -0.001 else ("▼ worse" if lgbm_delta > 0.001 else "≈ same")
    print(f"  {'DC + LightGBM':20s}  "
          f"{lgbm_val_ll:>12.4f}  {lgbm_test_ll:>13.4f}  "
          f"{lgbm_delta:>+6.4f} {better}")

    # ── Paired bootstrap: is the test-set difference statistically real? ─────
    boot = paired_bootstrap_test(
        ll_base=test_pred_df["log_loss"].values,
        ll_alt=cal_test_df["cal_log_loss"].values,
    )
    sig = (boot["ci_low"] > 0) or (boot["ci_high"] < 0)
    print(f"\n  Paired bootstrap on test set (n={boot['n']:,}, 10k resamples):")
    print(f"    Δ log-loss (LGBM − DC) = {boot['mean_diff']:+.4f}  "
          f"95% CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]  "
          f"p ≈ {boot['p_value']:.3f}")
    if sig and boot["mean_diff"] > 0:
        print(f"    → LGBM is significantly WORSE than raw DC. "
              f"Prefer raw DC probabilities.")
    elif sig and boot["mean_diff"] < 0:
        print(f"    → LGBM is significantly BETTER than raw DC.")
    else:
        print(f"    → No statistically detectable difference; "
              f"the simpler model (raw DC) wins by parsimony.")

    # ── 5. Draw calibration comparison ───────────────────────────────────────
    print(f"\n  Draw overconfidence fix (test set):")
    print(f"    {'Metric':30s}  {'DC':>8}  {'DC+LGBM':>9}")
    print(f"    " + "-" * 52)
    dc_draw_pred  = test_pred_df["p_draw"].mean()
    cal_draw_pred = cal_test_df["cal_draw"].mean()
    actual_draw   = (test_pred_df["actual"] == "draw").mean()
    print(f"    {'Mean predicted draw prob':30s}  {dc_draw_pred:>8.1%}  {cal_draw_pred:>9.1%}")
    print(f"    {'Actual draw rate':30s}  {actual_draw:>8.1%}  {actual_draw:>9.1%}")
    print(f"    {'Draw bias (pred − actual)':30s}  "
          f"{(dc_draw_pred - actual_draw):>+7.1%}  "
          f"{(cal_draw_pred - actual_draw):>+8.1%}")

    # ── 6. Outcome accuracy (modal prediction) ───────────────────────────────
    print(f"\n  Outcome accuracy (modal prediction):")
    for label, p_cols in [
        ("Dixon-Coles",  ("p_home",   "p_draw",   "p_away")),
        ("DC + LightGBM", ("cal_home", "cal_draw", "cal_away")),
    ]:
        df = cal_test_df  # both sets of cols are in cal_test_df
        pred_col = df[[p_cols[0], p_cols[1], p_cols[2]]].idxmax(axis=1).str.replace("cal_", "").str.replace("p_", "")
        acc = (pred_col == df["actual"]).mean()
        draw_recall = (pred_col[df["actual"] == "draw"] == "draw").mean() if (df["actual"] == "draw").sum() > 0 else 0.0
        print(f"    {label:20s}  acc={acc:.1%}  draw-recall={draw_recall:.1%}")

    print("=" * 62)

    results = {
        "dc_val":   {"log_loss": dc_val_ll,   "n": len(val_pred_df)},
        "dc_test":  {"log_loss": dc_test_ll,  "n": len(test_pred_df)},
        "lgbm_val": {"log_loss": lgbm_val_ll, "n": len(val_pred_df)},
        "lgbm_test":{"log_loss": lgbm_test_ll,"n": len(test_pred_df)},
        "bootstrap": boot,
        "val_pred_df":  val_pred_df,
        "test_pred_df": test_pred_df,
    }
    return results, calibrator


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.fetch_matches import fetch_and_process

    matches = fetch_and_process(force=False)
    model, results = fit_and_evaluate(matches, xi=0.003)
