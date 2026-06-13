"""
LightGBM Calibration Layer for Dixon-Coles Probabilities.

Why this matters
────────────────
Dixon-Coles is systematically overconfident on draws.  When it predicts
p_draw ≈ 44 %, actual draw rate in WC-era matches is only ~29 %.  A second-
stage gradient-boosting classifier can learn to correct these biases from
a held-out validation set without touching the score matrix used for
simulation.

Architecture
────────────
    Input   → DC output features (p_home, p_draw, p_away, λ_i, μ_j, ρ, ...)
    Model   → LGBMClassifier (multiclass, 3 classes)
    Output  → re-calibrated (p_home, p_draw, p_away) that sum to 1

Design principles
─────────────────
• DC *score matrices* are unchanged — simulation is unaffected.
• Only the *reported* outcome probabilities are corrected.
• Training data = validation-set DC predictions (honest: model never saw them).
• Regularised heavily (small trees, high min_child_samples) to avoid overfitting
  a ~5k-row training set.

Usage
─────
    from src.model.lgbm_calibrator import LGBMCalibrator

    cal = LGBMCalibrator()
    cal.fit(val_pred_df)           # val_pred_df from predict_matches(model, val)

    # Single match
    probs = cal.predict_proba_row(dc_pred_dict)

    # DataFrame of matches
    corrected_df = cal.predict_proba_df(pred_df)

    # Persist / reload
    cal.save("models/lgbm_calibrator.joblib")
    cal2 = LGBMCalibrator.load("models/lgbm_calibrator.joblib")
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError(
        "lightgbm is required for LGBMCalibrator. "
        "Install it with: pip install lightgbm"
    ) from exc

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore


# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "p_home",
    "p_draw",
    "p_away",
    "lambda_i",
    "mu_j",
    "rho",
    "log_lambda_ratio",   # computed internally from lambda_i / mu_j
    "abs_alpha_diff",
    "match_importance",
    "log_value_ratio",    # log(squad value_i / value_j), NaN if not covered
    "elo_diff",           # (pre-match elo_i − elo_j) / 400 — fast drift signal
]

LABEL_MAP: dict[str, int] = {"home": 0, "draw": 1, "away": 2}
LABEL_INV: dict[int, str] = {0: "home", 1: "draw", 2: "away"}


# ── Feature engineering ───────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add log_lambda_ratio column if not already present."""
    df = df.copy()
    if "log_lambda_ratio" not in df.columns:
        df["log_lambda_ratio"] = np.log(
            df["lambda_i"] / (df["mu_j"].clip(lower=1e-9))
        )
    return df


def _to_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy (n × len(FEATURE_COLS)) float32 DataFrame ready for LightGBM.
    Adds derived columns (log_lambda_ratio) if not already present.
    Raises ValueError if required source columns are missing.
    """
    df = _add_derived_features(df)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"LGBMCalibrator: prediction DataFrame is missing columns: {missing}\n"
            f"  Ensure you called predict_matches() with the extended output "
            f"(lambda_i, mu_j, abs_alpha_diff, match_importance)."
        )
    return df[FEATURE_COLS].astype(np.float32).reset_index(drop=True)


# ── Main class ────────────────────────────────────────────────────────────────

class LGBMCalibrator:
    """
    A LightGBM-based second-stage calibrator for Dixon-Coles probabilities.

    Attributes
    ----------
    model      : fitted LGBMClassifier (None before fit)
    feature_importance : dict mapping feature name → importance score (after fit)
    train_log_loss : log-loss on training data (after fit)
    """

    def __init__(self) -> None:
        self.model: Optional[lgb.LGBMClassifier] = None
        self.feature_importance_: dict[str, float] = {}
        self.train_log_loss_: float = float("nan")
        self.cv_log_loss_: float = float("nan")
        self.best_params_: dict = {}
        self._fitted: bool = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        pred_df: pd.DataFrame,
        n_estimators: int = 400,
        learning_rate: float = 0.04,
        max_depth: int = 4,
        num_leaves: int = 15,
        min_child_samples: int = 30,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.2,
        reg_lambda: float = 0.2,
        verbose: bool = False,
    ) -> "LGBMCalibrator":
        """
        Fit the calibrator on a validation-set prediction DataFrame.

        Parameters
        ----------
        pred_df : output of calibration.predict_matches() — must include
                  columns: p_home, p_draw, p_away, lambda_i, mu_j, rho,
                           abs_alpha_diff, match_importance, actual
        n_estimators, learning_rate, ... : LightGBM hyperparameters
            Defaults are chosen conservatively for a ~5k-row training set.

        Returns self for chaining.
        """
        if "actual" not in pred_df.columns:
            raise ValueError("pred_df must have an 'actual' column (home/draw/away).")

        X = _to_feature_df(pred_df)
        y = pred_df["actual"].map(LABEL_MAP).values

        if np.isnan(y).any():
            raise ValueError(
                "Some 'actual' values could not be mapped to 0/1/2. "
                f"Unexpected values: {pred_df['actual'][pd.isna(pred_df['actual'].map(LABEL_MAP))].unique()}"
            )

        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)

        self._fitted = True

        # Feature importance
        importances = self.model.feature_importances_
        self.feature_importance_ = dict(zip(FEATURE_COLS, importances))

        # Training log-loss (reuse X which is already a DataFrame)
        proba = self.model.predict_proba(X)
        ll = 0.0
        for i, true_label in enumerate(y):
            ll -= math.log(max(proba[i, true_label], 1e-10))
        self.train_log_loss_ = ll / len(y)

        if verbose:
            print(f"\nLGBMCalibrator fitted on {len(pred_df):,} matches")
            print(f"  Training log-loss: {self.train_log_loss_:.4f}")
            print(f"  Feature importances:")
            for feat, imp in sorted(
                self.feature_importance_.items(), key=lambda x: -x[1]
            ):
                print(f"    {feat:<20s}  {imp:.0f}")

        return self

    # ── Fit with internal cross-validation ────────────────────────────────────

    def fit_cv(
        self,
        pred_df: pd.DataFrame,
        n_splits: int = 5,
        max_boost_rounds: int = 2000,
        early_stopping_rounds: int = 50,
        param_grid: list[dict] | None = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> "LGBMCalibrator":
        """
        Fit with hyperparameters chosen by k-fold cross-validation on the
        training data. For each candidate config, lgb.cv selects the optimal
        number of boosting rounds via early stopping; the config with the
        lowest CV log-loss wins and is refitted on the full training set.

        The held-out test set is never touched here — this is honest tuning.
        """
        if "actual" not in pred_df.columns:
            raise ValueError("pred_df must have an 'actual' column (home/draw/away).")

        X = _to_feature_df(pred_df)
        y = pred_df["actual"].map(LABEL_MAP).values

        if param_grid is None:
            # Deliberately includes very small models (num_leaves=3) — with
            # little residual bias to learn, near-linear models may win.
            param_grid = [
                {"num_leaves": nl, "learning_rate": lr, "min_child_samples": mcs}
                for nl in (3, 7, 15)
                for lr in (0.03, 0.1)
                for mcs in (50, 100)
            ]

        base_params = {
            "objective":        "multiclass",
            "num_class":        3,
            "metric":           "multi_logloss",
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     1,
            "lambda_l1":        0.2,
            "lambda_l2":        0.2,
            "verbosity":        -1,
            "seed":             seed,
        }

        dtrain = lgb.Dataset(X, label=y)
        best: tuple[float, int, dict] | None = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for g in param_grid:
                cvres = lgb.cv(
                    {**base_params, **g},
                    dtrain,
                    num_boost_round=max_boost_rounds,
                    nfold=n_splits,
                    stratified=True,
                    seed=seed,
                    callbacks=[lgb.early_stopping(early_stopping_rounds,
                                                  verbose=False)],
                )
                losses = cvres["valid multi_logloss-mean"]
                best_iter = int(np.argmin(losses)) + 1
                best_loss = float(np.min(losses))
                if verbose:
                    print(f"    leaves={g['num_leaves']:>2} lr={g['learning_rate']:.2f} "
                          f"min_child={g['min_child_samples']:>3}  "
                          f"cv_logloss={best_loss:.4f}  rounds={best_iter}")
                if best is None or best_loss < best[0]:
                    best = (best_loss, best_iter, g)

        cv_loss, n_rounds, best_grid = best
        self.cv_log_loss_ = cv_loss
        self.best_params_ = {**best_grid, "n_estimators": n_rounds}

        if verbose:
            print(f"\n  Best config: {best_grid}, n_estimators={n_rounds}, "
                  f"cv_logloss={cv_loss:.4f}")

        # Refit on full training data with the winning config
        return self.fit(
            pred_df,
            n_estimators=n_rounds,
            learning_rate=best_grid["learning_rate"],
            max_depth=-1,
            num_leaves=best_grid["num_leaves"],
            min_child_samples=best_grid["min_child_samples"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.2,
            verbose=verbose,
        )

    # ── Predict (single dict) ─────────────────────────────────────────────────

    def predict_proba_row(self, dc_pred: dict) -> dict[str, float]:
        """
        Return calibrated (p_home, p_draw, p_away) for a single match.

        Parameters
        ----------
        dc_pred : dict from DixonColesModel.predict() — must have keys:
                  p_home, p_draw, p_away, lambda_i, mu_j, rho
                  Optional: abs_alpha_diff (default 0), match_importance (default 1.0)
        """
        if not self._fitted:
            raise RuntimeError("LGBMCalibrator must be fitted before prediction.")

        # Build a one-row DataFrame
        row = {
            "p_home":           dc_pred.get("home", dc_pred.get("p_home", 0.333)),
            "p_draw":           dc_pred.get("draw", dc_pred.get("p_draw", 0.333)),
            "p_away":           dc_pred.get("away", dc_pred.get("p_away", 0.333)),
            "lambda_i":         dc_pred.get("lambda_i", 1.0),
            "mu_j":             dc_pred.get("mu_j", 1.0),
            "rho":              dc_pred.get("rho", -0.1),
            "abs_alpha_diff":   dc_pred.get("abs_alpha_diff", 0.0),
            "match_importance": dc_pred.get("match_importance", 1.0),
            "log_value_ratio":  dc_pred.get("log_value_ratio", float("nan")),
            "elo_diff":         dc_pred.get("elo_diff", float("nan")),
        }
        df = pd.DataFrame([row])
        X = _to_feature_df(df)

        proba = self.model.predict_proba(X)[0]
        return {"home": float(proba[0]), "draw": float(proba[1]), "away": float(proba[2])}

    # ── Predict (DataFrame) ───────────────────────────────────────────────────

    def predict_proba_df(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with calibrated probability columns added.

        The input pred_df must be output of predict_matches() (with the
        extended column set). Returns a copy with new columns:
            cal_home, cal_draw, cal_away, cal_log_loss (if 'actual' is present)
        """
        if not self._fitted:
            raise RuntimeError("LGBMCalibrator must be fitted before prediction.")

        X = _to_feature_df(pred_df)
        proba = self.model.predict_proba(X)

        out = pred_df.copy()
        out["cal_home"] = proba[:, 0]
        out["cal_draw"] = proba[:, 1]
        out["cal_away"] = proba[:, 2]

        if "actual" in out.columns:
            label_nums = out["actual"].map(LABEL_MAP).values
            ll_vals = np.array([
                -math.log(max(proba[i, int(label_nums[i])], 1e-10))
                for i in range(len(out))
            ])
            out["cal_log_loss"] = ll_vals

        return out

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Save the fitted calibrator to disk using joblib.

        Args:
            path : file path (e.g. 'models/lgbm_calibrator.joblib')
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted calibrator.")
        if joblib is None:
            raise ImportError("joblib is required to save/load the calibrator. "
                              "Install it with: pip install joblib")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  LGBMCalibrator saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LGBMCalibrator":
        """
        Load a previously saved LGBMCalibrator from disk.

        Args:
            path : file path used in save()

        Returns:
            Fitted LGBMCalibrator instance
        """
        if joblib is None:
            raise ImportError("joblib is required to save/load the calibrator. "
                              "Install it with: pip install joblib")
        cal = joblib.load(Path(path))
        if not isinstance(cal, cls):
            raise TypeError(f"Loaded object is {type(cal)}, expected LGBMCalibrator")
        return cal

    # ── Log-loss scorer ───────────────────────────────────────────────────────

    @staticmethod
    def log_loss_from_df(df: pd.DataFrame, prefix: str = "cal_") -> float:
        """
        Compute mean log-loss from a DataFrame returned by predict_proba_df().

        If prefix='cal_', uses cal_home / cal_draw / cal_away columns.
        If prefix='p_', uses p_home / p_draw / p_away (raw DC columns).
        """
        col = f"{prefix}log_loss"
        if col in df.columns:
            return float(df[col].mean())

        # Fallback: recompute from probability columns
        p_col = {
            "home": f"{prefix}home",
            "draw": f"{prefix}draw",
            "away": f"{prefix}away",
        }
        ll = 0.0
        for _, row in df.iterrows():
            p = row[p_col[row["actual"]]]
            ll -= math.log(max(p, 1e-10))
        return ll / len(df)
