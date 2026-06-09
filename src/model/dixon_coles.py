"""
Dixon-Coles Bivariate Poisson Model for football match prediction.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores
           and Inefficiencies in the Football Betting Market"

The model:
    Goals scored by team i against team j:
        X ~ Poisson(lambda_ij),   lambda_ij = exp(alpha_i - beta_j + gamma + features)
        Y ~ Poisson(mu_ij),       mu_ij     = exp(alpha_j - beta_i + features)

    For a neutral venue (all WC games):
        gamma = 0

    Low-score correction (the key Dixon-Coles fix):
        P(X=x, Y=y) = tau(x, y, lam, mu, rho) * Poisson(x; lam) * Poisson(y; mu)

        tau(0,0) = 1 - rho * lam * mu
        tau(1,0) = 1 + rho * mu
        tau(0,1) = 1 + rho * lam
        tau(1,1) = 1 - rho
        tau(x,y) = 1   for all other (x, y)

Fitting:
    Maximum likelihood via scipy.optimize.minimize (L-BFGS-B).
    Time-weighted log-likelihood: each match weighted by exp(-xi * days_ago).
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Optional


MAX_GOALS = 10   # truncate Poisson sum at this value for matrix operations


# ── Low-score correction ─────────────────────────────────────────────────────

def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon-Coles correction factor for the joint probability of (x, y) goals.
    Only modifies the four low-score outcomes; all others return 1.
    """
    if x == 0 and y == 0:
        return 1 - rho * lam * mu
    elif x == 1 and y == 0:
        return 1 + rho * mu
    elif x == 0 and y == 1:
        return 1 + rho * lam
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def joint_prob(
    x: int,
    y: int,
    lam: float,
    mu: float,
    rho: float,
) -> float:
    """
    P(X=x, Y=y) under the Dixon-Coles model.
    Clamps to 0 if tau < 0 (invalid rho range).
    """
    t = tau(x, y, lam, mu, rho)
    if t <= 0:
        return 1e-10
    p = t * poisson.pmf(x, lam) * poisson.pmf(y, mu)
    return max(p, 1e-10)


def score_matrix(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Return a (max_goals+1) x (max_goals+1) matrix of joint probabilities.
    Entry [i, j] = P(home scores i, away scores j).
    """
    mat = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i, j] = joint_prob(i, j, lam, mu, rho)
    # Renormalise to account for truncation
    mat /= mat.sum()
    return mat


def outcome_probs(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> dict[str, float]:
    """
    Return P(home win), P(draw), P(away win) for given lambdas.
    """
    mat = score_matrix(lam, mu, rho, max_goals)
    p_home = float(np.tril(mat, -1).sum())   # i > j
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())    # j > i
    return {"home": p_home, "draw": p_draw, "away": p_away}


# ── Parameter management ─────────────────────────────────────────────────────

class DixonColesModel:
    """
    Fitted Dixon-Coles model storing team attack/defense strengths.

    Attributes:
        teams            : sorted list of team names
        alpha            : attack strength per team (dict)
        beta             : defense weakness per team (dict)
        rho_gamma        : coefficients of the context-dependent rho function
        rho_feature_names: names of context features driving rho
        gamma            : feature weights vector (for lambda covariates)
        feature_names    : names of lambda covariates used in fitting

    rho is no longer a scalar. Instead:
        rho(match) = -0.99 / (1 + exp(-(γ₀ + γ₁·|Δα| + γ₂·match_importance)))

    The sigmoid maps any real input to (-0.99, 0), so rho is always valid.
    Use rho_for_match(context) to evaluate it for a specific matchup.
    """

    # Context features that drive rho (order must match rho_gamma)
    RHO_FEATURE_NAMES: list[str] = ["intercept", "abs_alpha_diff", "match_importance"]

    def __init__(self):
        self.teams: list[str] = []
        self.alpha: dict[str, float] = {}
        self.beta:  dict[str, float] = {}
        self.rho_gamma: np.ndarray = np.array([-2.3, 0.0, 0.0])  # default: rho ≈ -0.09
        self.feature_names: list[str] = []
        self.gamma: np.ndarray = np.array([])
        self._fitted = False

    # ── Rho accessor ──────────────────────────────────────────────────────────

    def rho_for_match(self, context: dict) -> float:
        """
        Compute the match-specific rho from context features.

        context keys (all optional, default to 0):
            abs_alpha_diff   : |alpha_i - alpha_j|  — team-strength imbalance
            match_importance : 0=friendly … 1.0=WC knockout — stake level

        Returns rho in (-0.99, 0.0).
        The sigmoid parameterisation guarantees validity without clamping.
        """
        g = self.rho_gamma
        x = (g[0]                                              # intercept
             + g[1] * context.get("abs_alpha_diff", 0.0)      # strength gap
             + g[2] * context.get("match_importance", 0.0))   # stake level
        return -0.99 / (1.0 + math.exp(-x))

    @property
    def rho(self) -> float:
        """
        Baseline rho (zero context — backward-compatible accessor).
        Equivalent to rho_for_match({}).
        """
        return self.rho_for_match({})

    def _match_context(self, team_i: str, team_j: str, match_importance: float = 1.0) -> dict:
        """Build a rho context dict from two team names and a match-importance level."""
        abs_alpha_diff = abs(self.alpha.get(team_i, 0.0) - self.alpha.get(team_j, 0.0))
        return {"abs_alpha_diff": abs_alpha_diff, "match_importance": match_importance}

    # ── Lambda accessor ───────────────────────────────────────────────────────

    def lambda_ij(
        self,
        team_i: str,
        team_j: str,
        extra_features: Optional[np.ndarray] = None,
    ) -> float:
        """Expected goals for team_i against team_j."""
        base = self.alpha.get(team_i, 0.0) - self.beta.get(team_j, 0.0)
        if extra_features is not None and len(self.gamma) > 0:
            base += float(np.dot(self.gamma, extra_features))
        return math.exp(base)

    # ── Full match prediction ─────────────────────────────────────────────────

    def predict(
        self,
        team_i: str,
        team_j: str,
        extra_features: Optional[np.ndarray] = None,
        rho_context: Optional[dict] = None,
        match_importance: float = 1.0,
        max_goals: int = MAX_GOALS,
    ) -> dict:
        """
        Full prediction for a match between team_i and team_j.

        Args:
            rho_context     : explicit context dict for rho; if None, built
                              automatically from model.alpha + match_importance
            match_importance: 0=friendly … 1.0=WC knockout (used when
                              rho_context is None)

        Returns dict with keys:
            score_matrix, lambda_i, mu_j, rho, home, draw, away
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        lam = self.lambda_ij(team_i, team_j, extra_features)
        mu  = self.lambda_ij(team_j, team_i, extra_features)

        ctx = rho_context if rho_context is not None else \
              self._match_context(team_i, team_j, match_importance)
        rho = self.rho_for_match(ctx)

        mat   = score_matrix(lam, mu, rho, max_goals)
        probs = outcome_probs(lam, mu, rho, max_goals)

        return {
            "score_matrix": mat,
            "lambda_i":     lam,
            "mu_j":         mu,
            "rho":          rho,
            **probs,
        }

    def predict_shootout(self) -> float:
        """Coin-flip 50/50 for penalty shootout (after extra time draw)."""
        return 0.5


# ── MLE fitting ──────────────────────────────────────────────────────────────

def fit(
    matches: pd.DataFrame,
    xi: float = 0.003,
    feature_names: Optional[list[str]] = None,
    feature_matrix: Optional[np.ndarray] = None,
    max_goals_ll: int = 8,
) -> DixonColesModel:
    """
    Fit a Dixon-Coles model via weighted maximum likelihood.

    Args:
        matches       : DataFrame with columns: date, home_team, away_team,
                        home_score, away_score
        xi            : time-decay constant (higher = more weight on recent)
        feature_names : optional list of extra covariate names
        feature_matrix: optional (n_matches x n_features) array, aligned with matches
        max_goals_ll  : max goals considered in log-likelihood (truncation)

    Returns:
        Fitted DixonColesModel
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    # Time weights: most recent match gets weight 1.0
    latest = matches["date"].max()
    days_ago = (latest - matches["date"]).dt.days.values
    weights = np.exp(-xi * days_ago)

    n_extra = len(feature_names) if feature_names else 0
    n_rho   = 3   # [intercept, abs_alpha_diff, match_importance]

    # ── Precompute match_importance from tournament type ─────────────────────
    # Used as the third rho context feature. Scale: 0 (friendly) → 1 (WC).
    _importance_map = {
        "World Cup":                1.0,
        "Continental Championship": 0.7,
        "World Cup Qualifier":      0.3,
        "Continental Qualifier":    0.3,
        "Friendly":                 0.0,
    }
    match_importance_col = (
        matches["tournament_category"]
        .map(_importance_map)
        .fillna(0.0)
        .values
    )

    # ── Parameter vector layout ──────────────────────────────────────────────
    # [alpha_0..alpha_{n-1},          — attack strengths
    #  beta_0..beta_{n-1},            — defense weaknesses
    #  rho_g0, rho_g1, rho_g2,       — rho context coefficients
    #  lambda_gamma_0..k]             — optional lambda covariates
    def unpack(params: np.ndarray):
        alpha     = params[:n_teams]
        beta      = params[n_teams:2 * n_teams]
        rho_gamma = params[2 * n_teams: 2 * n_teams + n_rho]
        lam_gamma = params[2 * n_teams + n_rho:] if n_extra > 0 else np.array([])
        return alpha, beta, rho_gamma, lam_gamma

    # ── Pre-build index arrays for vectorised likelihood ─────────────────────
    home_idx = np.array([team_idx[t] for t in matches["home_team"]], dtype=np.int32)
    away_idx = np.array([team_idx[t] for t in matches["away_team"]], dtype=np.int32)
    home_goals = np.clip(matches["home_score"].values.astype(np.int32), 0, max_goals_ll)
    away_goals = np.clip(matches["away_score"].values.astype(np.int32), 0, max_goals_ll)

    # Precompute log(x!) for all observed goal counts
    lgamma_home = np.array([math.lgamma(g + 1) for g in home_goals])
    lgamma_away = np.array([math.lgamma(g + 1) for g in away_goals])

    # Low-score masks for tau correction
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    def neg_log_likelihood(params: np.ndarray) -> float:
        alpha, beta, rho_gamma, lam_gamma = unpack(params)

        # ── Lambda / mu (vectorised) ─────────────────────────────────────────
        log_lam = alpha[home_idx] - beta[away_idx]
        log_mu  = alpha[away_idx] - beta[home_idx]
        lam = np.exp(log_lam)
        mu  = np.exp(log_mu)

        # ── Per-match rho via sigmoid (no clamping needed) ───────────────────
        # rho(k) = -0.99 / (1 + exp(-(g0 + g1*|Δα| + g2*importance)))
        abs_alpha_diff = np.abs(alpha[home_idx] - alpha[away_idx])
        rho_linear = (rho_gamma[0]
                      + rho_gamma[1] * abs_alpha_diff
                      + rho_gamma[2] * match_importance_col)
        rho_vec = -0.99 / (1.0 + np.exp(-rho_linear))   # shape (n_matches,)

        # ── Log-Poisson PMF for actual scores ────────────────────────────────
        log_p_home = home_goals * log_lam - lam - lgamma_home
        log_p_away = away_goals * log_mu  - mu  - lgamma_away

        # ── Dixon-Coles tau correction (per-match rho) ───────────────────────
        tau_vals = np.ones(len(matches))
        tau_vals[mask_00] = 1 - rho_vec[mask_00] * lam[mask_00] * mu[mask_00]
        tau_vals[mask_10] = 1 + rho_vec[mask_10] * mu[mask_10]
        tau_vals[mask_01] = 1 + rho_vec[mask_01] * lam[mask_01]
        tau_vals[mask_11] = 1 - rho_vec[mask_11]

        tau_vals = np.maximum(tau_vals, 1e-10)
        log_joint = np.log(tau_vals) + log_p_home + log_p_away

        return -float(np.dot(weights, log_joint))

    # ── Initial parameters ───────────────────────────────────────────────────
    x0 = np.zeros(2 * n_teams + n_rho + n_extra)
    # Vectorised initialisation of alpha/beta from goal stats
    np.add.at(x0, home_idx,            0.01 * home_goals)
    np.add.at(x0, n_teams + away_idx, -0.01 * home_goals)
    np.add.at(x0, away_idx,            0.01 * away_goals)
    np.add.at(x0, n_teams + home_idx, -0.01 * away_goals)

    # rho_gamma initial: intercept → rho ≈ -0.09, slopes = 0
    x0[2 * n_teams]     = -2.3   # sigmoid(-2.3) * -0.99 ≈ -0.09
    x0[2 * n_teams + 1] =  0.0   # abs_alpha_diff slope
    x0[2 * n_teams + 2] =  0.0   # match_importance slope

    # ── Optimise (all params unconstrained — sigmoid handles rho bounds) ────
    print(f"Fitting Dixon-Coles on {len(matches):,} matches, {n_teams} teams "
          f"(context-dependent rho)...")
    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-8, "maxfun": 150000},
    )

    if not result.success:
        print(f"Warning: optimisation did not fully converge: {result.message}")

    alpha_fit, beta_fit, rho_gamma_fit, lam_gamma_fit = unpack(result.x)

    # ── Build model object ───────────────────────────────────────────────────
    model = DixonColesModel()
    model.teams         = teams
    model.alpha         = {t: float(alpha_fit[i]) for t, i in team_idx.items()}
    model.beta          = {t: float(beta_fit[i])  for t, i in team_idx.items()}
    model.rho_gamma     = rho_gamma_fit
    model.feature_names = feature_names or []
    model.gamma         = lam_gamma_fit
    model._fitted       = True

    # Print fitted rho at a few representative contexts
    rho_friendly = model.rho_for_match({"abs_alpha_diff": 0.0, "match_importance": 0.0})
    rho_equal_wc = model.rho_for_match({"abs_alpha_diff": 0.0, "match_importance": 1.0})
    rho_mismatch = model.rho_for_match({"abs_alpha_diff": 0.5, "match_importance": 1.0})
    print(f"Done. rho_gamma={rho_gamma_fit.round(3)}")
    print(f"  rho(friendly, equal teams)    = {rho_friendly:.4f}")
    print(f"  rho(WC, equal teams)          = {rho_equal_wc:.4f}")
    print(f"  rho(WC, |Δα|=0.5 mismatch)   = {rho_mismatch:.4f}")
    return model


# ── Convenience: predict from raw team names ─────────────────────────────────

def predict_match(
    model: DixonColesModel,
    team_i: str,
    team_j: str,
    match_importance: float = 1.0,
    max_goals: int = MAX_GOALS,
) -> dict:
    """Wrapper around model.predict() with pretty printing."""
    result = model.predict(team_i, team_j,
                           match_importance=match_importance,
                           max_goals=max_goals)
    print(f"\n{team_i} vs {team_j}")
    print(f"  E[goals]: {result['lambda_i']:.2f} – {result['mu_j']:.2f}  "
          f"rho={result['rho']:.4f}")
    print(f"  P(home win): {result['home']:.1%}  "
          f"P(draw): {result['draw']:.1%}  "
          f"P(away win): {result['away']:.1%}")

    mat = result["score_matrix"]
    best = np.unravel_index(mat.argmax(), mat.shape)
    print(f"  Most likely score: {best[0]}-{best[1]} "
          f"(p={mat[best]:.1%})")
    return result


if __name__ == "__main__":
    # Minimal smoke test
    test = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "home_team": "Spain",
         "away_team": "France", "home_score": 2, "away_score": 1},
        {"date": pd.Timestamp("2024-03-01"), "home_team": "France",
         "away_team": "Spain", "home_score": 1, "away_score": 2},
        {"date": pd.Timestamp("2024-06-01"), "home_team": "Spain",
         "away_team": "Germany", "home_score": 3, "away_score": 0},
        {"date": pd.Timestamp("2024-09-01"), "home_team": "Germany",
         "away_team": "France", "home_score": 0, "away_score": 2},
        {"date": pd.Timestamp("2024-12-01"), "home_team": "France",
         "away_team": "Germany", "home_score": 1, "away_score": 0},
    ])
    model = fit(test, xi=0.003)
    predict_match(model, "Spain", "France")
    predict_match(model, "Germany", "Spain")
