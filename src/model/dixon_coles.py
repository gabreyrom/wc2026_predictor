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
        teams      : sorted list of team names
        alpha      : attack strength per team (dict)
        beta       : defense weakness per team (dict)
        rho        : low-score correction parameter
        gamma      : feature weights vector (for additional covariates)
        feature_names: names of extra covariates used in fitting
    """

    def __init__(self):
        self.teams: list[str] = []
        self.alpha: dict[str, float] = {}
        self.beta:  dict[str, float] = {}
        self.rho:   float = 0.0
        self.feature_names: list[str] = []
        self.gamma: np.ndarray = np.array([])
        self._fitted = False

    def lambda_ij(
        self,
        team_i: str,
        team_j: str,
        extra_features: Optional[np.ndarray] = None,
    ) -> float:
        """
        Expected goals for team_i against team_j.
        extra_features: vector aligned with self.feature_names
        """
        base = self.alpha.get(team_i, 0.0) - self.beta.get(team_j, 0.0)
        if extra_features is not None and len(self.gamma) > 0:
            base += float(np.dot(self.gamma, extra_features))
        return math.exp(base)

    def predict(
        self,
        team_i: str,
        team_j: str,
        extra_features: Optional[np.ndarray] = None,
        max_goals: int = MAX_GOALS,
    ) -> dict:
        """
        Full prediction for a match between team_i and team_j.

        Returns:
            score_matrix  : np.ndarray of joint probabilities
            lambda_i      : expected goals for team_i
            mu_j          : expected goals for team_j
            p_home        : P(team_i wins)
            p_draw        : P(draw)
            p_away        : P(team_j wins)
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted yet.")

        lam = self.lambda_ij(team_i, team_j, extra_features)
        mu  = self.lambda_ij(team_j, team_i, extra_features)

        mat = score_matrix(lam, mu, self.rho, max_goals)
        probs = outcome_probs(lam, mu, self.rho, max_goals)

        return {
            "score_matrix": mat,
            "lambda_i":     lam,
            "mu_j":         mu,
            "rho":          self.rho,
            **probs,
        }

    def predict_shootout(self) -> float:
        """
        Coin-flip 50/50 for penalty shootout (after extra time ends in draw).
        """
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

    # ── Parameter vector layout ──────────────────────────────────────────────
    # [alpha_0..alpha_{n-1}, beta_0..beta_{n-1}, rho, gamma_0..gamma_{k-1}]
    # Constraint: sum(alpha) = 0  (identifiability — fixed by zeroing alpha[0])
    def unpack(params: np.ndarray):
        alpha = params[:n_teams]
        beta  = params[n_teams:2 * n_teams]
        rho   = params[2 * n_teams]
        gamma = params[2 * n_teams + 1:] if n_extra > 0 else np.array([])
        return alpha, beta, rho, gamma

    # ── Pre-build index arrays for vectorised likelihood ────────────────────
    home_idx = np.array([team_idx[t] for t in matches["home_team"]], dtype=np.int32)
    away_idx = np.array([team_idx[t] for t in matches["away_team"]], dtype=np.int32)
    home_goals = np.clip(matches["home_score"].values.astype(np.int32), 0, max_goals_ll)
    away_goals = np.clip(matches["away_score"].values.astype(np.int32), 0, max_goals_ll)

    # Precompute Poisson log-PMF table: shape (n_matches, max_goals+1)
    # log_poisson_pmf[k, g] = log P(X=g | lambda_k)  — recomputed each call
    # We'll precompute the score pairs as masks for tau correction
    mask_00 = (home_goals == 0) & (away_goals == 0)
    mask_10 = (home_goals == 1) & (away_goals == 0)
    mask_01 = (home_goals == 0) & (away_goals == 1)
    mask_11 = (home_goals == 1) & (away_goals == 1)

    def neg_log_likelihood(params: np.ndarray) -> float:
        alpha, beta, rho, gamma = unpack(params)
        rho_c = float(np.clip(rho, -0.99, 0.0))

        # Vectorised: lambda and mu for all matches at once
        log_lam = alpha[home_idx] - beta[away_idx]   # (n_matches,)
        log_mu  = alpha[away_idx] - beta[home_idx]
        lam = np.exp(log_lam)
        mu  = np.exp(log_mu)

        # Log-Poisson PMF for actual scores
        # log P(X=x | lam) = x*log(lam) - lam - log(x!)
        log_p_home = (home_goals * log_lam - lam
                      - np.array([math.lgamma(g + 1) for g in home_goals]))
        log_p_away = (away_goals * log_mu  - mu
                      - np.array([math.lgamma(g + 1) for g in away_goals]))

        # Dixon-Coles tau correction (vectorised, only low-score cases)
        tau_vals = np.ones(len(matches))
        tau_vals[mask_00] = 1 - rho_c * lam[mask_00] * mu[mask_00]
        tau_vals[mask_10] = 1 + rho_c * mu[mask_10]
        tau_vals[mask_01] = 1 + rho_c * lam[mask_01]
        tau_vals[mask_11] = 1 - rho_c

        tau_vals = np.maximum(tau_vals, 1e-10)
        log_joint = np.log(tau_vals) + log_p_home + log_p_away

        return -float(np.dot(weights, log_joint))

    # ── Initial parameters ───────────────────────────────────────────────────
    x0 = np.zeros(2 * n_teams + 1 + n_extra)
    # Vectorised initialisation from goal stats
    np.add.at(x0, home_idx,              0.01 * home_goals)
    np.add.at(x0, n_teams + away_idx,   -0.01 * home_goals)
    np.add.at(x0, away_idx,              0.01 * away_goals)
    np.add.at(x0, n_teams + home_idx,   -0.01 * away_goals)

    x0[2 * n_teams] = -0.1  # initial rho (slightly negative)

    # ── Optimise ────────────────────────────────────────────────────────────
    bounds = (
        [(None, None)] * (2 * n_teams)  # alpha, beta unconstrained
        + [(-0.99, 0.0)]                # rho must be in (-1, 0]
        + [(None, None)] * n_extra      # gamma unconstrained
    )

    print(f"Fitting Dixon-Coles on {len(matches):,} matches, {n_teams} teams...")
    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-8, "maxfun": 50000},
    )

    if not result.success:
        print(f"Warning: optimisation did not fully converge: {result.message}")

    alpha_fit, beta_fit, rho_fit, gamma_fit = unpack(result.x)

    # ── Build model object ───────────────────────────────────────────────────
    model = DixonColesModel()
    model.teams = teams
    model.alpha = {t: float(alpha_fit[i]) for t, i in team_idx.items()}
    model.beta  = {t: float(beta_fit[i])  for t, i in team_idx.items()}
    model.rho   = float(np.clip(rho_fit, -0.99, 0.0))
    model.feature_names = feature_names or []
    model.gamma = gamma_fit
    model._fitted = True

    print(f"Done. rho={model.rho:.4f}")
    return model


# ── Convenience: predict from raw team names ─────────────────────────────────

def predict_match(
    model: DixonColesModel,
    team_i: str,
    team_j: str,
    max_goals: int = MAX_GOALS,
) -> dict:
    """Wrapper around model.predict() with pretty printing."""
    result = model.predict(team_i, team_j, max_goals=max_goals)
    print(f"\n{team_i} vs {team_j}")
    print(f"  E[goals]: {result['lambda_i']:.2f} – {result['mu_j']:.2f}")
    print(f"  P(home win): {result['home']:.1%}  "
          f"P(draw): {result['draw']:.1%}  "
          f"P(away win): {result['away']:.1%}")

    # Most likely scoreline
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
