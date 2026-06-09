"""
Feature engineering for match outcome prediction.

Builds a feature vector for each matchup (team_i vs team_j) from:
  - Elo ratings
  - Rolling xG / xGA (scoring and defensive talent)
  - Recent form (points from last N games)
  - Market odds (implied probability)
  - Squad characteristics (avg age, market value proxy)

All features are computed from the perspective of team_i.
A negative value means team_j has the advantage.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path


# ── Form computation helpers ────────────────────────────────────────────────

def match_points(row: pd.Series, team: str) -> float:
    """Return points earned by `team` in a single match row (3/1/0)."""
    if row["home_team"] == team:
        gs, ga = row["home_score"], row["away_score"]
    else:
        gs, ga = row["away_score"], row["home_score"]

    if gs > ga:
        return 3.0
    elif gs == ga:
        return 1.0
    return 0.0


def match_gf_ga(row: pd.Series, team: str) -> tuple[int, int]:
    """Return (goals_for, goals_against) for `team` in this match."""
    if row["home_team"] == team:
        return int(row["home_score"]), int(row["away_score"])
    return int(row["away_score"]), int(row["home_score"])


def rolling_form(
    df: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
    n: int = 10,
    xi: float = 0.003,
) -> dict[str, float]:
    """
    Compute time-weighted rolling statistics for `team` in the last `n` games
    before `as_of`.

    Returns dict with keys:
        pts_per_game, gf_per_game, ga_per_game, win_rate, form_score
    """
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < as_of)
    )
    recent = df[mask].sort_values("date").tail(n).copy()

    if recent.empty:
        return {
            "pts_per_game": 1.5,   # neutral prior
            "gf_per_game":  1.2,
            "ga_per_game":  1.2,
            "win_rate":     0.33,
            "form_score":   0.0,
        }

    # Days from as_of -> time weight
    recent["days_ago"] = (as_of - recent["date"]).dt.days
    recent["weight"] = np.exp(-xi * recent["days_ago"])

    recent["pts"] = recent.apply(lambda r: match_points(r, team), axis=1)
    recent["gf"]  = recent.apply(lambda r: match_gf_ga(r, team)[0], axis=1)
    recent["ga"]  = recent.apply(lambda r: match_gf_ga(r, team)[1], axis=1)
    recent["win"] = (recent["pts"] == 3).astype(float)

    w = recent["weight"].values
    w_sum = w.sum()

    return {
        "pts_per_game": float(np.dot(w, recent["pts"]) / w_sum),
        "gf_per_game":  float(np.dot(w, recent["gf"]) / w_sum),
        "ga_per_game":  float(np.dot(w, recent["ga"]) / w_sum),
        "win_rate":     float(np.dot(w, recent["win"]) / w_sum),
        "form_score":   float(np.dot(w, recent["pts"] - 1.5) / w_sum),  # centered
    }


# ── xG proxy ────────────────────────────────────────────────────────────────

def xg_proxy(
    df: pd.DataFrame,
    team: str,
    as_of: pd.Timestamp,
    n: int = 15,
    xi: float = 0.003,
) -> dict[str, float]:
    """
    Proxy xG from actual goals scored/conceded (rolling time-weighted mean).
    Replace with real xG data when available (FBref).

    Returns: xg_att (attack proxy), xg_def (defense proxy = xGA)
    """
    stats = rolling_form(df, team, as_of, n=n, xi=xi)
    return {
        "xg_att": stats["gf_per_game"],
        "xg_def": stats["ga_per_game"],
    }


# ── Odds conversion ─────────────────────────────────────────────────────────

def implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to raw implied probability (before margin removal)."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def remove_vig(p_home: float, p_draw: float, p_away: float) -> tuple[float, float, float]:
    """
    Remove bookmaker margin (vig) from three-way implied probabilities.
    Returns fair probabilities that sum to 1.
    """
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


# ── Main feature builder ─────────────────────────────────────────────────────

def build_match_features(
    team_i: str,
    team_j: str,
    elo_ratings: dict[str, float],
    match_df: pd.DataFrame,
    as_of: pd.Timestamp,
    odds: dict | None = None,
    squad_data: dict | None = None,
    n_form: int = 10,
    n_xg: int = 15,
    xi: float = 0.003,
) -> dict[str, float]:
    """
    Build the full feature vector for a WC match: team_i vs team_j.
    All features are from the perspective of team_i (positive = team_i advantage).

    Args:
        team_i      : name of team i
        team_j      : name of team j
        elo_ratings : dict of team -> Elo rating
        match_df    : full historical match DataFrame
        as_of       : prediction date (use tournament start date)
        odds        : optional dict with keys 'home', 'draw', 'away' (decimal)
        squad_data  : optional dict with squad-level stats (age, value, etc.)
        n_form      : rolling window for form features
        n_xg        : rolling window for xG proxy
        xi          : time-decay constant

    Returns:
        Feature dict (used as input to Dixon-Coles)
    """
    elo_i = elo_ratings.get(team_i, 1500.0)
    elo_j = elo_ratings.get(team_j, 1500.0)

    form_i = rolling_form(match_df, team_i, as_of, n=n_form, xi=xi)
    form_j = rolling_form(match_df, team_j, as_of, n=n_form, xi=xi)

    xg_i = xg_proxy(match_df, team_i, as_of, n=n_xg, xi=xi)
    xg_j = xg_proxy(match_df, team_j, as_of, n=n_xg, xi=xi)

    features: dict[str, float] = {
        # Elo
        "elo_i":            elo_i,
        "elo_j":            elo_j,
        "elo_diff":         elo_i - elo_j,

        # Scoring / defensive talent (xG proxy)
        "xg_att_i":         xg_i["xg_att"],
        "xg_att_j":         xg_j["xg_att"],
        "xg_def_i":         xg_i["xg_def"],
        "xg_def_j":         xg_j["xg_def"],
        "xg_att_diff":      xg_i["xg_att"] - xg_j["xg_att"],
        "xg_def_diff":      xg_j["xg_def"] - xg_i["xg_def"],  # positive = i defends better

        # Form
        "form_diff":        form_i["form_score"] - form_j["form_score"],
        "win_rate_diff":    form_i["win_rate"]   - form_j["win_rate"],
        "pts_diff":         form_i["pts_per_game"] - form_j["pts_per_game"],

        # WC context
        "is_neutral":       1.0,  # all WC matches
    }

    # Market odds (optional)
    if odds is not None:
        p_i   = implied_prob(odds.get("home", 3.0))
        p_d   = implied_prob(odds.get("draw", 3.0))
        p_j   = implied_prob(odds.get("away", 3.0))
        p_i_f, p_d_f, p_j_f = remove_vig(p_i, p_d, p_j)
        features["odds_implied_i"]    = p_i_f
        features["odds_implied_j"]    = p_j_f
        features["odds_implied_draw"] = p_d_f
        features["odds_logit_i"]      = math.log(p_i_f / (1 - p_i_f + 1e-9))

    # Squad data (optional)
    if squad_data is not None:
        age_i  = squad_data.get(team_i, {}).get("avg_age", 26.0)
        age_j  = squad_data.get(team_j, {}).get("avg_age", 26.0)
        val_i  = squad_data.get(team_i, {}).get("squad_value", 1.0)
        val_j  = squad_data.get(team_j, {}).get("squad_value", 1.0)
        features["age_diff"]        = age_i - age_j
        features["log_value_diff"]  = math.log(val_i + 1) - math.log(val_j + 1)

    return features


def features_to_array(
    features: dict[str, float],
    feature_names: list[str],
) -> np.ndarray:
    """Convert a feature dict to a numpy array in the given column order."""
    return np.array([features[name] for name in feature_names], dtype=np.float64)


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.fetch_matches import fetch_and_process

    print("Loading match data...")
    df = fetch_and_process(force=False)

    elo_dummy = {"Spain": 2050, "France": 2010}
    as_of = pd.Timestamp("2026-06-11")

    feats = build_match_features(
        team_i="Spain",
        team_j="France",
        elo_ratings=elo_dummy,
        match_df=df,
        as_of=as_of,
    )
    for k, v in feats.items():
        print(f"  {k:30s} {v:.4f}")
