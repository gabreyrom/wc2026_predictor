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
            "draw_rate":    0.25,
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

    recent["draw"] = (recent["pts"] == 1).astype(float)

    return {
        "pts_per_game": float(np.dot(w, recent["pts"]) / w_sum),
        "gf_per_game":  float(np.dot(w, recent["gf"]) / w_sum),
        "ga_per_game":  float(np.dot(w, recent["ga"]) / w_sum),
        "win_rate":     float(np.dot(w, recent["win"]) / w_sum),
        "draw_rate":    float(np.dot(w, recent["draw"]) / w_sum),
        "form_score":   float(np.dot(w, recent["pts"] - 1.5) / w_sum),  # centered
    }


# ── Fast precomputed form table (for λ covariates) ──────────────────────────

def precompute_form_table(
    df: pd.DataFrame,
    n: int = 10,
    xi: float = 0.003,
    elo_k: float = 32.0,
) -> dict[tuple[str, int], float]:
    """
    One chronological pass over the full match history computing, for every
    (team, match_day) pair, the OPPONENT-ADJUSTED momentum: the time-weighted
    mean of (actual result − Elo-expected result) over the team's last `n`
    matches STRICTLY BEFORE that day, where result ∈ {0, 0.5, 1}.

    Why opponent-adjusted: raw points-per-game form is confounded by schedule
    strength — Haiti farming points off weak CONCACAF sides shows "better
    form" than Spain playing elite opposition. Momentum must measure
    performance ABOVE expectation, not absolute results.

    A simple flat-K Elo is maintained inside the same pass to provide the
    expectation; it only needs to rank opponents, not be perfectly tuned.

    Anti-leakage by construction: a match's own result never enters its own
    momentum value. Keys are (team, day_int), day_int = days since epoch.
    """
    from collections import defaultdict, deque

    df = df.sort_values("date")
    H  = df["home_team"].values
    A  = df["away_team"].values
    HS = df["home_score"].values.astype(int)
    AS = df["away_score"].values.astype(int)
    days = pd.to_datetime(df["date"]).values.astype("datetime64[D]").astype(int)

    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=n))
    elo:  dict[str, float] = defaultdict(lambda: 1500.0)
    table: dict[tuple[str, int], float] = {}

    for k in range(len(df)):
        h, a, hs, as_, d = H[k], A[k], HS[k], AS[k], int(days[k])

        # Momentum + draw rate BEFORE this match (strictly past matches only)
        for team in (h, a):
            dq = hist[team]
            if dq:
                d_arr = np.array([x[0] for x in dq], dtype=float)
                res   = np.array([x[1] for x in dq], dtype=float)  # actual − expected
                drw   = np.array([x[2] for x in dq], dtype=float)  # 1 if draw
                w = np.exp(-xi * (d - d_arr))
                table[(team, d)] = (
                    float(np.dot(w, res) / w.sum()),   # momentum
                    float(np.dot(w, drw) / w.sum()),   # draw rate
                )
            else:
                table[(team, d)] = (0.0, 0.25)         # neutral priors

        # Elo expectation and update (flat K — expectation provider only)
        e_h = 1.0 / (1.0 + 10 ** (-(elo[h] - elo[a]) / 400.0))
        s_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        is_draw = 1.0 if hs == as_ else 0.0

        hist[h].append((d, s_h - e_h, is_draw))
        hist[a].append((d, (1.0 - s_h) - (1.0 - e_h), is_draw))
        elo[h] += elo_k * (s_h - e_h)
        elo[a] += elo_k * ((1.0 - s_h) - (1.0 - e_h))

    # Stash final state so prediction-time momentum can be computed
    table["__final_hist__"] = hist          # type: ignore[assignment]
    return table


def make_form_diff_fn(df: pd.DataFrame, n: int = 10, xi: float = 0.003):
    """
    Build an antisymmetric λ-covariate function
        form_diff(team_i, team_j, date) = form_i − form_j
    backed by the precomputed table for historical dates, with an on-demand
    fallback (cached) for prediction dates not in the table (e.g. WC 2026).

    Suitable for DixonColesModel lambda_feature_fns.
    """
    table = precompute_form_table(df, n=n, xi=xi)
    final_hist = table.pop("__final_hist__")
    pred_cache: dict[tuple[str, int], tuple[float, float]] = {}

    def _stats(team: str, day: int) -> tuple[float, float]:
        """(momentum, draw_rate) for team strictly before `day`."""
        v = table.get((team, day))
        if v is not None:
            return v
        # Prediction date beyond the data: stats from the team's final
        # n results, time-decayed to the requested day
        if (team, day) not in pred_cache:
            dq = final_hist.get(team)
            if not dq:
                pred_cache[(team, day)] = (0.0, 0.25)
            else:
                d_arr = np.array([x[0] for x in dq], dtype=float)
                res   = np.array([x[1] for x in dq], dtype=float)
                drw   = np.array([x[2] for x in dq], dtype=float)
                w = np.exp(-xi * (day - d_arr))
                pred_cache[(team, day)] = (
                    float(np.dot(w, res) / w.sum()),
                    float(np.dot(w, drw) / w.sum()),
                )
        return pred_cache[(team, day)]

    def _day(date) -> int:
        return int(np.datetime64(pd.Timestamp(date), "D").astype(int))

    def form_diff(team_i: str, team_j: str, date) -> float:
        """Antisymmetric momentum difference (λ-covariate compatible)."""
        day = _day(date)
        return _stats(team_i, day)[0] - _stats(team_j, day)[0]

    def draw_rate_mean(team_i: str, team_j: str, date) -> float:
        """Symmetric mean draw-proneness of both teams (LGBM feature)."""
        day = _day(date)
        return (_stats(team_i, day)[1] + _stats(team_j, day)[1]) / 2.0

    form_diff.draw_rate_mean = draw_rate_mean   # piggy-back accessor
    return form_diff


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

        # ── Rho context features (drive match-specific low-score correction) ──
        # abs_elo_diff  : team-strength imbalance; larger → fewer draws expected
        # draw_rate_mean: historical draw tendency of both teams combined
        # match_importance: 0=friendly, 0.3=qualifier, 0.7=continental, 1.0=WC
        "abs_elo_diff":     abs(elo_i - elo_j),
        "draw_rate_mean":   (form_i["draw_rate"] + form_j["draw_rate"]) / 2,
        "match_importance": 1.0,   # all WC matches are max importance
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
