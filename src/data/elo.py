"""
Elo rating system for international football.

Computes and updates Elo ratings from historical match data.
Based on the World Football Elo Ratings methodology (eloratings.net):
  - K-factor varies by match importance
  - Goal difference multiplier applied
  - Neutral venue correction
"""

import math
import pandas as pd
from pathlib import Path

# ── K-factor by match type (confederation-aware) ─────────────────────────────
# The K-factor — how much a result moves the ratings — depends on the raw
# tournament name AND, for some categories, the confederations of the two
# teams. World Cup qualifiers are split by confederation (a UEFA qualifier
# carries more weight than a minor-confederation one). Friendlies are flat.
#
# Ladder:
#   100  FIFA World Cup, UEFA Euro
#    90  Copa América
#    80  UEFA WC qualifying
#    70  UEFA Nations League, CONMEBOL WC qualifying, Confederations Cup
#    60  AFCON, AFC Asian Cup, Gold Cup
#    50  rest-of-confederation WC qualifying, CONCACAF Nations League
#    40  continental qualifying (Euro/AFCON/Asian/Copa qual), all friendlies
#    30  minor/regional tournaments (Gulf Cup, AFF, COSAFA, ...)

# Raw tournament names treated as minor/regional (K=30). Everything not listed
# anywhere — including the literal "Friendly" and unlisted invitationals —
# falls through to K=40. Extend this set to reclassify more tournaments.
MINOR_REGIONAL: frozenset[str] = frozenset({
    "Gulf Cup", "AFF Championship", "COSAFA Cup", "SAFF Cup",
    "SAFF Championship", "EAFF Championship", "Asian Games", "Island Games",
    "CFU Caribbean Cup", "CFU Caribbean Cup qualification", "UNCAF Cup",
    "CECAFA Cup", "WAFF Championship", "Nations Cup", "Pacific Games",
})

# Continental qualifiers (all K=40)
_CONTINENTAL_QUAL: frozenset[str] = frozenset({
    "UEFA Euro qualification", "African Cup of Nations qualification",
    "AFC Asian Cup qualification", "Copa América qualification",
})


def match_k_factor(tournament: str, home: str, away: str) -> int:
    """
    K-factor for a single match, given the raw tournament name and the two
    teams (the teams determine the confederation for WC-qualifier splits).
    """
    from src.data.confederations import CONFEDERATIONS
    t = tournament

    # ── Fixed by tournament name ─────────────────────────────────────────────
    if t == "FIFA World Cup" or t == "UEFA Euro":
        return 100
    if t == "Copa América":
        return 90
    if t == "UEFA Nations League" or t == "Confederations Cup":
        return 70
    if t in ("African Cup of Nations", "AFC Asian Cup", "Gold Cup"):
        return 60
    if t == "CONCACAF Nations League":
        return 50
    if t in _CONTINENTAL_QUAL:
        return 40
    if t in MINOR_REGIONAL:
        return 30

    # ── World Cup qualifying — split by confederation ────────────────────────
    if t == "FIFA World Cup qualification":
        confs = {CONFEDERATIONS.get(home), CONFEDERATIONS.get(away)}
        if "UEFA" in confs:
            return 80          # at least one UEFA team
        if "CONMEBOL" in confs:
            return 70          # CONMEBOL (and no UEFA)
        return 50              # all other confederations

    # ── Friendlies and any unlisted tournament ───────────────────────────────
    return 40


# Default starting Elo for a new team
DEFAULT_ELO = 1500

# Home advantage correction (in Elo points) — 0 for neutral venues
HOME_ADVANTAGE = 100


def goal_diff_multiplier(goal_diff: int) -> float:
    """
    Weight a result by goal difference.
    Formula from eloratings.net:
        GD=1 -> 1.0
        GD=2 -> 1.5
        GD=3+ -> (11 + GD) / 8
    """
    gd = abs(goal_diff)
    if gd == 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11 + gd) / 8


def expected_result(elo_home: float, elo_away: float, neutral: bool = False) -> float:
    """
    Probability that the home team wins (or the first team on a neutral field).
    Uses the standard Elo formula with a 400-point scale.
    On neutral venues, no home advantage is added.
    """
    advantage = 0 if neutral else HOME_ADVANTAGE
    return 1 / (1 + 10 ** (-(elo_home + advantage - elo_away) / 400))


def update_elo(
    elo_home: float,
    elo_away: float,
    score_home: int,
    score_away: int,
    k: float,
    neutral: bool = False,
) -> tuple[float, float]:
    """
    Return updated (elo_home, elo_away) after a match result.

    Actual result W:
        1.0 if home wins
        0.5 if draw
        0.0 if away wins
    """
    if score_home > score_away:
        w = 1.0
    elif score_home == score_away:
        w = 0.5
    else:
        w = 0.0

    expected = expected_result(elo_home, elo_away, neutral)
    gd_mult = goal_diff_multiplier(score_home - score_away)

    delta = k * gd_mult * (w - expected)
    return elo_home + delta, elo_away - delta


def compute_elo_ratings(
    matches: pd.DataFrame,
    time_weight: bool = False,
    xi: float = 0.003,
) -> dict[str, float]:
    """
    Compute Elo ratings for all teams from a match history DataFrame.

    Expected columns:
        date         : datetime
        home_team    : str
        away_team    : str
        home_score   : int
        away_score   : int
        tournament   : str   (used to pick K-factor)
        neutral      : bool  (True if played at a neutral venue)

    Args:
        matches    : sorted DataFrame of historical matches (oldest first)
        time_weight: if True, scale K by exp(-xi * days_from_last)
                     so recent matches update ratings more aggressively
        xi         : decay constant for time weighting

    Returns:
        dict mapping team_name -> current Elo rating
    """
    ratings: dict[str, float] = {}
    matches = matches.sort_values("date").reset_index(drop=True)
    last_date = matches["date"].max()

    for _, row in matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        # Initialise unseen teams
        if home not in ratings:
            ratings[home] = DEFAULT_ELO
        if away not in ratings:
            ratings[away] = DEFAULT_ELO

        # K-factor (confederation-aware, from the raw tournament name + teams)
        k = match_k_factor(row.get("tournament", "Friendly"), home, away)

        # Optional time weighting on K
        if time_weight:
            delta_days = (last_date - row["date"]).days
            k = k * math.exp(-xi * delta_days)

        ratings[home], ratings[away] = update_elo(
            ratings[home],
            ratings[away],
            int(row["home_score"]),
            int(row["away_score"]),
            k=k,
            neutral=bool(row.get("neutral", False)),
        )

    return ratings


def make_elo_diff_fn(matches: pd.DataFrame, xi: float = 0.003):
    """
    Build an antisymmetric feature function
        elo_diff(team_i, team_j, date) = (elo_i − elo_j) / 400
    using each team's Elo STRICTLY BEFORE the given date (anti-leakage: a
    match's own result never enters its own feature).

    One chronological pass stores the pre-match rating per (team, day);
    dates beyond the data (e.g. WC 2026 fixtures) use the final ratings.
    The /400 scaling puts the feature on the natural Elo logistic scale.

    K-factors come from match_k_factor (confederation-aware: built from the
    raw tournament name and the two teams).
    """
    import numpy as np

    df = matches.sort_values("date").reset_index(drop=True)
    days = pd.to_datetime(df["date"]).values.astype("datetime64[D]").astype(int)
    # Raw tournament name per row (falls back to "Friendly" if column absent)
    tourns = (df["tournament"] if "tournament" in df.columns
              else pd.Series(["Friendly"] * len(df))).values

    ratings: dict[str, float] = {}
    table: dict[tuple[str, int], float] = {}

    for k_row in range(len(df)):
        row = df.iloc[k_row]
        home, away = row["home_team"], row["away_team"]
        d = int(days[k_row])

        ratings.setdefault(home, DEFAULT_ELO)
        ratings.setdefault(away, DEFAULT_ELO)

        # Pre-match snapshot (only first match of the day per team is stored;
        # same-day double-headers are essentially nonexistent internationally)
        table.setdefault((home, d), ratings[home])
        table.setdefault((away, d), ratings[away])

        k = match_k_factor(tourns[k_row], home, away)
        ratings[home], ratings[away] = update_elo(
            ratings[home], ratings[away],
            int(row["home_score"]), int(row["away_score"]),
            k=k, neutral=bool(row.get("neutral", False)),
        )

    def _elo(team: str, day: int) -> float:
        v = table.get((team, day))
        if v is None:
            v = ratings.get(team, DEFAULT_ELO)   # prediction date: final rating
        return v

    def elo_diff(team_i: str, team_j: str, date) -> float:
        import numpy as np
        day = int(np.datetime64(pd.Timestamp(date), "D").astype(int))
        return (_elo(team_i, day) - _elo(team_j, day)) / 400.0

    return elo_diff


_ELO_DIFF_FN = None


def elo_diff(team_i: str, team_j: str, date) -> float:
    """
    Module-level antisymmetric Elo-difference feature, (elo_i − elo_j)/400,
    using pre-match ratings (anti-leakage). Lazily builds the rating table
    from the processed match history on first call.
    """
    global _ELO_DIFF_FN
    if _ELO_DIFF_FN is None:
        from src.data.fetch_matches import fetch_and_process
        _ELO_DIFF_FN = make_elo_diff_fn(fetch_and_process(force=False))
    return _ELO_DIFF_FN(team_i, team_j, date)


def load_ratings_from_csv(path: str | Path) -> dict[str, float]:
    """Load pre-computed Elo ratings from a CSV with columns: team, elo."""
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["elo"]))


def save_ratings_to_csv(ratings: dict[str, float], path: str | Path) -> None:
    df = pd.DataFrame(list(ratings.items()), columns=["team", "elo"])
    df = df.sort_values("elo", ascending=False).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} team ratings -> {path}")


if __name__ == "__main__":
    # Quick smoke test with 3 synthetic matches
    test_matches = pd.DataFrame([
        {"date": pd.Timestamp("2025-01-01"), "home_team": "Spain",
         "away_team": "France", "home_score": 2, "away_score": 1,
         "tournament": "Friendly", "neutral": True},
        {"date": pd.Timestamp("2025-03-01"), "home_team": "Brazil",
         "away_team": "Argentina", "home_score": 1, "away_score": 1,
         "tournament": "World Cup Qualifier", "neutral": False},
        {"date": pd.Timestamp("2025-06-01"), "home_team": "England",
         "away_team": "Germany", "home_score": 0, "away_score": 2,
         "tournament": "Continental Championship", "neutral": True},
    ])

    ratings = compute_elo_ratings(test_matches)
    for team, elo in sorted(ratings.items(), key=lambda x: -x[1]):
        print(f"  {team:20s} {elo:.1f}")
